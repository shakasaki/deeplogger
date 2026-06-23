"""Second-generation ATV U-Net architectures.

Two models:

``UNetATV``
    Faithful reproduction of the Perritaz (2024) thesis architecture with
    per-sample min-max input normalization added.  Pool4 is kept as the
    original size-preserving MaxPool(k=3, s=1, p=1); skip connections 3 and 4
    use bilinear interpolation to handle the resulting spatial quirk.

``AttentionUNetATV``
    Extended version:
    - Attention gates on all four skip connections (Oktay et al., 2018).
    - Spatial dropout (Dropout2d) in deep encoders and bottleneck.
    - Pool4 removed (was a no-op); bottleneck follows enc4 directly, giving a
      clean 4×-downsampled bottleneck.
    - All skip connections use F.interpolate so any input size is accepted.
    - Per-sample normalization retained.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _norm_input(x: torch.Tensor) -> torch.Tensor:
    """Per-sample min-max normalisation to [0, 1].

    Operates on (B, C, H, W).  Each sample in the batch is normalised
    independently so the model is invariant to global amplitude offsets
    between boreholes.
    """
    b = x.shape[0]
    flat = x.view(b, -1)
    lo = flat.min(dim=1).values.view(b, 1, 1, 1)
    hi = flat.max(dim=1).values.view(b, 1, 1, 1)
    return (x - lo) / (hi - lo + 1e-8)


def _conv_block(in_ch: int, out_ch: int, name: str) -> nn.Sequential:
    """Two Conv3×3 + BN + ReLU layers (Perritaz block)."""
    return nn.Sequential(
        OrderedDict([
            (name + "conv1", nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)),
            (name + "norm1", nn.BatchNorm2d(out_ch)),
            (name + "relu1", nn.ReLU(inplace=True)),
            (name + "conv2", nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)),
            (name + "norm2", nn.BatchNorm2d(out_ch)),
            (name + "relu2", nn.ReLU(inplace=True)),
        ])
    )


def _conv_block_drop(in_ch: int, out_ch: int, name: str, p: float = 0.1) -> nn.Sequential:
    """Conv block with spatial dropout after the second ReLU."""
    return nn.Sequential(
        OrderedDict([
            (name + "conv1", nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)),
            (name + "norm1", nn.BatchNorm2d(out_ch)),
            (name + "relu1", nn.ReLU(inplace=True)),
            (name + "conv2", nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)),
            (name + "norm2", nn.BatchNorm2d(out_ch)),
            (name + "relu2", nn.ReLU(inplace=True)),
            (name + "drop",  nn.Dropout2d(p=p)),
        ])
    )


# ---------------------------------------------------------------------------
# Attention gate (Oktay et al., 2018)
# ---------------------------------------------------------------------------

class AttentionGate(nn.Module):
    """Soft spatial attention gate for U-Net skip connections.

    Given a gating signal ``g`` from the decoder and a skip feature ``x``
    from the encoder, computes a per-spatial-location attention map α ∈ (0,1)
    and returns ``x * α``.

    Args:
        F_g: channels in the gating signal (decoder side).
        F_x: channels in the skip feature (encoder side).
        F_int: intermediate projection channels (typically F_x // 2).
    """

    def __init__(self, F_g: int, F_x: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_x, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Resize g to x's spatial size before adding
        g_up = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=True)
        alpha = self.psi(F.relu(self.W_g(g_up) + self.W_x(x), inplace=True))
        return x * alpha


# ---------------------------------------------------------------------------
# Model 1: Perritaz reproduction + normalization
# ---------------------------------------------------------------------------

class UNetATV(nn.Module):
    """Perritaz (2024) ATV U-Net with per-sample input normalisation.

    Exact architecture from the thesis:
    - 4 encoder levels (×2 MaxPool for enc1–3, size-preserving k=3/s=1 for enc4)
    - Bilinear interpolation at skip3/4 to compensate the pool4 quirk
    - Direct cat at skip1/2 (requires H,W divisible by 4)
    - init_features = 32  → bottleneck = 512 channels
    - Sigmoid output

    Args:
        in_channels: 1 for ATV amplitude, 3 if stacking grayscale to RGB.
        out_channels: 1 (binary fracture probability).
        init_features: base feature count (32 in thesis).
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, init_features: int = 32):
        super().__init__()
        f = init_features

        self.encoder1 = _conv_block(in_channels, f,      "enc1")
        self.pool1    = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = _conv_block(f,      f * 2,  "enc2")
        self.pool2    = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = _conv_block(f * 2,  f * 4,  "enc3")
        self.pool3    = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = _conv_block(f * 4,  f * 8,  "enc4")
        self.pool4    = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # size-preserving

        self.bottleneck = _conv_block(f * 8, f * 16, "bottleneck")

        self.upconv4  = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.decoder4 = _conv_block(f * 16, f * 8,  "dec4")
        self.upconv3  = nn.ConvTranspose2d(f * 8,  f * 4, kernel_size=2, stride=2)
        self.decoder3 = _conv_block(f * 8,  f * 4,  "dec3")
        self.upconv2  = nn.ConvTranspose2d(f * 4,  f * 2, kernel_size=2, stride=2)
        self.decoder2 = _conv_block(f * 4,  f * 2,  "dec2")
        self.upconv1  = nn.ConvTranspose2d(f * 2,  f,     kernel_size=2, stride=2)
        self.decoder1 = _conv_block(f * 2,  f,      "dec1")

        self.head = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) — raw ATV amplitude; normalised inside.
        Returns:
            (B, H, W) probability map.
        """
        x = _norm_input(x)

        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        e3 = self.encoder3(self.pool2(e2))
        e4 = self.encoder4(self.pool3(e3))
        bn = self.bottleneck(self.pool4(e4))

        d4 = F.interpolate(self.upconv4(bn), e4.shape[2:], mode="bilinear", align_corners=True)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.decoder4(d4)

        d3 = F.interpolate(self.upconv3(d4), e3.shape[2:], mode="bilinear", align_corners=True)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.decoder3(d3)

        d2 = F.interpolate(self.upconv2(d3), e2.shape[2:], mode="bilinear", align_corners=True)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.decoder2(d2)

        d1 = F.interpolate(self.upconv1(d2), e1.shape[2:], mode="bilinear", align_corners=True)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.decoder1(d1)

        return torch.sigmoid(self.head(d1)).squeeze(1)


# ---------------------------------------------------------------------------
# Model 2: Attention U-Net with dropout
# ---------------------------------------------------------------------------

class AttentionUNetATV(nn.Module):
    """Attention U-Net for ATV fracture segmentation.

    Improvements over ``UNetATV``:

    - **Attention gates** on all four skip connections: the decoder gating
      signal suppresses irrelevant background features before they enter the
      decoder, directly addressing the class imbalance (<1 % foreground).

    - **Spatial dropout** (Dropout2d) in enc3, enc4, and bottleneck: randomly
      zeros entire feature maps during training, preventing co-adaptation and
      reducing overfitting on the ~1700-sample dataset.

    - **Pool4 removed**: the size-preserving MaxPool(k=3,s=1,p=1) was a no-op
      (no spatial downsampling).  Removing it gives a clean 4-level bottleneck
      at H/8, W/8 — no special padding or interpolation quirks.

    - **F.interpolate on all skips**: decoder output is always resized to the
      encoder skip size, so any (H, W) works without padding constraints.

    Args:
        in_channels: 1 (ATV grayscale).
        out_channels: 1 (binary mask).
        init_features: base channel count (default 32).
        dropout_p: spatial dropout probability in deep encoder/bottleneck.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        init_features: int = 32,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        f = init_features

        self.encoder1 = _conv_block(in_channels, f,     "enc1")
        self.pool1    = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = _conv_block(f,      f * 2,  "enc2")
        self.pool2    = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = _conv_block_drop(f * 2,  f * 4,  "enc3", p=dropout_p)
        self.pool3    = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = _conv_block_drop(f * 4,  f * 8,  "enc4", p=dropout_p)
        self.pool4    = nn.MaxPool2d(kernel_size=2, stride=2)
        # pool4 now does real downsampling → bottleneck at H/16, W/16

        self.bottleneck = _conv_block_drop(f * 8, f * 16, "bottleneck", p=dropout_p * 2)

        # Attention gates: F_g = decoder ch before cat, F_x = skip ch
        self.att4 = AttentionGate(F_g=f * 16, F_x=f * 8,  F_int=f * 4)
        self.att3 = AttentionGate(F_g=f * 8,  F_x=f * 4,  F_int=f * 2)
        self.att2 = AttentionGate(F_g=f * 4,  F_x=f * 2,  F_int=f)
        self.att1 = AttentionGate(F_g=f * 2,  F_x=f,      F_int=f // 2)

        self.upconv4  = nn.ConvTranspose2d(f * 16, f * 8,  kernel_size=2, stride=2)
        self.decoder4 = _conv_block(f * 16, f * 8,  "dec4")
        self.upconv3  = nn.ConvTranspose2d(f * 8,  f * 4,  kernel_size=2, stride=2)
        self.decoder3 = _conv_block(f * 8,  f * 4,  "dec3")
        self.upconv2  = nn.ConvTranspose2d(f * 4,  f * 2,  kernel_size=2, stride=2)
        self.decoder2 = _conv_block(f * 4,  f * 2,  "dec2")
        self.upconv1  = nn.ConvTranspose2d(f * 2,  f,      kernel_size=2, stride=2)
        self.decoder1 = _conv_block(f * 2,  f,      "dec1")

        self.head = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W) raw ATV amplitude.
        Returns:
            (B, H, W) probability map.
        """
        x = _norm_input(x)

        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        e3 = self.encoder3(self.pool2(e2))
        e4 = self.encoder4(self.pool3(e3))
        bn = self.bottleneck(self.pool4(e4))

        d4 = F.interpolate(self.upconv4(bn), e4.shape[2:], mode="bilinear", align_corners=True)
        d4 = torch.cat([d4, self.att4(bn, e4)], dim=1)
        d4 = self.decoder4(d4)

        d3 = F.interpolate(self.upconv3(d4), e3.shape[2:], mode="bilinear", align_corners=True)
        d3 = torch.cat([d3, self.att3(d4, e3)], dim=1)
        d3 = self.decoder3(d3)

        d2 = F.interpolate(self.upconv2(d3), e2.shape[2:], mode="bilinear", align_corners=True)
        d2 = torch.cat([d2, self.att2(d3, e2)], dim=1)
        d2 = self.decoder2(d2)

        d1 = F.interpolate(self.upconv1(d2), e1.shape[2:], mode="bilinear", align_corners=True)
        d1 = torch.cat([d1, self.att1(d2, e1)], dim=1)
        d1 = self.decoder1(d1)

        return torch.sigmoid(self.head(d1)).squeeze(1)


# ---------------------------------------------------------------------------
# Model 3: clean attention-only ablation (v2 + gates, nothing else)
# ---------------------------------------------------------------------------

class AttentionOnlyUNetATV(nn.Module):
    """``UNetATV`` (v2) plus attention gates only — a clean attention ablation.

    This model is identical to :class:`UNetATV` — same per-sample input
    normalisation, same ``_conv_block`` encoders/decoders with **no** spatial
    dropout, and the same **size-preserving** ``pool4`` (``MaxPool(k=3, s=1,
    p=1)``) — except that each of the four skip connections is passed through an
    :class:`AttentionGate` (Oktay et al., 2018) before concatenation.

    Its purpose is to isolate the effect of the attention gates. The existing
    :class:`AttentionUNetATV` confounds three changes at once (gates **+**
    ``Dropout2d`` **+** a real down-sampling ``pool4``); comparing *this* model
    to :class:`UNetATV` varies **only** the gates, so any performance difference
    is attributable to attention alone. By construction its parameter count
    equals :class:`AttentionUNetATV` (dropout and pool type add no parameters)
    and exceeds :class:`UNetATV` by exactly the gate parameters.

    Args:
        in_channels: 1 for ATV amplitude, 3 if stacking grayscale to RGB.
        out_channels: 1 (binary fracture probability).
        init_features: base feature count (32 in thesis).

    Examples:
        >>> import torch
        >>> model = AttentionOnlyUNetATV()
        >>> y = model(torch.rand(2, 1, 360, 360))
        >>> y.shape
        torch.Size([2, 360, 360])

    See Also:
        UNetATV: the no-attention baseline this ablation is compared against.
        AttentionUNetATV: the bundled variant (gates + dropout + real pool4).
        AttentionGate: the Oktay (2018) soft attention gate used on each skip.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, init_features: int = 32):
        super().__init__()
        f = init_features

        # Encoder / bottleneck — identical to UNetATV v2 (no dropout, pool4 size-preserving).
        self.encoder1 = _conv_block(in_channels, f,      "enc1")
        self.pool1    = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = _conv_block(f,      f * 2,  "enc2")
        self.pool2    = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = _conv_block(f * 2,  f * 4,  "enc3")
        self.pool3    = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = _conv_block(f * 4,  f * 8,  "enc4")
        self.pool4    = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # size-preserving (v2 quirk)

        self.bottleneck = _conv_block(f * 8, f * 16, "bottleneck")

        # Attention gates on every skip (same dims as AttentionUNetATV).
        self.att4 = AttentionGate(F_g=f * 16, F_x=f * 8, F_int=f * 4)
        self.att3 = AttentionGate(F_g=f * 8,  F_x=f * 4, F_int=f * 2)
        self.att2 = AttentionGate(F_g=f * 4,  F_x=f * 2, F_int=f)
        self.att1 = AttentionGate(F_g=f * 2,  F_x=f,     F_int=f // 2)

        self.upconv4  = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.decoder4 = _conv_block(f * 16, f * 8,  "dec4")
        self.upconv3  = nn.ConvTranspose2d(f * 8,  f * 4, kernel_size=2, stride=2)
        self.decoder3 = _conv_block(f * 8,  f * 4,  "dec3")
        self.upconv2  = nn.ConvTranspose2d(f * 4,  f * 2, kernel_size=2, stride=2)
        self.decoder2 = _conv_block(f * 4,  f * 2,  "dec2")
        self.upconv1  = nn.ConvTranspose2d(f * 2,  f,     kernel_size=2, stride=2)
        self.decoder1 = _conv_block(f * 2,  f,      "dec1")

        self.head = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) — raw ATV amplitude; normalised inside.
        Returns:
            (B, H, W) probability map.
        """
        x = _norm_input(x)

        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        e3 = self.encoder3(self.pool2(e2))
        e4 = self.encoder4(self.pool3(e3))
        bn = self.bottleneck(self.pool4(e4))

        d4 = F.interpolate(self.upconv4(bn), e4.shape[2:], mode="bilinear", align_corners=True)
        d4 = torch.cat([d4, self.att4(bn, e4)], dim=1)
        d4 = self.decoder4(d4)

        d3 = F.interpolate(self.upconv3(d4), e3.shape[2:], mode="bilinear", align_corners=True)
        d3 = torch.cat([d3, self.att3(d4, e3)], dim=1)
        d3 = self.decoder3(d3)

        d2 = F.interpolate(self.upconv2(d3), e2.shape[2:], mode="bilinear", align_corners=True)
        d2 = torch.cat([d2, self.att2(d3, e2)], dim=1)
        d2 = self.decoder2(d2)

        d1 = F.interpolate(self.upconv1(d2), e1.shape[2:], mode="bilinear", align_corners=True)
        d1 = torch.cat([d1, self.att1(d2, e1)], dim=1)
        d1 = self.decoder1(d1)

        return torch.sigmoid(self.head(d1)).squeeze(1)
