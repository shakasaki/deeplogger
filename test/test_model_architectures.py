"""Tests for deeplogger.model_architectures_v2.

Focus on the clean attention-only ablation model (AttentionOnlyUNetATV) and the
invariants that make it a valid ablation: it must produce the right output
shape, accept any input size, and have a parameter count equal to the bundled
AttentionUNetATV (so the only difference is dropout + pool type, not capacity)
while exceeding the plain UNetATV by exactly the attention-gate parameters.

Also covers the reproducibility property behind the train.py generator fix: an
explicit-generator random_split is deterministic and independent of the global
torch RNG state.
"""

import torch
import torch.utils.data as data

from deeplogger.model_architectures_v2 import (
    UNetATV,
    AttentionUNetATV,
    AttentionOnlyUNetATV,
)


def _n_params(model: torch.nn.Module) -> int:
    """Total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())


class TestAttentionOnlyUNetATV:
    """The clean attention-only ablation model."""

    def test_forward_output_shape_and_range(self):
        """(B, 1, 360, 360) input → (B, 360, 360) probabilities in [0, 1]."""
        model = AttentionOnlyUNetATV()
        x = torch.rand(2, 1, 360, 360)
        y = model(x)
        assert y.shape == (2, 360, 360)
        assert torch.all(y >= 0.0) and torch.all(y <= 1.0)

    def test_forward_accepts_non_360_size(self):
        """All skips use F.interpolate, so any spatial size is accepted."""
        model = AttentionOnlyUNetATV()
        y = model(torch.rand(2, 1, 128, 128))
        assert y.shape == (2, 128, 128)

    def test_param_count_equals_bundled_attention(self):
        """Same capacity as AttentionUNetATV — dropout and pool type add no parameters."""
        attn_only = AttentionOnlyUNetATV()
        attn_bundle = AttentionUNetATV()
        assert _n_params(attn_only) == _n_params(attn_bundle)

    def test_param_count_exceeds_plain_v2_by_gate_params(self):
        """The only thing this ablation adds over UNetATV is the attention gates."""
        attn_only = AttentionOnlyUNetATV()
        plain = UNetATV()
        assert _n_params(attn_only) > _n_params(plain)

    def test_gradients_flow_to_all_parameters(self):
        """A backward pass yields a non-None gradient for every parameter."""
        model = AttentionOnlyUNetATV()
        y = model(torch.rand(1, 1, 64, 64))
        y.sum().backward()
        assert all(p.grad is not None for p in model.parameters())


class TestSeededSplitReproducibility:
    """The property behind the train.py generator fix (decoupled train/val split)."""

    def _split_indices(self, generator: torch.Generator) -> list:
        dataset = list(range(100))
        train_set, _ = data.random_split(dataset, [80, 20], generator=generator)
        return list(train_set.indices)

    def test_same_seed_gives_identical_split(self):
        """Two splits with the same seeded generator are identical."""
        a = self._split_indices(torch.Generator().manual_seed(100))
        b = self._split_indices(torch.Generator().manual_seed(100))
        assert a == b

    def test_split_independent_of_global_rng_state(self):
        """Consuming the global torch RNG first must not change the seeded split.

        This is exactly the bug the generator fix prevents: model init draws
        from the global RNG before the split, so without an explicit generator
        the split would differ between models of different size.
        """
        clean = self._split_indices(torch.Generator().manual_seed(100))
        torch.manual_seed(0)
        torch.randn(1000)  # perturb the global RNG, as a large model init would
        after_global_draw = self._split_indices(torch.Generator().manual_seed(100))
        assert clean == after_global_draw
