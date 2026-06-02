# DeepLogger

Machine-Learning used for borehole televiewer data interpretation.

## Installation

Clone the repository:
```bash
git clone https://gitlab.com/shakasa/deeplogger.git
cd deeplogger
```

### Reproducible environments (recommended)

The repo ships two conda specs (both Python 3.12, editable install):

| Env | File | torch | Use for |
| --- | --- | --- | --- |
| `deeplogger` | `environment.yml` | CUDA (GPU) | training + development (`[dev,gui]`) |
| `deeplogger-gui` | `environment-gui.yml` | CPU-only | running the GUI / CPU inference (`[gui]`) |

**Training / development env (GPU):**
```bash
conda env create -f environment.yml   # or: mamba env create -f environment.yml
conda activate deeplogger
```

**GUI / inference env (lightweight, CPU-only):**
```bash
conda env create -f environment-gui.yml
conda activate deeplogger-gui
```
This skips the ~3-4 GB of CUDA wheels; inference runs on CPU.

The `environment*.yml` files pin only Python; dependencies are resolved from
`pyproject.toml`. For a bit-for-bit rebuild, install the matching fully-pinned
snapshot instead:
```bash
# training env (Linux + CUDA 13 wheels)
pip install -r requirements-lock-train.txt && pip install -e . --no-deps
# gui env (CPU-only, portable)
pip install -r requirements-lock-gui.txt   && pip install -e . --no-deps
```
Note: `requirements-lock-train.txt` is platform-specific (Linux + CUDA 13). On
other platforms use the `environment*.yml` files, which resolve fresh.

### Minimal install

If you only need the library (no GUI, no dev tools):
```bash
python -m pip install -e .
```
Optional extras: `.[dev]` (pytest), `.[gui]` (napari/streamlit viewer), `.[jax]`.

## Data

Download the borehole data from ETH Polybox using pooch:
```bash
python deeplogger/download_data.py
```
Running this a second time will not re-download files that already exist. The
data directory can be relocated with the `DEEPLOGGER_DATA_DIR` environment
variable (defaults to `./data`).

## Tests

```bash
pytest test/
```
