"""DeepLogger — deep learning for borehole televiewer interpretation."""
import os

__version__ = "0.1.0"

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_PACKAGE_DIR, os.pardir))


def _resolve_dir(env_var: str, default: str) -> str:
    return os.environ.get(env_var, default).rstrip(os.sep) + os.sep


DATA_DIR = _resolve_dir("DEEPLOGGER_DATA_DIR", os.path.join(_REPO_ROOT, "data"))
OUTPUT_DIR = _resolve_dir("DEEPLOGGER_OUTPUT_DIR", os.path.join(_REPO_ROOT, "output"))
