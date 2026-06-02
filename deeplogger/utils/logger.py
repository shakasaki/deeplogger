"""Optional logging setup for DeepLogger.

Importing this module has no side effects. Call :func:`setup_logging` from
CLI entry points or scripts to configure handlers.
"""
import logging
import os
import sys


def setup_logging(log_dir: str = ".", level: int = logging.DEBUG) -> logging.Logger:
    """Configure the ``deeplogger`` logger with file and console handlers.

    Args:
        log_dir: directory where ``main.log`` and ``deeplogger.log`` are written.
        level: log level for the ``deeplogger`` logger.

    Returns:
        The configured ``deeplogger`` logger.
    """
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, "main.log"),
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    logger = logging.getLogger("deeplogger")
    logger.setLevel(level)

    fh = logging.FileHandler(os.path.join(log_dir, "deeplogger.log"), mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(name)-18s | %(levelname)-8s | %(message)s"))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG)
    console.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y/%m/%d %I:%M:%S %p",
        )
    )
    logger.addHandler(console)
    return logger


def set_level(name_level: str) -> None:
    """Set the ``deeplogger`` logger level by name (e.g. ``"INFO"``)."""
    logging.getLogger("deeplogger").setLevel(logging.getLevelName(name_level))
