from __future__ import annotations

import logging
import os
from typing import Iterable, Any


def setup_logging(verbosity: int | str = 1) -> logging.Logger:
    """Configure and return root logger with a concise format.

    verbosity:
      0 -> WARNING, 1 -> INFO, 2 -> DEBUG
    """
    level = logging.INFO
    if isinstance(verbosity, str):
        verbosity = 2 if verbosity.lower() in ("2", "debug") else 1 if verbosity.lower() in ("1", "info") else 0
    if verbosity <= 0:
        level = logging.WARNING
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(level)
    handler = logging.StreamHandler()
    fmt = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    # Reduce noise from third parties
    noisy = [
        "PIL",
        "matplotlib",
        "ultralytics",
        "sam2",
        "hydra",
        "omegaconf",
        "torch",
    ]
    for name in noisy:
        lg = logging.getLogger(name)
        lg.setLevel(logging.WARNING)
        # Avoid bubbling up INFO messages if libraries misconfigure handlers
        lg.propagate = False
    return logger


def get_progress(iterable: Iterable[Any], *, desc: str = "", total: int | None = None, progress: bool = True):
    """Return tqdm iterator if progress enabled; otherwise original iterable."""
    if not progress:
        return iterable
    try:
        from tqdm.auto import tqdm
        return tqdm(iterable, desc=desc, total=total, dynamic_ncols=True)
    except Exception:
        return iterable
