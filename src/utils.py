"""Small shared utilities: YAML configs, seeding, timing, logging, JSON I/O."""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import random
import time
from contextlib import ContextDecorator
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping from ``path``."""
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a YAML mapping at {path}, got {type(payload).__name__}")
    return payload


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` and return a new mapping."""
    merged: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def set_seed(seed: int) -> None:
    """Seed the Python and NumPy RNGs used by the harness."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class Timer(ContextDecorator):
    """Wall-clock timer usable as a context manager or decorator."""

    def __init__(self) -> None:
        self.start: float | None = None
        self.end: float | None = None

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        self.end = None
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.end = time.perf_counter()

    @property
    def elapsed(self) -> float:
        if self.start is None:
            return 0.0
        stop = self.end if self.end is not None else time.perf_counter()
        return float(stop - self.start)


def get_logger(
    name: str = "wp4",
    log_file: str | Path | None = None,
    level: int | str = logging.INFO,
) -> logging.Logger:
    """Create a logger with a console handler and an optional file handler."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not logger.handlers:
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        target = str(path.resolve())
        already_attached = any(
            isinstance(handler, logging.FileHandler)
            and handler.baseFilename == target
            for handler in logger.handlers
        )
        if not already_attached:
            file_handler = logging.FileHandler(path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def configure_single_thread() -> None:
    """Best-effort single-thread configuration for fair wall-clock comparisons.

    Environment variables only take effect if set before BLAS is loaded, so
    experiment entry points set them before importing NumPy. ``threadpoolctl``
    covers the case where BLAS is already initialised.
    """
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[variable] = "1"
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=1)
    except Exception:  # pragma: no cover - optional dependency
        pass


def _to_jsonable(value: Any) -> Any:
    """Recursively convert NumPy/Path objects and non-finite floats for JSON."""
    if isinstance(value, np.ndarray):
        return [_to_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _to_jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(payload: Any, path: str | Path, indent: int = 2) -> Path:
    """Write strict JSON (no NaN/Infinity) and create parent directories."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(payload), handle, indent=indent, allow_nan=False)
        handle.write("\n")
    return target
