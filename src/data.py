"""Data loading, cleaning, reproducible splits, and feature preparation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TIME_COLUMN = "Time"
LABEL_COLUMN = "Class"
AMOUNT_COLUMN = "Amount"
V_COLUMNS = [f"V{i}" for i in range(1, 29)]
FEATURE_COLUMNS = V_COLUMNS + [AMOUNT_COLUMN]
REQUIRED_COLUMNS = [TIME_COLUMN] + FEATURE_COLUMNS + [LABEL_COLUMN]

HF_MIRROR_URL = (
    "https://huggingface.co/datasets/David-Egea/Creditcard-fraud-detection/"
    "resolve/main/creditcard.csv"
)


def load_raw(path: str | Path = "data/raw/creditcard.csv") -> pd.DataFrame:
    """Load the raw ULB CSV and check that the expected columns are present."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {source}. Download it (not into Git) with:\n"
            f'  curl -L --fail --retry 3 -C - -o {source} "{HF_MIRROR_URL}"'
        )
    frame = pd.read_csv(source)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset at {source} is missing required columns: {missing}")
    if frame.empty:
        raise ValueError(f"Dataset at {source} is empty")
    return frame


def clean_data(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """Drop exact duplicate rows and reset positional indexing.

    The split indices on disk refer to positions in the frame returned by this
    function, so the order (first occurrence) and reset index must stay stable.
    """
    frame = df.copy()
    if drop_duplicates:
        frame = frame.drop_duplicates()
    frame = frame.reset_index(drop=True)

    if frame[LABEL_COLUMN].isna().any():
        raise ValueError("Class column contains missing values")
    labels = set(pd.unique(frame[LABEL_COLUMN]).tolist())
    if not labels.issubset({0, 1}):
        raise ValueError(f"Class must contain only 0/1, found {sorted(labels)}")
    frame[LABEL_COLUMN] = frame[LABEL_COLUMN].astype(np.int8)
    return frame


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the model feature order: V1..V28 followed by Amount.

    Time is deliberately excluded; it is used only by the chronological split.
    """
    missing = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return list(FEATURE_COLUMNS)


def make_stratified_splits(
    y: Sequence[int] | np.ndarray,
    seed: int,
    test_size: float = 0.15,
    val_size: float = 0.15,
) -> dict[str, np.ndarray]:
    """Return disjoint train/val/test positional indices, stratified by label."""
    labels = np.asarray(y).ravel()
    n_samples = labels.size
    if n_samples < 3:
        raise ValueError("At least three samples are required")
    if not (0.0 < test_size < 1.0) or not (0.0 < val_size < 1.0):
        raise ValueError("test_size and val_size must lie in (0, 1)")
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be < 1")

    positions = np.arange(n_samples)
    train_val_idx, test_idx = train_test_split(
        positions,
        test_size=test_size,
        stratify=labels,
        random_state=int(seed),
    )
    relative_val_size = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=relative_val_size,
        stratify=labels[train_val_idx],
        random_state=int(seed),
    )
    return {
        "train": np.sort(train_idx).astype(np.int64),
        "val": np.sort(val_idx).astype(np.int64),
        "test": np.sort(test_idx).astype(np.int64),
    }


def make_chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> dict[str, np.ndarray]:
    """Sort rows by Time and return positional indices for train/val/test.

    The frames are ordered by Time, so the returned indices locate rows in the
    original (positional) DataFrame; the test block is always the latest block.
    """
    if TIME_COLUMN not in df.columns:
        raise ValueError(f"Missing {TIME_COLUMN} column")
    if not (0.0 < train_frac < 1.0) or not (0.0 < val_frac < 1.0):
        raise ValueError("train_frac and val_frac must lie in (0, 1)")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1")

    time_values = df[TIME_COLUMN].to_numpy()
    if not np.all(np.isfinite(time_values)):
        raise ValueError("Time column contains non-finite values")

    order = np.argsort(time_values, kind="mergesort")
    n_samples = len(order)
    n_train = int(np.floor(train_frac * n_samples))
    n_val = int(np.floor(val_frac * n_samples))
    if n_train < 1 or n_val < 1 or n_samples - n_train - n_val < 1:
        raise ValueError("Split fractions produce an empty partition")

    return {
        "train": order[:n_train].astype(np.int64),
        "val": order[n_train : n_train + n_val].astype(np.int64),
        "test": order[n_train + n_val :].astype(np.int64),
    }


def save_splits(
    splits: Mapping[str, Sequence[int]],
    path: str | Path,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save split indices as a small, deterministic JSON document."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = dict(metadata or {})
    for name in ("train", "val", "test"):
        if name not in splits:
            raise ValueError(f"Missing split partition: {name}")
        payload[name] = [int(index) for index in splits[name]]
    payload["train_size"] = len(payload["train"])
    payload["val_size"] = len(payload["val"])
    payload["test_size"] = len(payload["test"])
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"), allow_nan=False)
        handle.write("\n")
    return target


def load_splits(path: str | Path) -> dict[str, np.ndarray]:
    """Load split indices saved by :func:`save_splits`."""
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Split file {path} does not contain a JSON object")
    splits: dict[str, np.ndarray] = {}
    for name in ("train", "val", "test"):
        if name not in payload:
            raise ValueError(f"Split file {path} is missing '{name}'")
        splits[name] = np.asarray(payload[name], dtype=np.int64)
    return splits


def _standardize_amount(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    amount_index = FEATURE_COLUMNS.index(AMOUNT_COLUMN)
    mean = float(np.mean(X_train[:, amount_index]))
    std = float(np.std(X_train[:, amount_index], ddof=0))
    if not np.isfinite(std) or std < 1e-12:
        std = 1.0

    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()
    for matrix in (X_train, X_val, X_test):
        matrix[:, amount_index] = (matrix[:, amount_index] - mean) / std
    return X_train, X_val, X_test


def prepare_arrays(
    df: pd.DataFrame,
    splits: Mapping[str, Sequence[int]],
    standardize_amount: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build float64 feature matrices and int8 labels in {-1, +1}.

    ``Amount`` is standardized with the training-fold mean and population
    standard deviation; V1..V28 are left untouched. Returns
    ``X_train, y_train, X_val, y_val, X_test, y_test``.
    """
    columns = feature_columns(df)
    X_all = df[columns].to_numpy(dtype=np.float64, copy=True)
    y_all = np.where(df[LABEL_COLUMN].to_numpy() == 1, 1, -1).astype(np.int8)

    X_train = X_all[splits["train"]]
    X_val = X_all[splits["val"]]
    X_test = X_all[splits["test"]]
    if standardize_amount:
        X_train, X_val, X_test = _standardize_amount(X_train, X_val, X_test)

    return (
        X_train,
        y_all[splits["train"]],
        X_val,
        y_all[splits["val"]],
        X_test,
        y_all[splits["test"]],
    )


def stratified_subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Draw a class-stratified subset of ``n`` rows and reset its index."""
    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive")
    if n > len(df):
        raise ValueError(f"Cannot sample {n} rows from a frame with {len(df)} rows")
    if n == len(df):
        return df.reset_index(drop=True)
    subset, _ = train_test_split(
        df,
        train_size=n,
        stratify=df[LABEL_COLUMN],
        random_state=int(seed),
    )
    return subset.reset_index(drop=True)


def subsample_positive_indices(
    y: Sequence[int] | np.ndarray,
    fraud_rate: float,
    seed: int,
) -> np.ndarray:
    """Return positive indices to keep so the full training set hits ``fraud_rate``.

    All negatives are assumed to be retained. The returned indices are a
    deterministic random subset of the current positives (without replacement).
    """
    labels = np.asarray(y).ravel()
    if not (0.0 < fraud_rate < 1.0):
        raise ValueError("fraud_rate must lie in (0, 1)")
    positive_idx = np.flatnonzero(labels > 0)
    n_negative = int(np.sum(labels <= 0))
    if positive_idx.size == 0:
        return positive_idx.astype(np.int64)
    if n_negative == 0:
        return positive_idx.astype(np.int64)

    target = int(round(fraud_rate * n_negative / (1.0 - fraud_rate)))
    target = max(1, min(target, positive_idx.size))
    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(positive_idx, size=target, replace=False)
    return np.sort(chosen).astype(np.int64)


def subsample_training_to_rate(
    y: Sequence[int] | np.ndarray,
    fraud_rate: float,
    seed: int,
) -> tuple[np.ndarray, str]:
    """Return training indices whose positive prevalence approximates ``fraud_rate``.

    Positive-only subsampling can only *lower* prevalence below the natural rate.
    For a target above it, all positives are kept and training negatives are
    subsampled instead. The mechanism is returned for the run metadata; the
    validation and test partitions are never touched here.
    """
    labels = np.asarray(y).ravel()
    if not (0.0 < fraud_rate < 1.0):
        raise ValueError("fraud_rate must lie in (0, 1)")
    positive_idx = np.flatnonzero(labels > 0)
    negative_idx = np.flatnonzero(labels <= 0)
    if positive_idx.size == 0 or negative_idx.size == 0:
        return np.arange(labels.size, dtype=np.int64), "noop_single_class"

    current_rate = positive_idx.size / labels.size
    if fraud_rate <= current_rate:
        kept_positive = subsample_positive_indices(labels, fraud_rate, seed)
        keep = np.sort(np.concatenate([negative_idx, kept_positive]))
        return keep.astype(np.int64), "positive_subsample"

    target_negative = int(
        round(positive_idx.size * (1.0 - fraud_rate) / fraud_rate)
    )
    target_negative = max(1, min(target_negative, negative_idx.size))
    rng = np.random.default_rng(int(seed))
    kept_negative = np.sort(
        rng.choice(negative_idx, size=target_negative, replace=False)
    )
    keep = np.sort(np.concatenate([positive_idx, kept_negative]))
    return keep.astype(np.int64), "negative_subsample"


def flip_labels(
    y: Sequence[int] | np.ndarray,
    rate: float,
    seed: int,
) -> np.ndarray:
    """Flip a random ``rate`` fraction of labels (training-set noise only)."""
    labels = np.asarray(y).ravel().astype(np.int8)
    if not (0.0 <= rate <= 1.0):
        raise ValueError("rate must lie in [0, 1]")
    n_flip = int(round(rate * labels.size))
    if n_flip == 0:
        return labels.copy()
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(labels.size, size=n_flip, replace=False)
    flipped = labels.copy()
    flipped[indices] *= -1
    return flipped


def make_synthetic_fraud(
    n: int,
    d: int,
    fraud_rate: float,
    seed: int,
    separation: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a controlled linear-separable-ish fraud problem.

    Features are standard normal; a small subset of features is shifted for the
    fraud class. Labels are returned as int8 in {-1, +1}. The generator is used
    only for scaling and stress tests, not as a second application.
    """
    n = int(n)
    d = int(d)
    if n < 2 or d < 1:
        raise ValueError("n must be >= 2 and d must be >= 1")
    if not (0.0 < fraud_rate < 1.0):
        raise ValueError("fraud_rate must lie in (0, 1)")

    n_positive = max(1, int(round(n * fraud_rate)))
    n_negative = n - n_positive
    if n_negative < 1:
        raise ValueError("fraud_rate leaves no negatives")

    rng = np.random.default_rng(int(seed))
    X = rng.standard_normal((n, d), dtype=np.float64)
    y = np.concatenate(
        [
            np.ones(n_positive, dtype=np.int8),
            -np.ones(n_negative, dtype=np.int8),
        ]
    )
    n_informative = min(3, d)
    shift = separation / np.sqrt(n_informative)
    X[:n_positive, :n_informative] += shift
    permutation = rng.permutation(n)
    return X[permutation], y[permutation]
