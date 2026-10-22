"""Generate the committed split files from the locally cached ULB dataset."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from src import data as data_module  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_split(
    splits: dict[str, np.ndarray],
    n_rows: int,
    labels: np.ndarray,
) -> None:
    concatenated = np.concatenate([splits["train"], splits["val"], splits["test"]])
    if concatenated.size != n_rows or np.unique(concatenated).size != n_rows:
        raise ValueError("Split partitions are not a disjoint cover of all rows")
    for name in ("train", "val", "test"):
        positive = int(np.sum(labels[splits[name]] == 1))
        print(
            f"  {name:5s}: {len(splits[name]):7d} rows | "
            f"{positive:4d} frauds | rate {positive / len(splits[name]):.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", default="data/raw/creditcard.csv")
    parser.add_argument("--splits-dir", default="data/splits")
    args = parser.parse_args()

    raw_path = Path(args.raw_path)
    splits_dir = Path(args.splits_dir)
    raw = data_module.load_raw(raw_path)
    cleaned = data_module.clean_data(raw, drop_duplicates=True)
    labels = cleaned[data_module.LABEL_COLUMN].to_numpy()
    digest = sha256_file(raw_path)

    print(f"Raw rows: {len(raw)} | duplicates: {len(raw) - len(cleaned)}")
    print(f"Cleaned rows: {len(cleaned)} | positives: {int(np.sum(labels == 1))}")
    print(f"SHA-256: {digest}")

    base_metadata = {
        "dataset": "ulb",
        "source_file": str(raw_path),
        "source_sha256": digest,
        "raw_rows": int(len(raw)),
        "clean_rows": int(len(cleaned)),
        "clean_positives": int(np.sum(labels == 1)),
        "indexing": "positional indices into the deduplicated, reset-index frame",
        "generator": "experiments/generate_splits.py",
    }

    splits_dir.mkdir(parents=True, exist_ok=True)
    for seed in range(5):
        splits = data_module.make_stratified_splits(labels, seed=seed)
        validate_split(splits, len(cleaned), labels)
        path = splits_dir / f"stratified_seed{seed}.json"
        data_module.save_splits(
            splits,
            path,
            metadata={
                **base_metadata,
                "protocol": "stratified",
                "seed": seed,
                "train_fraction": 0.70,
                "val_fraction": 0.15,
                "test_fraction": 0.15,
            },
        )
        print(f"Wrote {path}")

    chronological = data_module.make_chronological_split(
        cleaned,
        train_frac=0.70,
        val_frac=0.15,
    )
    validate_split(chronological, len(cleaned), labels)
    path = splits_dir / "chronological_seed0.json"
    data_module.save_splits(
        chronological,
        path,
        metadata={
            **base_metadata,
            "protocol": "chronological",
            "seed": 0,
            "train_fraction": 0.70,
            "val_fraction": 0.15,
            "test_fraction": 0.15,
            "sort_column": data_module.TIME_COLUMN,
        },
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
