from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import data as data_module


def make_frame(n: int = 120, n_fraud: int = 12, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        rng.standard_normal((n, 28)),
        columns=[f"V{i}" for i in range(1, 29)],
    )
    frame.insert(0, "Time", np.arange(n, dtype=np.float64) * 10.0)
    frame["Amount"] = np.linspace(1.0, 500.0, n)
    frame["Class"] = np.zeros(n, dtype=np.int8)
    frame.loc[: n_fraud - 1, "Class"] = 1
    return frame


def test_clean_data_drops_duplicates_and_resets_index() -> None:
    frame = make_frame()
    with_duplicates = pd.concat([frame, frame.iloc[:5]], ignore_index=True)
    clean = data_module.clean_data(with_duplicates)
    assert len(clean) == len(frame)
    assert clean.duplicated().sum() == 0
    assert clean.index.tolist() == list(range(len(frame)))
    assert set(clean["Class"].unique()) == {0, 1}


def test_clean_data_rejects_non_binary_labels() -> None:
    frame = make_frame()
    frame.loc[0, "Class"] = 2
    with pytest.raises(ValueError):
        data_module.clean_data(frame)


def test_feature_columns_exclude_time_and_label() -> None:
    frame = make_frame()
    columns = data_module.feature_columns(frame)
    assert columns == [f"V{i}" for i in range(1, 29)] + ["Amount"]
    assert "Time" not in columns
    assert "Class" not in columns


def test_stratified_splits_are_disjoint_and_cover_all_rows() -> None:
    frame = data_module.clean_data(make_frame())
    labels = frame["Class"].to_numpy()
    splits = data_module.make_stratified_splits(labels, seed=3)

    concatenated = np.concatenate([splits["train"], splits["val"], splits["test"]])
    assert sorted(concatenated.tolist()) == list(range(len(frame)))
    assert len(set(concatenated.tolist())) == len(frame)
    assert len(splits["test"]) == 18
    assert len(splits["val"]) == 18

    for name in ("train", "val", "test"):
        positive_rate = labels[splits[name]].mean()
        assert positive_rate == pytest.approx(labels.mean(), abs=0.02)


def test_chronological_split_respects_time_order() -> None:
    frame = make_frame()
    shuffled = frame.sample(frac=1.0, random_state=7).reset_index(drop=True)
    splits = data_module.make_chronological_split(shuffled)

    train_times = shuffled.iloc[splits["train"]]["Time"]
    val_times = shuffled.iloc[splits["val"]]["Time"]
    test_times = shuffled.iloc[splits["test"]]["Time"]
    assert train_times.max() <= val_times.min()
    assert val_times.max() <= test_times.min()
    assert len(splits["train"]) == 84
    assert len(splits["val"]) == 18
    assert len(splits["test"]) == 18


def test_prepare_arrays_uses_training_amount_statistics_only() -> None:
    frame = data_module.clean_data(make_frame(n=200, n_fraud=20))
    splits = data_module.make_stratified_splits(frame["Class"].to_numpy(), seed=0)

    raw_amount = frame["Amount"].to_numpy()
    train_mean = raw_amount[splits["train"]].mean()
    train_std = raw_amount[splits["train"]].std(ddof=0)

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = data_module.prepare_arrays(frame, splits)

    amount_index = data_module.FEATURE_COLUMNS.index("Amount")
    assert X_train[:, amount_index].mean() == pytest.approx(0.0, abs=1e-10)
    assert X_train[:, amount_index].std(ddof=0) == pytest.approx(1.0, abs=1e-10)
    assert X_val[:, amount_index].mean() == pytest.approx(
        (raw_amount[splits["val"]].mean() - train_mean) / train_std,
        abs=1e-10,
    )
    np.testing.assert_allclose(X_val[:, 0], frame["V1"].to_numpy()[splits["val"]])
    assert set(np.unique(y_train)) == {-1, 1}
    assert y_train.dtype == np.int8
    assert len(y_val) == len(X_val)
    assert len(y_test) == len(X_test)


def test_prepare_arrays_can_skip_amount_standardization() -> None:
    frame = data_module.clean_data(make_frame())
    splits = data_module.make_stratified_splits(frame["Class"].to_numpy(), seed=0)
    X_train, _, _, _, _, _ = data_module.prepare_arrays(
        frame, splits, standardize_amount=False
    )
    amount_index = data_module.FEATURE_COLUMNS.index("Amount")
    np.testing.assert_allclose(
        X_train[:, amount_index],
        frame["Amount"].to_numpy()[splits["train"]],
    )


def test_save_and_load_splits_roundtrip(tmp_path) -> None:
    frame = data_module.clean_data(make_frame())
    splits = data_module.make_stratified_splits(frame["Class"].to_numpy(), seed=1)
    metadata = {
        "protocol": "stratified",
        "seed": 1,
        "clean_rows": len(frame),
        "clean_positives": int((frame["Class"] == 1).sum()),
        "source_sha256": "abc123",
    }
    path = tmp_path / "split.json"
    data_module.save_splits(splits, path, metadata=metadata)

    payload = json.loads(path.read_text())
    assert payload["protocol"] == "stratified"
    assert payload["train_size"] == len(splits["train"])

    loaded, loaded_metadata = data_module.load_splits_with_metadata(path)
    assert loaded_metadata["source_sha256"] == "abc123"
    for name in ("train", "val", "test"):
        np.testing.assert_array_equal(loaded[name], splits[name])
    np.testing.assert_array_equal(
        data_module.load_splits(path)["test"],
        splits["test"],
    )

    data_module.validate_split_metadata(
        loaded_metadata,
        n_rows=len(frame),
        n_positives=int((frame["Class"] == 1).sum()),
        source_sha256="abc123",
    )
    with pytest.raises(ValueError, match="clean_rows"):
        data_module.validate_split_metadata(
            loaded_metadata,
            n_rows=len(frame) + 1,
            n_positives=int((frame["Class"] == 1).sum()),
        )
    with pytest.raises(ValueError, match="source_sha256"):
        data_module.validate_split_metadata(
            loaded_metadata,
            n_rows=len(frame),
            n_positives=int((frame["Class"] == 1).sum()),
            source_sha256="different",
        )


def test_stratified_subsample_preserves_class_rate() -> None:
    frame = data_module.clean_data(make_frame(n=1000, n_fraud=50))
    subset = data_module.stratified_subsample(frame, n=400, seed=5)
    assert len(subset) == 400
    assert subset.index.tolist() == list(range(400))
    assert subset["Class"].mean() == pytest.approx(frame["Class"].mean(), abs=0.01)
    same = data_module.stratified_subsample(frame, n=len(frame), seed=5)
    pd.testing.assert_frame_equal(same, frame.reset_index(drop=True))


def test_subsample_positive_indices_hits_target_rate() -> None:
    labels = np.concatenate(
        [np.full(1980, -1, dtype=np.int8), np.full(20, 1, dtype=np.int8)]
    )
    selected = data_module.subsample_positive_indices(labels, fraud_rate=0.002, seed=0)
    assert selected.size == 4
    kept = 1980 + selected.size
    assert selected.size / kept == pytest.approx(0.002, rel=0.15)
    assert np.all(labels[selected] == 1)


def test_subsample_training_to_rate_uses_both_directions() -> None:
    labels = np.concatenate(
        [np.full(1980, -1, dtype=np.int8), np.full(20, 1, dtype=np.int8)]
    )

    keep, mechanism = data_module.subsample_training_to_rate(
        labels, fraud_rate=0.002, seed=0
    )
    assert mechanism == "positive_subsample"
    assert np.sum(labels[keep] > 0) == 4
    assert np.sum(labels[keep] <= 0) == 1980
    assert np.mean(labels[keep] > 0) == pytest.approx(0.002, rel=0.05)

    keep, mechanism = data_module.subsample_training_to_rate(
        labels, fraud_rate=0.05, seed=0
    )
    assert mechanism == "negative_subsample"
    assert np.sum(labels[keep] > 0) == 20
    assert np.sum(labels[keep] <= 0) == 380
    assert np.mean(labels[keep] > 0) == pytest.approx(0.05, rel=0.01)


def test_flip_labels_count_and_determinism() -> None:
    labels = np.array([1] * 10 + [-1] * 90, dtype=np.int8)
    first = data_module.flip_labels(labels, rate=0.1, seed=0)
    second = data_module.flip_labels(labels, rate=0.1, seed=0)
    np.testing.assert_array_equal(first, second)
    assert np.sum(first != labels) == 10


def test_make_synthetic_fraud_shape_rate_and_determinism() -> None:
    X, y = data_module.make_synthetic_fraud(500, 6, 0.05, seed=2)
    assert X.shape == (500, 6)
    assert y.shape == (500,)
    assert set(np.unique(y)) == {-1, 1}
    assert np.mean(y > 0) == pytest.approx(0.05, abs=0.005)

    X_again, y_again = data_module.make_synthetic_fraud(500, 6, 0.05, seed=2)
    np.testing.assert_array_equal(X, X_again)
    np.testing.assert_array_equal(y, y_again)

    X_other, _ = data_module.make_synthetic_fraud(500, 6, 0.05, seed=3)
    assert not np.array_equal(X, X_other)
