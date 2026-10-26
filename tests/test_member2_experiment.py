"""Small correctness checks for the Member 2 runner, independent of ULB data."""
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.run_member2 import (
    DEFAULTS, cases_for, compare_pair, data_for, load_verified_ulb,
    objective_diagnostics, parse_args, run_campaign, run_solver, write_json,
)


def test_formal_config_is_ten_fixed_objective_pairs():
    config, dry_run, _ = parse_args(["--config", "configs/member2_correctness.yaml", "--dry-run"])
    cases = cases_for(config)
    assert dry_run and len(cases) == 10
    assert sum(c["kernel"] == "linear" for c in cases) == 9
    assert sum(c["kernel"] == "rbf" for c in cases) == 1
    assert config["C"] == 1 and config["n"] == 10000


def test_total_size_split_and_data_fingerprint():
    config = dict(DEFAULTS, n=100, d=3, fraud_rate=0.20)
    arrays, meta = data_for(config, 0)
    # Preserve WP4's two-stage sklearn rounding at this small n.
    assert (meta["n_train"], meta["n_val"], meta["n_test"]) == (69, 16, 15)
    assert sum(meta[k] for k in ("train_positives", "val_positives", "test_positives")) == 20
    assert meta["training_arrays_sha256"] == data_for(config, 0)[1]["training_arrays_sha256"]
    assert len(arrays) == 6


def test_wrong_raw_checksum_fails_before_csv_parsing(tmp_path):
    path = tmp_path / "wrong.csv"
    path.write_text("not the canonical dataset")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_verified_ulb(path)


def test_json_never_overwrites_by_default(tmp_path):
    path = tmp_path / "record.json"
    write_json(path, {"original": 1})
    with pytest.raises(FileExistsError):
        write_json(path, {"replacement": 2})
    assert json.loads(path.read_text()) == {"original": 1}


def test_invalid_reference_cannot_certify_or_produce_gap():
    result = compare_pair({"status": "ok"}, {"status": "ok", "converged": False}, 1e-4)
    assert result["reference_valid"] is False
    assert result["correctness_passed"] is None
    assert result["objective_relative_gap_to_libsvm"] is None
    assert result["time_to_reference_tolerance_sec"] is None


def test_global_budget_does_not_start_solver():
    result = run_solver("smo", None, None, None, None, timeout=0)
    assert result["status"] == "not_run_global_budget"


def test_kernel_norm_recomputation_uses_training_scores():
    class FixedKernelModel:
        b = 0.5
        def decision_function(self, X):
            # K=I, beta=[0.25,-0.25]. No explicit Gram access is needed.
            return np.array([0.75, 0.25])
    diag, _ = objective_diagnostics(FixedKernelModel(), np.zeros((2, 1)),
                                    np.array([1, -1]), np.ones(2), np.array([0.25, 0.25]))
    assert diag["objective"] == pytest.approx(0.0625 + 1.5)
    assert diag["dual_objective"] == pytest.approx(0.5 - 0.0625)


def test_one_tiny_synthetic_pair_runs_and_keeps_real_histories(tmp_path):
    # Exactly two fits, each on 69 training rows. No real-data download or batch.
    config = dict(DEFAULTS, n=100, d=3, fraud_rate=0.20, tol=1e-4,
                  output=str(tmp_path / "run"), global_sec=30, hard_timeout_sec=10,
                  per_fit_sec=5, log_every=5)
    records = run_campaign(config)
    assert len(records) == 1
    record = records[0]
    assert record["smo"]["status"] == record["libsvm"]["status"] == "ok"
    assert record["comparison"]["correctness_passed"]
    assert len(record["smo"]["history"]["objective"]) >= 2
    assert len(record["libsvm"]["history"]["objective"]) == 1
    assert record["libsvm"]["history"]["time_to_tolerance_sec"] is None
    assert Path(config["output"], "summary.csv").exists()
    assert record["data"]["test_positives"] == 3


def test_ten_thousand_split_counts_match_existing_harness():
    from src.data import make_stratified_splits
    y = np.r_[np.ones(17), -np.ones(9983)]
    for seed in (0, 1, 2):
        split = make_stratified_splits(y, seed)
        assert [len(split[k]) for k in ("train", "val", "test")] == [7000, 1500, 1500]
        assert [int((y[split[k]] > 0).sum()) for k in ("train", "val", "test")] == [12, 2, 3]


@pytest.mark.parametrize("field,value", [
    ("relative_duality_gap", 0.00018125710191957878),
    ("relative_duality_gap", -2e-10),
    ("kkt_violation", 2e-4),
    ("box_violation", 2e-8),
    ("equality_violation", 2e-8),
    ("objective", float("nan")),
    ("duality_gap", None),
])
def test_success_flag_cannot_replace_reference_certificate(field, value):
    diagnostics = {"objective": 6.742943630178095, "dual_objective": 6.741721423757282,
                   "duality_gap": 0.0012222064208131656, "relative_duality_gap": 1e-8,
                   "kkt_violation": 7.40299177515169e-8, "box_violation": 0.0,
                   "equality_violation": 1.7763568394002505e-15}
    diagnostics[field] = value
    reference = {"status": "ok", "converged": True, "diagnostics": diagnostics}
    result = compare_pair({"status": "ok"}, reference, 1e-4)
    assert result["reference_valid"] is False
    assert result["correctness_status"] == "uncertified_reference"
    assert result["correctness_passed"] is None
    assert result["objective_relative_gap_to_libsvm"] is None
    assert result["time_to_reference_tolerance_sec"] is None
