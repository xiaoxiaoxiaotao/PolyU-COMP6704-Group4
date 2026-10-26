"""Reproducible, fixed-objective SMO versus LIBSVM validation for Member 2.

Default: one small synthetic pair. ULB runs reuse the WP4 data construction.
A configured subset_n of 10000 means 10000 TOTAL rows, normally split into
7000 training, 1500 validation and 1500 test rows. No solver sees test labels.
Each solver runs in a spawn subprocess with a hard deadline. This script does
not download data, change saved split files, or overwrite results by default.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
from datetime import datetime, timezone

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import sklearn
from sklearn.svm import SVC
import yaml
from threadpoolctl import threadpool_limits

from src import data as data_module
from src.evaluate import _build_ulb_data, make_per_sample_costs
from src.metrics import evaluate_scores, select_threshold_min_cost
from src.svm_smo import SMO

CANONICAL_SHA256 = "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89"
DEFAULTS = {
    "dataset": "synthetic", "n": 500, "d": 29, "fraud_rate": 0.10,
    "seeds": [0], "ratios": [10], "kernels": ["linear"], "extra_cases": [],
    "C": 1.0, "gamma": "scale", "tol": 1e-4, "per_fit_sec": 60.0,
    "hard_timeout_sec": 65.0, "global_sec": 1800.0, "max_iter": 100000,
    "cache_size_mb": 64.0, "log_every": 25, "subset_seed": 0,
    "raw_path": "data/raw/creditcard.csv", "output": "results/member2",
}


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path, record, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if overwrite else "x", encoding="utf-8") as handle:
        json.dump(json_ready(record), handle, indent=2, allow_nan=False)
        handle.write("\n")


def cases_for(config):
    cases = [{"kernel": k, "ratio": float(r), "seed": int(s)}
             for k in config["kernels"] for r in config["ratios"] for s in config["seeds"]]
    cases.extend(dict(case) for case in config.get("extra_cases", []))
    seen = set()
    for case in cases:
        key = (case["kernel"], float(case["ratio"]), int(case["seed"]))
        if key in seen:
            raise ValueError(f"Duplicate case: {key}")
        if key[0] not in ("linear", "rbf") or key[1] <= 0 or key[2] < 0:
            raise ValueError(f"Invalid case: {case}")
        seen.add(key)
    return cases


def data_for(config, seed, cleaned=None, raw_sha=None):
    if config["dataset"] == "ulb":
        if cleaned is None or raw_sha != CANONICAL_SHA256:
            raise ValueError("ULB data require the verified canonical SHA-256.")
        arrays = _build_ulb_data(
            {"dataset": "ulb", "data": {
                "subset_n": config["n"], "subset_seed": config["subset_seed"],
                "use_saved_splits": False, "split_protocol": "stratified",
                "standardize_amount": True}}, cleaned, seed, {}, None, raw_sha,
        )
        result, meta = tuple(arrays[:6]), dict(arrays[6])
    else:
        X, y = data_module.make_synthetic_fraud(
            config["n"], config["d"], config["fraud_rate"], seed, separation=2.5
        )
        splits = data_module.make_stratified_splits(y, seed)
        result = tuple(a for name in ("train", "val", "test")
                       for a in (X[splits[name]], y[splits[name]]))
        meta = {"source": "synthetic", "split_protocol": "stratified",
                "split_source": "generated-synthetic", "subset_n": config["n"]}
    Xtr, ytr, Xv, yv, Xt, yt = result
    meta.update({"n_total": len(ytr) + len(yv) + len(yt), "n_train": len(ytr),
                 "n_val": len(yv), "n_test": len(yt), "d": Xtr.shape[1],
                 "train_positives": int((ytr > 0).sum()),
                 "val_positives": int((yv > 0).sum()),
                 "test_positives": int((yt > 0).sum()), "raw_sha256": raw_sha})
    # Fingerprint the actual prepared training arrays, not merely their counts.
    fingerprint = hashlib.sha256()
    for a in (Xtr, ytr):
        fingerprint.update(np.ascontiguousarray(a).tobytes())
    meta["training_arrays_sha256"] = fingerprint.hexdigest()
    return result, meta


def load_verified_ulb(path):
    digest = data_module.sha256_file(path)
    if digest != CANONICAL_SHA256:
        raise ValueError(f"ULB SHA-256 mismatch: expected {CANONICAL_SHA256}, got {digest}")
    raw = data_module.load_raw(path)
    if raw.shape != (284807, 31) or int((raw.Class == 1).sum()) != 492:
        raise ValueError("Canonical ULB raw shape or positive count does not match.")
    cleaned = data_module.clean_data(raw)
    if len(cleaned) != 283726 or int((cleaned.Class == 1).sum()) != 473:
        raise ValueError("Canonical ULB deduplication counts do not match.")
    return cleaned, digest


def objective_diagnostics(model, X, y, C, alphas):
    """Recompute linear/RKHS objective without allocating a full Gram matrix."""
    scores = np.asarray(model.decision_function(X), dtype=float)
    b = float(model.b if hasattr(model, "b") else model.intercept_[0])
    beta = alphas * y
    # sum_i beta_i (f(x_i)-b) = beta^T K beta, for either kernel.
    norm_sq = float(beta @ (scores - b))
    scale = max(1.0, float(np.abs(beta * (scores - b)).sum()))
    if norm_sq < -1e-10 * scale:
        raise FloatingPointError("Negative kernel norm beyond numerical tolerance.")
    norm_sq = max(0.0, norm_sq)
    primal = 0.5 * norm_sq + float(C @ np.maximum(0.0, 1.0 - y * scores))
    dual = float(alphas.sum()) - 0.5 * norm_sq
    eps = 16.0 * np.finfo(float).eps * np.maximum(C, 1e-12)
    up = ((y > 0) & (alphas < C - eps)) | ((y < 0) & (alphas > eps))
    low = ((y > 0) & (alphas > eps)) | ((y < 0) & (alphas < C - eps))
    z = y - (scores - b)
    kkt = max(0.0, float(z[up].max() - z[low].min())) if up.any() and low.any() else None
    return {
        "objective": primal, "dual_objective": dual, "duality_gap": primal - dual,
        "relative_duality_gap": (primal - dual) / max(1.0, abs(primal), abs(dual)),
        "kkt_violation": kkt,
        "box_violation": max(0.0, float((-alphas).max()), float((alphas - C).max())),
        "equality_violation": abs(float(y @ alphas)),
    }, scores


def _solver_worker(method, arrays, C, config, case, result_file):
    try:
        with threadpool_limits(limits=1):
            X, y, Xv, yv, Xt, yt = arrays
            started = time.perf_counter()
            if method == "smo":
                model = SMO().fit(
                    X, y, C, seed=case["seed"], kernel=case["kernel"],
                    gamma=config["gamma"], tol=config["tol"], gap_tol=config["tol"],
                    max_iter=config["max_iter"], time_budget_sec=max(0.0, config["per_fit_sec"] - min(1.0, config["per_fit_sec"] * 0.05)),
                    cache_size_mb=config["cache_size_mb"], log_every=config["log_every"],
                )
                alphas = model.alphas
                converged = bool(model.converged)
                reason = model.stop_reason
                history = {"objective": model.objective_history, "time_sec": model.time_history,
                           "dual_objective": model.dual_objective_history,
                           "duality_gap": model.duality_gap_history,
                           "relative_duality_gap": model.relative_duality_gap_history,
                           "kkt_violation": model.kkt_violation_history,
                           "iterations": model.history_iters}
                reported = float(model.objective)
                extra = {"cache_bytes": model.cache_bytes, "n_iter": model.n_iter,
                         "gamma_resolved": model.gamma_, "dual_feasible": model.dual_feasible,
                         "returned_best_checkpoint": model.returned_best_checkpoint,
                         "returned_checkpoint_index": model.returned_checkpoint_index,
                         "returned_checkpoint_iteration": model.returned_checkpoint_iteration}
            else:
                model = SVC(
                    C=1.0, kernel=case["kernel"], gamma=config["gamma"],
                    tol=min(1e-7, config["tol"] * 0.01), shrinking=False,
                    cache_size=config["cache_size_mb"], max_iter=-1,
                ).fit(X, y, sample_weight=C)
                alphas = np.zeros(len(y))
                alphas[model.support_] = np.abs(model.dual_coef_[0])
                converged = int(model.fit_status_) == 0
                reason = "converged" if converged else "libsvm_fit_failed"
                history = None
                reported = None
                extra = {"fit_status": int(model.fit_status_), "n_iter": int(model.n_iter_.sum()),
                         "gamma_resolved": float(model._gamma)}
            fit_time = time.perf_counter() - started
            diagnostics, train_scores = objective_diagnostics(model, X, y, C, alphas)
            val_scores = model.decision_function(Xv)
            threshold, val_cost = select_threshold_min_cost(
                yv, val_scores, c_fn=case["ratio"], c_fp=1.0, return_cost=True
            )
            test_scores = model.decision_function(Xt)
            metrics = evaluate_scores(yt, test_scores, c_fn=case["ratio"], c_fp=1.0,
                                      threshold=threshold)
            if history is None:
                # A terminal observation is not an iterative convergence curve.
                history = {"objective": [diagnostics["objective"]], "time_sec": [fit_time],
                           "kind": "single_terminal_observation", "time_to_tolerance_sec": None}
            record = {"status": "ok", "method": method, "converged": converged,
                      "stop_reason": reason, "fit_time_sec": fit_time,
                      "worker_wall_sec": time.perf_counter() - started,
                      "reported_objective": reported, "diagnostics": diagnostics,
                      "history": history, "validation_threshold": threshold,
                      "validation_cost": val_cost, "test_metrics": metrics,
                      "train_predictions_at_zero": (train_scores >= 0).astype(int).tolist(),
                      "test_predictions_at_zero": (test_scores >= 0).astype(int).tolist(),
                      "test_scores": test_scores.tolist(), **extra}
            write_json(result_file, record)
    except Exception as exc:
        write_json(result_file, {"status": "failure", "method": method,
                                "error_type": type(exc).__name__, "error": str(exc)})


def run_solver(method, arrays, C, config, case, timeout):
    if timeout <= 0:
        return {"status": "not_run_global_budget", "method": method}
    scratch = ROOT / "work" / "member2_runs"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=method + "-", dir=scratch) as folder:
        result_file = str(Path(folder) / "result.json")
        process = mp.get_context("spawn").Process(
            target=_solver_worker, args=(method, arrays, C, config, case, result_file)
        )
        started = time.perf_counter()
        process.start()
        process.join(timeout)
        if process.is_alive():
            process.terminate()
            process.join(2)
            if process.is_alive():
                process.kill()
                process.join(2)
            return {"status": "timeout", "method": method, "hard_timeout_sec": timeout,
                    "process_wall_sec": time.perf_counter() - started}
        if not Path(result_file).exists():
            return {"status": "failure", "method": method, "exitcode": process.exitcode,
                    "error": "Solver worker exited without a result."}
        record = json.loads(Path(result_file).read_text())
        record["process_wall_sec"] = time.perf_counter() - started
        record["hard_timeout_sec"] = timeout
        return record


def certificate_check(result, tol):
    """Certify the returned model independently of a solver's success flag."""
    if result.get("status") != "ok":
        return False, "solver_status_not_ok"
    if not result.get("converged"):
        return False, "solver_not_converged"
    diagnostics = result.get("diagnostics", {})
    required = ("objective", "dual_objective", "duality_gap", "relative_duality_gap",
                "kkt_violation", "box_violation", "equality_violation")
    for key in required:
        value = diagnostics.get(key)
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            return False, f"missing_or_nonfinite_diagnostic:{key}"
    reasons = []
    if diagnostics["relative_duality_gap"] < -1e-10:
        reasons.append("negative_duality_gap")
    if diagnostics["relative_duality_gap"] > tol * (1 + 1e-6):
        reasons.append("relative_duality_gap_above_tolerance")
    if diagnostics["kkt_violation"] < 0 or diagnostics["kkt_violation"] > tol * (1 + 1e-6):
        reasons.append("kkt_above_tolerance")
    if not 0 <= diagnostics["box_violation"] <= 1e-8:
        reasons.append("box_infeasible")
    if not 0 <= diagnostics["equality_violation"] <= 1e-8:
        reasons.append("equality_infeasible")
    return not reasons, ";".join(reasons) if reasons else "certified"


def compare_pair(smo, reference, tol):
    smo_valid, smo_reason = certificate_check(smo, tol)
    reference_valid, reference_reason = certificate_check(reference, tol)
    result = {"smo_certificate_valid": smo_valid, "smo_certificate_reason": smo_reason,
              "reference_valid": reference_valid, "reference_certificate_reason": reference_reason,
              "objective_relative_gap_to_libsvm": None,
              "objective_relative_abs_difference": None,
              "descriptive_objective_relative_difference": None,
              "time_to_reference_tolerance_sec": None,
              "train_prediction_agreement": None, "test_prediction_agreement": None,
              "correctness_passed": None, "correctness_status": "uncertified_reference"}
    sd, rd = smo.get("diagnostics", {}), reference.get("diagnostics", {})
    have_objectives = all(isinstance(d.get("objective"), (int, float))
                          and np.isfinite(d["objective"]) for d in (sd, rd))
    if have_objectives:
        denominator = max(abs(rd["objective"]), np.finfo(float).tiny)
        descriptive_gap = (sd["objective"] - rd["objective"]) / denominator
        result["descriptive_objective_relative_difference"] = descriptive_gap
    for partition in ("train", "test"):
        key = f"{partition}_predictions_at_zero"
        if key in smo and key in reference and len(smo[key]) == len(reference[key]) and len(smo[key]):
            result[f"{partition}_prediction_agreement"] = float(np.mean(
                np.asarray(smo[key]) == np.asarray(reference[key])))
    # An uncertified reference cannot define a certified objective gap or a
    # time-to-tolerance target. This is inconclusive, not an SMO failure.
    if not reference_valid:
        return result
    result["correctness_passed"] = False
    result["correctness_status"] = "smo_certificate_not_met"
    if smo.get("status") != "ok" or not have_objectives:
        return result
    gap = result["descriptive_objective_relative_difference"]
    result["objective_relative_gap_to_libsvm"] = gap
    result["objective_relative_abs_difference"] = abs(gap)
    history = smo.get("history", {})
    objectives, times = history.get("objective", []), history.get("time_sec", [])
    if len(objectives) >= 2 and len(objectives) == len(times):
        for objective, seconds in zip(objectives, times):
            if (objective - rd["objective"]) / denominator <= tol:
                result["time_to_reference_tolerance_sec"] = seconds
                break
    result["correctness_passed"] = bool(smo_valid and abs(gap) <= tol)
    result["correctness_status"] = ("passed" if result["correctness_passed"] else
                                     "reference_disagreement" if smo_valid else "smo_certificate_not_met")
    return result


def _summary_row(record):
    row = {k: record.get(k) for k in ("case_id", "dataset", "kernel", "ratio", "seed", "C")}
    row.update(record.get("data", {}))
    row.update(record["comparison"])
    for method in ("smo", "libsvm"):
        result = record[method]
        for key in ("status", "converged", "stop_reason", "fit_time_sec", "n_iter"):
            row[f"{method}_{key}"] = result.get(key)
        row.update({f"{method}_{key}": value for key, value in result.get("diagnostics", {}).items()})
        row.update({f"{method}_{key}": value for key, value in result.get("test_metrics", {}).items()})
    return row


def run_campaign(config, overwrite=False):
    started = time.perf_counter()
    cases = cases_for(config)
    output = Path(config["output"])
    if output.joinpath("summary.csv").exists() and not overwrite:
        raise FileExistsError(f"{output / 'summary.csv'} exists; use a fresh --output directory.")
    cleaned, raw_sha = (load_verified_ulb(config["raw_path"])
                        if config["dataset"] == "ulb" else (None, None))
    cfg_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    source_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                     for name in ("src/svm_smo.py", "src/data.py", "src/evaluate.py", "src/metrics.py",
                                  "experiments/run_member2.py")}
    prepared = {}
    records = []
    for case in cases:
        case_id = f"{config['dataset']}-n{config['n']}-{case['kernel']}-ratio{case['ratio']:g}-seed{case['seed']}-C{config['C']:g}"
        path = output / "raw" / f"{case_id}.json"
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists; results were not overwritten.")
        if case["seed"] not in prepared:
            prepared[case["seed"]] = data_for(config, case["seed"], cleaned, raw_sha)
        arrays, metadata = prepared[case["seed"]]
        C = make_per_sample_costs(arrays[1], config["C"], case["ratio"])
        record = {"schema_version": 1, "case_id": case_id, "dataset": config["dataset"],
                  **case, "C": config["C"], "data": metadata, "config": config,
                  "config_hash": cfg_hash, "costs_source": "harness_placeholder_sample_mean",
                  "upstream_snapshot_revision": "ac49e434d8321459420f9176a72e688946307487",
                  "source_files_sha256": source_hashes,
                  "C_per_sample_mean": float(C.mean()),
                  "C_per_sample_sha256": hashlib.sha256(C.tobytes()).hexdigest(),
                  "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                  "thresholds": {"kkt_tol": config["tol"], "relative_duality_gap_tol": config["tol"],
                                 "relative_reference_gap_tol": config["tol"]},
                  "environment": {"platform": platform.platform(), "machine": platform.machine(),
                                  "python": platform.python_version(), "numpy": np.__version__,
                                  "sklearn": sklearn.__version__, "blas_threads": 1},
                  "evidence_scope": "Numerical correctness on this fixed dataset and objective. "
                                    "A tiny ULB test-positive count cannot establish generalization superiority."}
        for method in ("smo", "libsvm"):
            remaining = config["global_sec"] - (time.perf_counter() - started)
            # LIBSVM has no cooperative wall-clock limit, so its entire worker
            # (startup, fit, diagnostics) is bounded by the approved fit budget.
            # SMO reserves time inside that budget for its terminal checkpoint.
            method_limit = config["per_fit_sec"] if method == "libsvm" else config["per_fit_sec"] + 5.0
            record[method] = run_solver(method, arrays, C, config, case,
                                        min(config["hard_timeout_sec"], method_limit, remaining))
        record["comparison"] = compare_pair(record["smo"], record["libsvm"], config["tol"])
        record["campaign_elapsed_sec"] = time.perf_counter() - started
        write_json(path, record, overwrite)
        write_json(output / "histories" / f"{case_id}.json", {
            "case_id": case_id, "config_hash": cfg_hash,
            "smo": record["smo"].get("history"), "libsvm": record["libsvm"].get("history"),
            "reference_valid": record["comparison"]["reference_valid"],
            "reference_certificate_reason": record["comparison"]["reference_certificate_reason"],
            "reference_objective": record["libsvm"].get("diagnostics", {}).get("objective")
            if record["comparison"]["reference_valid"] else None,
        }, overwrite)
        records.append(record)
        print(f"{case_id}: SMO={record['smo']['status']}, LIBSVM={record['libsvm']['status']}, "
              f"correctness_passed={record['comparison']['correctness_passed']}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    rows = [_summary_row(r) for r in records]
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "summary.csv").open("w" if overwrite else "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, columns)
        writer.writeheader()
        writer.writerows(rows)
    return records


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="YAML configuration, overridden by explicit CLI arguments")
    parser.add_argument("--dataset", choices=["synthetic", "ulb"])
    for name in ("n", "d", "max-iter", "log-every", "subset-seed"):
        parser.add_argument("--" + name, type=int)
    for name in ("fraud-rate", "C", "tol", "per-fit-sec", "hard-timeout-sec", "global-sec", "cache-size-mb"):
        parser.add_argument("--" + name, type=float)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--ratios", type=float, nargs="+")
    parser.add_argument("--kernels", choices=["linear", "rbf"], nargs="+")
    parser.add_argument("--gamma")
    parser.add_argument("--raw-path")
    parser.add_argument("--output")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = vars(parser.parse_args(argv))
    config = dict(DEFAULTS)
    if args["config"]:
        loaded = yaml.safe_load(Path(args["config"]).read_text()) or {}
        unknown = set(loaded) - set(DEFAULTS)
        if unknown:
            parser.error(f"Unknown YAML key(s): {sorted(unknown)}")
        config.update(loaded)
    config.update({key: value for key, value in args.items() if key in DEFAULTS and value is not None})
    if isinstance(config["gamma"], str) and config["gamma"] not in ("scale", "auto"):
        config["gamma"] = float(config["gamma"])
    if config["n"] < 20 or config["d"] < 1:
        parser.error("Use n >= 20 and d >= 1; both held-out partitions need both classes.")
    if any(not np.isfinite(config[k]) or config[k] <= 0 for k in
           ("C", "tol", "per_fit_sec", "hard_timeout_sec", "global_sec", "cache_size_mb")):
        parser.error("C, tolerances, budgets and cache size must be finite and positive.")
    cases_for(config)
    return config, args["dry_run"], args["overwrite"]


def main(argv=None):
    config, dry_run, overwrite = parse_args(argv)
    if dry_run:
        print(json.dumps({"config": config, "cases": cases_for(config),
                          "n_pairs": len(cases_for(config)), "n_solver_fits": 2 * len(cases_for(config)),
                          "subset_size_means": "total rows before 70/15/15 split",
                          "downloads_or_fits_started": False}, indent=2))
        return 0
    records = run_campaign(config, overwrite)
    if any(r["comparison"]["correctness_passed"] is False for r in records):
        return 1
    return 0 if all(r["comparison"]["correctness_passed"] for r in records) else 2


if __name__ == "__main__":
    raise SystemExit(main())
