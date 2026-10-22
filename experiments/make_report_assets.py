"""Create final report tables/figures, or clearly marked placeholders.

Run after experiments:
    python experiments/make_report_assets.py

Existing non-empty tables are never overwritten. Figures are regenerated from
whatever tables/run JSONs exist; with no solver results they are placeholders
labelled "Pending solver integration" rather than fabricated plots.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PENDING = "Pending solver integration\nNo final numbers are shown."

TABLE_METRICS: dict[str, list[str]] = {
    "correctness": ["objective", "relative_gap"],
    "main_fast": [
        "time_to_tolerance_sec",
        "cost_per_example",
        "pr_auc",
    ],
    "main_full": [
        "time_to_tolerance_sec",
        "fit_time_sec",
        "cost_per_example",
        "pr_auc",
        "recall_at_fpr_0.001",
        "recall_at_fpr_0.01",
        "balanced_accuracy",
    ],
    "cost_sweep": ["cost_per_example", "time_to_tolerance_sec"],
    "ablation": ["time_to_tolerance_sec", "n_iter"],
    "stress": ["cost_per_example", "pr_auc", "recall_at_fpr_0.001"],
    "scaling": ["fit_time_sec", "peak_memory_mb"],
}

TABLE_HEADER = (
    "dataset,method,variant,cost_ratio,n_train,n,mean,std,ci_low,ci_high\n"
)


def table_path(tables_dir: Path, experiment: str, metric: str) -> Path:
    return tables_dir / f"{experiment}_{metric}.csv"


def read_table(tables_dir: Path, experiment: str, metric: str) -> pd.DataFrame | None:
    path = table_path(tables_dir, experiment, metric)
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None
    if frame.empty or "mean" not in frame.columns:
        return None
    frame = frame.dropna(subset=["mean"])
    return frame if not frame.empty else None


def create_empty_tables(tables_dir: Path) -> list[Path]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    for experiment, metrics in TABLE_METRICS.items():
        for metric in metrics:
            path = table_path(tables_dir, experiment, metric)
            if path.exists():
                continue
            path.write_text(TABLE_HEADER, encoding="utf-8")
            created.append(path)
    return created


def placeholder_figure(path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.0, 4.0))
    axis.axis("off")
    axis.set_title(title)
    axis.text(
        0.5,
        0.5,
        PENDING,
        ha="center",
        va="center",
        fontsize=12,
        color="#666666",
        transform=axis.transAxes,
    )
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _group_label(frame: pd.DataFrame) -> pd.Series:
    method = frame["method"].astype(str)
    variant = frame["variant"].fillna("base").astype(str)
    return method.where(variant == "base", method + ":" + variant)


def plot_metric_bars(
    tables_dir: Path,
    figures_dir: Path,
    experiment: str,
    metric: str,
    filename: str,
    title: str,
) -> Path:
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / filename
    frame = read_table(tables_dir, experiment, metric)
    if frame is None:
        placeholder_figure(path, title)
        return path

    frame = frame.copy()
    frame["label"] = _group_label(frame)
    pivot = frame.pivot_table(
        index="label",
        columns="cost_ratio",
        values="mean",
        aggfunc="mean",
    )
    axis = pivot.plot(kind="bar", figsize=(8.0, 4.5))
    axis.set_title(title)
    axis.set_ylabel(metric.replace("_", " "))
    axis.set_xlabel("")
    axis.legend(title="cost ratio", fontsize=8)
    axis.figure.tight_layout()
    axis.figure.savefig(path)
    plt.close(axis.figure)
    return path


def plot_metric_lines(
    tables_dir: Path,
    figures_dir: Path,
    experiment: str,
    metric: str,
    filename: str,
    title: str,
) -> Path:
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / filename
    frame = read_table(tables_dir, experiment, metric)
    if frame is None:
        placeholder_figure(path, title)
        return path

    frame = frame.copy()
    frame["label"] = _group_label(frame)
    figure, axis = plt.subplots(figsize=(8.0, 4.5))
    for label, group in frame.groupby("label"):
        group = group.sort_values("cost_ratio")
        axis.plot(
            group["cost_ratio"],
            group["mean"],
            marker="o",
            label=label,
        )
    axis.set_title(title)
    axis.set_xlabel("cost ratio c_FN / c_FP")
    axis.set_ylabel(metric.replace("_", " "))
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def plot_scaling(
    tables_dir: Path,
    figures_dir: Path,
    metric: str,
    filename: str,
    title: str,
) -> Path:
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / filename
    frame = read_table(tables_dir, "scaling", metric)
    if frame is None:
        placeholder_figure(path, title)
        return path

    frame = frame.copy()
    frame["label"] = _group_label(frame)
    figure, axis = plt.subplots(figsize=(8.0, 4.5))
    for label, group in frame.groupby("label"):
        group = group.sort_values("n_train")
        axis.plot(group["n_train"], group["mean"], marker="o", label=label)
    axis.set_title(title)
    axis.set_xlabel("training rows n")
    axis.set_ylabel(metric.replace("_", " "))
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def plot_ablation(tables_dir: Path, figures_dir: Path) -> Path:
    path = figures_dir / "ablation_rho_sensitivity.pdf"
    frame = read_table(tables_dir, "ablation", "time_to_tolerance_sec")
    if frame is None:
        placeholder_figure(path, "PA-ADMM component ablation")
        return path

    frame = frame.copy()
    frame = frame[frame["variant"].fillna("base") != "base"]
    if frame.empty:
        placeholder_figure(path, "PA-ADMM component ablation")
        return path
    grouped = frame.groupby("variant")["mean"].agg(["mean", "std"]).sort_values("mean")
    figure, axis = plt.subplots(figsize=(9.0, 4.8))
    positions = np.arange(len(grouped))
    axis.bar(
        positions,
        grouped["mean"].to_numpy(),
        yerr=grouped["std"].fillna(0.0).to_numpy(),
        capsize=3,
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(grouped.index, rotation=35, ha="right", fontsize=8)
    axis.set_ylabel("time to tolerance (s)")
    axis.set_title("PA-ADMM component ablation")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def plot_stress(tables_dir: Path, figures_dir: Path) -> Path:
    path = figures_dir / "stress_cost_under_shift.pdf"
    frame = read_table(tables_dir, "stress", "cost_per_example")
    if frame is None:
        placeholder_figure(path, "Stress tests: cost per example")
        return path

    frame = frame.copy()
    frame["label"] = _group_label(frame)
    pivot = frame.pivot_table(
        index="dataset",
        columns="label",
        values="mean",
        aggfunc="mean",
    )
    axis = pivot.plot(kind="bar", figsize=(10.0, 4.8))
    axis.set_title("Stress tests: cost per example")
    axis.set_ylabel("cost per example")
    axis.set_xlabel("")
    axis.tick_params(axis="x", rotation=30, labelsize=8)
    axis.legend(fontsize=7)
    axis.figure.tight_layout()
    axis.figure.savefig(path)
    plt.close(axis.figure)
    return path


def plot_recall_at_fpr(tables_dir: Path, figures_dir: Path) -> Path:
    path = figures_dir / "main_full_recall_at_fpr.pdf"
    first = read_table(tables_dir, "main_full", "recall_at_fpr_0.001")
    second = read_table(tables_dir, "main_full", "recall_at_fpr_0.01")
    if first is None or second is None:
        placeholder_figure(path, "Recall at fixed false-positive rates")
        return path

    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=True)
    for axis, frame, label in (
        (axes[0], first, "recall @ FPR 0.001"),
        (axes[1], second, "recall @ FPR 0.01"),
    ):
        grouped = frame.groupby("method")["mean"].mean().sort_values(ascending=False)
        axis.bar(grouped.index, grouped.to_numpy())
        axis.set_title(label)
        axis.tick_params(axis="x", rotation=25, labelsize=8)
    axes[0].set_ylabel("recall")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def _load_raw_runs(raw_root: Path, experiment: str) -> list[dict]:
    records = []
    for path in sorted(raw_root.glob(f"{experiment}/**/seed-*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if record.get("status") == "ok":
            records.append(record)
    return records


def plot_convergence(
    raw_root: Path,
    figures_dir: Path,
    experiment: str = "main_full",
    cost_ratio: int = 20,
    seed: int = 0,
) -> Path:
    path = figures_dir / f"{experiment}_convergence.pdf"
    records = [
        record
        for record in _load_raw_runs(raw_root, experiment)
        if record.get("cost_ratio") == cost_ratio
        and int(record.get("seed", -1)) == seed
        and record.get("objective_history")
    ]
    if not records:
        placeholder_figure(path, "Weighted objective vs time")
        return path

    figure, axis = plt.subplots(figsize=(8.0, 4.5))
    for record in records:
        history = np.asarray(record["objective_history"], dtype=np.float64)
        times = record.get("time_history")
        if times is not None:
            x_values = np.asarray(times, dtype=np.float64)
            x_label = "wall-clock time (s)"
        else:
            x_values = np.arange(history.size)
            x_label = "recorded iteration"
        label = record["method"]
        if record.get("variant"):
            label = f"{label}:{record['variant']}"
        axis.plot(x_values, history, label=label)
        axis.set_xlabel(x_label)
    axis.set_ylabel("weighted objective")
    axis.set_title(
        f"{experiment}: objective vs time (ratio {cost_ratio}, seed {seed})"
    )
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def plot_pr_curves(
    raw_root: Path,
    figures_dir: Path,
    experiment: str = "main_full",
    cost_ratio: int = 20,
    seed: int = 0,
) -> Path:
    path = figures_dir / f"{experiment}_pr_curves.pdf"
    records = [
        record
        for record in _load_raw_runs(raw_root, experiment)
        if record.get("cost_ratio") == cost_ratio
        and int(record.get("seed", -1)) == seed
        and record.get("pr_curve")
    ]
    if not records:
        placeholder_figure(path, "Precision-recall curves (test)")
        return path

    figure, axis = plt.subplots(figsize=(6.5, 4.8))
    for record in records:
        curve = record["pr_curve"]
        label = record["method"]
        if record.get("variant"):
            label = f"{label}:{record['variant']}"
        axis.plot(curve["recall"], curve["precision"], label=label)
    axis.set_xlabel("recall")
    axis.set_ylabel("precision")
    axis.set_title(
        f"{experiment}: test PR curves (ratio {cost_ratio}, seed {seed})"
    )
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def write_pending_manifest(tables_dir: Path) -> Path:
    rows = []
    for experiment, metrics in TABLE_METRICS.items():
        for metric in metrics:
            path = table_path(tables_dir, experiment, metric)
            if not path.exists():
                continue
            frame = read_table(tables_dir, experiment, metric)
            if frame is None:
                rows.append(
                    {
                        "experiment": experiment,
                        "artifact": path.name,
                        "status": "pending solver integration",
                        "note": "empty table; no fake numbers",
                    }
                )
    manifest = tables_dir / "_pending_solver_integration.csv"
    pd.DataFrame(
        rows,
        columns=["experiment", "artifact", "status", "note"],
    ).to_csv(manifest, index=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    raw_root = results_dir / "raw"

    created = create_empty_tables(tables_dir)
    print(f"Empty tables created: {len(created)}")

    figures: list[Path] = [
        plot_metric_bars(
            tables_dir,
            figures_dir,
            "correctness",
            "objective",
            "correctness_objective_agreement.pdf",
            "Correctness: weighted objective agreement",
        ),
        plot_metric_bars(
            tables_dir,
            figures_dir,
            "main_fast",
            "time_to_tolerance_sec",
            "main_fast_time_to_tolerance.pdf",
            "Main-fast: time to tolerance",
        ),
        plot_metric_bars(
            tables_dir,
            figures_dir,
            "main_full",
            "time_to_tolerance_sec",
            "main_full_time_to_tolerance.pdf",
            "Main-full: time to tolerance",
        ),
        plot_convergence(raw_root, figures_dir),
        plot_metric_lines(
            tables_dir,
            figures_dir,
            "main_full",
            "cost_per_example",
            "main_full_cost_vs_ratio.pdf",
            "Main-full: test cost vs cost ratio",
        ),
        plot_pr_curves(raw_root, figures_dir),
        plot_recall_at_fpr(tables_dir, figures_dir),
        plot_metric_lines(
            tables_dir,
            figures_dir,
            "cost_sweep",
            "cost_per_example",
            "cost_sweep_cost_vs_ratio.pdf",
            "Cost sweep: test cost vs cost ratio",
        ),
        plot_ablation(tables_dir, figures_dir),
        plot_stress(tables_dir, figures_dir),
        plot_scaling(
            tables_dir,
            figures_dir,
            "fit_time_sec",
            "scaling_runtime_vs_n.pdf",
            "Scaling: fit time vs n",
        ),
        plot_scaling(
            tables_dir,
            figures_dir,
            "peak_memory_mb",
            "scaling_memory_vs_n.pdf",
            "Scaling: peak memory vs n",
        ),
    ]
    manifest = write_pending_manifest(tables_dir)
    print(f"Figures written: {len(figures)}")
    print(f"Pending manifest: {manifest}")


if __name__ == "__main__":
    main()
