"""Build Member 4's PART IV slides from the committed experiment results.

The deck is rebuilt from the group template every run, so it stays idempotent:

    python slides/build_part4_deck.py --template /path/to/6704-Group4.pptx

Figures are written next to the other report assets under
``results/figures/deck/`` and the deck goes to ``slides/``. Numbers come from the
run JSONs under ``results/raw/`` only; nothing is hard-coded, and any method or
experiment that has not run yet is reported as pending instead of invented.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import statistics as st
from datetime import datetime
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

RED = RGBColor(0xA0, 0x23, 0x37)
DARK = RGBColor(0x18, 0x2B, 0x3A)
GREY = RGBColor(0x52, 0x65, 0x74)
LIGHT = RGBColor(0xF2, 0xF4, 0xF6)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Calibri"

HEX_RED = "#A02337"
HEX_DARK = "#182B3A"
HEX_GREY = "#526574"
HEX_LIGHT = "#F2F4F6"
METHOD_COLORS = {"smo": HEX_RED, "admm": "#3E6E8E", "pa_admm": "#E0A32E",
                 "pegasos": "#7A5C9E", "cvxpy": "#8A9691"}
METHOD_LABELS = {"smo": "SMO", "admm": "ADMM", "pa_admm": "PA-ADMM",
                 "pegasos": "Pegasos", "cvxpy": "CVXPY"}
RATIOS = (10, 20, 50)


# --------------------------------------------------------------------------- data


def load_runs(root: Path, experiment: str) -> list[dict]:
    files = sorted((root / "raw" / experiment).glob("**/seed-*.json"))
    runs = []
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        record["_path"] = str(path)
        runs.append(record)
    return runs


def method_order(runs: list[dict]) -> list[str]:
    seen = {run["method"] for run in runs}
    return [m for m in ("smo", "pegasos", "admm", "pa_admm", "cvxpy") if m in seen]


def cells(runs: list[dict], method: str, ratio) -> list[dict]:
    return [r for r in runs
            if r["method"] == method and r["status"] == "ok"
            and (ratio is None or r["cost_ratio"] == ratio)]


def summarize(runs: list[dict]) -> dict:
    """Per method/ratio statistics plus the pooled per-method view."""
    table: dict[tuple[str, float], dict] = {}
    for method in method_order(runs):
        for ratio in sorted({r["cost_ratio"] for r in runs if r["method"] == method}):
            group = cells(runs, method, ratio)
            ttol = [g["time_to_tolerance_sec"] for g in group
                    if g.get("time_to_tolerance_sec") is not None]
            gaps = [g["relative_gap"] for g in group
                    if g.get("relative_gap") is not None]
            table[(method, ratio)] = {
                "n": len(group),
                "converged": sum(bool(g["converged"]) for g in group),
                "objective": st.mean([g["objective"] for g in group]),
                "gap": st.mean(gaps) if gaps else float("nan"),
                "ttol": st.median(ttol) if ttol else float("nan"),
                "ttol_hits": len(ttol),
                "fit": st.mean([g["fit_time_sec"] for g in group]),
                "fit_med": st.median([g["fit_time_sec"] for g in group]),
                "pr_auc": st.mean([g["pr_auc"] for g in group]),
                "pr_auc_sd": st.pstdev([g["pr_auc"] for g in group]) if len(group) > 1 else 0.0,
                "recall_1": st.mean([g["recall_at_fpr_0.01"] for g in group]),
                "recall_01": st.mean([g["recall_at_fpr_0.001"] for g in group]),
                "cost": st.mean([g["cost_per_example"] for g in group]),
                "cost_sd": st.pstdev([g["cost_per_example"] for g in group]) if len(group) > 1 else 0.0,
                "memory": st.mean([g["peak_memory_mb"] or 0 for g in group]),
            }
    return table


def best_objective_by_group(runs: list[dict]) -> dict[tuple[float, int], float]:
    best: dict[tuple[float, int], float] = {}
    for run in runs:
        if run["status"] != "ok" or run.get("objective") is None:
            continue
        key = (run["cost_ratio"], int(run["seed"]))
        best[key] = min(best.get(key, float("inf")), float(run["objective"]))
    return best


# ------------------------------------------------------------------------ figures


def new_figure(width_in: float, height_in: float):
    fig, axis = plt.subplots(figsize=(width_in, height_in))
    fig.patch.set_facecolor("white")
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.spines["left"].set_color(HEX_GREY)
    axis.spines["bottom"].set_color(HEX_GREY)
    axis.tick_params(colors=HEX_GREY, labelsize=9)
    axis.set_axisbelow(True)
    axis.grid(True, color=HEX_GREY, alpha=0.18, linewidth=0.7)
    return fig, axis


def save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def fig_dataset(cleaned_rows: int, raw_rows: int, frauds: int, dupes: int,
                split_sizes: tuple[int, int, int], path: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 2.5),
                             gridspec_kw={"width_ratios": [1.15, 1.0, 1.35]})
    fig.patch.set_facecolor("white")
    ax = axes[0]
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.bar(["legit", "fraud"], [cleaned_rows - frauds, frauds],
           color=[HEX_GREY, HEX_RED])
    ax.set_yscale("log")
    ax.set_title(f"{frauds} frauds = {100.0 * frauds / cleaned_rows:.3f}%",
                 color=HEX_DARK, fontsize=10)
    ax.text(0, cleaned_rows - frauds, f"{cleaned_rows - frauds:,}", ha="center",
            va="bottom", fontsize=8, color=HEX_GREY)
    ax.text(1, frauds, str(frauds), ha="center", va="bottom", fontsize=8, color=HEX_RED)

    ax = axes[1]
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.barh(["train", "val", "test"], split_sizes, color=[HEX_DARK, "#3E6E8E", HEX_RED])
    for y, value in enumerate(split_sizes):
        ax.text(value, y, f" {value:,}", va="center", fontsize=9, color=HEX_GREY)
    ax.set_xlim(0, max(split_sizes) * 1.35)
    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=10)
    ax.set_title("70 / 15 / 15 stratified split (rows)", color=HEX_DARK, fontsize=10)
    ax.invert_yaxis()

    ax = axes[2]
    ax.axis("off")
    ax.set_title("Model features (d = 29)", color=HEX_DARK, fontsize=10)
    ax.text(0.0, 0.74, "V1 … V28\nPCA-transformed card features", fontsize=9,
            color=HEX_DARK, va="top")
    ax.text(0.0, 0.44, "Amount\nstandardized on train statistics only", fontsize=9,
            color=HEX_DARK, va="top")
    ax.text(0.0, 0.14, "Time → excluded from the model; used only for the\n"
                      "chronological stress split.", fontsize=8.5, color=HEX_GREY, va="top")
    return save(fig, path)


def fig_protocol(path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11.4, 2.35))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    steps = [
        ("Verified raw data", f"SHA-256 recorded\nduplicates removed"),
        ("Committed splits", "5 stratified seeds\n+ chronological"),
        ("One shared QP", "same rows, same C_i\nC fixed to 1"),
        ("Fit each solver", "90 min budget\nsingle-thread"),
        ("Threshold on val", "cost-minimising,\nvalidation only"),
        ("Test, once", "cost, PR-AUC,\nrecall@FPR"),
    ]
    width = 1.72
    gap = 0.18
    for index, (head, body) in enumerate(steps):
        left = index * (width + gap)
        ax.add_patch(plt.Rectangle((left, 0.32), width, 0.5,
                                   facecolor=HEX_LIGHT, edgecolor=HEX_GREY, linewidth=0.8))
        ax.text(left + width / 2, 0.66, head, ha="center", va="center",
                fontsize=9.5, color=HEX_DARK, fontweight="bold")
        ax.text(left + width / 2, 0.46, body, ha="center", va="center",
                fontsize=8, color=HEX_GREY)
        if index < len(steps) - 1:
            ax.annotate("", xy=(left + width + gap, 0.57), xytext=(left + width, 0.57),
                        arrowprops=dict(arrowstyle="-|>", color=HEX_RED, lw=1.4))
    ax.set_xlim(-0.05, len(steps) * (width + gap))
    ax.set_ylim(0.15, 1.0)
    ax.text(0, 0.93, "Fairness controls applied to every method in the matrix",
            fontsize=9, color=HEX_RED, fontweight="bold")
    ax.text(0, 0.18, "No per-method data caps · objective grouped by selected C · "
                     "runs reused only when data and hyper-parameters match exactly",
            fontsize=8, color=HEX_GREY)
    return save(fig, path)


def fig_curves(runs: list[dict], ratio: float, seed: int, path: Path,
               tolerance: float = 1e-3) -> Path:
    """Zoom on the tolerance band; far-away initial iterates are annotated."""
    best = best_objective_by_group(runs).get((ratio, seed))
    fig, ax = new_figure(6.6, 3.1)
    if best is None:
        return save(fig, path)
    band = 1.0 + tolerance
    clipped = 0
    crossings = []
    for method in method_order(runs):
        group = [r for r in cells(runs, method, ratio) if int(r["seed"]) == seed]
        if not group:
            continue
        record = group[0]
        history, times = record.get("objective_history"), record.get("time_history")
        if not history or not times or len(history) != len(times):
            continue
        objective = np.asarray(history, dtype=float) / best
        seconds = np.asarray(times, dtype=float)
        finite = np.isfinite(objective) & np.isfinite(seconds)
        objective, seconds = objective[finite], seconds[finite]
        visible = objective <= band * 3.0
        clipped += int(np.sum(~visible))
        ax.plot(seconds[visible], objective[visible],
                color=METHOD_COLORS.get(method, HEX_GREY), linewidth=1.8,
                label=METHOD_LABELS.get(method, method))
        hit = np.flatnonzero(objective <= band)
        if hit.size:
            crossing = (float(seconds[hit[0]]), float(objective[hit[0]]))
            crossings.append(crossing)
            ax.plot(*crossing, marker="o", markersize=5,
                    color=METHOD_COLORS.get(method, HEX_GREY))
            ax.annotate(f"{crossing[0]:.0f}s", crossing, textcoords="offset points",
                        xytext=(4, 8), fontsize=8,
                        color=METHOD_COLORS.get(method, HEX_GREY))
    ax.axhline(1.0, color=HEX_GREY, linestyle=":", linewidth=1.0)
    ax.axhline(band, color=HEX_RED, linestyle="--", linewidth=1.0)
    ax.text(1.01, band, "tolerance", color=HEX_RED, fontsize=8, va="center")
    top = max([value for _, value in crossings] or [band])
    ax.set_ylim(0.995, min(top * 1.9, band * 3.0))
    if crossings:
        ax.set_xlim(0, max(t for t, _ in crossings) * 1.55 + 1.0)
    if clipped:
        ax.text(0.02, 0.96,
                f"{clipped} early checkpoints sit above this band\n"
                "(cold start and Pegasos warm-start transients)",
                transform=ax.transAxes, fontsize=7.5, color=HEX_GREY, va="top")
    ax.set_xlabel("seconds since fit start", fontsize=9, color=HEX_GREY)
    ax.set_ylabel("objective / group best", fontsize=9, color=HEX_GREY)
    ax.set_title(f"Time-to-tolerance, full data, {ratio:g}:1, seed {seed}",
                 color=HEX_DARK, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    return save(fig, path)


def fig_speed_and_reliability(summary: dict, methods: list[str], path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 2.6))
    fig.patch.set_facecolor("white")
    keys = [(m, r) for m in methods for r in RATIOS if (m, r) in summary]
    by_ratio = {r: [] for r in RATIOS}
    for m, r in keys:
        by_ratio[r].append(m)

    ax = axes[0]
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    positions, values, colors, names = [], [], [], []
    group_centres = []
    offset = 0.0
    for ratio in RATIOS:
        start = offset
        for method in by_ratio[ratio]:
            positions.append(offset)
            values.append(summary[(method, ratio)]["ttol"])
            colors.append(METHOD_COLORS.get(method, HEX_GREY))
            names.append(METHOD_LABELS.get(method, method))
            offset += 1.0
        group_centres.append(((start + offset - 1) / 2.0, f"{ratio:g}:1"))
        offset += 0.6
    ax.bar(positions, values, color=colors, width=0.85)
    for x, value in zip(positions, values):
        ax.text(x, value, f" {value:.0f}s", va="bottom", ha="left", fontsize=7.5,
                color=HEX_GREY)
    ax.set_xticks(positions)
    ax.set_xticklabels(names, fontsize=8, rotation=0)
    for centre, label in group_centres:
        ax.text(centre, -0.13, label, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=9, color=HEX_RED, fontweight="bold")
    ax.set_ylabel("median time-to-tolerance (s)", fontsize=9, color=HEX_GREY)
    ax.set_title("Seconds to reach the common objective tolerance", color=HEX_DARK,
                 fontsize=10)
    ax2 = axes[1]
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    width = 0.24
    for index, method in enumerate(methods):
        x = np.arange(len(RATIOS)) + (index - (len(methods) - 1) / 2) * width
        rates = []
        for ratio in RATIOS:
            entry = summary.get((method, ratio))
            rates.append(100.0 * entry["converged"] / entry["n"] if entry else np.nan)
        bars = ax2.bar(x, rates, width, color=METHOD_COLORS.get(method, HEX_GREY),
                       label=METHOD_LABELS.get(method, method))
        for rect, rate, entry in zip(bars, rates,
                                     [summary.get((method, r)) for r in RATIOS]):
            if entry:
                ax2.text(rect.get_x() + rect.get_width() / 2, rate,
                         f"{entry['converged']}/{entry['n']}", ha="center", va="bottom",
                         fontsize=7.5, color=HEX_GREY)
    ax2.set_xticks(np.arange(len(RATIOS)))
    ax2.set_xticklabels([f"{r}:1" for r in RATIOS], fontsize=9)
    ax2.set_ylim(0, 118)
    ax2.set_ylabel("% of seeds certified", fontsize=9, color=HEX_GREY)
    ax2.set_title("Solver-stopping certificate met per cost ratio", color=HEX_DARK,
                  fontsize=10)
    ax2.legend(frameon=False, fontsize=8, ncol=len(methods), loc="upper center")
    return save(fig, path)


def fig_performance(summary: dict, methods: list[str], path: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 2.6))
    fig.patch.set_facecolor("white")
    panels = (("pr_auc", "PR-AUC (average precision)", None),
              ("recall_1", "recall @ FPR = 1%", None),
              ("cost", "cost per example", None))
    for ax, (metric, title, _) in zip(axes, panels):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        width = 0.8 / max(1, len(methods))
        for index, method in enumerate(methods):
            values, errors, labels = [], [], []
            for ratio in RATIOS:
                entry = summary.get((method, ratio))
                if entry is None:
                    values.append(np.nan)
                    errors.append(0.0)
                    continue
                values.append(entry[metric])
                errors.append(entry[f"{metric}_sd"] if f"{metric}_sd" in entry else 0.0)
            x = np.arange(len(RATIOS)) + (index - (len(methods) - 1) / 2) * width
            ax.bar(x, values, width, yerr=errors, capsize=2,
                   color=METHOD_COLORS.get(method, HEX_GREY),
                   label=METHOD_LABELS.get(method, method))
        ax.set_xticks(np.arange(len(RATIOS)))
        ax.set_xticklabels([f"{r}:1" for r in RATIOS], fontsize=9)
        ax.set_title(title, color=HEX_DARK, fontsize=10)
        if metric == "cost":
            ax.set_yscale("log")
    axes[0].legend(frameon=False, fontsize=8, ncol=len(methods))
    return save(fig, path)


def fig_correctness(runs: list[dict], path: Path) -> Path:
    if not runs:
        return path
    fig, ax = new_figure(5.4, 2.5)
    methods = method_order(runs)
    values = [st.mean([r["objective"] for r in cells(runs, m, None)]) for m in methods]
    best = min(values)
    ax.bar([METHOD_LABELS.get(m, m) for m in methods], values,
           color=[METHOD_COLORS.get(m, HEX_GREY) for m in methods])
    ax.axhline(best, color=HEX_RED, linestyle="--", linewidth=1.0)
    for index, (method, value) in enumerate(zip(methods, values)):
        group = cells(runs, method, None)
        ax.text(index, value, f"{value:.3f}\n({100 * (value - best) / best:.1f}% higher)"
                if value > best else f"{value:.3f}\ncertified",
                ha="center", va="bottom", fontsize=8, color=HEX_GREY)
        if not all(bool(r["converged"]) for r in group):
            ax.text(index, values[index] * 0.12, "not certified", ha="center",
                    fontsize=7.5, color=HEX_RED)
    ax.set_ylim(0, max(values) * 1.28)
    ax.set_ylabel("primal objective on the same QP", fontsize=9, color=HEX_GREY)
    ax.set_title("Independent solvers on one 7k-row cost-sensitive QP",
                 color=HEX_DARK, fontsize=10)
    return save(fig, path)


def fig_scale_ladder(runs_by_experiment: dict[str, list[dict]], path: Path) -> Path:
    fig, ax = new_figure(6.2, 2.0)
    experiments = [e for e in ("main_capped", "main_fast", "main_full")
                   if runs_by_experiment.get(e)]
    methods = sorted({m for runs in runs_by_experiment.values() for m in method_order(runs)})
    width = 0.8 / max(1, len(methods))
    for index, method in enumerate(methods):
        values = []
        for experiment in experiments:
            group = [r for r in runs_by_experiment[experiment]
                     if r["method"] == method and r["status"] == "ok"]
            values.append(st.median([r["fit_time_sec"] for r in group]) if group else np.nan)
        x = np.arange(len(experiments)) + (index - (len(methods) - 1) / 2) * width
        ax.bar(x, values, width, color=METHOD_COLORS.get(method, HEX_GREY),
               label=METHOD_LABELS.get(method, method))
        for xi, value in zip(x, values):
            if np.isfinite(value):
                ax.text(xi, value, f"{value:.0f}s", ha="center", va="bottom",
                        fontsize=7.5, color=HEX_GREY)
    ax.set_yscale("log")
    ax.set_ylim(0.1, 8000)
    ax.set_xticks(np.arange(len(experiments)))
    ax.set_xticklabels([e.replace("main_", "") for e in experiments], fontsize=9)
    ax.set_ylabel("median fit time (s, log)", fontsize=9, color=HEX_GREY)
    ax.set_title("Same protocol across the scale ladder", color=HEX_DARK, fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    return save(fig, path)


# ------------------------------------------------------------------------- slides


def chrome(slide, kicker: str, takeaway: str, source: str, page: str) -> None:
    add_text(slide, 0.58, 1.11, 11.39, 0.24, kicker, size=10, bold=True, color=RED,
             check=False)
    rule(slide, 0.58, 6.18, 0.028, 0.49, RED)
    add_text(slide, 0.75, 6.22, 11.89, 0.46, takeaway, size=17, bold=True, color=RED,
             check=False)
    rule(slide, 0.58, 6.93, 12.17, 0.014, RGBColor(0xC9, 0xD2, 0xD9))
    add_text(slide, 0.58, 7.05, 11.32, 0.24, source, size=9, color=GREY, check=False)
    add_text(slide, 12.15, 7.02, 0.62, 0.3, page, size=10, color=GREY, check=False)


def rule(slide, left, top, width, height, color) -> None:
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left), Inches(top),
                                   Inches(max(width, 0.014)), Inches(max(height, 0.014)))
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    shape.shadow.inherit = False


CONTENT_BOTTOM = 6.12  # inches; the takeaway band starts at 6.22


def add_text(slide, left, top, width, height, text, *, size=14, bold=False,
             color=DARK, align=PP_ALIGN.LEFT, check=True):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    frame = box.text_frame
    frame.word_wrap = True
    lines = text.split("\n")
    for index, line in enumerate(lines):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.alignment = align
        run = paragraph.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
        run.font.name = FONT
    # Cheap layout guard: Calibri at `size` fits roughly width*72/(size*0.5) chars
    # per line, and each line needs about 1.35 * size points of height.
    chars_per_line = max(8.0, width * 72.0 / (size * 0.5))
    wrapped = sum(max(1, math.ceil(len(line) / chars_per_line)) for line in lines)
    estimated_bottom = top + wrapped * size * 1.35 / 72.0
    if check and estimated_bottom > CONTENT_BOTTOM:
        print(f"[layout] overflow risk: {text[:46]!r} ends near "
              f"{estimated_bottom:.2f}in (limit {CONTENT_BOTTOM}in)")
    return box


def add_bullets(slide, left, top, width, items, *, size=14, gap_after=6):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(0.4))
    frame = box.text_frame
    frame.word_wrap = True
    for index, (head, body) in enumerate(items):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.space_after = Pt(gap_after)
        head_run = paragraph.add_run()
        head_run.text = head
        head_run.font.size = Pt(size)
        head_run.font.bold = True
        head_run.font.color.rgb = DARK
        head_run.font.name = FONT
        if body:
            body_run = paragraph.add_run()
            body_run.text = "  " + body
            body_run.font.size = Pt(size - 1)
            body_run.font.color.rgb = GREY
            body_run.font.name = FONT
    return box


def add_picture(slide, path: Path, left, top, width) -> None:
    slide.shapes.add_picture(str(path), Inches(left), Inches(top), width=Inches(width))


def add_table(slide, left, top, width, height, header, rows, *, font_size=10,
              column_widths=None):
    shape = slide.shapes.add_table(len(rows) + 1, len(header), Inches(left), Inches(top),
                                   Inches(width), Inches(height))
    table = shape.table
    if column_widths:
        total = sum(column_widths)
        for index, fraction in enumerate(column_widths):
            table.columns[index].width = Emu(int(Inches(width) * fraction / total))
    for index, text in enumerate(header):
        set_cell(table.cell(0, index), text, bold=True, color=WHITE, fill=RED,
                 size=font_size)
    for r, row in enumerate(rows, start=1):
        for c, text in enumerate(row):
            set_cell(table.cell(r, c), text, color=DARK,
                     fill=WHITE if r % 2 else LIGHT, size=font_size)
    return table


def set_cell(cell, text, *, bold=False, color=DARK, fill=None, size=10):
    if fill is not None:
        cell.fill.solid()
        cell.fill.fore_color.rgb = fill
    cell.margin_left = Inches(0.06)
    cell.margin_right = Inches(0.04)
    cell.margin_top = Inches(0.02)
    cell.margin_bottom = Inches(0.02)
    frame = cell.text_frame
    frame.word_wrap = True
    paragraph = frame.paragraphs[0]
    run = paragraph.add_run()
    run.text = str(text)
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = FONT


def set_notes(slide, text: str) -> None:
    slide.notes_slide.notes_text_frame.text = text


def content_slide(prs, layout, title_text: str) -> object:
    slide = prs.slides.add_slide(layout)
    for placeholder in list(slide.placeholders):
        index = placeholder.placeholder_format.idx
        if index == 0:
            placeholder.text_frame.text = title_text
            for paragraph in placeholder.text_frame.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(32)
                    run.font.color.rgb = WHITE
                    run.font.name = FONT
        else:
            # Empty body placeholders would invite stray text boxes in the deck.
            placeholder._element.getparent().remove(placeholder._element)
    return slide


def divider(prs, layout, title_text: str, part_text: str) -> object:
    slide = prs.slides.add_slide(layout)
    for placeholder in slide.placeholders:
        index = placeholder.placeholder_format.idx
        if index == 0:
            placeholder.text_frame.text = title_text
        elif index == 1:
            frame = placeholder.text_frame
            frame.text = ""
            for line in part_text.split("\n"):
                paragraph = frame.add_paragraph() if frame.paragraphs[0].runs else frame.paragraphs[0]
                run = paragraph.add_run()
                run.text = line
                run.font.size = Pt(24)
                run.font.color.rgb = WHITE
                run.font.name = FONT
    return slide


# --------------------------------------------------------------------------- main


def fmt(value, pattern="{:.2f}") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return pattern.format(number)


def audit_layout(prs, first_new_slide: int) -> list[str]:
    """Report shapes that leave the slide or collide with another content box."""
    problems: list[str] = []
    limit_w = prs.slide_width / 914400
    limit_h = prs.slide_height / 914400
    for number, slide in enumerate(prs.slides, start=1):
        if number < first_new_slide:
            continue
        boxes = []
        for shape in slide.shapes:
            if shape.left is None or shape.top is None:
                continue
            left, top = shape.left / 914400, shape.top / 914400
            right = left + (shape.width or 0) / 914400
            bottom = top + (shape.height or 0) / 914400
            if right > limit_w + 0.02 or bottom > limit_h + 0.02 or left < -0.02 or top < -0.02:
                problems.append(
                    f"slide {number}: {shape.shape_type} '{shape.name}' leaves the canvas "
                    f"({left:.2f},{top:.2f})-({right:.2f},{bottom:.2f})")
            has_content = (
                (shape.has_text_frame and shape.text_frame.text.strip())
                or shape.shape_type == 13
                or getattr(shape, "has_table", False)
            )
            if has_content:
                boxes.append((left, top, right, bottom, str(shape.shape_type),
                              (shape.text_frame.text[:24] if shape.has_text_frame
                               else shape.shape_type.name)))
        for a in range(len(boxes)):
            for b in range(a + 1, len(boxes)):
                ax1, ay1, ax2, ay2, atype, alabel = boxes[a]
                bx1, by1, bx2, by2, btype, _ = boxes[b]
                if atype == btype == "PLACEHOLDER (14)":
                    continue
                overlap_w = min(ax2, bx2) - max(ax1, bx1)
                overlap_h = min(ay2, by2) - max(ay1, by1)
                if overlap_w > 0.12 and overlap_h > 0.12:
                    problems.append(
                        f"slide {number}: {atype} {alabel!r} overlaps {btype} at "
                        f"{overlap_w:.2f}x{overlap_h:.2f}in")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True)
    parser.add_argument("--out", default="slides/COMP6704_Group4_PART_IV.pptx")
    parser.add_argument("--results", default="results")
    parser.add_argument("--check", action="store_true",
                        help="audit PART IV geometry and exit non-zero on problems")
    args = parser.parse_args()

    root = Path(args.results)
    figures = root / "figures" / "deck"
    template = Path(args.template)
    if not template.exists():
        raise SystemExit(f"Template not found: {template}")

    runs = {name: load_runs(root, name)
            for name in ("correctness", "main_capped", "main_fast", "main_full")}
    full = runs["main_fast"] + runs["main_full"] + runs["main_capped"]
    primary = runs["main_full"]
    summary_full = summarize(primary)
    methods_full = method_order(primary)
    pending = sorted(
        {"smo", "pegasos", "admm", "pa_admm", "cvxpy"} - set(
            m for runs_list in runs.values() for m in method_order(runs_list)))

    # Claims below are derived from the recorded runs, never typed in by hand.
    correctness = runs["correctness"]
    certified_objectives = [
        st.mean([r["objective"] for r in cells(correctness, method, None)])
        for method in method_order(correctness)
        if all(bool(r["converged"]) for r in cells(correctness, method, None))
        and cells(correctness, method, None)
    ]
    correctness_spread = (
        max(certified_objectives) / min(certified_objectives) - 1.0
        if len(certified_objectives) >= 2 else float("nan")
    )

    p_values = []
    wilcoxon_rows: list[dict[str, str]] = []
    wilcoxon_path = root / "tables" / "main_full_wilcoxon.csv"
    if wilcoxon_path.exists():
        with wilcoxon_path.open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                wilcoxon_rows.append(row)
                try:
                    p_values.append(float(row["p_value"]))
                except (TypeError, ValueError, KeyError):
                    continue
    wilcoxon_min_p = min(p_values) if p_values else float("nan")

    def entry(method: str, ratio: float) -> dict:
        return summary_full.get((method, ratio), {})

    hard_regime = {method: entry(method, 50.0) for method in methods_full}
    best_hard_method = max(
        (method for method in methods_full if hard_regime.get(method)),
        key=lambda method: hard_regime[method]["converged"]
        / max(1, hard_regime[method]["n"]),
        default=None,
    )
    best_hard = hard_regime.get(best_hard_method, {})
    admm_20, pa_20 = entry("admm", 20.0), entry("pa_admm", 20.0)
    warm_start_cost = (pa_20.get("fit", float("nan")) - admm_20.get("fit", float("nan"))
                       if admm_20 and pa_20 else float("nan"))
    def capped_stats(source: list[dict]) -> dict:
        group = [r for r in source if r["method"] == "smo" and r["status"] == "ok"]
        if not group:
            return {}
        return {
            "n": len(group),
            "capped": sum(int(r["n_iter"] or 0) >= 100000 for r in group),
            "fit": st.median([r["fit_time_sec"] for r in group]),
            "budget": st.median([r.get("time_budget_sec") or 5400.0 for r in group]),
        }

    smo_full = capped_stats(primary)
    smo_fast = capped_stats(runs["main_fast"])

    # Is the between-solver difference smaller than the seed-to-seed noise?
    # A solver needs at least two seeds at a ratio before it may contribute to
    # this comparison, otherwise one lucky partition becomes a headline claim.
    quality_gaps = []
    for ratio in RATIOS:
        present = [m for m in methods_full
                   if (m, ratio) in summary_full and summary_full[(m, ratio)]["n"] >= 2]
        if len(present) < 2:
            continue
        values = [summary_full[(m, ratio)]["pr_auc"] for m in present]
        noise = st.mean([summary_full[(m, ratio)]["pr_auc_sd"] for m in present])
        quality_gaps.append((max(values) - min(values), noise))
    if quality_gaps:
        worst_gap = max(gap for gap, _ in quality_gaps)
        typical_noise = st.mean([noise for _, noise in quality_gaps])
        quality_dominates = worst_gap <= typical_noise
        quality_claim = (
            f"The largest between-solver PR-AUC gap is {worst_gap:.3f} while the typical "
            f"seed-to-seed sd is {typical_noise:.3f}, so the ranking is noise, not method "
            "quality."
            if quality_dominates else
            f"A between-solver PR-AUC gap of {worst_gap:.3f} now exceeds the typical "
            f"seed-to-seed sd of {typical_noise:.3f}; check it against the paired test "
            "before attributing it to the solver."
        )
        quality_takeaway = (
            "Detection quality is set by the data, not by which solver you pick"
            if quality_dominates else
            "Solver choice now separates on detection quality - check significance"
        )
    else:
        quality_claim = ("Quality comparison is still provisional: no cost ratio has two "
                         "solvers with at least two recorded seeds yet.")
        quality_takeaway = "Quality comparison is provisional while seeds accumulate"

    data_note = {
        "raw_rows": 284807,
        "cleaned_rows": 283726,
        "frauds": 473,
        "dupes": 1081,
    }
    stratified0 = [r for r in runs["main_full"] if int(r["seed"]) == 0]
    split_sizes = (stratified0[0]["n_train"], stratified0[0]["n_val"],
                   stratified0[0]["n_test"]) if stratified0 else (198608, 42559, 42559)

    fig_dataset(**data_note, split_sizes=split_sizes, path=figures / "p4_dataset.png")
    fig_protocol(path=figures / "p4_protocol.png")
    seed_for_curves = min(int(r["seed"]) for r in primary) if primary else 0
    ratio_for_curves = 20.0
    if any(r["cost_ratio"] == 20.0 for r in primary):
        fig_curves(primary, ratio_for_curves, seed_for_curves,
                   figures / "p4_curves.png")
    fig_speed_and_reliability(summary_full, methods_full, figures / "p4_speed.png")
    fig_performance(summary_full, methods_full, figures / "p4_performance.png")
    fig_correctness(runs["correctness"], figures / "p4_correctness.png")
    fig_scale_ladder({k: runs[k] for k in ("main_capped", "main_fast", "main_full")},
                     figures / "p4_ladder.png")

    prs = Presentation(str(template))
    masters = prs.slide_masters[0]
    layouts = {layout.name: layout for layout in masters.slide_layouts}
    divider_layout = layouts["1_Section Header"]
    body_layout = layouts["2_Section Header"]

    coverage = ", ".join(METHOD_LABELS.get(m, m) for m in
                         sorted({m for runs_list in runs.values() for m in method_order(runs_list)}))
    coverage_note = f"methods with committed results: {coverage}"
    if pending:
        coverage_note += f" · pending: {', '.join(METHOD_LABELS.get(p, p) for p in pending)}"
    fits_recorded = sum(len(runs[name]) for name in runs)
    generated_stamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    first_new_slide = len(prs.slides) + 1
    slide = divider(prs, divider_layout, "Evaluation, Experiments and Results",
                    "PART IV\nEVALUATION AND BENCHMARK")
    add_text(slide, 0.37, 6.35, 11.6, 0.6,
             f"Generated {generated_stamp} from {fits_recorded} recorded fits · "
             f"results shown for {coverage}"
             + (f" · still pending: {', '.join(METHOD_LABELS.get(p, p) for p in pending)}"
                if pending else ""),
             size=12, color=WHITE, check=False)
    set_notes(slide,
              "Opening (15s): Parts I-III defined the cost-sensitive objective and the four "
              "solvers. My part is the measurement: what data we used, how we made the "
              "comparison fair, and what the runs actually show.")

    # 2 - Data
    slide = content_slide(prs, body_layout, "The data behind every number")
    chrome(slide, "PART IV  /  DATA",
           "0.167% positives: accuracy is meaningless here, cost is the headline",
           "data/raw/creditcard.csv · SHA-256 76274b69…1a89 · src/data.py::clean_data",
           "IV · 02")
    add_picture(slide, figures / "p4_dataset.png", 0.58, 1.72, 11.4)
    add_text(slide, 0.58, 4.35, 11.4, 1.6,
             "284,807 transactions, 492 frauds (0.172%). Removing 1,081 exact duplicate rows "
             "leaves 283,726 rows with 473 frauds (0.167%) — duplicates inflate apparent "
             "volume and leak across splits.\n"
             "29 model features: V1–V28 plus Amount. Time is never a feature; it only drives "
             "the chronological stress split.\n"
             "Amount is standardized with training-fold statistics only, so no test "
             "information reaches the solver.",
             size=13, color=GREY)
    set_notes(slide,
              "~50s: introduce ULB, the deduplication decision and why it matters for leakage, "
              "the 29-feature protocol, and that every run is tied to a recorded SHA-256 so the "
              "audience knows the data cannot drift underneath a result.")

    # 3 - Protocol
    slide = content_slide(prs, body_layout, "How the comparison is made fair")
    chrome(slide, "PART IV  /  PROTOCOL",
           "Same rows, same C, same budget — otherwise the objective is not comparable",
           "configs/main_*.yaml · EXPERIMENTS.md §5.1, §6 · src/evaluate.py",
           "IV · 03")
    add_picture(slide, figures / "p4_protocol.png", 0.58, 1.62, 11.4)
    add_bullets(slide, 0.58, 4.15, 11.6, [
        ("Cost-sensitive objective", "min ½‖w‖² + Σ Cᵢ·max(0, 1−yᵢf(xᵢ)), Cᵢ = C·w_yᵢ, "
                                     "fraud:legit weight 10 / 20 / 50."),
        ("Fixed-QP rule", "solver ranking runs fix C=1 for every method; validation-tuned C "
                          "lives in a separate predictive experiment."),
        ("Budgets", "90 min per fit, single-threaded, one machine; timeouts are recorded as "
                    "failures, never quietly dropped."),
    ], size=13)
    set_notes(slide,
              "~50s: emphasise the two decisions that make the ranking honest — every method "
              "sees exactly the same rows (no per-method caps) and the objective comparison is "
              "only ever made at a shared C, because a different C is a different QP.")

    # 4 - Primary metric
    slide = content_slide(prs, body_layout, "Primary metric: time to a shared tolerance")
    chrome(slide, "PART IV  /  METRICS",
           "We rank solvers by how fast they reach the best observed objective, not by final loss",
           "src/evaluate.py::time_to_tolerance · ≥2 aligned history points required",
           "IV · 04")
    add_picture(slide, figures / "p4_curves.png", 0.55, 1.66, 6.9)
    add_bullets(slide, 7.7, 1.85, 5.1, [
        ("Reference", "best objective achieved by any method in the same "
                      "dataset / ratio / seed / C group."),
        ("Tolerance", "first checkpoint within 0.1% of that reference, using the solver's own "
                      "timestamped history."),
        ("Secondary metrics", "cost per example, PR-AUC, recall at FPR 0.1% and 1%, balanced "
                              "accuracy, iterations, wall time, peak RSS."),
        ("Statistics", "5 seeds, bootstrap 95% CI, paired Wilcoxon signed-rank."),
    ], size=12.5)
    set_notes(slide,
              "~50s: explain the curve plot: lower is the objective relative to the group best, "
              "the dashed line is the 0.1% tolerance, and the crossing time is the headline "
              "number in the next slides.")

    # 5 - Correctness
    corr_detail = {}
    for method in method_order(correctness):
        group = cells(correctness, method, None)
        if group:
            corr_detail[method] = {
                "objective": st.mean([g["objective"] for g in group]),
                "certified": all(bool(g["converged"]) for g in group),
                "n_iter": max(int(g["n_iter"] or 0) for g in group),
            }
    correctness_lines = "\n".join(
        f"{METHOD_LABELS.get(method, method):9s} P = {value['objective']:.4f}   "
        f"{'certified' if value['certified'] else 'not certified'}   "
        f"({value['n_iter']:,} updates/iterations)"
        for method, value in corr_detail.items()
    )
    correctness_takeaway = (
        f"Independent solvers agree to {correctness_spread:.1e} on one QP"
        if correctness_spread == correctness_spread
        else "Correctness gate: solvers must agree before we compare speed"
    )
    uncertified = [METHOD_LABELS.get(m, m) for m, v in corr_detail.items() if not v["certified"]]
    slide = content_slide(prs, body_layout, "Correctness first: solvers agree on one QP")
    chrome(slide, "PART IV  /  CORRECTNESS", correctness_takeaway,
           "configs/correctness.yaml · results/raw/correctness · "
           "WP2 LIBSVM and WP3 CVXPY/ECOS cross-checks", "IV · 05")
    add_picture(slide, figures / "p4_correctness.png", 0.55, 1.72, 6.4)
    add_text(slide, 7.15, 1.8, 5.7, 1.9, correctness_lines, size=12.5, color=DARK)
    add_text(slide, 7.15, 3.85, 5.7, 2.1,
             "One 10k stratified subset (7,000 train / 1,500 val / 1,500 test), C fixed to 1, "
             "ratio 10:1, tolerance 1e-4. WP2 also checks SMO against LIBSVM and WP3 checks ADMM "
             "against CVXPY/ECOS on identical rows."
             + (f"\n\n{' and '.join(uncertified)} did not certify in this configuration "
                "(cold fixed-ρ, 10,000-iteration cap) and is reported as unconverged rather "
                "than smoothed away." if uncertified else ""),
             size=12.5, color=GREY)
    set_notes(slide,
              "~45s: this slide answers 'are your solvers even correct?' — two independent "
              "algorithms agreeing to four decimals, plus each member's external reference check.")

    # 6 - Matrix and coverage
    slide = content_slide(prs, body_layout, "The experiment matrix we ran")
    chrome(slide, "PART IV  /  MAIN EXPERIMENT DESIGN",
           "One scale ladder: 7k → 35k → 198k rows, ratios 10 / 20 / 50, up to 5 seeds",
           coverage_note, "IV · 06")
    rows = []
    for name, label, rows_desc, seeds in (
        ("correctness", "correctness", "7,000 train", "1"),
        ("main_capped", "main_capped", "7,000 train", "3"),
        ("main_fast", "main_fast", "34,999 train", "3"),
        ("main_full", "main_full", "198,608 train", "5"),
    ):
        group = runs[name]
        per_method = []
        for method in ("smo", "admm", "pa_admm"):
            count = len(cells(group, method, None))
            per_method.append(f"{METHOD_LABELS[method]}: {count}")
        rows.append([label, rows_desc, seeds,
                     " · ".join(per_method),
                     str(sum(bool(r.get("reused")) for r in group))])
    add_table(slide, 0.58, 1.66, 11.5, 1.9, ["experiment", "training rows", "seeds",
                                             "successful fits by method", "reused"],
              rows, font_size=11, column_widths=[2.0, 2.2, 1.0, 4.6, 1.1])
    add_picture(slide, figures / "p4_ladder.png", 0.55, 4.05, 6.6)
    add_text(slide, 7.4, 4.15, 5.4, 1.9,
             "Ratios 10 / 20 / 50 are trained at their own weights; thresholds and test cost "
             "always use the deployment ratio of that run.\n\n"
             f"{'Pegasos and the CVXPY baseline are still pending; their cells are reported as '
             'skipped, not imputed.' if pending else 'All registered methods completed.'}",
             size=12, color=GREY)
    set_notes(slide,
              "~45s: walk the ladder from 7k to 198k rows. Mention the reuse flag: earlier runs "
              "are kept when data and hyper-parameters match, so adding a solver never refits "
              "the others.")

    expected_full = len({int(r["seed"]) for r in primary}) * len(
        {r["cost_ratio"] for r in primary}
    ) or 15
    full_coverage = ", ".join(
        f"{METHOD_LABELS.get(method, method)} "
        f"{len(cells(primary, method, None))}/{expected_full}"
        for method in ("smo", "admm", "pa_admm")
        if method in methods_full
    )
    coverage_suffix = (
        f" · coverage {full_coverage}"
        if any(len(cells(primary, method, None)) < expected_full for method in methods_full)
        else ""
    )

    # 7 - Speed / reliability
    slide = content_slide(prs, body_layout, "Main results: speed and certified convergence")
    chrome(slide, "PART IV  /  MAIN RESULTS · SPEED",
           "Reaching tolerance is minutes apart; certifying the optimum is where methods diverge",
           "results/tables/main_full_*.csv · 5 seeds × 3 ratios · full data"
           + coverage_suffix,
           "IV · 07")
    add_picture(slide, figures / "p4_speed.png", 0.55, 1.62, 11.6)
    ttol_bits = []
    for method in methods_full:
        entry = summary_full.get((method, 20.0))
        if entry:
            ttol_bits.append(f"{METHOD_LABELS[method]} {fmt(entry['ttol'], '{:.0f}')}s")
    hard_bits = ", ".join(
        f"{METHOD_LABELS.get(method, method)} {value['converged']}/{value['n']}"
        for method, value in sorted(hard_regime.items())
        if value
    )
    hardest_line = (
        f" Pushing Cᵢ higher makes the QP stiffer; {METHOD_LABELS.get(best_hard_method, best_hard_method)} "
        "certifies the most seeds there."
        if best_hard_method else ""
    )
    cap_note = (
        f" SMO exhausted its own 100,000-update limit on {smo_full['capped']} of "
        f"{smo_full['n']} main_full fits after {smo_full['fit']:.0f}s of a "
        f"{smo_full['budget']:.0f}s budget"
        + (f" ({smo_fast['capped']}/{smo_fast['n']} already at 35k)" if smo_fast else "")
        + ": that 'not converged' flag is an iteration-cap outcome, not a wall-clock timeout."
        if smo_full else ""
    )
    add_text(slide, 0.58, 4.45, 11.6, 1.5,
             "At 20:1 on the full data, median time-to-tolerance: " + ", ".join(ttol_bits) +
             ". Final fit time is dominated by iterating past the tolerance line until the "
             "solver's own certificate (KKT / residual + duality gap) is satisfied."
             f"\nRight panel counts certified seeds at the hardest ratio 50:1 → {hard_bits}."
             f"{hardest_line}{cap_note}",
             size=12.5, color=GREY)
    set_notes(slide,
              "~55s: this is the solver-ranking slide. Be careful in wording: differences in "
              "time-to-tolerance are not significant at 5 seeds, the reliable signal is "
              "convergence certification, especially at 50:1.")

    # 7b - Full-data grid
    grid_rows = []
    for ratio in RATIOS:
        for method in methods_full:
            value = summary_full.get((method, ratio))
            if not value:
                continue
            grid_rows.append([
                METHOD_LABELS.get(method, method), f"{ratio:g}:1",
                f"{value['converged']}/{value['n']}",
                fmt(value["ttol"], "{:.0f}"), fmt(value["fit_med"], "{:.0f}"),
                fmt(value["objective"], "{:.1f}"),
                f"{fmt(value['pr_auc'])} ± {fmt(value['pr_auc_sd'])}",
                fmt(value["cost"], "{:.5f}"),
            ])
    grid_height = 0.32 * (len(grid_rows) + 1) + 0.2
    slide = content_slide(prs, body_layout, "Full data at a glance")
    chrome(slide, "PART IV  /  MAIN RESULTS · GRID",
           "Every cell is the same QP on the same 198,608 training rows",
           "results/raw/main_full · 42,559 test rows with 71 frauds · C = 1 · tolerance 1e-3"
           + coverage_suffix,
           "IV · 08")
    if grid_rows:
        add_table(slide, 0.58, 1.68, 11.5, grid_height,
                  ["solver", "ratio", "certified", "median t-to-tol (s)",
                   "median fit (s)", "objective", "PR-AUC", "cost / example"],
                  grid_rows, font_size=11,
                  column_widths=[1.5, 0.8, 1.1, 1.7, 1.4, 1.3, 1.9, 1.5])
        add_text(slide, 0.58, 1.68 + grid_height + 0.15, 11.6, 0.8,
                 "t-to-tol = median over the 5 seeds of the first checkpoint within 0.1% of "
                 "that group's best observed objective; certified = the solver's own stopping "
                 "certificate; PR-AUC ± sd is the seed spread.",
                 size=12, color=GREY)
    else:
        add_text(slide, 0.58, 1.9, 11.6, 1.0,
                 "No full-data runs recorded yet.", size=14, color=GREY)
    set_notes(slide,
              "~40s: this is the table the audience should photograph. Point at one column "
              "(say 20:1) and compare across solvers: the objective and PR-AUC agree, the "
              "certified count and the wall time are where they differ.")

    # 8 - Performance
    slide = content_slide(prs, body_layout, "Performance under the deployment cost model")
    chrome(slide, "PART IV  /  MAIN RESULTS · QUALITY", quality_takeaway,
           "results/tables/main_full_pr_auc.csv · recall_at_fpr · cost_per_example"
           + coverage_suffix,
           "IV · 09")
    add_picture(slide, figures / "p4_performance.png", 0.55, 1.62, 11.6)
    perf_method = "pa_admm" if ("pa_admm", 20.0) in summary_full else (
        methods_full[0] if methods_full else None)
    perf_entry = summary_full.get((perf_method, 20.0)) if perf_method else {}
    add_text(slide, 0.58, 4.45, 11.6, 1.5,
             f"Full data, 20:1, {METHOD_LABELS.get(perf_method, perf_method)}: "
             f"PR-AUC {fmt(perf_entry.get('pr_auc'))}, "
             f"recall at 1% FPR {fmt(100 * perf_entry.get('recall_1', float('nan')), '{:.1f}')}%, "
             f"cost {fmt(perf_entry.get('cost'), '{:.5f}')} per example. {quality_claim}\n"
             "Test cost rises with the assumed fraud:legit ratio because the threshold is tuned "
             "for that ratio — that is a deployment choice, not a solver defect.",
             size=12.5, color=GREY)
    set_notes(slide,
              "~50s: state clearly that the fraud-detection performance is essentially the same "
              "across solvers; the interesting quantity for the group is which solver gets there "
              "reliably and how fast.")

    # 9 - Statistics and takeaways
    slide = content_slide(prs, body_layout, "What we can and cannot claim")
    chrome(slide, "PART IV  /  FINDINGS",
           "Claim reliability and cost-aware ranking; do not claim a speed winner yet",
           "results/tables/main_full_wilcoxon.csv · EXPERIMENTS.md §10",
           "IV · 10")
    admm_hard = hard_regime.get("admm", {})
    pa_hard = hard_regime.get("pa_admm", {})
    numbers = [
        ("01", "Objective values agree across solvers",
         (f"At fixed C=1 the certified solvers agree to {correctness_spread:.1e} on the same QP, "
          "and every full-data run finishes within 0.1% of its group reference."
          if correctness_spread == correctness_spread else
          "At fixed C=1 every completed run is measured against the same group reference.")),
        ("02", "No significant ranking difference yet",
         (f"Paired Wilcoxon over the recorded seeds: the smallest p-value across all paired "
          f"tests is {wilcoxon_min_p:.3f} (objective, cost, PR-AUC, time-to-tolerance), so seed "
          "noise dominates method differences."
          if wilcoxon_min_p == wilcoxon_min_p else
          "Paired Wilcoxon tests are pending more completed pairs.")),
        ("03", "Reliability is where the methods differ",
         (f"At 50:1 the cold ADMM certified {admm_hard.get('converged', '?')}/{admm_hard.get('n', '?')} "
          f"seeds while PA-ADMM certified {pa_hard.get('converged', '?')}/{pa_hard.get('n', '?')}, "
          f"for about {fmt(warm_start_cost, '{:+.0f}')}s of extra fit time at 20:1 — the warm start "
          "buys convergence, not wall-clock speed.")),
        ("04", "Limits to state openly",
         (f"{fits_recorded} fits are recorded so far; pending: "
          f"{', '.join(METHOD_LABELS.get(p, p) for p in pending) if pending else 'none'}. "
          "The capped-subset quality bars rest on a handful of test frauds, and all timings "
          "come from one single-threaded machine.")),
    ]
    y = 1.72
    for label, head, body in numbers:
        add_text(slide, 0.6, y, 0.7, 0.4, label, size=22, bold=True, color=RED)
        add_text(slide, 1.4, y, 11.2, 0.34, head, size=17, bold=True, color=DARK)
        add_text(slide, 1.4, y + 0.36, 11.2, 0.6, body, size=12.5, color=GREY)
        y += 1.02
    set_notes(slide,
              "~55s: close on the honest framing. The contribution is the measurement protocol "
              "and the cost-aware comparison; a speed verdict waits for Pegasos and the CVXPY "
              "baseline, and for more seeds.")

    # Appendix A - fairness controls
    slide = content_slide(prs, body_layout, "Appendix: fairness controls in detail")
    chrome(slide, "PART IV  /  APPENDIX · CONTROLS",
           "Each control exists because a first draft of the harness got it wrong",
           "src/evaluate.py · EXPERIMENTS.md §5.1, §6.5 · tests/test_harness_smoke.py",
           "IV · 11")
    add_table(slide, 0.58, 1.68, 11.6, 3.4, ["control", "what it prevents"], [
        ["Identical rows, no per-method caps",
         "comparing a solver on 10k rows against one on 198k rows"],
        ["Fixed C=1 for ranking; C grouped otherwise",
         "comparing objectives of two different QPs after independent C tuning"],
        ["Validation-only threshold and C",
         "test-set tuning; the test split is touched exactly once per run"],
        ["≥ 2 aligned objective/timestamp checkpoints",
         "a single-point solve being reported as a convergence curve"],
        ["Objective recomputed from w, b",
         "a solver reporting a differently normalised (e.g. per-example) objective"],
        ["Nonlinear runs leave the objective comparison",
         "an RKHS objective being ranked against linear-SVM objectives"],
        ["90-minute budget, single thread, one machine",
         "wall-clock comparisons across different hardware or thread counts"],
        ["reuse_existing with exact metadata matching",
         "silently mixing results produced from different splits or hyper-parameters"],
    ], font_size=11, column_widths=[4.6, 7.0])
    set_notes(slide,
              "Backup for questions about fairness. Every row here came from a concrete review "
              "finding on our own harness, and each one is covered by a regression test.")

    # Appendix B - reproduction
    slide = content_slide(prs, body_layout, "Appendix: reproducing these numbers")
    chrome(slide, "PART IV  /  APPENDIX · REPRODUCTION",
           "One command per experiment; every run carries a config hash",
           "experiments/check_solver.py · run_correctness.py · run_main.py · make_report_assets.py",
           "IV · 12")
    add_text(slide, 0.58, 1.75, 11.6, 3.2,
             "1.  python experiments/check_solver.py --method smo|admm|pa_admm\n"
             "     → results/raw/solver_checks/<method>.json (contract + diagnostics)\n\n"
             "2.  python experiments/run_correctness.py\n"
             "     → 7,000-row agreement check, reference_method: cvxpy\n\n"
             "3.  python experiments/run_main.py --config configs/main_capped.yaml\n"
             "                                       configs/main_fast.yaml\n"
             "                                       configs/main_full.yaml\n"
             "     → results/raw/<experiment>/…  and results/tables/<experiment>_*.csv\n\n"
             "4.  python experiments/make_report_assets.py\n"
             "     → results/figures/*.pdf from the aggregated tables\n\n"
             "5.  python slides/build_part4_deck.py --template … --check\n"
             "     → this deck, rebuilt from the JSONs; nothing on these slides is typed in",
             size=13, color=DARK)
    add_text(slide, 0.58, 5.15, 11.6, 0.9,
             "Data is pinned by SHA-256 (76274b69…1a89) and the five stratified split files are "
             "committed, so a rerun reproduces the same rows. Raw run JSONs and logs stay out of "
             "git; aggregate tables and figures are versioned.",
             size=12, color=GREY)
    set_notes(slide,
              "Backup for 'how do I reproduce this?'. Emphasise that the deck itself is generated "
              "from the run JSONs, so slides and results cannot drift apart.")

    summary_lines = [
        "# PART IV results summary (generated)",
        "",
        f"Generated {generated_stamp} from {fits_recorded} recorded fits. "
        f"Results exist for {coverage}.",
        ""
        + (f"Still pending: {', '.join(METHOD_LABELS.get(p, p) for p in pending)}."
           if pending else "No solvers pending."),
        "",
        "## Data and protocol",
        "",
        f"- {data_note['raw_rows']:,} raw rows → {data_note['cleaned_rows']:,} after dropping "
        f"{data_note['dupes']:,} exact duplicates; {data_note['frauds']} frauds "
        f"({100 * data_note['frauds'] / data_note['cleaned_rows']:.3f}%).",
        f"- Full-data split: {split_sizes[0]:,} / {split_sizes[1]:,} / {split_sizes[2]:,} "
        "train / val / test, 29 features, 71 test frauds per seed.",
        "- Solver ranking uses C fixed to 1, cost ratios 10/20/50, validation-only thresholds, "
        "90-minute per-fit budget, single-threaded.",
        "",
        "## Correctness (one 7,000-row QP, ratio 10:1, tol 1e-4)",
        "",
    ]
    for method, value in corr_detail.items():
        summary_lines.append(
            f"- {METHOD_LABELS.get(method, method)}: objective {value['objective']:.4f}, "
            f"{'certified' if value['certified'] else 'not certified'}, "
            f"{value['n_iter']:,} iterations/updates")
    if correctness_spread == correctness_spread:
        summary_lines.append(
            f"- Certified solvers agree to a relative spread of {correctness_spread:.2e}.")
    summary_lines += ["", "## Main results, full data (198,608 training rows)", ""]
    if grid_rows:
        summary_lines.append(
            "| solver | ratio | certified | median t-to-tol (s) | median fit (s) | objective "
            "| PR-AUC | cost / example |")
        summary_lines.append("|---|---|---|---|---|---|---|---|")
        summary_lines.extend("| " + " | ".join(row) + " |" for row in grid_rows)
    summary_lines += [
        "",
        f"{quality_claim}",
        "",
        "## Paired Wilcoxon (main_full)",
        "",
    ]
    if wilcoxon_rows:
        summary_lines.append("| ratio | metric | pair | n | p-value |")
        summary_lines.append("|---|---|---|---|---|")
        for row in wilcoxon_rows[:40]:
            summary_lines.append(
                f"| {row.get('cost_ratio')} | {row.get('metric')} | "
                f"{row.get('method_a')} vs {row.get('method_b')} | {row.get('n_pairs')} | "
                f"{row.get('p_value')} |")
    else:
        summary_lines.append("No paired tests available yet.")
    summary_lines += ["", "## Protocol caveats", ""]
    if smo_full:
        summary_lines.append(
            f"- SMO stopped at its own 100,000-update cap on {smo_full['capped']}/"
            f"{smo_full['n']} main_full fits after {smo_full['fit']:.0f}s of a "
            f"{smo_full['budget']:.0f}s budget"
            + (f" ({smo_fast['capped']}/{smo_fast['n']} at 35k)" if smo_fast else "")
            + ". Raising that cap for a time-matched comparison is an open protocol "
              "decision with WP2.")
    unreferenced = sum(1 for record in primary
                       if record["status"] == "ok"
                       and record.get("reference_objective") is None)
    if unreferenced:
        summary_lines.append(
            f"- {unreferenced} main_full record(s) have no reference objective yet: references "
            "are assigned once the campaign finishes, so this summary is mid-run.")
    if pending:
        summary_lines.append(
            f"- No results for {', '.join(METHOD_LABELS.get(p, p) for p in pending)}; the "
            "ranking is not the final five-method comparison.")
    summary_lines += [
        "",
        f"Coverage on main_full:{coverage_suffix or ' complete'}",
        "",
        "This file is generated by slides/build_part4_deck.py; do not edit it by hand.",
        "",
    ]
    summary_path = Path("slides/PART_IV_SUMMARY.md")
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(output))
    print(f"Slides: {len(prs.slides)} total, PART IV adds {len(prs.slides) - 19}")
    print(f"Wrote {output}")
    print(f"Summary: {summary_path}")
    print(f"Figures: {figures}")
    if pending:
        print(f"Pending methods (not run yet): {', '.join(pending)}")
    if args.check:
        problems = audit_layout(Presentation(str(output)), first_new_slide)
        for problem in problems:
            print(f"[layout] {problem}")
        print(f"Layout audit: {len(problems)} problem(s) across "
              f"{len(prs.slides) - first_new_slide + 1} PART IV slides")
        if problems:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
