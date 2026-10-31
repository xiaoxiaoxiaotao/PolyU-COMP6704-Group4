"""Regenerate report tables and aggregate figures from the supplied run CSVs.

Run from this directory: python make_report_assets.py
The archived convergence figure is copied from the original project and is not
regenerated here because its per-iteration JSON source is absent from the upload.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
FIG = ROOT / 'figures'
TAB = ROOT / 'tables'
for directory in (FIG, TAB):
    directory.mkdir(exist_ok=True)
METHODS = ['smo', 'pegasos', 'admm', 'pa_admm']
LABEL = dict(zip(METHODS, ['SMO', 'Pegasos', 'ADMM', 'PA-ADMM']))
COLOR = dict(zip(METHODS, ['#7E536F', '#C37B2F', '#316895', '#24796B']))
MARKER = dict(zip(METHODS, ['s', '^', 'o', 'D']))
full = pd.read_csv(DATA / 'main_full_runs.csv')
small = pd.read_csv(DATA / 'main_capped_runs.csv')
medium = pd.read_csv(DATA / 'main_fast_runs.csv')
correct = pd.read_csv(DATA / 'correctness_runs.csv')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'axes.titlesize': 9, 'axes.labelsize': 9,
                     'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
                     'legend.fontsize': 8.5, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42,
                     'savefig.bbox': 'tight', 'axes.linewidth': .65})

def pm(values, decimals=2):
    values = pd.to_numeric(values, errors='coerce').dropna()
    if not len(values):
        return r'---'
    return f'{values.mean():.{decimals}f}' + r' $\pm$ ' + f'{values.std(ddof=1):.{decimals}f}'

def save(fig, name):
    fig.savefig(FIG / f'{name}.pdf')
    fig.savefig(FIG / f'{name}.png', dpi=200)
    plt.close(fig)

# Compact correctness table, in the same method order as every comparison.
rows = []
for method in METHODS:
    r = correct[correct.method.eq(method)].iloc[0]
    gap = float(r.relative_gap) * 100
    reached = bool(r.reached_tolerance)
    ttt = f'{r.time_to_tolerance_sec:.3f}' if reached else r'---'
    rows.append(f"{LABEL[method]} & {r.objective:.6f} & {gap:.4f} & "
                f"{r.fit_time_sec:.3f} & {ttt} & {'Yes' if r.converged else 'No'} " + r'\\')
(TAB / 'correctness.tex').write_text('\n'.join(rows) + '\n' + r'\bottomrule' + '\n')

# Full-data optimization: mean and sample SD where relevant.
rows = []
for ratio in [10,20,50]:
    for method in METHODS:
        g = full[full.cost_ratio.eq(ratio) & full.method.eq(method)]
        gap = g.relative_gap.mean() * 100
        rows.append(f'{ratio}:1 & {LABEL[method]} & {gap:.4f} & '
                    f"{pm(g.time_to_tolerance_sec,1)} & {pm(g.fit_time_sec,1)} & "
                    f"{int(g.reached_tolerance.sum())}/5 & {int(g.converged.sum())}/5 " + r'\\')
    if ratio != 50:
        rows.append(r'\addlinespace[3pt]')
(TAB / 'optimization.tex').write_text('\n'.join(rows) + '\n' + r'\bottomrule' + '\n')

# Prediction table, using cost per 1,000 examples for readable decimal places.
rows = []
for ratio in [10,20,50]:
    for method in METHODS:
        g = full[full.cost_ratio.eq(ratio) & full.method.eq(method)]
        rows.append(f'{ratio}:1 & {LABEL[method]} & {pm(g.pr_auc,3)} & '
                    f"{g['recall_at_fpr_0.001'].mean():.3f} & "
                    f"{g['recall_at_fpr_0.01'].mean():.3f} & "
                    f"{pm(g.cost_per_example*1000,2)} " + r'\\')
    if ratio != 50:
        rows.append(r'\addlinespace[3pt]')
(TAB / 'prediction.tex').write_text('\n'.join(rows) + '\n' + r'\bottomrule' + '\n')

# Time to common objective tolerance. Show every split, the paired design,
# and the sample-mean trend; no fabricated confidence band or censored bars.
fig, axes = plt.subplots(1,3,figsize=(6.8,2.4),sharey=True)
for ax,ratio in zip(axes,[10,20,50]):
    sub = full[full.cost_ratio.eq(ratio)]
    for seed in range(5):
        pair = [sub[(sub.method==m)&(sub.seed==seed)].time_to_tolerance_sec.iloc[0]
                for m in ['admm','pa_admm']]
        ax.plot([0,1],pair,color='#CFD3D7',lw=.8,zorder=1)
    for x,m in enumerate(['admm','pa_admm']):
        g = sub[sub.method.eq(m)].sort_values('seed')
        ax.scatter(np.full(5,x),g.time_to_tolerance_sec,s=23,
                   color=COLOR[m],marker=MARKER[m],zorder=3)
        ax.plot([x-.17,x+.17],[g.time_to_tolerance_sec.mean()]*2,
                color='black',lw=1.8,zorder=4)
    ax.set_xticks([0,1],['ADMM','PA-ADMM'])
    ax.set_xlim(-.4,1.4)
    ax.set_title(f'Cost ratio {ratio}:1',fontweight='bold')
    ax.grid(axis='y',alpha=.2)
axes[0].set_ylabel('Time to 0.1% objective tolerance (s)')
axes[0].set_ylim(0,135)
fig.tight_layout(w_pad=1.6)
save(fig,'paired_tolerance')

# Size ladder: ratio 20 only, all paired seeds; descriptive workload study.
fig,axes=plt.subplots(1,2,figsize=(6.8,2.65))
sizes=[7000,34999,198608]
for m in METHODS:
    runtimes=[]; gaps=[]
    for df in [small,medium,full]:
        g=df[df.method.eq(m)&df.cost_ratio.eq(20)]
        runtimes.append(g.fit_time_sec.median())
        gaps.append(g.relative_gap.median()*100)
    axes[0].plot(sizes,runtimes,color=COLOR[m],marker=MARKER[m],label=LABEL[m],lw=1.4,ms=4)
    axes[1].plot(sizes,gaps,color=COLOR[m],marker=MARKER[m],label=LABEL[m],lw=1.4,ms=4)
for ax in axes:
    ax.set_xscale('log')
    ax.set_xticks(sizes,['7,000','34,999','198,608'])
    ax.set_xlabel('Training rows')
    ax.grid(axis='y',alpha=.2)
axes[0].set_yscale('log')
axes[0].set_ylabel('Median total fit time (s)')
axes[0].set_title('(a) Runtime',fontweight='bold')
axes[1].set_yscale('symlog',linthresh=.0001)
axes[1].set_ylim(bottom=0)
axes[1].set_ylabel('Median final empirical gap (%)')
axes[1].set_title('(b) Returned objective quality',fontweight='bold')
axes[1].axhline(.1,color='#777777',ls='--',lw=.8)
axes[1].text(8200,.13,'0.1% target',fontsize=7,color='#555555')
fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=4,
           bbox_to_anchor=(.5,-.055),frameon=False)
fig.tight_layout(w_pad=2)
save(fig,'size_ladder')

# Same-QP optimization error versus average precision across full-data means.
fig,axes=plt.subplots(1,2,figsize=(6.8,2.5))
xx=np.arange(3)
for k,m in enumerate(METHODS):
    avg=full[full.method.eq(m)].groupby('cost_ratio')
    x=xx+(k-1.5)*.13
    axes[0].plot(x,avg.relative_gap.mean().to_numpy()*100,marker=MARKER[m],
                 color=COLOR[m],lw=1.2,ms=4,label=LABEL[m])
    axes[1].errorbar(x,avg.pr_auc.mean().to_numpy(),yerr=avg.pr_auc.std().to_numpy(),
                     marker=MARKER[m],color=COLOR[m],lw=1.1,ms=4,capsize=2)
for ax in axes:
    ax.set_xticks(xx,['10:1','20:1','50:1'])
    ax.set_xlabel('Cost ratio')
    ax.grid(axis='y',alpha=.2)
axes[0].set_yscale('log')
axes[0].set_ylabel('Mean final empirical gap (%)')
axes[0].set_title('(a) Training objective',fontweight='bold')
axes[0].axhline(.1,ls='--',lw=.8,color='#777777')
axes[1].set_ylabel('Test average precision')
axes[1].set_ylim(.67,.84)
axes[1].set_title('(b) Held-out ranking',fontweight='bold')
fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=4,
           bbox_to_anchor=(.5,-.055),frameon=False)
fig.tight_layout(w_pad=2)
save(fig,'objective_and_prediction')

summary=[]
for suite,df in [('correctness',correct),('main_capped',small),('main_fast',medium),('main_full',full)]:
    for (ratio,m),g in df.groupby(['cost_ratio','method']):
        summary.append({'suite':suite,'cost_ratio':int(ratio),'method':m,'n':len(g),
                        'n_train':int(g.n_train.iloc[0]),
                        'fit_median_seconds':float(g.fit_time_sec.median()),
                        'gap_median_percent':float(g.relative_gap.median()*100),
                        'reached_count':int(g.reached_tolerance.sum()),
                        'stop_count':int(g.converged.sum())})
(ROOT/'aggregate_values.json').write_text(json.dumps(summary,indent=2))
print('Created three aggregate figures, three table fragments, and aggregate_values.json.')
