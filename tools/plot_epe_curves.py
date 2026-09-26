"""Side-by-side SceneFlow train/val EPE curves: LightStereo-M (left), LightGEV (right).

Curves are re-drawn from the training logs (same parser as plot_val_metrics.py)
in the paper style of plot_efficiency.py. Run from the repo root.
"""
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_val_metrics import parse_log

OUT = Path('outputs')
OUT.mkdir(parents=True, exist_ok=True)
RUNS = Path('train_outputs/SceneFlowDataset')
MODELS = [
    ('LightStereo-M', RUNS / 'LightStereo/lightstereo_m_train_sceneflow/'
                             'pretrain_trainBS12_valBS12_LR0.0004_datafixed'),
    ('LightGEV (Ours)', RUNS / 'LightGEVModify_4/lightgev_modify_4_train_sceneflow_full_iter_6_r_075/'
                               'pretrain_trainBS12_valBS12_LR0.0004_modify_4_gate_normT_biasF'),
]

FIG_W, FIG_H = 7.16, 2.45
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'Nimbus Sans', 'DejaVu Sans'],
    'font.size': 7.5, 'axes.labelsize': 7.5, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#555555', 'axes.linewidth': .6,
    'xtick.major.width': .6, 'ytick.major.width': .6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.color': '#333333', 'ytick.color': '#333333',
    'text.color': '#111111', 'axes.labelcolor': '#111111'})
SERIES = [('train_epe', 'Train', '#EA580C'), ('epe', 'Validation', '#2563EB')]
GRID = dict(color='#E6E6E6', linewidth=.5, zorder=0)

# Shared y-axis so the two models are directly comparable.
fig, axs = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), sharey=True)
fig.subplots_adjust(left=.065, right=.99, bottom=.17, top=.82, wspace=.12)
for ax, (model, run_dir) in zip(axs, MODELS):
    log = sorted(run_dir.glob('train_*.log'))[-1]
    records = parse_log(log)
    for key, label, color in SERIES:
        epochs, values = zip(*records[key])
        ax.plot(epochs, values, color=color, linewidth=1.1, marker='o',
                markersize=1.8, label=label, zorder=3)
        ax.annotate(f'{values[-1]:.3f}', (epochs[-1], values[-1]),
                    xytext=(0, 5 if key == 'train_epe' else -6),
                    textcoords='offset points', ha='right', va='center',
                    fontsize=6.5, color=color)
    ax.set_title(model, loc='center')
    ax.set_xlabel('Epoch')
    ax.set_xticks(list(range(0, max(epochs) + 1, 10)))
    ax.set_xlim(-2, max(epochs) + 2)
    ax.grid(**GRID)
axs[0].set_ylabel('EPE (px)')
axs[0].set_ylim(.3, 3.1)

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, 1.0), ncol=2,
           frameon=False, fontsize=7.5, handlelength=1.6, columnspacing=1.6)
fig.savefig(OUT / 'epe_curves_lightstereo_vs_lightgev.png', dpi=600,
            bbox_inches='tight', pad_inches=.02, facecolor='white')
print('Saved', OUT / 'epe_curves_lightstereo_vs_lightgev.png')
