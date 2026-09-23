from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
from matplotlib.lines import Line2D
import numpy as np

OUT = Path('outputs')
OUT.mkdir(parents=True, exist_ok=True)
# Sized for a two-column paper's full text width (figure*), so fonts print at
# their nominal size. Device, input size and run counts belong in the caption.
FIG_W, FIG_H = 7.16, 2.45
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'Nimbus Sans', 'DejaVu Sans'],
    'font.size': 7.5, 'axes.labelsize': 7.5, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#555555', 'axes.linewidth': .6,
    'xtick.major.width': .6, 'ytick.major.width': .6,
    'xtick.minor.width': .4, 'ytick.minor.width': .4,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.color': '#333333', 'ytick.color': '#333333',
    'text.color': '#111111', 'axes.labelcolor': '#111111',
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
# One distinct hue per model, shared by both figures.
COLOR = {'LightStereo-M': '#1F77B4', 'BANet-2D': '#2CA02C',
         'LightGEV (Ours)': '#D62728', 'MobileStereoNet-2D': '#9467BD',
         'GwcNet': '#FF7F0E', 'ACVNet': '#17BECF', 'IGEV++': '#E377C2'}
CATEGORY = '#7F7F7F'  # neutral marker colour for the shape legend
GRID = dict(color='#E6E6E6', linewidth=.5, zorder=0)


def save(fig, name):
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(OUT / f'{name}.{ext}', dpi=600,
                    bbox_inches='tight', pad_inches=.02, facecolor='white')
    plt.close(fig)
    print(f'Saved {name}.{{pdf,png,svg}} in', OUT)


# ---------------------------------------------------------------------------
# Desktop: RTX PRO 6000 Blackwell Max-Q, PyTorch FP32
# All numbers are transcribed from the user-provided benchmark tables.
# (name, FLOPs [G], runtime [ms], peak memory [GB], marker, color,
#  label offset [pt], ha)
# ---------------------------------------------------------------------------
desktop = [
    ('LightStereo-M',      30.525,   7.532,  .151, 'o', COLOR['LightStereo-M'],  (0, -9),  'center'),
    ('BANet-2D',           33.148,   8.793,  .183, 'o', COLOR['BANet-2D'],  (3, 7),   'right'),
    ('LightGEV (Ours)',    52.941,   9.889,  .352, '*', COLOR['LightGEV (Ours)'],  (6, -1),  'left'),
    ('MobileStereoNet-2D', 130.323, 32.364, 1.259, 'o', COLOR['MobileStereoNet-2D'],   (0, 6),   'center'),
    ('GwcNet',             697.742, 56.596, 2.738, '^', COLOR['GwcNet'], (-5, -1), 'right'),
    ('ACVNet',             846.737, 72.581, 3.458, '^', COLOR['ACVNet'], (-5, 2),  'right'),
    ('IGEV++',            3145.177, 164.659, 1.397, '^', COLOR['IGEV++'], (-5, 1),  'right')]

fig, axs = plt.subplots(1, 2, figsize=(FIG_W, FIG_H),
                        gridspec_kw={'width_ratios': [1.1, 1]})
fig.subplots_adjust(left=.065, right=.99, bottom=.17, top=.98, wspace=.42)

ax = axs[0]
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(12, 5000); ax.set_ylim(5, 260)
for name, x, y, _, m, c, offset, ha in desktop:
    ours = m == '*'
    ax.scatter(x, y, s=90 if ours else 22, marker=m, c=c,
               edgecolors='white', linewidths=.5, zorder=4 if ours else 3)
    ax.annotate(name, (x, y), xytext=offset, textcoords='offset points',
                fontsize=6.5, ha=ha, va='center',
                fontweight='bold' if ours else 'normal',
                color=c if ours else '#333333')
ax.xaxis.set_major_locator(FixedLocator([30, 100, 300, 1000, 3000]))
ax.yaxis.set_major_locator(FixedLocator([10, 30, 100]))
for axis in [ax.xaxis, ax.yaxis]:
    axis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:g}'))
    axis.set_minor_formatter(NullFormatter())
ax.set_xlabel('FLOPs (G)')
ax.set_ylabel('Runtime (ms)')
ax.grid(which='major', **GRID)
ax.legend(handles=[Line2D([], [], marker='o', linestyle='', markersize=4,
                          color=CATEGORY, label='Lightweight'),
                   Line2D([], [], marker='^', linestyle='', markersize=4,
                          color=CATEGORY, label='Volumetric')],
          loc='upper left', frameon=False, handletextpad=.2,
          borderaxespad=.1, labelspacing=.3)

# Horizontal bars in table order (lightweight on top, volumetric below).
ax = axs[1]
ypos = np.arange(len(desktop))[::-1]
for yp, (name, _, _, mem, m, c, _, _) in zip(ypos, desktop):
    ours = m == '*'
    ax.barh(yp, mem, .68, color=c, linewidth=0, zorder=3)
    ax.text(mem + .05, yp, f'{mem:.3f}', va='center', fontsize=6.5,
            fontweight='bold' if ours else 'normal',
            color=c if ours else '#333333')
ax.set_yticks(ypos, [d[0] for d in desktop])
for label, d in zip(ax.get_yticklabels(), desktop):
    if d[4] == '*':
        label.set_fontweight('bold'); label.set_color(d[5])
ax.axhline(ypos[3] - .5, color='#BBBBBB', linewidth=.5, linestyle=(0, (3, 2)))
ax.set_xlim(0, 3.9); ax.set_xticks([0, 1, 2, 3])
ax.set_ylim(-.6, len(desktop) - .4)
ax.set_xlabel('Peak inference memory (GB)')
ax.grid(axis='x', **GRID)
ax.tick_params(axis='y', length=0, pad=3)
save(fig, 'lightgev_efficiency_desktop')

# ---------------------------------------------------------------------------
# Jetson Orin Nano Super Developer Kit, TensorRT engines
# ---------------------------------------------------------------------------
runtime = np.array([[106.623, 47.651, 36.600], [92.112, 48.877, 34.385], [161.663, 85.171, 69.626]])
memory = np.array([[233.0, 84.3, 79.9], [324.7, 72.5, 53.0], [342.2, 150.1, 144.1]])
models = ['LightStereo-M', 'BANet-2D', 'LightGEV (Ours)']

fig, axs = plt.subplots(1, 2, figsize=(FIG_W, FIG_H))
fig.subplots_adjust(left=.065, right=.99, bottom=.17, top=.88, wspace=.28)
width = .27
for ax, data, ylabel, limit, ticks in [
        (axs[0], runtime, 'Runtime (ms)', 185, [0, 50, 100, 150]),
        (axs[1], memory, 'Activation memory (MB)', 390, [0, 100, 200, 300])]:
    for i, model in enumerate(models):
        bars = ax.bar(np.arange(3) + (i - 1) * width, data[i], width * .92,
                      label=model, color=COLOR[model], linewidth=0, zorder=3)
        ax.bar_label(bars, labels=[f'{v:.1f}' for v in data[i]], padding=1.5,
                     fontsize=6, color='#333333')
    ax.set_xticks(np.arange(3), ['FP32', 'FP16', 'INT8'])
    ax.set_xlim(-.55, 2.55)
    ax.set_ylim(0, limit); ax.set_yticks(ticks)
    ax.set_ylabel(ylabel); ax.set_xlabel('TensorRT precision')
    ax.grid(axis='y', **GRID)
    ax.tick_params(axis='x', length=0, pad=3)

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, 1.0), ncol=3,
           frameon=False, fontsize=7.5, handlelength=1.2, handleheight=.8,
           columnspacing=1.6, handletextpad=.4)
save(fig, 'lightgev_efficiency_jetson')
