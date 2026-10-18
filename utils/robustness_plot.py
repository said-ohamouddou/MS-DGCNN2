"""One plotting style for the three robustness figures (canopy retention, noise, outliers):
same colors, markers, labels and in-figure legend for every model."""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from utils.plot_style import set_publication_style, save_figure, VARIANT_COLORS
from models.baselines import BASELINES, BASELINE_COLORS

# Drawn at the printed size (about 0.85 of a 390 pt text width) so fonts print at 7-10 pt
FIGSIZE = (4.6, 2.9)
COMPACT_STYLE = {'font.size': 8.5, 'axes.labelsize': 9.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                 'legend.fontsize': 6.5, 'axes.linewidth': 0.8, 'grid.linewidth': 0.5,
                 'xtick.major.size': 3, 'ytick.major.size': 3, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7}

# key -> (label, color, marker, dashed)
MODEL_STYLE = {
    'raw_only': ('MS-DGCNN++ (RO)', VARIANT_COLORS[0], 'o', False),
    'hybrid_everywhere': ('MS-DGCNN++ (HE)', VARIANT_COLORS[1], 'o', False),
    'default': ('MS-DGCNN++ (DA)', VARIANT_COLORS[2], 'o', False),
    'reversed': ('MS-DGCNN++ (RA)', VARIANT_COLORS[3], 'o', False),
    'dgcnn': (BASELINES['dgcnn']['label'], BASELINE_COLORS['dgcnn'], 's', True),
    'ms_dgcnn': (BASELINES['ms_dgcnn']['label'], BASELINE_COLORS['ms_dgcnn'], '^', True),
    'pointm2ae': (BASELINES['pointm2ae']['label'], BASELINE_COLORS['pointm2ae'], 'D', True),
}


def plot_robustness(stats, path, xlabel, x=None, xticklabels=None, log_x=False, linear_ticks=None,
                    right=None, legend_loc='lower right', ymin=None, legend_band=0.45):
    """OA mean +- SD over folds against a corruption level.

    stats: {model_key: (levels_in_order, means, sds)} with keys from MODEL_STYLE.
    x: numeric positions (None -> evenly spaced categories labelled by xticklabels).
    right: optional dict(values=..., label=..., ylabel=...) drawn on a right axis.
    The legend (2 columns x 4 rows) sits inside the axes, in a band kept free of data.
    """
    set_publication_style()
    plt.rcParams.update(COMPACT_STYLE)
    keys = [k for k in MODEL_STYLE if k in stats]
    n = len(next(iter(stats.values()))[1])
    pos = np.arange(n) if x is None else np.asarray(x, dtype=float)
    fig, ax1 = plt.subplots(figsize=FIGSIZE)
    lows, highs = [], []
    for key in keys:
        label, color, marker, dashed = MODEL_STYLE[key]
        _, mean, sd = stats[key]
        mean, sd = np.asarray(mean), np.asarray(sd)
        lows.append((mean - sd).min()); highs.append((mean + sd).max())
        ax1.plot(pos, mean, linestyle='--' if dashed else '-', marker=marker, color=color,
                 linewidth=1.0 if dashed else 1.4, markersize=3.5 if dashed else 4.2, label=label, zorder=3)
        ax1.fill_between(pos, mean - sd, mean + sd, color=color, alpha=0.10, linewidth=0, zorder=1)
    if x is None:
        ax1.set_xticks(pos)
        ax1.set_xticklabels(xticklabels)
        ax1.set_xlim(pos[0] - 0.4, pos[-1] + 0.4)
    else:
        if log_x:
            ax1.set_xscale('log')
        ticks = [t for t in (linear_ticks or []) if pos.min() <= t <= pos.max()] or list(pos)
        ax1.set_xticks(ticks)
        ax1.set_xticklabels([f'{t:g}' for t in ticks])
        ax1.minorticks_off()
        if not log_x:
            span = pos.max() - pos.min()
            ax1.set_xlim(pos.min() - 0.02 * span, pos.max() + 0.02 * span)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('OA (%)')
    ax1.spines['top'].set_visible(False)
    # Empty band at the bottom for the legend
    lo, hi = min(lows), max(highs)
    if ymin is None:
        # Blank band below the data for the legend; ticks are shown only from 0 upwards
        from matplotlib.ticker import MaxNLocator
        bottom, top = lo - legend_band * (hi - lo), hi + 0.03 * (hi - lo)
        ax1.set_ylim(bottom, top)
        ax1.set_yticks([t for t in MaxNLocator(nbins=6).tick_values(max(bottom, 0.0), top) if max(bottom, 0.0) <= t <= top])
    else:
        ax1.set_ylim(ymin, hi + 0.03 * (hi - ymin))  # fixed lower limit: lower values leave the plot
    handles, labels = ax1.get_legend_handles_labels()
    legend_ax = ax1
    if right is not None:
        ax2 = ax1.twinx()
        vals = np.asarray(right['values'])
        ax2.plot(pos, vals, linestyle=':', marker='x', color='black', linewidth=1.0, markersize=4,
                 label=right['label'], zorder=2)
        ax2.set_ylabel(right['ylabel'])
        ax2.grid(False)
        ax2.spines['top'].set_visible(False)
        v_lo, v_hi = vals.min(), vals.max()
        ax2.set_ylim(v_lo - 0.62 * (v_hi - v_lo), v_hi + 0.05 * (v_hi - v_lo))
        h2, l2 = ax2.get_legend_handles_labels()
        handles, labels, legend_ax = handles + h2, labels + l2, ax2
    leg = legend_ax.legend(handles, labels, loc=legend_loc, ncol=2, fontsize=6.5, frameon=True,
                           framealpha=0.92, edgecolor='0.8', columnspacing=0.8, handlelength=2.0,
                           borderpad=0.4, handletextpad=0.4, labelspacing=0.3)
    leg.set_zorder(10)
    fig.tight_layout()
    save_figure(fig, path)
    plt.close(fig)


def fold_stats(results, levels, key_map=None):
    """{name: {level: {fold: metrics}} or {level: [metrics]}} -> stats for plot_robustness."""
    stats = {}
    for name, by_level in results.items():
        key = key_map.get(name, name) if key_map else name
        if key not in MODEL_STYLE:
            continue
        values = []
        for level in levels:
            folds = by_level[level]
            folds = folds.values() if isinstance(folds, dict) else folds
            values.append([f['oa'] for f in folds])
        values = np.array(values)
        stats[key] = (levels, values.mean(1), values.std(1))
    return stats
