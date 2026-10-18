"""MS-DGCNN++ k-scale sensitivity (5-fold CV).

Protocol: every k configuration is trained once on clean data and tested at the
upper-canopy retention rates given by --retention_rates (default 100% and 5%, as in
Table 7). Others can be evaluated from the saved checkpoints with --eval_only.
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.msdgcnn2 import create_model
from utils.experiment_utils import (
    get_cv_dataloaders, train_model, set_seed, save_config,
    generate_cv_splits, TreeSpeciesDatasetCV
)

from utils.experiment_utils import run_shared_fold, aggregate_metrics, add_common_arguments, prepare_args, get_device, stable_seed
from utils.experiment_utils import evaluate_corrupted, parse_levels
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


BASELINE_K_SCALES = [5, 20, 30]
K_LOCAL_VALUES = [3, 5, 10, 15, 20, 25, 30]
K_INTERMEDIATE_VALUES = [5, 10, 15, 20, 25, 30, 35]
K_GLOBAL_VALUES = [10, 20, 25, 30, 35, 40, 45]


def create_default_asymmetric(num_classes: int, k_scales: List[int]) -> nn.Module:
    """Create MS-DGCNN++ (Default Asymmetric) model."""
    return create_model(
        num_classes=num_classes,
        k_scales=k_scales,
        use_multiscale=True,
        use_normalized_features=True,
        local_use_normalized=False,
        fusion_type='concat_conv',
    )


def get_checkpoint_filename(scale_name: str, k_value: int, fold: int, retention: float = 1.0) -> str:
    """Generate checkpoint filename."""
    ret_pct = int(retention * 100)
    return f'default_asymmetric_{scale_name}_{k_value}_r{ret_pct}_fold{fold}.pth'


def plot_kscale_sensitivity(results_by_rate: List[Tuple[float, Dict]], output_dir: Path,
                            time_results: Dict = None, name: str = 'kscale_sensitivity'):
    """Figure 6 in the paper's style, one figure: one OA/F1 row per test retention
    (e.g. 100% and 5%), then a single epoch-time row (training time does not depend on retention)."""
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from matplotlib.lines import Line2D

    style = {
        'font.family': 'serif', 'font.serif': ['Nimbus Roman', 'Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'cm', 'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
        'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.linewidth': 0.6,
        'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'pdf.fonttype': 3,
    }
    panels = [('k_local', K_LOCAL_VALUES, BASELINE_K_SCALES[0], '#2E86AB', r'$k_1$ (local)'),
              ('k_intermediate', K_INTERMEDIATE_VALUES, BASELINE_K_SCALES[1], '#A23B72', r'$k_2$ (intermediate)'),
              ('k_global', K_GLOBAL_VALUES, BASELINE_K_SCALES[2], '#F18F01', r'$k_3$ (learned space)')]
    xticks = {'k_local': [5, 10, 20, 30], 'k_intermediate': [5, 10, 15, 20, 25, 30, 35],
              'k_global': [10, 20, 30, 40]}
    default_line = dict(color='0.35', linestyle=(0, (4, 2)), linewidth=0.9, zorder=2.5)

    def lighter(color, amount=0.45):
        return tuple(c + (1 - c) * amount for c in to_rgb(color))

    def finish(ax, scale, default, xlabel, ylabel):
        ax.axvline(default, **default_line)
        ax.set_xticks(xticks[scale])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, color='0.88', linewidth=0.5)

    with matplotlib.rc_context(style):
        n_rows = len(results_by_rate) + 1
        fig, axes = plt.subplots(n_rows, 3, figsize=(6.52, 1.75 * n_rows))

        # One OA/F1 row per test retention, shared y-range within the row (as in the paper)
        for row, (rate, all_results) in enumerate(results_by_rate):
            row_values = []
            for col, (scale, k_values, default, color, xlabel) in enumerate(panels):
                ks = [k for k in k_values if k in all_results.get(scale, {})]
                if not ks:
                    continue
                stat = lambda s, m: np.array([all_results[scale][k][s][m] for k in ks]) * (100 if m == 'f1' else 1)
                oa, oa_sd, f1, f1_sd = stat('mean', 'oa'), stat('std', 'oa'), stat('mean', 'f1'), stat('std', 'f1')
                row_values += [oa - oa_sd, oa + oa_sd, f1 - f1_sd, f1 + f1_sd]
                ax = axes[row, col]
                ax.errorbar(ks, oa, yerr=oa_sd, color=color, marker='o', markersize=3.5, linewidth=1.1,
                            elinewidth=0.6, capsize=2, capthick=0.6, zorder=3)
                eb = ax.errorbar(ks, f1, yerr=f1_sd, color=lighter(color), marker='s', markersize=3,
                                 linewidth=1.1, linestyle='--', elinewidth=0.6, capsize=2, capthick=0.6, zorder=2)
                eb[-1][0].set_linestyle('--')
                if default in ks:
                    ax.plot([default], [oa[ks.index(default)]], marker='*', markersize=6, color='black',
                            markerfacecolor='none', markeredgewidth=0.6, linestyle='none', zorder=4)
                finish(ax, scale, default, xlabel, f'Score (%), $r$ = {rate * 100:g}%')
            if row_values:
                lo, hi = np.min(np.concatenate(row_values)), np.max(np.concatenate(row_values))
                pad = 0.04 * (hi - lo)
                for ax in axes[row]:
                    ax.set_ylim(lo - pad, hi + pad)

        # Single epoch-time row
        for col, (scale, k_values, default, color, xlabel) in enumerate(panels):
            ax = axes[-1, col]
            ks = [k for k in k_values if time_results and k in time_results.get(scale, {})]
            if ks:
                t = [time_results[scale][k] for k in ks]
                ax.plot(ks, t, color=color, marker='s', markersize=3, linewidth=1.1, zorder=3)
                if default in ks:
                    ax.plot([default], [t[ks.index(default)]], marker='*', markersize=6, color='black',
                            linestyle='none', zorder=4)
            finish(ax, scale, default, xlabel, 'Epoch time (s)')

        c0 = panels[0][3]
        handles = [Line2D([], [], color=c0, marker='o', markersize=3.5, linewidth=1.1),
                   Line2D([], [], color=lighter(c0), marker='s', markersize=3, linewidth=1.1, linestyle='--')]
        fig.legend(handles, ['OA', 'F1'], loc='upper left', bbox_to_anchor=(0.11, 1.0), ncol=2, frameon=False,
                   handlelength=3.0, columnspacing=1.5)
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.1 / n_rows), h_pad=1.0, w_pad=1.2)
        out_path = output_dir / f'{name}.png'
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        fig.savefig(str(out_path).replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
        plt.close(fig)
    logger.info(f"Saved k-scale plot to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='K-Scale Ablation Study')
    parser.add_argument('--data_path', type=str, default='data/STPCTLC',
                        help='Path to STPCTLC data directory')
    parser.add_argument('--output_dir', type=str, default='results/paper_v2/kscale_ablation',
                        help='Output directory for results')
    parser.add_argument('--k_folds', type=int, default=5,
                        help='Number of CV folds')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='Number of points per sample')
    parser.add_argument('--patience', type=int, default=0,
                        help='Early stopping patience; 0 runs all epochs (paper default)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--eval_only', action='store_true',
                        help='Evaluation only mode (load existing checkpoints)')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume from a previous run directory (e.g., 20260314_144548). Skips models with existing checkpoints.')
    parser.add_argument('--retention_rates', type=float, nargs='+', default=[1.0, 0.05],
                        help='Test-time canopy retention rates for the clean-trained models '
                             '(default 100%% and 5%%, as in Table 7; Figure 6 uses 5%%)')
    
    add_common_arguments(parser)
    args = prepare_args(parser.parse_args())
    
    set_seed(args.seed)
    stamp = args.resume_from or datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(dict(vars(args), protocol_name='clean_train_corrupted_test'), output_dir / 'config.json')
    scale_experiments = [('k_local', K_LOCAL_VALUES, 0),
                         ('k_intermediate', K_INTERMEDIATE_VALUES, 1),
                         ('k_global', K_GLOBAL_VALUES, 2)]
    rates = parse_levels(args.retention_rates, '--retention_rates')
    keys = {r: f'r{int(round(r*100))}' for r in rates}
    all_results = {keys[r]: {} for r in rates}
    time_results = {}
    for name, values, pos in scale_experiments:
        for r in rates:
            all_results[keys[r]][name] = {}
        time_results[name] = {}
        for k in values:
            scales = BASELINE_K_SCALES.copy()
            scales[pos] = k
            folds = {r: [] for r in rates}
            for fold in range(args.k_folds):
                logger.info('Neighborhood (clean training): %s=%s, fold %s', name, k, fold)
                clean, model = run_shared_fold(args, fold,
                    lambda nc: create_default_asymmetric(nc, scales),
                    checkpoint_path=output_dir / 'checkpoints' / get_checkpoint_filename(name, k, fold, 1.0))
                # Retention is applied to the validation clouds only (same evaluations as Table 7)
                for r in rates:
                    folds[r].append(evaluate_corrupted(args, fold, model, clean, retention=r,
                                                       output_dir=output_dir))
                del model
            for r in rates:
                all_results[keys[r]][name][k] = aggregate_metrics(folds[r], args.k_folds)
            time_results[name][k] = all_results[keys[rates[0]]][name][k]['mean']['epoch_time']
    from utils.experiment_utils import save_results_json
    save_results_json(all_results, output_dir / 'results.json')
    save_results_json(time_results, output_dir / 'time_results.json')
    plot_kscale_sensitivity([(r, all_results[keys[r]]) for r in rates], output_dir, time_results)
    write_kscale_table(all_results, rates, keys, output_dir / 'kscale_results.tex')


def write_kscale_table(all_results, rates, keys, path):
    """LaTeX: one row per k value, OA and F1 at each test retention (mean $\\pm$ SD over folds)."""
    titles = {'k_local': '$k_1$ (local)', 'k_intermediate': '$k_2$ (intermediate)', 'k_global': '$k_3$ (learned space)'}
    defaults = dict(zip(['k_local', 'k_intermediate', 'k_global'], BASELINE_K_SCALES))
    n = len(rates)
    rate_labels = ' & '.join([rf'{r*100:g}\%' for r in rates] * 2)
    lines = [r'\begin{table}[htbp]', r'\centering',
             r'\caption{Neighborhood size sensitivity. Models trained on clean data and tested at upper-canopy '
             r'retention $r$; each scale is varied with the others at their defaults ($k_1=5$, $k_2=20$, $k_3=30$; '
             r'default marked *). Mean $\pm$ standard deviation across five folds (ddof=0).}',
             r'\label{tab:kscale_results}', r'\resizebox{\columnwidth}{!}{%',
             r'\begin{tabular}{l' + 'c' * (2 * n) + '}', r'\toprule',
             rf'$k$ & \multicolumn{{{n}}}{{c}}{{OA (\%)}} & \multicolumn{{{n}}}{{c}}{{F1 (\%)}} \\',
             r' & ' + rate_labels + r' \\']
    for scale in ['k_local', 'k_intermediate', 'k_global']:
        lines += [r'\midrule', rf'\multicolumn{{{1 + 2 * n}}}{{l}}{{\textbf{{{titles[scale]}}}}} \\']
        for k in all_results[keys[rates[0]]][scale]:
            cells = [f'{k}*' if k == defaults[scale] else f'{k}']
            for metric, scale_factor in [('oa', 1), ('f1', 100)]:
                for r in rates:
                    a = all_results[keys[r]][scale][k]
                    cells.append(f"{scale_factor*a['mean'][metric]:.1f}$\\pm${scale_factor*a['std'][metric]:.1f}")
            lines.append(' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}}', r'\end{table}']
    Path(path).write_text('\n'.join(lines) + '\n')

if __name__ == '__main__':
    main()
