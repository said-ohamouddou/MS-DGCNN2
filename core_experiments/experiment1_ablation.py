"""Experiment 1: per-scale edge encoding ablations."""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.msdgcnn2 import create_model
from utils.experiment_utils import (
    get_cv_dataloaders, train_model, set_seed, save_config,
    ExperimentResults,
    generate_cv_splits,
    evaluate
)
import torch.nn as nn
import json

from utils.experiment_utils import run_shared_fold, aggregate_metrics, add_common_arguments, prepare_args, get_device, stable_seed
from utils.experiment_utils import evaluate_corrupted
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


VARIANTS = {
    'raw_only': {
        'use_normalized_features': False,
        'local_use_normalized': False,
        'label': 'MS-DGCNN++ (RO)'
    },
    'hybrid_everywhere': {
        'use_normalized_features': True,
        'local_use_normalized': True,
        'label': 'MS-DGCNN++ (HE)'
    },
    'default': {
        'use_normalized_features': True,
        'local_use_normalized': False,
        'label': 'MS-DGCNN++ (DA)'
    },
    'reversed': {
        'use_normalized_features': False,
        'local_use_normalized': True,
        'label': 'MS-DGCNN++ (RA)'
    }
}


def run_experiment(args):
    """Table 6; identical clean configurations are reused by density/outlier studies."""
    set_seed(args.seed)
    output_dir = Path(args.resume_dir) if args.resume_dir else Path(args.output_dir) / f'experiment1_{datetime.now():%Y%m%d_%H%M%S}'
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(vars(args), output_dir / 'config.json')
    _, _, _, class_names = get_cv_dataloaders(args.data_path, fold=0, num_points=args.num_points, seed=args.seed)
    results = ExperimentResults('experiment1', output_dir)
    for variant, config in VARIANTS.items():
        for fold in range(args.k_folds):
            logger.info('Encoding: %s, fold %s', variant, fold)
            factory = lambda nc: create_model(num_classes=nc, k_scales=[5,20,30],
                use_multiscale=True, use_normalized_features=config['use_normalized_features'],
                local_use_normalized=config['local_use_normalized'], fusion_type='concat_conv', emb_dims=1024, dropout=0.5)
            clean, model = run_shared_fold(args, fold, factory,
                checkpoint_path=output_dir / 'checkpoints' / f'{variant}_fold{fold}.pth')
            # Same saved clean evaluation as Figure 4 / Table 7 at 100% retention
            metrics = evaluate_corrupted(args, fold, model, clean, retention=1.0, output_dir=output_dir)
            results.add_fold_result(variant, fold, metrics)
            del model
    results.save()
    generate_visualizations(results, class_names, output_dir)


def generate_visualizations(results, class_names, output_dir):
    """Export the encoding table used in the paper; bold = best displayed value per column."""
    import pandas as pd
    tables_dir = output_dir / 'tables'
    tables_dir.mkdir(exist_ok=True)
    agg = {v: results.get_aggregated_metrics(v) for v in VARIANTS}
    rows = []
    for variant in VARIANTS:
        a = agg[variant]
        rows.append([VARIANTS[variant]['label'], f"{a['oa_mean']:.2f}$\\pm${a['oa_std']:.2f}",
                     f"{a['macc_mean']:.2f}$\\pm${a['macc_std']:.2f}"] + [f'{100 * x:.1f}' for x in a['f1_mean']])
    df = pd.DataFrame(rows, columns=['Model', r'OA (\%)', r'mAcc (\%)', *class_names])
    df.to_csv(tables_dir / 'full_results.csv', index=False)

    # Best displayed value per column (ties all bold)
    best_oa = max(round(agg[v]['oa_mean'], 2) for v in VARIANTS)
    best_macc = max(round(agg[v]['macc_mean'], 2) for v in VARIANTS)
    best_f1 = [max(round(100 * agg[v]['f1_mean'][i], 1) for v in VARIANTS) for i in range(len(class_names))]
    bold = lambda text, flag: r'\best{' + text + '}' if flag else text  # \best = bold (defined in the paper)
    lines = [r'\begin{table}[htbp]', r'\centering',
             r'\caption{Encoding ablation. Models trained on clean data and tested on clean data. OA and mAcc: '
             r'mean $\pm$ standard deviation across five folds (ddof=0). Per-class F1: fold means. '
             r'Bold: best value per column.}',
             r'\label{tab:exp1_results}', r'\resizebox{\textwidth}{!}{%',
             r'\begin{tabular}{l' + 'c' * (2 + len(class_names)) + '}', r'\toprule',
             rf'& & & \multicolumn{{{len(class_names)}}}{{c}}{{Per-class F1 (\%)}} \\',
             rf'\cmidrule(lr){{4-{3 + len(class_names)}}}',
             r'Model & OA (\%) & mAcc (\%) & ' + ' & '.join(class_names) + r' \\', r'\midrule']
    for variant in VARIANTS:
        a = agg[variant]
        cells = [VARIANTS[variant]['label'],
                 bold(f"{a['oa_mean']:.2f}$\\pm${a['oa_std']:.2f}", round(a['oa_mean'], 2) == best_oa),
                 bold(f"{a['macc_mean']:.2f}$\\pm${a['macc_std']:.2f}", round(a['macc_mean'], 2) == best_macc)]
        cells += [bold(f'{100 * x:.1f}', round(100 * x, 1) == best_f1[i]) for i, x in enumerate(a['f1_mean'])]
        lines.append(' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}}', r'\end{table}']
    (tables_dir / 'full_results.tex').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Experiment 1: Per-Scale Encoding Ablation')
    parser.add_argument('--data_path', type=str, 
                        default='data/STPCTLC',
                        help='Path to STPCTLC data directory')
    parser.add_argument('--output_dir', type=str, default='results/paper_v2',
                        help='Output directory for results')
    parser.add_argument('--resume_dir', type=str, default=None,
                        help='Path to existing experiment directory to resume/regenerate')
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
    
    add_common_arguments(parser)
    args = prepare_args(parser.parse_args())
    
    # Convert relative path to absolute
    if not os.path.isabs(args.data_path):
        args.data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.data_path)
    
    run_experiment(args)


if __name__ == '__main__':
    main()
