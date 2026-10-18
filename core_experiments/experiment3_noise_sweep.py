"""Experiment 3: Gaussian noise sweep across encoding variants.

Protocol: models are trained once on clean data (shared with Experiment 1) and
tested on validation clouds with physical Gaussian noise. New sigma values can be
evaluated from the saved clean checkpoints with --eval_only --noise_levels ...
"""

import os
import sys
import argparse
import numpy as np
import torch
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.msdgcnn2 import create_model
from utils.experiment_utils import (
    get_cv_dataloaders, train_model, set_seed, save_config,
    generate_cv_splits, TreeSpeciesDatasetCV, evaluate
)

import matplotlib.pyplot as plt
import pandas as pd
from utils.experiment_utils import run_shared_fold, aggregate_metrics, add_common_arguments, prepare_args, get_device, stable_seed
from utils.experiment_utils import evaluate_corrupted, parse_levels
from models.baselines import BASELINES, BASELINE_COLORS
import logging
import torch.nn as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


VARIANTS = {
    'raw_only': {
        'use_normalized_features': False,
        'local_use_normalized': False,
        'branch_feature_mode': 'full',
        'label': 'MS-DGCNN++ (RO)'
    },
    'hybrid_everywhere': {
        'use_normalized_features': True,
        'local_use_normalized': True,
        'branch_feature_mode': 'full',
        'label': 'MS-DGCNN++ (HE)'
    },
    'default': {
        'use_normalized_features': True,
        'local_use_normalized': False,
        'branch_feature_mode': 'full',
        'label': 'MS-DGCNN++ (DA)'
    },
    'reversed': {
        'use_normalized_features': False,
        'local_use_normalized': True,
        'branch_feature_mode': 'full',
        'label': 'MS-DGCNN++ (RA)'
    }
}

NOISE_LEVELS = [0, 0.5, 1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 300, 500, 1000]  # mm; 0 = clean


def all_models():
    """The four MS-DGCNN++ encodings plus the shared baselines (as in the outlier study)."""
    models = {}
    for variant, config in VARIANTS.items():
        models[variant] = {'label': config['label'], 'factory':
            lambda nc, config=config: create_model(num_classes=nc, k_scales=[5,20,30],
                use_multiscale=True, use_normalized_features=config['use_normalized_features'],
                local_use_normalized=config['local_use_normalized'], fusion_type='concat_conv',
                emb_dims=1024, dropout=0.5)}
    models.update(BASELINES)
    return models


def run_experiment(args):
    """Figure 5: clean training, noisy test at each sigma (physical units)."""
    if args.coordinate_unit is None:
        raise ValueError('Confirm XYZ units, then pass --coordinate_unit m, cm, or mm.')
    set_seed(args.seed)
    levels = parse_levels(args.noise_levels, '--noise_levels')
    output_dir = Path(args.resume_dir) if args.resume_dir else Path(args.output_dir) / f'experiment3_{datetime.now():%Y%m%d_%H%M%S}'
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(dict(vars(args), protocol_name='clean_train_corrupted_test'), output_dir / 'config.json')
    all_results = {v: {sigma: {} for sigma in levels} for v in all_models()}
    for variant, spec in all_models().items():
        factory = spec['factory']
        for fold in range(args.k_folds):
            logger.info('Noise (clean training): %s, fold %s', variant, fold)
            clean, model = run_shared_fold(args, fold, factory,
                checkpoint_path=output_dir / 'checkpoints' / f'{variant}_clean_fold{fold}.pth')
            for sigma in levels:
                all_results[variant][sigma][fold] = evaluate_corrupted(
                    args, fold, model, clean, noise_sigma=sigma, output_dir=output_dir)
            del model
            save_results_json(all_results, output_dir / 'all_results.json')
    generate_visualizations(all_results, output_dir, levels)
    from utils.plot_style import save_paper_transformations
    save_paper_transformations(args, output_dir / 'figures')


def save_results_json(results, path):
    """Save results to JSON."""
    serializable = {}
    for variant, sigmas in results.items():
        serializable[variant] = {}
        for sigma, folds in sigmas.items():
            serializable[variant][str(sigma)] = {}
            for fold, metrics in folds.items():
                serializable[variant][str(sigma)][str(fold)] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in metrics.items()
                }
    
    with open(path, 'w') as f:
        json.dump(serializable, f, indent=2)


def generate_visualizations(results, output_dir, levels):
    """Only the OA-versus-noise plot present in the paper (Figure 5)."""
    from utils.plot_style import set_publication_style, get_variant_colors, save_figure
    set_publication_style()
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    from utils.robustness_plot import plot_robustness, fold_stats
    for variant in VARIANTS:
        for sigma in levels:
            if sorted(results[variant][sigma]) != list(range(5)):
                raise ValueError('Five completed folds required for the noise plot.')
    plot_robustness(fold_stats(results, levels), figures_dir / 'noise_robustness',
                    r'Gaussian noise $\sigma$ at test time (mm)', xticklabels=[f'{s:g}' for s in levels],
                    legend_loc='lower left', ymin=60)
    models = all_models()

    # Numbers behind Figure 5 (OA and mAcc), one row per noise level
    from utils.latex_tables import level_table
    labels = {m: models[m]['label'] for m in models}
    for metric, name in [('oa', 'OA'), ('macc', 'mAcc')]:
        level_table(results, labels, levels, r'$\sigma$ (mm)', lambda s: f'{s:g}',
                    f'Gaussian coordinate noise: models trained on clean data, tested with noise $\\sigma$. '
                    f'{name} (\\%), mean $\\pm$ standard deviation across five folds (ddof=0). Bold: best mean per row.',
                    f'tab:noise_{metric}', output_dir / 'tables' / f'noise_{metric}.tex', metric=metric)


def main():
    parser = argparse.ArgumentParser(description='Experiment 3: Synthetic Noise Sweep (Resume/Extend)')
    parser.add_argument('--data_path', type=str, default='data/STPCTLC')
    parser.add_argument('--output_dir', type=str, default='results/paper_v2')
    parser.add_argument('--resume_dir', type=str, default=None,
                        help='Path to existing experiment directory to resume/extend')
    parser.add_argument('--k_folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--num_points', type=int, default=1024)
    parser.add_argument('--patience', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--noise_levels', type=float, nargs='+', default=NOISE_LEVELS,
                        help='Test-time Gaussian sigma levels in mm (clean-trained models)')
    
    add_common_arguments(parser)
    args = prepare_args(parser.parse_args())
    
    if not os.path.isabs(args.data_path):
        args.data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.data_path)
    
    run_experiment(args)


if __name__ == '__main__':
    main()
