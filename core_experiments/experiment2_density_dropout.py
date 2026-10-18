"""Experiment 2: upper-canopy density dropout across retention rates.

Protocol: models are trained once on clean data (shared with Experiment 1) and
tested on thinned validation clouds. A small matched-training supplement trains
selected variants on thinned data as well. New retention rates can be evaluated
from the saved clean checkpoints with --eval_only --retention_rates ...
"""

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
    generate_cv_splits, TreeSpeciesDatasetCV, evaluate
)

import matplotlib.pyplot as plt
import pandas as pd
from utils.experiment_utils import run_shared_fold, aggregate_metrics, add_common_arguments, prepare_args, get_device, stable_seed
from utils.experiment_utils import evaluate_corrupted, parse_levels
from models.baselines import BASELINES, BASELINE_COLORS
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

RETENTION_RATES = [1.0, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.02, 0.01]  # 1.0 = clean


def compute_k2_neighbor_distance(data_path, fold, retention_rate, k2=20,
                                  num_points=1024, seed=42, k_folds=5):
    """Original upper-canopy queries to all input points; k includes the center.

    Uses the same thinning and padding as training, in normalized coordinates.
    Average within each query, then cloud; the caller combines all validation folds.
    """
    from scipy.spatial.distance import cdist
    from utils.experiment_utils import normalize_pc
    dataset = TreeSpeciesDatasetCV(data_path, fold=fold, partition='val',
        num_points=num_points, augment=False, seed=seed, k_folds=k_folds,
        density_retention=retention_rate, density_seed=seed)
    means = []
    for idx in range(len(dataset)):
        original = dataset.data[idx]
        points = dataset.apply_density_dropout(original, stable_seed(seed, int(dataset.indices[idx]), 'density'))
        queries = points[:,2] > np.median(original[:,2])
        points = normalize_pc(points)
        if queries.any():
            distances = cdist(points[queries], points)
            means.append(float(np.partition(distances, k2-1, axis=1)[:,:k2].mean()))
    return means


def make_factory(config):
    return lambda nc: create_model(num_classes=nc, k_scales=[5,20,30],
        use_multiscale=True, use_normalized_features=config['use_normalized_features'],
        local_use_normalized=config['local_use_normalized'], fusion_type='concat_conv', emb_dims=1024, dropout=0.5)


def all_models():
    """The four MS-DGCNN++ encodings plus the shared baselines (as in the outlier study)."""
    models = {v: {'label': c['label'], 'factory': make_factory(c)} for v, c in VARIANTS.items()}
    models.update(BASELINES)
    return models


def run_experiment(args):
    """Figure 4: clean training, thinned test; plus matched (thinned) training for a subset."""
    set_seed(args.seed)
    rates = parse_levels(args.retention_rates, '--retention_rates')
    output_dir = Path(args.resume_dir) if args.resume_dir else Path(args.output_dir) / f'experiment2_{datetime.now():%Y%m%d_%H%M%S}'
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(dict(vars(args), protocol_name='clean_train_corrupted_test'), output_dir / 'config.json')
    _, _, _, class_names = get_cv_dataloaders(args.data_path, fold=0, num_points=args.num_points, seed=args.seed)

    # Main protocol: one clean model per model and fold, tested at every retention rate
    all_results = {v: {r: {} for r in rates} for v in all_models()}
    for variant, spec in all_models().items():
        for fold in range(args.k_folds):
            logger.info('Density (clean training): %s, fold %s', variant, fold)
            clean, model = run_shared_fold(args, fold, spec['factory'],
                checkpoint_path=output_dir / 'checkpoints' / f'{variant}_clean_fold{fold}.pth')
            for rate in rates:
                all_results[variant][rate][fold] = evaluate_corrupted(
                    args, fold, model, clean, retention=rate, output_dir=output_dir)
            del model
            save_results_json(all_results, output_dir / 'all_results.json')

    # Supplement: matched training (trained and tested on thinned data)
    matched = {v: {r: {} for r in args.matched_retentions} for v in args.matched_variants}
    for variant in args.matched_variants:
        for rate in args.matched_retentions:
            for fold in range(args.k_folds):
                logger.info('Density (matched training): %s, retention %s, fold %s', variant, rate, fold)
                metrics, model = run_shared_fold(args, fold, make_factory(VARIANTS[variant]), retention=rate,
                    checkpoint_path=output_dir / 'checkpoints' / f'{variant}_matched_r{int(rate*100)}_fold{fold}.pth')
                matched[variant][rate][fold] = metrics
                del model
    if matched:
        save_results_json(matched, output_dir / 'matched_results.json')

    k2_distances = {}
    for rate in rates:
        means = []
        for fold in range(args.k_folds):
            means.extend(compute_k2_neighbor_distance(args.data_path, fold, rate,
                num_points=args.num_points, seed=args.seed, k_folds=args.k_folds))
        k2_distances[rate] = {'mean': float(np.mean(means)), 'std': float(np.std(means)), 'clouds': len(means)}
    from utils.experiment_utils import save_results_json as save_json
    save_json(k2_distances, output_dir / 'k2_distances.json')
    generate_visualizations(all_results, k2_distances, class_names, output_dir, rates, matched,
                            min_retention=args.figure_min_retention)


def save_results_json(results, path):
    """Save results to JSON with numpy array conversion."""
    import json
    
    serializable = {}
    for variant, rates in results.items():
        serializable[variant] = {}
        for rate, folds in rates.items():
            serializable[variant][str(rate)] = {}
            for fold, metrics in folds.items():
                serializable[variant][str(rate)][fold] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in metrics.items()
                }
    
    with open(path, 'w') as f:
        json.dump(serializable, f, indent=2)


def plot_degradation(results, k2_distances, rates, models, colors, path, log_x=False):
    """OA (mean +- SD over folds) vs upper-canopy retention, with the mean k2-neighbor distance."""
    from utils.robustness_plot import plot_robustness, fold_stats
    plot_robustness(fold_stats(results, rates), path, 'Upper-canopy retention (%)',
                    x=[r * 100 for r in rates], log_x=log_x,
                    linear_ticks=(1, 2, 5, 10, 20, 50, 100) if log_x else (5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
                    right=dict(values=[k2_distances[r]['mean'] for r in rates],
                               label=r'Mean $k_2$ distance (right axis)',
                               ylabel=r'Mean $k_2$-neighbor distance'),
                    legend_loc='lower right')


def generate_visualizations(results, k2_distances, class_names, output_dir, rates, matched=None,
                            min_retention=0.05):
    """Generate visualizations for Experiment 2."""
    
    from utils.plot_style import set_publication_style, get_variant_colors, add_panel_label, save_figure
    set_publication_style()
    
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    models = all_models()
    variant_names = list(models.keys())
    colors = get_variant_colors(len(VARIANTS)) + [BASELINE_COLORS[b] for b in BASELINES]
    
    # 1. Degradation curves: main figure (>= min_retention, linear axis) + all levels (log axis)
    shown = [r for r in rates if r >= min_retention]
    plot_degradation(results, k2_distances, shown, models, colors, figures_dir / 'degradation_curves', log_x=False)
    if len(shown) < len(rates):
        plot_degradation(results, k2_distances, rates, models, colors,
                         figures_dir / 'degradation_curves_all_levels', log_x=True)

    # Numbers behind Figure 4 (OA and mAcc), one row per retention rate
    from utils.latex_tables import level_table
    labels = {m: models[m]['label'] for m in models}
    for metric, name in [('oa', 'OA'), ('macc', 'mAcc')]:
        level_table(results, labels, rates, 'Retention', lambda r: f'{r*100:g}\\%',
                    f'Upper-canopy density dropout: models trained on clean data, tested at retention $r$. '
                    f'{name} (\\%), mean $\\pm$ standard deviation across five folds (ddof=0). Bold: best mean per row.',
                    f'tab:density_{metric}', output_dir / 'tables' / f'density_{metric}.tex', metric=metric)

    # Supplement: clean-trained vs matched-trained OA at the matched retention rates
    if matched:
        rows = []
        for variant, by_rate in matched.items():
            for rate, folds in by_rate.items():
                clean_oa = [results[variant][rate][f]['oa'] for f in sorted(results[variant][rate])] \
                    if rate in results[variant] else []
                matched_oa = [folds[f]['oa'] for f in sorted(folds)]
                rows.append({'variant': VARIANTS[variant]['label'], 'retention': rate,
                             'clean_train_oa_mean': float(np.mean(clean_oa)) if clean_oa else None,
                             'clean_train_oa_std': float(np.std(clean_oa)) if clean_oa else None,
                             'matched_train_oa_mean': float(np.mean(matched_oa)),
                             'matched_train_oa_std': float(np.std(matched_oa))})
        tables_dir = output_dir / 'tables'
        tables_dir.mkdir(exist_ok=True)
        pd.DataFrame(rows).to_csv(tables_dir / 'matched_vs_clean_training.csv', index=False)
        from utils.latex_tables import matched_table
        matched_table(rows, tables_dir / 'matched_vs_clean_training.tex')
        logger.info('Matched-training comparison saved to %s', tables_dir / 'matched_vs_clean_training.csv')
    

def main():
    parser = argparse.ArgumentParser(description='Experiment 2: Controlled Density Dropout')
    parser.add_argument('--data_path', type=str, default='data/STPCTLC')
    parser.add_argument('--output_dir', type=str, default='results/paper_v2')
    parser.add_argument('--resume_dir', type=str, default=None,
                        help='Path to existing experiment directory to resume/regenerate')
    parser.add_argument('--k_folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--num_points', type=int, default=1024)
    parser.add_argument('--patience', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--retention_rates', type=float, nargs='+', default=RETENTION_RATES,
                        help='Test-time upper-canopy retention rates (clean-trained models)')
    parser.add_argument('--figure_min_retention', type=float, default=0.05,
                        help='Lowest retention drawn in the main figure (all levels stay in the tables '
                             'and in degradation_curves_all_levels)')
    parser.add_argument('--matched_retentions', type=float, nargs='*', default=[0.05],
                        help='Retention rates for the matched-training supplement (empty to skip)')
    parser.add_argument('--matched_variants', nargs='*', default=['hybrid_everywhere', 'default'],
                        choices=list(VARIANTS), help='Variants for the matched-training supplement')
    
    add_common_arguments(parser)
    args = prepare_args(parser.parse_args())
    
    if not os.path.isabs(args.data_path):
        args.data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.data_path)
    
    run_experiment(args)


if __name__ == '__main__':
    main()
