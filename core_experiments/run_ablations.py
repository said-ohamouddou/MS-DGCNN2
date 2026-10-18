"""MS-DGCNN++ component ablations (5-fold CV, density dropout at 100% / 5% retention)."""

import os
import sys
import json
import argparse
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


K_SCALES = [5, 20, 30]
RETENTION_RATES = [1.0, 0.50, 0.05]

BASELINE_CONFIG = {
    'MS-DGCNN++ Default': {
        'factory': lambda nc: create_model(
            num_classes=nc,
            k_scales=K_SCALES,
            use_multiscale=True,
            use_normalized_features=True,
            local_use_normalized=False,
            fusion_type='concat_conv',
            emb_dims=1024,
            dropout=0.5
        ),
        'label': 'MS-DGCNN++ Default'
    }
}

SINGLE_SCALE_CONFIGS = {
    'DGCNN (k1=5, k=30)': {
        'factory': lambda nc: create_model(
            num_classes=nc,
            k_scales=[5, 5, 30],
            use_multiscale=False,
            use_normalized_features=False,
            local_use_normalized=False,
            fusion_type='concat_conv',
        ),
        'label': 'DGCNN (k1=5, k=30)'
    },
    'DGCNN (k1=20, k=30)': {
        'factory': lambda nc: create_model(
            num_classes=nc,
            k_scales=[20, 20, 30],
            use_multiscale=False,
            use_normalized_features=False,
            local_use_normalized=False,
            fusion_type='concat_conv',
        ),
        'label': 'DGCNN (k1=20, k=30)'
    },
    'DGCNN (k1=30, k=30)': {
        'factory': lambda nc: create_model(
            num_classes=nc,
            k_scales=[30, 30, 30],
            use_multiscale=False,
            use_normalized_features=False,
            local_use_normalized=False,
            fusion_type='concat_conv',
        ),
        'label': 'DGCNN (k1=30, k=30)'
    },
    'DGCNN+Norm (k1=5, k=30)': {
        'factory': lambda nc: create_model(
            num_classes=nc,
            k_scales=[5, 5, 30],
            use_multiscale=False,
            use_normalized_features=False,
            local_use_normalized=True,
            fusion_type='concat_conv',
        ),
        'label': 'DGCNN+Norm (k1=5, k=30)'
    },
    'DGCNN+Norm (k1=20, k=30)': {
        'factory': lambda nc: create_model(
            num_classes=nc,
            k_scales=[20, 20, 30],
            use_multiscale=False,
            use_normalized_features=False,
            local_use_normalized=True,
            fusion_type='concat_conv',
        ),
        'label': 'DGCNN+Norm (k1=20, k=30)'
    },
    'DGCNN+Norm (k1=30, k=30)': {
        'factory': lambda nc: create_model(
            num_classes=nc,
            k_scales=[30, 30, 30],
            use_multiscale=False,
            use_normalized_features=False,
            local_use_normalized=True,
            fusion_type='concat_conv',
        ),
        'label': 'DGCNN+Norm (k1=30, k=30)'
    },
}

FUSION_CONFIGS = {
    'MS-DGCNN++ (concat_only)': {
        'factory': lambda nc: create_model(
            num_classes=nc,
            k_scales=K_SCALES,
            use_multiscale=True,
            use_normalized_features=True,
            local_use_normalized=False,
            fusion_type='concat_only',
        ),
        'label': 'concat_only'
    },
    'MS-DGCNN++ (add)': {
        'factory': lambda nc: create_model(
            num_classes=nc,
            k_scales=K_SCALES,
            use_multiscale=True,
            use_normalized_features=True,
            local_use_normalized=False,
            fusion_type='add',
        ),
        'label': 'add'
    },
    'MS-DGCNN++ (attention)': {
        'factory': lambda nc: create_model(
            num_classes=nc,
            k_scales=K_SCALES,
            use_multiscale=True,
            use_normalized_features=True,
            local_use_normalized=False,
            fusion_type='attention',
        ),
        'label': 'attention'
    },
    'MS-DGCNN++ (gated)': {
        'factory': lambda nc: create_model(
            num_classes=nc,
            k_scales=K_SCALES,
            use_multiscale=True,
            use_normalized_features=True,
            local_use_normalized=False,
            fusion_type='gated',
        ),
        'label': 'gated'
    },
    'MS-DGCNN++ (se_fusion)': {
        'factory': lambda nc: create_model(
            num_classes=nc,
            k_scales=K_SCALES,
            use_multiscale=True,
            use_normalized_features=True,
            local_use_normalized=False,
            fusion_type='se_fusion',
        ),
        'label': 'se_fusion'
    },
}


def get_checkpoint_filename(model_name: str, fold: int) -> str:
    """Generate checkpoint filename."""
    safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'plus').replace('=', '')
    return f'{safe_name}_fold{fold}.pth'


def generate_combined_latex_table_dual_retention(baseline_results, single_scale_results,
                                                fusion_results, retention_rates):
    """Table 7 (any number of test retention rates); bold the highest displayed mean per column."""
    groups = [('Baseline', BASELINE_CONFIG, baseline_results),
              ('Single-scale DGCNN', SINGLE_SCALE_CONFIGS, single_scale_results),
              ('Multi-scale fusion', FUSION_CONFIGS, fusion_results)]
    keys = [f'r{int(round(r*100))}' for r in retention_rates]
    n = len(keys)
    best = {}
    for metric in ['oa', 'f1']:
        for key in keys:
            scale = 100 if metric == 'f1' else 1
            best[(metric,key)] = max(round(scale*d[name][key]['mean'][metric],1)
                                    for _, configs,d in groups for name in configs)
    rate_labels = ' & '.join([rf'{int(round(r*100))}\%' for r in retention_rates] * 2)
    lines = [r'\begin{table}[htbp]', r'\centering',
             r'\caption{Component ablation. Models trained on clean data and tested at upper-canopy retention $r$. Mean $\pm$ standard deviation across five folds (ddof=0). Bold: highest displayed mean per column.}',
             r'\label{tab:ablation_results}', r'\resizebox{\textwidth}{!}{%',
             r'\begin{tabular}{l' + 'c'*(2*n) + '}', r'\toprule',
             rf'Model & \multicolumn{{{n}}}{{c}}{{OA (\%)}} & \multicolumn{{{n}}}{{c}}{{F1 (\%)}} \\',
             rf'\cmidrule(lr){{2-{1+n}}}\cmidrule(lr){{{2+n}-{1+2*n}}}',
             r' & ' + rate_labels + r' \\']
    labels = {'concat_only':'Concat Only','add':'Add','attention':'Attention','gated':'Gated','se_fusion':'SE Fusion'}
    for heading, configs, data in groups:
        lines += [r'\midrule', rf'\multicolumn{{{1+2*n}}}{{l}}{{\textbf{{{heading}}}}} \\']
        for name, conf in configs.items():
            label = labels.get(conf['label'], conf['label']).replace('k1=',r'$k_1$=').replace('k=',r'$k$=')
            if name == 'MS-DGCNN++ Default':
                label = 'MS-DGCNN++ (DA, Concat Conv)'
            cells = [label]
            for metric in ['oa', 'f1']:
                scale = 100 if metric == 'f1' else 1
                for key in keys:
                    mean, sd = (scale*data[name][key][stat][metric] for stat in ['mean','std'])
                    cell = f'{mean:.1f}$\\pm${sd:.1f}'
                    if round(mean,1) == best[(metric,key)]:
                        cell = r'\best{' + cell + '}'
                    cells.append(cell)
            lines.append(' & '.join(cells) + r' \\')
    return '\n'.join(lines + [r'\bottomrule',r'\end{tabular}}',r'\end{table}'])


def get_checkpoint_filename_with_retention(model_name: str, fold: int, retention: float) -> str:
    """Generate checkpoint filename with retention rate."""
    safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'plus').replace('=', '')
    ret_pct = int(retention * 100)
    return f'{safe_name}_r{ret_pct}_fold{fold}.pth'


def train_clean_and_evaluate(model_config, model_name, fold, retention_rates, checkpoints_dir,
                             import_checkpoints, args, output_dir):
    """Clean training (shared cache), then testing at each upper-canopy retention rate."""
    ckpt_path = checkpoints_dir / get_checkpoint_filename(model_name, fold)
    if import_checkpoints:
        candidate = import_checkpoints / ckpt_path.name
        if candidate.exists():
            # The shared loader validates metadata before accepting this import.
            ckpt_path = candidate
    clean, model = run_shared_fold(args, fold, model_config['factory'], checkpoint_path=ckpt_path)
    results = {r: evaluate_corrupted(args, fold, model, clean, retention=r, output_dir=output_dir)
               for r in retention_rates}
    del model
    return results


def aggregate_fold_results(folds_data, epochs=100):
    """Macro-F1 is averaged over classes WITHIN each fold, then across folds."""
    return aggregate_metrics(folds_data, k_folds=5)


def run_outlier_study(args):
    """Table 5: clean training, fixed contaminated validation, matched scales."""
    from models.baselines import BASELINES, POINTM2AE_CKPT
    from utils.experiment_utils import save_results_json
    m2ae_ckpt = str(POINTM2AE_CKPT)
    if not POINTM2AE_CKPT.exists():
        raise FileNotFoundError(f'Point-M2AE pretrained weights not found: {m2ae_ckpt}')
    output_dir = Path(args.output_dir) / f'outliers_{datetime.now():%Y%m%d_%H%M%S}'
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(dict(vars(args), dgcnn_edgeconv_k=[5,30,30,30], ms_dgcnn_scales=[5,20,30], pointm2ae_pretrained=m2ae_ckpt,
                     outliers='append floor(rate*N) uniform bounding-box points before normalization; N=1024',
                     protocol_name='clean_train_corrupted_test'),
                output_dir / 'config.json')
    configs = {
        # Shared baseline definitions (models/baselines.py): same clean checkpoints as
        # Figures 4-5; DGCNN (k1=5, k=30) is also Table 7's single-scale DGCNN
        'DGCNN': BASELINES['dgcnn']['factory'],
        'MS-DGCNN': BASELINES['ms_dgcnn']['factory'],
        'MS-DGCNN++ (RO)': lambda nc: create_model(num_classes=nc, k_scales=[5,20,30],
                    use_normalized_features=False, local_use_normalized=False),
        'MS-DGCNN++ (HE)': lambda nc: create_model(num_classes=nc, k_scales=[5,20,30],
                    use_normalized_features=True, local_use_normalized=True),
        'MS-DGCNN++ (DA)': lambda nc: create_model(num_classes=nc, k_scales=[5,20,30],
                    use_normalized_features=True, local_use_normalized=False),
        'MS-DGCNN++ (RA)': lambda nc: create_model(num_classes=nc, k_scales=[5,20,30],
                    use_normalized_features=False, local_use_normalized=True),
        # Full fine-tuning from the ShapeNet-pretrained Point-M2AE weights
        'Point-M2AE': BASELINES['pointm2ae']['factory'],
    }
    rates = parse_levels(args.outlier_rates, '--outlier_rates')
    all_results = {name: {r: [] for r in rates} for name in configs}
    clean_results = {name: [] for name in configs}
    for name, factory in configs.items():
        for fold in range(args.k_folds):
            clean_metrics, model = run_shared_fold(args, fold, factory)
            # Same saved clean evaluation as Table 6 / Figure 4 / Table 7 at 100%
            clean_results[name].append(evaluate_corrupted(args, fold, model, clean_metrics,
                                                          retention=1.0, output_dir=output_dir))
            for rate in rates:
                all_results[name][rate].append(evaluate_corrupted(
                    args, fold, model, clean_metrics, outlier_rate=rate, output_dir=output_dir))
            del model
        save_results_json(all_results, output_dir / 'outlier_results.json')
    aggregated = {name: {r: aggregate_metrics(folds) for r,folds in rates_data.items()}
                  for name,rates_data in all_results.items()}
    clean_agg = {name: aggregate_metrics(folds) for name, folds in clean_results.items()}
    save_results_json({'outlier_rates': rates, 'clean': {n: {k: v for k, v in a.items() if k != 'folds'}
                       for n, a in clean_agg.items()},
                       'corrupted': {n: {str(r): {k: v for k, v in a.items() if k != 'folds'}
                                         for r, a in by_rate.items()} for n, by_rate in aggregated.items()}},
                      output_dir / 'outlier_summary.json')
    show_clean = 0.0 not in rates  # 0% outliers is the same clean evaluation
    header = ' & '.join((['Clean'] if show_clean else []) + [rf'{r*100:g}\%' for r in rates])
    lines = [r'\begin{table*}[htbp]',r'\centering',
        r'\caption{Outlier robustness. OA (\%), mean $\pm$ standard deviation across five folds (ddof=0). Models trained on clean data and tested with added outliers (percentage of the original input count). DGCNN: DGCNN ($k_1=5$, $k=30$); MS-DGCNN scales: [5,20,30], no internal FPS; Point-M2AE: pretrained, fully fine-tuned.}',
        r'\label{tab:robustness_all}', r'\resizebox{\textwidth}{!}{%',
        r'\begin{tabular}{l' + 'c'*(len(rates)+show_clean) + '}', r'\toprule',
        'Model & ' + header + r' \\', r'\midrule']
    for name in configs:
        c = clean_agg[name]
        cells=[name] + ([f"{c['mean']['oa']:.1f}$\\pm${c['std']['oa']:.1f}"] if show_clean else [])
        for rate in rates:
            a=aggregated[name][rate]
            cell=f"{a['mean']['oa']:.1f}$\\pm${a['std']['oa']:.1f}"
            best=max(round(aggregated[n][rate]['mean']['oa'],1) for n in configs)
            if round(a['mean']['oa'],1)==best:
                cell=r'\textbf{'+cell+'}'
            cells.append(cell)
        lines.append(' & '.join(cells)+r' \\')
    # Figure in the same style as the density (Fig. 4) and noise (Fig. 5) figures
    from utils.robustness_plot import plot_robustness, fold_stats
    key_map = {'MS-DGCNN++ (RO)': 'raw_only', 'MS-DGCNN++ (HE)': 'hybrid_everywhere',
               'MS-DGCNN++ (DA)': 'default', 'MS-DGCNN++ (RA)': 'reversed', 'DGCNN': 'dgcnn',
               'MS-DGCNN': 'ms_dgcnn', 'Point-M2AE': 'pointm2ae'}
    plot_robustness(fold_stats(all_results, rates, key_map), output_dir / 'outlier_robustness',
                    'Added outliers (% of input points)', xticklabels=[f'{r*100:g}' for r in rates],
                    legend_loc='lower left')
    (output_dir / 'outlier_results.tex').write_text('\n'.join(lines+[r'\bottomrule',r'\end{tabular}}',r'\end{table*}']))
    logger.info('Table 5 saved to %s',output_dir)


def main():
    global RETENTION_RATES
    parser = argparse.ArgumentParser(description='MS-DGCNN2 Component Ablation Studies')
    parser.add_argument('--data_path', type=str, default='data/STPCTLC',
                        help='Path to STPCTLC data directory')
    parser.add_argument('--output_dir', type=str, default='results/paper_v2/ablations',
                        help='Output directory for results')
    parser.add_argument('--import_checkpoints', type=str, 
                        default=None,
                        help='Path to import existing checkpoints (all retention rates)')
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
                        help='Evaluation only mode - load checkpoints and generate LaTeX, no training')
    
    parser.add_argument('--outliers', action='store_true', help='Run only the paper Table 5 outlier evaluation')
    parser.add_argument('--retention_rates', type=float, nargs='+', default=RETENTION_RATES,
                        help='Test-time upper-canopy retention rates for Table 7 (clean-trained models)')
    parser.add_argument('--outlier_rates', type=float, nargs='+',
                        default=[0, .01, .02, .05, .10, .15, .20, .25, .30, .40, .50, .60, .75, .90, 1.0],
                        help='Test-time outlier fractions for Table 5 (clean-trained models)')
    add_common_arguments(parser)
    args = prepare_args(parser.parse_args())
    RETENTION_RATES = parse_levels(args.retention_rates, '--retention_rates')
    if args.outliers:
        run_outlier_study(args)
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.data_path):
        args.data_path = os.path.join(script_dir, args.data_path)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(script_dir, args.output_dir)
    if args.import_checkpoints and not os.path.isabs(args.import_checkpoints):
        args.import_checkpoints = os.path.join(script_dir, args.import_checkpoints)

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / timestamp
    checkpoints_dir = output_dir / 'checkpoints'
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(exist_ok=True)

    import_checkpoints = None
    if args.import_checkpoints:
        import_path = Path(args.import_checkpoints)
        if import_path.exists():
            import_checkpoints = import_path
            logger.info(f"Will import checkpoints from: {import_path}")
        else:
            logger.warning(f"Import path not found: {import_path}")

    generate_cv_splits(args.data_path, k_folds=args.k_folds, seed=args.seed)

    train_loader, val_loader, num_classes, class_names = get_cv_dataloaders(
        args.data_path, fold=0, batch_size=args.batch_size,
        num_points=args.num_points, seed=args.seed, k_folds=args.k_folds
    )
    
    all_configs = {**BASELINE_CONFIG, **SINGLE_SCALE_CONFIGS, **FUSION_CONFIGS}
    total_experiments = len(all_configs) * args.k_folds * len(RETENTION_RATES)
    
    print("=" * 70)
    print("MS-DGCNN2 COMPONENT ABLATION STUDIES")
    print("=" * 70)
    print(f"Data: {args.data_path}")
    print(f"Output: {output_dir}")
    print(f"Retention rates: {[f'{int(r*100)}%' for r in RETENTION_RATES]}")
    print(f"Baseline: {list(BASELINE_CONFIG.keys())}")
    print(f"Single-scale variants: {list(SINGLE_SCALE_CONFIGS.keys())}")
    print(f"Fusion variants: {list(FUSION_CONFIGS.keys())}")
    print(f"Folds: {args.k_folds}")
    print(f"Total experiments: {total_experiments}")
    print("=" * 70)

    config = {
        'experiment': 'Component Ablation Studies with Density Dropout',
        'protocol_name': 'clean_train_corrupted_test',
        'data_path': args.data_path,
        'k_folds': args.k_folds,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'num_points': args.num_points,
        'k_scales': K_SCALES,
        'retention_rates': RETENTION_RATES,
        'seed': args.seed,
        'emb_dims': 1024,
        'dropout': 0.5,
        'timestamp': datetime.now().isoformat()
    }
    save_config(dict(config, arguments=vars(args)), output_dir / 'config.json')

    groups = [('BASELINE: MS-DGCNN++ Default', BASELINE_CONFIG, {m: {} for m in BASELINE_CONFIG}),
              ('ABLATION 1: Single-Scale DGCNN (k value effect)', SINGLE_SCALE_CONFIGS, {m: {} for m in SINGLE_SCALE_CONFIGS}),
              ('ABLATION 2: Fusion Method Comparison', FUSION_CONFIGS, {m: {} for m in FUSION_CONFIGS})]
    for heading, configs, store in groups:
        print(f"\n{'='*60}\n{heading}\n{'='*60}")
        for model_name, model_config in configs.items():
            print(f"\n  {model_config['label']}")
            per_rate = {r: [] for r in RETENTION_RATES}
            for fold in range(args.k_folds):
                fold_results = train_clean_and_evaluate(model_config, model_name, fold, RETENTION_RATES,
                                                        checkpoints_dir, import_checkpoints, args, output_dir)
                for r, metrics in fold_results.items():
                    per_rate[r].append(metrics)
            for r, folds_data in per_rate.items():
                store[model_name][f'r{int(round(r*100))}'] = aggregate_fold_results(folds_data, args.epochs)
    baseline_results, single_scale_results, fusion_results = (g[2] for g in groups)

    results_save = {
        'k_folds': args.k_folds,
        'retention_rates': RETENTION_RATES,
        'baseline': baseline_results,
        'single_scale': single_scale_results,
        'fusion': fusion_results,
    }
    
    from utils.experiment_utils import save_results_json
    save_results_json(results_save, output_dir / 'results.json')

    latex_combined = generate_combined_latex_table_dual_retention(
        baseline_results, single_scale_results, fusion_results, RETENTION_RATES
    )
    with open(output_dir / 'ablation_results.tex', 'w') as f:
        f.write(latex_combined)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for retention in RETENTION_RATES:
        ret_pct = int(round(retention * 100))
        ret_key = f'r{ret_pct}'
        print(f"\n--- Retention = {ret_pct}% ---")
        
        print("\nBaseline:")
        for m in BASELINE_CONFIG.keys():
            if m in baseline_results and ret_key in baseline_results[m]:
                r = baseline_results[m][ret_key]
                if 'mean' in r:
                    print(f"  {m}: OA={r['mean']['oa']:.1f}±{r['std']['oa']:.1f}")
        
        print("\nSingle-Scale DGCNN:")
        for m in SINGLE_SCALE_CONFIGS.keys():
            if m in single_scale_results and ret_key in single_scale_results[m]:
                r = single_scale_results[m][ret_key]
                if 'mean' in r:
                    print(f"  {m}: OA={r['mean']['oa']:.1f}±{r['std']['oa']:.1f}")
        
        print("\nFusion Methods:")
        for m in FUSION_CONFIGS.keys():
            if m in fusion_results and ret_key in fusion_results[m]:
                r = fusion_results[m][ret_key]
                if 'mean' in r:
                    label = FUSION_CONFIGS[m]['label']
                    print(f"  {label}: OA={r['mean']['oa']:.1f}±{r['std']['oa']:.1f}")
    
    print("\n" + "=" * 70)
    print("OUTPUTS SAVED")
    print("=" * 70)
    print(f"  - {output_dir}/results.json")
    print(f"  - {output_dir}/ablation_results.tex")
    print("=" * 70)


if __name__ == '__main__':
    main()
