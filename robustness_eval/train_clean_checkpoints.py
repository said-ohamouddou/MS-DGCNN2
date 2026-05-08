"""Train multi-model checkpoints (5-fold CV) for reuse in robustness scripts."""

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
from models.msdgcnn import create_ms_dgcnn
from models.pointm2ae.pointm2ae import create_pointm2ae
from utils.experiment_utils import (
    get_cv_dataloaders, train_model, set_seed, save_config,
    generate_cv_splits, evaluate
)

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



K_SCALES = [5, 20, 30]

MODEL_CONFIGS = {
    'DGCNN': {
        'factory': lambda nc: create_model(
            num_classes=nc,
            k_scales=K_SCALES,
            use_multiscale=False,
            use_normalized_features=False,
            fusion_type='concat_conv',
        ),
        'label': 'DGCNN'
    },
    'MS-DGCNN': {
        'factory': lambda nc: create_ms_dgcnn(
            num_classes=nc,
            k_scales=K_SCALES,
        ),
        'label': 'MS-DGCNN'
    },
    'Point-M2AE': {
        'factory': lambda nc: create_pointm2ae(
            num_classes=nc,
            ckpt_path=None,
        ),
        'label': 'Point-M2AE'
    },
    'MS-DGCNN++ (Raw-only)': {
        'factory': lambda nc: create_model(
            num_classes=nc,
            k_scales=K_SCALES,
            use_multiscale=True,
            use_normalized_features=False,
            local_use_normalized=False,
            fusion_type='concat_conv',
        ),
        'label': 'MS-DGCNN++ (Raw-only)'
    },
    'MS-DGCNN++ (Hybrid-everywhere)': {
        'factory': lambda nc: create_model(
            num_classes=nc,
            k_scales=K_SCALES,
            use_multiscale=True,
            use_normalized_features=True,
            local_use_normalized=True,
            fusion_type='concat_conv',
        ),
        'label': 'MS-DGCNN++ (Hybrid-everywhere)'
    },
    'MS-DGCNN++ (Default Asymmetric)': {
        'factory': lambda nc: create_model(
            num_classes=nc,
            k_scales=K_SCALES,
            use_multiscale=True,
            use_normalized_features=True,
            local_use_normalized=False,
            fusion_type='concat_conv',
        ),
        'label': 'MS-DGCNN++ (Default Asymmetric)'
    },
}


def get_checkpoint_filename(model_name: str, fold: int) -> str:
    """Generate checkpoint filename."""
    safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'plus')
    return f'{safe_name}_fold{fold}.pth'


def train_all_models(args):
    """Train all models on clean data with 5-fold CV."""
    
    # Setup
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate CV splits
    generate_cv_splits(args.data_path, k_folds=args.k_folds, seed=args.seed)
    
    # Save config
    config = {
        'experiment': 'Train Clean Checkpoints',
        'data_path': args.data_path,
        'k_folds': args.k_folds,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'num_points': args.num_points,
        'k_scales': K_SCALES,
        'patience': args.patience,
        'seed': args.seed,
        'models': list(MODEL_CONFIGS.keys()),
        'timestamp': datetime.now().isoformat()
    }
    save_config(config, output_dir / 'config.json')
    
    # Get class info from first fold
    train_loader, val_loader, num_classes, class_names = get_cv_dataloaders(
        args.data_path, fold=0, batch_size=args.batch_size,
        num_points=args.num_points, seed=args.seed, k_folds=args.k_folds
    )
    logger.info(f"Dataset: {num_classes} classes - {class_names}")
    
    # Results tracker
    all_results = {}
    
    # Count existing checkpoints
    total_runs = len(MODEL_CONFIGS) * args.k_folds
    existing_runs = 0
    for model_name in MODEL_CONFIGS.keys():
        for fold in range(args.k_folds):
            ckpt_path = output_dir / get_checkpoint_filename(model_name, fold)
            if ckpt_path.exists():
                existing_runs += 1
    
    logger.info(f"Total runs: {total_runs}, Existing: {existing_runs}, To train: {total_runs - existing_runs}")
    
    # Train each model
    for model_name, model_config in MODEL_CONFIGS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Model: {model_config['label']}")
        logger.info(f"{'='*60}")
        
        model_results = {'folds': [], 'mean': {}, 'std': {}}
        
        for fold in range(args.k_folds):
            ckpt_path = output_dir / get_checkpoint_filename(model_name, fold)
            
            # Check if checkpoint exists
            if ckpt_path.exists() and not args.force_retrain:
                logger.info(f"  Fold {fold}: Loading existing checkpoint")
                checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
                fold_metrics = checkpoint.get('metrics', {})
                model_results['folds'].append(fold_metrics)
                continue
            
            logger.info(f"  Fold {fold}: Training...")
            
            # Set seed for reproducibility (different per fold, matching experiment1_ablation.py)
            set_seed(args.seed + fold)
            
            # Get dataloaders for this fold
            train_loader, val_loader, num_classes, _ = get_cv_dataloaders(
                args.data_path, fold=fold, batch_size=args.batch_size,
                num_points=args.num_points, seed=args.seed, k_folds=args.k_folds
            )
            
            # Create model
            try:
                model = model_config['factory'](num_classes)
                model = model.to(device)
            except Exception as e:
                logger.error(f"  Failed to create model: {e}")
                continue
            
            # Train
            best_metrics, model = train_model(
                model, train_loader, val_loader, device,
                epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                save_path=str(ckpt_path), patience=args.patience
            )
            
            model_results['folds'].append(best_metrics)
            logger.info(f"  Fold {fold}: OA={best_metrics.get('oa', 0):.2f}%, mAcc={best_metrics.get('macc', 0):.2f}%")
        
        # Compute mean and std across folds
        if model_results['folds']:
            metrics_keys = ['oa', 'macc']
            for key in metrics_keys:
                values = [f.get(key, 0) for f in model_results['folds'] if key in f]
                if values:
                    model_results['mean'][key] = np.mean(values)
                    model_results['std'][key] = np.std(values)
            
            logger.info(f"  Mean OA: {model_results['mean'].get('oa', 0):.2f} ± {model_results['std'].get('oa', 0):.2f}%")
        
        all_results[model_name] = model_results
    
    # Save summary results
    results_path = output_dir / 'training_results.json'
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj
    
    serializable_results = convert_to_serializable(all_results)
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"\nResults saved to: {results_path}")
    logger.info(f"Checkpoints saved to: {output_dir}")
    
    # Print summary table
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"{'Model':<40} {'OA (mean±std)':<15} {'mAcc (mean±std)':<15}")
    print("-"*70)
    for model_name, results in all_results.items():
        oa_mean = results['mean'].get('oa', 0)
        oa_std = results['std'].get('oa', 0)
        macc_mean = results['mean'].get('macc', 0)
        macc_std = results['std'].get('macc', 0)
        print(f"{model_name:<40} {oa_mean:.2f}±{oa_std:.2f}%      {macc_mean:.2f}±{macc_std:.2f}%")
    print("="*70)
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Train Clean Checkpoints for Robustness Evaluation')
    parser.add_argument('--data_path', type=str, default='data/STPCTLC',
                        help='Path to STPCTLC data directory')
    parser.add_argument('--output_dir', type=str, default='results/clean_checkpoints/clean_cv',
                        help='Output directory for checkpoints')
    parser.add_argument('--k_folds', type=int, default=5,
                        help='Number of CV folds')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='Number of points per sample')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--force_retrain', action='store_true',
                        help='Force retraining even if checkpoints exist')
    
    args = parser.parse_args()
    
    # Handle relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.data_path):
        args.data_path = os.path.join(script_dir, args.data_path)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(script_dir, args.output_dir)
    
    train_all_models(args)


if __name__ == '__main__':
    main()
