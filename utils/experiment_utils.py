"""Training, CV splits, metrics, and plotting helpers for MS-DGCNN++ experiments."""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import h5py
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_recall_fscore_support, f1_score
)
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import math
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class RandomZRotate:
    """Random rotation around Z-axis (vertical) for tree point clouds."""
    
    def __call__(self, points):
        theta = np.random.uniform(0, 2 * math.pi)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array([
            [cos_t, -sin_t, 0],
            [sin_t, cos_t, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        return points @ rotation_matrix.T


def normalize_pc(points):
    """Normalize point cloud to unit sphere."""
    centroid = np.mean(points, axis=0)
    points = points - centroid
    furthest_distance = np.max(np.sqrt(np.sum(points ** 2, axis=-1)))
    points = points / (furthest_distance + 1e-8)
    return points



def generate_cv_splits(data_path, k_folds=5, seed=42, force_regenerate=False):
    """
    Generate and save k-fold CV splits to a JSON file.
    Ensures identical splits across all experiments.
    
    Returns:
        splits_path: Path to the JSON file containing splits
    """
    splits_path = os.path.join(data_path, f'cv_splits_k{k_folds}_seed{seed}.json')
    
    if os.path.exists(splits_path) and not force_regenerate:
        logger.info(f"Using existing CV splits: {splits_path}")
        return splits_path
    
    # Load data to get labels for stratification
    h5_path = os.path.join(data_path, 'point_cloud_data.h5')
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Data file not found: {h5_path}")
    
    with h5py.File(h5_path, 'r') as f:
        labels = f['labels'][:]
        classes = [c.decode() if isinstance(c, bytes) else c for c in f['classes'][:]]
    
    if labels.ndim > 1:
        labels = labels.flatten()
    
    # Generate stratified k-fold splits
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    indices = np.arange(len(labels))
    
    splits = {
        'k_folds': k_folds,
        'seed': seed,
        'classes': classes,
        'total_samples': len(labels),
        'folds': []
    }
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(indices, labels)):
        splits['folds'].append({
            'fold': fold_idx,
            'train_indices': train_idx.tolist(),
            'val_indices': val_idx.tolist()
        })
    
    with open(splits_path, 'w') as f:
        json.dump(splits, f, indent=2)
    
    logger.info(f"Generated CV splits: {splits_path}")
    return splits_path


def load_cv_splits(splits_path):
    """Load CV splits from JSON file."""
    with open(splits_path, 'r') as f:
        return json.load(f)



class TreeSpeciesDatasetCV(Dataset):
    """
    Tree Species Dataset with CV fold support.
    Supports density dropout and noise injection for experiments.
    """
    
    def __init__(self, data_path, fold, partition='train', num_points=1024,
                 augment=True, seed=42, k_folds=5,
                 density_retention=1.0, noise_sigma=0.0,
                 density_seed=None):
        """
        Args:
            data_path: Path to data directory
            fold: Fold index (0 to k_folds-1)
            partition: 'train' or 'val'
            num_points: Number of points per sample
            augment: Apply RandomZRotate augmentation (train only)
            seed: Random seed for CV splits
            k_folds: Number of folds
            density_retention: Retention rate for upper canopy (Exp 2)
            noise_sigma: Gaussian noise std in mm (Exp 3)
            density_seed: Seed for density dropout (for reproducibility)
        """
        self.num_points = num_points
        self.partition = partition
        self.augment = augment and (partition == 'train')
        self.density_retention = density_retention
        self.noise_sigma = noise_sigma
        self.density_seed = density_seed
        
        # Load data
        h5_path = os.path.join(data_path, 'point_cloud_data.h5')
        with h5py.File(h5_path, 'r') as f:
            self.all_data = f['point_clouds'][:]
            self.all_labels = f['labels'][:]
            self.classes = [c.decode() if isinstance(c, bytes) else c 
                           for c in f['classes'][:]]
        
        if self.all_labels.ndim > 1:
            self.all_labels = self.all_labels.flatten()
        
        # Load CV splits
        splits_path = generate_cv_splits(data_path, k_folds=k_folds, seed=seed)
        splits = load_cv_splits(splits_path)
        
        fold_data = splits['folds'][fold]
        if partition == 'train':
            indices = fold_data['train_indices']
        else:
            indices = fold_data['val_indices']
        
        self.data = self.all_data[indices]
        self.labels = self.all_labels[indices]
        self.num_classes = len(self.classes)
        
        self.augmentation = RandomZRotate() if self.augment else None
        
        logger.info(f"Loaded {len(self.data)} samples for fold {fold} {partition}")
    
    def apply_density_dropout(self, points, seed=None):
        """Apply density dropout to upper canopy points."""
        if self.density_retention >= 1.0:
            return points
        
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random
        
        # Compute median height
        median_z = np.median(points[:, 2])
        
        # Split into upper and lower
        upper_mask = points[:, 2] > median_z
        lower_mask = ~upper_mask
        
        upper_points = points[upper_mask]
        lower_points = points[lower_mask]
        
        # Subsample upper canopy
        n_upper = len(upper_points)
        n_keep = max(1, int(n_upper * self.density_retention))
        
        if n_keep < n_upper:
            keep_idx = rng.choice(n_upper, n_keep, replace=False)
            upper_points = upper_points[keep_idx]
        
        # Combine
        combined = np.vstack([lower_points, upper_points])
        
        # Resample to num_points if needed
        if len(combined) < self.num_points:
            # Duplicate points
            repeat_factor = (self.num_points // len(combined)) + 1
            combined = np.tile(combined, (repeat_factor, 1))[:self.num_points]
        elif len(combined) > self.num_points:
            idx = rng.choice(len(combined), self.num_points, replace=False)
            combined = combined[idx]
        
        return combined
    
    def apply_noise(self, points):
        """Apply Gaussian noise to point coordinates."""
        if self.noise_sigma <= 0:
            return points
        
        # Convert mm to normalized units (assuming unit sphere normalization)
        # This is applied after normalization, so we need to scale appropriately
        noise = np.random.normal(0, self.noise_sigma / 1000.0, points.shape)
        return points + noise.astype(np.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        points = self.data[idx][:self.num_points].copy()
        label = int(self.labels[idx])
        
        # Apply density dropout (before normalization)
        if self.density_retention < 1.0:
            seed = None if self.density_seed is None else self.density_seed + idx
            points = self.apply_density_dropout(points, seed=seed)
        
        # Normalize
        points = normalize_pc(points)
        
        # Apply noise (after normalization)
        if self.noise_sigma > 0:
            points = self.apply_noise(points)
        
        # Apply augmentation
        if self.augmentation is not None:
            points = self.augmentation(points)
        
        return torch.from_numpy(points.astype(np.float32)), label


def get_cv_dataloaders(data_path, fold, batch_size=16, num_points=1024,
                       num_workers=4, seed=42, k_folds=5,
                       density_retention=1.0, noise_sigma=0.0,
                       density_seed=None):
    """Create train and val dataloaders for a specific fold."""
    
    train_dataset = TreeSpeciesDatasetCV(
        data_path, fold=fold, partition='train', num_points=num_points,
        augment=True, seed=seed, k_folds=k_folds,
        density_retention=density_retention, noise_sigma=noise_sigma,
        density_seed=density_seed
    )
    
    val_dataset = TreeSpeciesDatasetCV(
        data_path, fold=fold, partition='val', num_points=num_points,
        augment=False, seed=seed, k_folds=k_folds,
        density_retention=density_retention, noise_sigma=noise_sigma,
        density_seed=density_seed
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )
    
    return train_loader, val_loader, train_dataset.num_classes, train_dataset.classes



def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)
    for points, labels in pbar:
        points = points.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(points)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return total_loss / len(train_loader), 100. * correct / total


def evaluate(model, val_loader, criterion, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for points, labels in val_loader:
            points = points.to(device)
            labels = labels.to(device)
            
            outputs = model(points)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    oa = accuracy_score(all_labels, all_preds) * 100
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    # Mean class accuracy
    cm = confusion_matrix(all_labels, all_preds)
    per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-8)
    macc = per_class_acc.mean() * 100
    
    return {
        'loss': total_loss / len(val_loader),
        'oa': oa,
        'macc': macc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels
    }


def train_model(model, train_loader, val_loader, device, 
                epochs=100, lr=0.0005, weight_decay=0.0001,
                save_path=None, patience=20):
    """
    Full training loop with early stopping.
    
    Returns:
        best_metrics: Best validation metrics (includes 'training_time' in seconds)
        model: Trained model (loaded with best weights)
    """
    import time
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_oa = 0
    best_metrics = None
    patience_counter = 0
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        if val_metrics['oa'] > best_oa:
            best_oa = val_metrics['oa']
            best_metrics = val_metrics.copy()
            best_metrics['training_time'] = time.time() - start_time
            patience_counter = 0
            logger.info(f"New best model at epoch {epoch}: OA={best_oa:.2f}%, mAcc={val_metrics['macc']:.2f}%")
            
            if save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_oa': best_oa,
                    'metrics': best_metrics
                }, save_path)
        else:
            patience_counter += 1
        
        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                f"Val OA={val_metrics['oa']:.2f}%, Val mAcc={val_metrics['macc']:.2f}%"
            )
        
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    total_time = time.time() - start_time
    if best_metrics:
        best_metrics['total_training_time'] = total_time
    
    # Load best model
    if save_path and os.path.exists(save_path):
        checkpoint = torch.load(save_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    return best_metrics, model



def plot_grouped_bar_chart(data, labels, group_labels, title, ylabel, 
                           save_path=None, figsize=(10, 6), error_bars=None):
    """
    Create grouped bar chart.
    
    Args:
        data: 2D array [n_groups, n_bars]
        labels: Bar labels
        group_labels: Group labels
        error_bars: Optional error bars [n_groups, n_bars]
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    n_groups = len(group_labels)
    n_bars = len(labels)
    x = np.arange(n_groups)
    width = 0.8 / n_bars
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_bars))
    
    for i, (label, color) in enumerate(zip(labels, colors)):
        offset = (i - n_bars/2 + 0.5) * width
        yerr = error_bars[:, i] if error_bars is not None else None
        ax.bar(x + offset, data[:, i], width, label=label, color=color, yerr=yerr, capsize=3)
    
    ax.set_xlabel('Variant')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    
    plt.close()


def plot_heatmap(data, row_labels, col_labels, title, save_path=None,
                 figsize=(10, 8), cmap='RdYlGn', vmin=0, vmax=1, annot=True):
    """Create heatmap visualization."""
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(data, annot=annot, fmt='.2f', cmap=cmap,
                xticklabels=col_labels, yticklabels=row_labels,
                vmin=vmin, vmax=vmax, ax=ax, cbar_kws={'label': 'F1 Score'})
    
    ax.set_title(title)
    ax.set_xlabel('Variant')
    ax.set_ylabel('Species')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    
    plt.close()


def plot_confusion_matrix_grid(cms, titles, class_names, save_path=None,
                               figsize=(16, 16), normalize=True):
    """Plot 2x2 grid of confusion matrices."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    for idx, (cm, title, ax) in enumerate(zip(cms, titles, axes)):
        if normalize:
            cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
        else:
            cm_norm = cm
        
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax, vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    
    plt.close()


def plot_degradation_curves(x_values, y_data, labels, title, xlabel, ylabel,
                            save_path=None, figsize=(10, 6), log_x=True,
                            error_bands=None, secondary_y=None, secondary_label=None):
    """
    Plot degradation curves with optional error bands and secondary y-axis.
    
    Args:
        x_values: X-axis values
        y_data: Dict of {label: y_values}
        error_bands: Dict of {label: std_values}
        secondary_y: Optional secondary y-axis data
    """
    fig, ax1 = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(labels)))
    
    for label, color in zip(labels, colors):
        y = y_data[label]
        ax1.plot(x_values, y, 'o-', label=label, color=color, linewidth=2, markersize=6)
        
        if error_bands and label in error_bands:
            std = error_bands[label]
            ax1.fill_between(x_values, y - std, y + std, alpha=0.2, color=color)
    
    if log_x:
        ax1.set_xscale('log')
    
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(title)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    if secondary_y is not None:
        ax2 = ax1.twinx()
        ax2.plot(x_values, secondary_y, 's--', color='gray', label=secondary_label, alpha=0.7)
        ax2.set_ylabel(secondary_label, color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    
    plt.close()


def save_results_table(data, columns, index, save_path, title="Results"):
    """Save results as CSV and LaTeX table."""
    import pandas as pd
    
    df = pd.DataFrame(data, columns=columns, index=index)
    
    # Save CSV
    csv_path = save_path.replace('.tex', '.csv')
    df.to_csv(csv_path)
    
    # Save LaTeX
    latex = df.to_latex(float_format='%.2f', caption=title, label=f'tab:{Path(save_path).stem}')
    with open(save_path, 'w') as f:
        f.write(latex)
    
    return df


def save_config(config, save_path):
    """Save experiment configuration to JSON."""
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



class ExperimentResults:
    """Manage experiment results across folds and variants."""
    
    def __init__(self, experiment_name, output_dir):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def add_fold_result(self, variant, fold, metrics):
        """Add results for a specific variant and fold."""
        if variant not in self.results:
            self.results[variant] = {}
        self.results[variant][fold] = metrics
    
    def get_aggregated_metrics(self, variant):
        """Get mean and std of metrics across folds."""
        fold_results = self.results.get(variant, {})
        if not fold_results:
            return None
        
        oas = [r['oa'] for r in fold_results.values()]
        maccs = [r['macc'] for r in fold_results.values()]
        f1s = np.array([r['f1'] for r in fold_results.values()])
        
        return {
            'oa_mean': np.mean(oas),
            'oa_std': np.std(oas),
            'macc_mean': np.mean(maccs),
            'macc_std': np.std(maccs),
            'f1_mean': np.mean(f1s, axis=0),
            'f1_std': np.std(f1s, axis=0)
        }
    
    def save(self):
        """Save all results to JSON."""
        save_path = self.output_dir / f'{self.experiment_name}_results_{self.timestamp}.json'
        
        # Convert numpy arrays to lists for JSON serialization
        serializable = {}
        for variant, folds in self.results.items():
            serializable[variant] = {}
            for fold, metrics in folds.items():
                serializable[variant][fold] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in metrics.items()
                }
        
        with open(save_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        logger.info(f"Results saved to {save_path}")
        return save_path
