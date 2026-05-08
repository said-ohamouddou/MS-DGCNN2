"""
Data utilities for MS-DGCNN2 ablation studies.
Loads STPCTLC dataset with z_rotate_tree augmentation.
"""

import os
import sys
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RandomZRotate:
    """Random rotation around Z-axis (vertical) for tree point clouds."""
    
    def __call__(self, points):
        """
        Args:
            points: (N, 3) numpy array
        Returns:
            Rotated points (N, 3)
        """
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


class TreeSpeciesDataset(Dataset):
    """
    STPCTLC Tree Species Dataset for ablation studies.
    """
    
    def __init__(self, data_path, subset='train', num_points=1024, 
                 augment=True, seed=42):
        """
        Args:
            data_path: Path to data directory (e.g., 'data/STPCTLC')
            subset: 'train' or 'val'
            num_points: Number of points per sample
            augment: Apply data augmentation (z_rotate_tree)
            seed: Random seed for reproducibility
        """
        self.num_points = num_points
        self.subset = subset
        self.augment = augment and (subset == 'train')
        
        # Load data from h5 file
        split_h5_path = os.path.join(data_path, 'data_split_simple.h5')
        
        if not os.path.exists(split_h5_path):
            raise FileNotFoundError(
                f"Split file not found: {split_h5_path}\n"
                f"Please run the main dataset loader first to create the split."
            )
        
        with h5py.File(split_h5_path, 'r') as f:
            self.classes = [c.decode() if isinstance(c, bytes) else c 
                           for c in f['classes'][:]]
            
            if subset not in f:
                available = [k for k in f.keys() if k != 'classes']
                raise ValueError(f"Subset '{subset}' not found. Available: {available}")
            
            self.data = f[subset]['point_clouds'][:]
            self.labels = f[subset]['labels'][:]
        
        # Ensure labels are 1D
        if self.labels.ndim > 1:
            self.labels = self.labels.flatten()
        
        self.num_classes = len(self.classes)
        self.augmentation = RandomZRotate() if self.augment else None
        
        print(f"Loaded {len(self.data)} samples for '{subset}' "
              f"({self.num_classes} classes, augment={self.augment})")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        points = self.data[idx][:self.num_points].copy()
        label = int(self.labels[idx])
        
        # Normalize
        points = normalize_pc(points)
        
        # Apply augmentation
        if self.augmentation is not None:
            points = self.augmentation(points)
        
        return torch.from_numpy(points.astype(np.float32)), label


def get_dataloaders(data_path, batch_size=16, num_points=1024, 
                    num_workers=4, seed=42):
    """
    Create train and validation dataloaders.
    
    Args:
        data_path: Path to data directory
        batch_size: Batch size
        num_points: Number of points per sample
        num_workers: Number of data loading workers
        seed: Random seed
    
    Returns:
        train_loader, val_loader, num_classes, class_names
    """
    train_dataset = TreeSpeciesDataset(
        data_path, subset='train', num_points=num_points,
        augment=True, seed=seed
    )
    val_dataset = TreeSpeciesDataset(
        data_path, subset='val', num_points=num_points,
        augment=False, seed=seed
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader, train_dataset.num_classes, train_dataset.classes


if __name__ == '__main__':
    # Test data loading
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'STPCTLC')
    
    train_loader, val_loader, num_classes, class_names = get_dataloaders(
        data_path, batch_size=16, num_points=1024
    )
    
    print(f"\nClasses ({num_classes}): {class_names}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Test one batch
    points, labels = next(iter(train_loader))
    print(f"\nBatch shape: {points.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Label values: {labels.tolist()}")
