"""
@Author: Said Ohamouddou
@File: models.py
@Time: 2025/07/13 13:18 PM
"""
"""
Paper: MS-DGCNN++: A Multi-Scale Fusion Dynamic Graph Neural Network with Biological Knowledge Integration for LiDAR Tree Species Classification
Said Ohamouddou, Abdellatif El Afia, Hanaa El Afia, Raddouane Chiheb
"""


import os
import sys
import glob
import numpy as np
import h5py
from torch.utils.data import Dataset
import open3d as o3d
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
np.random.seed(42)

def load_point_cloud(file_path):
    """Load point cloud data from various formats (xyz, pts, txt)."""
    file_extension = Path(file_path).suffix.lower()
    try:
        if file_extension == '.xyz':
            points = np.loadtxt(file_path)
            return points[:, :3]  # Take only x, y, z coordinates
        elif file_extension == '.pts':
            with open(file_path, 'r') as f:
                # Skip header if present
                first_line = f.readline().strip()
                if not all(c.isdigit() or c == '.' or c == '-' or c.isspace() for c in first_line):
                    points = np.loadtxt(file_path, skiprows=1)
                else:
                    f.seek(0)
                    points = np.loadtxt(file_path)
            return points[:, :3]
        elif file_extension == '.txt':
            points = np.loadtxt(file_path)
            return points[:, :3]
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {str(e)}")
        raise

def point_selection(point_cloud_path, target_point_count):
    """Downsample point cloud to target point count using farthest point sampling."""
    try:
        points = load_point_cloud(point_cloud_path)
        if len(points) < target_point_count:
            logger.warning(f"Point cloud {point_cloud_path} has fewer points ({len(points)}) than target ({target_point_count})")
            # Duplicate points to reach target count
            points = np.repeat(points, (target_point_count // len(points)) + 1, axis=0)[:target_point_count]
            return points

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        pcd_down = point_cloud.farthest_point_down_sample(target_point_count)
        return np.asarray(pcd_down.points)
    except Exception as e:
        logger.error(f"Error in point selection for {point_cloud_path}: {str(e)}")
        raise

def load_data(num_points):
    """Load and process point cloud data, saving to H5 format."""
    folder_path = "data_tree"
    h5_path = "point_cloud_data.h5"
    
    try:
        classes = sorted([d for d in os.listdir(folder_path) 
                        if os.path.isdir(os.path.join(folder_path, d))])
        
        if not classes:
            raise ValueError(f"No class directories found in {folder_path}")

        point_clouds = []
        labels = []
        
        for class_idx, class_name in enumerate(classes):
            class_path = os.path.join(folder_path, class_name)
            files = [f for f in os.listdir(class_path) 
                    if f.endswith(('.xyz', '.pts', '.txt'))]
            
            if not files:
                logger.warning(f"No valid files found in class {class_name}")
                continue
                
            for file_name in files:
                try:
                    file_path = os.path.join(class_path, file_name)
                    points = point_selection(file_path, num_points)
                    point_clouds.append(points)
                    labels.append(np.array([class_idx]))
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
                    continue

        if not point_clouds:
            raise ValueError("No valid point cloud data was loaded")

        point_clouds = np.array(point_clouds)
        labels = np.array(labels)

        # Save to H5 file
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('point_clouds', data=point_clouds)
            f.create_dataset('labels', data=labels)
            # Store classes as a dataset instead of attributes
            f.create_dataset('classes', data=np.array(classes, dtype='S'))

        logger.info("Data loading completed successfully")
        return h5_path
    except Exception as e:
        logger.error(f"Error in load_data: {str(e)}")
        raise

def data_split(h5_path, test_ratio=0.2):
    """Split data into train and test sets."""
    try:
        with h5py.File(h5_path, 'r') as f:
            point_clouds = f['point_clouds'][:]
            labels = f['labels'][:]
            classes = [c.decode() if isinstance(c, bytes) else c for c in f['classes'][:]]

        # Split data into train and test sets
        train_point_clouds, test_point_clouds, train_labels, test_labels = train_test_split(
            point_clouds, labels, test_size=test_ratio, stratify=labels, random_state=42
        )

        # Save split data to H5 file
        with h5py.File('data_tree/stpctlc_data_split.h5', 'w') as f:
            # Train data
            train_group = f.create_group('train')
            train_group.create_dataset('point_clouds', data=train_point_clouds)
            train_group.create_dataset('labels', data=train_labels)
            
            # Test data
            test_group = f.create_group('test')
            test_group.create_dataset('point_clouds', data=test_point_clouds)
            test_group.create_dataset('labels', data=test_labels)
            
            # Store class names as a dataset
            f.create_dataset('classes', data=np.array(classes, dtype='S'))

        logger.info("Data splitting completed successfully")
    except Exception as e:
        logger.error(f"Error in data_split: {str(e)}")
        raise

class TreePointCloudDataset(Dataset):
    def __init__(self, num_points, partition='train'):
        self.num_points = num_points
        self.partition = partition  # Store partition type
        
        try:
            if not os.path.exists('data_tree/stpctlc_data_split.h5'):
                h5_path = load_data(self.num_points)
                data_split(h5_path)

            with h5py.File('data_tree/stpctlc_data_split.h5', 'r') as f:
                self.data = f[partition]['point_clouds'][:]
                self.label = f[partition]['labels'][:]
                self.classes = [c.decode() if isinstance(c, bytes) else c for c in f['classes'][:]]
        except Exception as e:
            logger.error(f"Error initializing dataset: {str(e)}")
            raise

    def __getitem__(self, item):
        try:
            pointcloud = self.data[item][:self.num_points].copy()  # Make a copy to avoid modifying original data
            label = self.label[item]
            
            # Apply augmentation only for training data
            if self.partition == 'train':
                pointcloud = translate_pointcloud(pointcloud)
                np.random.shuffle(pointcloud)
            
            # Always normalize the point cloud (this is not augmentation)
            pointcloud = normalize_pc(pointcloud)
            
            return pointcloud, label
        except Exception as e:
            logger.error(f"Error getting item {item}: {str(e)}")
            raise

    def __len__(self):
        return len(self.data)

# Data augmentation functions
def translate_pointcloud(pointcloud):
    """Apply random translation augmentation."""
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def normalize_pc(points):
    """Normalize point cloud to unit sphere (this is not augmentation)."""
    centroid = np.mean(points, axis=0)
    points -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2, axis=-1)))
    points /= furthest_distance
    return points


def analyze_class_distribution(dataset, title="Class Distribution"):
    """Analyze and display class distribution in a dataset."""
    class_counts = {}
    total_samples = len(dataset)
    
    # Count samples per class
    for i in range(len(dataset)):
        label = dataset.label[i].item()
        class_name = dataset.classes[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # Display distribution
    logger.info(f"\n{title}:")
    logger.info("-" * 50)
    logger.info(f"{'Class':<20} {'Count':<10} {'Percentage':<10}")
    logger.info("-" * 50)
    
    for class_name, count in class_counts.items():
        percentage = (count / total_samples) * 100
        logger.info(f"{class_name:<20} {count:<10} {percentage:>6.2f}%")
    
    logger.info("-" * 50)
    logger.info(f"Total samples: {total_samples}\n")
    
    return class_counts

if __name__ == '__main__':
    try:
        train_dataset = TreePointCloudDataset(2048, 'train')
        test_dataset = TreePointCloudDataset(2048, 'test')
        print(train_dataset.classes)
        # Analyze class distributions
        train_dist = analyze_class_distribution(train_dataset, "Training Set Distribution")
        test_dist = analyze_class_distribution(test_dataset, "Test Set Distribution")
        
        # Compare ratios between train and test
        logger.info("Train/Test Ratio Comparison:")
        logger.info("-" * 50)
        logger.info(f"{'Class':<20} {'Train %':<10} {'Test %':<10} {'Ratio':<10}")
        logger.info("-" * 50)
        
        total_train = len(train_dataset)
        total_test = len(test_dataset)
        
        for class_name in train_dist.keys():
            train_pct = (train_dist[class_name] / total_train) * 100
            test_pct = (test_dist[class_name] / total_test) * 100
            ratio = train_pct / test_pct if test_pct > 0 else float('inf')
            
            logger.info(f"{class_name:<20} {train_pct:>6.2f}%    {test_pct:>6.2f}%    {ratio:>6.2f}")
        
        logger.info("-" * 50)
        
        # Verify data loading
        sample_data, sample_label = train_dataset[0]
        logger.info(f"\nSample point cloud shape: {sample_data.shape}")
        logger.info(f"Sample label shape: {sample_label.shape}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        sys.exit(1)
