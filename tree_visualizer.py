
"""
@Author: Said Ohamouddou
@File: models.py
@Time: 2025/07/13 13:18 PM
"""
"""
Paper: MS-DGCNN++: A Multi-Scale Fusion Dynamic Graph Neural Network with Biological Knowledge Integration for LiDAR Tree Species Classification
Said Ohamouddou, Abdellatif El Afia, Hanaa El Afia, Raddouane Chiheb
"""
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import logging

# Import your data_loader module
from data import TreePointCloudDataset, analyze_class_distribution

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def visualize_point_cloud(points, title="Point Cloud", point_size=2.0):
    """Visualize a single point cloud using Open3D."""
    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Color the points based on height (Z coordinate) for better visualization
    colors = plt.cm.viridis((points[:, 2] - points[:, 2].min()) / (points[:, 2].max() - points[:, 2].min()))[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Visualize
    print(f"Visualizing: {title}")
    print(f"Number of points: {len(points)}")
    o3d.visualization.draw_geometries([pcd], 
                                    window_name=title,
                                    point_show_normal=False)

def visualize_all_classes(dataset):
    """Visualize one sample from each of the 7 classes."""
    print("\n" + "="*60)
    print("VISUALIZING ALL 7 CLASSES")
    print("="*60)
    
    classes = dataset.classes
    print(f"Classes found: {classes}")
    
    for class_idx, class_name in enumerate(classes):
        print(f"\n--- Class {class_idx + 1}/7: {class_name} ---")
        
        # Find samples from this class
        class_indices = [i for i in range(len(dataset)) 
                        if dataset.label[i].item() == class_idx]
        
        if not class_indices:
            print(f"No samples found for class {class_name}")
            continue
            
        # Select a random sample from this class
        sample_idx = np.random.choice(class_indices)
        points, label = dataset[sample_idx]
        
        print(f"Sample index: {sample_idx}")
        print(f"Number of points: {len(points)}")
        print(f"Point cloud shape: {points.shape}")
        
        # Visualize the point cloud
        visualize_point_cloud(points, f"Class {class_idx + 1}: {class_name}")

def main():
    """Main function to visualize point clouds from all 7 classes."""
    print("Point Cloud Dataset Visualization - All 7 Classes")
    print("="*60)
    
    # Load dataset
    print("Loading dataset...")
    num_points = 2048
    train_dataset = TreePointCloudDataset(num_points, 'train')
    
    print(f"✓ Dataset loaded: {len(train_dataset)} samples")
    print(f"✓ Number of classes: {len(train_dataset.classes)}")
    print(f"✓ Classes: {train_dataset.classes}")
    
    # Analyze class distribution
    analyze_class_distribution(train_dataset, "Dataset Class Distribution")
    
    # Visualize one sample from each class
    visualize_all_classes(train_dataset)

if __name__ == "__main__":
    main()
