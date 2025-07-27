"""
@Author: Said Ohamouddou
@File: models.py
@Time: 2025/07/13 13:18 PM
"""
"""
Paper: MS-DGCNN++: A Multi-Scale Fusion Dynamic Graph Neural Network with Biological Knowledge Integration for LiDAR Tree Species Classification
Said Ohamouddou, Abdellatif El Afia, Hanaa El Afia, Raddouane Chiheb
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn import metrics
import pandas as pd
import os
import argparse

from ms_dgcnn2_model import MS_DGCNN2

from data import TreePointCloudDataset

def test(args):
    test_loader = DataLoader(TreePointCloudDataset(partition='test', num_points=args.num_points), batch_size=args.test_batch_size)
    device = torch.device("cuda" if not args.no_cuda else "cpu")
    model_path = args.model_path
    
    # Try to load models
    model = MS_DGCNN2(args).to(device)
    #model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path))
    model = model.eval()
    
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    
    # Get test dataset for class names
    test_dataset = TreePointCloudDataset(partition='test', num_points=args.num_points)
    
    for data, label in test_loader:
        data, label = data.to(device).float(), label.to(device).squeeze()  # Convert to float32
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        logits = model(data)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    
    # Calculate metrics
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    kappa = metrics.cohen_kappa_score(test_true, test_pred)
    
    # Get classification report
    report = metrics.classification_report(test_true, test_pred, target_names=test_dataset.classes, output_dict=True)
    
    # Extract precision for each class
    best_precision_values = {f"best_precision_{label}": metrics_dict['precision'] * 100 
                           for label, metrics_dict in report.items() 
                           if label in test_dataset.classes}
    
    # Create metrics dictionary
    metrics_dict = {
        'overall_test_acc': test_acc * 100,  # Convert to percentage
        'balanced_acc': avg_per_class_acc * 100,  # Convert to percentage
        'kappa': kappa
    }
    
    # Add class precisions
    metrics_dict.update(best_precision_values)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame([metrics_dict])
    
    # Create directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Save to CSV
    csv_path = 'results/test_metrics.csv'
    df.to_csv(csv_path, index=False)
    
    # Print results
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f, kappa: %.6f' % (test_acc, avg_per_class_acc, kappa)
    print(outstr)
    print(f'Metrics saved to: {csv_path}')
    
    # Print class-wise precisions
    for class_name, precision in best_precision_values.items():
        print(f'{class_name}: {precision:.2f}%')
    
    return metrics_dict

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Baseline')
    parser.add_argument('--exp_name', type=str, default='exp', 
                        help='Name of the experiment')
    parser.add_argument('--seed', type=int, default=1, 
                        help='Random seed for reproducibility')
    parser.add_argument('--model_path', type=str, default='./checkpoints/MS-DGCNN2_5_20_30/models/best_acc.t7', 
                        help='Path to pretrained model')
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Training batch size')
    parser.add_argument('--test_batch_size', type=int, default=16, 
                        help='Testing batch size')
    parser.add_argument('--epochs', type=int, default=250, 
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='Initial learning rate')
    parser.add_argument('--use_sgd', action='store_true', 
                        help='Use SGD optimizer instead of Adam')
    parser.add_argument('--momentum', type=float, default=0.9, 
                        help='SGD momentum (only used with --use_sgd)')
    parser.add_argument('--dropout', type=float, default=0.5, 
                        help='Dropout rate for MLP')
                        
    # Hardware settings
    parser.add_argument('--no_cuda', action='store_true', 
                        help='Disable CUDA training')
                        
    # Model architecture
    parser.add_argument('--num_points', type=int, default=1024, 
                        help='Number of points in point cloud')
    parser.add_argument('--emb_dims', type=int, default=1024, 
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=8, 
                        help='Number of nearest neighbors to use')
    parser.add_argument('--aggr', type=str, default='max', choices=['max', 'mean', 'sum'], 
                        help='Aggregation method (max, mean, sum)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if not args.no_cuda:
        torch.cuda.manual_seed(args.seed)
    

    print("Running evaluation...")
    metrics = test(args)
    print("Evaluation completed!")

