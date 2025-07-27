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
import argparse
import time
import numpy as np
import wandb
import sklearn.metrics as metrics
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from data import TreePointCloudDataset
from collections import Counter
from ms_dgcnn2_model import MS_DGCNN2
from ms_dgcnn_model import MS_DGCNN
from dgcnn_model import DGCNN
import math

def count_trainable_parameters(model):
    """
    Counts the number of trainable parameters in a PyTorch model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_class_weights(train_loader, device):
    """
    Compute class weights based on class distribution in the training dataset.
    Handles standard PyTorch DataLoader objects.
    
    Args:
        train_loader: Standard PyTorch DataLoader
        device: torch device to place the resulting weights tensor
        
    Returns:
        torch.Tensor: Class weights tensor on specified device
    """
    from collections import Counter
    import torch
    
    labels = []
    
    for batch_data, batch_labels in train_loader:
        try:
            # Handle both single label and batch of labels
            if batch_labels.dim() > 1:
                batch_labels = batch_labels.squeeze()
            
            # Convert to list and add to labels
            labels.extend(batch_labels.cpu().numpy().tolist())
            
        except (IndexError, ValueError, AttributeError) as e:
            print(f"Warning: Skipping a batch due to error: {e}")
    
    class_counts = Counter(labels)
    num_classes = len(class_counts)
    mean_samples = sum(class_counts.values()) / num_classes
    
    weights = []
    max_count = max(class_counts.values())
    
    for i in range(num_classes):
        count = class_counts.get(i, 1)
        if count < mean_samples:
            # For minority classes, use mean sample count instead
            weight = max_count / mean_samples
        else:
            weight = max_count / count
        weights.append(weight)
    
    return torch.tensor(weights, dtype=torch.float32).to(device)
    
# Helper function for tree species-specific data augmentation
def augment_tree_point_cloud(point_cloud, jitter_sigma=0.01, rotate=True, scale_min=0.8, scale_max=1.2):
    """
    Apply domain-specific augmentations for tree point clouds.
    - Jittering: Simulates natural variation in foliage
    - Rotation: Random rotation around vertical axis (trees grow upward)
    - Scaling: Simulates different tree sizes within species
    - Partial deletion: Simulates occlusion/missing data
    """
    batch_size, num_dims, num_points = point_cloud.size()
    device = point_cloud.device
    
    # Clone point cloud
    result = point_cloud.clone()
    
    # Add jitter (more to upper parts of tree)
    if jitter_sigma > 0:
        # Identify height (assuming z-axis is up)
        height = result[:, 2, :].clone()
        height_normalized = (height - height.min(dim=1, keepdim=True)[0]) / (height.max(dim=1, keepdim=True)[0] - height.min(dim=1, keepdim=True)[0] + 1e-6)
        
        # Apply more jitter to upper parts (leaves are more variable than trunk)
        jitter = torch.randn_like(result) * jitter_sigma
        jitter = jitter * height_normalized.unsqueeze(1).expand_as(jitter)
        result += jitter
    
    # Random rotation around vertical (z) axis
    if rotate:
        for i in range(batch_size):
            theta = torch.rand(1, device=device) * 2 * math.pi
            rot_matrix = torch.tensor([
                [torch.cos(theta), -torch.sin(theta), 0],
                [torch.sin(theta), torch.cos(theta), 0],
                [0, 0, 1]
            ], device=device)
            
            # Apply rotation
            result[i, :, :] = torch.matmul(rot_matrix, result[i, :, :])
    
    # Random scaling (uniform in all dimensions)
    scales = torch.rand(batch_size, 1, 1, device=device) * (scale_max - scale_min) + scale_min
    result = result * scales.expand_as(result)
    
    # Random partial deletion (simulates occlusion/missing data)
    if torch.rand(1).item() > 0.5:
        mask = torch.rand(batch_size, 1, num_points, device=device) > 0.1  # Keep 90% of points
        # Ensure we don't delete too many points
        min_points_to_keep = int(0.8 * num_points)
        for i in range(batch_size):
            if mask[i].sum() < min_points_to_keep:
                # If too many points were deleted, randomly select points to keep
                perm = torch.randperm(num_points, device=device)
                mask[i, 0, perm[:min_points_to_keep]] = True
        
        # Set deleted points to zero (or another marker value)
        deleted_mask = ~mask.expand_as(result)
        result[deleted_mask] = 0
    
    return result

def train(args):
    # Start tracking total training time
    total_training_start_time = time.time()

    # Initialize Weights & Biases
    run = wandb.init(
        project='STPCTLC',
        name=args.exp_name,
        reinit=True
    )
  
    # Create datasets with transforms
    train_dataset = TreePointCloudDataset(
        num_points=args.num_points, 
        partition='train'
    )
    test_dataset = TreePointCloudDataset(
        num_points=args.num_points,
        partition='test'
    )
    
    # Create data loaders using standard DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=6
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=6
    )

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of testing batches: {len(test_loader)}")
    num_classes = len(train_dataset.classes)
    print(f'Number of classes: {num_classes}')
    device = torch.device("cuda" if args.cuda else "cpu")
    
    if args.model == 'MS_DGCNN2':
        model = MS_DGCNN2(args, num_classes)
    elif args.model == 'MS_DGCNN':
        model = MS_DGCNN(args, num_classes)
    elif args.model == 'DGCNN':
        model = DGCNN(args, num_classes)
    else:
        raise ValueError(f"Unknown model type: {args.model}. Supported models: 'MS_DGCNN2', 'MS_DGCNN', 'DGCNN'")
    model = model.to(device)
    
    print(model)
    num_trainable_params = count_trainable_parameters(model)
    print(f"The model has {num_trainable_params:,} trainable parameters.")
    wandb.run.summary["trainable_parameters"] = num_trainable_params
    
    # Initialize optimizer
    if args.use_sgd:
        print("Use SGD")
        optimizer = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-3)
    
    # Compute class weights
    class_weights = compute_class_weights(train_loader, device)
    scaled_weights = class_weights / class_weights.max()
    print(scaled_weights)
    criterion = torch.nn.CrossEntropyLoss(weight=scaled_weights)
    
    # Initialize best metrics
    best_test_acc = 0.0
    best_balanced_acc = 0.0
    best_kappa = 0.0
    # Initialize metrics at best accuracy
    best_acc_balanced_acc = 0.0
    best_acc_kappa = 0.0
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()

        model.train()
        train_loss = 0.0
        train_total = 0
        train_pred = []
        train_true = []

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)
            
            optimizer.zero_grad()
            
            data = data.permute(0, 2, 1)
            
            data = augment_tree_point_cloud(data)
            # Forward pass
          
            logits = model(data)
            
            # Calculate loss
            loss = criterion(logits, target.squeeze(1))
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Get predictions
            preds = logits.max(dim=1)[1]
            
            train_loss += loss.item() * batch_size
            train_total += batch_size
            
            train_true.append(target.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        scheduler.step()
        
        # Calculate training metrics
        avg_train_loss = train_loss / train_total
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_acc = metrics.accuracy_score(train_true, train_pred)
        train_avg_per_class_acc = metrics.balanced_accuracy_score(train_true, train_pred)
        train_kappa = metrics.cohen_kappa_score(train_true, train_pred)
        
        print(f"Train Epoch: {epoch} | "
              f"Loss: {avg_train_loss:.6f} | "
              f"Accuracy: {train_acc:.6f} | "
              f"Balanced Accuracy: {train_avg_per_class_acc:.6f} | "
              f"Kappa: {train_kappa:.6f}")
       
        # Testing Phase
        model.eval()
        test_loss = 0.0
        test_total = 0
        test_pred = []
        test_true = []

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.float().to(device), target.to(device)
                batch_size = data.size(0)
                
                data = data.permute(0, 2, 1)
                # Forward pass
                logits = model(data)
                
                # Calculate loss
                loss = criterion(logits, target.squeeze(1))
                
                # Get predictions
                preds = logits.max(dim=1)[1]
                
                test_loss += loss.item() * batch_size
                test_total += batch_size
                
                test_true.append(target.cpu().numpy())
                test_pred.append(preds.detach().cpu().numpy())
        
        # Calculate testing metrics
        avg_test_loss = test_loss / test_total
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        test_avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        test_kappa = metrics.cohen_kappa_score(test_true, test_pred)
        
        epoch_time = time.time() - epoch_start_time
        current_total_time = time.time() - total_training_start_time
    
        # Log metrics to WandB
        wandb.log({
            "train_loss": avg_train_loss,
            "train_acc": train_acc,
            "test_loss": avg_test_loss,
            "test_acc": test_acc,
            "test_avg_per_class_acc": test_avg_per_class_acc,
            "test_kappa": test_kappa,
            "epoch_time": epoch_time,
            "total_training_time": current_total_time,
            "epoch": epoch
        })
        
        print(f"Test Epoch: {epoch} | "
              f"Loss: {avg_test_loss:.6f} | "
              f"Accuracy: {test_acc:.6f} | "
              f"Balanced Accuracy: {test_avg_per_class_acc:.6f} | "
              f"Kappa: {test_kappa:.6f} | "
              f"Epoch Time: {epoch_time:.2f}s | "
              f"Total Time: {current_total_time/60:.2f}m")
        
        # Create save directory
        save_path = os.path.join('checkpoints', args.exp_name, 'models')
        os.makedirs(save_path, exist_ok=True)
        
        # Save best model based on accuracy
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            # Track balanced accuracy and kappa at best accuracy epoch
            best_acc_balanced_acc = test_avg_per_class_acc
            best_acc_kappa = test_kappa
            
            torch.save(model.state_dict(), os.path.join(save_path, 'best_acc.t7'))
            report = metrics.classification_report(test_true, test_pred, target_names=test_dataset.classes, output_dict=True)
            # Extract precision for each class and store it for later use
            best_precision_values = {f"best_precision_{label}": metrics['precision'] * 100 
                           for label, metrics in report.items() 
                           if label in test_dataset.classes}
    
            with open(os.path.join(save_path, 'best_acc_epoch.txt'), 'w') as f:
                f.write(f'Best accuracy model saved at epoch {epoch}\n')
                f.write(f'Accuracy: {best_test_acc:.6f}\n')
                f.write(f'Balanced Accuracy: {best_acc_balanced_acc:.6f}\n')
                f.write(f'Kappa: {best_acc_kappa:.6f}')
            print(f"Best accuracy model saved with accuracy: {best_test_acc:.6f}")
            print(f"At best accuracy, balanced accuracy: {best_acc_balanced_acc:.6f}, kappa: {best_acc_kappa:.6f}")
            
        # Save best model based on balanced accuracy
        if test_avg_per_class_acc > best_balanced_acc:
            best_balanced_acc = test_avg_per_class_acc
            torch.save(model.state_dict(), os.path.join(save_path, 'best_balanced_acc.t7'))
            with open(os.path.join(save_path, 'best_balanced_acc_epoch.txt'), 'w') as f:
                f.write(f'Best balanced accuracy model saved at epoch {epoch}\n')
                f.write(f'Accuracy: {test_acc:.6f}\n')
                f.write(f'Balanced Accuracy: {best_balanced_acc:.6f}\n')
                f.write(f'Kappa: {test_kappa:.6f}')
            print(f"Best balanced accuracy model saved with balanced accuracy: {best_balanced_acc:.6f}")
        
        # Save best model based on kappa
        if test_kappa > best_kappa:
            best_kappa = test_kappa
            torch.save(model.state_dict(), os.path.join(save_path, 'best_kappa.t7'))
            with open(os.path.join(save_path, 'best_kappa_epoch.txt'), 'w') as f:
                f.write(f'Best kappa model saved at epoch {epoch}\n')
                f.write(f'Accuracy: {test_acc:.6f}\n')
                f.write(f'Balanced Accuracy: {test_avg_per_class_acc:.6f}\n')
                f.write(f'Kappa: {best_kappa:.6f}')
            print(f"Best kappa model saved with kappa: {best_kappa:.6f}")
    
    # Calculate and log total training time
    total_training_time = time.time() - total_training_start_time
    
    # Save final results
    print(f'Best test accuracy: {best_test_acc:.4f}')
    print(f'Best balanced accuracy: {best_balanced_acc:.4f}')
    print(f'Best kappa: {best_kappa:.4f}')
    print(f'At best accuracy epoch - balanced acc: {best_acc_balanced_acc:.4f}, kappa: {best_acc_kappa:.4f}')
    print(f'Total training time: {total_training_time :.4f} s')
    
    # Add total training time to wandb summary
    wandb.run.summary["best_test_acc"] = best_test_acc*100
    wandb.run.summary["best_balanced_acc"] = best_balanced_acc*100
    wandb.run.summary["best_kappa"] = best_kappa*100
    # Add metrics at best accuracy epoch
    wandb.run.summary["balanced_acc_at_best_acc"] = best_acc_balanced_acc*100
    wandb.run.summary["kappa_at_best_acc"] = best_acc_kappa*100
    wandb.run.summary["total_training_time"] = total_training_time
    wandb.run.summary["time_per_epoch"] = total_training_time / args.epochs
    # Log precision values associated with the best model in overall accuracy
    for class_name, value in best_precision_values.items():
        wandb.run.summary[class_name] = value
    
    # Save final metrics summary
    with open(os.path.join(save_path, 'final_results.txt'), 'w') as f:
        f.write(f'Best test accuracy: {best_test_acc:.4f}\n')
        f.write(f'Best balanced accuracy: {best_balanced_acc:.4f}\n')
        f.write(f'Best kappa: {best_kappa:.4f}\n')
        f.write(f'At best accuracy epoch - balanced acc: {best_acc_balanced_acc:.4f}, kappa: {best_acc_kappa:.4f}\n')
        f.write(f'Total training time: {total_training_time:.4f} s\n')
        
    # Save final model
    final_model_path = os.path.join(save_path, 'last.t7')
    torch.save(model.state_dict(), final_model_path)
    print(f'Final model saved to {final_model_path}')
    run.finish()


if __name__ == "__main__":
    # Training settings
    print("ok")
    parser = argparse.ArgumentParser(description='Baseline')
    parser.add_argument('--exp_name', type=str, default='exp', 
                        help='Name of the experiment')
    parser.add_argument('--seed', type=int, default=1, 
                        help='Random seed for reproducibility')
    # Training hyperparameters
    
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Training batch size')
    parser.add_argument('--test_batch_size', type=int, default=16, 
                        help='Testing batch size')
    parser.add_argument('--epochs', type=int, default=300, 
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
    parser.add_argument('--model', type=str, default='MS_DGCNN2', choices=['MS_DGCNN2', 'MS_DGCNN', 'DGCNN'], 
                        help='Model to train')
    
    args = parser.parse_args()

    
    print(str(args))
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    if args.cuda:
        print(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        print('Using CPU')


    train(args)


