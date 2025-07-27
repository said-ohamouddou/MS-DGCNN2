"""
@Author: Said ohamouddou
@Contact: said_ohamouddou@um5.ac.ma
@Time: 2025/07/13 3:46 PM
"""


"""
Multi-Scale Dynamic Graph Convolution Network for Point Clouds Classification
Based on the paper: "Multi-Scale Dynamic Graph Convolution Network for Point Clouds Classification"
by Zhengli Zhai, Xin Zhang, and Luyao Yao
"""

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pointnet2_ops._ext as _ext


def knn(x, k):
    """
    K-nearest neighbors search using pairwise distance computation
    Args:
        x: input points, [B, dims, N] - batch, dimensions, number of points
        k: number of nearest neighbors
    Returns:
        idx: indices of k-nearest neighbors, [B, N, k]
    """
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    """
    Extract graph features using k-NN for EdgeConv operation
    Creates edge features by concatenating (neighbor_feat - center_feat, center_feat)
    Args:
        x: input points, [B, dims, N] - batch, feature dimensions, number of points
        k: number of nearest neighbors
        idx: precomputed k-NN indices (optional)
    Returns:
        feature: graph features, [B, 2*dims, N, k] - doubled feature dimension due to concatenation
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base
    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    # EdgeConv feature: concatenate (neighbor - center, center)
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature


def farthest_point_sample(xyz, npoint):
    """
    Farthest Point Sampling using the installed pointnet2_ops
    Args:
        xyz: input points, [B, N, 3] - batch, number of points, xyz coordinates
        npoint: number of points to sample
    Returns:
        idx: sampled point indices, [B, npoint]
    """
    return _ext.furthest_point_sampling(xyz, npoint)


def index_points(points, idx):
    """
    Index points using given indices
    Args:
        points: input points, [B, N, C] - batch, number of points, channels
        idx: sample indices, [B, S] - batch, number of samples
    Returns:
        new_points: indexed points, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

class MS_DGCNN(nn.Module):
    """
    Multi-Scale Dynamic Graph Convolution Network
    Architecture consists of three parallel branches with different k values:
    - Branch 1: k=20, single EdgeConv -> 64 channels
    - Branch 2: k=30, two EdgeConv layers -> 128 channels  
    - Branch 3: k=40, three EdgeConv layers with shortcut -> 256 channels
    Total concatenated features: 64 + 128 + 256 = 448 channels
    """
    def __init__(self, args, output_channels=7):
        super(MS_DGCNN, self).__init__()
        self.args = args
        self.k1 = getattr(args, 'k1', 20)  # Scale 1: k=20
        self.k2 = getattr(args, 'k2', 30)  # Scale 2: k=30  
        self.k3 = getattr(args, 'k3', 40)  # Scale 3: k=40
        self.fps_points = 512  # Extract 512 points using FPS
        
        # Branch 1: k=20 (single EdgeConv layer)
        # Input: [B, 6, N, k] -> Output: [B, 64, N] after max pooling
        self.bn1_1 = nn.BatchNorm2d(64)
        self.edge_conv1_1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),  # 3*2=6 input channels from EdgeConv
            self.bn1_1,
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        # Branch 2: k=30 (two EdgeConv layers)
        # First EdgeConv: [B, 6, N, k] -> [B, 64, N]
        # Second EdgeConv: [B, 128, N, k] -> [B, 128, N] (64*2=128 input after get_graph_feature)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.edge_conv2_1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            self.bn2_1,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.edge_conv2_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, bias=False),  # 64*2=128 input from get_graph_feature
            self.bn2_2,
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        # Branch 3: k=40 (three EdgeConv layers with shortcut connection)
        # First EdgeConv: [B, 6, N, k] -> [B, 64, N]
        # Second EdgeConv: [B, 128, N, k] -> [B, 128, N]
        # Third EdgeConv with shortcut: [B, 384, N, k] -> [B, 256, N] (192*2=384 input)
        self.bn3_1 = nn.BatchNorm2d(64)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.bn3_3 = nn.BatchNorm2d(256)
        self.edge_conv3_1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            self.bn3_1,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.edge_conv3_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, bias=False),  # 64*2=128 input
            self.bn3_2,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.edge_conv3_3 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=1, bias=False),  # (64+128)*2=384 input after shortcut
            self.bn3_3,
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        # Feature aggregation layers
        # Input: 64 + 128 + 256 = 448 channels from multi-scale concatenation
        # Architecture: 448 -> 512 -> 1024
        self.bn_agg1 = nn.BatchNorm1d(512)
        self.bn_agg2 = nn.BatchNorm1d(1024)
        
        self.conv_agg1 = nn.Sequential(
            nn.Conv1d(448, 512, kernel_size=1, bias=False),  # Fixed: 448 input channels
            self.bn_agg1,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv_agg2 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            self.bn_agg2,
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        # Classification head: 1024 -> 512 -> 256 -> output_channels
        self.bn_cls1 = nn.BatchNorm1d(512)
        self.bn_cls2 = nn.BatchNorm1d(256)

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.dp1 = nn.Dropout(p=0.5)
        
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.dp2 = nn.Dropout(p=0.5)
        
        # Final classification layer
        self.linear_final = nn.Linear(256, output_channels)

    def forward(self, x):
        """
        Forward pass through multi-scale DGCNN
        Args:
            x: input point cloud, [B, 3, N] - batch, xyz coordinates, number of points
        Returns:
            x: classification logits, [B, output_channels]
        """
        batch_size = x.size(0)
        
        # Step 1: Farthest Point Sampling to extract 512 representative points
        # This reduces computational complexity while preserving geometric structure
        xyz = x.transpose(2, 1).contiguous()  # [B, N, 3] - required format for FPS
        fps_idx = farthest_point_sample(xyz, self.fps_points)  # [B, 512]
        sampled_points = index_points(xyz, fps_idx).transpose(2, 1).contiguous()  # [B, 3, 512]
        
        # Step 2: Multi-scale EdgeConv branches with different neighborhood sizes
        
        # Branch 1: k=20 (small neighborhood - captures fine local details)
        x1 = get_graph_feature(sampled_points, k=self.k1)  # [B, 6, 512, 20]
        x1 = self.edge_conv1_1(x1)  # [B, 64, 512, 20]
        x1 = x1.max(dim=-1, keepdim=False)[0]  # [B, 64, 512] - max pooling over neighbors
        
        # Branch 2: k=30 (medium neighborhood - captures medium-scale patterns)
        x2 = get_graph_feature(sampled_points, k=self.k2)  # [B, 6, 512, 30]
        x2 = self.edge_conv2_1(x2)  # [B, 64, 512, 30]
        x2 = x2.max(dim=-1, keepdim=False)[0]  # [B, 64, 512]
        
        # Second EdgeConv in branch 2
        x2 = get_graph_feature(x2, k=self.k2)  # [B, 128, 512, 30] - feature dimension doubled
        x2 = self.edge_conv2_2(x2)  # [B, 128, 512, 30]
        x2 = x2.max(dim=-1, keepdim=False)[0]  # [B, 128, 512]
        
        # Branch 3: k=40 (large neighborhood - captures coarse global structure)
        x3 = get_graph_feature(sampled_points, k=self.k3)  # [B, 6, 512, 40]
        x3 = self.edge_conv3_1(x3)  # [B, 64, 512, 40]
        x3_1 = x3.max(dim=-1, keepdim=False)[0]  # [B, 64, 512] - store for shortcut
        
        # Second EdgeConv in branch 3
        x3 = get_graph_feature(x3_1, k=self.k3)  # [B, 128, 512, 40]
        x3 = self.edge_conv3_2(x3)  # [B, 128, 512, 40]
        x3_2 = x3.max(dim=-1, keepdim=False)[0]  # [B, 128, 512] - store for shortcut
        
        # Shortcut connection: concatenate features from first two EdgeConvs
        # This preserves multi-level features and helps with gradient flow
        x3_shortcut = torch.cat([x3_1, x3_2], dim=1)  # [B, 192, 512] - (64+128=192)
        
        # Third EdgeConv with shortcut input
        x3 = get_graph_feature(x3_shortcut, k=self.k3)  # [B, 384, 512, 40] - (192*2=384)
        x3 = self.edge_conv3_3(x3)  # [B, 256, 512, 40]
        x3_final = x3.max(dim=-1, keepdim=False)[0]  # [B, 256, 512]
        
        # Step 3: Multi-scale feature fusion
        # Concatenate features from all three branches to capture multi-scale information
        x = torch.cat((x1, x2, x3_final), dim=1)  # [B, 448, 512] - (64+128+256=448)
        
        # Step 4: Feature aggregation with MLP layers
        # Transform concatenated multi-scale features to higher-level representations
        x = self.conv_agg1(x)  # [B, 512, 512] 
        x = self.conv_agg2(x)  # [B, 1024, 512]
        
        # Step 5: Global feature extraction
        # Max pooling extracts the most salient features across all points
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)  # [B, 1024]
        
        # Step 6: Classification head with dropout for regularization
        x = F.leaky_relu(self.bn_cls1(self.linear1(x)), negative_slope=0.2)  # [B, 512]
        x = self.dp1(x)
        
        x = F.leaky_relu(self.bn_cls2(self.linear2(x)), negative_slope=0.2)  # [B, 256]
        x = self.dp2(x)
        
        # Final classification layer (no activation - raw logits for CrossEntropy loss)
        x = self.linear_final(x)  # [B, output_channels]
        
        return x

