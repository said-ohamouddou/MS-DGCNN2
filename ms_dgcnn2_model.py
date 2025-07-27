
"""
@Author: Said ohamouddou
@Contact: said_ohamouddou@um5.ac.ma
@Time: 2025/07/13 3:46 PM
"""
"""
Paper: MS-DGCNN++: A Multi-Scale Fusion Dynamic Graph Neural Network with Biological Knowledge Integration for LiDAR Tree Species Classification
Said Ohamouddou, Abdellatif El Afia, Hanaa El Afia, Raddouane Chiheb
"""


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn_multiscale(x, k_scales=[5, 20, 50]):
    """
    Compute k-nearest neighbors at multiple scales for hierarchical graph construction.
    
    Args:
        x: Input point cloud [B, C, N]
        k_scales: List of k values for different scales (local, branch, canopy)
    
    Returns:
        List of indices for each scale
    """
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    
    idx_list = []
    for k in k_scales:
        idx = pairwise_distance.topk(k=k, dim=-1)[1]
        idx_list.append(idx)
    
    return idx_list


def get_hierarchical_graph_feature(x, k_scales=[5, 20, 50], idx_list=None):
    """
    Extract hierarchical graph features at multiple scales.
    
    Args:
        x: Input features [B, C, N]
        k_scales: List of k values for different scales
        idx_list: Pre-computed neighbor indices (optional)
    
    Returns:
        Concatenated features from all scales
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    
    if idx_list is None:
        idx_list = knn_multiscale(x, k_scales)
    
    device = x.device
    _, num_dims, _ = x.size()
    x_transposed = x.transpose(2, 1).contiguous()
    
    hierarchical_features = []
    
    for scale_idx, (k, idx) in enumerate(zip(k_scales, idx_list)):
        # Prepare indices for batched indexing
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        idx_batch = idx + idx_base
        idx_batch = idx_batch.view(-1)
        
        # Extract neighbor features
        neighbor_features = x_transposed.view(batch_size * num_points, -1)[idx_batch, :]
        neighbor_features = neighbor_features.view(batch_size, num_points, k, num_dims)
        
        # Repeat center point features
        center_features = x_transposed.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
        
        # Compute relative features (neighbor - center) and absolute features (center)
        relative_features = neighbor_features - center_features
        
        # For different scales, we can apply different feature transformations
        if scale_idx == 0:  # Local scale - focus on fine details
            scale_features = torch.cat((relative_features, center_features), dim=3)
        elif scale_idx == 1:  # Branch scale - focus on structural patterns
            # Add angular information by normalizing relative vectors
            relative_norm = torch.norm(relative_features, dim=3, keepdim=True) + 1e-8
            normalized_relative = relative_features / relative_norm
            scale_features = torch.cat((relative_features, normalized_relative, center_features), dim=3)
        else:  # Canopy scale - focus on global shape
            # Add distance information
            distances = torch.norm(relative_features, dim=3, keepdim=True)
            scale_features = torch.cat((relative_features, center_features, distances), dim=3)
        
        # Permute to [B, C, N, K] format
        scale_features = scale_features.permute(0, 3, 1, 2).contiguous()
        hierarchical_features.append(scale_features)
    
    return hierarchical_features


class MS_DGCNN2(nn.Module):
    def __init__(self, args, output_channels=7):
        super(MS_DGCNN2, self).__init__()
        self.args = args
        self.k_scales = getattr(args, 'k_scales', [5, 20, 30])  # Local, Branch, Canopy scales
        print("we use ", self.k_scales)
        # Batch normalization layers
        self.bn1_local = nn.BatchNorm2d(64)
        self.bn1_branch = nn.BatchNorm2d(64)
        self.bn1_canopy = nn.BatchNorm2d(64)
        
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        
        # Multi-scale convolution layers for first layer
        # Local scale (fine details like leaves, bark texture)
        self.conv1_local = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),  # Standard relative + absolute features
            self.bn1_local,
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        # Branch scale (structural patterns)
        self.conv1_branch = nn.Sequential(
            nn.Conv2d(9, 64, kernel_size=1, bias=False),  # Relative + normalized + absolute features
            self.bn1_branch,
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        # Canopy scale (global shape)
        self.conv1_canopy = nn.Sequential(
            nn.Conv2d(7, 64, kernel_size=1, bias=False),  # Relative + absolute + distance features
            self.bn1_canopy,
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        # Feature fusion layer
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=1, bias=False),  # 64 * 3 scales = 192
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        # Subsequent layers remain similar but use single scale (branch scale for balance)
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        # Classification head
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        
        # First layer: Hierarchical multi-scale feature extraction
        hierarchical_features = get_hierarchical_graph_feature(x, k_scales=self.k_scales)
        
        # Process each scale separately
        x_local = self.conv1_local(hierarchical_features[0])    # Local features (fine details)
        x_branch = self.conv1_branch(hierarchical_features[1])  # Branch features (structure)
        x_canopy = self.conv1_canopy(hierarchical_features[2])  # Canopy features (global shape)
        
        # Max pooling for each scale
        x_local = x_local.max(dim=-1, keepdim=False)[0]
        x_branch = x_branch.max(dim=-1, keepdim=False)[0]
        x_canopy = x_canopy.max(dim=-1, keepdim=False)[0]
        
        # Concatenate multi-scale features
        x_multi = torch.cat((x_local, x_branch, x_canopy), dim=1)
        
        # Fuse multi-scale features
        # Expand dims for conv2d compatibility
        x_multi_expanded = x_multi.unsqueeze(-1)
        x_fused = self.fusion_conv(x_multi_expanded).squeeze(-1)
        x1 = x_fused
        
        # Continue with standard DGCNN layers using branch-scale connections
        k_branch = self.k_scales[1]  # Use branch scale for subsequent layers
        
        x = get_graph_feature(x1, k=k_branch)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=k_branch)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=k_branch)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        # Concatenate features from all layers
        x = torch.cat((x1, x2, x3, x4), dim=1)

        # Global feature extraction
        x = self.conv5(x)
        x1_global = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2_global = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1_global, x2_global), 1)

        # Classification layers
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x


def get_graph_feature(x, k=20, idx=None):
    """Original graph feature function for backward compatibility"""
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        inner = -2*torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        idx = pairwise_distance.topk(k=k, dim=-1)[1]
    
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature
