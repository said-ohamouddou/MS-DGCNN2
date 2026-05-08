#!/usr/bin/env python3
"""
Standalone MS-DGCNN implementation for use without the registry pattern.
Adapted from msdgcnn.py for direct instantiation in experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    """K-nearest neighbors using pairwise distances."""
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=20, idx=None):
    """Extract graph features for EdgeConv."""
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)
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


def farthest_point_sample_cpu(xyz, npoint):
    """CPU implementation of farthest point sampling."""
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    
    return centroids


def farthest_point_sample(xyz, npoint):
    """Farthest point sampling with CUDA fallback."""
    try:
        import pointnet2_ops._ext as _ext
        return _ext.furthest_point_sampling(xyz, npoint)
    except ImportError:
        return farthest_point_sample_cpu(xyz, npoint)


def index_points(points, idx):
    """Index points based on indices."""
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


class MS_DGCNN_Standalone(nn.Module):
    """
    Multi-Scale Dynamic Graph CNN for Point Cloud Classification.
    
    Standalone version without registry dependencies.
    """
    
    def __init__(self, num_classes=7, k1=5, k2=20, k3=30, fps_points=512, dropout=0.5):
        super(MS_DGCNN_Standalone, self).__init__()
        
        self.num_classes = num_classes
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.fps_points = fps_points

        # Scale 1 (k1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.edge_conv1_1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            self.bn1_1,
            nn.LeakyReLU(negative_slope=0.2)
        )

        # Scale 2 (k2)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.edge_conv2_1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            self.bn2_1,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.edge_conv2_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            self.bn2_2,
            nn.LeakyReLU(negative_slope=0.2)
        )

        # Scale 3 (k3)
        self.bn3_1 = nn.BatchNorm2d(64)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.bn3_3 = nn.BatchNorm2d(256)
        self.edge_conv3_1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            self.bn3_1,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.edge_conv3_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            self.bn3_2,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.edge_conv3_3 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=1, bias=False),
            self.bn3_3,
            nn.LeakyReLU(negative_slope=0.2)
        )

        # Aggregation
        self.bn_agg1 = nn.BatchNorm1d(512)
        self.bn_agg2 = nn.BatchNorm1d(1024)
        self.conv_agg1 = nn.Sequential(
            nn.Conv1d(448, 512, kernel_size=1, bias=False),
            self.bn_agg1,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv_agg2 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            self.bn_agg2,
            nn.LeakyReLU(negative_slope=0.2)
        )

        # Classifier
        self.bn_cls1 = nn.BatchNorm1d(512)
        self.bn_cls2 = nn.BatchNorm1d(256)
        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear_final = nn.Linear(256, self.num_classes)

    def forward(self, x):
        # Handle input shape
        if x.dim() == 3 and x.shape[-1] == 3:
            x = x.permute(0, 2, 1)

        batch_size = x.size(0)

        # FPS sampling
        xyz = x.transpose(2, 1).contiguous()
        fps_idx = farthest_point_sample(xyz, self.fps_points)
        sampled_points = index_points(xyz, fps_idx).transpose(2, 1).contiguous()

        # Scale 1
        x1 = get_graph_feature(sampled_points, k=self.k1)
        x1 = self.edge_conv1_1(x1)
        x1 = x1.max(dim=-1, keepdim=False)[0]

        # Scale 2
        x2 = get_graph_feature(sampled_points, k=self.k2)
        x2 = self.edge_conv2_1(x2)
        x2 = x2.max(dim=-1, keepdim=False)[0]

        x2 = get_graph_feature(x2, k=self.k2)
        x2 = self.edge_conv2_2(x2)
        x2 = x2.max(dim=-1, keepdim=False)[0]

        # Scale 3
        x3 = get_graph_feature(sampled_points, k=self.k3)
        x3 = self.edge_conv3_1(x3)
        x3_1 = x3.max(dim=-1, keepdim=False)[0]

        x3 = get_graph_feature(x3_1, k=self.k3)
        x3 = self.edge_conv3_2(x3)
        x3_2 = x3.max(dim=-1, keepdim=False)[0]

        x3_shortcut = torch.cat([x3_1, x3_2], dim=1)

        x3 = get_graph_feature(x3_shortcut, k=self.k3)
        x3 = self.edge_conv3_3(x3)
        x3_final = x3.max(dim=-1, keepdim=False)[0]

        # Concatenate multi-scale features
        x = torch.cat((x1, x2, x3_final), dim=1)

        # Aggregation
        x = self.conv_agg1(x)
        x = self.conv_agg2(x)

        # Global pooling
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)

        # Classifier
        x = F.leaky_relu(self.bn_cls1(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)

        x = F.leaky_relu(self.bn_cls2(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)

        x = self.linear_final(x)

        return x


def create_ms_dgcnn(num_classes: int, k_scales: list = None, dropout: float = 0.5) -> nn.Module:
    """
    Factory function to create MS-DGCNN model.
    
    Args:
        num_classes: Number of output classes
        k_scales: List of k values [k1, k2, k3], default [5, 20, 30]
        dropout: Dropout rate
    
    Returns:
        MS_DGCNN_Standalone model instance
    """
    if k_scales is None:
        k_scales = [5, 20, 30]
    
    return MS_DGCNN_Standalone(
        num_classes=num_classes,
        k1=k_scales[0],
        k2=k_scales[1],
        k3=k_scales[2],
        dropout=dropout
    )
