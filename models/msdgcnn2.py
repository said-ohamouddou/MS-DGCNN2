"""MS-DGCNN++ model variants for ablations (multi-scale, normalized edge features, fusion types, k-scale configs)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AblationConfig:
    """Configuration for ablation experiments."""
    num_classes: int = 7
    emb_dims: int = 1024
    dropout: float = 0.5
    k_scales: List[int] = None  # [k_local, k_branch, k_global]
    use_multiscale: bool = True  # False = single-scale (uses k_scales[0] only)
    use_normalized_features: bool = True  # Normalized relative features in branch
    fusion_type: str = 'concat_conv'  # 'concat_conv', 'concat_only', 'add', 'attention', 'gated', 'bilinear', 'se_fusion'
    branch_feature_mode: str = 'full'  # 'full' or 'normalized_only'
    local_use_normalized: bool = False  # Whether to use normalized features in local scale (Scale 1)
    
    def __post_init__(self):
        if self.k_scales is None:
            self.k_scales = [5, 20, 30]


def knn(x, k):
    """Compute k-nearest neighbors."""
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=20, idx=None):
    """Standard DGCNN graph feature extraction."""
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    
    if idx is None:
        idx = knn(x, k)
    
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx_base + idx
    idx = idx.view(-1)
    
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


def get_graph_feature_norm(x, k=20, idx=None):
    """DGCNN graph feature extraction with normalized relative features.
    
    Returns 9 channels: relative + normalized_relative + center
    Used for single-scale DGCNN+Norm variants.
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    
    if idx is None:
        idx = knn(x, k)
    
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx_base + idx
    idx = idx.view(-1)
    
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    center = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    relative = feature - center
    relative_norm = torch.norm(relative, dim=3, keepdim=True) + 1e-8
    normalized_relative = relative / relative_norm
    
    # 9 channels: relative (3) + normalized_relative (3) + center (3)
    out = torch.cat((relative, normalized_relative, center), dim=3).permute(0, 3, 1, 2).contiguous()
    return out


def get_hierarchical_graph_feature(x, k_scales, use_normalized=True, branch_feature_mode='full', local_use_normalized=False):
    """Extract hierarchical graph features at multiple scales.
    
    Args:
        x: Input point cloud [B, 3, N]
        k_scales: List of k values for each scale
        use_normalized: Whether to include normalized relative features in branch
        branch_feature_mode: Feature mode for branch scale:
            - 'full': xyz + relative + normalized (if use_normalized) [default]
            - 'normalized_only': ONLY normalized relative features (no xyz, no relative)
        local_use_normalized: Whether to include normalized features in local scale (Scale 1)
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    
    device = x.device
    _, num_dims, _ = x.size()
    x_transposed = x.transpose(2, 1).contiguous()
    
    # Compute pairwise distances once
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    
    hierarchical_features = []
    
    for scale_idx, k in enumerate(k_scales):
        idx = pairwise_distance.topk(k=k, dim=-1)[1]
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        idx_batch = (idx + idx_base).view(-1)
        
        neighbor_features = x_transposed.view(batch_size * num_points, -1)[idx_batch, :]
        neighbor_features = neighbor_features.view(batch_size, num_points, k, num_dims)
        center_features = x_transposed.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
        relative_features = neighbor_features - center_features
        
        if scale_idx == 0:
            # Local scale: optionally include normalized features
            if local_use_normalized:
                # With normalization: relative + normalized + center (9 channels)
                relative_norm = torch.norm(relative_features, dim=3, keepdim=True) + 1e-8
                normalized_relative = relative_features / relative_norm
                scale_features = torch.cat((relative_features, normalized_relative, center_features), dim=3)
            else:
                # Standard edge features: relative + center (6 channels)
                scale_features = torch.cat((relative_features, center_features), dim=3)
        else:
            # Branch scale: different feature modes
            if branch_feature_mode == 'normalized_only':
                # ONLY normalized relative features (3 channels)
                relative_norm = torch.norm(relative_features, dim=3, keepdim=True) + 1e-8
                normalized_relative = relative_features / relative_norm
                scale_features = normalized_relative
            elif use_normalized:
                # Full features with normalization: relative + normalized + center (9 channels)
                relative_norm = torch.norm(relative_features, dim=3, keepdim=True) + 1e-8
                normalized_relative = relative_features / relative_norm
                scale_features = torch.cat((relative_features, normalized_relative, center_features), dim=3)
            else:
                # Without normalization: relative + center (6 channels)
                scale_features = torch.cat((relative_features, center_features), dim=3)
        
        scale_features = scale_features.permute(0, 3, 1, 2).contiguous()
        hierarchical_features.append(scale_features)
    
    return hierarchical_features


class AttentionFusion(nn.Module):
    """Cross-attention fusion between local and branch features."""
    def __init__(self, in_channels=64, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (in_channels // num_heads) ** -0.5
        self.qkv = nn.Conv1d(in_channels * 2, in_channels * 3, 1, bias=False)
        self.proj = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x_local, x_branch):
        B, C, N = x_local.shape
        x = torch.cat([x_local, x_branch], dim=1)  # [B, 2C, N]
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, C // self.num_heads, N)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        out = (v @ attn.transpose(-2, -1)).reshape(B, C, N)
        return self.proj(out)


class GatedFusion(nn.Module):
    """Gated fusion with learnable gate weights."""
    def __init__(self, in_channels=64):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv1d(in_channels * 2, in_channels, 1, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.Sigmoid()
        )
        self.proj = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x_local, x_branch):
        gate = self.gate(torch.cat([x_local, x_branch], dim=1))
        fused = gate * x_local + (1 - gate) * x_branch
        return self.proj(fused)


class BilinearFusion(nn.Module):
    """Bilinear pooling fusion for richer feature interaction."""
    def __init__(self, in_channels=64, out_channels=64):
        super().__init__()
        self.bilinear = nn.Bilinear(in_channels, in_channels, out_channels, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.LeakyReLU(0.2)
    
    def forward(self, x_local, x_branch):
        B, C, N = x_local.shape
        x_local_t = x_local.permute(0, 2, 1)  # [B, N, C]
        x_branch_t = x_branch.permute(0, 2, 1)  # [B, N, C]
        out = self.bilinear(x_local_t, x_branch_t)  # [B, N, out_channels]
        out = out.permute(0, 2, 1)  # [B, out_channels, N]
        return self.act(self.bn(out))


class SEFusion(nn.Module):
    """Squeeze-and-Excitation fusion with channel attention."""
    def __init__(self, in_channels=128, reduction=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excite = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
        self.proj = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x_local, x_branch):
        x = torch.cat([x_local, x_branch], dim=1)  # [B, 2C, N]
        B, C, N = x.shape
        se = self.squeeze(x).view(B, C)
        se = self.excite(se).view(B, C, 1)
        x = x * se
        return self.proj(x)


class MS_DGCNN2_Ablation(nn.Module):
    """
    MS-DGCNN2 with configurable ablation options.
    
    Ablation flags:
    - use_multiscale: True=multi-scale, False=single-scale
    - use_normalized_features: True=with normalization, False=without
    - fusion_type: 'concat_conv', 'concat_only', 'add', 'attention', 'gated', 'bilinear', 'se_fusion'
    """
    
    def __init__(self, config: AblationConfig):
        super().__init__()
        self.config = config
        self.num_classes = config.num_classes
        
        k_scales = config.k_scales
        self.k_local = k_scales[0]
        self.k_branch = k_scales[1] if len(k_scales) > 1 else k_scales[0]
        self.k_global = k_scales[2] if len(k_scales) > 2 else self.k_branch
        
        self.use_multiscale = config.use_multiscale
        self.use_normalized_features = config.use_normalized_features
        self.fusion_type = config.fusion_type
        self.branch_feature_mode = config.branch_feature_mode
        self.local_use_normalized = config.local_use_normalized
        
        # Input channels depend on feature mode
        if config.branch_feature_mode == 'normalized_only':
            branch_in_channels = 3  # Only normalized relative features
        elif config.use_normalized_features:
            branch_in_channels = 9  # relative + normalized + center
        else:
            branch_in_channels = 6  # relative + center
        
        # Local input channels depend on local_use_normalized
        local_in_channels = 9 if config.local_use_normalized else 6
        
        # Local branch (always used)
        self.conv1_local = nn.Sequential(
            nn.Conv2d(local_in_channels, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        # Branch (only used in multi-scale mode)
        if self.use_multiscale:
            self.conv1_branch = nn.Sequential(
                nn.Conv2d(branch_in_channels, 64, kernel_size=1, bias=False),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(negative_slope=0.2)
            )
        
        # Fusion layer (depends on fusion_type)
        if self.use_multiscale:
            if self.fusion_type == 'concat_conv':
                self.fusion_conv = nn.Sequential(
                    nn.Conv2d(128, 64, kernel_size=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU(negative_slope=0.2)
                )
            elif self.fusion_type == 'concat_only':
                # True concat-only: no projection, pass 128 channels to conv2
                pass  # No fusion layer needed
            elif self.fusion_type == 'attention':
                self.fusion_attention = AttentionFusion(in_channels=64, num_heads=4)
            elif self.fusion_type == 'gated':
                self.fusion_gated = GatedFusion(in_channels=64)
            elif self.fusion_type == 'bilinear':
                self.fusion_bilinear = BilinearFusion(in_channels=64, out_channels=64)
            elif self.fusion_type == 'se_fusion':
                self.fusion_se = SEFusion(in_channels=128, reduction=4)
            # 'add' fusion doesn't need extra layers
        
        # EdgeConv layers - input channels depend on fusion type
        # concat_only passes 128 channels (no projection), others pass 64
        conv2_in_channels = 128 if (self.use_multiscale and self.fusion_type == 'concat_only') else 64
        self.conv2 = nn.Sequential(
            nn.Conv2d(conv2_in_channels * 2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        # Global feature aggregation
        # x1 channels: 128 for concat_only, 64 for others
        # Total: x1 + x2(64) + x3(128) + x4(256) = 576 or 512
        conv5_in_channels = 576 if (self.use_multiscale and self.fusion_type == 'concat_only') else 512
        self.conv5 = nn.Sequential(
            nn.Conv1d(conv5_in_channels, config.emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(config.emb_dims),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(config.emb_dims * 2, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=config.dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=config.dropout),
            nn.Linear(256, self.num_classes)
        )
    
    def forward(self, x):
        # Handle input shape
        if x.dim() == 3 and x.shape[-1] == 3:
            x = x.permute(0, 2, 1)
        elif x.dim() == 3 and x.shape[1] > 10:
            x = x.permute(0, 2, 1)
        
        batch_size = x.size(0)
        
        if self.use_multiscale:
            # Multi-scale feature extraction
            k_scales = [self.k_local, self.k_branch]
            hierarchical_features = get_hierarchical_graph_feature(
                x, k_scales, use_normalized=self.use_normalized_features,
                branch_feature_mode=self.branch_feature_mode,
                local_use_normalized=self.local_use_normalized
            )
            
            x_local = self.conv1_local(hierarchical_features[0])
            x_branch = self.conv1_branch(hierarchical_features[1])
            
            x_local = x_local.max(dim=-1, keepdim=False)[0]
            x_branch = x_branch.max(dim=-1, keepdim=False)[0]
            
            # Fusion
            if self.fusion_type == 'concat_conv':
                x_multi = torch.cat((x_local, x_branch), dim=1)
                x_multi_expanded = x_multi.unsqueeze(-1)
                x1 = self.fusion_conv(x_multi_expanded).squeeze(-1)
            elif self.fusion_type == 'concat_only':
                # True concat-only: no projection, just concatenate (128 channels)
                x1 = torch.cat((x_local, x_branch), dim=1)
            elif self.fusion_type == 'add':
                x1 = x_local + x_branch
            elif self.fusion_type == 'attention':
                x1 = self.fusion_attention(x_local, x_branch)
            elif self.fusion_type == 'gated':
                x1 = self.fusion_gated(x_local, x_branch)
            elif self.fusion_type == 'bilinear':
                x1 = self.fusion_bilinear(x_local, x_branch)
            elif self.fusion_type == 'se_fusion':
                x1 = self.fusion_se(x_local, x_branch)
            else:
                raise ValueError(f"Unknown fusion type: {self.fusion_type}")
        else:
            # Single-scale (local only)
            if self.local_use_normalized:
                local_features = get_graph_feature_norm(x, k=self.k_local)
            else:
                local_features = get_graph_feature(x, k=self.k_local)
            x_local = self.conv1_local(local_features)
            x1 = x_local.max(dim=-1, keepdim=False)[0]
        
        # EdgeConv layers
        x = get_graph_feature(x1, k=self.k_global)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
        
        x = get_graph_feature(x2, k=self.k_global)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        
        x = get_graph_feature(x3, k=self.k_global)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]
        
        # Concatenate multi-scale features
        x = torch.cat((x1, x2, x3, x4), dim=1)
        
        # Global feature
        x = self.conv5(x)
        x1_global = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2_global = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1_global, x2_global), 1)
        
        # Classification
        x = self.classifier(x)
        return x
    
    def get_loss(self, pred, label, smoothing=0.0):
        """Cross-entropy loss with optional label smoothing."""
        if smoothing > 0:
            n_classes = pred.size(-1)
            one_hot = torch.zeros_like(pred).scatter(1, label.unsqueeze(1), 1)
            one_hot = one_hot * (1 - smoothing) + smoothing / n_classes
            log_probs = F.log_softmax(pred, dim=-1)
            loss = -(one_hot * log_probs).sum(dim=-1).mean()
        else:
            loss = F.cross_entropy(pred, label)
        return loss

    def forward_with_intermediates(self, x):
        """Forward pass that returns logits AND intermediate per-point features for interpretability.
        
        Returns:
            logits: [B, num_classes]
            intermediates: dict with keys 'x_local', 'x_branch', 'x1', 'x2', 'x3', 'x4'
                           each tensor is [B, C, N] (per-point features)
        """
        # Handle input shape
        if x.dim() == 3 and x.shape[-1] == 3:
            x = x.permute(0, 2, 1)
        elif x.dim() == 3 and x.shape[1] > 10:
            x = x.permute(0, 2, 1)
        
        batch_size = x.size(0)
        intermediates = {}
        
        if self.use_multiscale:
            # Multi-scale feature extraction
            k_scales = [self.k_local, self.k_branch]
            hierarchical_features = get_hierarchical_graph_feature(
                x, k_scales, use_normalized=self.use_normalized_features,
                branch_feature_mode=self.branch_feature_mode,
                local_use_normalized=self.local_use_normalized
            )
            
            x_local_pre = self.conv1_local(hierarchical_features[0])
            x_branch_pre = self.conv1_branch(hierarchical_features[1])
            
            x_local = x_local_pre.max(dim=-1, keepdim=False)[0]
            x_branch = x_branch_pre.max(dim=-1, keepdim=False)[0]
            
            intermediates['x_local'] = x_local  # [B, 64, N]
            intermediates['x_branch'] = x_branch  # [B, 64, N]
            
            # Fusion
            if self.fusion_type == 'concat_conv':
                x_multi = torch.cat((x_local, x_branch), dim=1)
                x_multi_expanded = x_multi.unsqueeze(-1)
                x1 = self.fusion_conv(x_multi_expanded).squeeze(-1)
            elif self.fusion_type == 'concat_only':
                # True concat-only: no projection, just concatenate (128 channels)
                x1 = torch.cat((x_local, x_branch), dim=1)
            elif self.fusion_type == 'add':
                x1 = x_local + x_branch
            elif self.fusion_type == 'attention':
                x1 = self.fusion_attention(x_local, x_branch)
            elif self.fusion_type == 'gated':
                x1 = self.fusion_gated(x_local, x_branch)
            elif self.fusion_type == 'bilinear':
                x1 = self.fusion_bilinear(x_local, x_branch)
            elif self.fusion_type == 'se_fusion':
                x1 = self.fusion_se(x_local, x_branch)
            else:
                raise ValueError(f"Unknown fusion type: {self.fusion_type}")
        else:
            # Single-scale (local only)
            local_features = get_graph_feature(x, k=self.k_local)
            x_local_pre = self.conv1_local(local_features)
            x_local = x_local_pre.max(dim=-1, keepdim=False)[0]
            x1 = x_local
            intermediates['x_local'] = x_local
            intermediates['x_branch'] = None
        
        intermediates['x1'] = x1  # [B, 64 or 128, N] (fused)
        
        # EdgeConv layers
        x = get_graph_feature(x1, k=self.k_global)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
        intermediates['x2'] = x2  # [B, 64, N]
        
        x = get_graph_feature(x2, k=self.k_global)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        intermediates['x3'] = x3  # [B, 128, N]
        
        x = get_graph_feature(x3, k=self.k_global)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]
        intermediates['x4'] = x4  # [B, 256, N]
        
        # Concatenate multi-scale features
        x = torch.cat((x1, x2, x3, x4), dim=1)
        
        # Global feature
        x = self.conv5(x)
        x1_global = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2_global = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1_global, x2_global), 1)
        
        # Classification
        logits = self.classifier(x)
        return logits, intermediates


def create_model(
    num_classes: int = 7,
    k_scales: List[int] = None,
    use_multiscale: bool = True,
    use_normalized_features: bool = True,
    fusion_type: str = 'concat_conv',
    emb_dims: int = 1024,
    dropout: float = 0.5,
    branch_feature_mode: str = 'full',
    local_use_normalized: bool = False
) -> MS_DGCNN2_Ablation:
    """Factory function to create ablation model variants.
    
    Args:
        branch_feature_mode: 'full' (xyz+relative+normalized) or 'normalized_only' (only normalized)
        local_use_normalized: Whether to use normalized features in local scale (Scale 1)
    """
    if k_scales is None:
        k_scales = [5, 20, 30]
    
    config = AblationConfig(
        num_classes=num_classes,
        emb_dims=emb_dims,
        dropout=dropout,
        k_scales=k_scales,
        use_multiscale=use_multiscale,
        use_normalized_features=use_normalized_features,
        fusion_type=fusion_type,
        branch_feature_mode=branch_feature_mode,
        local_use_normalized=local_use_normalized
    )
    return MS_DGCNN2_Ablation(config)
