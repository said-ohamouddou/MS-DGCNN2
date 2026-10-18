"""Baselines shared by the robustness studies (density dropout, noise, outliers).

Defined once so every experiment builds exactly the same models and therefore
reuses the same clean checkpoints (and evaluations) from the shared cache.
"""

from pathlib import Path

from models.msdgcnn2 import create_model
from models.msdgcnn import create_ms_dgcnn

POINTM2AE_CKPT = Path(__file__).resolve().parent / 'pointm2ae' / 'pretrained_m2ae.pth'


def create_dgcnn(num_classes):
    """DGCNN (k1=5, k=30): EdgeConv neighborhoods 5, 30, 30, 30 (same as Table 7)."""
    return create_model(num_classes=num_classes, k_scales=[5, 5, 30], use_multiscale=False,
                        use_normalized_features=False, local_use_normalized=False,
                        fusion_type='concat_conv')


def create_msdgcnn(num_classes):
    """MS-DGCNN, parallel scales [5, 20, 30], all input points (no internal FPS)."""
    return create_ms_dgcnn(num_classes=num_classes, k_scales=[5, 20, 30])


def create_pointm2ae_finetuned(num_classes):
    """Point-M2AE initialized from the ShapeNet-pretrained weights, fully fine-tuned."""
    from models.pointm2ae import create_pointm2ae
    if not POINTM2AE_CKPT.exists():
        raise FileNotFoundError(f'Point-M2AE pretrained weights not found: {POINTM2AE_CKPT}')
    return create_pointm2ae(num_classes=num_classes, ckpt_path=str(POINTM2AE_CKPT))


BASELINES = {
    'dgcnn': {'label': 'DGCNN ($k_1$=5, $k$=30)', 'factory': create_dgcnn},
    'ms_dgcnn': {'label': 'MS-DGCNN', 'factory': create_msdgcnn},
    'pointm2ae': {'label': 'Point-M2AE', 'factory': create_pointm2ae_finetuned},
}
# Distinct from the MS-DGCNN++ variant colors; drawn with dashed lines
BASELINE_COLORS = {'dgcnn': '#6C757D', 'ms_dgcnn': '#8D6E63', 'pointm2ae': '#7B2CBF'}
