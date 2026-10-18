"""Training, CV splits, metrics, and plotting helpers for MS-DGCNN++ experiments."""

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import random
import hashlib
import dataclasses
import inspect
import shutil
import time
from functools import lru_cache
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import h5py
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_recall_fscore_support, f1_score
)
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
from tqdm import tqdm
import math
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RandomZRotate:
    """Random rotation around Z-axis (vertical) for tree point clouds."""
    
    def __call__(self, points):
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


def generate_cv_splits(data_path, k_folds=5, seed=42, force_regenerate=False):
    """
    Generate and save k-fold CV splits to a JSON file.
    Ensures identical splits across all experiments.
    
    Returns:
        splits_path: Path to the JSON file containing splits
    """
    splits_path = os.path.join(data_path, f'cv_splits_k{k_folds}_seed{seed}.json')
    
    if os.path.exists(splits_path) and not force_regenerate:
        with h5py.File(os.path.join(data_path, 'point_cloud_data.h5'), 'r') as h:
            labels = h['labels'][:].reshape(-1)
            classes = [x.decode() if isinstance(x, bytes) else str(x) for x in h['classes'][:]]
        saved = load_cv_splits(splits_path)
        if saved.get('seed') != seed or saved.get('k_folds') != k_folds or saved.get('classes') != classes or saved.get('total_samples') != len(labels):
            raise ValueError('CV split metadata does not match the requested dataset/seed.')
        expected = set(range(len(labels)))
        validation_ids = []
        if len(saved['folds']) != k_folds:
            raise ValueError('Incorrect fold count in saved splits.')
        for i, fold in enumerate(saved['folds']):
            tr, va = fold['train_indices'], fold['val_indices']
            if fold.get('fold') != i or set(tr) & set(va) or set(tr) | set(va) != expected or len(set(tr)) != len(tr) or len(set(va)) != len(va):
                raise ValueError('Invalid, overlapping, or repeated CV split indices.')
            if set(labels[tr]) != set(range(len(classes))) or set(labels[va]) != set(range(len(classes))):
                raise ValueError('Every fold must contain all species.')
            validation_ids.extend(va)
        if sorted(validation_ids) != list(range(len(labels))):
            raise ValueError('Validation folds do not partition the data exactly once.')
        return splits_path
    
    # Load data to get labels for stratification
    h5_path = os.path.join(data_path, 'point_cloud_data.h5')
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Data file not found: {h5_path}")
    
    with h5py.File(h5_path, 'r') as f:
        labels = f['labels'][:]
        classes = [c.decode() if isinstance(c, bytes) else c for c in f['classes'][:]]
    
    if labels.ndim > 1:
        labels = labels.flatten()
    
    # Generate stratified k-fold splits
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    indices = np.arange(len(labels))
    
    splits = {
        'k_folds': k_folds,
        'seed': seed,
        'classes': classes,
        'total_samples': len(labels),
        'folds': []
    }
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(indices, labels)):
        splits['folds'].append({
            'fold': fold_idx,
            'train_indices': train_idx.tolist(),
            'val_indices': val_idx.tolist()
        })
    
    with open(splits_path, 'w') as f:
        json.dump(splits, f, indent=2)
    
    logger.info(f"Generated CV splits: {splits_path}")
    return splits_path


def load_cv_splits(splits_path):
    """Load CV splits from JSON file."""
    with open(splits_path, 'r') as f:
        return json.load(f)


class TreeSpeciesDatasetCV(Dataset):
    """
    Tree Species Dataset with CV fold support.
    Supports density dropout and noise injection for experiments.
    """
    
    def __init__(self, data_path, fold, partition='train', num_points=1024,
                 augment=True, seed=42, k_folds=5,
                 density_retention=1.0, noise_sigma=0.0,
                 density_seed=None, noise_seed=None, coordinate_unit=None, outlier_rate=0.0):
        """
        Args:
            data_path: Path to data directory
            fold: Fold index (0 to k_folds-1)
            partition: 'train' or 'val'
            num_points: Number of points per sample
            augment: Apply RandomZRotate augmentation (train only)
            seed: Random seed for CV splits
            k_folds: Number of folds
            density_retention: Retention rate for upper canopy (Exp 2)
            noise_sigma: Gaussian noise std in mm (Exp 3)
            density_seed: Seed for density dropout (for reproducibility)
        """
        self.num_points = num_points
        self.partition = partition
        self.augment = augment and (partition == 'train')
        self.density_retention = density_retention
        self.noise_sigma = noise_sigma
        self.seed, self.fold, self.epoch = seed, fold, 0
        self.density_seed = seed if density_seed is None else density_seed
        self.noise_seed = seed if noise_seed is None else noise_seed
        self.coordinate_unit, self.outlier_rate = coordinate_unit, outlier_rate
        if not 0 < density_retention <= 1 or not 0 <= outlier_rate <= 1:
            raise ValueError('Invalid retention or outlier fraction.')
        if noise_sigma and coordinate_unit not in ('m', 'cm', 'mm'):
            raise ValueError('Specify --coordinate_unit m, cm, or mm for physical Gaussian noise.')
        
        # Load data
        h5_path = os.path.join(data_path, 'point_cloud_data.h5')
        with h5py.File(h5_path, 'r') as f:
            self.all_data = f['point_clouds'][:]
            self.all_labels = f['labels'][:]
            self.classes = [c.decode() if isinstance(c, bytes) else c 
                           for c in f['classes'][:]]
        
        if self.all_labels.ndim > 1:
            self.all_labels = self.all_labels.flatten()
        
        # Load CV splits
        splits_path = generate_cv_splits(data_path, k_folds=k_folds, seed=seed)
        splits = load_cv_splits(splits_path)
        
        fold_data = splits['folds'][fold]
        if partition == 'train':
            indices = fold_data['train_indices']
        else:
            indices = fold_data['val_indices']
        
        self.indices = np.asarray(indices)
        if self.all_data.shape[1:] != (num_points, 3):
            raise ValueError(f'Expected the FPS cache to have exactly {num_points} XYZ points, got {self.all_data.shape}.')
        self.data = self.all_data[indices]
        self.labels = self.all_labels[indices]
        self.num_classes = len(self.classes)
        
        self.augmentation = RandomZRotate() if self.augment else None
        
        logger.info(f"Loaded {len(self.data)} samples for fold {fold} {partition}")
    
    def apply_density_dropout(self, points, seed=None):
        """Thin the original upper canopy; preserve all survivors and pad uniformly."""
        if self.density_retention >= 1.0:
            return points.copy()
        rng = np.random.default_rng(seed)
        upper = np.flatnonzero(points[:, 2] > np.median(points[:, 2]))
        lower = np.flatnonzero(points[:, 2] <= np.median(points[:, 2]))
        n_keep = max(1, int(len(upper) * self.density_retention)) if len(upper) else 0
        # A common permutation makes retained subsets nested across retention rates.
        indices = np.concatenate((lower, rng.permutation(upper)[:n_keep]))
        if len(indices) < self.num_points:
            indices = np.concatenate((indices, rng.choice(indices, self.num_points-len(indices), replace=True)))
        return points[indices].copy()
    
    def apply_noise(self, points, seed=None):
        """Add physical coordinate noise BEFORE unit-sphere normalization."""
        if self.noise_sigma <= 0:
            return points
        mm_per_unit = {'m': 1000.0, 'cm': 10.0, 'mm': 1.0}
        sigma = self.noise_sigma / mm_per_unit[self.coordinate_unit]
        rng = np.random.default_rng(seed)
        return points + rng.normal(0, sigma, points.shape).astype(np.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        points = self.data[idx].copy()
        sample_id = int(self.indices[idx])
        points = self.apply_density_dropout(points, stable_seed(self.density_seed, sample_id, 'density'))
        points = self.apply_noise(points, stable_seed(self.noise_seed, sample_id, 'noise'))
        if self.outlier_rate:
            rng = np.random.default_rng(stable_seed(self.seed, sample_id, 'outliers'))
            count = int(self.num_points * self.outlier_rate)
            extra = rng.uniform(points.min(0), points.max(0), (count, 3)).astype(np.float32)
            points = np.concatenate((points, extra))
        points = normalize_pc(points)
        if self.augment:
            rng = np.random.default_rng(stable_seed(self.seed, sample_id, self.fold, self.epoch, 'rotation'))
            theta = rng.uniform(0, 2 * np.pi)
            c, t = np.cos(theta), np.sin(theta)
            rotation = np.array([[c, -t, 0], [t, c, 0], [0, 0, 1]], dtype=np.float32)
            points = points @ rotation.T
        return torch.from_numpy(points.astype(np.float32)), int(self.labels[idx])


def get_cv_dataloaders(data_path, fold, batch_size=16, num_points=1024,
                       num_workers=4, seed=42, k_folds=5, density_retention=1.0,
                       noise_sigma=0.0, density_seed=None, noise_seed=None,
                       coordinate_unit=None, outlier_rate=0.0):
    common = dict(fold=fold, num_points=num_points, seed=seed, k_folds=k_folds,
                  density_retention=density_retention, noise_sigma=noise_sigma,
                  density_seed=seed if density_seed is None else density_seed,
                  noise_seed=seed if noise_seed is None else noise_seed,
                  coordinate_unit=coordinate_unit)
    train_dataset = TreeSpeciesDatasetCV(data_path, partition='train', augment=True, **common)
    val_dataset = TreeSpeciesDatasetCV(data_path, partition='val', augment=False,
                                      outlier_rate=outlier_rate, **common)
    generator = torch.Generator().manual_seed(stable_seed(seed, fold, 'loader'))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=True,
                             generator=generator, persistent_workers=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=False,
                           generator=torch.Generator().manual_seed(stable_seed(seed, fold, 'validation')))
    return train_loader, val_loader, train_dataset.num_classes, train_dataset.classes


def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)
    for points, labels in pbar:
        points = points.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(points)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return total_loss / len(train_loader), 100. * correct / total


def evaluate(model, val_loader, criterion, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for points, labels in val_loader:
            points = points.to(device)
            labels = labels.to(device)
            
            outputs = model(points)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    oa = accuracy_score(all_labels, all_preds) * 100
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=np.arange(val_loader.dataset.num_classes), average=None, zero_division=0
    )
    
    # Mean class accuracy
    cm = confusion_matrix(all_labels, all_preds, labels=np.arange(val_loader.dataset.num_classes))
    per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-8)
    macc = per_class_acc.mean() * 100
    
    return {
        'loss': total_loss / len(val_loader),
        'oa': oa,
        'macc': macc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'macro_f1': float(np.mean(f1)),
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels
    }


def train_model(model, train_loader, val_loader, device,
                epochs=100, lr=0.0005, weight_decay=0.0001,
                save_path=None, patience=0):
    """Train for a fixed budget (10-epoch warm-up, cosine decay, clipping); keep the FINAL epoch.

    The validation fold is evaluated after every epoch only to record the learning
    curve (history); it never chooses the checkpoint, so the reported fold is not
    used for any decision. The best-validation epoch is recorded for reference only.
    """
    if len(train_loader) == 0 or epochs < 1:
        raise ValueError('Empty training loader or invalid epoch budget.')
    if patience:
        raise ValueError('Early stopping would select on the reported fold; patience must be 0.')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_oa, best_epoch = -1., 0
    epoch_times, history = [], []
    start_time = time.perf_counter()
    for epoch in range(1, epochs + 1):
        warmup = min(10, epochs)
        factor = epoch / warmup if epoch <= warmup else 0.5 * (1 + math.cos(math.pi * (epoch-warmup) / (epochs-warmup)))
        for group in optimizer.param_groups:
            group['lr'] = lr * factor
        train_loader.dataset.epoch = epoch
        train_loader.generator.manual_seed(stable_seed(train_loader.dataset.seed, train_loader.dataset.fold, epoch, 'loader'))
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        epoch_times.append(time.perf_counter() - start)
        val_metrics = evaluate(model, val_loader, criterion, device)
        history.append({'epoch': epoch, 'lr': lr*factor, 'train_loss': train_loss, 'train_acc': train_acc,
                        'val_oa': val_metrics['oa'], 'val_macc': val_metrics['macc']})
        if val_metrics['oa'] > best_oa:
            best_oa, best_epoch = val_metrics['oa'], epoch
        logger.info('Fold %s epoch %s/%s: OA=%.2f', train_loader.dataset.fold, epoch, epochs, val_metrics['oa'])
    final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    final_metrics = val_metrics.copy()
    final_metrics.update(fold=train_loader.dataset.fold, selected_epoch=epochs, epochs_completed=epoch,
                         selection='final epoch (validation fold not used for selection)',
                         best_val_epoch_for_reference=best_epoch, best_val_oa_for_reference=best_oa,
                         total_training_time=time.perf_counter()-start_time,
                         epoch_time=float(np.mean(epoch_times[min(10,len(epoch_times)-1):])),
                         epoch_times=epoch_times, history=history,
                         sample_ids=val_loader.dataset.indices.tolist())
    if save_path:
        atomic_torch_save({'epoch': epochs, 'model_state_dict': final_state,
                           'metrics': final_metrics, 'complete': True}, save_path)
    return final_metrics, model


def save_config(config, save_path):
    """Save experiment configuration to JSON."""
    config = dict(config, protocol=PROTOCOL_VERSION, warmup_epochs=10,
                  gradient_clip=10, sd_ddof=0, seed_policy='base seed + stable sample ID',
                  checkpoint_selection='final epoch; validation fold never used for selection')
    if Path(save_path).exists():
        old = json.loads(Path(save_path).read_text())
        if old.get('protocol') != PROTOCOL_VERSION:
            raise ValueError('Use a new output directory for the corrected protocol; legacy outputs are preserved.')
    save_results_json(config, save_path)


def set_seed(seed):
    """Seed all generators and require deterministic operations on this environment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class ExperimentResults:
    """Manage experiment results across folds and variants."""
    
    def __init__(self, experiment_name, output_dir):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def add_fold_result(self, variant, fold, metrics):
        """Add results for a specific variant and fold."""
        if variant not in self.results:
            self.results[variant] = {}
        self.results[variant][fold] = metrics
    
    def get_aggregated_metrics(self, variant):
        """Get mean and std of metrics across folds."""
        fold_results = self.results.get(variant, {})
        if not fold_results:
            return None
        
        oas = [r['oa'] for r in fold_results.values()]
        maccs = [r['macc'] for r in fold_results.values()]
        if sorted(fold_results) != list(range(5)):
            raise ValueError('All five folds are required.')
        f1s = np.array([r['f1'] for r in fold_results.values()])
        
        return {
            'oa_mean': np.mean(oas),
            'oa_std': np.std(oas),
            'macc_mean': np.mean(maccs),
            'macc_std': np.std(maccs),
            'f1_mean': np.mean(f1s, axis=0),
            'f1_std': np.std(f1s, axis=0)
        }
    
    def save(self):
        """Save all results to JSON."""
        save_path = self.output_dir / f'{self.experiment_name}_results_{self.timestamp}.json'
        
        # Convert numpy arrays to lists for JSON serialization
        serializable = {}
        for variant, folds in self.results.items():
            serializable[variant] = {}
            for fold, metrics in folds.items():
                serializable[variant][fold] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in metrics.items()
                }
        
        with open(save_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        logger.info(f"Results saved to {save_path}")
        return save_path


# Shared checks used by the existing experiment entry points.
PROTOCOL_VERSION = 'paper-consistent-v3'
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def stable_seed(seed, *parts):
    payload = json.dumps([int(seed), *parts], sort_keys=True).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], 'little')


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


@lru_cache(maxsize=32)
def _file_digest(path, size, mtime_ns):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def file_digest(path):
    path = Path(path).resolve()
    info = path.stat()
    return _file_digest(str(path), info.st_size, info.st_mtime_ns)


def add_common_arguments(parser):
    """Keep existing CLIs, adding only options needed for shared reproducibility."""
    options = parser._option_string_actions
    specs = [('--cache_dir', dict(default='results/paper_shared_v2')),
             ('--num_workers', dict(type=int, default=4)),
             ('--device', dict(default='cuda')),
             ('--eval_only', dict(action='store_true')),
             ('--coordinate_unit', dict(choices=['m', 'cm', 'mm'], default='m',
                                        help='Input XYZ coordinate unit (default: m; the STPCTLS '
                                             'H5 stores meters, trees are ~28 units tall)'))]
    for name, kwargs in specs:
        if name not in options:
            parser.add_argument(name, **kwargs)


# One training recipe for every script, so a model configuration always maps to the
# same shared checkpoint (and therefore the same values) in every table and figure.
PAPER_TRAINING = {'epochs': 100, 'batch_size': 16, 'lr': 0.0005, 'weight_decay': 0.0001,
                  'num_points': 1024, 'patience': 0, 'seed': 42, 'k_folds': 5}


def prepare_args(args):
    differing = {k: getattr(args, k) for k, v in PAPER_TRAINING.items() if getattr(args, k, v) != v}
    if differing:
        raise ValueError(f'Training settings differ from the shared paper recipe {PAPER_TRAINING}: '
                         f'{differing}. Shared checkpoints would no longer match across experiments.')
    for name in ['data_path', 'output_dir', 'cache_dir', 'resume_dir', 'import_checkpoints']:
        value = getattr(args, name, None)
        if value and not Path(value).is_absolute():
            setattr(args, name, str(PROJECT_ROOT / value))
    if args.k_folds != 5:
        raise ValueError('The paper reports five folds; --k_folds must be 5.')
    if args.epochs < 1 or args.batch_size < 2:
        raise ValueError('Positive epochs and batch_size >= 2 required.')
    return args


def get_device(args):
    return torch.device(args.device if torch.cuda.is_available() else 'cpu')


def checkpoint_metadata(model, train_loader, args, device):
    dataset = train_loader.dataset
    split_path = Path(args.data_path) / f'cv_splits_k{args.k_folds}_seed{args.seed}.json'
    if hasattr(model, 'config') and dataclasses.is_dataclass(model.config):
        architecture = dataclasses.asdict(model.config)
    else:
        architecture = {'num_classes': model.num_classes,
                        'k_scales': [model.k1, model.k2, model.k3],
                        'fps_points': model.fps_points, 'dropout': model.dp1.p}
    # Unit affects only physical-noise experiments; clean runs share one identity.
    return {'protocol': PROTOCOL_VERSION, 'model_class': type(model).__name__,
            'architecture': architecture,
            'source': file_digest(inspect.getfile(type(model))),
            'training_source': file_digest(__file__),
            'dataset': file_digest(Path(args.data_path) / 'point_cloud_data.h5'),
            'splits': file_digest(split_path), 'fold': dataset.fold,
            'seed': args.seed, 'model_seed': args.seed + dataset.fold,
            'density_seed': args.seed, 'noise_seed': args.seed,
            'retention': dataset.density_retention, 'noise_sigma_mm': dataset.noise_sigma,
            'coordinate_unit': args.coordinate_unit if dataset.noise_sigma else None,
            'num_points': args.num_points, 'batch_size': args.batch_size,
            'num_workers': args.num_workers, 'epochs': args.epochs,
            'lr': args.lr, 'weight_decay': args.weight_decay, 'patience': args.patience,
            'warmup_epochs': 10, 'gradient_clip': 10.,
            'selection': 'final epoch; validation fold never used for selection',
            'torch': torch.__version__, 'numpy': np.__version__,
            'device': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
            'cuda': torch.version.cuda if device.type == 'cuda' else None}


# Recorded in every checkpoint but not part of its identity, so a configuration trained
# on one machine is reused (not retrained with possibly different values) on another.
ENVIRONMENT_FIELDS = ('torch', 'numpy', 'device', 'cuda', 'num_workers')


def checkpoint_identity(metadata):
    return {k: v for k, v in metadata.items() if k not in ENVIRONMENT_FIELDS}


def run_shared_fold(args, fold, factory, retention=1., noise_sigma=0., checkpoint_path=None):
    """One canonical checkpoint for an identical model/data/training configuration.

    Completed folds are reused. An interrupted fold restarts with the same seed;
    incomplete or legacy checkpoints never supply metrics to a paper output.
    """
    device = get_device(args)
    set_seed(args.seed + fold)
    train_loader, val_loader, nc, _ = get_cv_dataloaders(
        args.data_path, fold=fold, batch_size=args.batch_size, num_points=args.num_points,
        num_workers=args.num_workers, seed=args.seed, k_folds=args.k_folds,
        density_retention=retention, density_seed=args.seed, noise_sigma=noise_sigma,
        noise_seed=args.seed, coordinate_unit=args.coordinate_unit)
    set_seed(args.seed + fold)
    model = factory(nc).to(device)
    metadata = checkpoint_metadata(model, train_loader, args, device)
    identity = checkpoint_identity(metadata)
    cache_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache_path = Path(args.cache_dir) / f'{cache_id}.pth'
    cached = None
    if cache_path.exists():
        cached = torch.load(cache_path, map_location=device, weights_only=False)
        if checkpoint_identity(cached.get('metadata', {})) != identity or not cached.get('complete'):
            raise ValueError(f'Invalid cached checkpoint: {cache_path}')
    if cached is None and checkpoint_path and Path(checkpoint_path).exists():
        candidate = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint_identity(candidate.get('metadata', {})) != identity or not candidate.get('complete'):
            raise ValueError(f'Legacy/incompatible checkpoint: {checkpoint_path}. Use a new output directory.')
        cached = candidate
        atomic_torch_save(cached, cache_path)
    if cached is None:
        if args.eval_only:
            raise FileNotFoundError(f'Missing completed fold {fold}: {cache_path}. No partial table will be produced.')
        metrics, model = train_model(model, train_loader, val_loader, device,
                                     epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                                     patience=args.patience)
        metrics['cache_id'] = cache_id
        cached = {'metadata': metadata, 'complete': True, 'metrics': metrics,
                  'epoch': metrics['selected_epoch'],
                  'model_state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()}}
        atomic_torch_save(cached, cache_path)
    else:
        model.load_state_dict(cached['model_state_dict'])
        logger.info('Reusing fold %s: %s, OA=%.2f', fold, cache_id[:12], cached['metrics']['oa'])
    sidecar = cache_path.with_suffix('.json')
    if not sidecar.exists():
        save_results_json({'cache_id': cache_id, 'metadata': cached['metadata'],
                           'metrics': cached['metrics']}, sidecar)
    if checkpoint_path:
        target = Path(checkpoint_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            alias = torch.load(target, map_location='cpu', weights_only=False)
            if checkpoint_identity(alias.get('metadata', {})) != identity or not alias.get('complete'):
                raise ValueError(f'Incompatible output checkpoint: {target}. Use a new run directory.')
        else:
            try:
                os.link(cache_path, target)
            except OSError:
                shutil.copy2(cache_path, target)
    return cached['metrics'], model


def get_val_loader(args, fold, retention=1., noise_sigma=0., outlier_rate=0.):
    """Validation loader of one fold with a test-time corruption (no training split built)."""
    dataset = TreeSpeciesDatasetCV(args.data_path, fold=fold, partition='val', augment=False,
                                   num_points=args.num_points, seed=args.seed, k_folds=args.k_folds,
                                   density_retention=retention, noise_sigma=noise_sigma,
                                   density_seed=args.seed, noise_seed=args.seed,
                                   coordinate_unit=args.coordinate_unit, outlier_rate=outlier_rate)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                      num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), drop_last=False)


def evaluate_corrupted(args, fold, model, clean_metrics, retention=1., noise_sigma=0., outlier_rate=0.,
                       output_dir=None):
    """Test a clean-trained checkpoint on corrupted validation clouds (no retraining).

    Every evaluation is stored as JSON (metrics, confusion matrix, predictions, labels,
    sample IDs) in <cache_dir>/evaluations, keyed by the checkpoint and the corruption,
    and copied to <output_dir>/evaluations. New corruption levels only need --eval_only.
    """
    device = get_device(args)
    corruption = {'retention': float(retention), 'noise_sigma_mm': float(noise_sigma),
                  'coordinate_unit': args.coordinate_unit if noise_sigma else None,
                  'outlier_rate': float(outlier_rate)}
    key = {'checkpoint': clean_metrics['cache_id'], 'fold': fold, 'corruption': corruption,
           'seed': args.seed, 'num_points': args.num_points, 'protocol': PROTOCOL_VERSION,
           'evaluation_source': file_digest(__file__)}
    eval_id = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
    eval_path = Path(args.cache_dir) / 'evaluations' / f'{eval_id}.json'
    if eval_path.exists():
        result = json.loads(eval_path.read_text())
    else:
        loader = get_val_loader(args, fold, retention, noise_sigma, outlier_rate)
        metrics = evaluate(model.to(device), loader, nn.CrossEntropyLoss(), device)
        result = dict(metrics, **key, eval_id=eval_id, sample_ids=loader.dataset.indices.tolist(),
                      protocol_name='clean_train_corrupted_test')
        save_results_json(result, eval_path)
        result = json.loads(eval_path.read_text())
    # Fields needed by aggregate_metrics; training time comes from the clean run
    result.update(fold=fold, epoch_time=clean_metrics['epoch_time'], clean_oa=clean_metrics['oa'])
    if output_dir is not None:
        save_results_json(result, Path(output_dir) / 'evaluations' / f'{eval_id}.json')
    logger.info('Fold %s %s: OA=%.2f (clean %.2f)', fold, corruption, result['oa'], clean_metrics['oa'])
    return result


def parse_levels(values, name):
    """Validate a CLI list of corruption levels."""
    values = [float(v) for v in values]
    if not values or len(set(values)) != len(values):
        raise ValueError(f'{name}: give distinct values.')
    return values


def aggregate_metrics(folds_data, k_folds=5):
    if sorted(f['fold'] for f in folds_data) != list(range(k_folds)):
        raise ValueError('Exactly one completed result per fold is required.')
    values = {'oa': [f['oa'] for f in folds_data], 'macc': [f['macc'] for f in folds_data],
              'f1': [float(np.mean(f['f1'])) for f in folds_data],
              'epoch_time': [f['epoch_time'] for f in folds_data]}
    if not all(np.isfinite(v).all() for v in values.values()):
        raise ValueError('Non-finite fold metrics.')
    return {'mean': {k: float(np.mean(v)) for k, v in values.items()},
            'std': {k: float(np.std(v, ddof=0)) for k, v in values.items()},
            'folds': folds_data}


def save_results_json(value, path):
    def encode(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        raise TypeError(type(v).__name__)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=encode, allow_nan=False))
    temporary.replace(path)
