"""Publication-oriented matplotlib defaults."""

import matplotlib.pyplot as plt
import matplotlib as mpl

COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'tertiary': '#F18F01',
    'quaternary': '#C73E1D',
    'quinary': '#3B1F2B',
    'success': '#28A745',
    'warning': '#FFC107',
    'info': '#17A2B8',
}

VARIANT_COLORS = [
    '#E63946',
    '#457B9D',
    '#2A9D8F',
    '#E9C46A',
    '#F4A261',
]

ALT_COLORS = [
    '#264653',
    '#2A9D8F',
    '#E9C46A',
    '#F4A261',
    '#E76F51',
]


def set_publication_style():
    """Set matplotlib style for publication-quality figures."""

    plt.style.use('seaborn-v0_8-whitegrid')

    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 18,
        'font.weight': 'normal',

        'text.usetex': False,
        'mathtext.fontset': 'dejavusans',

        'axes.labelsize': 21,
        'axes.labelweight': 'normal',
        'axes.titlesize': 24,
        'axes.titleweight': 'normal',
        'axes.linewidth': 2.0,
        'axes.edgecolor': '#333333',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'axes.axisbelow': True,

        'grid.color': '#E0E0E0',
        'grid.linewidth': 1.0,
        'grid.alpha': 0.7,

        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'xtick.major.width': 2.0,
        'ytick.major.width': 2.0,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.direction': 'out',
        'ytick.direction': 'out',

        'legend.fontsize': 16,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '#CCCCCC',
        'legend.fancybox': False,

        'figure.facecolor': 'white',
        'figure.edgecolor': 'white',
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,

        'lines.linewidth': 3.0,
        'lines.markersize': 12,
    })


def get_variant_colors(n=5):
    """Get publication-friendly colors for variants."""
    return VARIANT_COLORS[:n]


def get_figure_size(aspect='wide'):
    """Get standard figure sizes for publication."""
    sizes = {
        'wide': (12, 6),
        'square': (8, 8),
        'tall': (8, 10),
        'single_column': (6, 5),
        'double_column': (12, 5),
        'heatmap': (10, 8),
    }
    return sizes.get(aspect, (10, 6))


def add_panel_label(ax, label, loc='upper left', fontsize=16):
    """Add panel label (a), (b), etc. to figure."""
    loc_coords = {
        'upper left': (0.02, 0.98),
        'upper right': (0.98, 0.98),
        'lower left': (0.02, 0.02),
        'lower right': (0.98, 0.02),
        'bottom left': (0.02, 0.02),
    }
    
    x, y = loc_coords.get(loc, (0.02, 0.98))
    ha = 'left' if 'left' in loc else 'right'
    va = 'top' if 'upper' in loc else 'bottom'
    
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold',
            ha=ha, va=va,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor='none', alpha=0.8))


def format_axis_labels(ax, xlabel=None, ylabel=None, title=None):
    """Format axis labels with bold font."""
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    if title:
        ax.set_title(title, fontsize=16, fontweight='bold', pad=10)


def save_figure(fig, path, formats=['png', 'pdf']):
    """Save figure in multiple formats."""
    from pathlib import Path
    path = Path(path)
    
    for fmt in formats:
        save_path = path.with_suffix(f'.{fmt}')
        fig.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')


TERM_MAPPING = {
    'branch': 'intermediate',
    'Branch': 'Intermediate',
    'BRANCH': 'INTERMEDIATE',
    'F_branch': 'F_intermediate',
    'x_branch': 'x_intermediate',
    'conv1_branch': 'conv1_intermediate',
}


def replace_branch_terminology(text):
    """Replace 'branch' with 'intermediate' in text."""
    if text is None:
        return None
    for old, new in TERM_MAPPING.items():
        text = text.replace(old, new)
    return text


def save_paper_transformations(args, figures_dir, sample_id=None, noise_levels=(0, 50, 300),
                               retentions=(1.0, 0.25, 0.05), outlier_rates=(0.0, 0.10, 0.25)):
    """Figure 3: side views of one validation tree under each perturbation (paper layout,
    coloured by height). Uses exactly the loader's perturbations and seeds."""
    from pathlib import Path
    import numpy as np
    from utils.experiment_utils import TreeSpeciesDatasetCV, save_results_json, stable_seed
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    # sample_id: tree index in the H5 file (default: first validation tree of fold 0)
    fold = 0
    if sample_id is not None:
        for f in range(args.k_folds):
            if int(sample_id) in TreeSpeciesDatasetCV(args.data_path, fold=f, partition='val', augment=False,
                                                      num_points=args.num_points, seed=args.seed,
                                                      k_folds=args.k_folds).indices:
                fold = f
                break
    common = dict(fold=fold, partition='val', augment=False, num_points=args.num_points,
                  seed=args.seed, k_folds=args.k_folds, coordinate_unit=args.coordinate_unit)
    base = TreeSpeciesDatasetCV(args.data_path, **common)
    position = 0 if sample_id is None else int(np.flatnonzero(base.indices == int(sample_id))[0])
    sample_id = int(base.indices[position])
    raw = base.data[position].astype(np.float32)
    centroid = raw.mean(0)
    radius = np.max(np.linalg.norm(raw - centroid, axis=1))
    frame = lambda p: (p - centroid) / radius          # one frame for all panels of a figure
    z_range = (frame(raw)[:, 2].min(), frame(raw)[:, 2].max())

    def noisy(sigma):
        if not sigma:
            return frame(raw)
        d = TreeSpeciesDatasetCV(args.data_path, noise_sigma=float(sigma), **common)
        return frame(d.apply_noise(raw.copy(), stable_seed(args.seed, sample_id, 'noise')))

    def thinned(r):
        d = TreeSpeciesDatasetCV(args.data_path, density_retention=float(r), **common)
        kept = d.apply_density_dropout(raw.copy(), stable_seed(args.seed, sample_id, 'density'))
        kept_rows = {tuple(p) for p in kept}
        removed = np.array([p for p in raw if tuple(p) not in kept_rows]).reshape(-1, 3)
        return frame(np.unique(kept, axis=0)), frame(removed)

    def contaminated(rate):
        rng = np.random.default_rng(stable_seed(args.seed, sample_id, 'outliers'))
        extra = rng.uniform(raw.min(0), raw.max(0), (int(args.num_points * rate), 3)).astype(np.float32)
        return frame(raw), frame(extra)

    style = {'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans', 'font.size': 8,
             'font.weight': 'normal', 'axes.titleweight': 'normal'}
    cmap = 'viridis'

    def draw(panels, name):
        with mpl.rc_context(style):
            fig, axes = plt.subplots(1, 3, figsize=(5.596, 2.384))
            everything = np.concatenate([np.concatenate(layers) for layers, _ in panels])
            lim_x = (everything[:, 0].min(), everything[:, 0].max())
            lim_z = (everything[:, 2].min(), everything[:, 2].max())
            for ax, (layers, title) in zip(axes, panels):
                tree, grey, red = layers + [np.empty((0, 3))] * (3 - len(layers))
                if len(grey):
                    ax.scatter(grey[:, 0], grey[:, 2], s=2.2, color='0.78', linewidths=0)
                ax.scatter(tree[:, 0], tree[:, 2], s=2.2, c=tree[:, 2], cmap=cmap,
                           vmin=z_range[0], vmax=z_range[1], linewidths=0)
                if len(red):
                    ax.scatter(red[:, 0], red[:, 2], s=2.6, color='#D62728', linewidths=0)
                ax.set_xlim(*lim_x); ax.set_ylim(*lim_z)
                ax.set_aspect('equal'); ax.set_axis_off()
                ax.set_title(title, pad=2)
            fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.90, wspace=0.05)
            for ext in ('pdf', 'png'):
                fig.savefig(figures_dir / f'transformation_{name}.{ext}', dpi=300, facecolor='white')
            plt.close(fig)

    draw([([noisy(s)], rf'$\sigma$ = {s:g} mm') for s in noise_levels], 'noise')
    draw([([*thinned(r)][::1], f'r = {r * 100:g}%') if r < 1 else ([frame(raw)], f'r = {r * 100:g}%')
          for r in retentions], 'dropout')
    draw([([frame(raw)], '0% outliers') if not rate else
          ([contaminated(rate)[0], np.empty((0, 3)), contaminated(rate)[1]], f'+{rate * 100:g}% outliers')
          for rate in outlier_rates], 'outlier')
    save_results_json({'sample_id': sample_id, 'coordinate_unit': args.coordinate_unit,
                       'noise_sigma_mm': list(noise_levels), 'retention': list(retentions),
                       'outlier_rate': list(outlier_rates), 'seed': args.seed,
                       'colouring': 'height (viridis); removed points grey; added outliers red'},
                      figures_dir / 'transformation_settings.json')
