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
        'font.weight': 'bold',

        'text.usetex': False,
        'mathtext.fontset': 'dejavusans',

        'axes.labelsize': 21,
        'axes.labelweight': 'bold',
        'axes.titlesize': 24,
        'axes.titleweight': 'bold',
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
