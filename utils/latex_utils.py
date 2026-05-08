"""
LaTeX table generation utilities for ablation studies.
Generates booktabs-style tables for publication.
"""

from typing import Dict, List, Tuple, Optional
import os


LATEX_PREAMBLE = r"""\documentclass[11pt]{article}

% Required packages
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[table]{xcolor}
\usepackage{geometry}
\geometry{margin=1in}

% Custom colors
\definecolor{bestcolor}{RGB}{220, 240, 220}

\begin{document}

"""

LATEX_POSTAMBLE = r"""
\end{document}
"""


def wrap_standalone(table_content: str, title: str = None) -> str:
    """Wrap table in a complete LaTeX document."""
    doc = LATEX_PREAMBLE
    if title:
        doc += f"\\section*{{{title}}}\n\n"
    doc += table_content
    doc += LATEX_POSTAMBLE
    return doc


def format_metric(mean: float, std: float, bold: bool = False) -> str:
    """Format metric as 'mean ± std' with optional bold."""
    text = f"{mean:.2f} $\\pm$ {std:.2f}"
    if bold:
        text = f"\\textbf{{{text}}}"
    return text


def generate_ablation_table(
    results: Dict[str, Dict[str, Tuple[float, float]]],
    study_groups: Dict[str, List[str]],
    metrics: List[str] = None,
    metric_labels: Dict[str, str] = None,
    caption: str = "Ablation Study Results",
    label: str = "tab:ablation",
    feature_descriptions: Dict[str, str] = None
) -> str:
    """
    Generate a combined LaTeX table for multiple ablation studies.
    
    Args:
        results: Dict mapping variant_name -> {metric_name: (mean, std)}
        study_groups: Dict mapping study_name -> [variant_names]
                     e.g., {'Multi-scale': ['Single (k=5)', 'Multi (k=[5,20])']}
        metrics: List of metric names to include
        metric_labels: Dict mapping metric_name -> display_label
        caption: Table caption
        label: LaTeX label
        feature_descriptions: Dict mapping variant_name -> feature description string
    
    Returns:
        LaTeX table string
    """
    if metrics is None:
        metrics = ['accuracy', 'balanced_accuracy', 'f1_macro', 'kappa', 'mcc']
    
    if metric_labels is None:
        metric_labels = {
            'accuracy': 'Acc',
            'balanced_accuracy': 'Bal. Acc',
            'f1_macro': 'F1',
            'kappa': '$\\kappa$',
            'mcc': 'MCC'
        }
    
    include_features = feature_descriptions is not None and len(feature_descriptions) > 0
    
    # Find best values for each metric within each study group
    best_per_study = {}
    for study_name, variants in study_groups.items():
        best_per_study[study_name] = {}
        for metric in metrics:
            best_val = -float('inf')
            for variant in variants:
                if variant in results:
                    val = results[variant][metric][0]
                    if val > best_val:
                        best_val = val
            best_per_study[study_name][metric] = best_val
    
    # Build table
    lines = []
    
    # Preamble
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    
    # Column spec
    num_metrics = len(metrics)
    if include_features:
        col_spec = "lll" + "c" * num_metrics  # Study, Variant, Features, Metrics...
    else:
        col_spec = "ll" + "c" * num_metrics
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    
    # Header row
    if include_features:
        header_cols = ["Study", "Variant", "Features"] + [metric_labels.get(m, m) for m in metrics]
    else:
        header_cols = ["Study", "Variant"] + [metric_labels.get(m, m) for m in metrics]
    lines.append(" & ".join(header_cols) + " \\\\")
    lines.append("\\midrule")
    
    # Data rows
    for study_idx, (study_name, variants) in enumerate(study_groups.items()):
        num_variants = len(variants)
        
        for var_idx, variant in enumerate(variants):
            row = []
            
            # Study name (only on first row, use multirow)
            if var_idx == 0:
                row.append(f"\\multirow{{{num_variants}}}{{*}}{{{study_name}}}")
            else:
                row.append("")
            
            # Variant name
            row.append(variant)
            
            # Feature description (if provided)
            if include_features:
                feat_desc = feature_descriptions.get(variant, '--')
                row.append(feat_desc)
            
            # Metrics
            if variant in results:
                for metric in metrics:
                    mean, std = results[variant][metric]
                    is_best = abs(mean - best_per_study[study_name][metric]) < 0.01
                    row.append(format_metric(mean, std, bold=is_best))
            else:
                row.extend(["--"] * num_metrics)
            
            lines.append(" & ".join(row) + " \\\\")
        
        # Add midrule between study groups (except after last)
        if study_idx < len(study_groups) - 1:
            lines.append("\\midrule")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def generate_kscale_table(
    results: Dict[Tuple[int, int], Dict[str, Tuple[float, float]]],
    k_local_values: List[int],
    k_branch_values: List[int],
    metric: str = 'accuracy',
    caption: str = "K-Scale Sensitivity Analysis",
    label: str = "tab:kscale"
) -> str:
    """
    Generate a LaTeX table for k-scale sensitivity analysis.
    
    Args:
        results: Dict mapping (k_local, k_branch) -> {metric: (mean, std)}
        k_local_values: List of k_local values tested
        k_branch_values: List of k_branch values tested
        metric: Metric to display
        caption: Table caption
        label: LaTeX label
    
    Returns:
        LaTeX table string
    """
    lines = []
    
    # Find best value
    best_val = -float('inf')
    for key, metrics_dict in results.items():
        if metric in metrics_dict:
            val = metrics_dict[metric][0]
            if val > best_val:
                best_val = val
    
    # Preamble
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    
    # Column spec
    num_cols = len(k_branch_values) + 1
    col_spec = "c" * num_cols
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    
    # Header row
    header = ["$k_{local}$ / $k_{branch}$"] + [str(k) for k in k_branch_values]
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")
    
    # Data rows
    for k_local in k_local_values:
        row = [str(k_local)]
        for k_branch in k_branch_values:
            key = (k_local, k_branch)
            if key in results and metric in results[key]:
                mean, std = results[key][metric]
                is_best = abs(mean - best_val) < 0.01
                row.append(format_metric(mean, std, bold=is_best))
            else:
                row.append("--")
        lines.append(" & ".join(row) + " \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def save_latex_table(latex_str: str, filepath: str, standalone: bool = True, title: str = None):
    """
    Save LaTeX table to file.
    
    Args:
        latex_str: LaTeX table content
        filepath: Output file path
        standalone: If True, wrap in complete document (compilation-ready)
        title: Optional title for standalone document
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if standalone:
        content = wrap_standalone(latex_str, title=title)
    else:
        content = latex_str
    
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Saved LaTeX {'document' if standalone else 'table'} to: {filepath}")


if __name__ == '__main__':
    # Test table generation
    results = {
        'Single (k=5)': {
            'accuracy': (85.2, 1.3),
            'balanced_accuracy': (84.1, 1.5),
            'f1_macro': (83.5, 1.4),
            'kappa': (82.0, 1.6),
            'mcc': (81.5, 1.7)
        },
        'Single (k=20)': {
            'accuracy': (86.8, 1.1),
            'balanced_accuracy': (85.5, 1.2),
            'f1_macro': (85.0, 1.3),
            'kappa': (83.5, 1.4),
            'mcc': (83.0, 1.5)
        },
        'Multi (k=[5,20])': {
            'accuracy': (89.5, 0.8),
            'balanced_accuracy': (88.2, 0.9),
            'f1_macro': (87.8, 1.0),
            'kappa': (86.5, 1.1),
            'mcc': (86.0, 1.2)
        }
    }
    
    study_groups = {
        'Multi-scale': ['Single (k=5)', 'Single (k=20)', 'Multi (k=[5,20])']
    }
    
    latex = generate_ablation_table(results, study_groups)
    print(latex)
