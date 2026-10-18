"""LaTeX tables for the robustness studies (kept separate from experiment_utils.py,
whose content is part of the checkpoint identity)."""

from pathlib import Path

import numpy as np


def level_table(results, labels, levels, level_header, level_fmt, caption, label, path, metric='oa'):
    """Rows = corruption levels, columns = models; mean $\\pm$ SD over folds; bold = best mean per row.

    results[model][level][fold] -> metrics dict (as saved by the experiment scripts).
    """
    models = list(labels)
    cols = 'l' + 'c' * len(models)
    lines = [r'\begin{table*}[htbp]', r'\centering', rf'\caption{{{caption}}}', rf'\label{{{label}}}',
             r'\resizebox{\textwidth}{!}{%', rf'\begin{{tabular}}{{{cols}}}', r'\toprule',
             level_header + ' & ' + ' & '.join(labels[m] for m in models) + r' \\', r'\midrule']
    for level in levels:
        stats = {}
        for m in models:
            folds = results[m][level]
            values = [folds[f][metric] for f in sorted(folds)]
            stats[m] = (float(np.mean(values)), float(np.std(values, ddof=0)))
        best = max(round(v[0], 1) for v in stats.values())
        cells = [level_fmt(level)]
        for m in models:
            mean, sd = stats[m]
            cell = f'{mean:.1f}$\\pm${sd:.1f}'
            cells.append(r'\textbf{' + cell + '}' if round(mean, 1) == best else cell)
        lines.append(' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}}', r'\end{table*}']
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text('\n'.join(lines) + '\n')


def matched_table(rows, path):
    """Clean-trained vs matched-trained OA at the matched retention rates."""
    lines = [r'\begin{table}[htbp]', r'\centering',
             r'\caption{Density dropout: models trained on clean data vs.\ trained on thinned data (matched), '
             r'both tested at the same upper-canopy retention. OA (\%), mean $\pm$ standard deviation across five folds (ddof=0).}',
             r'\label{tab:matched_training}', r'\begin{tabular}{lccc}', r'\toprule',
             r'Variant & Retention & Clean training & Matched training \\', r'\midrule']
    for r in rows:
        clean = (f"{r['clean_train_oa_mean']:.1f}$\\pm${r['clean_train_oa_std']:.1f}"
                 if r['clean_train_oa_mean'] is not None else '--')
        lines.append(f"{r['variant']} & {r['retention']*100:g}\\% & {clean} & "
                     f"{r['matched_train_oa_mean']:.1f}$\\pm${r['matched_train_oa_std']:.1f} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text('\n'.join(lines) + '\n')
