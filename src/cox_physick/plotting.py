"""Publication plots from the supplied TCOM result coordinates."""
from pathlib import Path
import csv
import numpy as np

METHODS = {
    'maxsinr_ttt': ('Max-SINR+TTT', '#222222', 'x'),
    'maxrst': ('Max-RST', '#777777', '+'),
    'greedy_analytic': ('Load-aware greedy', '#E69F00', 'v'),
    'cox_only': ('Cox-only', '#56B4E9', 'h'),
    'mlp213': ('TGN-MLP', '#D55E00', 'o'),
    'kan_generic': ('TGN-KAN', '#0072B2', 's'),
    'leo_madrl': ('LEO-MADRL', '#8B62A8', 'D'),
    'full': ('Cox-PhysiCK', '#009E73', '^'),
}
LABELS = {'outage_percent': 'Outage (%)', 'hof_percent': 'Conditional HOF (%)',
          'pp_percent': 'PP (%)', 'tp_Mbps': 'TP (Mbps)',
          'peak_normalized': 'Peak', 'J_user': 'Cost per user–epoch'}


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def plot_results(data_root, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    data_root, output = Path(data_root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    coordinates = data_root / 'figure_points'
    rows = read_csv(coordinates / 'density.csv')
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6))
    fig.subplots_adjust(top=.78, bottom=.11, left=.08, right=.98, hspace=.5, wspace=.4)
    for ax, panel, metric in zip(axes.flat, 'abcdef', LABELS):
        for method, (label, color, marker) in METHODS.items():
            r = sorted([r for r in rows if r['method_id'] == method and r['metric'] == metric], key=lambda r: int(r['K']))
            if not r:
                raise ValueError(f'Missing density coordinates: {method} {metric}')
            ax.errorbar([int(x['K']) for x in r], [float(x['mean']) for x in r],
                        yerr=[float(x['sample_sd']) for x in r], color=color,
                        marker=marker, markersize=3, capsize=1.6, linewidth=.9, label=label)
        ax.set(xlabel='Users, K', ylabel=LABELS[metric], title=f'({panel})')
        ax.set_xticks([200,300,400,500]); ax.grid(alpha=.15)
        if metric != 'J_user': ax.set_ylim(bottom=0)
        if metric == 'tp_Mbps': ax.set_ylim(top=240)
        if metric == 'peak_normalized': ax.set_ylim(top=1.05)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False, fontsize=7.5)
    _save(fig, output / 'figure_3_density', plt)

    rows = read_csv(coordinates / 'kappa.csv')
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), layout='constrained')
    for ax, panel in zip(axes, ['a','b']):
        subset = [r for r in rows if r['panel'] == panel]
        for series in dict.fromkeys(r['series'] for r in subset):
            r = sorted([r for r in subset if r['series'] == series], key=lambda x:float(x['kappa']))
            ax.errorbar([float(x['kappa']) for x in r], [float(x['mean']) for x in r],
                        yerr=[float(x['sample_sd']) for x in r], marker='o', markersize=3,
                        capsize=2, linewidth=1, label=series)
        ax.axvline(.6, color='.55', linestyle=':', linewidth=.8)
        ax.set(xlabel=r'$\kappa$', ylabel='Probability' if panel=='a' else 'Brier score',title=f'({panel})')
        ax.legend(frameon=False,fontsize=7); ax.grid(alpha=.15)
    _save(fig, output / 'figure_4_kappa', plt)

    rows = read_csv(coordinates / 'weight_ratios.csv')
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.5))
    fig.subplots_adjust(top=.85,bottom=.11,left=.09,right=.97,hspace=.48,wspace=.30)
    names = {'a': r'$w_o$', 'b': r'$w_h$', 'c': r'$w_\ell$', 'd': r'$w_r$'}
    for ax, panel in zip(axes.flat, 'abcd'):
        subset = [r for r in rows if r['panel']==panel]
        for metric in dict.fromkeys(r['metric'] for r in subset):
            r = sorted([r for r in subset if r['metric']==metric],key=lambda x:float(x['multiplier']))
            ax.errorbar([float(x['multiplier']) for x in r], [float(x['mean_seed_ratio']) for x in r],
                        yerr=[float(x['sample_sd_seed_ratio']) for x in r], marker='o', markersize=3,
                        capsize=1.6, linewidth=.9, label=LABELS.get(metric,metric))
        ax.axhline(1,color='.55',linewidth=.7,linestyle=':')
        ax.set(title=f'({panel}) {names[panel]}',xlabel='Weight multiplier',ylabel='Ratio to nominal')
        ax.grid(alpha=.15)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False,fontsize=7.5)
    _save(fig, output / 'figure_5_weights', plt)
    return [str(p) for p in sorted(output.glob('figure_*'))]


def _save(fig, path, plt):
    for ext in ('pdf','svg','png'):
        fig.savefig(path.with_suffix('.'+ext), dpi=220, bbox_inches='tight')
    plt.close(fig)
