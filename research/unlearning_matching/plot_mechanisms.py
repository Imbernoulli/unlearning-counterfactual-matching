"""Render calibration-only mechanism evidence; never opens audit/test artifacts."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def ecdf(ax, values, label, color, linestyle='-'):
    ordered = np.sort(values)
    ax.step(ordered, np.arange(1, len(ordered) + 1) / len(ordered),
            where='post', label=label, color=color, linestyle=linestyle, linewidth=1.6)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    root = a.artifacts / 'out'
    run_a = root / 'optimization/r1_A_inherited_lr003_a01/seed42_split42/A'
    run_c = root / 'optimization/r1_C_frozen_lr003_a01/seed42_split42/C'
    probe_a = json.loads((root / 'mechanism/probe_A_epoch10/seed42_split42/mechanism.json').read_text())
    probe_c = json.loads((root / 'mechanism/probe_Cf_epoch2/seed42_split42/mechanism.json').read_text())
    assert all(x['audit_predicted'] is False and x['test_used'] is False for x in (probe_a, probe_c))
    records = [json.loads(line) for line in (run_a / 'history.jsonl').read_text().splitlines()]
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axs = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    colors = {'forget': '#B33A3A', 'selection': '#2768A6', 'A': '#E39435', 'C': '#249B7B'}
    ax = axs[0, 0]
    for stage, label, col in [('after_forget', 'After forget pass', colors['forget']),
                              ('after_retain', 'After retain pass', colors['selection'])]:
        rows = [r for r in records if r['stage'] == stage]
        ax.plot([r['epoch'] for r in rows], [100*r['metrics']['forget_selection_confidence_ks'] for r in rows],
                marker='o', markersize=3, label=label, color=col)
    ax.set(title='(a) Retain passes partially undo matching', xlabel='Epoch', ylabel='Forget–selection confidence KS (%)')
    ax.legend(frameon=False, fontsize=9)
    ax = axs[0, 1]
    for name, linestyle, label in [('epoch000_initial_distributions.npz', '--', 'Original'),
                                  ('epoch002_epoch_end_distributions.npz', '-', 'Joint + fixed BN, epoch 2')]:
        with np.load(run_c / name, allow_pickle=False) as z:
            for group in ['forget', 'selection']:
                ecdf(ax, z[f'{group}_logodds'], f'{label}: {group}', colors[group], linestyle)
    ax.set(title='(b) Forget and selection log-odds', xlabel='True-class log-odds', ylabel='Empirical CDF', xlim=(-10, 16))
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    ax = axs[1, 0]
    blocks = ['stem', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']
    yy = np.arange(len(blocks))
    for offset, probe, model, col in [(-.25, probe_c, 'original', '#888888'),
                                     (0, probe_a, 'candidate', colors['A']),
                                     (.25, probe_c, 'candidate', colors['C'])]:
        vals = [probe['gradients'][model]['blocks'][b]['cosine'] for b in blocks]
        label = {'original': 'Original', 'candidate': 'Sequential, epoch 10' if probe is probe_a else 'Joint + fixed BN, epoch 2'}[model]
        ax.barh(yy + offset, vals, height=.23, label=label, color=col)
    ax.axvline(0, color='black', linewidth=.7)
    ax.set(yticks=yy, yticklabels=blocks, xlabel='Cosine: forget-target vs retain-CE parameter gradients', title='(c) Layerwise gradient conflict', xlim=(-1, 1))
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    ax = axs[1, 1]
    keys = ['original_feature_original_head', 'original_feature_candidate_head',
            'candidate_feature_original_head', 'candidate_feature_candidate_head']
    labels = ['Original', 'New head\nonly', 'New features\nonly', 'New features\n+ head']
    for shift, group, col in [(-.17, 'forget', colors['forget']), (.17, 'selection', colors['selection'])]:
        comb = probe_c['groups'][group]['head_feature_combinations']
        ax.bar(np.arange(4)+shift, [comb[k]['error_pct'] for k in keys], width=.32, label=group, color=col)
    ax.set(xticks=np.arange(4), xticklabels=labels, ylabel='Classification error (%)', title='(d) Head vs. feature intervention')
    ax.tick_params(axis='x', labelsize=8)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(a.out_dir / 'mechanism_probes.pdf')
    fig.savefig(a.out_dir / 'mechanism_probes.png', dpi=180)
    (a.out_dir / 'caption.md').write_text(
        'Calibration-only probes, seed 42; audit/test/Retrain predictions were not read. '
        'Panels (a,b) use the full 4,500 forget / 1,250 selection sets; '
        '(c) uses fixed clean batches of 256 per objective; '
        '(d) uses fixed subsets of 1,024 per group. Gradient cosines are unweighted '
        'parameter-gradient measurements, with no optimizer momentum. Head/feature '
        'hybrids intervene on already trained components and do not prove that '
        'training a different head alone would fail. The compared checkpoints '
        'also differ in schedule, anchor, BN handling, and epoch. These are '
        'descriptive probes, not causal isolation of one component.\n')


if __name__ == '__main__':
    main()
