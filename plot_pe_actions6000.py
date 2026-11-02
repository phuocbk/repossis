"""Export original PE-RLHF takeover requests and selected HUMAN/PHYSICS steering."""
import csv
from datetime import datetime
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'PE_RLHF_SCANeR'))
from plot_human_actions import phase_data, draw
from storage import load_chunks


def export(branch):
    data, summary = phase_data(branch, 'train', 6000)
    chunks = []
    for session in summary['sessions']:
        folder = branch / session['session']
        raw = load_chunks(folder)
        meta = json.loads((folder / 'metadata.json').read_text())
        chunks.append(dict(source=raw['source'], eligible=raw['eligible'],
                           executed=raw['executed_actions'][:, 0] * meta['config']['environment']['steering_limit_deg']))
    source = np.concatenate([c['source'] for c in chunks])
    executed = np.concatenate([c['executed'] for c in chunks])
    assert len(source) == 6000 and np.isfinite(executed).all()
    summary['executed_source_counts'] = {str(s): int((source == s).sum()) for s in np.unique(source)}
    summary['branch'] = str(branch)
    out = HERE / 'reports' / datetime.now().strftime('pe_rlhf_actions6000_%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    extent = max(300., float(np.ceil((np.abs(data['steering'][data['human']]).max(initial=0) + 20) / 50) * 50))
    fig, ax = plt.subplots(figsize=(18, 6.4), facecolor='#f4f6f8')
    draw(ax, data, summary, 'online training', extent)
    fig.subplots_adjust(left=.065, right=.985, top=.81, bottom=.19)
    fig.text(.065, .05, 'Original HUMAN takeover requests only; AUTO steps and replay-prefill copies excluded.\n'
             'Red: HUMAN not selected or replay-ineligible; not an RM36 rejection or a ground-truth wrong-action label.',
             fontsize=9, color='#656c73')
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'pe_rlhf_human_actions_online6000.{ext}', dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(18, 6.4), facecolor='#f4f6f8')
    ax.set_facecolor('white')
    for name, color, marker in [('HUMAN', '#2385ad', 'o'), ('PHYSICS', '#e74747', 'x')]:
        mask = source == name
        ax.scatter(data['step'][mask], executed[mask], s=10 if marker == 'o' else 24,
                   marker=marker, color=color, alpha=.65,
                   linewidths=0 if marker == 'o' else 1.1,
                   label=f'{name} selected ({mask.sum():,})')
    ax.axhline(0, color='#707070', linewidth=.9)
    for bound in (-200, 200):
        ax.axhline(bound, color='#929292', linestyle='--', linewidth=.9)
    for session in summary['sessions'][:-1]:
        ax.axvline(session['last_step'] + .5, color='#a3a8ad', linestyle=':', linewidth=1)
    ax.set(xlim=(0, 6000), ylim=(-250, 250), xlabel='Cumulative recorded online training step',
           ylabel='Executed steering (deg)')
    ax.set_xticks(np.arange(0, 6001, 500))
    ax.grid(color='#d6dade', linewidth=.7)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_title('PE-RLHF - Selected HUMAN and PHYSICS actions during online training\n'
                 '6,000 recorded steps | executed steering after control limits',
                 loc='left', fontsize=16, weight='bold', pad=14)
    ax.legend(loc='upper right')
    fig.subplots_adjust(left=.065, right=.985, top=.81, bottom=.19)
    fig.text(.065, .05, 'Only actions whose recorded execution source is HUMAN or PHYSICS are plotted; AGENT steps are excluded.\n'
             'Colors identify execution source, not accept/reject. Dotted lines: session boundaries.', fontsize=9, color='#656c73')
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'pe_rlhf_human_physics_executed6000.{ext}', dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    with (out / 'pe_rlhf_actions6000.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow(['training_step', 'human_takeover_requested', 'human_steering_deg',
                         'human_selected_and_eligible', 'executed_source', 'executed_steering_deg'])
        for i in range(6000):
            writer.writerow([i + 1, int(data['human'][i]), float(data['steering'][i]),
                             int(data['accepted'][i]), str(source[i]), float(executed[i])])
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(dict(output=str(out), **summary), indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branch-id', required=True)
    args = parser.parse_args()
    export(HERE.parent / 'PE_RLHF_SCANeR/experiments' / args.branch_id)
