"""Plot one flattened RM36 lineage without counting replay-prefix copies twice."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def load_history(branch):
    plan = json.loads((branch / 'resume_plan.json').read_text())
    keys = ('intervention', 'rm36_steering_deg', 'rm36_rejected',
            'replay_eligible', 'rm36_reference_deg', 'rm36_active',
            'rm36_reason', 'scenario_id')
    parts, sources = [], []
    sessions = sorted(p for p in branch.iterdir() if (p / 'metadata.json').is_file())
    for session in sessions:
        meta = json.loads((session / 'metadata.json').read_text())
        arrays = []
        for path in sorted((session / 'chunks').glob('transitions_*.npz')):
            with np.load(path, allow_pickle=False) as chunk:
                arrays.append({key: chunk[key].copy() for key in keys})
        combined = {k: np.concatenate([a[k] for a in arrays]) for k in keys}
        with (session / 'sensor_manifest.csv').open(encoding='utf-8-sig') as handle:
            count = sum(1 for _ in csv.DictReader(handle))
        if count != len(combined['intervention']):
            raise ValueError('Manifest/replay count differs')
        offset = 0 if meta.get('archived_combined_prefix') else plan['completed_steps']
        combined['step'] = np.arange(offset + 1, offset + count + 1)
        parts.append(combined)
        sources.append(str(session))
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    if not np.array_equal(data['step'], np.arange(1, plan['target_steps'] + 1)):
        raise ValueError('Missing or duplicate original decisions')
    human = data['intervention'].astype(bool)
    rejected = data['rm36_rejected'].astype(bool) & human
    if not np.isfinite(data['rm36_steering_deg'][human]).all():
        raise ValueError('Nonfinite HUMAN steering')
    boundaries = (np.flatnonzero(data['scenario_id'][1:] != data['scenario_id'][:-1]) + 1).tolist()
    summary = dict(recorded_steps=len(human), human_actions=int(human.sum()),
                   accepted_human=int((human & ~rejected).sum()),
                   rejected_human=int(rejected.sum()),
                   reject_rate_pct=100 * float(rejected.sum()) / max(1, int(human.sum())),
                   accepted_but_replay_ineligible=int((human & ~rejected & ~data['replay_eligible'].astype(bool)).sum()),
                   session_boundaries=boundaries, source_storage=sources,
                   blue_meaning='RM36 accepted; other replay-quality gates may still exclude a transition',
                   red_meaning='Recorded RM36 rejection, not an inferred wrong-action ground truth')
    return data, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branch-id', required=True)
    args = parser.parse_args()
    data, summary = load_history(HERE.parent / 'RM36/experiments' / args.branch_id)
    out = HERE / 'reports' / datetime.now().strftime('rm36_human_steering_%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    human = data['intervention'].astype(bool)
    rejected = data['rm36_rejected'].astype(bool)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, ax = plt.subplots(figsize=(18, 6.4), facecolor='#f4f6f8')
    ax.set_facecolor('white')
    for mask, color, marker, size, label in (
            (human & ~rejected, '#2385ad', 'o', 10, 'RM36 accepted HUMAN action'),
            (human & rejected, '#e74747', 'x', 24, 'RM36 rejected HUMAN action')):
        ax.scatter(data['step'][mask], data['rm36_steering_deg'][mask],
                   color=color, marker=marker, s=size, alpha=.65 if marker == 'o' else .8,
                   linewidths=0 if marker == 'o' else 1.1, label=f'{label} ({mask.sum():,})', zorder=3)
    ax.axhline(0, color='#707070', linewidth=.9)
    for limit in (-200, 200):
        ax.axhline(limit, color='#929292', linestyle='--', linewidth=.9)
    for boundary in summary['session_boundaries']:
        ax.axvline(boundary + .5, color='#a3a8ad', linestyle=':', linewidth=1)
    extent = max(300, np.ceil((np.abs(data['rm36_steering_deg'][human]).max(initial=0) + 20) / 100) * 100)
    ax.set(xlim=(0, summary['recorded_steps']), ylim=(-extent, extent),
           xlabel='Cumulative recorded online training step', ylabel='Raw HUMAN steering (deg)')
    ax.set_xticks(np.arange(0, summary['recorded_steps'] + 1, 500))
    ax.grid(color='#d6dade', linewidth=.7)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_title('RM36 - HUMAN actions during online training\n'
                 f'{summary["recorded_steps"]:,} recorded steps | {summary["human_actions"]:,} HUMAN actions | '
                 f'{summary["rejected_human"]:,} rejected ({summary["reject_rate_pct"]:.1f}%)',
                 loc='left', fontsize=16, weight='bold', pad=14)
    ax.legend(loc='upper right', framealpha=.95)
    fig.subplots_adjust(left=.065, right=.985, top=.81, bottom=.19)
    fig.text(.065, .045, 'Original HUMAN actions only; AUTO steps and replay-prefill copies are excluded. Dotted lines: session boundaries.\n'
             'Blue: accepted by RM36 (not necessarily replay-eligible). Red: rejected by RM36; not a ground-truth wrong-action label.',
             fontsize=9, color='#656c73')
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'rm36_human_actions_online6000.{ext}', dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    columns = ('step', 'rm36_steering_deg', 'rm36_rejected', 'replay_eligible',
               'rm36_reference_deg', 'rm36_active', 'rm36_reason')
    with (out / 'rm36_human_actions.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for i in np.flatnonzero(human):
            writer.writerow([data[k][i].item() for k in columns])
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(dict(output=str(out), **summary), indent=2))


if __name__ == '__main__':
    main()
