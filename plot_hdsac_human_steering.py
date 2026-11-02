"""Plot recorded HDSAC HUMAN steering without counting replay-prefix copies twice."""
import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent


def load_history(branch):
    plan = json.loads((branch / 'resume_plan.json').read_text())
    if plan['completed_steps'] != 5000 or plan['target_steps'] != 6000:
        raise ValueError('Expected the verified 5000-to-6000 continuation')
    sessions = sorted(p for p in branch.iterdir() if (p / 'snapshot_index.json').is_file())
    parts, provenance, index_steps = [], [], []
    for session in sessions:
        meta = json.loads((session / 'metadata.json').read_text())
        if meta['agent']['variant'] != 'official' or not meta['agent']['online_train']:
            raise ValueError('Expected official HDSAC training')
        arrays = []
        for path in sorted((session / 'chunks').glob('transitions_*.npz')):
            with np.load(path, allow_pickle=False) as chunk:
                arrays.append({key: chunk[key].copy() for key in
                               ('human_action', 'intervention', 'replay_eligible', 'human_controls_saturated')})
        combined = {key: np.concatenate([a[key] for a in arrays]) for key in arrays[0]}
        with (session / 'sensor_manifest.csv').open(encoding='utf-8-sig', newline='') as handle:
            count = sum(1 for _ in csv.DictReader(handle))
        index = json.loads((session / 'snapshot_index.json').read_text())
        offset = index[0].get('training_step_offset', 0)
        if len(combined['intervention']) != count or index[-1]['recorded_training_steps'] != offset + count:
            raise ValueError('Recorded data/snapshot counters differ')
        scale_deg = math.degrees(meta['control_mapping']['human_steering_full_scale_rad'])
        human = combined['intervention'].astype(bool)
        eligible = combined['replay_eligible'].astype(bool)
        parts.append(dict(step=np.arange(offset + 1, offset + count + 1),
                          steering=combined['human_action'][:, 0] * scale_deg,
                          human=human, accepted=human & eligible, omitted=human & ~eligible,
                          saturated=human & combined['human_controls_saturated'].astype(bool)))
        provenance.append(dict(session=str(session), first_step=offset + 1,
                               last_step=offset + count, archived=bool(meta.get('archived_combined_prefix'))))
        index_steps.extend(range(offset + 1, offset + count + 1))
    if index_steps != list(range(1, 6001)):
        raise ValueError('Expected exactly 6000 original recorded decisions without overlapping replay copies')
    data = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    if not np.isfinite(data['steering']).all():
        raise ValueError('Nonfinite recorded steering')
    summary = dict(recorded_steps=6000, human_actions=int(data['human'].sum()),
                   replay_eligible_human=int(data['accepted'].sum()),
                   replay_ineligible_human=int(data['omitted'].sum()),
                   saturated_human=int(data['saturated'].sum()),
                   source_storage=provenance, original_session_boundaries=[4000, 5000],
                   steering_units='Recorded normalized HUMAN action converted to degrees; clipped at +/-200 deg',
                   red_meaning='HUMAN transition not eligible for replay, not an RM36 filter rejection')
    assert summary['replay_eligible_human'] + summary['replay_ineligible_human'] == summary['human_actions']
    return data, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branch-id', default='phuoc8_hdsac_no_calibration_old5000_to6000_v2')
    args = parser.parse_args()
    branch = PROJECT / 'HDSAC_SNAPSHOT100/experiments' / args.branch_id
    data, summary = load_history(branch)
    output = HERE / 'reports' / datetime.now().strftime('hdsac_human_steering6000_%Y%m%d_%H%M%S')
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, ax = plt.subplots(figsize=(18, 6.4), facecolor='#f4f6f8')
    ax.set_facecolor('white')
    for mask, marker, color, size, alpha, label in (
            (data['accepted'], 'o', '#2385ad', 10, .6, 'HUMAN replay-eligible'),
            (data['omitted'], 'x', '#e74747', 24, .8, 'HUMAN replay-ineligible')):
        ax.scatter(data['step'][mask], data['steering'][mask], s=size, marker=marker,
                   color=color, alpha=alpha, linewidths=0 if marker == 'o' else 1.1,
                   label=f'{label} ({mask.sum():,})', zorder=3 if marker == 'o' else 4)
    ax.axhline(0, color='#707070', linewidth=.9)
    for bound in (-200, 200):
        ax.axhline(bound, color='#929292', linestyle='--', linewidth=.9)
    for boundary in summary['original_session_boundaries']:
        ax.axvline(boundary + .5, color='#a3a8ad', linestyle=':', linewidth=1)
    ax.set_xlim(0, 6000)
    ax.set_ylim(-300, 300)
    ax.xaxis.set_major_locator(MultipleLocator(500))
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.grid(color='#d6dade', linewidth=.7, zorder=0)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('bottom', 'left'):
        ax.spines[side].set_color('#a5a9ad')
    ax.set_xlabel('Cumulative recorded online training step', fontsize=11)
    ax.set_ylabel('Recorded HUMAN steering (deg)', fontsize=11)
    ax.set_title('HDSAC - HUMAN actions during online training\n'
                 f'6,000 recorded steps | 3 sessions | {summary["human_actions"]:,} HUMAN actions',
                 loc='left', fontsize=16, weight='bold', pad=14)
    ax.legend(loc='upper right', fontsize=10, framealpha=.95)
    fig.subplots_adjust(left=.065, right=.985, top=.81, bottom=.19)
    fig.text(.065, .05, 'Original HUMAN actions only; AUTO steps and replay-prefill copies are excluded. '
             'Red: replay-ineligible, not an RM36 rejection.\n'
             'Recorded steering is clipped at +/-200 deg; out-of-range physical wheel angles cannot be recovered. '
             'Dotted lines: session boundaries.', fontsize=9, color='#656c73')
    fig.savefig(output / 'hdsac_human_actions_online6000.png', dpi=200, facecolor=fig.get_facecolor())
    fig.savefig(output / 'hdsac_human_actions_online6000.pdf', facecolor=fig.get_facecolor())
    plt.close(fig)
    with (output / 'hdsac_human_actions.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow(['training_step', 'recorded_steering_deg', 'replay_eligible', 'controls_saturated'])
        for i in np.flatnonzero(data['human']):
            writer.writerow([int(data['step'][i]), float(data['steering'][i]),
                             int(data['accepted'][i]), int(data['saturated'][i])])
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(dict(output=str(output), **summary), indent=2))


if __name__ == '__main__':
    main()
