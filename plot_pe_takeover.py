"""Plot recorded takeover requests and executed HUMAN control during online training."""
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


def rolling(values, window):
    sums = np.concatenate(([0.], np.cumsum(values)))
    ends = np.arange(1, len(values)+1)
    starts = np.maximum(0, ends-window)
    return (sums[ends]-sums[starts]) / (ends-starts) * 100


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branch-id', default='phuoc8_pe_rlhf_online4000_noise_retest_v3')
    args = parser.parse_args()
    branch = HERE.parent / 'PE_RLHF_SCANeR/experiments' / args.branch_id
    parts, sessions, count = [], [], 0
    for session in sorted(branch.glob('train_*')):
        meta = json.loads((session / 'metadata.json').read_text())
        if meta.get('mock'):
            continue
        for chunk in sorted(session.glob('transitions_*.npz')):
            with np.load(chunk, allow_pickle=False) as data:
                parts.append({k: data[k].copy() for k in ('takeover', 'source', 'eligible')})
        last = sum(len(p['takeover']) for p in parts)
        sessions.append(dict(name=session.name, first_step=count+1, last_step=last))
        count = last
    if count != 4000:
        raise ValueError(f'Expected exactly 4000 original online transitions, found {count}')
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    requested = data['takeover'].astype(bool)
    human = data['source'] == 'HUMAN'
    physics = data['source'] == 'PHYSICS'
    x = np.arange(1, count+1)
    summary = dict(branch=args.branch_id, recorded_steps=count,
                   takeover_request_steps=int(requested.sum()), takeover_request_pct=float(requested.mean()*100),
                   human_selected_steps=int(human.sum()), human_selected_pct=float(human.mean()*100),
                   physics_selected_steps=int(physics.sum()), physics_selected_pct=float(physics.mean()*100),
                   human_selected_replay_eligible_steps=int((human & data['eligible'].astype(bool)).sum()),
                   takeover_starts=int((requested & ~np.concatenate(([False], requested[:-1]))).sum()),
                   last250_request_pct=float(requested[-250:].mean()*100), sessions=sessions)
    output = HERE / 'reports' / args.branch_id / datetime.now().strftime('takeover_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.8), sharex=True)
    ax = axes[0]
    ax.plot(x, rolling(requested, 250), color='#238bb8', label='Rolling request rate (250 steps)', linewidth=1.7)
    ax.plot(x, np.cumsum(requested)/x*100, color='#8a9296', label='Cumulative request rate', linewidth=1.2)
    ax.axhline(requested.mean()*100, color='#238bb8', linestyle='--', alpha=.6,
               label=f'Overall requests: {requested.mean()*100:.1f}%')
    ax.set_title(f"A. HUMAN takeover requests | {requested.sum():,} / {count:,} steps", loc='left', fontweight='bold')
    ax = axes[1]
    ax.plot(x, rolling(human, 250), color='#27895f', label='HUMAN selected (rolling 250)', linewidth=1.7)
    ax.plot(x, rolling(physics, 250), color='#c37a37', label='PHYSICS selected (rolling 250)', linewidth=1.5)
    ax.set_title(f'B. Executed intervention source | HUMAN {human.mean()*100:.1f}% | PHYSICS {physics.mean()*100:.1f}%',
                 loc='left', fontweight='bold')
    for ax in axes:
        ax.set_ylim(0, 100)
        ax.set_xlim(0, count)
        ax.set_ylabel('Recorded steps (%)')
        ax.grid(alpha=.2)
        ax.legend(loc='upper right', fontsize=8, framealpha=.9)
        ax.spines[['top', 'right']].set_visible(False)
        for session in sessions[:-1]:
            ax.axvline(session['last_step'], linestyle=':', color='gray')
    axes[1].set_xlabel('Recorded online training steps')
    fig.suptitle('PE-RLHF - HUMAN Takeover During 4,000 Online Training Steps', fontsize=15, fontweight='bold')
    fig.subplots_adjust(top=.87, bottom=.14, left=.075, right=.985, hspace=.36)
    fig.text(.075, .035, 'Requests measure operator intervention time. PE-HAI can execute PHYSICS instead of HUMAN.\n'
             'Warm-up, evaluation episodes and replay-prefill copies are excluded. Initial rolling windows use available samples.', fontsize=9)
    fig.savefig(output / 'pe_rlhf_takeover.png', dpi=200)
    fig.savefig(output / 'pe_rlhf_takeover.pdf')
    plt.close(fig)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with (output / 'takeover_steps.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['training_step', 'requested', 'human_selected', 'physics_selected', 'request_rolling250_pct'])
        writer.writerows(zip(x, requested.astype(int), human.astype(int), physics.astype(int), rolling(requested, 250)))
    print(json.dumps(dict(output=str(output), **summary), indent=2))


if __name__ == '__main__':
    main()
