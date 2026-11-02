"""Plot measured checkpoint evaluation results without running the simulator."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_rows(suite):
    manifest = json.loads((suite / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('mock'):
        raise ValueError('This report requires real evaluation results')
    rows = []
    for task in manifest['tasks']:
        root = suite / task['result_dir']
        result_path = root / 'result.json'
        if not result_path.is_file():
            raise ValueError(f'Missing completed result: {root}')
        result = json.loads(result_path.read_text(encoding='utf-8'))
        if (result.get('status') != 'complete' or result.get('mock') or
                result['completed_episodes'] != manifest['tests'] or
                result['evaluation_horizon_steps'] != manifest['protocol']['horizon_steps'] or
                result['updates_before'] != result['updates_after'] or
                result['checkpoint_sha256'] != task['checkpoint_sha256']):
            raise ValueError(f'Inconsistent evaluation provenance: {root}')
        # Resolve the recorded session locally even if evaluated on another mapped drive.
        session_name = result['session'].replace('\\', '/').rstrip('/').split('/')[-1]
        session = root / session_name
        episodes = json.loads((session / 'evaluation_reward_episodes.json').read_text())
        if len(episodes) != manifest['tests']:
            raise ValueError('Episode count mismatch')
        valid_episodes = {int(ep['episode']): int(ep['steps']) for ep in episodes}
        speeds = {ep: [] for ep in valid_episodes}
        with (session / 'steps.csv').open(newline='', encoding='utf-8-sig') as handle:
            for row in csv.DictReader(handle):
                ep = int(row['episode'])
                if ep in valid_episodes and int(row['episode_step']) <= valid_episodes[ep]:
                    if row['source'] != 'AGENT' or int(row['takeover']) != 0:
                        raise ValueError('Non-AUTO action found')
                    speed = abs(float(row['speed_kmh']))
                    if not np.isfinite(speed):
                        raise ValueError('Nonfinite speed')
                    speeds[ep].append(speed)
        if any(len(speeds[ep]) != steps for ep, steps in valid_episodes.items()):
            raise ValueError(f'Missing step telemetry: {root}')
        returns = [ep['episode_return'] for ep in episodes]
        row = dict(training_steps=task['training_steps'],
                   reward=float(np.mean(returns)),
                   success_pct=100 * sum(ep['goal'] for ep in episodes) / len(episodes),
                   mean_speed_kmh=float(np.mean([np.mean(s) for s in speeds.values()])),
                   offroad_pct=100 * sum(ep['offroad'] for ep in episodes) / len(episodes),
                   mean_episode_steps=float(np.mean([ep['steps'] for ep in episodes])),
                   terminal_reasons=';'.join(ep['terminal_reason'] for ep in episodes),
                   checkpoint=task['checkpoint'])
        if not np.isclose(row['reward'], result['mean_return']):
            raise ValueError('Return differs between summary and episode records')
        rows.append(row)
    return manifest, sorted(rows, key=lambda row: row['training_steps'])


def draw_axis(ax, rows, metric, title, ylabel, algorithm='PE-RLHF'):
    x = [row['training_steps'] for row in rows]
    y = [row[metric] for row in rows]
    color = '#239bce' if algorithm == 'PE-RLHF' else '#272c30'
    ax.plot(x, y, color=color, linewidth=1.7, marker='o', markersize=3.5, label=algorithm)
    ax.set_title(title, loc='left', fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Online training steps')
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, max(x) + 70)
    ax.set_xticks(np.arange(0, max(x) + 1, 1000))
    if metric in ('success_pct', 'offroad_pct'):
        ax.set_ylim(-4, 104)
        ax.set_yticks([0, 25, 50, 75, 100])
    elif metric == 'mean_speed_kmh':
        ax.set_ylim(bottom=0)
    ax.grid(True, alpha=.2)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='best', frameon=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    args = parser.parse_args()
    suite = args.suite.resolve(strict=True)
    manifest, rows = load_rows(suite)
    algorithm = manifest['algorithm']
    prefix = 'pe_rlhf' if algorithm == 'PE-RLHF' else 'rm36'
    if not rows:
        raise ValueError('No measured results')
    output = suite / 'reports' / datetime.now().strftime('four_metrics_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'figure.facecolor': 'white', 'axes.facecolor': 'white'})
    panels = [('reward', 'A. Evaluation Reward', 'Episode return'),
              ('success_pct', 'B. Goal Success', 'Successful episodes (%)'),
              ('mean_speed_kmh', 'C. Mean Vehicle Speed', 'Mean speed (km/h)'),
              ('offroad_pct', 'D. Off-road', 'Off-road episodes (%)')]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (metric, title, ylabel) in zip(axes.flat, panels):
        draw_axis(ax, rows, metric, title, ylabel, algorithm)
    fig.suptitle(f'{algorithm} - AUTO Evaluation Across Training Checkpoints', fontsize=16, fontweight='bold', y=.98)
    count = manifest['tests']
    horizon = manifest['protocol']['horizon_steps']
    fig.text(.5, .925, f'{len(rows)} saved checkpoints | {count} episode per checkpoint | maximum {horizon} steps per episode',
             ha='center', fontsize=10, color='#444444')
    fig.text(.5, .025, 'Reward = delta progress (m) + 0.1 x speed (km/h) / 80 + 20 x terminal goal.\n'
             'Raw measured outcomes; no smoothing. Each checkpoint has one trial, so goal/off-road outcomes are binary.'
             if count == 1 else 'Raw measured checkpoint averages; no smoothing.',
             ha='center', fontsize=9, color='#444444')
    fig.subplots_adjust(left=.085, right=.98, top=.86, bottom=.115, wspace=.25, hspace=.40)
    fig.savefig(output / f'{prefix}_four_metrics.png', dpi=220)
    fig.savefig(output / f'{prefix}_four_metrics.pdf')
    plt.close(fig)
    for metric, title, ylabel in panels:
        fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
        draw_axis(ax, rows, metric, title, ylabel, algorithm)
        fig.savefig(output / f'{prefix}_{metric}.png', dpi=220)
        plt.close(fig)
    with (output / 'plotted_metrics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(source_suite=str(suite), checkpoints=len(rows), episodes_per_checkpoint=count,
                   horizon_steps=horizon,
                   successful_checkpoint_steps=[r['training_steps'] for r in rows if r['success_pct'] == 100],
                   offroad_checkpoint_steps=[r['training_steps'] for r in rows if r['offroad_pct'] == 100],
                   best_reward_checkpoint=max(rows, key=lambda row: row['reward'])['training_steps'],
                   final_checkpoint_return=rows[-1]['reward'])
    (output / 'report_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(dict(output=str(output), **summary), indent=2))


if __name__ == '__main__':
    main()
