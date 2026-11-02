"""Export the same four evaluation metrics as PE-RLHF, with an HDSAC score table."""
import argparse
import csv
import json
from pathlib import Path

from plot_four_metrics import load_rows, draw_axis, plt


def report(suite):
    manifest, rows = load_rows(suite)
    if manifest['algorithm'] != 'HDSAC' or not rows:
        raise ValueError('Expected completed real HDSAC evaluation results')
    by_step = {task['training_steps']: task for task in manifest['tasks']}
    for row in rows:
        root = suite / by_step[row['training_steps']]['result_dir']
        result = json.loads((root / 'result.json').read_text())
        name = result['session'].replace('\\', '/').rstrip('/').split('/')[-1]
        episodes = json.loads((root / name / 'evaluation_reward_episodes.json').read_text())
        for key in ('reward_driving_sum', 'reward_speed_sum', 'reward_destination_sum', 'lane_invasion_rate'):
            row[key] = sum(ep[key] for ep in episodes) / len(episodes)
    output = suite / 'reports'
    output.mkdir(exist_ok=True)
    with (output / 'hdsac_checkpoint_scores.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    panels = [('reward', 'A. Evaluation Reward', 'Episode return'),
              ('success_pct', 'B. Goal Success', 'Successful episodes (%)'),
              ('mean_speed_kmh', 'C. Mean Vehicle Speed', 'Mean speed (km/h)'),
              ('offroad_pct', 'D. Off-road', 'Off-road episodes (%)')]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (metric, title, ylabel) in zip(axes.flat, panels):
        draw_axis(ax, rows, metric, title, ylabel, 'HDSAC')
    fig.suptitle('HDSAC - AUTO Evaluation Across Training Checkpoints', fontsize=15, fontweight='bold')
    fig.text(.5, .025, f'{len(rows)} checkpoints | 1 AUTO trial per checkpoint | '
             f'maximum {manifest["protocol"]["horizon_steps"]} steps per trial\n'
             'Reward = delta progress (m) + 0.1 x speed (km/h) / 80 + 20 x terminal goal.\n'
             'Measured outcomes only; no smoothing. Single-trial success/off-road outcomes are binary.',
             ha='center', fontsize=9)
    fig.subplots_adjust(left=.085, right=.98, top=.90, bottom=.15, wspace=.25, hspace=.42)
    fig.savefig(output / 'hdsac_four_metrics.png', dpi=220)
    fig.savefig(output / 'hdsac_four_metrics.pdf')
    plt.close(fig)
    for metric, title, ylabel in panels:
        fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
        draw_axis(ax, rows, metric, title, ylabel, 'HDSAC')
        fig.savefig(output / f'hdsac_{metric}.png', dpi=220)
        plt.close(fig)
    print(json.dumps(dict(output=str(output), checkpoints=len(rows),
                          final_training_steps=rows[-1]['training_steps'],
                          final_reward=rows[-1]['reward']), indent=2))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    args = parser.parse_args()
    report(args.suite.resolve(strict=True))
