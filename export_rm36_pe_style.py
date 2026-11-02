"""Export measured RM36 results using the existing blue PE/HDSAC figure template."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path

from plot_four_metrics import load_rows, draw_axis, plt


def export(suite):
    manifest, rows = load_rows(suite)
    if manifest['algorithm'] != 'RM36' or not rows:
        raise ValueError('Expected completed RM36 evaluation results')
    output = Path(__file__).resolve().parent / 'reports' / datetime.now().strftime('rm36_full6000_pe_style_%Y%m%d_%H%M%S')
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'figure.facecolor': 'white', 'axes.facecolor': 'white'})
    panels = [('reward', 'A. Evaluation Reward', 'Episode return'),
              ('success_pct', 'B. Goal Success', 'Successful episodes (%)'),
              ('mean_speed_kmh', 'C. Mean Vehicle Speed', 'Mean speed (km/h)'),
              ('offroad_pct', 'D. Off-road', 'Off-road episodes (%)')]

    def draw(ax, metric, title, ylabel):
        draw_axis(ax, rows, metric, title, ylabel, 'RM36')
        for line in ax.lines:
            line.set_color('#239bce')
        ax.legend(loc='best', frameon=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (metric, title, ylabel) in zip(axes.flat, panels):
        draw(ax, metric, title, ylabel)
    fig.suptitle('RM36 - AUTO Evaluation Across Training Checkpoints', fontsize=16, fontweight='bold', y=.98)
    fig.text(.5, .925, f'{len(rows)} saved checkpoints | {manifest["tests"]} episode per checkpoint | '
             f'maximum {manifest["protocol"]["horizon_steps"]} steps per episode',
             ha='center', fontsize=10, color='#444444')
    fig.text(.5, .025, 'Reward = delta progress (m) + 0.1 x speed (km/h) / 80 + 20 x terminal goal.\n'
             'Raw measured outcomes; no smoothing. Each checkpoint has one trial, so goal/off-road outcomes are binary.',
             ha='center', fontsize=9, color='#444444')
    fig.subplots_adjust(left=.085, right=.98, top=.86, bottom=.115, wspace=.25, hspace=.40)
    for ext in ('png', 'pdf'):
        fig.savefig(output / f'rm36_four_metrics.{ext}', dpi=220)
    plt.close(fig)
    for metric, title, ylabel in panels:
        fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
        draw(ax, metric, title, ylabel)
        name = 'total_reward' if metric == 'reward' else metric
        for ext in ('png', 'pdf'):
            fig.savefig(output / f'rm36_{name}.{ext}', dpi=220)
        plt.close(fig)
    with (output / 'rm36_checkpoint_scores.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(source_suite=str(suite), output=str(output), checkpoints=len(rows),
                   final_reward=rows[-1]['reward'], final_success_pct=rows[-1]['success_pct'],
                   style='DejaVu Sans 10, #239bce, 12x8 combined, 7x4.2 individual', smoothing=False)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    export(parser.parse_args().suite.resolve(strict=True))
