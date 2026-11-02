"""Combine measured HDSAC lineage results and export the existing PE-RLHF plot style."""
import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path

from checkpoint_sweep import file_hash
from plot_four_metrics import load_rows, draw_axis, plt

HERE = Path(__file__).resolve().parent


def export(suites):
    loaded = [(suite, *load_rows(suite)) for suite in suites]
    reference = loaded[0][1]
    steps = [row['training_steps'] for _, _, rows in loaded for row in rows]
    if steps != list(range(100, 6001, 100)):
        raise ValueError('Expected exactly 60 measured checkpoints, 100 through 6000')
    for _, manifest, _ in loaded:
        if (manifest['algorithm'] != 'HDSAC' or manifest['tests'] != 1 or
                manifest['config'] != reference['config'] or manifest['protocol'] != reference['protocol']):
            raise ValueError('Evaluation protocols differ')
    first_final = reference['tasks'][-1]
    continuation = loaded[1][1]['tasks'][-1]
    meta = json.loads((Path(continuation['checkpoint']).parent / 'metadata.json').read_text())
    if file_hash(meta['agent']['checkpoint_loaded']) != first_final['checkpoint_sha256']:
        raise ValueError('The 6000-step continuation did not start from the evaluated old 5000 checkpoint')
    output = HERE / 'reports' / datetime.now().strftime('hdsac_full6000_pe_style_%Y%m%d_%H%M%S')
    output.mkdir(parents=True, exist_ok=False)
    tasks = []
    for suite, manifest, _ in loaded:
        for task in manifest['tasks']:
            tasks.append(dict(task, result_dir=os.path.relpath(suite / task['result_dir'], output)))
    combined = dict(reference, tasks=tasks, branch_id='original_5000_plus_verified_6000_continuation',
                    source_suites=[str(suite) for suite in suites],
                    source_manifest_sha256=[file_hash(suite / 'manifest.json') for suite in suites])
    (output / 'manifest.json').write_text(json.dumps(combined, indent=2), encoding='utf-8')
    _, rows = load_rows(output)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'figure.facecolor': 'white', 'axes.facecolor': 'white'})
    panels = [('reward', 'A. Evaluation Reward', 'Episode return'),
              ('success_pct', 'B. Goal Success', 'Successful episodes (%)'),
              ('mean_speed_kmh', 'C. Mean Vehicle Speed', 'Mean speed (km/h)'),
              ('offroad_pct', 'D. Off-road', 'Off-road episodes (%)')]

    def draw(ax, metric, title, ylabel):
        draw_axis(ax, rows, metric, title, ylabel, 'HDSAC')
        for line in ax.lines:
            line.set_color('#239bce')
        ax.legend(loc='best', frameon=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (metric, title, ylabel) in zip(axes.flat, panels):
        draw(ax, metric, title, ylabel)
    fig.suptitle('HDSAC - AUTO Evaluation Across Training Checkpoints',
                 fontsize=16, fontweight='bold', y=.98)
    fig.text(.5, .925, '60 saved checkpoints | 1 episode per checkpoint | maximum 200 steps per episode',
             ha='center', fontsize=10, color='#444444')
    fig.text(.5, .025, 'Reward = delta progress (m) + 0.1 x speed (km/h) / 80 + 20 x terminal goal.\n'
             'Raw measured outcomes; no smoothing. Each checkpoint has one trial, so goal/off-road outcomes are binary.',
             ha='center', fontsize=9, color='#444444')
    fig.subplots_adjust(left=.085, right=.98, top=.86, bottom=.115, wspace=.25, hspace=.40)
    fig.savefig(output / 'hdsac_four_metrics.png', dpi=220)
    fig.savefig(output / 'hdsac_four_metrics.pdf')
    plt.close(fig)
    for metric, title, ylabel in panels:
        fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
        draw(ax, metric, title, ylabel)
        filename = 'total_reward' if metric == 'reward' else metric
        fig.savefig(output / f'hdsac_{filename}.png', dpi=220)
        fig.savefig(output / f'hdsac_{filename}.pdf')
        plt.close(fig)
    with (output / 'hdsac_checkpoint_scores.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(output=str(output), checkpoints=len(rows), smoothing=False,
                   style='PE-RLHF template: DejaVu Sans 10, #239bce, 12x8 combined, 7x4.2 individual',
                   final_reward=rows[-1]['reward'], final_success_pct=rows[-1]['success_pct'],
                   final_terminal_reason=rows[-1]['terminal_reasons'])
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, action='append', required=True)
    args = parser.parse_args()
    if len(args.suite) != 2:
        parser.error('Specify the old 5000-step suite and its 6000-step continuation suite')
    export([suite.resolve(strict=True) for suite in args.suite])
