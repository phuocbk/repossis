"""Render the retained online HUMAN actions in the original scatter template."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PE_ROOT = HERE.parent / 'PE_RLHF_SCANeR'
sys.path.insert(0, str(PE_ROOT))
from plot_human_actions import phase_data, draw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branch-id', default='phuoc8_pe_rlhf_online4000_noise_retest_v3')
    args = parser.parse_args()
    branch = PE_ROOT / 'experiments' / args.branch_id
    data, summary = phase_data(branch, 'train', 4000)
    summary['branch'] = args.branch_id
    summary['red_meaning'] = 'HUMAN not selected by PE-HAI or replay-ineligible; not an RM36 rejection label'
    values = np.abs(data['steering'][data['human']])
    limit = max(300., float(np.ceil((values.max(initial=0) + 20) / 50) * 50))
    output = HERE / 'reports' / args.branch_id / datetime.now().strftime('human_steering_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, ax = plt.subplots(figsize=(18, 6.4), facecolor='#f4f6f8')
    draw(ax, data, summary, 'online training', limit)
    if len(summary['sessions']) == 1:
        ax.set_title(ax.get_title(loc='left').replace('1 sessions', '1 session'), loc='left',
                     fontsize=16, weight='bold', pad=14)
    fig.subplots_adjust(left=.065, right=.985, top=.81, bottom=.19)
    fig.text(.065, .05, 'Only HUMAN takeover requests are plotted; AUTO steps and replay-prefill copies are excluded. '
             'Red includes PE-HAI choosing PHYSICS or an ineligible action; it is not an RM36 rejection label.',
             fontsize=9, color='#656c73')
    fig.savefig(output / 'pe_rlhf_human_actions_online4000.png', dpi=200, facecolor=fig.get_facecolor())
    fig.savefig(output / 'pe_rlhf_human_actions_online4000.pdf', facecolor=fig.get_facecolor())
    plt.close(fig)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(dict(output=str(output), **summary), indent=2))


if __name__ == '__main__':
    main()
