"""Plan, run and resume deterministic PE-RLHF checkpoint evaluations."""
import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from run_pe_rlhf_auto import (HERE, PE_ROOT, ScoredJournal, file_hash, pe,
                              PERLHFLearner, OBS_DIM, save_json)


def discover(branch, every):
    candidates = {}
    for session in sorted(branch.glob('train_*')):
        metadata_path = session / 'metadata.json'
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        if metadata.get('mock') or metadata.get('mode') != 'train':
            continue
        for path in sorted(session.glob('*.pt')):
            if path.name != 'pe_rlhf_latest.pt' and not path.name.startswith('actor_critic_step_'):
                continue
            checkpoint = pe.torch.load(path, map_location='cpu', weights_only=False)
            step = checkpoint.get('config', {}).get('online_recorded_steps')
            if not isinstance(step, int) or step <= 0:
                raise ValueError(f'Missing reliable training-step counter: {path}')
            # Prefer the final saved snapshot at a duplicated step.
            rank = (path.name == 'pe_rlhf_latest.pt', path.stat().st_mtime_ns)
            if step not in candidates or rank > candidates[step][0]:
                candidates[step] = (rank, path)
    if not candidates:
        raise ValueError(f'No saved online checkpoints in {branch}')
    final = max(candidates)
    selected = [step for step in sorted(candidates) if step % every == 0 or step == final]
    return [dict(algorithm='PE-RLHF', training_steps=step,
                 checkpoint=str(candidates[step][1].resolve()),
                 checkpoint_sha256=file_hash(candidates[step][1]),
                 result_dir=f'step_{step:06d}') for step in selected]


def wilson(successes, n):
    p = successes / n
    z = 1.96
    center = (p + z*z / (2*n)) / (1 + z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return center - half, center + half


def summarize(episodes):
    n = len(episodes)
    if not n:
        raise ValueError('No completed episodes')
    returns = np.array([row['episode_return'] for row in episodes], dtype=float)
    goals = sum(row['goal'] for row in episodes)
    offroads = sum(row['offroad'] for row in episodes)
    std = float(returns.std(ddof=1)) if n > 1 else None
    return dict(completed_episodes=n, goals=goals, offroads=offroads,
                mean_return=float(returns.mean()), return_std=std,
                return_approx_ci95_half=1.96*std/math.sqrt(n) if std is not None else None,
                success_rate=goals/n, success_ci95=wilson(goals, n),
                offroad_rate=offroads/n, offroad_ci95=wilson(offroads, n))


def render(suite, manifest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = []
    for task in manifest['tasks']:
        result = suite / task['result_dir'] / 'result.json'
        if result.is_file():
            item = json.loads(result.read_text(encoding='utf-8'))
            if item.get('status') == 'complete':
                rows.append(item)
    if not rows:
        return
    rows.sort(key=lambda row: row['training_steps'])
    columns = ['training_steps', 'completed_episodes', 'mean_return', 'return_std',
               'success_rate', 'offroad_rate', 'goals', 'offroads', 'checkpoint']
    with (suite / 'checkpoint_metrics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    x = [row['training_steps'] for row in rows]
    color = '#248fbc'
    for ax, metric, title in zip(axes, ['mean_return', 'success_rate', 'offroad_rate'],
                                ['Episode Return', 'Goal Success Rate', 'Off-road Rate']):
        factor = 1 if metric == 'mean_return' else 100
        y = np.array([row[metric] * factor for row in rows])
        ax.plot(x, y, 'o-', color=color, label='PE-RLHF', markersize=4)
        if metric == 'mean_return':
            half = np.array([row['return_approx_ci95_half'] or 0 for row in rows])
            lower, upper = y-half, y+half
        else:
            key = 'success_ci95' if metric == 'success_rate' else 'offroad_ci95'
            lower, upper = np.array([row[key] for row in rows]).T * 100
            ax.set_ylim(0, 100)
        ax.fill_between(x, lower, upper, color=color, alpha=.15)
        ax.set_title(title)
        ax.set_xlabel('Recorded online training steps')
        ax.set_ylabel('Undiscounted return' if factor == 1 else 'Episodes (%)')
        ax.grid(alpha=.2)
        ax.legend(loc='best')
    prefix = 'MOCK DATA - ' if manifest['mock'] else ''
    fig.suptitle(prefix + f"PE-RLHF AUTO | {manifest['tests']} episodes per checkpoint")
    fig.text(.5, -.035, 'Reward band: approximate 95% mean CI; rate bands: Wilson 95% CI. '
             'Single training run; bands describe evaluation episodes only.', ha='center', fontsize=8)
    fig.savefig(suite / 'checkpoint_curves.png', dpi=180, bbox_inches='tight')
    fig.savefig(suite / 'checkpoint_curves.pdf', bbox_inches='tight')
    plt.close(fig)


class BorrowedAdapter:
    """Allow native evaluation sessions to reuse one SDK connection."""
    def __init__(self, adapter):
        self.adapter = adapter

    def __getattr__(self, name):
        return getattr(self.adapter, name)

    def connect(self):
        return None

    def close(self):
        return None


def execute(suite, manifest):
    if manifest['tests'] < 1 or not manifest['tasks']:
        raise ValueError('Invalid empty evaluation plan')
    protocol = manifest['protocol']
    configs = []
    for task in manifest['tasks']:
        path = Path(task['checkpoint'])
        if file_hash(path) != task['checkpoint_sha256']:
            raise ValueError(f'Checkpoint changed since planning: {path}')
        config = json.loads((path.parent / 'metadata.json').read_text(encoding='utf-8'))['config']
        for field, key in [('frequency_hz', 'frequency_hz'), ('max_speed_kmh', 'max_speed_kmh'),
                           ('stall_sec', 'stall_timeout_sec'),
                           ('goal_region', 'goal_region')]:
            if config['environment'][field] != protocol[key]:
                raise ValueError(f'Protocol mismatch: {field} in {path}')
        # Evaluation horizon is independent of the horizon used during training.
        config['environment']['horizon'] = protocol['horizon_steps']
        if config['environment']['horizon'] < 1:
            raise ValueError('Evaluation horizon must be positive')
        configs.append(config)
    if any(c['environment'] != configs[0]['environment'] for c in configs):
        raise ValueError('Checkpoints have different environment settings; use separate suites')
    pending = []
    for task, config in zip(manifest['tasks'], configs):
        result = suite / task['result_dir'] / 'result.json'
        if result.is_file():
            saved = json.loads(result.read_text(encoding='utf-8'))
            if saved.get('status') == 'complete':
                if saved['checkpoint_sha256'] != task['checkpoint_sha256']:
                    raise ValueError('Completed result has a different checkpoint hash')
                if (saved['completed_episodes'] != manifest['tests'] or
                        saved.get('evaluation_horizon_steps') != protocol['horizon_steps']):
                    raise ValueError('Existing results use another episode count/horizon; create a new suite')
                continue
        pending.append((task, config))
    if not pending:
        render(suite, manifest)
        print('All checkpoints already evaluated; plots refreshed.')
        return
    pe.torch.set_num_threads(2)
    factory = pe.ScanerAdapter
    journal_factory = pe.Journal
    cfg = configs[0]['environment']
    adapter = (pe.MockAdapter(cfg, pe.Route(cfg['route'])) if manifest['mock'] else
               factory(cfg, PE_ROOT.parent))
    try:
        adapter.connect()
        pe.ScanerAdapter = lambda *args, **kwargs: BorrowedAdapter(adapter)
        pe.Journal = ScoredJournal
        for task, config in pending:
            root = suite / task['result_dir']
            root.mkdir(parents=True, exist_ok=True)
            print(f"[SWEEP] step={task['training_steps']} tests={manifest['tests']}", flush=True)
            learner = PERLHFLearner(OBS_DIM, config['learning'])
            learner.load(task['checkpoint'])
            updates_before = learner.updates
            options = SimpleNamespace(mock=manifest['mock'], checkpoint=task['checkpoint'],
                                      tests=manifest['tests'], experimental_selector=False)
            sessions_before = set(root.glob('eval_*'))
            try:
                pe.collect(config, options, root, 'eval', learner,
                           budget=manifest['tests'] * cfg['horizon'])
                sessions = set(root.glob('eval_*')) - sessions_before
                session = next(iter(sessions))
                episodes_file = session / 'evaluation_reward_episodes.json'
                episodes = json.loads(episodes_file.read_text()) if episodes_file.exists() else []
                if len(episodes) != manifest['tests']:
                    raise RuntimeError('Evaluation interrupted or incomplete; resume reruns this checkpoint')
                if learner.updates != updates_before or file_hash(task['checkpoint']) != task['checkpoint_sha256']:
                    raise RuntimeError('Checkpoint/update count changed during evaluation')
                result = dict(task, status='complete', mock=manifest['mock'],
                              session=str(session), updates_before=updates_before,
                              evaluation_horizon_steps=cfg['horizon'],
                              updates_after=learner.updates, **summarize(episodes))
                save_json(root / 'result.json', result)
                render(suite, manifest)
            except BaseException as exc:
                save_json(root / 'last_failure.json', dict(error=type(exc).__name__, detail=str(exc)))
                raise
    finally:
        pe.ScanerAdapter = factory
        pe.Journal = journal_factory
        render(suite, manifest)
        try:
            adapter.close()
        except KeyboardInterrupt:
            print('SDK close interrupted; completed evaluation results retained.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branch-id', default='phuoc8_pe_rlhf_online4000_noise_retest_v3')
    parser.add_argument('--every', type=int, default=1)
    parser.add_argument('--tests', type=int, default=1)
    parser.add_argument('--max-episode-steps', type=int, default=None,
                        help='Evaluation horizon; default 200 for new suites, saved horizon on resume')
    parser.add_argument('--suite', help='Resume or run an existing suite folder')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--mock', action='store_true', help='Synthetic verification only; never research results')
    args = parser.parse_args()
    if args.every < 1 or args.tests < 1 or (args.max_episode_steps is not None and args.max_episode_steps < 1):
        parser.error('Checkpoint interval, episode count and horizon must be positive')
    if args.suite:
        suite = Path(args.suite).resolve(strict=True)
        manifest = json.loads((suite / 'manifest.json').read_text())
        if args.max_episode_steps is not None and args.max_episode_steps != manifest['protocol']['horizon_steps']:
            parser.error('Saved suite uses another horizon; create a new suite without --suite')
    else:
        if Path(args.branch_id).name != args.branch_id or any(c not in
                'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.branch_id):
            parser.error('Invalid branch id')
        branch = PE_ROOT / 'experiments' / args.branch_id
        tasks = discover(branch, args.every)
        suite = HERE / 'results' / ('mock_sweeps' if args.mock else 'sweeps') / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        suite.mkdir(parents=True, exist_ok=False)
        manifest = dict(algorithm='PE-RLHF', branch_id=args.branch_id, tests=args.tests,
                        every=args.every, mock=args.mock, tasks=tasks,
                        online_step_axis_excludes_warmup=True,
                        protocol=json.loads((HERE / 'protocol.json').read_text()))
        manifest['protocol']['horizon_steps'] = args.max_episode_steps or 200
        manifest['protocol']['episodes_per_algorithm'] = args.tests
        save_json(suite / 'manifest.json', manifest)
    print(f'[SWEEP] suite={suite}', flush=True)
    print('[SWEEP] training steps=' + ', '.join(str(t['training_steps']) for t in manifest['tasks']), flush=True)
    print(f"[SWEEP] max episode steps={manifest['protocol']['horizon_steps']}", flush=True)
    if args.plot_only:
        render(suite, manifest)
    elif not args.plan_only:
        execute(suite, manifest)


if __name__ == '__main__':
    main()
