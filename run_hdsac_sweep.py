"""Evaluate all retained HDSAC snapshots under the unchanged PE-RLHF AUTO protocol."""
import argparse
import copy
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
import sys

from reward import EpisodeReward
from checkpoint_sweep import HERE, PE_ROOT, pe, file_hash, save_json, summarize
import numpy as np

PROJECT = HERE.parent
sys.path.append(str(PROJECT))
from hdsac_official import OfficialActor
from collect_hdsac_score_eval import observation, ScenarioPolylineRoute


class FrozenHDSAC:
    def __init__(self, path):
        data = pe.torch.load(path, map_location='cpu', weights_only=False)
        if data.get('schema') != 'hdsac_official_aligned_v1':
            raise ValueError('Unexpected HDSAC checkpoint schema')
        self.actor = OfficialActor(int(data['state_dim']), hidden_dim=int(data['hidden_dim']))
        self.actor.load_state_dict(data['actor'])
        self.actor.eval().requires_grad_(False)
        self.low = np.asarray(data['action_low'], dtype=np.float32)
        self.high = np.asarray(data['action_high'], dtype=np.float32)
        if data.get('action_scaling') != 'zero_centered_piecewise_v1' and not np.allclose(self.low, -self.high):
            raise ValueError('Unsupported asymmetric legacy action scaling')
        self.updates = int(data['step'])

    def act(self, obs):
        with pe.torch.inference_mode():
            mean, _ = self.actor(pe.torch.as_tensor(obs, dtype=pe.torch.float32)[None])
            unit = pe.torch.tanh(mean)[0].numpy()
        return unit * np.where(unit >= 0, self.high, -self.low)


def validate_metadata(metadata, config):
    env = config['environment']
    schema = metadata['observation_schema']
    if schema['name'] != 'path100' or schema['state_dim'] != 444:
        raise ValueError('This evaluator supports retained path100 HDSAC checkpoints only')
    if metadata['scenario_file'] != env['scenario'] or metadata['route']['source'] != 'waypoints':
        raise ValueError('HDSAC scenario/route does not match the shared protocol')
    points = np.array([[float(x) for x in point.split(',')]
                       for point in metadata['route']['waypoints'].split(';')])
    if not np.allclose(points, env['route'], atol=1e-3, rtol=0):
        raise ValueError('HDSAC reference route differs from the shared evaluation route')
    mapping = metadata['control_mapping']
    if (mapping['mechanical_steering_sync'] or
            not math.isclose(mapping['agent_steering_full_scale_rad'], math.radians(200), abs_tol=1e-6) or
            mapping['brake_full_scale'] != env['brake_full_scale']):
        raise ValueError('HDSAC requires another actuator mapping')


def plan(branch, horizon):
    config = json.loads((PE_ROOT / 'config.json').read_text())
    protocol = json.loads((HERE / 'protocol.json').read_text())
    config['environment']['horizon'] = horizon
    protocol.update(horizon_steps=horizon, episodes_per_algorithm=1,
                    purpose='HDSAC checkpoint evaluation; same reward/protocol as PE-RLHF',
                    algorithms=['HDSAC'], status='planned')
    sessions = sorted(p for p in branch.iterdir() if (p / 'hdsac_online_latest.pt').is_file())
    tasks, cumulative, previous = [], 0, None
    for session in sessions:
        metadata = json.loads((session / 'metadata.json').read_text())
        if not metadata['agent']['online_train']:
            continue
        if metadata['agent']['variant'] != 'official':
            raise ValueError('Expected official HDSAC training metadata')
        validate_metadata(metadata, config)
        if previous is not None:
            source = metadata['agent']['checkpoint_loaded'].replace('\\', '/').split('/')[-2]
            if source != previous:
                raise ValueError('Sessions do not form one continuous training lineage')
        with (session / 'sensor_manifest.csv').open(encoding='utf-8-sig', newline='') as handle:
            count = sum(1 for _ in csv.DictReader(handle))
        if count < 1:
            continue
        checkpoint = session / 'hdsac_online_latest.pt'
        FrozenHDSAC(checkpoint)  # Verify retained weights before connecting to SCANeR.
        index = session / 'snapshot_index.json'
        entries = json.loads(index.read_text()) if index.is_file() else []
        last_snapshot = 0
        for entry in entries:
            local_step = entry.get('session_recorded_steps', entry['recorded_training_steps'])
            if not last_snapshot < local_step <= count:
                raise ValueError('Snapshot index does not match recorded transition count')
            name = entry['checkpoint']
            if Path(name).name != name:
                raise ValueError('Invalid snapshot filename')
            snapshot = session / name
            if file_hash(snapshot) != entry['sha256']:
                raise ValueError('Snapshot hash mismatch')
            total = cumulative + local_step
            if entry['recorded_training_steps'] != total:
                raise ValueError('Snapshot global step does not match the retained training lineage')
            tasks.append(dict(algorithm='HDSAC', training_steps=total, checkpoint=str(snapshot.resolve()),
                              checkpoint_sha256=entry['sha256'], result_dir=f'step_{total:06d}'))
            last_snapshot = local_step
        cumulative += count
        if last_snapshot < count:
            tasks.append(dict(algorithm='HDSAC', training_steps=cumulative, checkpoint=str(checkpoint.resolve()),
                              checkpoint_sha256=file_hash(checkpoint), result_dir=f'step_{cumulative:06d}'))
        previous = session.name
    if not tasks:
        raise ValueError('No retained HDSAC checkpoints found')
    return dict(algorithm='HDSAC', branch_id=branch.name, tests=1, mock=False, protocol=protocol,
                config=config, tasks=tasks, step_counter='cumulative recorded sensor manifest rows',
                initialization='fine-tuned from straight-driving HDSAC; prior training is excluded from x-axis',
                omitted_intermediate_checkpoints='not retained; cannot reconstruct historical weights')


def evaluate_task(adapter, task, manifest, suite):
    config = manifest['config']
    cfg = config['environment']
    root = suite / task['result_dir']
    root.mkdir(parents=True, exist_ok=True)
    if file_hash(task['checkpoint']) != task['checkpoint_sha256']:
        raise ValueError('Checkpoint changed since planning')
    metadata = json.loads((Path(task['checkpoint']).parent / 'metadata.json').read_text())
    validate_metadata(metadata, config)
    policy = FrozenHDSAC(task['checkpoint'])
    route = pe.Route(cfg['route'])
    native_route = ScenarioPolylineRoute(metadata['route']['waypoints'])
    schema = metadata['observation_schema']
    session = root / datetime.now().strftime('eval_%Y%m%d_%H%M%S_%f')
    session.mkdir(exist_ok=False)
    save_json(session / 'metadata.json', dict(algorithm='HDSAC', mode='eval', mock=False,
              checkpoint=task['checkpoint'], protocol=manifest['protocol'], native_training_metadata=metadata))
    state, lidar = adapter.reset()
    accumulator = EpisodeReward(route.project(state['pos_x'], state['pos_y'])['progress_m'])
    stalled = 0.
    lane_steps = 0
    with (session / 'steps.csv').open('x', encoding='utf-8', newline='') as handle:
        writer = None
        for step in range(1, cfg['horizon'] + 1):
            lidar_path = Path(adapter.lidar_reader.source)
            obs, quality = observation(state, native_route, 49.9, 8., lidar_path,
                policy_goal_mode=metadata['policy_goal_mode'], observation_schema=schema['name'],
                local_path_count=schema['local_path_points'], local_path_horizon_m=schema['local_path_horizon_m'],
                steering_full_scale_rad=metadata['control_mapping']['agent_steering_full_scale_rad'])
            if not quality['lidar_valid']:
                raise RuntimeError('Invalid LiDAR input; evaluation not scored')
            action = pe.capped_action(policy.act(obs), state, cfg)
            nxt, lidar = adapter.step(action)
            stalled = stalled + max(0., nxt['time'] - state['time']) if abs(nxt['speed_x']) < 1/3.6 else 0.
            terminated, truncated, reason, geom = pe.status(nxt, route, cfg, step, stalled)
            parts = accumulator.add(geom['progress_m'], abs(nxt['speed_x'])*3.6, reason)
            lane_steps += int(abs(geom['lateral_m']) > cfg['evaluation_lane_boundary_m'])
            row = dict(episode=1, episode_step=step, source='AGENT', takeover=0,
                       speed_kmh=nxt['speed_x']*3.6, progress_m=geom['progress_m'],
                       lane_gap_m=geom['lateral_m'], pos_x=nxt['pos_x'], pos_y=nxt['pos_y'],
                       simulation_time=nxt['time'], steering=float(action[0]), acceleration=float(action[1]),
                       evaluation_reward=parts.total, reward_driving=parts.driving,
                       reward_speed=parts.speed, reward_destination=parts.destination, terminal_reason=reason)
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            if step % 20 == 0 or reason:
                handle.flush()
                print(f'[HDSAC AUTO] trained={task["training_steps"]} step={step} speed={abs(nxt["speed_x"])*3.6:.1f} reason={reason}', flush=True)
            state = nxt
            if terminated or truncated:
                break
    if not accumulator.closed:
        raise RuntimeError('Incomplete evaluation')
    if file_hash(task['checkpoint']) != task['checkpoint_sha256']:
        raise RuntimeError('Checkpoint changed during evaluation')
    total = accumulator.parts
    episode = dict(episode=1, steps=step, terminal_reason=reason, goal=int(reason=='goal'),
                   offroad=int(reason=='offroad'), collision=int(reason=='collision'),
                   lane_invasion_rate=lane_steps/step, reward_driving_sum=total.driving,
                   reward_speed_sum=total.speed, reward_destination_sum=total.destination,
                   episode_return=total.total)
    save_json(session / 'evaluation_reward_episodes.json', [episode])
    result = dict(task, status='complete', mock=False, session=str(session),
                  updates_before=policy.updates, updates_after=policy.updates,
                  evaluation_horizon_steps=cfg['horizon'], **summarize([episode]))
    save_json(root / 'result.json', result)


def code_hashes():
    paths = [Path(__file__), HERE / 'plot_hdsac_metrics.py', HERE / 'plot_four_metrics.py',
             HERE / 'reward.py', HERE / 'checkpoint_sweep.py', PE_ROOT / 'train.py',
             PE_ROOT / 'environment.py', PROJECT / 'hdsac_official.py',
             PROJECT / 'collect_hdsac_score_eval.py', PROJECT / 'scaner_env.py',
             PROJECT / 'scaner_live_fusion_waypoint_agent.py']
    return {str(path.relative_to(PROJECT)): file_hash(path) for path in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path)
    parser.add_argument('--branch-id', default='phuoc8_hdsac_no_calibration_noise1000_v1')
    parser.add_argument('--max-episode-steps', type=int, default=200)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    pe.torch.set_num_threads(2)
    if args.suite:
        suite = args.suite.resolve(strict=True)
        manifest = json.loads((suite / 'manifest.json').read_text())
        if manifest['algorithm'] != 'HDSAC':
            raise ValueError('Expected an HDSAC suite')
        if manifest.get('evaluation_code_sha256') != code_hashes():
            raise ValueError('Evaluation code changed since this suite was prepared')
    else:
        if args.max_episode_steps < 1 or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.branch_id):
            parser.error('Invalid branch or horizon')
        manifest = plan(PROJECT / 'HDSAC_SNAPSHOT100' / 'experiments' / args.branch_id, args.max_episode_steps)
        manifest['evaluation_code_sha256'] = code_hashes()
        suite = HERE / 'results' / 'hdsac_sweeps' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        suite.mkdir(parents=True, exist_ok=False)
        save_json(suite / 'manifest.json', manifest)
    print(f'[HDSAC] suite={suite}', flush=True)
    print('[HDSAC] retained training steps=' + ', '.join(str(t['training_steps']) for t in manifest['tasks']), flush=True)
    if args.plan_only:
        return
    pending = []
    for task in manifest['tasks']:
        result = suite / task['result_dir'] / 'result.json'
        if result.is_file():
            saved = json.loads(result.read_text())
            if (saved.get('status') == 'complete' and saved['completed_episodes'] == 1 and
                    saved['evaluation_horizon_steps'] == manifest['config']['environment']['horizon'] and
                    saved['checkpoint_sha256'] == task['checkpoint_sha256'] and
                    file_hash(task['checkpoint']) == task['checkpoint_sha256']):
                continue
            raise ValueError('Saved result does not match suite')
        pending.append(task)
    if pending:
        adapter = pe.ScanerAdapter(copy.deepcopy(manifest['config']['environment']), PROJECT)
        try:
            adapter.connect()
            for task in pending:
                evaluate_task(adapter, task, manifest, suite)
        finally:
            adapter.close()
    subprocess.run([sys.executable, str(HERE / 'plot_hdsac_metrics.py'), '--suite', str(suite)], check=True)


if __name__ == '__main__':
    main()
