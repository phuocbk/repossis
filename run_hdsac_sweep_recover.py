"""Resume the existing AUTO suite with stable LiDAR reads and bounded sensor retries."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import tempfile
import time

import run_hdsac_sweep as base

NATIVE_OBSERVATION = base.observation
NATIVE_EVALUATE = base.evaluate_task
MAX_ATTEMPTS = 3


class InvalidLidar(RuntimeError):
    pass


def snapshot_observation(state, route, goal_x, goal_y, lidar_path, **kwargs):
    """Keep the training parser/normalization; never parse a changing producer file."""
    source = Path(lidar_path)
    try:
        before = source.stat()
        raw = source.read_bytes()
        after = source.stat()
    except OSError as exc:
        raise InvalidLidar(f'LiDAR snapshot unavailable: {source}: {exc}') from exc
    if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
        raise InvalidLidar(f'LiDAR file changed during reading: {source}')
    if not raw or len(raw) != after.st_size:
        raise InvalidLidar(f'LiDAR file is empty/incomplete: {source}')
    if time.time() - after.st_mtime > 2.0:
        raise InvalidLidar(f'LiDAR input is older than the shared 2-second limit: {source}')
    with tempfile.TemporaryDirectory(prefix='hdsac_lidar_') as temp:
        snapshot = Path(temp) / 'frame.csv'
        snapshot.write_bytes(raw)
        obs, quality = NATIVE_OBSERVATION(state, route, goal_x, goal_y, snapshot, **kwargs)
    if not quality['lidar_valid']:
        raise InvalidLidar(f'HDSAC LiDAR validation failed (requires >=10 hits in 80m): '
                           f'{source}; bytes={len(raw)}')
    return obs, quality


def is_lidar_failure(exc):
    if isinstance(exc, InvalidLidar):
        return True
    message = str(exc)
    return (isinstance(exc, RuntimeError) and
            (message.startswith('Invalid LiDAR input;') or message.startswith('LiDAR frame is stale') or
             message.startswith('LiDAR produced no fresh complete frame')))


def evaluate_with_recovery(adapter, task, manifest, suite):
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return NATIVE_EVALUATE(adapter, task, manifest, suite)
        except RuntimeError as exc:
            if not is_lidar_failure(exc):
                raise
            # The failed attempt has no result.json, so it is never scored.
            adapter.send([0., -1.])
            log = suite / 'sensor_failures'
            log.mkdir(exist_ok=True)
            detail = dict(training_steps=task['training_steps'], attempt=attempt,
                          scored=False, error=str(exc),
                          source=getattr(adapter.lidar_reader, 'source', None),
                          retry_resets_to_home=True)
            base.save_json(log / datetime.now().strftime('failure_%Y%m%d_%H%M%S_%f.json'), detail)
            print(f'[HDSAC][LIDAR] trained={task["training_steps"]} attempt={attempt}/{MAX_ATTEMPTS}: '
                  f'{exc}. Invalid attempt NOT scored.', flush=True)
            if attempt == MAX_ATTEMPTS:
                raise RuntimeError('LiDAR still invalid after 3 attempts. Completed checkpoint results '
                                   'are retained; check LASERMETER/logger before resuming.') from exc
            time.sleep(1.)
            # evaluate_task starts a new session and resets the vehicle to home.


def verify_recovery_version(suite):
    manifest = json.loads((suite / 'manifest.json').read_text())
    if manifest['algorithm'] != 'HDSAC' or manifest['evaluation_code_sha256'] != base.code_hashes():
        raise ValueError('Original HDSAC evaluation suite/code does not match')
    if manifest['config']['environment']['lidar_max_age_sec'] != 2:
        raise ValueError('Expected the existing two-second freshness protocol')
    record = dict(version='hdsac_stable_lidar_recovery_v1',
                  recovery_source_sha256=base.file_hash(__file__),
                  original_manifest_sha256=base.file_hash(suite / 'manifest.json'),
                  maximum_attempts=MAX_ATTEMPTS, reward_changed=False,
                  checkpoint_changed=False, invalid_attempts_scored=False,
                  completed_results_reused=True)
    path = suite / 'lidar_recovery_version.json'
    if path.exists():
        if json.loads(path.read_text()) != record:
            raise ValueError('Recovery code or suite changed; do not silently mix versions')
    else:
        with path.open('x', encoding='utf-8') as handle:
            json.dump(record, handle, indent=2)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--suite', type=Path, required=True)
    args, _ = parser.parse_known_args()
    suite = args.suite.resolve(strict=True)
    verify_recovery_version(suite)
    base.observation = snapshot_observation
    base.evaluate_task = evaluate_with_recovery
    base.main()


if __name__ == '__main__':
    main()
