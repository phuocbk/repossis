"""Read-only checkpoint evaluation with the shared episode reward."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from reward import EpisodeReward

HERE = Path(__file__).resolve().parent
PE_ROOT = HERE.parent / 'PE_RLHF_SCANeR'
sys.path.insert(0, str(PE_ROOT))
import train as pe
from learner import PERLHFLearner
from observation import OBS_DIM
from storage import Journal, save_json


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class ScoredJournal(Journal):
    def __init__(self, root):
        super().__init__(root)
        self.accumulator = None
        self.episode_id = None
        self.completed = []

    def add(self, arrays, diagnostics):
        episode = int(arrays['episode_id'])
        if episode != self.episode_id:
            self.episode_id = episode
            self.accumulator = EpisodeReward(float(arrays['progress_m']))
        reason = diagnostics['terminal_reason']
        parts = self.accumulator.add(float(diagnostics['progress_m']),
                                     abs(float(diagnostics['speed_kmh'])), reason)
        diagnostics = dict(diagnostics, evaluation_reward=parts.total,
                           evaluation_reward_driving=parts.driving,
                           evaluation_reward_speed=parts.speed,
                           evaluation_reward_destination=parts.destination)
        arrays = dict(arrays, evaluation_reward=parts.total)
        super().add(arrays, diagnostics)
        if reason:
            total = self.accumulator.parts
            self.completed.append(dict(
                episode=episode, steps=self.accumulator.steps, terminal_reason=reason,
                goal=int(reason == 'goal'), offroad=int(reason == 'offroad'),
                collision=int(reason == 'collision'), reward_driving_sum=total.driving,
                reward_speed_sum=total.speed, reward_destination_sum=total.destination,
                episode_return=total.total))
            save_json(self.root / 'evaluation_reward_episodes.json', self.completed)
            print(f'[AUTO EVAL] episode={episode} reason={reason} return={total.total:.4f}', flush=True)

    def close(self):
        super().close()
        count = len(self.completed)
        save_json(self.root / 'evaluation_reward_summary.json', dict(
            algorithm='PE-RLHF', completed_episodes=count,
            goals=sum(e['goal'] for e in self.completed),
            success_rate=sum(e['goal'] for e in self.completed) / count if count else None,
            mean_episode_return=sum(e['episode_return'] for e in self.completed) / count if count else None,
            formula='delta_progress_m + 0.1 * speed_kmh / 80 + 20 * terminal_goal',
            normalization_to_100=False, incomplete_episode_excluded=bool(
                self.accumulator and not self.accumulator.closed)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--tests', type=int, default=20)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    if args.tests < 1:
        parser.error('--tests must be positive')
    # Use the exact environment and observation settings recorded with this model.
    metadata = json.loads((checkpoint.parent / 'metadata.json').read_text(encoding='utf-8'))
    config = metadata['config']
    protocol = json.loads((HERE / 'protocol.json').read_text(encoding='utf-8'))
    for field in ('frequency_hz', 'max_speed_kmh', 'stall_sec', 'horizon', 'goal_region'):
        key = {'stall_sec': 'stall_timeout_sec', 'horizon': 'horizon_steps'}.get(field, field)
        if config['environment'][field] != protocol[key]:
            raise ValueError(f'Checkpoint environment differs from shared protocol: {field}')
    pe.torch.set_num_threads(2)
    learner = PERLHFLearner(OBS_DIM, config['learning'])
    before_hash = file_hash(checkpoint)
    learner.load(checkpoint)
    updates = learner.updates
    output = HERE / 'results' / 'PE_RLHF'
    options = SimpleNamespace(mock=False, checkpoint=str(checkpoint), tests=args.tests,
                              experimental_selector=False)
    original_journal = pe.Journal
    sessions_before = set(output.glob('eval_*'))
    pe.Journal = ScoredJournal
    try:
        pe.collect(config, options, output, 'eval', learner,
                   budget=args.tests * config['environment']['horizon'])
    finally:
        pe.Journal = original_journal
        after_hash = file_hash(checkpoint)
        for session in set(output.glob('eval_*')) - sessions_before:
            save_json(session / 'evaluation_protocol.json', dict(
                protocol=protocol, requested_episodes=args.tests,
                checkpoint=str(checkpoint), sha256_before=before_hash, sha256_after=after_hash,
                checkpoint_unchanged=before_hash == after_hash,
                updates_before=updates, updates_after=learner.updates,
                mode='deterministic_actor_only', training_updates=False))
        if before_hash != after_hash or updates != learner.updates:
            raise RuntimeError('Checkpoint or learner update count changed during evaluation')


if __name__ == '__main__':
    main()
