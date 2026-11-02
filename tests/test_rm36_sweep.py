import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_rm36_sweep import (FrozenRM36, PROJECT, PE_ROOT, file_hash, plan, evaluate_task)
from hdsac_official import OfficialHDSACLearner

BRANCH = PROJECT / 'RM36/experiments/phuoc8_threshold30_episode_band30_v1'
CHECKPOINT = BRANCH / '20261007_152652/hdsac_online_latest.pt'


class RM36Tests(unittest.TestCase):
    def test_retained_actor_matches_native_deterministic_actor(self):
        policy = FrozenRM36(CHECKPOINT)
        native = OfficialHDSACLearner(state_dim=444, checkpoint=str(CHECKPOINT), device='cpu')
        for obs in np.random.default_rng(71).normal(size=(3, 444)).astype(np.float32):
            np.testing.assert_allclose(policy.act(obs), native.act(obs, deterministic=True), atol=1e-6)
        self.assertFalse(any(param.requires_grad for param in policy.actor.parameters()))

    def test_step_counter_includes_rejected_recorded_samples(self):
        manifest = plan(BRANCH, 200)
        self.assertEqual([task['training_steps'] for task in manifest['tasks']], [3000, 4000])

    def test_timeout_scores_once_and_preserves_checkpoint(self):
        manifest = plan(BRANCH, 200)
        task = manifest['tasks'][-1]
        before = file_hash(CHECKPOINT)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            lidar = root / 'frame.csv'
            lidar.write_text('hit,distance,hAngleSensor\n' + '1,20,0\n'*12, encoding='utf-8')

            class Adapter:
                lidar_reader = SimpleNamespace(source=str(lidar))
                calls = 0
                def reset(self):
                    return dict(pos_x=-4.4445, pos_y=-45.1308, speed_x=2., heading=6.275,
                                steer=0., time=0.), None
                def step(self, action):
                    self.calls += 1
                    state, _ = self.reset()
                    state['time'] = self.calls / 10
                    return state, None

            adapter = Adapter()
            evaluate_task(adapter, task, manifest, root)
            result = json.loads((root / task['result_dir'] / 'result.json').read_text())
            self.assertEqual(adapter.calls, 200)
            self.assertEqual(result['goals'], 0)
            self.assertAlmostEqual(result['mean_return'], 200*.1*7.2/80)
            session = Path(result['session'])
            episode = json.loads((session / 'evaluation_reward_episodes.json').read_text())[0]
            self.assertEqual(episode['terminal_reason'], 'time_limit')
            self.assertEqual(episode['reward_destination_sum'], 0)
        self.assertEqual(file_hash(CHECKPOINT), before)


if __name__ == '__main__':
    unittest.main()
