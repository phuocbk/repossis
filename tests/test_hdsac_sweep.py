import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_hdsac_sweep import FrozenHDSAC, PROJECT, file_hash, plan, evaluate_task, pe
from plot_hdsac_metrics import report
from hdsac_official import OfficialHDSACLearner

BRANCH = PROJECT / 'HDSAC_SNAPSHOT100/experiments/phuoc8_hdsac_no_calibration_noise1000_v1'
CHECKPOINT = BRANCH / '20261027_164141/actor_critic_step_005000.pt'


class HDSACTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pe.torch.set_num_threads(2)
        cls.manifest = plan(BRANCH, 200)

    def test_all_fifty_steps_include_prefix_and_continuation_without_duplicates(self):
        tasks = self.manifest['tasks']
        self.assertEqual([task['training_steps'] for task in tasks], list(range(100, 5001, 100)))
        self.assertTrue(all(task['algorithm'] == 'HDSAC' for task in tasks))
        self.assertEqual(self.manifest['tests'], 1)
        self.assertFalse(self.manifest['protocol']['training_updates'])
        self.assertFalse(self.manifest['protocol']['human_takeover'])
        self.assertEqual(self.manifest['protocol']['horizon_steps'], 200)

    def test_frozen_actor_matches_native_deterministic_action(self):
        policy = FrozenHDSAC(CHECKPOINT)
        native = OfficialHDSACLearner(state_dim=444, checkpoint=str(CHECKPOINT), device='cpu')
        for obs in np.random.default_rng(71).normal(size=(3, 444)).astype(np.float32):
            np.testing.assert_allclose(policy.act(obs), native.act(obs, deterministic=True), atol=1e-6)
        self.assertEqual(policy.updates, 886)
        self.assertFalse(any(param.requires_grad for param in policy.actor.parameters()))

    def test_timeout_score_report_and_checkpoint_immutability(self):
        manifest = dict(self.manifest, tasks=[self.manifest['tasks'][-1]])
        task = manifest['tasks'][0]
        before = file_hash(CHECKPOINT)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            lidar = root / 'frame.csv'
            lidar.write_text('hit,distance,hAngleSensor\n' + '1,20,0\n' * 12, encoding='utf-8')

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
            self.assertAlmostEqual(result['mean_return'], 200 * .1 * 7.2 / 80)
            self.assertEqual(result['updates_before'], result['updates_after'])
            episode = json.loads((Path(result['session']) / 'evaluation_reward_episodes.json').read_text())[0]
            self.assertEqual(episode['terminal_reason'], 'time_limit')
            self.assertEqual(episode['reward_destination_sum'], 0)
            (root / 'manifest.json').write_text(json.dumps(manifest))
            output = report(root)
            with (output / 'hdsac_checkpoint_scores.csv').open(encoding='utf-8-sig', newline='') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(int(rows[0]['training_steps']), 5000)
            self.assertAlmostEqual(float(rows[0]['reward']), 1.8)
            self.assertTrue((output / 'hdsac_four_metrics.png').is_file())
        self.assertEqual(file_hash(CHECKPOINT), before)


if __name__ == '__main__':
    unittest.main()
