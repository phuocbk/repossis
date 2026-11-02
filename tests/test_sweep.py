import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checkpoint_sweep import discover, summarize, execute, pe, HERE, PE_ROOT, file_hash


class SweepTests(unittest.TestCase):
    def test_discovery_uses_recorded_step_counter_not_filename_or_updates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            session = root / 'train_1'
            session.mkdir()
            (session / 'metadata.json').write_text(json.dumps(dict(mode='train', mock=False)))
            for name, step in [('actor_critic_step_000100.pt', 500),
                               ('actor_critic_step_000200.pt', 600),
                               ('pe_rlhf_latest.pt', 600)]:
                pe.torch.save(dict(config=dict(online_recorded_steps=step), updates=7), session / name)
            result = discover(root, 500)
            self.assertEqual([item['training_steps'] for item in result], [500, 600])
            self.assertTrue(result[-1]['checkpoint'].endswith('pe_rlhf_latest.pt'))
            self.assertEqual(len(result[0]['checkpoint_sha256']), 64)

    def test_summary_uses_completed_outcomes_and_real_return(self):
        result = summarize([dict(episode_return=10., goal=0, offroad=1),
                            dict(episode_return=30., goal=1, offroad=0)])
        self.assertEqual(result['mean_return'], 20.)
        self.assertEqual(result['success_rate'], .5)
        self.assertEqual(result['offroad_rate'], .5)
        self.assertLess(result['success_ci95'][0], .5)
        self.assertGreater(result['success_ci95'][1], .5)

    def test_shared_connection_across_checkpoints_and_resume_skips_completed(self):
        config = json.loads((PE_ROOT / 'config.json').read_text())
        protocol = json.loads((HERE / 'protocol.json').read_text())
        protocol['horizon_steps'] = 200
        instances = []

        class Adapter:
            def __init__(self, *args):
                self.connects = self.closes = 0
                instances.append(self)
            def connect(self):
                self.connects += 1
            def close(self):
                self.closes += 1

        def collect(cfg, args, root, mode, learner, budget):
            self.assertEqual(mode, 'eval')
            self.assertEqual(cfg['environment']['horizon'], 200)
            self.assertEqual(budget, 400)
            self.assertFalse(args.experimental_selector)
            borrowed = pe.ScanerAdapter()
            borrowed.connect()
            journal = pe.Journal(root / 'eval_fixture')
            for ep in range(1, 3):
                journal.add(dict(episode_id=ep, progress_m=10.),
                            dict(terminal_reason='goal', progress_m=11., speed_kmh=20.))
            journal.close()
            borrowed.close()

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            session = root / 'source'
            session.mkdir()
            (session / 'metadata.json').write_text(json.dumps(dict(config=config)))
            # Checkpoints are loaded by a stub; this test isolates orchestration.
            path = session / 'model.pt'
            path.write_bytes(b'test fixture, not a real checkpoint')
            tasks = [dict(training_steps=s, checkpoint=str(path), checkpoint_sha256=file_hash(path),
                          result_dir=f'step_{s}') for s in (500, 1000)]
            manifest = dict(tests=2, mock=False, protocol=protocol, tasks=tasks)
            with patch.object(pe, 'ScanerAdapter', Adapter), \
                 patch.object(pe, 'collect', collect), \
                 patch('checkpoint_sweep.PERLHFLearner') as learner, \
                 patch('checkpoint_sweep.render'):
                learner.return_value.updates = 7
                execute(root, manifest)
                execute(root, manifest)
            self.assertEqual(len(instances), 1)
            self.assertEqual((instances[0].connects, instances[0].closes), (1, 1))
            self.assertEqual(learner.call_count, 2)


if __name__ == '__main__':
    unittest.main()
