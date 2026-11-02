from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_pe_rlhf_auto import ScoredJournal


class ScoredJournalTests(unittest.TestCase):
    def test_episode_reset_and_incomplete_exclusion(self):
        with tempfile.TemporaryDirectory() as temp:
            journal = ScoredJournal(Path(temp) / 'eval')
            for ep, before, after, reason in ((1, 10., 11., 'goal'), (2, 0., 1., '')):
                journal.add(dict(episode_id=ep, progress_m=before),
                            dict(terminal_reason=reason, progress_m=after, speed_kmh=20.))
            journal.close()
            summary = json.loads((journal.root / 'evaluation_reward_summary.json').read_text())
            self.assertEqual(summary['completed_episodes'], 1)
            self.assertAlmostEqual(summary['mean_episode_return'], 21.025)
            self.assertTrue(summary['incomplete_episode_excluded'])


if __name__ == '__main__':
    unittest.main()
