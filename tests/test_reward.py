from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reward import EpisodeReward, evaluation_reward


class RewardTests(unittest.TestCase):
    def test_units_and_terminal_bonus(self):
        parts = evaluation_reward(10, 11.5, 20, True)
        self.assertAlmostEqual(parts.driving, 1.5)
        self.assertAlmostEqual(parts.speed, .025)
        self.assertEqual(parts.destination, 20)
        self.assertAlmostEqual(parts.total, 21.525)

    def test_reset_not_counted_and_goal_not_repeated(self):
        ep = EpisodeReward(50)
        ep.add(51, 15)
        ep.add(50.5, 0, 'goal')
        self.assertAlmostEqual(ep.parts.total, .5 + .01875 + 20)
        with self.assertRaises(ValueError):
            ep.add(50.5, 0, 'goal')
        new = EpisodeReward(5)
        self.assertEqual(new.add(5, 0).total, 0)

    def test_reverse_and_failure_have_no_unspecified_penalties(self):
        ep = EpisodeReward(10)
        self.assertAlmostEqual(ep.add(9, 8, 'offroad').total, -.99)
        self.assertEqual(ep.parts.destination, 0)

    def test_invalid_telemetry_rejected(self):
        for speed in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                evaluation_reward(0, 1, speed)


if __name__ == '__main__':
    unittest.main()
