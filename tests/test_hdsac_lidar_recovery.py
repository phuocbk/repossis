import csv
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_hdsac_sweep_recover as recovery
from collect_hdsac_score_eval import lidar_ranges


class RecoveryTests(unittest.TestCase):
    def test_snapshot_preserves_native_ranges_and_normalization(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'lidar.csv'
            path.write_text('hit,distance,hAngleSensor\n' + '1,20,0\n' * 12)
            expected, valid = lidar_ranges(path)
            self.assertTrue(valid)

            def observation(state, route, x, y, snapshot, **kwargs):
                # Even if the producer file changes, the parser sees original bytes.
                path.write_text('hit,distance,hAngleSensor\n')
                rays, valid = lidar_ranges(snapshot)
                return rays, {'lidar_valid': valid}

            with patch.object(recovery, 'NATIVE_OBSERVATION', observation):
                actual, quality = recovery.snapshot_observation({}, None, 0, 0, path)
            np.testing.assert_array_equal(actual, expected)
            self.assertTrue(quality['lidar_valid'])

    def test_insufficient_hits_still_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'lidar.csv'
            path.write_text('hit,distance,hAngleSensor\n' + '1,20,0\n' * 9)

            def observation(state, route, x, y, snapshot, **kwargs):
                rays, valid = lidar_ranges(snapshot)
                return rays, {'lidar_valid': valid}

            with patch.object(recovery, 'NATIVE_OBSERVATION', observation):
                with self.assertRaises(recovery.InvalidLidar):
                    recovery.snapshot_observation({}, None, 0, 0, path)

    def test_sensor_failure_retries_whole_task_then_returns_result(self):
        adapter = SimpleNamespace(lidar_reader=SimpleNamespace(source='test.csv'), send=lambda action: None)
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(recovery, 'NATIVE_EVALUATE',
                              side_effect=[recovery.InvalidLidar('test'), 'complete']) as run:
                with patch.object(recovery.time, 'sleep'):
                    result = recovery.evaluate_with_recovery(adapter, {'training_steps': 2000}, {}, Path(temp))
            self.assertEqual(result, 'complete')
            self.assertEqual(run.call_count, 2)
            self.assertEqual(len(list((Path(temp) / 'sensor_failures').glob('*.json'))), 1)

    def test_persistent_failure_stops_after_three_attempts(self):
        adapter = SimpleNamespace(lidar_reader=SimpleNamespace(source='test.csv'), send=lambda action: None)
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(recovery, 'NATIVE_EVALUATE', side_effect=recovery.InvalidLidar('test')) as run:
                with patch.object(recovery.time, 'sleep'):
                    with self.assertRaisesRegex(RuntimeError, '3 attempts'):
                        recovery.evaluate_with_recovery(adapter, {'training_steps': 2000}, {}, Path(temp))
            self.assertEqual(run.call_count, 3)
            self.assertFalse(list(Path(temp).glob('**/result.json')))

    def test_unrelated_failure_is_not_retried(self):
        with patch.object(recovery, 'NATIVE_EVALUATE', side_effect=RuntimeError('bad checkpoint')) as run:
            with self.assertRaisesRegex(RuntimeError, 'bad checkpoint'):
                recovery.evaluate_with_recovery(None, {}, {}, Path('.'))
        self.assertEqual(run.call_count, 1)


if __name__ == '__main__':
    unittest.main()
