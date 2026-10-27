"""Offline accumulator contracts; no capture, Redis or model dependencies."""
import io
import pickle
import unittest
from unittest.mock import patch

import numpy as np

from emma.metrics.correlationlist import CorrelationList
from emma.metrics.distancelist import DistanceList


class MetricTests(unittest.TestCase):
    def test_incremental_and_merged_metrics(self):
        x = np.array([1., 2., 3., 4.])
        y = 2 * x + 3
        for metric_type, expected in ((CorrelationList, 1.), (DistanceList, 22.)):
            with self.subTest(metric=metric_type.__name__):
                left, right = metric_type((1,)), metric_type((1,))
                left.update(0, x[:2], y[:2])
                right.update(0, x[2:], y[2:])
                left.merge(right)
                self.assertEqual(left[0], expected)
                self.assertEqual(left[(0,)], expected)
                np.testing.assert_array_equal(left[[0]], [expected])
                with self.assertRaises(NotImplementedError):
                    left[:]
                with self.assertRaises(TypeError):
                    left["0"]
                with self.assertRaises(TypeError):
                    left.merge(object())

    def test_integer_inputs_do_not_overflow(self):
        for dtype in (np.uint8, np.int16):
            with self.subTest(dtype=dtype):
                x = np.array([100, 101, 102, 103], dtype=dtype)
                y = x[::-1]
                correlation = CorrelationList((1,))
                correlation.update(0, x, y)
                self.assertEqual(correlation[0], -1.)
                distance = DistanceList((1,))
                distance.update(0, x, y)
                self.assertEqual(distance[0], 8.)

    def test_large_offsets_remain_correlated_after_updates_and_merges(self):
        x = 1e12 + np.arange(1., 9.)
        y = 3 * x + 5
        whole, left, right, incremental = [CorrelationList((1,)) for _ in range(4)]
        whole.update(0, x, y)
        left.update(0, x[:4], y[:4])
        right.update(0, x[4:], y[4:])
        left.merge(right)
        for a, b in zip(x, y):
            incremental.update(0, [a], [b])
        for metric in (whole, left, incremental):
            self.assertAlmostEqual(metric[0], 1., places=13)
        empty = CorrelationList((1,))
        empty.merge(left)
        self.assertAlmostEqual(empty[0], 1., places=13)
        empty.merge(CorrelationList((1,)))
        self.assertAlmostEqual(empty[0], 1., places=13)

    def test_correlation_does_not_depend_on_amplitude_units(self):
        for scale in (1e-100, 1e-12, 1., 1e12, 1e100):
            x = scale * np.array([1., 2., 3., 4.])
            metric = CorrelationList((1,))
            metric.update(0, x, -3 * x)
            self.assertAlmostEqual(metric[0], -1., places=13)

    def test_legacy_serialized_statistics_are_accepted(self):
        x, y = np.array([1., 2., 3., 4.]), np.array([7., 9., 11., 13.])
        metric = CorrelationList.__new__(CorrelationList)
        metric.__dict__.update({
            '_sumx': np.array([x.sum()]), '_sumy': np.array([y.sum()]),
            '_sumxy': np.array([np.dot(x, y)]),
            '_sumxsq': np.array([np.dot(x, x)]), '_sumysq': np.array([np.dot(y, y)]),
            '_n': np.array([4.]),
        })
        restored = pickle.loads(pickle.dumps(metric))
        self.assertAlmostEqual(restored[0], 1., places=13)
        restored.update(0, [5.], [15.])
        self.assertAlmostEqual(restored[0], 1., places=13)

    def test_invalid_batch_does_not_partially_update_statistics(self):
        metric = CorrelationList((1,))
        metric.update(0, [1., 2.], [3., 5.])
        before = pickle.dumps(metric)
        for x, y in (([1., 2.], [3.]), ([[1., 2.]], [[3., 4.]])):
            with self.assertRaises(ValueError):
                metric.update(0, x, y)
            self.assertEqual(pickle.dumps(metric), before)
        metric.update(0, [], [])
        self.assertEqual(pickle.dumps(metric), before)

    def test_save_keeps_the_serialized_accumulator(self):
        for metric_type in (CorrelationList, DistanceList):
            with self.subTest(metric=metric_type.__name__):
                metric = metric_type((1,))
                metric.update(0, np.array([1., 2.]), np.array([3., 4.]))
                stream = io.BytesIO()
                with patch(metric_type.__module__ + ".open", return_value=stream, create=True):
                    # Keep the in-memory bytes available after a correct close.
                    with patch.object(stream, "close"):
                        metric.save()
                restored = pickle.loads(stream.getvalue())
                self.assertEqual(restored[0], metric[0])
                stream.close()

    def test_save_closes_the_file_if_serialization_fails(self):
        for metric_type in (CorrelationList, DistanceList):
            with self.subTest(metric=metric_type.__name__):
                stream = io.BytesIO()
                with patch(metric_type.__module__ + ".open", return_value=stream, create=True):
                    with patch("pickle.dump", side_effect=OSError("write failed")):
                        with self.assertRaisesRegex(OSError, "write failed"):
                            metric_type((1,)).save()
                self.assertTrue(stream.closed)
