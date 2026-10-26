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
