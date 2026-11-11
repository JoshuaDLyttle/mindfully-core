"""
Mindfully AI - Unit & Integration Tests for eeg_receiver.py
==========================================================
File: ml_pipeline/test_eeg_receiver.py
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eeg_receiver import (
    EEGRingBuffer,
    MindfullyEdgeInference,
    format_urge_bar,
    classify_urge_level,
)


class TestEEGRingBuffer(unittest.TestCase):
    def test_initialization(self):
        rb = EEGRingBuffer(window_size=1024, num_channels=2)
        self.assertEqual(rb.window_size, 1024)
        self.assertEqual(rb.num_channels, 2)
        self.assertFalse(rb.is_full())
        self.assertEqual(rb.total_samples_received, 0)
        with self.assertRaises(ValueError):
            rb.get_window()

    def test_partial_and_full_append(self):
        rb = EEGRingBuffer(window_size=4, num_channels=2)
        chunk1 = np.array([[1.0, 10.0], [2.0, 20.0]], dtype=np.float32)
        rb.append_chunk(chunk1)
        self.assertEqual(rb.total_samples_received, 2)
        self.assertFalse(rb.is_full())

        chunk2 = np.array([[3.0, 30.0], [4.0, 40.0]], dtype=np.float32)
        rb.append_chunk(chunk2)
        self.assertEqual(rb.total_samples_received, 4)
        self.assertTrue(rb.is_full())

        window = rb.get_window()
        self.assertEqual(window.shape, (4, 2))
        np.testing.assert_array_almost_equal(
            window,
            np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]], dtype=np.float32),
        )

    def test_wrap_around_ordering(self):
        rb = EEGRingBuffer(window_size=4, num_channels=1)
        rb.append_chunk([[1], [2], [3], [4]])
        self.assertTrue(rb.is_full())
        np.testing.assert_array_equal(rb.get_window().flatten(), [1, 2, 3, 4])

        # Push 1 sample -> should drop [1] and have [2, 3, 4, 5]
        rb.append_chunk([[5]])
        np.testing.assert_array_equal(rb.get_window().flatten(), [2, 3, 4, 5])

        # Push 2 samples -> should have [4, 5, 6, 7]
        rb.append_chunk([[6], [7]])
        np.testing.assert_array_equal(rb.get_window().flatten(), [4, 5, 6, 7])

    def test_large_chunk_exceeding_window(self):
        rb = EEGRingBuffer(window_size=4, num_channels=1)
        large_chunk = np.array([[10], [20], [30], [40], [50], [60]], dtype=np.float32)
        rb.append_chunk(large_chunk)
        self.assertTrue(rb.is_full())
        # Should keep the last 4: [30, 40, 50, 60]
        np.testing.assert_array_equal(rb.get_window().flatten(), [30, 40, 50, 60])


class TestFormatters(unittest.TestCase):
    def test_urge_bar(self):
        bar_low = format_urge_bar(0.0, width=10)
        self.assertEqual(bar_low, "░" * 10)
        bar_high = format_urge_bar(1.0, width=10)
        self.assertEqual(bar_high, "█" * 10)

    def test_classify_urge_level(self):
        level, _ = classify_urge_level(0.90)
        self.assertEqual(level, "CRITICAL")
        level, _ = classify_urge_level(0.75)
        self.assertEqual(level, "HIGH")
        level, _ = classify_urge_level(0.50)
        self.assertEqual(level, "MODERATE")
        level, _ = classify_urge_level(0.20)
        self.assertEqual(level, "LOW")


class TestTFLiteInference(unittest.TestCase):
    def test_model_prediction(self):
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "edge_models",
            "mindfully_edge_v1.tflite",
        )
        self.assertTrue(os.path.exists(model_path), f"Model path not found: {model_path}")

        engine = MindfullyEdgeInference(model_path=model_path)
        dummy_window = np.random.normal(0, 15.0, (1024, 2)).astype(np.float32)
        out = engine.predict(dummy_window)

        self.assertIn("substance_urge", out)
        self.assertIn("behavioral_urge", out)
        self.assertIn("willpower_capacity", out)
        self.assertIsInstance(out["substance_urge"], float)
        self.assertGreaterEqual(out["substance_urge"], 0.0)
        self.assertLessEqual(out["substance_urge"], 1.0)


if __name__ == "__main__":
    unittest.main()
