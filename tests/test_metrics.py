"""Tests for MAC accounting."""
import unittest
from src.utils.metrics import compute_macs

class TestMetrics(unittest.TestCase):
    def test_macs(self):
        self.assertEqual(compute_macs([1, 2]), 2)
