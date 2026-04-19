"""Tests for Mamba selective state space model."""
import unittest
from src.models.mamba import Mamba

class TestMamba(unittest.TestCase):
    def test_init(self):
        m = Mamba()
        self.assertIsNotNone(m)
