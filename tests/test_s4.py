"""Tests for S4 model."""
import unittest
from src.models.s4 import S4

class TestS4(unittest.TestCase):
    def test_init(self):
        s = S4()
        self.assertIsNotNone(s)
