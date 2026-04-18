"""Tests for ALCS event trigger."""
import unittest
from src.core.event_trigger import alcs_trigger

class TestTrigger(unittest.TestCase):
    def test_alcs(self):
        self.assertEqual(alcs_trigger([1, 2, 3], 1.5), [2, 3])
