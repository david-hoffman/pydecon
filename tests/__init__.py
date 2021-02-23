#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __init__.py
"""
Test test.

Copyright (c) 2021, David Hoffman
"""

import unittest


class InitializationTests(unittest.TestCase):
    """Test basics."""

    def test_initialization(self):
        """Check the test suite runs by affirming 2+2=4."""
        self.assertEqual(2 + 2, 4)

    def test_import(self):
        """Ensure the test suite can import our module."""
        try:
            import pydecon
        except ImportError:
            self.fail("Was not able to import the pyDecon package")
