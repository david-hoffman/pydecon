#!/usr/bin/env python
# -*- coding: utf-8 -*-
# test_utils.py
"""
Test utils.

Copyright (c) 2021, David Hoffman
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose
import unittest
from pydecon.utils import _ensure_positive, radialavg, expand_radialavg


def make_random_blob(ndims, size):
    """Make a random sized and placed blob."""
    # make coordinates
    x = np.linspace(-1, 1, size)
    mesh = np.meshgrid(*((x,) * ndims), indexing="ij")
    # randomly generate radii and center
    radii = (np.random.rand(ndims) - 0.5) * 0.2
    center = np.random.rand(ndims) - 0.5
    # make sure they'll broadcast correctly with mesh
    radii.shape += (1,) * ndims
    center.shape += (1,) * ndims
    # calc distances
    distances = (mesh - center) / radii
    r2 = (distances ** 2).sum(0)
    # return blob
    return np.exp(-r2)


class TestRadialAverageFuncs(unittest.TestCase):
    """Radial average function tester class."""

    def setUp(self):
        """Set up variables for testing."""
        self.test_data2d = make_random_blob(2, 128)
        self.test_data3d = make_random_blob(3, 128)
        self.radavg2d = radialavg(self.test_data2d)
        # radial average along z
        self.radavg3d = radialavg(self.test_data3d)

    def test_returns2d(self):
        """Make sure 2d data turns into 1d."""
        assert self.radavg2d.ndim == 1

    def test_returns3d(self):
        """Make sure 3d data turns into 2d."""
        assert self.radavg3d.ndim == 2

    def test_returns2d_even(self):
        """Make sure 2d data turns into 1d that is odd in length."""
        assert self.radavg2d.size % 2

    def test_returns3d_even(self):
        """Make sure 3d data turns into 2d that is odd along the r direction."""
        assert self.radavg3d.shape[1] % 2

    def test_self_consistency2d(self):
        """Make sure that expanded radialavg makes sense."""
        # expand
        expanded = expand_radialavg(self.radavg2d)
        center = np.array(expanded.shape) // 2
        # test center lines
        assert_allclose(expanded[center[0], center[1] :], self.radavg2d)
        assert_allclose(expanded[center[0], center[1] :], expanded[center[0] :, center[1]])

    def test_self_consistency3d(self):
        """Make sure that expanded radialavg is radially averaged to the same thing, 3d."""
        # expand
        expanded = expand_radialavg(self.radavg3d)
        center = np.array(expanded.shape) // 2
        # test center lines
        assert_allclose(expanded[:, center[1], center[2] :], self.radavg3d)
        assert_allclose(expanded[:, center[1], center[2] :], expanded[:, center[1] :, center[2]])

    def test_fft_2d(self):
        """Test that fft is real."""
        expanded = expand_radialavg(self.radavg2d)
        fft_data = np.fft.ifftn(np.fft.fftshift(expanded))
        assert abs(fft_data.imag).max() < np.finfo(fft_data.dtype).eps

    def test_fft_3d(self):
        """Test that fft is real."""
        expanded = expand_radialavg(self.radavg3d)
        # we don't know where the max value is along z so we only want
        # to take the fft along the radially averaged direction.
        fft_data = np.fft.ifftn(np.fft.fftshift(expanded), axes=(1, 2))
        max_val = abs(fft_data.imag).max()
        eps = np.finfo(fft_data.dtype).eps
        assert max_val < eps, "{} !< {}".format(max_val, eps)


class TestEnsurepositive(unittest.TestCase):
    """test ensure positive for proper behaviour."""

    def setUp(self):
        """Set up internal variables for test."""
        self.test_data = np.random.randn(1024)
        self.test_data_copy = self.test_data.copy()
        self.test_data_positive = _ensure_positive(self.test_data)

    def test_action(self):
        """Make sure it works as advertise."""
        assert (self.test_data_positive >= 0).all()

    def test_ensurepositive_copy(self):
        """Ensure return of a copy of the data and that the original data remains unchanged."""
        assert self.test_data is not self.test_data_positive
