from nose.tools import *
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import unittest
from pyDecon.utils import _ensure_positive, radialavg, expand_radialavg


def make_random_blob(ndims, size):
    """make a random sized and placed blob"""
    # make coordinates
    x = np.linspace(-1, 1, size)
    mesh = np.meshgrid(*((x, ) * ndims), indexing="ij")
    # randomly generate radii and center
    radii = (np.random.rand(ndims) - 0.5) * 0.2
    center = np.random.rand(ndims) - 0.5
    # make sure they'll broadcast correctly with mesh
    radii.shape += (1, ) * ndims
    center.shape += (1, ) * ndims
    # calc distances
    distances = (mesh - center) / radii
    r2 = (distances ** 2).sum(0)
    # return blob
    return np.exp(-r2)


class TestRadialAverage(unittest.TestCase):
    """Radial average function tester class"""

    def setUp(self):
        """set up variables for testing"""
        self.test_data2d = make_random_blob(2, 128)
        self.test_data3d = make_random_blob(3, 128)

    def test_returns2d(self):
        """make sure 2d data turns into 1d"""
        radavg = radialavg(self.test_data2d)
        assert_equals(radavg.ndim, 1)

    def test_returns3d(self):
        """make sure 3d data turns into 2d"""
        radavg = radialavg(self.test_data3d)
        assert_equals(radavg.ndim, 2)

    def test_returns2d_odd(self):
        """make sure 2d data turns into 1d that is odd in length"""
        radavg = radialavg(self.test_data2d)
        assert_true(radavg.size % 2)

    def test_returns3d_odd(self):
        """make sure 3d data turns into 2d that is odd along the r direction."""
        radavg = radialavg(self.test_data3d)
        assert_true(radavg.shape[1] % 2)

    def test_self_consistency2d(self):
        """Make sure that expanded radialavg makes sense"""
        # radialavg
        radavg = radialavg(self.test_data2d)
        # expand
        expanded = expand_radialavg(radavg)
        center = np.array(expanded.shape) // 2
        # test center lines
        assert_allclose(expanded[center[0], center[1]:], radavg)
        assert_allclose(expanded[center[0], center[1]:],
                        expanded[center[0]:, center[1]])

    def test_self_consistency3d(self):
        """Make sure that expanded radialavg is radially averaged to the
        same thing, 3d"""
        # radialavg
        radavg = radialavg(self.test_data3d)
        # expand
        expanded = expand_radialavg(radavg)
        center = np.array(expanded.shape) // 2
        # test center lines
        assert_allclose(expanded[:, center[1], center[2]:], radavg)
        assert_allclose(expanded[:, center[1], center[2]:],
                        expanded[:, center[1]:, center[2]])


class TestEnsurepositive(unittest.TestCase):
    """test ensure positive for proper behaviour"""

    def setUp(self):
        """setup internal variables for test"""
        self.test_data = np.random.randn(1024)
        self.test_data_copy = self.test_data.copy()
        self.test_data_positive = _ensure_positive(self.test_data)

    def test_action(self):
        """Make sure it works as advertise"""
        assert (self.test_data_positive >= 0).all()

    def test_ensurepositive_copy(self):
        """make sure that we return a copy of the data and that the original data
        has not been changed"""
        assert self.test_data is not self.test_data_positive
