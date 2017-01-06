from nose.tools import *
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import unittest
from pyDecon.utils import _ensure_positive, radialavg, expand_radialavg


class TestRadialAverage(unittest.TestCase):
    """Radial average function tester class"""

    def setUp(self):
        """set up variables for testing"""
        self.test_data2d = np.random.randn(128, 128)
        self.test_data3d = np.random.randn(128, 128, 128)

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
        """Make sure that expanded radialavg is radially averaged to
        the same thing"""
        radavg = radialavg(self.test_data2d)
        expanded = expand_radialavg(radavg)
        expanded_avg = radialavg(expanded)
        assert_allclose(expanded_avg, radavg)

    def test_self_consistency3d(self):
        """Make sure that expanded radialavg is radially averaged to the
        same thing"""
        radavg = radialavg(self.test_data3d)
        expanded = expand_radialavg(radavg)
        expanded_avg = radialavg(expanded)
        assert_allclose(expanded_avg, radavg)


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
