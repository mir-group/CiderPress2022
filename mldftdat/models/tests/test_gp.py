from nose import SkipTest
from nose.tools import nottest
from nose.plugins.skip import Skip
from numpy.testing import assert_almost_equal, assert_equal

import numpy as np

import unittest

class TestGPR(unittest.TestCase):

    def test_fit(self):
        # from https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy_targets.html
        pass
