from pyscf import scf, gto, lib
from nose import SkipTest
from nose.tools import nottest
from nose.plugins.skip import Skip
from numpy.testing import assert_almost_equal, assert_equal

from mldftdat.xcutil.transform_data import *
import numpy as np
import numbers
import os

TMP_TEST = 'test_files/tmp'

TEST_VEC = np.array([[0.0, 0.4, 8.0],
                     [0.0, 0.8, 5.0],
                     [0.0, 2.1, 3.0],
                     [0.0, 5.0, 20.0],
                     [-12.0, 0.0, 5.0],
                     [0.0, 5.0, 20.0]])

def get_grad_fd(vec, feat_list, delta=1e-8):
    deriv = np.zeros(vec.shape)
    for i in range(vec.shape[0]):
        dvec = vec.copy()
        dvec[i] += delta
        deriv[i] += np.sum((feat_list(dvec.T).T - feat_list(vec.T).T) / delta, axis=0)
    return deriv

def get_grad_a(vec, feat_list):
    deriv = np.zeros(vec.shape)
    feat = np.zeros((feat_list.nfeat, vec.shape[1]))
    feat_list.fill_vals_(feat, vec)
    fderiv = np.ones(feat.shape)
    feat_list.fill_derivs_(deriv, fderiv, vec)
    return deriv

class TestFeatureNormalizer():

    def test_umap(self):
        flist = FeatureList([UMap(0, 0, 0.25), UMap(1, 3, 0.25)])
        gradfd = get_grad_fd(TEST_VEC, flist)
        grada = get_grad_a(TEST_VEC, flist)
        assert_almost_equal(gradfd, grada)

    def test_vmap(self):
        flist = FeatureList([VMap(0, 1, 0.25, scale=2.0, center=1.0),
                             VMap(1, 5, 0.5, scale=2.0, center=1.0)])
        gradfd = get_grad_fd(TEST_VEC, flist)
        grada = get_grad_a(TEST_VEC, flist)
        assert_almost_equal(gradfd, grada)

    def test_wmap(self):
        flist = FeatureList([WMap(0, 0, 1, 4, 0.5, 0.5)])
        gradfd = get_grad_fd(TEST_VEC, flist)
        grada = get_grad_a(TEST_VEC, flist)
        assert_almost_equal(gradfd, grada)

    def test_xmap(self):
        flist = FeatureList([XMap(0, 0, 1, 4, 0.5, 0.5)])
        gradfd = get_grad_fd(TEST_VEC, flist)
        grada = get_grad_a(TEST_VEC, flist)
        assert_almost_equal(gradfd, grada)

    def test_ymap(self):
        flist = FeatureList([YMap(0, 0, 1, 3, 4, 0.5, 0.3, 0.2)])
        gradfd = get_grad_fd(TEST_VEC, flist)
        grada = get_grad_a(TEST_VEC, flist)
        assert_almost_equal(gradfd, grada)

    def test_zmap(self):
        flist = FeatureList([ZMap(0, 1, 0.25, scale=2.0, center=1.0),
                             ZMap(1, 5, 0.5, scale=2.0, center=1.0)])
        gradfd = get_grad_fd(TEST_VEC, flist)
        grada = get_grad_a(TEST_VEC, flist)
        assert_almost_equal(gradfd, grada)

    def test_integration(self):
        flist = FeatureList([UMap(0, 0, 0.25), UMap(1, 3, 0.25),
                             VMap(1, 1, 0.25, scale=2.0, center=1.0),
                             WMap(2, 0, 1, 4, 0.5, 0.5),
                             XMap(3, 0, 1, 4, 0.5, 0.5),
                             YMap(4, 0, 1, 3, 4, 0.5, 0.3, 0.2)])