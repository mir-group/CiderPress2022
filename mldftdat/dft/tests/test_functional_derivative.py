import numpy as np
from numpy.testing import assert_almost_equal
from pyscf import gto
from mldftdat.dft.xc_models import NormGPFunctional
import os

def check_spin(mf):
    dmu = mf.make_rdm1()
    dmd = dmu[[1,0],:,:]
    vu = mf.get_veff(dm=dmu)
    eu = vu.exc
    vu -= vu.vj
    vd = mf.get_veff(dm=dmd)
    ed = vd.exc
    vd -= vd.vj
    maxerr_v = np.max(np.abs(vu-vd[[1,0],:]))
    print('SPIN', eu, ed, eu-ed, maxerr_v)
    return abs(eu-ed), maxerr_v

def check_dm_uks(mf,s,i,j):
    dm0 = mf.make_rdm1()
    dm1 = dm0.copy()
    delta = np.maximum(dm0[s,i,j], 1) * 1e-8
    dm1[s,i,j] += delta / 2
    dm1[s,j,i] += delta / 2
    v0 = mf.get_veff(dm=dm0)
    e0 = v0.exc
    v0 -= v0.vj
    v1 = mf.get_veff(dm=dm1)
    e1 = v1.exc
    v1 -= v1.vj
    err = (e1-e0)/(delta) - v0[s,i,j]
    print('ERR', s,i,j, err)
    return err

def check_dm_rks(mf,i,j):
    dm0 = mf.make_rdm1()
    dm1 = dm0.copy()
    delta = np.maximum(dm0[i,j], 1) * 1e-8
    dm1[i,j] += delta / 2
    dm1[j,i] += delta / 2
    v0 = mf.get_veff(dm=dm0)
    e0 = v0.exc
    v0 -= v0.vj
    v1 = mf.get_veff(dm=dm1)
    e1 = v1.exc
    v1 -= v1.vj
    err = (e1-e0)/(delta) - v0[i,j]
    print('ERR', i,j, err)
    return err

BAS='def2-tzvp'
functional = 'CIDER_B3LYP_AHW'
mol = gto.M(atom='O', basis=BAS, spin=2, verbose=4)

from mldftdat.dft import numint

settings = {
    'mlfunc_file': 'SPLINE_A_HEG_WIDE.joblib',
    'xc': '.08*SLATER + .72*B88, .81*LYP + .19*VWN',
    'xmix': 0.2
}

mlfunc = NormGPFunctional.load('functionals/B3LYP_CIDER.yaml')

ks = numint.setup_uks_calc(mol, mlfunc, **settings)
ks.kernel()

err_e, err_v = check_spin(ks)
assert_almost_equal(err_e, 0, 10)
assert_almost_equal(err_v, 0, 10)

mol2 = gto.M(atom='H 0 0 0; F 0 0 0.93', basis=BAS, spin=0, verbose=4)
ks2 = numint.setup_rks_calc(mol2, mlfunc, **settings)
ks2.kernel()

for i in range(1,6):
    for s in range(2):
        err = check_dm_uks(ks,s,i,i)
        assert_almost_equal(err, 0, 5)
    err = check_dm_rks(ks2,i,i)
    assert_almost_equal(err, 0, 5)

for i in range(0,7,2):
    for j in range(i,7,2):
        for s in range(2):
            err = check_dm_uks(ks,s,i,j)
            assert_almost_equal(err, 0, 5)
        err = check_dm_rks(ks2,i,j)
        assert_almost_equal(err, 0, 5)
