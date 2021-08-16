import numpy as np
from numpy.testing import assert_almost_equal
from pyscf import gto, dft
from mldftdat.dft.xc_models import NormGPFunctional, GPFunctional
import os
import joblib

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
mol = gto.M(atom='Ne', basis=BAS, spin=0, verbose=4)

ks_ref = dft.RKS(mol)
ks_ref.xc = 'PBE0'
ks_ref.kernel()

from mldftdat.dft import numint

# Construct settings. 'xc' should compute all XC contributions EXCEPT CIDER.
# xmix should be the fraction of CIDER.
# The below settings represent PBE0 with 25% HF replaced with 25% CIDER.
settings = {
    'xc': 'GGA_C_PBE + 0.75*GGA_X_PBE',
    'xmix': 0.25
}

##### EXAMPLE 1: Simple GP Functional #####
# WARNING: Un-mapped GP functional provided as reference,
# but functional derivatives are buggy.
# Can load functionals using joblib
mlfunc = joblib.load('test_files/gpr_example.joblib')
mlfunc = GPFunctional(mlfunc)

ks = numint.setup_rks_calc(mol, mlfunc, **settings)
ks.kernel()

# This routine tests the functional derivatives.
# There are significant errors for the pure GP functional,
# but not for the spline functional.
for i in range(0,10):
    check_dm_rks(ks,i,i)

##### EXAMPLE 2: Simple Spline Functional #####
# Spline-based functionals are faster.
mlfunc = joblib.load('test_files/agpr_spline_example.joblib')
ks = numint.setup_rks_calc(mol, mlfunc, **settings)
ks.kernel()

for i in range(0,10):
    check_dm_rks(ks,i,i)

##### EXAMPLE 3: Accurate CIDER Functional #####
mlfunc = NormGPFunctional.load('functionals/B3LYP_CIDER.yaml')
ks = numint.setup_rks_calc(mol, mlfunc, **settings)
ks.kernel()

for i in range(0,10):
    check_dm_rks(ks,i,i)
