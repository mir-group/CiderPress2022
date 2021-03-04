# derivative check for HHGGA with SGX-HF

from pyscf import gto, scf, dft
from mldftdat.dft.numint import setup_rks_calc, setup_uks_calc
from mldftdat.workflow_utils import SAVE_ROOT
import os
from joblib import load
import numpy as np
from pyscf.sgx.sgx import sgx_fit

corr_model = load(os.path.join(SAVE_ROOT, 'MLFUNCTIONALS', 'GLH', 'CIDER_BIG_ITER0.joblib'))
mlfunc = load("/n/holystore01/LABS/kozinsky_lab/Lab/Data/MLDFTDBv3/MLFUNCTIONALS/CIDER/SPLINE_A_HEG_WIDE.joblib")

mol = gto.Mole(atom='Cl', basis='def2-qzvppd', spin=1, verbose=4)
mol.ecp = 'def2-qzvppd'
mol.build()
hf = scf.UHF(mol)
hf.kernel()
ks = dft.UKS(mol)
ks.xc = 'PBE'
ks.conv_tol = 1e-7
ks.kernel()

mf = setup_uks_calc(mol, mlfunc, corr_model)

#ks.xc = 'HF'
#mf = sgx_fit(ks)

mf.damp = 10
mf.diis_start_cycle = 30
mf.max_cycle = 150
mf.kernel(ks.make_rdm1())

"""
mf.conv_tol = 1e-8
mf.damp = 4
mf.diis_start_cycle = 20
mf.max_cycle = 200
mf.kernel()
"""

def check_spin():
    dmu = mf.make_rdm1()
    dmd = dmu[[1,0],:,:]
    vu = mf.get_veff(dm=dmu)
    eu = vu.exc
    vu -= vu.vj
    vd = mf.get_veff(dm=dmd)
    ed = vd.exc
    vd -= vd.vj
    print('SPIN', eu, ed, eu-ed, np.max(np.abs(vu-vd[[1,0],:])))

def check_dm(s,i,j):
    dm0 = mf.make_rdm1()
    dm1 = dm0.copy()
    delta = np.maximum(dm0[s,i,j], 1) * 1e-8
    #delta = dm0[s,i,j] * 1e-8
    dm1[s,i,j] += delta / 2
    dm1[s,j,i] += delta / 2
    #dm2[s,i,j] -= delta
    v0 = mf.get_veff(dm=dm0)
    e0 = v0.exc
    v0 -= v0.vj
    v1 = mf.get_veff(dm=dm1)
    e1 = v1.exc
    v1 -= v1.vj
    #print(delta, (e1-e0)/(delta), e1, e0, v0[s,i,j], dm0[s,i,j])
    print('ERR', s,i,j, (e1-e0)/(delta) - v0[s,i,j])

check_spin()
check_dm(0,5,5)
check_dm(1,5,5)
check_dm(0,10,20)
check_dm(1,10,20)
check_dm(0,60,60)
check_dm(1,60,60)
check_dm(0,20,78)
check_dm(1,20,78)


