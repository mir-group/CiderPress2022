# derivative check for CIDER with semi-local correlation

from mldftdat.dft.numint import run_mlscf
from pyscf import gto, scf, dft
from mldftdat.workflow_utils import SAVE_ROOT
import os
import numpy as np

mol = gto.Mole(atom='Co', basis='def2-qzvppd', spin=3)
mol.ecp = 'def2-qzvppd'
mol.build()
hf = scf.UHF(mol)
hf.kernel()
ks = dft.UKS(mol)
ks.xc = 'PBE'
ks.conv_tol = 1e-7
ks.kernel()
mf = run_mlscf(mol, 'UKS', SAVE_ROOT, 'CIDER_B3LYP_AHW')
"""
mf.conv_tol = 1e-8
mf.damp = 4
mf.diis_start_cycle = 20
mf.max_cycle = 200
mf.kernel()
"""
def check_dm(s,i,j):
    dm0 = mf.make_rdm1()
    dm1 = dm0.copy()
    delta = np.maximum(dm0[s,i,j], 1) * 1e-8
    dm1[s,i,j] += delta
    n0, e0, v0 = mf._numint.nr_uks(mf.mol, mf.grids, None, dm0)
    n1, e1, v1 = mf._numint.nr_uks(mf.mol, mf.grids, None, dm1)
    print(delta, (e1-e0)/delta, e1, e0, v0[s,i,j], dm0[s,i,j])
    print('ERR', s,i,j, (e1-e0)/delta - v0[s,i,j])

check_dm(0,5,5)
check_dm(1,5,5)
check_dm(0,10,20)
check_dm(1,10,20)
check_dm(0,60,60)
check_dm(1,60,60)
check_dm(0,20,80)
check_dm(1,20,80)

