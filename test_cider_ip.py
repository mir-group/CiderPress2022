from mldftdat.models import map_glh2
from pyscf import gto, dft
from mldftdat.dft.numint import setup_rks_calc, setup_uks_calc
from mldftdat.workflow_utils import SAVE_ROOT
import os
from joblib import load

BASIS = 'def2-qzvppd'
#BASIS = 'def2-tzvpp'

import numpy as np

mlfunc = load("/n/holystore01/LABS/kozinsky_lab/Lab/Data/MLDFTDBv3/MLFUNCTIONALS/CIDER/SPLINE_A_HEG_WIDE.joblib")
corr_model = load(os.path.join(SAVE_ROOT, 'MLFUNCTIONALS', 'GLH', 'CIDER_BIG_ITER0.joblib'))

#mol = gto.M(atom='O', basis=BASIS, spin=2)
mol = gto.M(atom='P', basis=BASIS, spin=3)

ks = dft.UKS(mol)
ks.xc = 'PBE'
ks.kernel()

calc = setup_uks_calc(mol, mlfunc, corr_model)
calc.build()
E_static = calc.energy_tot(dm=ks.make_rdm1())
E_O = calc.kernel()

print(E_static, E_O, E_static-E_O)

mol.charge = 1
#mol.spin = 3
mol.spin = 2
mol.build()

ks = dft.UKS(mol)
ks.xc = 'PBE'
ks.kernel()

calc = setup_uks_calc(mol, mlfunc, corr_model)
calc.build()
E_static_p = calc.energy_tot(dm=ks.make_rdm1())
E_O_p = calc.kernel()

print(E_static_p-E_static, E_O_p-E_O)


