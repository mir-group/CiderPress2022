from mldftdat.models import map_glh
from pyscf import gto
from mldftdat.dft.numint import setup_rks_calc, setup_uks_calc
from mldftdat.workflow_utils import SAVE_ROOT
import os

#BASIS = 'def2-qzvppd'
BASIS = 'def2-tzvpp'

import numpy as np

c = np.array([ 0.09458185,  0.42005969, -0.53499793,  0.39486842,  0.47500668,  0.14274847,
 -0.04226082,  0.6947042,  -0.66793513,  0.76889933, -0.5669118,   0.19841883,
  0.40328066,  0.72598104, -1.18196255, -0.34286538, -0.43444723, -0.10741351,
 -0.05860007, -1.18938227])

corr_model = map_glh.VSXCContribs(c)
from joblib import dump, load
dump(corr_model, os.path.join(SAVE_ROOT, 'MLFUNCTIONALS', 'GLH', 'CIDER_ITER0.joblib'))

corr_model = load(os.path.join(SAVE_ROOT, 'MLFUNCTIONALS', 'GLH', 'CIDER_ITER0.joblib'))
mlfunc = load("/n/holystore01/LABS/kozinsky_lab/Lab/Data/MLDFTDBv3/MLFUNCTIONALS/CIDER/SPLINE_A_HEG_WIDE.joblib")

mol = gto.M(atom='H', basis=BASIS, spin=1)
calc = setup_uks_calc(mol, mlfunc, corr_model)
E_H = calc.kernel()

mol = gto.M(atom='O', basis=BASIS, spin=2)
calc = setup_uks_calc(mol, mlfunc, corr_model)
E_O = calc.kernel()

mol = gto.M(
    atom='''O    0.   0.       0.
            H    0.   -0.757   0.587
            H    0.   0.757    0.587
    ''',
    basis = BASIS,
)

calc = setup_rks_calc(mol, mlfunc, corr_model)
E_H2O = calc.kernel()

print(2*E_H+E_O-E_H2O)

