from mldftdat.models import map_glh2
from pyscf import gto
from mldftdat.dft.numint import setup_rks_calc, setup_uks_calc
from mldftdat.workflow_utils import SAVE_ROOT
import os

BASIS = 'def2-qzvppd'
#BASIS = 'def2-tzvpp'

import numpy as np

c=np.array([0.12209126818937932, 0.24700740850425262, -0.20082448078454362, 0.395771904554465, 0.9059549427421132, 0.09159739269279044, -0.0029410239445279984, 1.1326625558748447, -0.7300578971637037, 1.122899384622471, -0.9285897309515114, 0.9419945468153514, -0.44712394983395143, 0.9509195132887669, -1.4154434872832553, -0.09352067367740347, -1.024548672972287, -0.12627662540157303, -0.21762188272617777, -1.5659719923991418, -1.499420682805905, 0.7919180469615235, 0.17238967362814606, 1.1171472472987034, -0.984495323932066, -0.1194773234099813, -0.11824659999840037, -0.21413644741670623, -0.9417719186850491, 0.8824380437510797]
)

corr_model = map_glh2.VSXCContribs(c)
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

