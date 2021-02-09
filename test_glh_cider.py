from mldftdat.models import map_glh
from pyscf import gto
from mldftdat.dft.numint import setup_rks_calc, setup_uks_calc
from mldftdat.workflow_utils import SAVE_ROOT
import os

#BASIS = 'def2-qzvppd'
BASIS = 'def2-tzvpp'

import numpy as np

c = np.array([ 0.08893247,  0.34775039, -0.55727308,  0.3983381,   0.47733845,  0.14178125,
 -0.04135754,  0.68747863, -0.64866956,  0.73496249, -0.47801523,  0.17958114,
  0.40809167,  0.75531126, -1.18289192, -0.37781771, -0.47749114, -0.11893029,
 -0.07837232, -1.19085653])

corr_model = map_glh.VSXCContribs(c)
from joblib import dump, load
#dump(corr_model, os.path.join(SAVE_ROOT, 'MLFUNCTIONALS', 'GLH', 'TEST2.joblib'))

#corr_model = load(os.path.join(SAVE_ROOT, 'MLFUNCTIONALS', 'GLH', 'TEST.joblib'))
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

