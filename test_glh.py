from mldftdat.models import map_glh
from pyscf import gto
from mldftdat.dft.glh_corr import setup_rks_calc, setup_uks_calc
from mldftdat.workflow_utils import SAVE_ROOT
import os

BASIS = 'def2-qzvppd'
#BASIS = 'def2-tzvpp'

import numpy as np

c = np.array([-0.22039573923869682, -0.17590781860481286, -0.20493729688694629, 0.8175838270111342, 1.1100168270101243, 0.1882321981553403, -0.15084703956412682, 0.922285325379562, -0.9199522032213849, 1.2312954385629666, -1.37649411079191, 0.43977254026810897, -0.09208488020766481, 0.7956898876179821, -0.5301598314922273, -0.9348283848106576, -0.2704210332626644, -0.08791736156541674, 0.1713262762289105, -1.3160652110382784])

corr_model = map_glh.VSXCContribs(c)
from joblib import dump, load
dump(corr_model, os.path.join(SAVE_ROOT, 'MLFUNCTIONALS', 'GLH', 'HHGGA_ITER0.joblib'))

corr_model = load(os.path.join(SAVE_ROOT, 'MLFUNCTIONALS', 'GLH', 'HHGGA_ITER0.joblib'))

mol = gto.M(atom='H', basis=BASIS, spin=1)
calc = setup_uks_calc(mol, corr_model)
#calc.with_df.atom_grid = {'H': (35,86), 'O': (35,86)}
E_H = calc.kernel()

mol = gto.M(atom='O', basis=BASIS, spin=2)
calc = setup_uks_calc(mol, corr_model)
#calc.with_df.atom_grid = {'H': (35,86), 'O': (35,86)}
E_O = calc.kernel()

mol = gto.M(
    atom='''O    0.   0.       0.
            H    0.   -0.757   0.587
            H    0.   0.757    0.587
    ''',
    basis = BASIS,
)

calc = setup_rks_calc(mol, corr_model)
#calc.with_df.atom_grid = {'H': (35,86), 'O': (35,86)}
E_H2O = calc.kernel()

print(2*E_H+E_O-E_H2O)

