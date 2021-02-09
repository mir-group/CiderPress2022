from mldftdat.models import map_glh
from pyscf import gto
from mldftdat.dft.glh_corr import setup_rks_calc, setup_uks_calc
from mldftdat.workflow_utils import SAVE_ROOT
import os

BASIS = 'def2-qzvppd'
#BASIS = 'def2-tzvpp'

import numpy as np

#c = np.array([-0.27138309, -0.21281959, -0.15623003,  1.24502537,  1.36510367,  0.17758945,
# -0.15952195,  0.9210639,  -0.79849808,  1.04656752, -0.93030217,  0.3040587,
# -0.06360054, 0.62487103, -0.39025499, -0.67329425, -0.27931914, -0.00622262,
#  0.16108911, -1.3067447 ])
#c = np.array([-0.35118628, -0.04116897, -0.1170049,   0.86801398,  0.64597561,  0.18425069,
# -0.13548765,  0.78579114, -0.55266733,  0.65062711, -0.4354578,  -0.11685605,
#  0.23603571,  0.4448409,  -0.5634864,  -0.23806251, -0.11930092, -0.10052114,
# -0.00996895, -1.2325281 ])
c = np.array([-0.51503662, -0.08172883, -0.2087729,   1.0914231,   0.85512226,  0.17399686,
 -0.13850834,  0.84846596, -0.62145956,  0.7605562,  -0.55463653, -0.03137007,
  0.24104494,  0.55393462, -0.56450321, -0.35437724, -0.13770352, -0.1233943,
  0.01527737, -1.2672252 ])
#c = np.array([-0.94546821,  0.03325448, -0.62417548,  1.26081618,  1.41712944,  0.17136032,
# -0.15442698,  0.93881024, -0.81708588,  0.92286681, -0.67492348,  0.45673972,
# -0.02563903,  0.70418737, -0.5186625,  -0.60217404, -0.11218421,  0.03068857,
#  0.27292559, -1.31344457])

corr_model = map_glh.VSXCContribs(c)
from joblib import dump, load
#dump(corr_model, os.path.join(SAVE_ROOT, 'MLFUNCTIONALS', 'GLH', 'TEST2.joblib'))

#corr_model = load(os.path.join(SAVE_ROOT, 'MLFUNCTIONALS', 'GLH', 'TEST.joblib'))

mol = gto.M(atom='H', basis=BASIS, spin=1)
calc = setup_uks_calc(mol, corr_model)
E_H = calc.kernel()

mol = gto.M(atom='O', basis=BASIS, spin=2)
calc = setup_uks_calc(mol, corr_model)
E_O = calc.kernel()

mol = gto.M(
    atom='''O    0.   0.       0.
            H    0.   -0.757   0.587
            H    0.   0.757    0.587
    ''',
    basis = BASIS,
)

calc = setup_rks_calc(mol, corr_model)
E_H2O = calc.kernel()

print(2*E_H+E_O-E_H2O)

