from mldftdat.models import map_glh
from pyscf import gto, dft
from mldftdat.dft.glh_corr import setup_rks_calc, setup_uks_calc
from mldftdat.workflow_utils import SAVE_ROOT
import os
from joblib import load

BASIS = 'def2-qzvppd'
#BASIS = 'def2-tzvpp'

import numpy as np

#corr_model = load(os.path.join(SAVE_ROOT, 'MLFUNCTIONALS', 'GLH', 'HHGGA_ALT.joblib'))

c = np.array(
[-0.2415724413135365, -0.17712220930242495, -0.22188722056386467, 0.7508844614764811, 1.169530511564787, 0.19949931461316694, -0.15945122573025827, 0.8992853934859397, -0.985860136390329, 1.2532517468495286, -1.2386514419428352, 0.4455556850002722, 0.040394457977527054, 0.7999035945331343, -0.5135519848467425, -1.020713654581641, -0.25778849712582996, -0.2061101105492753, 0.15619285571062846, -1.315941066335455])
c=np.array([-0.22039573923869682, -0.17590781860481286, -0.20493729688694629, 0.8175838270111342, 1.1100168270101243, 0.1882321981553403, -0.15084703956412682, 0.922285325379562, -0.9199522032213849, 1.2312954385629666, -1.37649411079191, 0.43977254026810897, -0.09208488020766481, 0.7956898876179821, -0.5301598314922273, -0.9348283848106576, -0.2704210332626644, -0.08791736156541674, 0.1713262762289105, -1.3160652110382784])
corr_model = map_glh.VSXCContribs(c)

#mol = gto.M(atom='O', basis=BASIS, spin=2)
mol = gto.M(atom='P', basis=BASIS, spin=3)

ks = dft.UKS(mol)
ks.xc = 'PBE'
ks.kernel()

calc = setup_uks_calc(mol, corr_model)
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

calc = setup_uks_calc(mol, corr_model)
calc.build()
E_static_p = calc.energy_tot(dm=ks.make_rdm1())
E_O_p = calc.kernel()

print(E_static_p-E_static, E_O_p-E_O)


