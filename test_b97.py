from mldftdat.dft.correlation import *
from pyscf.dft import RKS
from pyscf import gto
import numpy as np
from pyscf.dft.libxc import eval_xc

mol = gto.Mole(atom='He', basis='aug-cc-pvqz')
mol.build()
rks = RKS(mol)
rks.xc = 'B97M-V'
rks.kernel()

from mldftdat.pyscf_utils import get_mgga_data

ao_data, rho_data = get_mgga_data(mol, rks.grids, rks.make_rdm1())
numint = ProjNumInt()
e, v, _, _ = numint.eval_xc('', rho_data)

print(rks._numint.nlc_coeff('B97M-V'))

eref, vref, _, _ = eval_xc('B97M-V', rho_data)
_, Evv, _ = nr_rks_vv10(rks._numint, mol, rks.grids, 'B97M-V', rks.make_rdm1(), b = 6.0, c = 0.01)

print(e.shape, eref.shape, rks.grids.weights.shape)
print(np.dot(np.abs(e - eref) * rho_data[0], rks.grids.weights))
print(np.dot(np.abs(e) * rho_data[0], rks.grids.weights))
print(np.dot(np.abs(eref) * rho_data[0], rks.grids.weights))
eb97 = np.dot(eref * rho_data[0], rks.grids.weights)
eest = np.dot(e * rho_data[0], rks.grids.weights)

numint = ProjNumInt(xterms = [], ssterms = [], osterms = [])
e, v, _, _ = numint.eval_xc('', rho_data)
eref, vref, _, _ = eval_xc('LDA,VWN', rho_data)
print(np.dot(e * rho_data[0], rks.grids.weights))
Eref = np.dot(eref * rho_data[0], rks.grids.weights) 
print(np.dot(eref * rho_data[0], rks.grids.weights))
print(eb97 - Eref)
print(eest - Eref)
print(eb97 - (Evv + eest))