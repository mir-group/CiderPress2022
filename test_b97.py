from mldftdat.dft.correlation import *
from pyscf.dft import RKS
from pyscf import gto
import numpy as np
from pyscf.dft.libxc import eval_xc
from mldftdat.analyzers import RHFAnalyzer
from mldftdat.pyscf_utils import get_uniform_tau

SPIN = 0
mol = gto.Mole(atom='Ar', basis='aug-cc-pvqz', spin=SPIN)
mol.build()
rks = RKS(mol)
rks.xc = 'B97M-V'
rks.kernel()

from mldftdat.pyscf_utils import get_mgga_data

ao_data, rho_data = get_mgga_data(mol, rks.grids, rks.make_rdm1())
numint = ProjNumInt()
rks._numint = numint
rks.kernel()
analyzer = RHFAnalyzer(rks)
print(analyzer.fx_total)
ao_data, rho_data = get_mgga_data(mol, rks.grids, rks.make_rdm1())
e, v, _, _ = numint.eval_xc('', rho_data, spin=SPIN)

print(rks._numint.nlc_coeff('B97M-V'))

eref, vref, _, _ = eval_xc('B97M-V', rho_data)
_, Evv, _ = nr_rks_vv10(rks._numint, mol, rks.grids, 'B97M-V', rks.make_rdm1(), b = 6.0, c = 0.01)

#print(e.shape, eref.shape, rho.shape, rks.grids.weights.shape)

print(e.shape, eref.shape, rks.grids.weights.shape)
print(np.dot(np.abs(e - eref) * rho_data[0], rks.grids.weights))
print(np.dot(np.abs(e) * rho_data[0], rks.grids.weights))
print(np.dot(np.abs(eref) * rho_data[0], rks.grids.weights))
eb97 = np.dot(vref[1] * np.linalg.norm(rho_data[1:4],axis=0)**2, rks.grids.weights)
eest = np.dot(v[1] * np.linalg.norm(rho_data[1:4],axis=0)**2, rks.grids.weights)
#eb97 = np.dot(vref[3] * rho_data[5], rks.grids.weights)
#eest = np.dot(v[3] * rho_data[5], rks.grids.weights)
eb97 = np.dot(vref[0] * rho_data[0], rks.grids.weights)
eest = np.dot(v[0] * rho_data[0], rks.grids.weights)
#eb97 = np.dot(eref * rho_data[0], rks.grids.weights)
#eest = np.dot(e * rho_data[0], rks.grids.weights)
print(eb97, eest)

#numint = ProjNumInt(xterms = [], ssterms = [], osterms = [])
pseudo_rho = rho_data.copy()
pseudo_rho[1:4] = 0
pseudo_rho[5] = get_uniform_tau(pseudo_rho[0])
e, v, _, _ = numint.eval_xc('', pseudo_rho)
#eref, vref, _, _ = eval_xc('LDA,LDA_C_PW', rho_data)
eref, vref, _, _ = eval_xc('B97M-V', pseudo_rho)
print(np.dot(e * rho_data[0], rks.grids.weights))
Eref = np.dot(eref * rho_data[0], rks.grids.weights) 
print('chk')
print(np.dot(vref[0] * rho_data[0], rks.grids.weights)\
    -np.dot(v[0] * rho_data[0], rks.grids.weights))
print(np.dot(eref * rho_data[0], rks.grids.weights))
print(np.dot(e * rho_data[0], rks.grids.weights) - Eref)
print(eb97 - Eref)
print(eest - Eref)
print(eb97 - eest)

#b97mv_wx_ss := (t, dummy) -> (K_FACTOR_C - t)/(K_FACTOR_C + t):
#b97mv_wx_os := (ts0, ts1) -> (K_FACTOR_C*(ts0 + ts1) - 2*ts0*ts1)/(K_FACTOR_C*(ts0 + ts1) + 2*ts0*ts1):
#3/10*(6*pi^2)^(2/3)
#UEG_FACTOR = 0.3*(6*np.pi**2)**(2/3)
"""
w = (UEG_FACTOR - t) / (UEG_FACTOR + t)
    if i > 1:
        w = (UEG_FACTOR * (tu + td) - 2 * tu * td) / (UEG_FACTOR * (tu + td) + 2 * tu * td)
    i += 1
"""
