from mldftdat.loc_analyzers import RHFAnalyzer
import numpy as np 
from pyscf.dft.libxc import eval_xc
from mldftdat.pyscf_utils import get_single_orbital_tau, get_gradient_magnitude,\
    get_hartree_potential

A = 0.704 # maybe replace with sqrt(6/5)?
B = 2 * np.pi / 9 * np.sqrt(6.0/5)
FXP0 = 27 / 50 * 10 / 81
FXI = 1.0 / 3 * (4*np.pi**2 / 3)**(1.0/3)
#MU = 10/81
MU = 0.21
C1 = 1.0 / 3 * (4*np.pi**2 / 3)**(1.0/3)
C2 = 1 - C1
C3 = 0.19697 * np.sqrt(0.704)
C4 = (C3**2 - 0.09834 * MU) / C3**3

def edmgga(rho_data):
    print(rho_data.shape)
    gradn = np.linalg.norm(rho_data[1:4], axis=0)
    tau0 = 3 / 10 * 3 * np.pi**2 * rho_data[0]**(5/3) + 1e-6
    tauw = 1 / 8 * gradn**2 / (rho_data[0] + 1e-6)
    QB = tau0 - rho_data[5] + tauw + 0.25 * rho_data[4]
    QB /= tau0
    x = A * QB + np.sqrt(1 + (A*QB)**2)
    FX = C1 + (C2 * x) / (1 + C3 * np.sqrt(x) * np.arcsinh(C4 * (x-1)))
    return FX

def edmgga_loc(rho_data):
    print(rho_data.shape)
    gradn = np.linalg.norm(rho_data[1:4], axis=0)
    tau0 = 3 / 10 * 3 * np.pi**2 * rho_data[0]**(5/3) + 1e-6
    tauw = 1 / 8 * gradn**2 / (rho_data[0] + 1e-6)
    QB = tau0 - rho_data[5] + 0.125 * rho_data[4]
    QB /= tau0
    x = A * QB + np.sqrt(1 + (A*QB)**2)
    FX = C1 + (C2 * x) / (1 + C3 * np.sqrt(x) * np.arcsinh(C4 * (x-1)))
    return FX

analyzer = RHFAnalyzer.load('test_files/RHF_HF.hdf5')
edmgga(analyzer.rho_data)
rho = analyzer.rho_data[0,:]


#vh = get_hartree_potential(analyzer.rho_data, analyzer.grid.coords, analyzer.grid.weights)

eps = - analyzer.get_fx_energy_density() / (rho + 1e-7)
#eps = - loc_dens / (rho + 1e-7)
analyzer.setup_rho_basis()
Ceps, tst = analyzer.fit_vals_to_aux2(eps)
print(tst.shape)
fx2 = np.dot(analyzer.aux_v, Ceps)
#fx2 = get_hartree_potential(tst[:4], analyzer.grid.coords, analyzer.grid.weights)
#fx2, gradfx2 = fx2[0], fx2[1:]
print(np.dot(np.abs(fx2 - eps), analyzer.grid.weights))
print(np.dot(rho * fx2, analyzer.grid.weights))
print(np.dot(analyzer.get_fx_energy_density(), analyzer.grid.weights))

"""
tauw = get_single_orbital_tau(rho, get_gradient_magnitude(analyzer.rho_data))
tau = analyzer.rho_data[5]
cterm = ( rho / (eps**3 + 1e-7) )**2
zeta = 0.015 * rho / (eps**2 + 1e-7) / (1 + 0.04 * cterm) * (tauw / (tau + 1e-7))**4
Czeta = analyzer.fit_vals_to_aux(zeta)

gradeps = np.dot(analyzer.aux_ao[1:], Ceps)
nabla = np.sum(analyzer.aux_ao[(4,7,9),:,:], axis=0)
nablaeps = np.dot(nabla, Ceps)
zeta2 = np.dot(analyzer.aux_ao[0], Czeta)
gradzeta = np.dot(analyzer.aux_ao[1:], Czeta)
gfunc = np.sum(gradzeta * gradeps, axis=0) + zeta * nablaeps
"""

from mldftdat import data
import matplotlib.pyplot as plt

#data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords,
#                    vh[0], 'vh', 'a.u.', [-2,5])
#data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords,
#                    np.linalg.norm(vh[1:4], axis=0), 'vh', 'a.u.', [-2,5])
#plt.show()

data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords,
                    tst[0], 'rho', 'a.u.', [-2,5])
plt.show()

data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords,
                    fx2, 'fit',
                    'Ha/Bohr$^3$', [-2,5])
data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords,
                    eps, 'conv',
                    'Ha/Bohr$^3$', [-2,5])
plt.show()