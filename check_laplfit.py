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


xcpbe = eval_xc('PBE,', analyzer.rho_data)[0] * rho
neps = analyzer.get_fx_energy_density()
diff = neps - xcpbe
"""
analyzer.setup_lapl_basis()
Cdiff = analyzer.fit_vals_to_lapl(diff)
fx2 = np.dot(analyzer.aux_d2, Cdiff)
"""
neps_pred = analyzer.get_smooth_fx_energy_density()
fx2 = neps - neps_pred

print(np.dot(np.abs(fx2 - diff), analyzer.grid.weights))
print(np.dot(fx2 - diff, analyzer.grid.weights))
print(np.dot(fx2, analyzer.grid.weights), 'should be 0')

from mldftdat import data
import matplotlib.pyplot as plt

"""
data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords,
                    diff, 'conv-pbe',
                    'Ha/Bohr$^3$', [-2,5])
data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords,
                    fx2, 'fit',
                    'Ha/Bohr$^3$', [-2,5])
data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords,
                    fx2 - diff, 'fit',
                    'Ha/Bohr$^3$', [-2,5])
plt.show()

data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords,
                    neps, 'exact',
                    'Ha/Bohr$^3$', [-2,5])
data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords,
                    xcpbe, 'PBE',
                    'Ha/Bohr$^3$', [-2,5])
data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords,
                    neps - fx2, 'exact smooth',
                    'Ha/Bohr$^3$', [-2,5])
plt.show()
"""

cond = rho > 1e-2
fig, axs = plt.subplots(1,3, figsize=(12, 3))
data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords[cond],
                    neps[cond] / (data.ldax(rho[cond]) + 1e-7), 'exact',
                    'Ha/Bohr$^3$', [-2,5], ax = axs[0])
data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords[cond],
                    xcpbe[cond] / (data.ldax(rho[cond]) + 1e-7), 'SCAN',
                    'Ha/Bohr$^3$', [-2,5], ax = axs[1])
data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords[cond],
                    (neps_pred)[cond] / (data.ldax(rho[cond]) + 1e-7), 'exact smooth',
                    'Ha/Bohr$^3$', [-2,5], ax = axs[2])
for i in range(3):
    axs[i].set_ylim(0.6, 1.6)
plt.tight_layout()
plt.show()

data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords[cond],
                    (neps_pred - xcpbe)[cond] / (data.ldax(rho[cond]) + 1e-7), 'exact smooth',
                    'Ha/Bohr$^3$', [-2,5])
plt.show()
