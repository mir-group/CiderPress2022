from mldftdat.loc_analyzers import RHFAnalyzer
import numpy as np 
from pyscf.dft.libxc import eval_xc
from mldftdat.pyscf_utils import get_single_orbital_tau, get_gradient_magnitude
from mldftdat.data import ldax, ldax_dens

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
print(analyzer.mol.atom)
edmgga(analyzer.rho_data)
rho = analyzer.rho_data[0,:]
xcscan = eval_xc('SCAN,', analyzer.rho_data)[0] * rho
xcpbe = eval_xc('PBE,', analyzer.rho_data)[0] * rho
#xcpbe = eval_xc('MGGA_X_EDMGGA,', analyzer.rho_data)[0] * rho
#analyzer.setup_etb()
loc_dens = analyzer.get_loc_fx_energy_density()
tot = np.dot(loc_dens, analyzer.small_grid.weights)
print('loc_dens data')
print(tot, analyzer.fx_total)
print(np.linalg.norm(loc_dens - analyzer.fx_energy_density))
print(np.dot( (loc_dens - analyzer.fx_energy_density)**2,
                analyzer.small_grid.weights ))

eps = - analyzer.get_fx_energy_density() / (rho + 1e-7)
#eps = - loc_dens / (rho + 1e-7)
analyzer.setup_eps_basis()
Ceps = analyzer.fit_vals_to_aux(eps)
fx2 = np.dot(analyzer.aux_ao[0], Ceps)
print(fx2.shape)
print(np.dot(np.abs(fx2 - eps), analyzer.grid.weights))
print(np.dot(fx2, analyzer.grid.weights))
print(np.dot(analyzer.get_fx_energy_density(), analyzer.grid.weights))

tauw = get_single_orbital_tau(rho, get_gradient_magnitude(analyzer.rho_data))
tau = analyzer.rho_data[5]
cterm = ( rho / (eps**3 + 1e-7) )**2
zeta = 0.015 * rho / (eps**2 + 1e-7) / (1 + 0.04 * cterm) * (tauw / (tau + 1e-7))**4
Czeta = analyzer.fit_vals_to_aux(zeta)

gradeps = np.dot(analyzer.aux_ao[1:4], Ceps)
nabla = np.sum(analyzer.aux_ao[(4,7,9),:,:], axis=0)
nablaeps = np.dot(nabla, Ceps)
zeta2 = np.dot(analyzer.aux_ao[0], Czeta)
gradzeta = np.dot(analyzer.aux_ao[1:4], Czeta)
gfunc = np.sum(gradzeta * gradeps, axis=0) + zeta * nablaeps
print(np.dot(gfunc, analyzer.grid.weights))

from mldftdat import data
import matplotlib.pyplot as plt

"""
data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords,
                    analyzer.rho_data[0], 'rho',
                    '1/Bohr$^3$', [-2,5])
plt.show()

data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords,
                    eps, 'conv',
                    '1/Bohr$^3$', [-2,5])
data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords,
                    - xcpbe / (rho + 1e-7), 'PBE',
                    '1/Bohr$^3$', [-2,5])
data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords,
                    - edmgga(analyzer.rho_data) * ldax_dens(rho), 'EDM',
                    '1/Bohr$^3$', [-2,5])
data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords,
                    eps + xcpbe / (rho + 1e-7), 'diff',
                    '1/Bohr$^3$', [-2,5])
plt.show()

data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords,
                    eps, 'conv',
                    'Ha/Bohr$^3$', [-2,5])
data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords,
                    fx2, 'fit',
                    'Ha/Bohr$^3$', [-2,5])
plt.show()

data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords,
                    zeta, 'conv',
                    'Ha/Bohr$^3$', [-2,5])
data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords,
                    zeta2, 'fit',
                    'Ha/Bohr$^3$', [-2,5])
data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords,
                    np.linalg.norm(gradzeta, axis=0), 'gradzeta',
                    'Ha/Bohr$^3$', [-2,5])
plt.show()

data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords,
                    nablaeps, 'nablaeps',
                    'Ha/Bohr$^3$', [-2,5])
data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords,
                    np.linalg.norm(gradeps, axis=0), 'gradeps',
                    'Ha/Bohr$^3$', [-2,5])
data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords,
                    gfunc, 'G',
                    'Ha/Bohr$^3$', [-2,5])
plt.show()

F = zeta * np.linalg.norm(gradeps, axis=0) / (rho + 1e-7)
data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords[rho > 1e-3],
                    F[rho > 1e-3], 'F',
                    'Ha/Bohr$^3$', [-2,5])
plt.show()

data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords[rho > 1e-3],
                    gfunc[rho > 1e-3] / (rho[rho > 1e-3] + 1e-7), 'G',
                    'Ha/Bohr$^3$', [-2,5])
data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords[rho > 1e-3],
                    (gfunc[rho > 1e-3]) / (rho[rho > 1e-3] + 1e-7) - eps[rho > 1e-3], 'Gpconv',
                    'Ha/Bohr$^3$', [-2,5])
data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords[rho > 1e-3],
                    xcpbe[rho > 1e-3] / (rho[rho > 1e-3] + 1e-7), 'PBE',
                    'Ha/Bohr$^3$', [-2,5])
plt.show()

exit(0)

data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords,
                    -analyzer.fx_energy_density, 'conv',
                    'Ha/Bohr$^3$', [-3, 6])
data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords,
                    -loc_dens, 'localized',
                    'Ha/Bohr$^3$', [-3, 6])
plt.show()

data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords,
                    analyzer.fx_energy_density / (data.ldax(rho) - 1e-7), 'conventional',
                    '$F_x$ (a.u)', [-3, 6])
data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords,
                    loc_dens / (data.ldax(rho) - 1e-7), 'localized',
                    '$F_x$ (a.u)', [-3, 6])
plt.show()
"""

condition = rho > 3e-3
fig, ax = plt.subplots()
data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords[condition],
                    loc_dens[condition] / (data.ldax(rho[condition]) - 1e-7), 'localized',
                    '$F_x$ (a.u)', [-3, 6], ax=ax)
print('hi')
ax.set_ylim(0.0, 2.5)
ax.scatter([0],[0], s=10, color='black')
ax.scatter([1.1/0.5291],[0], s=10, color='black')
ax.annotate('H', xy=(0, 0), xytext=(-0.1, 0.05), fontsize=15)
ax.annotate('F', xy=(1.1/0.5291, 0), xytext=(1.1/0.5291-0.1, 0.05), fontsize=15)
plt.show()

fig, ax = plt.subplots()
data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords[condition],
                    analyzer.fx_energy_density[condition] / (data.ldax(rho[condition]) - 1e-7),
                    'conventional', '$F_x$ (a.u)', [-3, 6])
ax.set_ylim(0.0, 2.5)
ax.scatter([0],[0], s=10, color='black')
ax.scatter([1.1/0.5291],[0], s=10, color='black')
ax.annotate('H', xy=(0, 0), xytext=(-0.1, 0.05), fontsize=15)
ax.annotate('F', xy=(1.1/0.5291, 0), xytext=(1.1/0.5291-0.1, 0.05), fontsize=15)
plt.show()
exit()

data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords[condition],
                    edmgga(analyzer.rho_data)[condition],
                    'EDM', '$F_x$ (a.u)', [-3, 6])
data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords[condition],
                    xcpbe[condition] / (data.ldax(rho[condition]) - 1e-7),
                    'PBE', '$F_x$ (a.u)', [-3, 6])
#data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords[condition],
#                    xcscan[condition] / (data.ldax(rho[condition]) - 1e-7),
#                    'SCAN', '$F_x$ (a.u)', [-3, 6])
plt.show()

data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords[condition],
                    analyzer.fx_energy_density[condition] / (data.ldax(rho[condition]) - 1e-7),
                    'conventional', '$F_x$ (a.u)', [-3, 6])
data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords[condition],
                    edmgga(analyzer.rho_data)[condition],
                    'EDM', '$F_x$ (a.u)', [-3, 6])
plt.show()

data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords[condition],
                    loc_dens[condition] / (data.ldax(rho[condition]) - 1e-7), 'localized',
                    '$F_x$ (a.u)', [-3, 6])
data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords[condition],
                    edmgga_loc(analyzer.rho_data)[condition],
                    'EDM', '$F_x$ (a.u)', [-3, 6])
plt.show()
