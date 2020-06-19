from mldftdat.loc_analyzers import RHFAnalyzer
import numpy as np 
from pyscf.dft.libxc import eval_xc
from mldftdat.pyscf_utils import get_single_orbital_tau, get_gradient_magnitude
from mldftdat.data import ldax, ldax_dens
from pyscf import dft, scf, gto

mol = gto.Mole(atom='H 0 0 0; H 0 0 0.74', basis = 'aug-cc-pvtz')
#mol = gto.Mole(atom='H 0 0 0; F 0 0 0.74', basis = 'aug-cc-pvtz')
mol.build()
myhf = dft.RKS(mol)
myhf.xc = 'PBE'
myhf.kernel()

analyzer = RHFAnalyzer(myhf)
analyzer.perform_full_analysis()

rho = analyzer.rho_data[0,:]
rho_data = analyzer.rho_data

from mldftdat import data
import matplotlib.pyplot as plt

#lams = [0.5, 0.62, 0.68, 0.74, 0.80, 0.86, 0.92, 0.96]
lams = [0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98]

condition = rho > 3e-3
fig, axs = plt.subplots(3,3, figsize=(12,11))

print(type(axs))
for i in range(0):
    print(i)
    lam = lams[i]
    ax = axs[i//3,i%3]
    loc_dens = analyzer.get_loc_fx_energy_density(lam = lam, overwrite = True)
    data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords[condition],
                        loc_dens[condition] / (data.ldax(rho[condition]) - 1e-7),
                        '$\\lambda$=%.2f'%lam, '$F_x$ (a.u)', [-3, 6], ax=ax)
    ax.set_ylim(0.0, 2.5)
    ax.scatter([0],[0], s=10, color='black')
    ax.scatter([1.1/0.5291],[0], s=10, color='black')
    ax.annotate('H', xy=(0, 0), xytext=(-0.1, 0.05), fontsize=15)
    ax.annotate('F', xy=(0.74/0.5291, 0), xytext=(0.74/0.5291-0.1, 0.05), fontsize=15)

#https://stackoverflow.com/questions/55767312/how-to-position-suptitle
def make_space_above(axes, topmargin=1):
    """ increase figure size to make topmargin (in inches) space for 
        titles, without changing the axes sizes"""
    fig = axes.flatten()[0].figure
    s = fig.subplotpars
    w, h = fig.get_size_inches()

    figh = h - (1-s.top)*h  + topmargin
    fig.subplots_adjust(bottom=s.bottom*h/figh, top=1-topmargin/figh)
    fig.set_figheight(figh)

ax = axs[-1,-1]
data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords[condition],
                    analyzer.fx_energy_density[condition] / (data.ldax(rho[condition]) - 1e-7),
                    '$\\lambda$=1.00', '$F_x$ (a.u)', [-3, 6], ax=ax)
ax.set_ylim(0.0, 2.5)
ax.scatter([0],[0], s=10, color='black')
ax.scatter([0.74/0.5291],[0], s=10, color='black')
ax.annotate('H', xy=(0, 0), xytext=(-0.1, 0.05), fontsize=15)
ax.annotate('F', xy=(0.74/0.5291, 0), xytext=(0.74/0.5291-0.1, 0.05), fontsize=15)
fig.tight_layout()
fig.suptitle('XEF of Transformed Exchange Holes for Varying $\\lambda$')
make_space_above(axs)
#fig.tight_layout()
plt.show()

data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords[condition],
                    analyzer.fx_energy_density[condition] / (data.ldax(rho[condition]) - 1e-7),
                    '$\\lambda$=1.00', '$F_x$ (a.u)', [-3, 6])
data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords[condition], hex(s)[condition],
                    '$\\lambda$=1.00', '$F_x$ (a.u)', [-3, 6])
plt.show()
