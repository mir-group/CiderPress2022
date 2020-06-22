from mldftdat.loc_analyzers import RHFAnalyzer
import numpy as np 
from pyscf.dft.libxc import eval_xc
from mldftdat.pyscf_utils import get_single_orbital_tau, get_gradient_magnitude
import numpy as np 
from pyscf.dft.libxc import eval_xc
from mldftdat.pyscf_utils import get_single_orbital_tau, get_gradient_magnitude
from mldftdat.data import ldax, ldax_dens
from pyscf import dft, scf, gto
from mldftdat.density import tail_fx

mol = gto.Mole(atom='He', basis = 'aug-cc-pvtz')
#mol = gto.Mole(atom='H 0 0 0; F 0 0 0.74', basis = 'aug-cc-pvtz')
mol.build()
myhf = dft.RKS(mol)
myhf.xc = 'PBE'
myhf.kernel()

analyzer = RHFAnalyzer(myhf)
analyzer.perform_full_analysis()

rho = analyzer.rho_data[0,:]
rho_data = analyzer.rho_data
tfx = tail_fx(rho_data)

from mldftdat import data
import matplotlib.pyplot as plt

#lams = [0.5, 0.62, 0.68, 0.74, 0.80, 0.86, 0.92, 0.96]
lams = [0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89]

condition = rho > 3e-5
fig, axs = plt.subplots(3,3, figsize=(12,11))

print(type(axs))
for i in range(8):
    print(i)
    lam = lams[i]
    ax = axs[i//3,i%3]
    loc_dens = analyzer.get_loc_fx_energy_density(lam = lam, overwrite = True)
    data.plot_data_atom(analyzer.mol, analyzer.grid.coords[condition],
                        loc_dens[condition] / (data.ldax(rho[condition]) - 1e-7),
                        '$\\lambda$=%.2f'%lam, 5, '$F_x$ (a.u)', ax=ax)
    ax.set_ylim(0.0, 2.5)

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

lam = 1.00
ax = axs[-1,-1]
data.plot_data_atom(analyzer.mol, analyzer.grid.coords[condition],
                    analyzer.fx_energy_density[condition] / (data.ldax(rho[condition]) - 1e-7),
                    '$\\lambda$=%.2f'%lam, 5, '$F_x$ (a.u)', ax=ax)
data.plot_data_atom(analyzer.mol, analyzer.grid.coords[condition],
                    tfx[condition],
                    '$\\lambda$=%.2f'%lam, 5, 'tail $F_x$ (a.u)', ax=ax)
tauw = np.linalg.norm(rho_data[1:4])**2 / (8 * rho)
frac = tauw / rho_data[5]
#data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords[condition],
#                    eval_xc('MGGA_X_TM,', analyzer.rho_data)[0][condition]\
#                    / (data.ldax_dens(rho[condition]) - 1e-7),
#                    'TM16', '$F_x$ (a.u)', [-3, 6])
#data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords[condition],
#                    eval_xc('MGGA_X_TM,', analyzer.rho_data)[0][condition]\
#                    / (data.ldax_dens(rho[condition]) - 1e-7),
#                    'TM16', '$F_x$ (a.u)', [-3, 6])
ax.set_ylim(0.0, 2.5)
fig.tight_layout()
fig.suptitle('XEF of Transformed Exchange Holes for Varying $\\lambda$')
make_space_above(axs)
#fig.tight_layout()
plt.show()

data.plot_data_diatomic(analyzer.mol, analyzer.grid.coords[condition],
                    frac[condition],
                    'tauw/tau', '$F_x$ (a.u)', [-3, 6])
plt.show()
