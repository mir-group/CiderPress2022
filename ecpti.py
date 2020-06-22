from make_ecp import make_ecp_restricted, make_ecp_unrestricted, gaussian
from pyscf import gto
import numpy as np
import matplotlib.pyplot as plt

fit_xlim = np.array([0.5, 2.2]) * 1.88973

with open('basis.nw', 'r') as f:
    defbasis = f.read()

print("Fit to a*exp(-b*r**2)")

for functional in 'B3LYP', 'PBE0', 'HYB_GGA_XC_HSE06':
    for charge, spin in [(4,0), (2,2)]:
        mol = gto.Mole(atom='Ti')
        mol.spin = spin
        mol.charge = charge
        mol.basis = defbasis
        mol.build()
        if spin == 0:
            res, rs, xcvals, vh, rho, vxc, chg = make_ecp_restricted(mol, fit_xlim,
                                                xc=functional)
        else:
            res, rs, xcvals, vh, rho, vxc, chg = make_ecp_unrestricted(mol, fit_xlim,
                                                xc=functional)

        condition = rs > 0.8
        plt.scatter(rs[condition], vh[condition] + vxc[condition] - chg/rs[condition],
                                    label='actual')
        plt.scatter(rs[condition], gaussian(rs[condition],
                                            res.x[0], res.x[1]), label='my fit')
        if charge == 4:
            plt.scatter(rs[condition], gaussian(rs[condition], -0.935, 0.356), label='paper')
        plt.ylim(-1,0)
        plt.xlim(0.2,5)
        plt.xlabel('r (Bohr)')
        plt.ylabel('V (Ha/e, elec pot)')
        plt.legend()
        plt.savefig('Ti{}+spin{}_{}'.format(charge, spin, functional))
        plt.cla()
        print("charge {}, spin {}, functional {}:\n   a={}, b={}".format(
                        charge, spin, functional, res.x[0], res.x[1]))
