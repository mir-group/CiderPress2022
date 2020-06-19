import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import minimize
from scipy.special import erf, erfc
from pyscf import gto, scf, dft
from pyscf.dft.gen_grid import Grids
from mldftdat.pyscf_utils import *

with open('basis.nw', 'r') as f:
    defbasis = f.read()

def gaussian(r, a, b):
    return a * np.exp(-b * r**2)

def gaussian_func_getter(r, pot):
    def gaussian_error(x):
        a, b = x
        #return np.linalg.norm((gaussian(r, a, b) - pot) / 1)
        return np.linalg.norm((gaussian(r, a, b) - pot) / pot)
    return gaussian_error

def make_ecp_restricted(mol, fit_xlim, xc='HYB_GGA_XC_HSE06'):
    mol.build()
    rks = scf.RKS(mol)
    rks.init_guess = 'atom'
    rks.conv_tol = 1e-7
    rks.diis_space = 16
    rks.diis_start_cycle = 4
    rks.xc = xc
    rks.kernel()
    grid = Grids(mol)
    grid.level = 3
    grid.kernel()
    coords, weights = grid.coords, grid.weights
    ao = dft.numint.eval_ao(mol, coords, deriv=2)
    rho = dft.numint.eval_rho2(mol, ao[0], rks.mo_coeff, rks.mo_occ)
    #rho_data = dft.numint.eval_rho2(mol, ao, rks.mo_coeff, rks.mo_occ, xctype='mGGA')
    xcvals = dft.xcfun.eval_xc('LDA,VWN', rho)
    rs = np.linalg.norm(coords, axis=1)

    chg = mol.nelectron

    vh = get_hartree_potential(rho, coords, weights)

    condition = np.logical_and(rs < fit_xlim[1], rs > fit_xlim[0])
    res = minimize(gaussian_func_getter(rs[condition],
        vh[condition] + xcvals[1][0][condition] - chg/rs[condition]), [0.935, 0.356])
    #print(res.x, res.success)

    return res, rs, xcvals, vh, rho, xcvals[1][0], chg

def make_ecp_unrestricted(mol, fit_xlim, xc='HYB_GGA_XC_HSE06'):
    mol.build()
    uks = scf.UKS(mol)
    uks.xc = xc
    uks.kernel()
    grid = Grids(mol)
    grid.level = 3
    grid.kernel()
    coords, weights = grid.coords, grid.weights
    ao = dft.numint.eval_ao(mol, coords, deriv=2)
    rhou = dft.numint.eval_rho2(mol, ao[0], uks.mo_coeff[0], uks.mo_occ[0])
    rhod = dft.numint.eval_rho2(mol, ao[0], uks.mo_coeff[1], uks.mo_occ[1])
    rho = rhou + rhod

    chg = mol.nelectron

    #rho_data = dft.numint.eval_rho2(mol, ao, uks.mo_coeff, uks.mo_occ, xctype='mGGA')
    #print(rho.shape)
    xcvals = dft.xcfun.eval_xc('LDA,VWN', (rhou, rhod), spin=mol.spin)
    rs = np.linalg.norm(coords, axis=1)
    vh = get_hartree_potential(rho, coords, weights)
    #print(len(xcvals), xcvals[1][0].shape)
    vxc = xcvals[1][0].sum(axis=-1) / 2
    #xcvals = [xcval.sum(axis=-1) for xcval in xcvals]

    condition = np.logical_and(rs < fit_xlim[1], rs > fit_xlim[0])
    res = minimize(gaussian_func_getter(rs[condition],
        vh[condition] + vxc[condition] - chg/rs[condition]), [0.935, 0.356])
    #print(res.x, res.success)

    return res, rs, xcvals, vh, rho, vxc, chg

if __name__ == '__main__':

    mol = gto.Mole(atom='Ti')
    mol.spin = 2
    mol.charge = 2
    mol.basis = defbasis
    mol.build()
    fit_xlim = np.array([0.5, 2.2]) * 1.88973

    res, rs, xcvals, vh, rho, vxc = make_ecp_unrestricted(mol, fit_xlim)

    #vhs = get_hartree_potential(rho * erfc(0.2 * drs), drs, weights)

    """
    for i in range(weights.shape[0]):
        vecs = coords - coords[i]
        drs = np.linalg.norm(vecs, axis=1)
        drs[i] = (2.0/3) * (3 * weights[i] / (4 * np.pi))**(1.0 / 3)
        vh[i] = get_hartree_potential(rho, drs, weights)
        vhs[i] = get_hartree_potential(rho * erfc(0.2 * drs), drs, weights)
        vhl[i] = vh[i] - vhs[i]
    """

    #plt.scatter(rs, rho_data[0])
    #plt.show()

    #plt.scatter(rs, rho)
    plt.scatter(rs, vh)
    #plt.scatter(rs, vhs)
    #plt.scatter(rs, vhl)
    plt.scatter(rs, vxc)
    plt.scatter(rs, xcvals[0])
    #plt.scatter(rs, 18/rs)
    plt.show()

    condition = rs > 0.8
    plt.scatter(rs[condition], gaussian(rs[condition], res.x[0], res.x[1]), label='myfit')
    plt.scatter(rs[condition], gaussian(rs[condition], -0.935, 0.356), label='paper')
    plt.scatter(rs[condition], gaussian(rs[condition], -0.935, 0.356 / 1.88973**2), label='paper')
    plt.scatter(rs[condition], vh[condition] + vxc[condition] - 20/rs[condition],
                label='actual')
    #plt.scatter(rs[condition], vh[condition] + xcvals[0][condition] - 18/rs[condition],
    #            label='actual_en')
    plt.ylim(-1,0)
    plt.xlim(0.2,5)
    #plt.scatter(rs, gaussian(rs/1.88973, 0.935, 0.356))
    #plt.scatter(rs, gaussian(rs/1.88973, 0.935*13.7, 0.356))
    plt.legend()
    plt.show()

    mol = gto.Mole(atom='Ti')
    mol.spin = 0
    mol.charge = 4
    mol.basis = defbasis
    mol.build()
    fit_xlim = np.array([0.5, 2.2]) * 1.88973
    _, rsr, xcvalsr, vhr, rhor = make_ecp_restricted(mol, fit_xlim)

    plt.scatter(rs[condition], vh[condition] + vxc[condition] - 20/rs[condition],
        label='unrestricted')
    condition = rsr > 0.8
    plt.scatter(rsr[condition], vhr[condition] + xcvalsr[1][0][condition] - 18/rsr[condition],
        label='restricted')
    plt.ylim(-1,0)
    plt.xlim(0.2,5)
    plt.legend()
    plt.show()

    print(xcvals[1][1])
