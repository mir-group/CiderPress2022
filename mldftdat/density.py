import numpy as np
from pyscf import gto, df
import scipy.linalg
from scipy.linalg.lapack import dgetrf, dgetri
from scipy.linalg.blas import dgemm, dgemv
from mldftdat.pyscf_utils import *

LDA_FACTOR = - 3.0 / 4.0 * (3.0 / np.pi)**(1.0/3)

def ldax(n):
    return LDA_FACTOR * n**(4.0/3)

def ldaxp(n):
    return 0.5 * ldax(2 * n)

def lsda(nu, nd):
    return ldaxp(nu) + ldaxp(nd)

# check this all makes sense with scaling

def get_x_nonlocal_descriptors_nsp(rho_data, tau_data, coords, weights):
    # calc ws_radii for single spin (1/n is factor of 2 larger)
    ws_radii = get_ws_radii(rho_data[0]) * 2**(1.0/3)
    nonlocal_data = get_nonlocal_data(rho_data, tau_data, ws_radii, coords, weights)
    if np.isnan(nonlocal_data).any():
        raise ValueError('Part of nonlocal_data is nan %d' % np.count_nonzero(np.isnan(nonlocal_data)))
    # note: ws_radii calculated in the regularization call does not have the
    # factor of 2^(1/3), but it only comes in linearly so it should be fine
    res = get_regularized_nonlocal_data(nonlocal_data, rho_data)
    if np.isnan(res).any():
        raise ValueError('Part of regularized result is nan %d' % np.count_nonzero(np.isnan(res)))
    return res

def get_exchange_descriptors(rho_data, tau_data, coords,
                             weights, restricted = True):
    if restricted:
        lc = get_dft_input(rho_data)[:3]
        nlc = get_x_nonlocal_descriptors_nsp(rho_data, tau_data,
                                                coords, weights)
        return np.append(lc, nlc, axis=0)
    else:
        lcu = get_dft_input(rho_data[0] * 2)[:3]
        nlcu = get_x_nonlocal_descriptors_nsp(rho_data[0] * 2,
                                                tau_data[0] * 2,
                                                coords, weights)
        lcd = get_dft_input(rho_data[1] * 2)[:3]
        nlcd = get_x_nonlocal_descriptors_nsp(rho_data[1] * 2,
                                                tau_data[1] * 2,
                                                coords, weights)
        return np.append(lcu, nlcu, axis=0),\
               np.append(lcd, nlcd, axis=0)


def _get_x_helper(auxmol, rho_data, ddrho, grid, rdm1, ao_to_aux):
    # desc[0:6]   = rho_data
    # desc[6:12]  = ddrho
    # desc[12:13] = g0
    # desc[13:16] = g1
    # desc[16:21] = g2
    # g1 order: x, y, z
    # g2 order: xy, yz, z^2, xz, x^2-y^2
    lc = get_dft_input2(rho_data)[:3]
    # size naux
    density = np.einsum('npq,pq->n', ao_to_aux, rdm1)
    desc = np.append(rho_data, ddrho, axis=0)
    N = grid.weights.shape[0]
    for l in range(3):
        atm, bas, env = get_gaussian_grid(grid.coords, rho_data[0],
                                          l = l, s = lc[1], alpha=lc[2])
        gridmol = gto.Mole(_atm = atm, _bas = bas, _env = env)
        # (ngrid * (2l+1), naux)
        ovlp = gto.mole.intor_cross('int1e_ovlp', gridmol, auxmol)
        proj = np.dot(ovlp, density).reshape(N, 2*l+1).transpose()
        desc = np.append(desc, proj, axis=0)
    return contract_exchange_descriptors(desc)

def get_exchange_descriptors2(analyzer, restricted = True):
    """
    A length-21 descriptor containing semi-local information
    and a few Gaussian integrals. The descriptors are not
    normalized to be scale-invariant or rotation-invariant.
    
    Args:
        analyzer (RHFAnalyzer)

    Returns 2D numpy array desc:
        desc[0:6]   = rho_data
        desc[6:12]  = ddrho
        desc[12:13] = g0
        desc[13:16] = g1
        desc[16:21] = g2
        g1 order: x, y, z
        g2 order: xy, yz, z^2, xz, x^2-y^2
    """
    auxbasis = df.aug_etb(analyzer.mol, beta=1.6)
    nao = analyzer.mol.nao_nr()
    auxmol = df.make_auxmol(analyzer.mol, auxbasis)
    naux = auxmol.nao_nr()
    # shape (naux, naux), symmetric
    aug_J = auxmol.intor('int2c2e')
    # shape (nao, nao, naux)
    aux_e2 = df.incore.aux_e2(analyzer.mol, auxmol)
    #print(aux_e2.shape)
    # shape (naux, nao * nao)
    aux_e2 = aux_e2.reshape((-1, aux_e2.shape[-1])).transpose()
    aux_e2 = np.ascontiguousarray(aux_e2)
    lu, piv, info = dgetrf(aug_J, overwrite_a = True)
    inv_aug_J, info = dgetri(lu, piv, overwrite_lu = True)
    ao_to_aux = dgemm(1, inv_aug_J, aux_e2)
    ao_to_aux = ao_to_aux.reshape(naux, nao, nao)

    # rho_dat aand rrdho are polarized if calc is unrestricted
    ao_data, rho_data = get_mgga_data(analyzer.mol,
                                      analyzer.grid,
                                      analyzer.rdm1)
    ddrho = get_rho_second_deriv(analyzer.mol,
                                analyzer.grid,
                                analyzer.rdm1,
                                ao_data)

    if restricted:
        return _get_x_helper(auxmol, rho_data, ddrho, analyzer.grid,
                             analyzer.rdm1, ao_to_aux)
    else:
        desc0 = _get_x_helper(auxmol, 2*rho_data[0], 2*ddrho[0], analyzer.grid,
                              2*analyzer.rdm1[0], ao_to_aux)
        desc1 = _get_x_helper(auxmol, 2*rho_data[1], 2*ddrho[1], analyzer.grid,
                              2*analyzer.rdm1[1], ao_to_aux)
        return desc0, desc1

def contract_exchange_descriptors(desc):
    # desc[0:6]   = rho_data
    # desc[6:12]  = ddrho
    # desc[12:13] = g0
    # desc[13:16] = g1
    # desc[16:21] = g2
    # g1 order: x, y, z
    # g2 order: xy, yz, z^2, xz, x^2-y^2

    N = desc.shape[1]
    res = np.zeros((15,N))
    rho_data = desc[:6]

    # rho, g0, s, alpha, nabla
    rho, s, alpha, tau_w, tau_unif = get_dft_input2(desc[:6])
    sprefac = 2 * (3 * np.pi * np.pi)**(1.0/3)
    n43 = rho**(4.0/3)
    svec = desc[1:4] / (sprefac * n43 + 1e-7)
    nabla = rho_data[4] / (tau_unif + 1e-7)

    res[0] = rho
    res[1] = s
    res[2] = alpha
    res[3] = nabla

    # other setup
    g0 = desc[12]
    g1 = desc[13:16]
    g2 = desc[16:21]
    ddrho = desc[6:12]
    ddrho_mat = np.zeros((3, 3, N))
    inds = [[0, 1, 2], [1, 3, 4], [2, 4, 5]]
    for i in range(3):
        ddrho_mat[:,i,:] = ddrho[inds[i],:]
        ddrho_mat[i,i,:] -= rho_data[4] / 3
    ddrho_mat /= tau_unif + 1e-7
    g2_mat = np.zeros((3, 3, N))
    # y^2 = -(1/2) (z^2 + (x^2-y^2))
    g2_mat[1,1,:] = -0.5 * (g2[2] + g2[4])
    g2_mat[2,2,:] = g2[4]
    g2_mat[0,0,:] = - (g2_mat[1,1,:] + g2_mat[2,2,:])
    g2_mat[1,0,:] = g2[0]
    g2_mat[0,1,:] = g2[0]
    g2_mat[2,0,:] = g2[3]
    g2_mat[0,2,:] = g2[3]
    g2_mat[1,2,:] = g2[1]
    g2_mat[2,1,:] = g2[1]

    # g1_norm and 1d dot product
    g1_norm = np.linalg.norm(g1, axis=0)
    dot1 = np.einsum('an,an->n', svec, g1)

    # nabla and g2 norms
    g2_norm = np.sqrt(np.einsum('pqn,pqn->n', g2_mat, g2_mat))
    d2_norm = np.sqrt(np.einsum('pqn,pqn->n', ddrho_mat, ddrho_mat))

    res[4] = g0
    res[5] = g1_norm
    res[6] = dot1
    res[7] = d2_norm
    res[8] = g2_norm

    res[9] = np.einsum('pn,pqn,qn->n', svec, ddrho_mat, svec)
    res[10] = np.einsum('pn,pqn,qn->n', g1, ddrho_mat, svec)
    res[11] = np.einsum('pn,pqn,qn->n', g1, ddrho_mat, g1)

    res[12] = np.einsum('pn,pqn,qn->n', svec, g2_mat, svec)
    res[13] = np.einsum('pn,pqn,qn->n', g1, g2_mat, svec)
    res[14] = np.einsum('pn,pqn,qn->n', g1, g2_mat, g1)

    # res
    # 0:  rho
    # 1:  s
    # 2:  alpha
    # 3:  nabla
    # 4:  g0
    # 5:  norm(g1)
    # 6:  g1 dot svec
    # 7:  norm(ddrho_{l=2})
    # 8:  norm(g2)
    # 9:  svec dot ddrho_{l=2} dot svec
    # 10: g1 dot ddrho_{l=2} dot svec
    # 11: g1 dot ddrho_{l=2} dot g1
    # 12: svec dot g2 dot svec
    # 13: g1 dot g2 dot svec
    # 14: g1 dot g2 dot g1
    return res





    # rho, g0

"""
The following two routines are from
J. Tao, J. Chem. Phys. 115, 3519 (2001) (doi: 10.1063/1.1388047)
"""

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
    gradn = np.linalg.norm(rho_data[1:4], axis=0)
    tau0 = get_uniform_tau(rho_data[0]) + 1e-6
    tauw = get_single_orbital_tau(rho_data[0], gradn)
    QB = tau0 - rho_data[5] + tauw + 0.25 * rho_data[4]
    QB /= tau0
    x = A * QB + np.sqrt(1 + (A*QB)**2)
    FX = C1 + (C2 * x) / (1 + C3 * np.sqrt(x) * np.arcsinh(C4 * (x-1)))
    return FX

def edmgga_loc(rho_data):
    gradn = np.linalg.norm(rho_data[1:4], axis=0)
    tau0 = get_uniform_tau(rho_data[0]) + 1e-6
    tauw = get_single_orbital_tau(rho_data[0], gradn)
    QB = tau0 - rho_data[5] + 0.125 * rho_data[4]
    QB /= tau0
    x = A * QB + np.sqrt(1 + (A*QB)**2)
    FX = C1 + (C2 * x) / (1 + C3 * np.sqrt(x) * np.arcsinh(C4 * (x-1)))
    return FX
