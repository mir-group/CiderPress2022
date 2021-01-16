import numpy as np
from pyscf import gto, df
import scipy.linalg
from scipy.linalg import cho_factor, cho_solve
from mldftdat.pyscf_utils import *

LDA_FACTOR = - 3.0 / 4.0 * (3.0 / np.pi)**(1.0/3)

GG_SMUL = 1.0
GG_AMUL = 1.0
GG_AMIN = 1.0 / 18

def ldax(n):
    return LDA_FACTOR * n**(4.0/3)

def ldaxp(n):
    return 0.5 * ldax(2 * n)

def lsda(nu, nd):
    return ldaxp(nu) + ldaxp(nd)

def get_ldax_dens(n):
    return LDA_FACTOR * n**(4.0/3)

def get_ldax(n):
    return LDA_FACTOR * n**(1.0/3)

def get_xed_from_y(y, rho):
    """
    Get the exchange energy density (n * epsilon_x)
    from the exchange enhancement factor y
    and density rho.
    """
    return rho * get_x(y, rho)

def get_x(y, rho):
    return (y + 1) * get_ldax(rho)

def get_y_from_xed(xed, rho):
    """
    Get the exchange enhancement factor minus one.
    """
    return xed / (get_ldax_dens(rho) - 1e-12) - 1


def get_gaussian_grid(coords, rho, l = 0, s = None, alpha = None):
    N = coords.shape[0]
    auxmol = gto.fakemol_for_charges(coords)
    atm = auxmol._atm.copy()
    bas = auxmol._bas.copy()
    start = auxmol._env.shape[0] - 2
    env = np.zeros(start + 2 * N)
    env[:start] = auxmol._env[:-2]
    bas[:,5] = start + np.arange(N)
    bas[:,6] = start + N + np.arange(N)

    a = np.pi * (rho / 2 + 1e-16)**(2.0 / 3)
    scale = 1
    fac = (6 * np.pi**2)**(2.0/3) / (16 * np.pi)
    if s is not None:
        scale += GG_SMUL * fac * s**2
    if alpha is not None:
        scale += GG_AMUL * 0.6 * fac * (alpha - 1)
    bas[:,1] = l
    ascale = a * scale
    cond = ascale < GG_AMIN
    ascale[cond] = GG_AMIN * np.exp(ascale[cond] / GG_AMIN - 1)
    env[bas[:,5]] = ascale
    logging.debug('GAUSS GRID MIN EXPONENT {}'.format(np.sqrt(np.min(env[bas[:,5]]))))
    #env[bas[:,6]] = np.sqrt(4 * np.pi) * (4 * np.pi * rho / 3)**(l / 3.0) * np.sqrt(scale)**l
    env[bas[:,6]] = np.sqrt(4 * np.pi**(1-l)) * (8 * np.pi / 3)**(l/3.0) * ascale**(l/2.0)

    return atm, bas, env

def get_gaussian_grid_c(coords, rho, l=0, s=None, alpha=None,
                        a0=8.0, fac_mul=0.25, amin=GG_AMIN):
    N = coords.shape[0]
    auxmol = gto.fakemol_for_charges(coords)
    atm = auxmol._atm.copy()
    bas = auxmol._bas.copy()
    start = auxmol._env.shape[0] - 2
    env = np.zeros(start + 2 * N)
    env[:start] = auxmol._env[:-2]
    bas[:,5] = start + np.arange(N)
    bas[:,6] = start + N + np.arange(N)
    ratio = alpha + 5./3 * s**2

    fac = fac_mul * 1.2 * (6 * np.pi**2)**(2.0/3) / np.pi
    a = np.pi * (rho / 2 + 1e-16)**(2.0 / 3)
    scale = a0 + (ratio-1) * fac
    bas[:,1] = l
    ascale = a * scale
    cond = ascale < amin
    ascale[cond] = amin * np.exp(ascale[cond] / amin - 1)
    env[bas[:,5]] = ascale
    logging.debug('GAUSS GRID MIN EXPONENT {}'.format(np.sqrt(np.min(env[bas[:,5]]))))
    env[bas[:,6]] = a0**1.5 * np.sqrt(4 * np.pi**(1-l)) \
                    * (8 * np.pi / 3)**(l/3.0) * ascale**(l/2.0)

    return atm, bas, env

def get_gaussian_grid_b(coords, rho, l=0, s=None, alpha=None):
    N = coords.shape[0]
    auxmol = gto.fakemol_for_charges(coords)
    atm = auxmol._atm.copy()
    bas = auxmol._bas.copy()
    start = auxmol._env.shape[0] - 2
    env = np.zeros(start + 2 * N)
    env[:start] = auxmol._env[:-2]
    bas[:,5] = start + np.arange(N)
    bas[:,6] = start + N + np.arange(N)

    a = np.pi * (rho / 2 + 1e-6)**(2.0 / 3)
    scale = 1
    #fac = (6 * np.pi**2)**(2.0/3) / (16 * np.pi)
    fac = (6 * np.pi**2)**(2.0/3) / (16 * np.pi)
    if s is not None:
        scale += fac * s**2
    if alpha is not None:
        scale += 3.0 / 5 * fac * (alpha - 1)
    bas[:,1] = l
    env[bas[:,5]] = a * scale
    env[bas[rho<1e-8,5]] = 1e16
    logging.debug('GAUSS GRID MIN EXPONENT {}'.format(np.sqrt(np.min(env[bas[:,5]]))))
    env[bas[:,6]] = np.sqrt(4 * np.pi) * (4 * np.pi * rho / 3)**(l / 3.0) * np.sqrt(scale)**l

    return atm, bas, env, (4 * np.pi * rho / 3)**(1.0 / 3), scale


def get_x_helper_full_a(auxmol, rho_data, grid, density,
                        ao_to_aux, deriv=False,
                        return_ovlp=False,
                        a0=8.0, fac_mul=0.25, amin=GG_AMIN):
    # desc[0:6]   = rho_data
    # desc[6:12]  = 0
    # desc[12:13] = g0
    # desc[13:16] = g1
    # desc[16:21] = g2
    # desc[21] = g0-0.5
    # desc[22] = g0-2
    # g1 order: x, y, z
    # g2 order: xy, yz, z^2, xz, x^2-y^2
    lc = get_dft_input2(rho_data)[:3]
    # size naux
    integral_name = 'int1e_r2_origj' if deriv else 'int1e_ovlp'
    desc = np.append(rho_data, 0 * rho_data, axis=0)
    N = grid.weights.shape[0]
    if return_ovlp:
        ovlps = []
    for l in range(3):
        atm, bas, env = get_gaussian_grid_c(grid.coords, rho_data[0],
                                            l=l, s=lc[1], alpha=lc[2],
                                            a0=a0, fac_mul=fac_mul,
                                            amin=amin)
        gridmol = gto.Mole(_atm=atm, _bas=bas, _env=env)
        # (ngrid * (2l+1), naux)
        ovlp = gto.mole.intor_cross(integral_name, auxmol, gridmol).T
        proj = np.dot(ovlp, density).reshape(N, 2*l+1).transpose()
        desc = np.append(desc, proj, axis=0)
        if return_ovlp:
            ovlps.append(ovlp)
    l = 0
    for mul in [0.25, 4.00]:
        atm, bas, env = get_gaussian_grid_c(grid.coords, rho_data[0],
                                           l=0, s=lc[1], alpha=lc[2],
                                           a0=a0*mul**(2./3),
                                           fac_mul=fac_mul*mul**(2./3),
                                           amin=amin*mul**(2./3))
        gridmol = gto.Mole(_atm=atm, _bas=bas, _env=env)
        # (ngrid * (2l+1), naux)
        ovlp = gto.mole.intor_cross(integral_name, auxmol, gridmol).T
        proj = np.dot(ovlp, density).reshape(N, 2*l+1).transpose()
        desc = np.append(desc, proj, axis=0)
        if return_ovlp:
            ovlps.append(ovlp)
    if return_ovlp:
        return desc, ovlps
    else:
        return desc


def get_x_helper_full_c(auxmol, rho_data, grid, density,
                        ao_to_aux, deriv=False,
                        return_ovlp=False,
                        a0=8.0, fac_mul=0.25, amin=GG_AMIN):
    # Contains convolutions up to second-order
    # desc[0:6]   = rho_data
    # desc[6:7] = g0
    # desc[7:10] = g1
    # desc[10:15] = g2
    # desc[16] = g0-r^2
    # g1 order: x, y, z
    # g2 order: xy, yz, z^2, xz, x^2-y^2
    lc = get_dft_input2(rho_data)[:3]
    # size naux
    desc = rho_data.copy()
    N = grid.weights.shape[0]
    integral_name = 'int1e_r2_origj' if deriv else 'int1e_ovlp'
    if return_ovlp:
        ovlps = []
    for l in range(3):
        atm, bas, env = get_gaussian_grid_c(grid.coords, rho_data[0],
                                            l=l, s=lc[1], alpha=lc[2],
                                            a0=a0, fac_mul=fac_mul,
                                            amin=amin)
        gridmol = gto.Mole(_atm=atm, _bas=bas, _env=env)
        # (ngrid * (2l+1), naux)
        ovlp = gto.mole.intor_cross(integral_name, auxmol, gridmol).T
        proj = np.dot(ovlp, density).reshape(N, 2*l+1).transpose()
        desc = np.append(desc, proj, axis=0)
        if return_ovlp:
            ovlps.append(ovlp)

    l = 0
    integral_name = 'int1e_r4_origj' if deriv else 'int1e_r2_origj'
    atm, bas, env = get_gaussian_grid_c(grid.coords, rho_data[0],
                                        l=0, s=lc[1], alpha=lc[2],
                                        a0=a0, fac_mul=fac_mul,
                                        amin=amin)
    env[bas[:,6]] *= env[bas[:,5]]
    gridmol = gto.Mole(_atm=atm, _bas=bas, _env=env)
    # (ngrid * (2l+1), naux)
    ovlp = gto.mole.intor_cross(integral_name, auxmol, gridmol).T
    proj = np.dot(ovlp, density).reshape(N, 2*l+1).transpose()
    desc = np.append(desc, proj, axis=0)
    if return_ovlp:
        ovlps.append(ovlp)
    
    integral_name = 'int1e_r6_origj' if deriv else 'int1e_r4_origj'
    atm, bas, env = get_gaussian_grid_c(grid.coords, rho_data[0],
                                        l=0, s=lc[1], alpha=lc[2],
                                        a0=a0, fac_mul=fac_mul,
                                        amin=amin)
    env[bas[:,6]] *= env[bas[:,5]]**2
    gridmol = gto.Mole(_atm=atm, _bas=bas, _env=env)
    # (ngrid * (2l+1), naux)
    ovlp = gto.mole.intor_cross(integral_name, auxmol, gridmol).T
    proj = np.dot(ovlp, density).reshape(N, 2*l+1).transpose()
    desc = np.append(desc, proj, axis=0)
    if return_ovlp:
        ovlps.append(ovlp)

    if return_ovlp:
        return desc, ovlps
    else:
        return desc


def _get_x_helper_a(auxmol, rho_data, ddrho, grid, rdm1, ao_to_aux,
                    a0=8.0, fac_mul=0.25, amin=GG_AMIN, **kwargs):
    # desc[0:6]   = rho_data
    # desc[6:12]  = ddrho
    # desc[12:13] = g0
    # desc[13:16] = g1
    # desc[16:21] = g2
    # desc[21] = g0-0.5
    # desc[22] = g0-2
    # g1 order: x, y, z
    # g2 order: xy, yz, z^2, xz, x^2-y^2
    lc = get_dft_input2(rho_data)[:3]
    # size naux
    density = np.einsum('npq,pq->n', ao_to_aux, rdm1)
    desc = rho_data.copy()
    N = grid.weights.shape[0]
    for l in range(3):
        atm, bas, env = get_gaussian_grid_c(grid.coords, rho_data[0],
                                            l=l, s=lc[1], alpha=lc[2],
                                            a0=a0, fac_mul=fac_mul,
                                            amin=amin)
        gridmol = gto.Mole(_atm=atm, _bas=bas, _env=env)
        # (ngrid * (2l+1), naux)
        ovlp = gto.mole.intor_cross('int1e_ovlp', gridmol, auxmol)
        proj = np.dot(ovlp, density).reshape(N, 2*l+1).transpose()
        desc = np.append(desc, proj, axis=0)
    l = 0
    for mul in [0.25, 4.00]:
        atm, bas, env = get_gaussian_grid_c(grid.coords, rho_data[0],
                                           l=0, s=lc[1], alpha=lc[2],
                                           a0=a0*mul**(2./3),
                                           fac_mul=fac_mul*mul**(2./3),
                                           amin=amin*mul**(2./3))
        gridmol = gto.Mole(_atm=atm, _bas=bas, _env=env)
        # (ngrid * (2l+1), naux)
        ovlp = gto.mole.intor_cross('int1e_ovlp', gridmol, auxmol)
        proj = np.dot(ovlp, density).reshape(N, 2*l+1).transpose()
        desc = np.append(desc, proj, axis=0)
    return contract_exchange_descriptors_c(desc)


def _get_x_helper_c(auxmol, rho_data, ddrho, grid, rdm1, ao_to_aux,
                    a0=8.0, fac_mul=0.25, amin=GG_AMIN, **kwargs):
    print(a0, fac_mul, amin)
    # desc[0:6]   = rho_data
    # desc[6] = g0
    # desc[7:10] = g1
    # desc[10:15] = g2
    # desc[15] = g0-r^2
    # g1 order: x, y, z
    # g2 order: xy, yz, z^2, xz, x^2-y^2
    lc = get_dft_input2(rho_data)[:3]
    # size naux
    density = np.einsum('npq,pq->n', ao_to_aux, rdm1)
    desc = rho_data.copy()
    N = grid.weights.shape[0]
    for l in range(3):
        atm, bas, env = get_gaussian_grid_c(grid.coords, rho_data[0],
                                            l=l, s=lc[1], alpha=lc[2],
                                            a0=a0, fac_mul=fac_mul,
                                            amin=amin)
        gridmol = gto.Mole(_atm = atm, _bas = bas, _env = env)
        # (ngrid * (2l+1), naux)
        ovlp = gto.mole.intor_cross('int1e_ovlp', gridmol, auxmol)
        proj = np.dot(ovlp, density).reshape(N, 2*l+1).transpose()
        desc = np.append(desc, proj, axis=0)
    l = 0
    atm, bas, env = get_gaussian_grid_c(grid.coords, rho_data[0],
                                        l=0, s=lc[1], alpha=lc[2],
                                        a0=a0, fac_mul=fac_mul,
                                        amin=amin)
    env[bas[:,6]] *= env[bas[:,5]]
    gridmol = gto.Mole(_atm=atm, _bas=bas, _env=env)
    ovlp = gto.mole.intor_cross('int1e_r2_origj', auxmol, gridmol).T
    proj = np.dot(ovlp, density).reshape(N, 2*l+1).transpose()
    desc = np.append(desc, proj, axis=0)
    atm, bas, env = get_gaussian_grid_c(grid.coords, rho_data[0],
                                        l=0, s=lc[1], alpha=lc[2],
                                        a0=a0, fac_mul=fac_mul,
                                        amin=amin)
    env[bas[:,6]] *= env[bas[:,5]]**2
    gridmol = gto.Mole(_atm=atm, _bas=bas, _env=env)
    ovlp = gto.mole.intor_cross('int1e_r4_origj', auxmol, gridmol).T
    proj = np.dot(ovlp, density).reshape(N, 2*l+1).transpose()
    desc = np.append(desc, proj, axis=0)
    return contract_exchange_descriptors_c(desc)


def get_exchange_descriptors2(analyzer, restricted=True, version='a',
                              **kwargs):
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
    if version == 'a':
        _get_x_helper = _get_x_helper_a
    elif version == 'b':
        _get_x_helper = _get_x_helper_b
    elif version == 'c':
        _get_x_helper = _get_x_helper_c
    else:
        raise ValueError('unknown descriptor version')
    #auxbasis = df.aug_etb(analyzer.mol, beta=1.6)
    nao = analyzer.mol.nao_nr()
    auxmol = df.make_auxmol(analyzer.mol, auxbasis='weigend+etb')
    naux = auxmol.nao_nr()
    # shape (naux, naux), symmetric
    aug_J = auxmol.intor('int2c2e')
    # shape (nao, nao, naux)
    aux_e2 = df.incore.aux_e2(analyzer.mol, auxmol)
    #print(aux_e2.shape)
    # shape (naux, nao * nao)
    aux_e2 = aux_e2.reshape((-1, aux_e2.shape[-1])).T
    c_and_lower = cho_factor(aug_J)
    ao_to_aux = cho_solve(c_and_lower, aux_e2)
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
                             analyzer.rdm1, ao_to_aux, **kwargs)
    else:
        desc0 = _get_x_helper(auxmol, 2*rho_data[0], 2*ddrho[0], analyzer.grid,
                              2*analyzer.rdm1[0], ao_to_aux, **kwargs)
        desc1 = _get_x_helper(auxmol, 2*rho_data[1], 2*ddrho[1], analyzer.grid,
                              2*analyzer.rdm1[1], ao_to_aux, **kwargs)
        return desc0, desc1

# TODO: Check the math
def contract21(t2, t1):
    # xy, yz, z2, xz, x2-y2
    # x, y, z
    t2c = np.zeros(t2.shape, dtype=np.complex128)
    t2c[4] = (t2[4] + 1j * t2[0]) / np.sqrt(2) # +2
    t2c[3] = (-t2[3] - 1j * t2[1]) / np.sqrt(2) # +1
    t2c[2] = t2[2] # 0
    t2c[1] = (t2[3] - 1j * t2[1]) / np.sqrt(2) # -1
    t2c[0] = (t2[4] - 1j * t2[0]) / np.sqrt(2) # -2

    t1c = np.zeros(t1.shape, dtype=np.complex128)
    t1c[2] = -(t1[0] + 1j * t1[1]) / np.sqrt(2) # +1
    t1c[1] = t1[2] # 0
    t1c[0] = (t1[0] - 1j * t1[1]) / np.sqrt(2) # -1

    res = np.zeros(t1.shape, dtype=np.complex128)
    res[0] = np.sqrt(0.6) * t2c[0] * t1c[2]\
             - np.sqrt(0.3) * t2c[1] * t1c[1]\
             + np.sqrt(0.1) * t2c[2] * t1c[0] # -1
    res[1] = np.sqrt(0.3) * t2c[1] * t1c[2]\
             - np.sqrt(0.4) * t2c[2] * t1c[1]\
             + np.sqrt(0.3) * t2c[3] * t1c[0]
    res[2] = np.sqrt(0.6) * t2c[4] * t1c[0]\
             - np.sqrt(0.3) * t2c[3] * t1c[1]\
             + np.sqrt(0.1) * t2c[2] * t1c[2]

    xterm = (res[0] - res[2]) / np.sqrt(2)
    yterm = 1j * (res[0] + res[2]) / np.sqrt(2)
    zterm = res[1]

    return np.real(np.array([xterm, yterm, zterm]))

def contract21_deriv(t1, t1b = None):
    if t1b is None:
        t1b = t1
    tmp = np.identity(5)
    derivs = np.zeros((5, 3, t1.shape[1]))
    for i in range(5):
        derivs[i] = contract21(tmp[i], t1)
    return np.einsum('map,ap->mp', derivs, t1b)

def contract_exchange_descriptors_a(desc):
    # desc[0:6]   = rho_data
    # desc[6:12]  = ddrho
    # desc[12:13] = g0
    # desc[13:16] = g1
    # desc[16:21] = g2
    # desc[21] = g0-0.5
    # desc[22] = g0-2
    # g1 order: x, y, z
    # g2 order: xy, yz, z^2, xz, x^2-y^2

    N = desc.shape[1]
    res = np.zeros((17,N))
    rho_data = desc[:6]

    # rho, g0, s, alpha, nabla
    rho, s, alpha, tau_w, tau_unif = get_dft_input2(desc[:6])
    sprefac = 2 * (3 * np.pi * np.pi)**(1.0/3)
    n43 = rho**(4.0/3)
    svec = desc[1:4] / (sprefac * n43 + 1e-7)
    nabla = rho_data[4] / (tau_unif + 1e-7)

    res[0] = rho
    res[1] = s**2
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
    g1_norm = np.linalg.norm(g1, axis=0)**2
    dot1 = np.einsum('an,an->n', svec, g1)

    # nabla and g2 norms
    #g2_norm = np.sqrt(np.einsum('pqn,pqn->n', g2_mat, g2_mat))

    # Clebsch Gordan https://en.wikipedia.org/wiki/Table_of_Clebsch%E2%80%93Gordan_coefficients
    # TODO need to adjust for the fact that these are real sph_harm?
    g2_norm = 0
    """
    g2 = np.array([(g2[-1] + 1j * g2[0]) / np.sqrt(2),
                   (-g2[-2] - 1j * g2[1]) / np.sqrt(2),
                   g2[2],
                   (g2[-2] - 1j * g2[1]) / np.sqrt(2),
                   (g2[-1] - 1j * g2[0]) / np.sqrt(2)])

    for i in range(5):
        g2_norm += g2[i] * g2[-1-i] * (-1)**i
    """
    for i in range(5):
        g2_norm += g2[i] * g2[i]
    g2_norm /= np.sqrt(5)

    d2_norm = np.sqrt(np.einsum('pqn,pqn->n', ddrho_mat, ddrho_mat))

    res[4] = g0
    res[5] = g1_norm
    res[6] = dot1
    res[7] = d2_norm
    res[8] = g2_norm

    res[9] = np.einsum('pn,pqn,qn->n', svec, ddrho_mat, svec)
    res[10] = np.einsum('pn,pqn,qn->n', g1, ddrho_mat, svec)
    res[11] = np.einsum('pn,pqn,qn->n', g1, ddrho_mat, g1)

    sgc = contract21(g2, svec)
    sgg = contract21(g2, g1)

    res[12] = np.einsum('pn,pn->n', sgc, svec)
    res[13] = np.einsum('pn,pn->n', sgc, g1)
    res[14] = np.einsum('pn,pn->n', sgg, g1)

    res[15] = desc[21]
    res[16] = desc[22]

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
    # 15: g0-0.5
    # 16: g0-2
    return res


def contract_exchange_descriptors_c(desc):
    # desc[0:6] = rho_data
    # desc[6:7] = g0
    # desc[7:10] = g1
    # desc[10:15] = g2
    # desc[15] = g0-r^2
    # g1 order: x, y, z
    # g2 order: xy, yz, z^2, xz, x^2-y^2

    N = desc.shape[1]
    res = np.zeros((12,N))
    rho_data = desc[:6]

    # rho, g0, s, alpha, nabla
    rho, s, alpha, tau_w, tau_unif = get_dft_input2(desc[:6])
    sprefac = 2 * (3 * np.pi * np.pi)**(1.0/3)
    n43 = rho**(4.0/3)
    svec = desc[1:4] / (sprefac * n43 + 1e-7)
    nabla = rho_data[4] / (tau_unif + 1e-7)

    res[0] = rho
    res[1] = s**2
    res[2] = alpha

    # other setup
    g0 = desc[6]
    g1 = desc[7:10]
    g2 = desc[10:15]

    # g1_norm and 1d dot product
    g1_norm = np.linalg.norm(g1, axis=0)**2
    dot1 = np.einsum('an,an->n', svec, g1)

    # Clebsch Gordan https://en.wikipedia.org/wiki/Table_of_Clebsch%E2%80%93Gordan_coefficients
    # TODO need to adjust for the fact that these are real sph_harm?
    g2_norm = 0
    for i in range(5):
        g2_norm += g2[i] * g2[i]
    g2_norm /= np.sqrt(5)

    res[3] = g0
    res[4] = g1_norm
    res[5] = dot1
    res[6] = g2_norm

    sgc = contract21(g2, svec)
    sgg = contract21(g2, g1)

    res[7] = np.einsum('pn,pn->n', sgc, svec)
    res[8] = np.einsum('pn,pn->n', sgc, g1)
    res[9] = np.einsum('pn,pn->n', sgg, g1)

    res[10] = desc[15]
    res[11] = desc[16]

    # res
    # 0:  rho
    # 1:  s
    # 2:  alpha
    # 3:  g0
    # 4:  norm(g1)**2
    # 5:  g1 dot svec
    # 6:  norm(g2)**2
    # 7:  svec dot g2 dot svec
    # 8:  g1 dot g2 dot svec
    # 9:  g1 dot g2 dot g1
    # 10: g0-r^2
    # 11: g0-r^4
    return res


sprefac = 2 * (3 * np.pi * np.pi)**(1.0/3)
s0 = 1 / (0.5 * sprefac / np.pi**(1.0/3) * 2**(1.0/3))
hprefac = 1.0 / 3 * (4 * np.pi**2 / 3)**(1.0 / 3)
mu = 0.2195
l = 0.5 * sprefac / np.pi**(1.0/3) * 2**(1.0/3)
a = 1 - hprefac * 4 / 3
b = mu - l**2 * hprefac / 18

def tail_fx(rho_data):
    gradn = np.linalg.norm(rho_data[1:4], axis=0)
    s = get_normalized_grad(rho_data[0], gradn)
    return tail_fx_direct(s)

def tail_fx_direct(s):
    sp = 0.5 * sprefac * s / np.pi**(1.0/3) * 2**(1.0/3)
    term1 = hprefac * 2.0 / 3 * sp / np.arcsinh(0.5 * sp)
    term3 = 2 + sp**2 / 12 - 17 * sp**4 / 2880 + 367 * sp**6 / 483840\
            - 27859 * sp**8 / 232243200 + 1295803 * sp**10 / 61312204800
    term3 *= hprefac * 2.0 / 3
    term2 = (a + b * s**2) / (1 + (l*s/2)**4)
    f = term2 + term3
    f[s > 0.025] = term2[s > 0.025] + term1[s > 0.025]
    return f

def tail_fx_deriv(rho_data):
    gradn = np.linalg.norm(rho_data[1:4], axis=0)
    s = get_normalized_grad(rho_data[0], gradn)
    return tail_fx_deriv_direct(s)

def tail_fx_deriv_direct(s):
    sp = 0.5 * sprefac * s / np.pi**(1.0/3) * 2**(1.0/3) 
    sfac = 0.5 * sprefac / np.pi**(1.0/3) * 2**(1.0/3)
    term1 = hprefac * 2.0 / 3 * sfac * (1.0 / np.arcsinh(0.5 * sp)\
            - sp / (2 * np.sqrt(1+sp**2/4) * np.arcsinh(sp/2)**2))
    term3 = sp / 6 - 17 * sp**3 / 720 + 367 * sp**5 / 80640 - 27859 * sp**7 / 29030400\
            + 1295803 * sp**9 / 6131220480
    term3 *= hprefac * 2.0 / 3 * sfac
    denom = (1 + (l*s/2)**4)
    term2 = 2 * b * s / denom - l**4 * s**3 * (a + b * s**2) / (4 * denom**2)
    f = term2 + term3
    f[s > 0.025] = term2[s > 0.025] + term1[s > 0.025]
    return f

def tail_fx_deriv_p(s):
    return tail_fx_deriv_direct(s) / (2 * s + 1e-16)

