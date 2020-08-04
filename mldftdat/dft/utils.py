import numpy as np 
from mldftdat.pyscf_utils import *
from mldftdat.workflow_utils import safe_mem_cap_mb
from pyscf.dft.numint import eval_ao, make_mask

def dtauw(rho_data):
    return - get_gradient_magnitude(rho_data)**2 / (8 * rho_data[0,:]**2 + 1e-16),\
           1 / (8 * rho_data[0,:] + 1e-16)

def dsdp(s):
    return 1 / (2 * s)

def dasinhsdp(s):
    return arcsinh_deriv(s) / (2 * s + 1e-10)

def ds2(rho_data):
    # s = |nabla n| / (b * n)
    rho = rho_data[0,:]
    b = 2 * (3 * np.pi * np.pi)**(1.0/3)
    s = get_gradient_magnitude(rho_data) / (b * rho**(4.0/3) + 1e-16)
    s2 = s**2
    return -8.0 * s2 / (3 * rho + 1e-16),\
            1 / (b * rho**(4.0/3) + 1e-16)**2

def dalpha(rho_data):
    rho = rho_data[0,:]
    tau = rho_data[5,:]
    tau0 = get_uniform_tau(rho) + 1e-16
    mag_grad = get_gradient_magnitude(rho_data)
    tauw = get_single_orbital_tau(rho, mag_grad)
    dwdn, dwds = dtauw(rho_data)
    return 5.0 * (tauw - tau) / (3 * tau0 * rho + 1e-16) - dwdn / tau0,\
           - dwds / tau0,\
           1 / tau0

def project_xc_basis_l0(dedb, auxmol, grid):
    """
    Args:
        dedb (np.array(N)): The derivatives of the energy density
            at each grid point with respect to the basis function
            at that grid point.
        auxmol (gto.Mole): the Mole object containing the functional
            basis function
        grid (Grids): Grids object containing the real space grid for
            the molecule
        elda (np.array(N)): LDA exchange energy density
    """
    max_mem = safe_mem_cap_mb()
    N = grid.weights.shape[0]
    num_chunks = int(N**2 * 8 / (1e6 * max_mem)) + 1
    chunk_size = N // num_chunks
    if N % num_chunks != 0:
        chunk_size += 1
    vbas = np.zeros(N)
    for chunk in range(num_chunks):
        # consider making a mask?
        # shape (N, norb)
        min_ind = chunk * chunk_size 
        max_ind = min((chunk+1) * chunk_size, N)
        non0tab = make_mask(auxmol, grid.coords,
                            shls_slice = (min_ind, max_ind))
        ao = eval_ao(auxmol, grid.coords, shls_slice = (min_ind, max_ind),
                     non0tab = non0tab)
        vbas += np.dot(ao, dedb[min_ind:max_ind] * grid.weights[min_ind:max_ind])
    return vbas

def project_xc_basis(dedb, auxmol, grid, l = 0):
    """
    Args:
        dedb (np.array(2*l+1, N)): The derivatives of the energy density
            at each grid point with respect to the basis function(s)
            at that grid point. 2*l+1 is the number of basis functions
            for  the given l value
        auxmol (gto.Mole): the Mole object containing the functional
            basis function
        grid (Grids): Grids object containing the real space grid for
            the molecule
        elda (np.array(N)): LDA exchange energy density
    """
    max_mem = safe_mem_cap_mb()
    N = grid.weights.shape[0]
    num_chunks = int(N**2 * (2*l+1) * 8 / (1e6 * max_mem)) + 1
    chunk_size = N // num_chunks
    if N % num_chunks != 0:
        chunk_size += 1
    vbas = np.zeros(N)
    # has shape (N, 2l+1)
    energy_deriv = (dedb * grid.weights).T.flatten()
    r = np.linalg.norm(grid.coords, axis=1)
    for chunk in range(num_chunks):
        # consider making a mask?
        # shape (N, norb)
        min_ind = chunk * chunk_size
        max_ind = min((chunk+1) * chunk_size, N)
        non0tab = make_mask(auxmol, grid.coords,
                            shls_slice = (min_ind, max_ind))
        # ao has shape (N, nao) so (N, chunk_size * (2l+1))
        ao = eval_ao(auxmol, grid.coords, shls_slice = (min_ind, max_ind))
                     #non0tab = non0tab)
        #print(vbas.shape, ao.shape, energy_deriv.shape, energy_deriv[min_ind * (2*l+1) : max_ind * (2*l+1)].shape)
        vbas += np.dot(ao, energy_deriv[min_ind * (2*l+1) : max_ind * (2*l+1)])
    return vbas

def get_grid_nuc_distances(mol, grid):
    # shape (natm, 3)
    atm_coords
    coords = grid.coords
    dists = np.zeros((N, natm))
    for i in range(natm):
        dists[:,i] = np.linalg.norm(coords - atm_coords[i], axis=1)
    # shape (N, natm)
    return dists

def get_fbas_std(auxmol):
    """
    Get the standard deviations of the Gaussian basis at each point.
    """
    # shape (N)
    pass

def get_ind_sets(dists, std):
    # index of closest atm
    closest = np.argmin(dists, axis=1, dtype=np.int32)
    atm_in_range = np.zeros(N, natm, dtype=bool)
    for i in range(natm):
        atm_in_range[:,i] = np.logical_or(dists[:,i] < (3 * std),
                                          i == closest)
    sets = np.unique(atm_in_range, axis=0)
    return closest, atm_in_range, sets

def project_xc_basis_fast(dedb, auxmol, mol, grid, elda):
    """
    Args:
        dedb (np.array(N)): The derivatives of the energy density
            at each grid point with respect to the basis function
            at that grid point.
        auxmol (gto.Mole): the Mole object containing the functional
            basis function
        grid (Grids): Grids object containing the real space grid for
            the molecule
        elda (np.array(N)): LDA exchange energy density
    """
    dists = get_grid_nuc_distances(mol, grid)
    std = get_fbas_std(auxmol)
    closest, atm_in_range, sets = get_ind_sets(dists, std)
    for atm_set in sets:
        inds = np.where((atm_in_range == atm_set).all(axis=1))
        # need to reorder and sort the auxmol basis set so
        # I can do proper shell slicing, since pyscf only allows
        # indexing by a two-index range

    max_mem = safe_mem_cap_mb()
    N = grid.weights.shape[0]
    num_chunks = int(N**2 * 8 / (1e6 * max_mem)) + 1
    chunk_size = N // num_chunks
    if N % num_chunks != 0:
        chunk_size += 1
    vbas = np.zeros(N)
    for chunk in num_chunks:
        # consider making a mask?
        # shape (N, norb)
        min_ind = chunk * chunk_size 
        max_ind = np.min((chunk+1) * chunk_size, N)
        non0tab = make_mask(auxmol, grid.coords,
                            shls_slice = (min_ind, max_ind))
        ao = eval_ao(auxmol, grid.coords, shls_slice = (min_ind, max_ind),
                     non0tab = non0tab)
        vbas += np.dot(ao, elda * dedb[min_ind:max_ind] * grid.weights[min_ind:max_ind])
    return vbas

LDA_FACTOR = - 3.0 / 4.0 * (3.0 / np.pi)**(1.0/3)

"""
def v_semilocal(rho_data, F, dfds2, dfdalpha):
    # 0 - n, 1 - sigma, 2 - nabla^2, 3 - alpha
    v = np.zeros((4, rho_data.shape[1]))
    rho = rho_data[0,:]
    elda = LDA_FACTOR * rho**(4.0/3)
    # dedn line 1
    v[0] = 4.0 / 3 * LDA_FACTOR * rho**(1.0/3) * F
    ds2dn, ds2dsigma = ds2(rho_data)
    # dedn line 4a
    v[0] += elda * dfds2 * ds2dn
    v[1] += elda * d2ds2 * ds2dsigma
    dn, dsigma, dtau = dalpha(rho_data)
    # 
    v[0] += elda * dfdalpha * dn
    v[1] += elda * dfdalpha * dsigma
    v[3] += elda * dfdalpha * dtau
    return v
"""

def v_semilocal(rho_data, F, dfdp, dfdalpha):
    # 0 - n, 1 - p, 2 - nabla^2, 3 - alpha
    v = np.zeros((4, rho_data.shape[1]))
    rho = rho_data[0,:]
    elda = LDA_FACTOR * rho**(4.0/3)
    # dE/dn line 1
    v[0] = 4.0 / 3 * LDA_FACTOR * rho**(1.0/3) * F
    # dE/dp line 1
    v[1] = elda * dfdp
    # dE/dalpha line 1
    v[3] = elda * dfdalpha
    return v

def v_basis_transform(rho_data, v_npalpha):
    """
    Transforms the basis of the exchange potential from
    density, reduced gradient, and alpha to
    density, contracted gradient, and kinetic energy.
    v_npalpha is a 3xN array:
        0 - Functional derivative of the exchange energy
            explicitly with respect to the density, i.e.
            not accounting for derivatives of the XEF features
            wrt density
        1 - Functional derivative wrt the square of the reduced
            gradient p
        2 - ZERO (Functional derivative wrt normalized laplacian)
        3 - Functional derivative wrt the isoorbital indicator
            alpha
    Returns a 3xN array:
        0 - Full functional derivative of the exchange energy
            wrt the density, accounting for dp/dn and dalpha/dn
        1 - Derivative wrt sigma, the contracted gradient |nabla n|^2
        2 - ZERO (Derivative wrt the laplacian fo the density)
        3 - Derivative wrt tau, the kinetic energy density
    """
    v_nst = np.zeros(v_npalpha.shape)
    # dE/dn lines 1-3
    v_nst[0] = v_npalpha[0]
    dpdn, dpdsigma = ds2(rho_data)
    # dE/dn line 4 term 1
    v_nst[0] += v_npalpha[1] * dpdn
    # dE/dsigma term 1
    v_nst[1] += v_npalpha[1] * dpdsigma
    dadn, dadsigma, dadtau = dalpha(rho_data)
    # dE/dn line 4 term 2
    v_nst[0] += v_npalpha[3] * dadn
    # dE/dsigma term 2
    v_nst[1] += v_npalpha[3] * dadsigma
    # dE/dtau
    v_nst[3] = v_npalpha[3] * dadtau
    return v_nst

def v_nonlocal(rho_data, grid, dfdg, density, auxmol, g, l = 0, mul = 1.0):
    # g should have shape (2l+1, N)
    elda = LDA_FACTOR * rho_data[0]**(4.0/3)
    N = grid.weights.shape[0]
    lc = get_dft_input2(rho_data)[:3]
    if l == 0:
        dedb = (elda * dfdg).reshape(1, -1)
    elif l == 1:
        #dedb = 2 * elda * g * dfdg
        dedb = elda * g * dfdg / (np.linalg.norm(g, axis=0) + 1e-10)
    elif l == 2:
        dedb = 2 * elda * g * dfdg / np.sqrt(5)
    elif l == -2:
        dedb = elda * dfdg
        l = 2
    elif l == -1:
        dedb = elda * dfdg
        l = 1
    else:
        raise ValueError('angular momentum code l=%d unknown' % l)
    atm, bas, env = get_gaussian_grid(grid.coords, mul * rho_data[0],
                                      l = l, s = lc[1], alpha=lc[2])
    a = env[bas[:,5]]
    gridmol = gto.Mole(_atm = atm, _bas = bas, _env = env)
    # dE/dn line 2
    vbas = project_xc_basis(dedb, gridmol, grid, l)

    # (ngrid * (2l+1), naux)
    ovlp = gto.mole.intor_cross('int1e_ovlp', auxmol, gridmol).transpose()
    ovlp_deriv = gto.mole.intor_cross('int1e_r2_origj', auxmol, gridmol).transpose()
    g = np.dot(ovlp, density).reshape(N, 2*l+1).transpose()
    gr2 = np.dot(ovlp_deriv, density).reshape(N, 2*l+1).transpose()
    dgda = l / (2 * a) * g - gr2

    fac = (6 * np.pi**2)**(2.0/3) / (16 * np.pi)
    dadn = 2 * a / (3 * lc[0] + 1e-10)
    dadp = np.pi * fac * (lc[0] / 2)**(2.0/3)
    dadalpha = 0.6 * np.pi * fac * (lc[0] / 2)**(2.0/3)
    # add in line 3 of dE/dn, line 2 of dE/dp and dE/dalpha
    v_npa = np.zeros((4, N))
    #print('shapes', dedb.shape, dgda.shape)
    deda = np.einsum('mi,mi->i', dedb, dgda)
    v_npa[0] = vbas + deda * dadn
    v_npa[1] = deda * dadp
    v_npa[3] = deda * dadalpha
    return v_npa

def v_nonlocal_fast(rho_data, grid, dfdg, density, auxmol, g, l = 0, mul = 1.0):
    # g should have shape (2l+1, N)
    elda = LDA_FACTOR * rho_data[0]**(4.0/3)
    N = grid.weights.shape[0]
    lc = get_dft_input2(rho_data)[:3]
    if l == 0:
        dedb = (elda * dfdg).reshape(1, -1)
    elif l == 1:
        #dedb = 2 * elda * g * dfdg
        dedb = elda * g * dfdg / (np.linalg.norm(g, axis=0) + 1e-10)
    elif l == 2:
        dedb = 2 * elda * g * dfdg / np.sqrt(5)
    elif l == -2:
        dedb = elda * dfdg
        l = 2
    elif l == -1:
        dedb = elda * dfdg
        l = 1
    else:
        raise ValueError('angular momentum code l=%d unknown' % l)
    atm, bas, env = get_gaussian_grid(grid.coords, mul * rho_data[0],
                                      l = l, s = lc[1], alpha=lc[2])
    a = env[bas[:,5]]
    gridmol = gto.Mole(_atm = atm, _bas = bas, _env = env)

    # (ngrid * (2l+1), naux)
    ovlp = gto.mole.intor_cross('int1e_ovlp', auxmol, gridmol).transpose()
    ovlp_deriv = gto.mole.intor_cross('int1e_r2_origj', auxmol, gridmol).transpose()
    g = np.dot(ovlp, density).reshape(N, 2*l+1).transpose()
    gr2 = np.dot(ovlp_deriv, density).reshape(N, 2*l+1).transpose()
    dedaux = np.dot((dedb * grid.weights).T.flatten(), ovlp)
    dgda = l / (2 * a) * g - gr2

    fac = (6 * np.pi**2)**(2.0/3) / (16 * np.pi)
    dadn = 1 * a / (3 * (lc[0] / 2 + 1e-6))
    dadp = np.pi * fac * (lc[0] / 2 + 1e-6)**(2.0/3)
    dadalpha = 0.6 * np.pi * fac * (lc[0] / 2 + 1e-6)**(2.0/3)
    # add in line 3 of dE/dn, line 2 of dE/dp and dE/dalpha
    v_npa = np.zeros((4, N))
    #print('shapes', dedb.shape, dgda.shape)
    deda = np.einsum('mi,mi->i', dedb, dgda)
    v_npa[0] = deda * dadn
    v_npa[1] = deda * dadp
    v_npa[3] = deda * dadalpha
    return v_npa, dedaux

def v_nonlocal_extra_fast(rho_data, grid, dfdg, density, auxmol,
                          g, gr2, ovlp, l = 0, mul = 1.0):
    # g should have shape (2l+1, N)
    elda = LDA_FACTOR * rho_data[0]**(4.0/3)
    N = grid.weights.shape[0]
    lc = get_dft_input2(rho_data)[:3]
    if l == 0:
        dedb = (elda * dfdg).reshape(1, -1)
    elif l == 1:
        #dedb = 2 * elda * g * dfdg
        dedb = elda * g * dfdg / (np.linalg.norm(g, axis=0) + 1e-10)
    elif l == 2:
        dedb = 2 * elda * g * dfdg / np.sqrt(5)
    elif l == -2:
        dedb = elda * dfdg
        l = 2
    elif l == -1:
        dedb = elda * dfdg
        l = 1
    else:
        raise ValueError('angular momentum code l=%d unknown' % l)

    rho, s, alpha = lc
    a = np.pi * (mul * rho / 2 + 1e-6)**(2.0 / 3)
    scale = 1
    #fac = (6 * np.pi**2)**(2.0/3) / (16 * np.pi)
    fac = (6 * np.pi**2)**(2.0/3) / (16 * np.pi)
    scale += fac * s**2
    scale += 3.0 / 5 * fac * (alpha - 1)
    a = a * scale
    a[rho<1e-8] = 1e16

    # (ngrid * (2l+1), naux)
    dedaux = np.dot((dedb * grid.weights).T.flatten(), ovlp)
    dgda = l / (2 * a) * g - gr2
    dgda[:,rho<1e-8] = 0

    fac = (6 * np.pi**2)**(2.0/3) / (16 * np.pi)
    dadn = mul * a / (3 * (mul * rho / 2 + 1e-6))
    dadp = np.pi * fac * (mul * rho / 2 + 1e-6)**(2.0/3)
    dadalpha = 0.6 * np.pi * fac * (mul * rho / 2 + 1e-6)**(2.0/3)
    # add in line 3 of dE/dn, line 2 of dE/dp and dE/dalpha
    v_npa = np.zeros((4, N))
    deda = np.einsum('mi,mi->i', dedb, dgda)
    v_npa[0] = deda * dadn
    v_npa[1] = deda * dadp
    v_npa[3] = deda * dadalpha
    return v_npa, dedaux

def v_nonlocal_general(rho_data, grid, dedg, density, auxmol,
                          g, gr2, ovlp, l = 0, mul = 1.0):
    # g should have shape (2l+1, N)
    N = grid.weights.shape[0]
    lc = get_dft_input2(rho_data)[:3]
    if l == 0:
        dedb = dedg.reshape(1, -1)
    elif l == 1:
        #dedb = 2 * elda * g * dfdg
        dedb = dedg * g / (np.linalg.norm(g, axis=0) + 1e-10)
    elif l == 2:
        dedb = 2 * dedg * g / np.sqrt(5)
    elif l == -2:
        dedb = dedg
        l = 2
    elif l == -1:
        dedb = dedg
        l = 1
    else:
        raise ValueError('angular momentum code l=%d unknown' % l)

    rho, s, alpha = lc
    a = np.pi * (mul * rho / 2 + 1e-6)**(2.0 / 3)
    scale = 1
    #fac = (6 * np.pi**2)**(2.0/3) / (16 * np.pi)
    fac = (6 * np.pi**2)**(2.0/3) / (16 * np.pi)
    scale += fac * s**2
    scale += 3.0 / 5 * fac * (alpha - 1)
    a = a * scale
    a[rho<1e-8] = 1e16

    # (ngrid * (2l+1), naux)
    dedaux = np.dot((dedb * grid.weights).T.flatten(), ovlp)
    dgda = l / (2 * a) * g - gr2
    dgda[:,rho<1e-8] = 0

    fac = (6 * np.pi**2)**(2.0/3) / (16 * np.pi)
    dadn = mul * a / (3 * (mul * rho / 2 + 1e-6))
    dadp = np.pi * fac * (mul * rho / 2 + 1e-6)**(2.0/3)
    dadalpha = 0.6 * np.pi * fac * (mul * rho / 2 + 1e-6)**(2.0/3)
    # add in line 3 of dE/dn, line 2 of dE/dp and dE/dalpha
    v_npa = np.zeros((4, N))
    deda = np.einsum('mi,mi->i', dedb, dgda)
    v_npa[0] = deda * dadn
    v_npa[1] = deda * dadp
    v_npa[3] = deda * dadalpha
    return v_npa, dedaux

def functional_derivative_loop(mol, mlfunc, dEddesc, contracted_desc,
                               raw_desc, raw_desc_r2,
                               rho_data, density, ovlps, grid):

    N = grid.weights.shape[0]
    naux = mol.auxmol.nao_nr()
    sprefac = 2 * (3 * np.pi * np.pi)**(1.0/3)
    n43 = rho_data[0]**(4.0/3)
    svec = rho_data[1:4] / (sprefac * n43 + 1e-20)
    v_aniso = np.zeros((3, N))
    v_aux = np.zeros(naux)

    for i, d in enumerate(mlfunc.desc_list):
        if d.code == 0:
            continue
        elif d.code == 1:
            v_npa[1] += dEddesc[:,i]
        elif d.code == 2:
            v_npa[3] += dEddesc[:,i]
        else:
            if d.code in [4, 15, 16]:
                g = contracted_desc[d.code]
                if d.code == 4:
                    ovlp = ovlps[0]
                    gr2 = raw_desc_r2[12:13]
                elif d.code == 15:
                    ovlp = ovlps[3]
                    gr2 = raw_desc_r2[21:22]
                else:
                    ovlp = ovlps[4]
                    gr2 = raw_desc_r2[22:23]
                l = 0
            elif d.code == 5:
                g = raw_desc[13:16]
                gr2 = raw_desc_r2[13:16]
                ovlp = ovlps[1]
                l = 1
            elif d.code == 8:
                g = raw_desc[16:21]
                gr2 = raw_desc_r2[16:21]
                ovlp = ovlps[2]
                l = 2
            elif d.code == 6:
                g = raw_desc[13:16]
                gr2 = raw_desc_r2[13:16]
                ovlp = ovlps[1]
                dfmul = svec
                v_aniso += dEddesc[:,i] * g
                l = -1
            elif d.code == 12:
                l = -2
                g = raw_desc[16:21]
                gr2 = raw_desc_r2[16:21]
                ovlp = ovlps[2]
                dfmul = contract21_deriv(svec)
                ddesc_dsvec = contract21(g, svec)
                v_aniso += dEddesc[:,i] * 2 * ddesc_dsvec
            elif d.code == 13:
                g2 = raw_desc[16:21]
                g2r2 = raw_desc_r2[16:21]
                ovlp2 = ovlps[2]
                g1 = raw_desc[13:16]
                g1r2 = raw_desc_r2[13:16]
                ovlp1 = ovlps[1]
                dfmul = contract21_deriv(svec, g1)
                ddesc_dsvec = contract21(g2, g1)
                ddesc_dg1 = contract21(g2, svec)
                v_aniso += dEddesc[:,i] * ddesc_dsvec
                vtmp1, dedaux1 = v_nonlocal_general(rho_data, grid,
                                         dEddesc[:,i] * ddesc_dg1,
                                         density, mol.auxmol, g1,
                                         g1r2, ovlp1, l = -1,
                                         mul = d.mul)
                vtmp2, dedaux2 = v_nonlocal_general(rho_data, grid,
                                         dEddesc[:,i] * dfmul,
                                         density, mol.auxmol, g2,
                                         g2r2, ovlp2, l = -2,
                                         mul = d.mul)
                vtmp = vtmp1 + vtmp2
                dedaux = dedaux1 + dedaux2
            else:
                raise NotImplementedError('Cannot take derivative for code %d' % d.code)

            if d.code in [6, 12]:
                vtmp, dedaux = v_nonlocal_general(rho_data, grid,
                                         dEddesc[:,i] * dfmul,
                                         density, mol.auxmol, g,
                                         gr2, ovlp, l = l,
                                         mul = d.mul)
            elif d.code == 13:
                pass
            else:
                vtmp, dedaux = v_nonlocal_general(rho_data, grid,
                                         dEddesc[:,i],
                                         density, mol.auxmol, g,
                                         gr2, ovlp, l = l,
                                         mul = d.mul)
            v_npa += vtmp
            v_aux += dedaux

    vmol = np.einsum('a,aij->ij', v_aux, mol.ao_to_aux)
    v_nst = v_basis_transform(rho_data, v_npa)
    v_nst[0] += np.einsum('ap,ap->p', -4.0 * svec / (3 * rho_data[0] + 1e-20), v_aniso)
    v_grad = v_aniso / (sprefac * n43 + 1e-20)

    return v_nst, v_grad, vmol


def get_density_in_basis(ao_to_aux, rdm1):
    return np.einsum('npq,pq->n', ao_to_aux, rdm1)

def arcsinh_deriv(x):
    return 1 / np.sqrt(x * x + 1)

def get_chi(alpha):
    return 1 / (1 + alpha**2)

def chi_deriv(alpha):
    return -2 * alpha / (1 + alpha**2)**2

