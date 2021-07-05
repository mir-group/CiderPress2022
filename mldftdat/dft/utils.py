import numpy as np 
from mldftdat.pyscf_utils import *
from mldftdat.workflow_utils import safe_mem_cap_mb
from pyscf.dft.numint import eval_ao, make_mask
from mldftdat.density import LDA_FACTOR,\
                             contract21_deriv, contract21, GG_AMIN

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

LDA_FACTOR = - 3.0 / 4.0 * (3.0 / np.pi)**(1.0/3)

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


def v_nonlocal_general(rho_data, grid, dedg, density, auxmol,
                       g, gr2, ovlp, l = 0, mul = 1.0):
    # g should have shape (2l+1, N)
    N = grid.weights.shape[0]
    lc = get_dft_input2(rho_data)[:3]
    if l == 0:
        dedb = dedg.reshape(1, -1)
    elif l == 1:
        #dedb = 2 * elda * g * dfdg
        dedb = 2 * dedg * g #/ (np.linalg.norm(g, axis=0) + 1e-10)
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
    a = np.pi * (mul * rho / 2 + 1e-16)**(2.0 / 3)
    scale = 1
    fac = (6 * np.pi**2)**(2.0/3) / (16 * np.pi)
    scale += GG_SMUL * fac * s**2
    scale += GG_AMUL * 0.6 * fac * (alpha - 1)
    a = a * scale
    cond = a < GG_AMIN
    da = np.exp(a[cond] / GG_AMIN - 1)
    a[cond] = GG_AMIN * np.exp(a[cond] / GG_AMIN - 1)

    # (ngrid * (2l+1), naux)
    dedb[:,rho<1e-8] = 0
    dedaux = np.dot((dedb * grid.weights).T.flatten(), ovlp)
    dgda = l / (2 * a) * g - gr2
    #print(dgda.shape, gr2.shape)
    dgda[:,rho<1e-8] = 0

    dadn = mul * a / (3 * (mul * rho / 2 + 1e-16))
    dadp = GG_SMUL * np.pi * fac * (mul * rho / 2 + 1e-16)**(2.0/3)
    dadalpha = GG_AMUL * 0.6 * np.pi * fac * (mul * rho / 2 + 1e-16)**(2.0/3)
    dadn[cond] *= da
    dadp[cond] *= da
    dadalpha[cond] *= da
    # add in line 3 of dE/dn, line 2 of dE/dp and dE/dalpha

    v_npa = np.zeros((4, N))
    deda = np.einsum('mi,mi->i', dedb, dgda)
    v_npa[0] = deda * dadn
    v_npa[1] = deda * dadp
    v_npa[3] = deda * dadalpha
    return v_npa, dedaux

def v_nonlocal(rho_data, grid, dedg, density, auxmol,
               g, gr2, ovlp, l=0, a0=8.0, fac_mul=0.25,
               amin=GG_AMIN, l_add=0, **kwargs):
    #print(l, l_add, a0, fac_mul, amin)
    # g should have shape (2l+1, N)
    N = grid.weights.shape[0]
    lc = get_dft_input2(rho_data)[:3]
    if l == 0:
        dedb = dedg.reshape(1, -1)
    elif l == 1:
        dedb = 2 * dedg * g
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
    ratio = alpha + 5./3 * s**2
    fac = fac_mul * 1.2 * (6 * np.pi**2)**(2.0/3) / np.pi
    a = np.pi * (rho / 2 + 1e-16)**(2.0 / 3)
    scale = a0 + (ratio-1) * fac
    a = a * scale
    cond = a < amin
    da = np.exp(a[cond] / amin - 1)
    a[cond] = amin * np.exp(a[cond] / amin - 1)

    # (ngrid * (2l+1), naux)
    dedb[:,rho<1e-8] = 0
    dedaux = np.dot((dedb * grid.weights).T.flatten(), ovlp)
    dgda = (l + l_add) / (2 * a) * g - gr2
    dgda[:,rho<1e-8] = 0

    dadn = 2 * a / (3 * rho + 1e-16)
    dadalpha = np.pi * fac * (rho / 2 + 1e-16)**(2.0/3)
    dadp = 5./3 * dadalpha
    dadn[cond] *= da
    dadp[cond] *= da
    dadalpha[cond] *= da
    # add in line 3 of dE/dn, line 2 of dE/dp and dE/dalpha

    v_npa = np.zeros((4, N))
    deda = np.einsum('mi,mi->i', dedb, dgda)
    v_npa[0] = deda * dadn
    v_npa[1] = deda * dadp
    v_npa[3] = deda * dadalpha
    return v_npa, dedaux


def functional_derivative_loop(mol, mlfunc, dEddesc,
                                 raw_desc, raw_desc_r2,
                                 rho_data, density, ovlps, grid):
    """
    Core functional derivative loop for the CIDER features,
    called by NLNumInt.
    Args:
        mol (pyscf.gto.Mole): molecule object
        mlfunc (MLFunctional): Exchange functional
        dEddesc (np.ndarray): ngrid x ndesc array of energy derivatives
            with respect to the descriptors.
        raw_desc (np.ndarray): raw CIDER descriptor vectors
        raw_desc_r2 (np.ndarray): raw CIDER descriptor vectors <r^2>
            for use in functional derivative with respect to the Gaussian
            exponents
        rho_data (np.ndarray): 6 x ngrid
        density (np.ndarray): density in DF basis space
        ovlps (np.ndarray): Overlaps of the CIDER descriptor functions with
            the DF basis
        grid: contains coords and weights of the real-space grid
    """

    gg_dict = {
        'a0': mlfunc.a0,
        'amin': mlfunc.amin,
        'fac_mul': mlfunc.fac_mul
    }
    N = grid.weights.shape[0]
    naux = mol.auxmol.nao_nr()
    sprefac = 2 * (3 * np.pi * np.pi)**(1.0/3)
    n43 = rho_data[0]**(4.0/3)
    svec = rho_data[1:4] / (sprefac * n43 + 1e-20)
    v_npa = np.zeros((4, N))
    v_aniso = np.zeros((3, N))
    v_aux = np.zeros(naux)

    for i, d in enumerate(mlfunc.desc_order):
        if d == 0:
            v_npa[0] += dEddesc[:,i]
        elif d == 1:
            v_npa[1] += dEddesc[:,i]
        elif d == 2:
            v_npa[3] += dEddesc[:,i]
        else:
            gg_kwargs = gg_dict
            l_add = 0
            if d in [3, 10, 11]:
                if d == 3:
                    g = raw_desc[6]
                    ovlp = ovlps[0]
                    gr2 = raw_desc_r2[6:7]
                elif d == 10:
                    g = raw_desc[15]
                    ovlp = ovlps[3]
                    gr2 = raw_desc_r2[15:16]
                    if mlfunc.desc_version == 'c':
                        l_add = 2
                        mul = 1.0
                    else:
                        mul = 0.25**(2./3)
                    gg_kwargs = {
                        'a0': mlfunc.a0 * mul,
                        'fac_mul': mlfunc.fac_mul * mul,
                        'amin': mlfunc.amin * mul
                    }
                else:
                    g = raw_desc[16]
                    ovlp = ovlps[4]
                    gr2 = raw_desc_r2[16:17]
                    if mlfunc.desc_version == 'c':
                        mul = 2.0
                    else:
                        mul = 4**(2./3)
                    gg_kwargs = {
                        'a0': mlfunc.a0 * mul,
                        'fac_mul': mlfunc.fac_mul * mul,
                        'amin': mlfunc.amin * mul
                    }
                l = 0
            elif d == 4:
                g = raw_desc[7:10]
                gr2 = raw_desc_r2[7:10]
                ovlp = ovlps[1]
                l = 1
            elif d == 6:
                g = raw_desc[10:15]
                gr2 = raw_desc_r2[10:15]
                ovlp = ovlps[2]
                l = 2
            elif d == 5:
                g = raw_desc[7:10]
                gr2 = raw_desc_r2[7:10]
                ovlp = ovlps[1]
                dfmul = svec
                v_aniso += dEddesc[:,i] * g
                l = -1
            elif d == 7:
                l = -2
                g = raw_desc[10:15]
                gr2 = raw_desc_r2[10:15]
                ovlp = ovlps[2]
                dfmul = contract21_deriv(svec)
                ddesc_dsvec = contract21(g, svec)
                v_aniso += dEddesc[:,i] * 2 * ddesc_dsvec
            elif d == 8:
                g2 = raw_desc[10:15]
                g2r2 = raw_desc_r2[10:15]
                ovlp2 = ovlps[2]
                g1 = raw_desc[7:10]
                g1r2 = raw_desc_r2[7:10]
                ovlp1 = ovlps[1]
                dfmul = contract21_deriv(svec, g1)
                ddesc_dsvec = contract21(g2, g1)
                ddesc_dg1 = contract21(g2, svec)
                v_aniso += dEddesc[:,i] * ddesc_dsvec
                vtmp1, dedaux1 = v_nonlocal(rho_data, grid,
                                         dEddesc[:,i] * ddesc_dg1,
                                         density, mol.auxmol, g1,
                                         g1r2, ovlp1, l=-1, **gg_kwargs)
                vtmp2, dedaux2 = v_nonlocal(rho_data, grid,
                                         dEddesc[:,i] * dfmul,
                                         density, mol.auxmol, g2,
                                         g2r2, ovlp2, l=-2, **gg_kwargs)
                vtmp = vtmp1 + vtmp2
                dedaux = dedaux1 + dedaux2
            elif d == 9:
                g2 = raw_desc[10:15]
                g2r2 = raw_desc_r2[10:15]
                ovlp2 = ovlps[2]
                g1 = raw_desc[7:10]
                g1r2 = raw_desc_r2[7:10]
                ovlp1 = ovlps[1]
                dfmul = contract21_deriv(g1)
                ddesc_dg1 = 2 * contract21(g2, g1)
                vtmp1, dedaux1 = v_nonlocal(rho_data, grid,
                                         dEddesc[:,i] * ddesc_dg1,
                                         density, mol.auxmol, g1,
                                         g1r2, ovlp1, l=-1, **gg_kwargs)
                vtmp2, dedaux2 = v_nonlocal(rho_data, grid,
                                         dEddesc[:,i] * dfmul,
                                         density, mol.auxmol, g2,
                                         g2r2, ovlp2, l=-2, **gg_kwargs)
                vtmp = vtmp1 + vtmp2
                dedaux = dedaux1 + dedaux2
            else:
                raise NotImplementedError('Cannot take derivative for code %d' % d)

            if d in [5, 7]:
                vtmp, dedaux = v_nonlocal(rho_data, grid,
                                         dEddesc[:,i] * dfmul,
                                         density, mol.auxmol, g,
                                         gr2, ovlp, l=l, **gg_kwargs)
            elif d in [8, 9]:
                pass
            else:
                vtmp, dedaux = v_nonlocal(rho_data, grid,
                                         dEddesc[:,i],
                                         density, mol.auxmol, g,
                                         gr2, ovlp, l=l, l_add=l_add,
                                         **gg_kwargs)
            
            v_npa += vtmp
            v_aux += dedaux
            vtmp = None
            dedaux = None

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

