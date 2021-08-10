import pyscf.dft.numint as pyscf_numint
from pyscf.dft.numint import _rks_gga_wv0, _scale_ao, _dot_ao_ao, _format_uks_dm
from pyscf.dft.libxc import eval_xc
from pyscf.dft.gen_grid import Grids
from pyscf import df, scf, dft

from mldftdat.density import get_x_helper_full_a, get_x_helper_full_c, LDA_FACTOR,\
                             contract_exchange_descriptors,\
                             contract21_deriv, contract21
from scipy.linalg import cho_factor, cho_solve
from mldftdat.dft.utils import *

import numpy as np
import logging
import time

from mldftdat.pyscf_utils import get_dft_input2
from mldftdat.density import get_gaussian_grid_c
from mldftdat.dft.utils import v_nonlocal, v_basis_transform



###################################################################
# Adjusted version of PySCF numint helper functions that accounts #
# for anisotropic gradient potential.                             #
###################################################################


def _rks_gga_wv0a(rho, vxc, weight):
    vrho, vgamma, vgrad = vxc[0], vxc[1], vxc[4]
    ngrid = vrho.size
    wv = np.empty((4,ngrid))
    wv[0]  = weight * vrho
    wv[1:] = (weight * vgamma * 2) * rho[1:4]
    # anisotropic component of the gradient derivative
    wv[1:] += weight * vgrad
    wv[0] *= .5  # v+v.T should be applied in the caller
    return wv

def _uks_gga_wv0a(rho, vxc, weight):
    rhoa, rhob = rho
    vrho, vsigma = vxc[:2]
    vgrad = vxc[4]
    ngrid = vrho.shape[0]
    wva = np.empty((4,ngrid))
    wva[0]  = weight * vrho[:,0] * .5  # v+v.T should be applied in the caller
    wva[1:] = rhoa[1:4] * (weight * vsigma[:,0] * 2)  # sigma_uu
    wva[1:]+= rhob[1:4] * (weight * vsigma[:,1])      # sigma_ud
    wva[1:]+= weight * vgrad[:,:,0]
    wvb = np.empty((4,ngrid))
    wvb[0]  = weight * vrho[:,1] * .5  # v+v.T should be applied in the caller
    wvb[1:] = rhob[1:4] * (weight * vsigma[:,2] * 2)  # sigma_dd
    wvb[1:]+= rhoa[1:4] * (weight * vsigma[:,1])      # sigma_ud
    wvb[1:]+= weight * vgrad[:,:,1]
    return wva, wvb

def get_gaussian_grid_simple(coords):
    N = coords.shape[0]
    auxmol = gto.fakemol_for_charges(coords)
    atm = auxmol._atm.copy()
    bas = auxmol._bas.copy()
    start = auxmol._env.shape[0] - 2
    env = np.zeros(start + 2 * N)
    env[:start] = auxmol._env[:-2]
    bas[:,5] = start + np.arange(N)
    bas[:,6] = start + N + np.arange(N)
    bas[:,1] = 0
    env[bas[:,5]] = 0.2
    env[bas[:,6]] = np.sqrt(4*np.pi)

    return atm, bas, env

def get_x_helper_simple(auxmol, rho_data, grid, density,
                        deriv=False, return_ovlp=False,
                        a0=8.0, fac_mul=0.25, amin=GG_AMIN):
    lc = get_dft_input2(rho_data)[:3]
    # size naux
    integral_name = 'int1e_r2_origj' if deriv else 'int1e_ovlp'
    N = grid.weights.shape[0]
    if return_ovlp:
        ovlps = []
    l = 0
    atm, bas, env = get_gaussian_grid_c(grid.coords, rho_data[0],
                                        l=l, s=lc[1], alpha=lc[2],
                                        a0=a0, fac_mul=fac_mul,
                                        amin=amin)
    gridmol = gto.Mole(_atm=atm, _bas=bas, _env=env)
    # (ngrid * (2l+1), naux)
    ovlp = gto.mole.intor_cross(integral_name, auxmol, gridmol).T
    proj = np.dot(ovlp, density).reshape(N, 2*l+1).transpose()
    if return_ovlp:
        return proj, ovlp
    else:
        return proj

def project_density_simple(coords, density, auxmol):
    atm, bas, env = get_gaussian_grid_simple(coords)
    gridmol = gto.Mole(_atm=atm, _bas=bas, _env=env)
    # (ngrid * (2l+1), naux)
    ovlp = gto.mole.intor_cross('int1e_ovlp', auxmol, gridmol).T
    proj = np.dot(ovlp, density).reshape(-1).transpose()
    return proj, ovlp

def project_density_cider(grid, density, auxmol, rho_data, gg_kwargs):
    proj, ovlp = get_x_helper_simple(auxmol, rho_data, grid,
                                     density, return_ovlp=True, **gg_kwargs)
    dproj = get_x_helper_simple(auxmol, rho_data, grid,
                                density, deriv=True, **gg_kwargs)
    return proj, ovlp, dproj


class QuickGrid():
    def __init__(self, coords, weights):
        self.coords = coords
        self.weights = weights

def nr_rks(ni, mol, grids, xc_code, dms, relativity=0, hermi=0,
           max_memory=2000, verbose=None):

    if not (hasattr(mol, 'ao_to_aux') and hasattr(mol, 'auxmol')):
        mol.auxmol, mol.ao_to_aux = setup_aux(mol)

    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    density = []
    if nset > 1:
        for idm in range(nset):
            density.append(np.einsum('npq,pq->n', mol.ao_to_aux, dms[idm]))
    else:
        density.append(np.einsum('npq,pq->n', mol.ao_to_aux, dms))

    nelec = np.zeros(nset)
    excsum = np.zeros(nset)
    vmat = np.zeros((nset, nao, nao))
    aow = None

    ao_deriv = 2
    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
        ngrid = weight.size
        aow = np.ndarray(ao[0].shape, order='F', buffer=aow)
        for idm in range(nset):
            logging.debug('dm shape %s', str(dms.shape))
            rho = make_rho(idm, ao, mask, 'MGGA')
            exc, vxc = ni.eval_xc_cider(
                xc_code, mol, rho, QuickGrid(coords, weight), density[idm],
                0, relativity, 1,
                verbose=verbose)[:2]
            vrho, vsigma, vlapl, vtau, vgrad, vmol = vxc[:6]
            den = rho[0] * weight
            nelec[idm] += den.sum()
            excsum[idm] += np.dot(den, exc)

            wv = _rks_gga_wv0a(rho, vxc, weight)
            #:aow = numpy.einsum('npi,np->pi', ao[:4], wv, out=aow)
            aow = _scale_ao(ao[:4], wv, out=aow)
            vmat[idm] += _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)

# FIXME: .5 * .5   First 0.5 for v+v.T symmetrization.
# Second 0.5 is due to the Libxc convention tau = 1/2 \nabla\phi\dot\nabla\phi
            wv = (.5 * .5 * weight * vtau).reshape(-1,1)
            vmat[idm] += _dot_ao_ao(mol, ao[1], wv*ao[1], mask, shls_slice, ao_loc)
            vmat[idm] += _dot_ao_ao(mol, ao[2], wv*ao[2], mask, shls_slice, ao_loc)
            vmat[idm] += _dot_ao_ao(mol, ao[3], wv*ao[3], mask, shls_slice, ao_loc)

            vmat[idm] += 0.5 * vmol

            rho = exc = vxc = vrho = vsigma = wv = None

    for i in range(nset):
        vmat[i] = vmat[i] + vmat[i].T
    if nset == 1:
        nelec = nelec[0]
        excsum = excsum[0]
        vmat = vmat.reshape(nao,nao)
    return nelec, excsum, vmat


def nr_uks(ni, mol, grids, xc_code, dms, relativity=0, hermi=0,
           max_memory=2000, verbose=None):
    if not (hasattr(mol, 'ao_to_aux') and hasattr(mol, 'auxmol')):
        mol.auxmol, mol.ao_to_aux = setup_aux(mol)

    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    dma, dmb = _format_uks_dm(dms)

    nao = dma.shape[-1]
    make_rhoa, nset = ni._gen_rho_evaluator(mol, dma, hermi)[:2]
    make_rhob       = ni._gen_rho_evaluator(mol, dmb, hermi)[0]

    density = []
    if nset > 1:
        for idm in range(nset):
            density.append((np.einsum('npq,pq->n', mol.ao_to_aux, dma[idm]),\
                   np.einsum('npq,pq->n', mol.ao_to_aux, dmb[idm])))
    else:
        density = [(np.einsum('npq,pq->n', mol.ao_to_aux, dma),\
                                   np.einsum('npq,pq->n', mol.ao_to_aux, dmb))]

    nelec = np.zeros((2,nset))
    excsum = np.zeros(nset)
    vmat = np.zeros((2,nset,nao,nao))
    aow = None
    ao_deriv = 2
    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
        ngrid = weight.size
        aow = np.ndarray(ao[0].shape, order='F', buffer=aow)
        for idm in range(nset):
            logging.debug('dm shape %s %s', str(dma.shape), str(dmb.shape))
            rho_a = make_rhoa(idm, ao, mask, 'MGGA')
            rho_b = make_rhob(idm, ao, mask, 'MGGA')
            exc, vxc = ni.eval_xc_cider(
                xc_code, mol, (rho_a, rho_b),
                QuickGrid(coords, weight), density[idm],
                1, relativity, 1, verbose=verbose)[:2]
            vrho, vsigma, vlapl, vtau, vgrad, vmol = vxc[:6]
            den = rho_a[0]*weight
            nelec[0,idm] += den.sum()
            excsum[idm] += np.dot(den, exc)
            den = rho_b[0]*weight
            nelec[1,idm] += den.sum()
            excsum[idm] += np.dot(den, exc)

            wva, wvb = _uks_gga_wv0a((rho_a,rho_b), vxc, weight)
            #:aow = np.einsum('npi,np->pi', ao[:4], wva, out=aow)
            aow = _scale_ao(ao[:4], wva, out=aow)
            vmat[0,idm] += _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
            #:aow = np.einsum('npi,np->pi', ao[:4], wvb, out=aow)
            aow = _scale_ao(ao[:4], wvb, out=aow)
            vmat[1,idm] += _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
            logging.debug(np.max(np.abs(vmat[1,idm])))
# FIXME: .5 * .5   First 0.5 for v+v.T symmetrization.
# Second 0.5 is due to the Libxc convention tau = 1/2 \nabla\phi\dot\nabla\phi
            wv = (.25 * weight * vtau[:,0]).reshape(-1,1)
            vmat[0,idm] += _dot_ao_ao(mol, ao[1], wv*ao[1], mask, shls_slice, ao_loc)
            vmat[0,idm] += _dot_ao_ao(mol, ao[2], wv*ao[2], mask, shls_slice, ao_loc)
            vmat[0,idm] += _dot_ao_ao(mol, ao[3], wv*ao[3], mask, shls_slice, ao_loc)
            wv = (.25 * weight * vtau[:,1]).reshape(-1,1)
            vmat[1,idm] += _dot_ao_ao(mol, ao[1], wv*ao[1], mask, shls_slice, ao_loc)
            vmat[1,idm] += _dot_ao_ao(mol, ao[2], wv*ao[2], mask, shls_slice, ao_loc)
            vmat[1,idm] += _dot_ao_ao(mol, ao[3], wv*ao[3], mask, shls_slice, ao_loc)
            logging.debug(np.max(np.abs(vmat[1,idm])))
            vmat[0,idm] += 0.5 * vmol[0,:,:]
            vmat[1,idm] += 0.5 * vmol[1,:,:]
            logging.debug(np.max(np.abs(vmat[1,idm])))

            rho_a = rho_b = exc = vxc = vrho = vsigma = wva = wvb = None

    for i in range(nset):
        vmat[0,i] = vmat[0,i] + vmat[0,i].T
        vmat[1,i] = vmat[1,i] + vmat[1,i].T
        logging.debug("VMAT {}".format(np.max(np.abs(vmat[0,i]))))
        logging.debug("VMAT {}".format(np.max(np.abs(vmat[1,i]))))
    if isinstance(dma, np.ndarray) and dma.ndim == 2:
        vmat = vmat[:,0]
        nelec = nelec.reshape(2)
        excsum = excsum[0]
    return nelec, excsum, vmat



##############################################
# Numerical integrator for CIDER functionals #
##############################################


class NLNumInt(pyscf_numint.NumInt):

    def __init__(self, xf, cider=False):
        self.xf = xf
        self.cider = cider
        super(NLNumInt, self).__init__()

    def nr_rks(ni, mol, grids, xc_code, dms, relativity=0, hermi=0,
               max_memory=2000, verbose=None):
        if '__VV10' in xc_code:
            return super(NLNumInt, ni).nr_rks(
                mol, grids, xc_code, dms, relativity, hermi,
                max_memory, verbose
            )
        else:
            return nr_rks(ni,
                mol, grids, xc_code, dms, relativity, hermi,
                max_memory, verbose
            )

    def nr_uks(ni, mol, grids, xc_code, dms, relativity=0, hermi=0,
               max_memory=2000, verbose=None):
        if '__VV10' in xc_code:
            return super(NLNumInt, ni).nr_uks(
                mol, grids, xc_code, dms, relativity, hermi,
                max_memory, verbose
            )
        else:
            return nr_uks(ni,
                mol, grids, xc_code, dms, relativity, hermi,
                max_memory, verbose
            )

    def rsh_and_hybrid_coeff(self, xc_code, spin=0):
        return 0, 0, 0

    def eval_xc_cider(self, xc_code, mol, rho_data, grid, density, spin=0,
                      relativity=0, deriv=1, omega=None,
                      verbose=None):
        """
        Args:
            mol (gto.Mole) should be assigned a few additional attributes:
                mlfunc (MLFunctional): The nonlocal functional object.
                auxmol (gto.Mole): auxiliary molecule containing the density basis.
                ao_to_aux(np.array): Matrix to convert atomic orbital basis to auxiliary
                    basis, shape (naux, nao, nao)
            rho_data (array (6, N)): The density, gradient, laplacian, and tau
            grid (Grids): The molecular grid
            density: density in auxiliary space
        """
        N = grid.weights.shape[0]
        logging.debug('XCCODE {} {}'.format(xc_code, spin))
        has_base_xc = (xc_code is not None) and (xc_code != '')
        if has_base_xc:
            exc0, vxc0, _, _ = eval_xc(xc_code, rho_data, spin, relativity,
                                       deriv, omega, verbose)

        if spin == 0:
            exc, vxc, _, _ = _eval_xc_0(mol,
                                      (rho_data / 2, rho_data / 2), grid,
                                      (density / 2, density / 2), spin=0,
                                      xf=self.xf, cider=self.cider)
            vxc = [vxc[0][:,1], 0.5 * vxc[1][:,2] + 0.25 * vxc[1][:,1],\
                   vxc[2][:,1], vxc[3][:,1], vxc[4][:,:,1], vxc[5][1,:,:]]
        else:
            exc, vxc, _, _ = _eval_xc_0(mol,
                                        (rho_data[0], rho_data[1]),
                                        grid, (density[0], density[1]),
                                        spin=1, xf=self.xf, cider=self.cider)
        if has_base_xc:
            exc += exc0
            if vxc0[0] is not None:
                vxc[0][:] += vxc0[0]
            if vxc0[1] is not None:
                vxc[1][:] += vxc0[1]
            if vxc0[2] is not None:
                vxc[2][:] += vxc0[2]
            if vxc0[3] is not None:
                vxc[3][:] += vxc0[3]
        return exc, vxc, None, None 


def _eval_xc_0(mol, rho_data, grid, density, spin=1, xf=None, cider=False):

    CF = 0.3 * (6 * np.pi**2)**(2.0/3)

    chkpt = time.monotonic()

    auxmol = mol.auxmol
    naux = auxmol.nao_nr()
    ao_to_aux = mol.ao_to_aux
    N = grid.weights.shape[0]

    desc = [0,0]
    raw_desc = [0,0]
    raw_desc_r2 = [0,0]
    ovlps = [0,0]
    contracted_desc = [0,0]
    F = [0, 0]
    dF = [0, 0]
    dEddesc = [0, 0]

    rhou = rho_data[0][0] + 1e-20
    g2u = np.einsum('ir,ir->r', rho_data[0][1:4], rho_data[0][1:4])
    tu = rho_data[0][5] + 1e-20
    rhod = rho_data[1][0] + 1e-20
    g2d = np.einsum('ir,ir->r', rho_data[1][1:4], rho_data[1][1:4])
    td = rho_data[1][5] + 1e-20
    ntup = (rhou, rhod)
    gtup = (g2u, g2d)
    ttup = (tu, td)
    rhot = rhou + rhod
    g2o = np.einsum('ir,ir->r', rho_data[0][1:4], rho_data[1][1:4])

    vtot = [np.zeros((N,2)), np.zeros((N,3)), np.zeros((N,2)), np.zeros((N,2))]
    vmol = [None, None]
    if xf is None:
        xf = 0.0
    #0.1 / np.sqrt(4 * np.pi)

    gg_kwargs = {
        'a0': 1.0,
        'fac_mul': 0.03125,
        'amin': 0.0625
    }
    
    if cider:
        if spin == 0:
            proj, ovlp, dproj = project_density_cider(grid, 2 * density[0], auxmol, 
                                                      2 * rho_data[0], gg_kwargs)
            v_npa, dedaux = v_nonlocal(2 * rho_data[0], grid,
                                       xf * rhou**(4./3),
                                       2 * density[0], mol.auxmol, proj,
                                       dproj, ovlp, l=0, l_add=0,
                                       **gg_kwargs)
            proj = proj[0]
            dproj = dproj[0]
            exc = 2 * xf * (proj/2) * rhou**(4./3)
            vtot[0][:,0] += xf * (proj/2) * 4./3 * rhou**(1./3)
            vtot[0][:,1] += xf * (proj/2) * 4./3 * rhod**(1./3)
            vmol[0] = np.einsum('a,aij->ij', dedaux, mol.ao_to_aux)
            vmol[1] = vmol[0]

            v_nst = v_basis_transform(rho_data[0]*2, v_npa)
            vtot[0] += v_nst[0][:,None]
            vtot[1][:,0] += 2 * v_nst[1]
            vtot[1][:,2] += 2 * v_nst[1]
            vtot[2] += v_nst[2][:,None]
            vtot[3] += v_nst[3][:,None]
        else:
            exc = 0
            for spin in range(2):
                proj, ovlp, dproj = project_density_cider(grid, 2 * density[spin], auxmol, 
                                                          2 * rho_data[spin], gg_kwargs)
                v_npa, dedaux = v_nonlocal(2 * rho_data[spin], grid,
                                           2 * xf * rhou**(4./3),
                                           2 * density[spin], mol.auxmol, proj,
                                           dproj, ovlp, l=0, l_add=0,
                                           **gg_kwargs)
                proj = proj[0]
                dproj = dproj[0]
                exc += xf * proj * rho_data[spin][0]**(4./3)
                vtot[0][:,spin] += xf * proj * 4./3 * rho_data[spin][0]**(1./3)
                vmol[spin] = np.einsum('a,aij->ij', dedaux, mol.ao_to_aux)

                v_nst = v_basis_transform(rho_data[0]*2, v_npa)
                vtot[0][:,spin] += v_nst[0]
                vtot[1][:,2*spin] += 2 * v_nst[1]
                vtot[2][:,spin] += v_nst[2]
                vtot[3][:,spin] += v_nst[3]
    else:
        if spin == 0:
            proj, ovlp = project_density_simple(grid.coords, density[0], auxmol)
            exc = 2 * xf * proj * rhou**(4./3)
            vtot[0][:,0] += xf * proj * 4./3 * rhou**(1./3)
            vtot[0][:,1] += xf * proj * 4./3 * rhod**(1./3)
            vmol[0] = np.dot(xf * rhou**(4./3) * grid.weights, ovlp)
            vmol[0] = np.einsum('a,aij->ij', vmol[0], mol.ao_to_aux)
            vmol[1] = vmol[0]
        else:
            proj, ovlp = project_density_simple(grid.coords, density[0], auxmol)
            exc = xf * proj * rhou**(4./3)
            vtot[0][:,0] += xf * proj * 4./3 * rhou**(1./3)
            vmol[0] = np.dot(xf * rhou**(4./3) * grid.weights, ovlp)
            vmol[0] = np.einsum('a,aij->ij', vmol[0], mol.ao_to_aux)
            proj, ovlp = project_density_simple(grid.coords, density[1], auxmol)
            exc += xf * proj * rhod**(4./3)
            vtot[0][:,1] += xf * proj * 4./3 * rhod**(1./3)
            vmol[1] = np.dot(xf * rhod**(4./3) * grid.weights, ovlp)
            vmol[1] = np.einsum('a,aij->ij', vmol[1], mol.ao_to_aux)

    v_aniso = np.zeros((3, N))
    v_grad = np.array([v_aniso, v_aniso]).transpose(1,2,0)

    return exc / (rhot + 1e-20), (vtot[0], vtot[1], vtot[2], \
           vtot[3], v_grad, np.array(vmol)), None, None



##########################
# Setup helper functions #
##########################


def setup_aux(mol):
    nao = mol.nao_nr()
    auxmol = df.make_auxmol(mol, 'def2-universal-jkfit')
    #auxmol = df.make_auxmol(mol, auxbasis)
    naux = auxmol.nao_nr()
    # shape (naux, naux), symmetric
    aug_J = auxmol.intor('int2c2e')
    # shape (nao, nao, naux)
    aux_e2 = df.incore.aux_e2(mol, auxmol)
    aux_e2 = aux_e2.reshape((-1, aux_e2.shape[-1])).T
    c_and_lower = cho_factor(aug_J)
    ao_to_aux = cho_solve(c_and_lower, aux_e2)
    ao_to_aux = ao_to_aux.reshape(naux, nao, nao)

    return auxmol, ao_to_aux


def setup_rks_calc(mol, xc='SCAN', xf=0.0, cider=False, **kwargs):
    rks = dft.RKS(mol)
    rks.xc = xc
    rks._numint = NLNumInt(xf, cider)
    rks.grids.level = 3
    rks.grids.build()
    return rks

def setup_uks_calc(mol, xc='SCAN', xf=0.0, cider=False, **kwargs):
    uks = dft.UKS(mol)
    uks.xc = xc
    uks._numint = NLNumInt(xf, cider)
    uks.grids.level = 3
    uks.grids.build()
    return uks

# xc example:
# set corr_model=None, xc='0.75*GGA_X_PBE + GGA_C_PBE',
# and xmix=0.25 to approximate PBE0




def check_dm_uks(mf,s,i,j):
    dm0 = mf.make_rdm1()
    dm1 = dm0.copy()
    delta = np.maximum(dm0[s,i,j], 1) * 1e-8
    dm1[s,i,j] += delta / 2
    dm1[s,j,i] += delta / 2
    v0 = mf.get_veff(dm=dm0)
    e0 = v0.exc
    v0 -= v0.vj
    v1 = mf.get_veff(dm=dm1)
    e1 = v1.exc
    v1 -= v1.vj
    err = (e1-e0)/(delta) - v0[s,i,j]
    print('ERR', s,i,j, err)
    return err

def check_dm_rks(mf,i,j):
    dm0 = mf.make_rdm1()
    dm1 = dm0.copy()
    delta = np.maximum(dm0[i,j], 1) * 1e-8
    dm1[i,j] += delta / 2
    dm1[j,i] += delta / 2
    v0 = mf.get_veff(dm=dm0)
    e0 = v0.exc
    v0 -= v0.vj
    v1 = mf.get_veff(dm=dm1)
    e1 = v1.exc
    v1 -= v1.vj
    err = (e1-e0)/(delta) - v0[i,j]
    print('ERR', i,j, err)
    return err

if __name__ == '__main__':
    from pyscf import gto
    mol1 = gto.M(atom='H 0 0 -0.7; H 0 0 0.7', basis='def2-qzvppd', unit='bohr')
    mol2 = gto.M(atom='H 0 0 -0.7; F 0 0 1.0', basis='def2-qzvppd', unit='bohr')
    mol3 = gto.M(atom='F 0 0 -1.33; F 0 0 1.33', basis='def2-qzvppd', unit='bohr')
    mol4 = gto.M(atom='He', basis='def2-qzvppd', unit='bohr')

    #for mol in [mol1, mol2, mol3]:
    for mol in [mol1]:
        calc = setup_rks_calc(mol, xc='SCAN', xf=0.1,
                              cider=False)
        calc.kernel()
        calc = setup_rks_calc(mol, xc='SCAN', xf=0.1,
                              cider=True)
        calc.kernel()
        for i in range(5):
            check_dm_rks(calc, i, i)

        calc = setup_uks_calc(mol, xc='SCAN', xf=0.1,
                              cider=True)
        calc.kernel()
        for i in range(5):
            check_dm_uks(calc, 0, i, i)
            check_dm_uks(calc, 1, i, i)
