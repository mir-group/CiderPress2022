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
from mldftdat.dft.qe_interface import TestPyFort

import numpy as np
import logging
import time



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

    def __init__(self, mlfunc_x, corr_model=None, xmix=1.0):
        """
        Args:
            mlfunc_x (MLFunctional): Exchange model
            xmix (float): Fraction of CIDER exchange
            corr_model (class, None): Optional class to add hyper-GGA
                correlation model. Must contain a function called xefc1,
                which takes the following arguments:
                    (rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb,
                     tau_a, tau_b, fx_a, fx_b)
                and returns the following
                    rho*e_xc, (vrho, vsigma, vtau, vexx)
                NOTE the convention difference from libxc, the XC energy
                *density* must be returned, not the XC energy per particle.
                fx_a and fx_b are the predicted exchange enhancement factors
                from the CIDER model.
        """
        super(NLNumInt, self).__init__()
        self.mlfunc_x = mlfunc_x
        self.mlfunc_x.caller = TestPyFort(mlfunc_x)
        self.mlfunc_x.xmix = xmix
        if mlfunc_x.desc_version == 'a':
            mlfunc_x.get_x_helper_full = get_x_helper_full_a
            mlfunc_x.functional_derivative_loop = functional_derivative_loop
        elif mlfunc_x.desc_version == 'c':
            mlfunc_x.get_x_helper_full = get_x_helper_full_a
            mlfunc_x.functional_derivative_loop = functional_derivative_loop
        else:
            raise ValueError('Invalid version for descriptors')

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
            exc, vxc, _, _ = _eval_xc_0(self.mlfunc_x, mol,
                                      rho_data, grid,
                                      density, spin=0)
        else:
            exc, vxc, _, _ = _eval_xc_0(self.mlfunc_x, mol,
                                        np.array([rho_data[0], rho_data[1]]),
                                        grid, np.array([density[0], density[1]]),
                                        spin=1)
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

def v_nl(rho_data, grid, dedb, density, auxmol,
         g, gr2, ovlp, l=0, a0=1.0, fac_mul=0.03125,
         amin=0.0625, l_add=0):
    #print(l, l_add, a0, fac_mul, amin)
    # g should have shape (2l+1, N)
    N = grid.weights.shape[0]
    lc = get_dft_input2(rho_data)[:3]

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


def _eval_xc_0(mlfunc, mol, rho_data, grid, density, spin=1):

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

    vtot = [np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)]
    v_grad = np.zeros((3,N))
    exc = 0

    if spin == 0:
        rho = rho_data[0]
        raw_desc, ovlps = mlfunc.get_x_helper_full(
                                          auxmol, rho_data, grid,
                                          density, ao_to_aux,
                                          return_ovlp=True, a0=1.0,
                                          fac_mul=0.03125,
                                          amin=0.0625)
        raw_desc_r2 = mlfunc.get_x_helper_full(
                                         auxmol, rho_data, grid,
                                         density, ao_to_aux,
                                         deriv=True, a0=1.0,
                                         fac_mul=0.03125,
                                         amin=0.0625)
        ex = np.zeros(N)
        v1x = np.zeros((N,1))
        v2x = np.zeros((N,1))
        v3x = np.zeros((N,1))
        vfeat = np.zeros((N,11,1))
        mlfunc.caller.get_xc_fortran(
            rho_data[0].reshape(N,1),
            rho_data[1:4].reshape(3,N,1),
            rho_data[5].reshape(N,1),
            raw_desc[6:].T.reshape(N,11,1),
            ex, v1x, v2x, v3x, vfeat,
            mlfunc.xmix, v_grad.reshape(3,N,1),
            no_swap=True
        )
        #print(np.sum(v1x[:,0]*rho*grid.weights), np.sum(v2x[:,0]*rho*grid.weights), np.sum(v3x[:,0]*rho*grid.weights))
        vfeat = vfeat.reshape(N,11).T
        #print(np.sum(v_grad*grid.weights))
        v_npa = 0
        v_aux = 0
        for i, j, f in [(6,0,1.0),(15,3,0.5**(4./3)),(16,4,2.0**(4./3))]:
            vtmp, dedaux = v_nl(rho_data, grid, vfeat[i-6:i-5], density, auxmol,
                                raw_desc[i], raw_desc_r2[i].reshape(1,-1), ovlps[j],
                                l=0, a0=1.0*f, fac_mul=0.03125*f, amin=0.0625*f)
            v_npa += vtmp
            v_aux += dedaux
        vtmp, dedaux = v_nl(rho_data, grid, vfeat[1:4], density, auxmol,
                            raw_desc[7:10], raw_desc_r2[7:10], ovlps[1],
                            l=1, a0=1.0, fac_mul=0.03125, amin=0.0625)
        v_npa += vtmp
        v_aux += dedaux
        vtmp, dedaux = v_nl(rho_data, grid, vfeat[4:9], density, auxmol,
                            raw_desc[10:15], raw_desc_r2[10:15], ovlps[2],
                            l=2, a0=1.0, fac_mul=0.03125, amin=0.0625)
        v_npa += vtmp
        v_aux += dedaux

        vmol = np.einsum('a,aij->ij', v_aux, mol.ao_to_aux)
        v_nst = v_basis_transform(rho_data, v_npa)

        vtot[0] += v1x[:,0] + v_nst[0]
        vtot[1] += v2x[:,0] + v_nst[1]
        vtot[3] += v3x[:,0] + v_nst[3]

        return ex / (rho_data[0] + 1e-20), (vtot[0], vtot[1], vtot[2], vtot[3], v_grad, vmol), None, None
    else:
        raise NotImplementedError


##########################
# Setup helper functions #
##########################


def setup_aux(mol):
    nao = mol.nao_nr()
    auxmol = df.make_auxmol(mol, 'weigend+etb')
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


def setup_rks_calc(mol, mlfunc_x, corr_model=None,
                   grid_level=3,
                   xc=None, xmix=1.0, **kwargs):
    """
    Initialize a PySCF RKS calculation from pyscf.gto.Mole object mol
    and ML exchange model mlfunc_x.
    Args:
        mol (pyscf.gto.Mole): molecular structure object
        mlfunc_x (MLFunctional): ML exchange functional
        corr_model: See NLNumInt.__init__
        grid_level (int): PySCF integration grid level for XC
        xc (str): Semi-local exchange-correlation functional to add to CIDER,
        xmix (float): fraction of CIDER exchange

    Example:
        For a PBE0-CIDER calculation (i.e. 75% PBE exchange, 25% CIDER
        exchange, 100% PBE correlation), use
        xc = '0.75 * GGA_X_PBE + GGA_C_PBE'
        xmix = 0.25
    """
    rks = dft.RKS(mol)
    rks.xc = xc
    rks._numint = NLNumInt(mlfunc_x, corr_model, xmix)
    rks.grids.level = grid_level
    rks.grids.build()
    return rks

def setup_uks_calc(mol, mlfunc_x, corr_model=None,
                   grid_level=3,
                   xc=None, xmix=1.0, **kwargs):
    """
    Initialize a PySCF UKS calculation, see setup_rks_calc for docs.
    """
    uks = dft.UKS(mol)
    uks.xc = xc
    uks._numint = NLNumInt(mlfunc_x, corr_model, xmix)
    uks.grids.level = grid_level
    uks.grids.build()
    return uks

# xc example:
# set corr_model=None, xc='0.75*GGA_X_PBE + GGA_C_PBE',
# and xmix=0.25 to approximate PBE0
