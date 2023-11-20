#
# The following module conatains modifed code from the PySCF package.
# This code is subject to the Apache 2.0 License per the terms
# of the PySCF license:
#
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

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

###################################################################
# Modified versions of PySCF numint helper functions that account #
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


###################################################################
# Modified versions of PySCF KS potential helper functions that   #
# include nonlocal CIDER contributions.                           #
###################################################################

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
        self.mlfunc_x.corr_model = corr_model
        self.mlfunc_x.xmix = xmix
        if mlfunc_x.desc_version == 'c':
            mlfunc_x.get_x_helper_full = get_x_helper_full_c
            mlfunc_x.functional_derivative_loop = functional_derivative_loop
        elif mlfunc_x.desc_version == 'a':
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
                                      (rho_data / 2, rho_data / 2), grid,
                                      (density, density), spin=0)
            vxc = [vxc[0][:,1], 0.5 * vxc[1][:,2] + 0.25 * vxc[1][:,1],\
                   vxc[2][:,1], vxc[3][:,1], vxc[4][:,:,1], vxc[5][1,:,:]]
        else:
            exc, vxc, _, _ = _eval_xc_0(self.mlfunc_x, mol,
                                        (rho_data[0], rho_data[1]),
                                        grid, (2 * density[0], 2 * density[1]),
                                        spin=1)
        if has_base_xc:
            exc += exc0
            if vxc0[0] is not None:
                vxc[0][:] += vxc0[0]
            if vxc0[1] is not None:
                vxc[1][:] += vxc0[1]
            if len(vxc0) > 2 and vxc0[2] is not None:
                vxc[2][:] += vxc0[2]
            if len(vxc0) > 3 and vxc0[3] is not None:
                vxc[3][:] += vxc0[3]
        return exc, vxc, None, None 


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

    exc = 0

    for spin in range(1 if spin==0 else 2):
        pr2 = 2 * rho_data[spin] * np.linalg.norm(grid.coords, axis=1)**2
        logging.debug('r2 {} {}'.format(spin, np.dot(pr2, grid.weights)))
        rho43 = ntup[spin]**(4.0/3)
        rho13 = ntup[spin]**(1.0/3)
        desc[spin] = np.zeros((N, mlfunc.nfeat))
        raw_desc[spin], ovlps[spin] = mlfunc.get_x_helper_full(
                                                auxmol, 2 * rho_data[spin], grid,
                                                density[spin], ao_to_aux,
                                                return_ovlp=True, a0=mlfunc.a0,
                                                fac_mul=mlfunc.fac_mul,
                                                amin=mlfunc.amin)
        raw_desc_r2[spin] = mlfunc.get_x_helper_full(auxmol, 2 * rho_data[spin], grid,
                                               density[spin], ao_to_aux,
                                               deriv=True, a0=mlfunc.a0,
                                               fac_mul=mlfunc.fac_mul,
                                               amin=mlfunc.amin)
        contracted_desc[spin] = contract_exchange_descriptors(raw_desc[spin])
        contracted_desc[spin] = contracted_desc[spin][mlfunc.desc_order]
        F[spin], dF[spin] = mlfunc.get_F_and_derivative(contracted_desc[spin])
        #F[spin][(ntup[spin]<1e-8)] = 0
        #dF[spin][(ntup[spin]<1e-8)] = 0
        exc += (mlfunc.xmix * 2**(1.0/3) * LDA_FACTOR) * rho43 * F[spin]
        vtot[0][:,spin] += (mlfunc.xmix * 2**(1.0/3) * 4.0 / 3 * LDA_FACTOR) * rho13 * F[spin]
        dEddesc[spin] = (mlfunc.xmix * 2**(4.0/3) * LDA_FACTOR) * rho43.reshape(-1,1) * dF[spin]
    if spin == 0:
        raw_desc[1] = raw_desc[0]
        ovlps[1] = ovlps[0]
        desc[1] = desc[0]
        raw_desc_r2[1] = raw_desc_r2[0]
        contracted_desc[1] = contracted_desc[0]
        F[1] = F[0]
        dF[1] = dF[0]
        spin = 1
        exc += (mlfunc.xmix * 2**(1.0/3) * LDA_FACTOR) * rho43 * F[spin]
        vtot[0][:,spin] += (mlfunc.xmix * 2**(1.0/3) * 4.0 / 3 * LDA_FACTOR) * rho13 * F[spin]
        dEddesc[spin] = (mlfunc.xmix * 2**(4.0/3) * LDA_FACTOR) * rho43.reshape(-1,1) * dF[spin]

    logging.info('TIME TO SETUP DESCRIPTORS AND RUN GP %f', time.monotonic() - chkpt)
    chkpt = time.monotonic()

    if mlfunc.corr_model is not None:
        tot, vxc = mlfunc.corr_model.xefc1(rhou, rhod, g2u, g2o, g2d,
                                           tu, td, F[0], F[1])
        weights = grid.weights
        
        exc += tot
        vtot[0][:,:] += vxc[0]
        vtot[1][:,:] += vxc[1]
        vtot[3][:,:] += vxc[2]
        dEddesc[0] += 2 * vxc[3][:,0,np.newaxis] * dF[0]
        dEddesc[1] += 2 * vxc[3][:,1,np.newaxis] * dF[1]
    else:
        logging.debug('No correlation, exchange-only evaluation')

    v_nst = [None, None]
    v_grad = [None, None]
    vmol = [None, None]

    for spin in range(2):
        v_nst[spin], v_grad[spin], vmol[spin] = \
            mlfunc.functional_derivative_loop(
                mol, mlfunc, dEddesc[spin],
                raw_desc[spin], raw_desc_r2[spin],
                2 * rho_data[spin], density[spin],
                ovlps[spin], grid)

    v_nst = np.stack(v_nst, axis=-1)
    v_grad = np.stack(v_grad, axis=-1)
    vmol = np.stack(vmol, axis=0)

    logging.info('TIME TO CALCULATE NONLOCAL POTENTIAL %f', time.monotonic() - chkpt)
    chkpt = time.monotonic()

    vtot[0] += v_nst[0]
    vtot[1][:,0] += 2 * v_nst[1][:,0]
    vtot[1][:,2] += 2 * v_nst[1][:,1]
    vtot[2] += v_nst[2]
    vtot[3] += v_nst[3]

    return exc / (rhot + 1e-20), (vtot[0], vtot[1], vtot[2], vtot[3], v_grad, vmol), None, None



##########################
# Setup helper functions #
##########################


def setup_aux(mol):
    nao = mol.nao_nr()
    auxmol = df.make_auxmol(mol, 'weigend+etb')
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
