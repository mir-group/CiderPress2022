#!/usr/bin/env python
# Copyright 2018-2019 The PySCF Developers. All Rights Reserved.
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
# Author: Peng Bao <baopeng@iccas.ac.cn>
#         Qiming Sun <osirpt.sun@gmail.com>
# Adapted and edited by Kyle Bystrom <kylebystrom@gmail.com> for use in the
# mldftdat module.
#

from pyscf.sgx.sgx import *
from pyscf.sgx.sgx_jk import *
from pyscf.sgx.sgx_jk import _gen_jk_direct, _gen_batch_nuc
from pyscf.sgx.sgx import _make_opt
from pyscf.dft.numint import eval_ao, eval_rho
from pyscf.dft.libxc import eval_xc
from pyscf.dft.gen_grid import make_mask, BLKSIZE
from mldftdat.models.map_c6 import VSXCContribs
from pyscf import __config__
import pyscf.dft.numint as pyscf_numint
from pyscf.dft.numint import _scale_ao, _dot_ao_ao
from pyscf.dft.rks import prune_small_rho_grids_
from pyscf.scf.hf import RHF 
from pyscf.scf.uhf import UHF
np = numpy

LDA_FACTOR = - 3.0 / 4.0 * (3.0 / np.pi)**(1.0/3)

def get_veff(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
    if mol is None: mol = ks.mol
    if dm is None: dm = ks.make_rdm1()
    t0 = (time.clock(), time.time())

    ground_state = (isinstance(dm, numpy.ndarray) and dm.ndim == 2)

    if ks.grids.coords is None:
        ks.grids.build(with_non0tab=True)
        if ks.small_rho_cutoff > 1e-20 and ground_state:
            ks.grids = prune_small_rho_grids_(ks, mol, dm, ks.grids)
        t0 = logger.timer(ks, 'setting up grids', *t0)

    if ks.nlc != '':
        if ks.nlcgrids.coords is None:
            ks.nlcgrids.build(with_non0tab=True)
            if ks.small_rho_cutoff > 1e-20 and ground_state:
                # Filter grids the first time setup grids
                ks.nlcgrids = prune_small_rho_grids_(ks, mol, dm, ks.nlcgrids)
            t0 = logger.timer(ks, 'setting up nlc grids', *t0)

    ni = ks._numint
    if hermi == 2:  # because rho = 0
        n, exc, vxc = 0, 0, 0
    else:
        max_memory = ks.max_memory - lib.current_memory()[0]
        n, exc, vxc = ni.nr_rks(mol, ks.grids, ks.xc, dm, max_memory=max_memory)
        if ks.nlc != '':
            assert('VV10' in ks.nlc.upper())
            _, enlc, vnlc = ni.nr_rks(mol, ks.nlcgrids, ks.xc+'__'+ks.nlc, dm,
                                      max_memory=max_memory)
            exc += enlc
            vxc += vnlc
        logger.debug(ks, 'nelec by numeric integration = %s', n)
        t0 = logger.timer(ks, 'vxc', *t0)

    vj, vk = ks.get_jk(mol, dm, hermi)
    vxc += vj - vk * .5
    # vk array must be tagged with these attributes,
    # vc_contrib and ec_contrib
    vxc += vk.vc_contrib
    exc += vk.ec_contrib

    if ground_state:
        exc -= numpy.einsum('ij,ji', dm, vk).real * .5 * .5

    if ground_state:
        ecoul = numpy.einsum('ij,ji', dm, vj).real * .5
    else:
        ecoul = None

    vxc = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=vk)
    return vxc


def get_uveff(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
    '''Coulomb + XC functional for UKS.  See pyscf/dft/rks.py
    :func:`get_veff` fore more details.
    '''
    if mol is None: mol = ks.mol
    if dm is None: dm = ks.make_rdm1()
    if not isinstance(dm, numpy.ndarray):
        dm = numpy.asarray(dm)
    if dm.ndim == 2:  # RHF DM
        dm = numpy.asarray((dm*.5,dm*.5))
    ground_state = (dm.ndim == 3 and dm.shape[0] == 2)

    t0 = (time.clock(), time.time())

    if ks.grids.coords is None:
        ks.grids.build(with_non0tab=True)
        if ks.small_rho_cutoff > 1e-20 and ground_state:
            ks.grids = prune_small_rho_grids_(ks, mol, dm[0]+dm[1], ks.grids)
        t0 = logger.timer(ks, 'setting up grids', *t0)
    if ks.nlc != '':
        if ks.nlcgrids.coords is None:
            ks.nlcgrids.build(with_non0tab=True)
            if ks.small_rho_cutoff > 1e-20 and ground_state:
                ks.nlcgrids = prune_small_rho_grids_(ks, mol, dm[0]+dm[1], ks.nlcgrids)
            t0 = logger.timer(ks, 'setting up nlc grids', *t0)

    ni = ks._numint
    if hermi == 2:  # because rho = 0
        n, exc, vxc = (0,0), 0, 0
    else:
        max_memory = ks.max_memory - lib.current_memory()[0]
        n, exc, vxc = ni.nr_uks(mol, ks.grids, ks.xc, dm, max_memory=max_memory)
        if ks.nlc != '':
            assert('VV10' in ks.nlc.upper())
            _, enlc, vnlc = ni.nr_rks(mol, ks.nlcgrids, ks.xc+'__'+ks.nlc, dm[0]+dm[1],
                                      max_memory=max_memory)
            exc += enlc
            vxc += vnlc
        logger.debug(ks, 'nelec by numeric integration = %s', n)
        t0 = logger.timer(ks, 'vxc', *t0)

    #enabling range-separated hybrids
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(ks.xc, spin=mol.spin)

    vj, vk = ks.get_jk(mol, dm, hermi)
    vj = vj[0] + vj[1]
    vxc += vj - vk
    # vk array must be tagged with these attributes,
    # vc_contrib and ec_contrib
    vxc += vk.vc_contrib
    exc += vk.ec_contrib

    if ground_state:
        exc -=(numpy.einsum('ij,ji', dm[0], vk[0]).real +
               numpy.einsum('ij,ji', dm[1], vk[1]).real) * .5

    if ground_state:
        ecoul = numpy.einsum('ij,ji', dm[0]+dm[1], vj).real * .5
    else:
        ecoul = None

    vxc = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=vk)
    return vxc


def get_gridss_with_non0tab(mol, level=1, gthrd=1e-10):
    Ktime = (time.clock(), time.time())
    grids = dft.gen_grid.Grids(mol)
    grids.level = level
    grids.build(with_non0tab=True)

    ngrids = grids.weights.size
    mask = []
    for p0, p1 in lib.prange(0, ngrids, 10000):
        ao_v = mol.eval_gto('GTOval', grids.coords[p0:p1])
        ao_v *= grids.weights[p0:p1,None]
        wao_v0 = ao_v
        mask.append(numpy.any(wao_v0>gthrd, axis=1) |
                    numpy.any(wao_v0<-gthrd, axis=1))

    mask = numpy.hstack(mask)
    grids.coords = grids.coords[mask]
    grids.weights = grids.weights[mask]
    logger.debug(mol, 'threshold for grids screening %g', gthrd)
    logger.debug(mol, 'number of grids %d', grids.weights.size)
    logger.timer_debug1(mol, "Xg screening", *Ktime)
    return grids

def sgx_fit_corr(mf, auxbasis=None, with_df=None):
    # needs to:
    # 1. Wrap in typical SGX but with get_jkc function
    #    instead of the normal get_jk function
    # 2. Find a way to pass the correlation energy and
    #    vc by attaching it to sgx and then returning
    #    it when nr_uks is called.
    # 3. I.e. attach the sgx to the numint, then
    #    when numint is called, return the current
    #    Ec and Vc attached to sgx
    from pyscf import scf
    from pyscf import df
    from pyscf.soscf import newton_ah
    assert(isinstance(mf, scf.hf.SCF))

    if with_df is None:
        with_df = SGXCorr(mf.mol)
        with_df.max_memory = mf.max_memory
        with_df.stdout = mf.stdout
        with_df.verbose = mf.verbose
        with_df.auxbasis = auxbasis

    mf._numint.sgx = with_df
    with_df.corr_model = mf._numint.corr_model

    new_mf = sgx_fit(mf, auxbasis=auxbasis, with_df=with_df)

    cls = new_mf.__class__
    if isinstance(mf, RHF):
        cls.get_veff = get_veff
    elif isinstance(mf, UHF):
        cls.get_veff = get_uveff
    else:
        raise ValueError('SGXCorr requires RHF or UHF type input')

    new_mf = cls(mf, with_df, auxbasis)

    return new_mf


def _eval_corr_uks(corr_model, rho_data, F):
    N = rho_data.shape[-1]
    rhou = rho_data[0][0]
    g2u = np.einsum('ir,ir->r', rho_data[0][1:4], rho_data[0][1:4])
    tu = rho_data[0][5]
    rhod = rho_data[1][0]
    g2d = np.einsum('ir,ir->r', rho_data[1][1:4], rho_data[1][1:4])
    td = rho_data[1][5]
    rhot = rhou + rhod
    g2o = np.einsum('ir,ir->r', rho_data[0][1:4], rho_data[1][1:4])

    #vtot = [np.zeros((N,2)), np.zeros((N,3)), np.zeros((N,2)),
    #        np.zeros((N,2)), np.zeros((N,2))]
    exc, vxc = corr_model.xefc2(rhou, rhod, g2u, g2o, g2d,
                                tu, td, F[0], F[1])

    return exc / (rhot + 1e-16), [vxc[0], vxc[1], np.zeros((N,2)), vxc[2], vxc[3]]

def _eval_corr_rks(corr_model, rho_data, F):
    rho_data = np.stack([rho_data/2, rho_data/2], axis=0)
    F = np.stack([F/2, F/2], axis=0)
    exc, vxc = _eval_corr_uks(corr_model, rho_data, F)[:2]
    vxc = [vxc[0][:,0], 0.5 * vxc[1][:,0] + 0.25 * vxc[1][:,1],\
           vxc[2][:,0], vxc[3][:,0], vxc[4][:,0]]
    return exc, vxc

from pyscf.dft.numint import _rks_gga_wv0, _uks_gga_wv0

def _contract_corr_rks(vmat, mol, exc, vxc, weight, ao, rho, mask):

    ngrid = weight.size
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    aow = np.ndarray(ao[0].shape, order='F')
    vrho, vsigma, vlap, vtau = vxc[:4]
    den = rho[0]*weight
    excsum = np.dot(den, exc)

    wv = _rks_gga_wv0(rho, vxc, weight)
    aow = _scale_ao(ao[:4], wv, out=aow)
    vmat += _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)

    wv = (0.5 * 0.5 * weight * vtau).reshape(-1,1)
    vmat += _dot_ao_ao(mol, ao[1], wv*ao[1], mask, shls_slice, ao_loc)
    vmat += _dot_ao_ao(mol, ao[2], wv*ao[2], mask, shls_slice, ao_loc)
    vmat += _dot_ao_ao(mol, ao[3], wv*ao[3], mask, shls_slice, ao_loc)

    return excsum, vmat

def _contract_corr_uks(vmat, mol, exc, vxc, weight, ao, rho, mask):

    ngrid = weight.size
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    aow = np.ndarray(ao[0].shape, order='F')
    rho_a = rho[0]
    rho_b = rho[1]
    vrho, vsigma, vlpal, vtau = vxc[:4]
    den = rho_a[0]*weight
    excsum = np.dot(den, exc)
    den = rho_b[0]*weight
    excsum += np.dot(den, exc)
    
    wva, wvb = _uks_gga_wv0((rho_a,rho_b), vxc, weight)
    #:aow = np.einsum('npi,np->pi', ao[:4], wva, out=aow)
    aow = _scale_ao(ao[:4], wva, out=aow)
    vmat[0] += _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
    #:aow = np.einsum('npi,np->pi', ao[:4], wvb, out=aow)
    aow = _scale_ao(ao[:4], wvb, out=aow)
    vmat[1] += _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)

    wv = (.25 * weight * vtau[:,0]).reshape(-1,1)
    vmat[0] += _dot_ao_ao(mol, ao[1], wv*ao[1], mask, shls_slice, ao_loc)
    vmat[0] += _dot_ao_ao(mol, ao[2], wv*ao[2], mask, shls_slice, ao_loc)
    vmat[0] += _dot_ao_ao(mol, ao[3], wv*ao[3], mask, shls_slice, ao_loc)
    wv = (.25 * weight * vtau[:,1]).reshape(-1,1)
    vmat[1] += _dot_ao_ao(mol, ao[1], wv*ao[1], mask, shls_slice, ao_loc)
    vmat[1] += _dot_ao_ao(mol, ao[2], wv*ao[2], mask, shls_slice, ao_loc)
    vmat[1] += _dot_ao_ao(mol, ao[3], wv*ao[3], mask, shls_slice, ao_loc)

    return excsum, vmat


def get_jkc(sgx, dm, hermi=1, with_j=True, with_k=True,
            direct_scf_tol=1e-13):
    """
    WARNING: Assumes dm.shape=(1,nao,nao) if restricted
    and dm.shape=(2,nao,nao) for unrestricted for correlation
    to be calculated correctly.
    """
    t0 = time.clock(), time.time()
    mol = sgx.mol
    nao = mol.nao_nr()
    grids = sgx.grids
    non0tab = grids.non0tab
    if non0tab is None:
        raise ValueError('Grids object must have non0tab!')
    gthrd = sgx.grids_thrd

    dms = numpy.asarray(dm)
    dm_shape = dms.shape
    nao = dm_shape[-1]
    dms = dms.reshape(-1,nao,nao)
    nset = dms.shape[0]

    if sgx.debug:
        batch_nuc = _gen_batch_nuc(mol)
    else:
        batch_jk = _gen_jk_direct(mol, 's2', with_j, with_k, direct_scf_tol,
                                  sgx._opt)

    sn = numpy.zeros((nao,nao))
    ngrids = grids.coords.shape[0]
    max_memory = sgx.max_memory - lib.current_memory()[0]
    sblk = sgx.blockdim
    blksize = min(ngrids, max(4, int(min(sblk, max_memory*1e6/8/nao**2))))
    
    for i0, i1 in lib.prange(0, ngrids, blksize):
        coords = grids.coords[i0:i1]
        ao = mol.eval_gto('GTOval', coords)
        wao = ao * grids.weights[i0:i1,None]
        sn += lib.dot(ao.T, wao)

    ovlp = mol.intor_symmetric('int1e_ovlp')
    proj = scipy.linalg.solve(sn, ovlp)
    proj_dm = lib.einsum('ki,xij->xkj', proj, dms)
    
    #proj_dm = dms.copy()

    t1 = logger.timer_debug1(mol, "sgX initialziation", *t0)
    vj = numpy.zeros_like(dms)
    vk = numpy.zeros_like(dms)
    vc = numpy.zeros_like(dms)
    vc2 = numpy.zeros_like(dms)
    if nset == 1:
        contract_corr = _contract_corr_rks
        eval_corr = _eval_corr_rks
    elif nset == 2:
        contract_corr = _contract_corr_uks
        eval_corr = _eval_corr_uks
    else:
        raise ValueError('Can only call sgx correlation model with nset=1,2')
    Ec = 0
    tnuc = 0, 0
    fxtot = 0
    NELEC = 0
    for i0, i1 in lib.prange(0, ngrids, blksize):
        non0 = non0tab[i0//BLKSIZE:]
        coords = grids.coords[i0:i1]
        ao = mol.eval_gto('GTOval', coords)
        wao = ao * grids.weights[i0:i1,None]
        weights = grids.weights[i0:i1]

        fg = lib.einsum('gi,xij->xgj', wao, proj_dm)
        mask = numpy.zeros(i1-i0, dtype=bool)
        for i in range(nset):
            mask |= numpy.any(fg[i]>gthrd, axis=1)
            mask |= numpy.any(fg[i]<-gthrd, axis=1)
        if not numpy.all(mask):
            ao = ao[mask]
            fg = fg[:,mask]
            coords = coords[mask]
            weights = weights[mask]

        if with_j:
            rhog = numpy.einsum('xgu,gu->xg', fg, ao)
            #NELEC += np.sum(rhog)
        else:
            rhog = None
        ex = numpy.zeros(rhog.shape[-1])
        ao_data = eval_ao(mol, coords, deriv=2)#, non0tab=non0)
        # should make mask for rho_data in the future.
        if nset == 1:
            #rho_data = eval_rho(mol, ao_data, proj_dm[0], non0tab=non0, xctype='MGGA')
            rho_data = eval_rho(mol, ao_data, dms[0], xctype='MGGA')
            NELEC += np.dot(weights, rho_data[0])
        else:
            #rho_data_0 = eval_rho(mol, ao_data, dms[0], non0tab=non0, xctype='MGGA')
            rho_data_0 = eval_rho(mol, ao_data, dms[0], xctype='MGGA')
            #rho_data_1 = eval_rho(mol, ao_data, dms[1], non0tab=non0, xctype='MGGA')
            rho_data_1 = eval_rho(mol, ao_data, dms[1], xctype='MGGA')
            rho_data = np.stack([rho_data_0, rho_data_1], axis=0)

        if sgx.debug:
            tnuc = tnuc[0] - time.clock(), tnuc[1] - time.time()
            gbn = batch_nuc(mol, coords)
            tnuc = tnuc[0] + time.clock(), tnuc[1] + time.time()
            if with_j:
                jpart = numpy.einsum('guv,xg->xuv', gbn, rhog)
            if with_k:
                gv = lib.einsum('gtv,xgt->xgv', gbn, fg)
            gbn = None
        else:
            tnuc = tnuc[0] - time.clock(), tnuc[1] - time.time()
            if with_j:
                rhog = rhog.copy()
            jpart, gv = batch_jk(mol, coords, rhog, fg.copy())
            tnuc = tnuc[0] + time.clock(), tnuc[1] + time.time()

        if with_j:
            vj += jpart
        if with_k:
            if nset == 2:
                FX = [0,0]
            for i in range(nset):
                vk[i] += lib.einsum('gu,gv->uv', ao, gv[i])
                ex = lib.einsum('gu,gu->g', fg[i]/weights[:,None], gv[i]/weights[:,None]) / 4 * nset
                if nset == 1:
                    FX = -ex# / (LDA_FACTOR * rho_data[0]**(4.0/3) - 1e-20)
                else:
                    FX[i] = -ex# / (LDA_FACTOR * rho_data[i][0]**(4.0/3) * 2**(1.0/3) - 1e-20)
                # vctmp = (vrho, vsigma, vlapl, vtau, vxdens)
                fxtot += np.dot(ex, weights)
            ec, vctmp = eval_corr(sgx.corr_model, rho_data, FX)
            if nset == 1:
                Ec += numpy.dot(ec, rho_data[0] * weights)
            else:
                Ec += numpy.dot(ec, (rho_data[0][0] + rho_data[1][0]) * weights)
            contract_corr(vc, mol, ec, vctmp[:-1], weights,
                          ao_data, rho_data, None)
            if nset == 1:
                vc2[0] -= lib.einsum('gu,gv->uv', ao, gv[0] * vctmp[-1][:,None]) / 4
            else:
                for i in range(nset):
                    vc2[i] -= lib.einsum('gu,gv->uv', ao, gv[i] * vctmp[-1][:,i,None]) / 2

        jpart = gv = None

    t2 = logger.timer_debug1(mol, "sgX J/K builder", *t1)
    tdot = t2[0] - t1[0] - tnuc[0] , t2[1] - t1[1] - tnuc[1]
    logger.debug1(sgx, '(CPU, wall) time for integrals (%.2f, %.2f); '
                  'for tensor contraction (%.2f, %.2f)',
                  tnuc[0], tnuc[1], tdot[0], tdot[1])

    for i in range(nset):
        lib.hermi_triu(vj[i], inplace=True)
    if with_k and hermi == 1:
        vk = (vk + vk.transpose(0,2,1))*.5
        vc = (vc + vc.transpose(0,2,1))
        vc += (vc2 + vc2.transpose(0,2,1))
    logger.timer(mol, "vj and vk", *t0)

    vk = vk.reshape(dm_shape)
    vk = lib.tag_array(vk, vc_contrib=vc.reshape(dm_shape), ec_contrib=Ec)

    return vj.reshape(dm_shape), vk


class SGXCorr(SGX):

    def __init__(self, mol, auxbasis=None):
        super(SGXCorr, self).__init__(mol, auxbasis)
        self.grids_level_i = 3
        self.grids_level_f = 3

    def build(self, level=None):
        if level is None:
            level = self.grids_level_f

        self.grids = get_gridss_with_non0tab(self.mol, level, self.grids_thrd)
        self._opt = _make_opt(self.mol)

        # TODO no rsh currently

        return self

    def get_jk(self, dm, hermi=1, with_j=True, with_k=True,
               direct_scf_tol=getattr(__config__, 'scf_hf_SCF_direct_scf_tol', 1e-13),
               omega=None):
        # omega not used
        if with_j and self.dfj:
            vj = df_jk.get_j(self, dm, hermi, direct_scf_tol)
            if with_k:
                vk = get_jkc(self, dm, hermi, False, with_k, direct_scf_tol)[1]
            else:
                vk = None
        else:
            vj, vk = get_jkc(self, dm, hermi, with_j, with_k, direct_scf_tol)
        return vj, vk


class HFCNumInt(pyscf_numint.NumInt):

    def __init__(self, corr_model, vv10_coeff=None):
        super(HFCNumInt, self).__init__()
        self.corr_model = corr_model

        if vv10_coeff is None:
            self.vv10 = False
        else:
            self.vv10 = True
            self.vv10_b, self.vv10_c = vv10_coeff

    def nr_rks(self, mol, grids, xc_code, dms, relativity=0, hermi=0,
               max_memory=2000, verbose=None):
        nelec, excsum, vmat = super(HFCNumInt, self).nr_uks(
                                mol, grids, xc_code, np.stack([dms/2, dms/2]),
                                relativity, hermi,
                                max_memory, verbose)
        return nelec, excsum, vmat[0]

    def nr_uks(self, mol, grids, xc_code, dms, relativity=0, hermi=0,
               max_memory=2000, verbose=None):
        nelec, excsum, vmat = super(HFCNumInt, self).nr_uks(
                                mol, grids, xc_code, dms,
                                relativity, hermi,
                                max_memory, verbose)
        return nelec, excsum, vmat

    def eval_xc(self, xc_code, rho, spin=0, relativity=0, deriv=1, omega=None,
                verbose=None):
        if deriv > 1:
            raise NotImplementedError('Only 1st derivative supported')
        N = rho[0].shape[-1]
        e = np.zeros(N)
        vxc = [np.zeros((N,2)), np.zeros((N,3)), None,
               np.zeros((N,2))]
        return e, vxc, None, None


def setup_rks_calc(mol, corr_model, vv10_coeff=None, **kwargs):
    rks = dft.RKS(mol)
    rks.xc = 'SCAN'
    rks._numint = HFCNumInt(corr_model, vv10_coeff=vv10_coeff)
    rks = sgx_fit_corr(rks)
    return rks

def setup_uks_calc(mol, corr_model, vv10_coeff=None, **kwargs):
    uks = dft.UKS(mol)
    uks.xc = 'SCAN'
    uks._numint = HFCNumInt(corr_model, vv10_coeff=vv10_coeff)
    uks = sgx_fit_corr(uks)
    return uks
