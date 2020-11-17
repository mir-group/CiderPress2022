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
    # 1. Wrap in typical SGX but with get get_jkc function
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
                                tu, td, F[0], F[1],
                                include_baseline=False,
                                include_aug_sl=True,
                                include_aug_nl=True)

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
            jpart, gv = batch_jk(mol, coords, rhog, fg)
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
        self.grids_level_i = 1
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

    def __init__(self, css, cos, cx, cm, ca,
                 dss, dos, dx, dm, da, vv10_coeff = None,
                 fterm_scale=2.0):
        print ("FTERM SCALE", fterm_scale)
        super(HFCNumInt, self).__init__()
        from mldftdat.models import map_c6
        self.corr_model = map_c6.VSXCContribs(
                                css, cos, cx, cm, ca,
                                dss, dos, dx, dm, da,
                                fterm_scale=fterm_scale)

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
        rho_data = rho
        N = rho_data[0].shape[-1]
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

        vtot = [np.zeros((N,2)), np.zeros((N,3)), np.zeros((N,2)),
                np.zeros((N,2))]
        F = np.zeros(N) 
        exc, vxc = self.corr_model.xefc2(rhou, rhod, g2u, g2o, g2d,
                                         tu, td, F, F,
                                         include_baseline=True,
                                         include_aug_sl=False,
                                         include_aug_nl=False)
        
        vtot[0][:,:] += vxc[0]
        vtot[1][:,:] += vxc[1]
        vtot[3][:,:] += vxc[2]
        for i in range(4):
            vtot[i][rhou<1e-9,0] = 0
            vtot[i][rhod<1e-9,1] = 0
        
        #exc *= 0
        return exc / (rhot + 1e-20), vtot, None, None


class HFCNumInt2(HFCNumInt):

    def __init__(self, cx, c0, c1, dx, d0, d1,
                 vv10_coeff = None, fterm_scale=2.0):
        print ("FTERM SCALE", fterm_scale)
        super(HFCNumInt, self).__init__()
        from mldftdat.models import map_c8
        self.corr_model = map_c8.VSXCContribs(
                                cx, c0, c1, dx, d0, d1,
                                fterm_scale=fterm_scale)

        if vv10_coeff is None:
            self.vv10 = False
        else:
            self.vv10 = True
            self.vv10_b, self.vv10_c = vv10_coeff

    def eval_xc(self, xc_code, rho, spin=0, relativity=0, deriv=1, omega=None,
                verbose=None):
        print("NEW ITERATION")
        e, vxc, _, _ = eval_xc(',MGGA_C_REVSCAN', rho, spin=1)
        e *= 0
        for i in [0, 1, 3]:
            vxc[i][:] *= 0
        """
        thr = 1e-8
        rhou, rhod = rho[0][0], rho[1][0]
        vxc[0][rhou<thr,0] = 0
        vxc[1][rhou<thr,0] = 0
        vxc[2][rhou<thr,0] = 0
        vxc[3][rhou<thr,0] = 0
        vxc[0][rhod<thr,1] = 0
        vxc[1][rhod<thr,2] = 0
        vxc[2][rhod<thr,1] = 0
        vxc[3][rhod<thr,1] = 0
        vxc[1][np.sqrt(rhou*rhod)<thr,1] = 0
        """
        return e, vxc, None, None

DEFAULT_COS = [-0.02481797,  0.00303413,  0.00054502,  0.00054913]
DEFAULT_CX = [-0.03483633, -0.00522109, -0.00299816, -0.0022187 ]
DEFAULT_CA = [-0.60154365, -0.06004444, -0.04293853, -0.03146755]
DEFAULT_DOS = [-0.00041445, -0.01881556,  0.03186469,  0.00100642, -0.00333434,
          0.00472453]
DEFAULT_DX = [ 0.00094936,  0.09238444, -0.21472824, -0.00118991,  0.0023009 ,
         -0.00096118]
DEFAULT_DA = [ 7.92248007e-03, -2.11963128e-03,  2.72918353e-02,  4.57295468e-05,
         -1.00450001e-05, -3.47808331e-04]
DEFAULT_CM = None
DEFAULT_DM = None
DEFAULT_CSS = None
DEFAULT_DSS = None
"""
DEFAULT_COS = [-3.33117906e-02,  3.42204832e-04, -5.12710933e-04,  7.23656600e-05]
DEFAULT_CX = [-0.04660942, -0.00958024, -0.00461958, -0.00240451]
DEFAULT_CA = [-0.60633633, -0.08913412, -0.06914276, -0.04692656]
DEFAULT_DOS = [-0.00036989, -0.01618721,  0.04825873,  0.00083393, -0.00350275,
          0.00700518]
DEFAULT_DX = [ 0.00225735,  0.1098632 , -0.28697265, -0.00159427,  0.00498835,
         -0.00549777]
DEFAULT_DA = [ 1.50175274e-02, -2.45387696e-03,  3.01268545e-02,  3.90737069e-05,
          2.57511213e-05, -3.90349785e-04]
"""
DEFAULT_COS = [-2.39744286e-02,  1.75450797e-03, -2.42249328e-04,  6.97484137e-05]
DEFAULT_CX = [-0.03658478, -0.00711547, -0.00381782, -0.00216612]
DEFAULT_CA = [-0.58796507, -0.07989303, -0.05758388, -0.03791681]
DEFAULT_DOS = [-0.00019688, -0.02070159,  0.05712587,  0.00092075, -0.00387164,
          0.00739471]
DEFAULT_DX = [ 0.00214469,  0.118391  , -0.30686047, -0.00179011,  0.00568496,
         -0.00611958]
DEFAULT_DA = [ 9.86513001e-03, -2.74320674e-03,  3.13055397e-02,  3.85063943e-05,
          3.93533240e-05, -4.02402091e-04]

def setup_rks_calc(mol, css=DEFAULT_CSS, cos=DEFAULT_COS,
                   cx=DEFAULT_CX, cm=DEFAULT_CM, ca=DEFAULT_CA,
                   dss=DEFAULT_DSS, dos=DEFAULT_DOS, dx=DEFAULT_DX,
                   dm=DEFAULT_DM, da=DEFAULT_DA,
                   vv10_coeff = None, fterm_scale=2.0):
    rks = dft.RKS(mol)
    rks.xc = 'SCAN'
    rks._numint = HFCNumInt(css, cos, cx, cm, ca,
                           dss, dos, dx, dm, da,
                           vv10_coeff=vv10_coeff,
                           fterm_scale=fterm_scale)
    rks = sgx_fit_corr(rks)
    rks.with_df.debug = False
    return rks

def setup_uks_calc(mol, css=DEFAULT_CSS, cos=DEFAULT_COS,
                   cx=DEFAULT_CX, cm=DEFAULT_CM, ca=DEFAULT_CA,
                   dss=DEFAULT_DSS, dos=DEFAULT_DOS, dx=DEFAULT_DX,
                   dm=DEFAULT_DM, da=DEFAULT_DA,
                   vv10_coeff = None, fterm_scale=2.0):
    uks = dft.UKS(mol)
    uks.xc = 'SCAN'
    uks._numint = HFCNumInt(css, cos, cx, cm, ca,
                           dss, dos, dx, dm, da,
                           vv10_coeff=vv10_coeff,
                           fterm_scale=fterm_scale)
    uks = sgx_fit_corr(uks)
    uks.with_df.debug = True
    return uks


C0 = [-0.04414256, -0.2250942 , -0.41182839, -0.31122479]
CX = [-0.92800667,  1.01815189, -2.85159579, -2.07850633]
D0 = [   0.82964304,  -54.23500274,   38.17884089,   57.62829873,   57.58797719,
    2.19364839, -112.13336295,  -61.13620897,  -40.81538857,   61.01520428,
    7.6658853 ,  -23.83122386,   -0.37110221,    3.47509178]
DX = [ -0.15470902,  -0.27707704,   2.37939525,   4.48691893,   3.11176978,
 -21.71645175, -14.90105348,  -7.52872736,  44.16545793,  16.19272888,
   7.25302051, -25.04934218,  -0.30376073,   0.16515621]
C1 =  [-0.2746799 ,  0.41379366, -0.23417838, -0.17982577]
D1 =  [-4.38135465, 14.38630401, -4.25170122, -1.86469137,  7.0068854 ,
 -9.35327294, -3.88512236,  3.28150407, -5.95836117, -0.05788019,
  3.58034473, -2.96382182,  1.88698092, -1.45027681]

C0 = [-0.17153411, -0.16331153, -0.29449155, -0.20487423]
CX = [-0.92652918,  0.73056813, -2.81120786, -2.24898139]
D0 = [  -8.5363603 ,  -23.06716595,   84.24841682,   54.63919587,  -19.44483216,
          -67.57367958, -108.45072687,  -38.25011227,  -28.86475648,   81.12211618,
             72.48438959,   36.09737289,   -2.59114045,   -0.29264352]
DX = [ -0.12277572,  -0.69697444,  -1.37163731,   3.37604513,   0.57383642,
         -10.78542408, -11.06255767,   4.53483696,  31.77094369,  12.5202736 ,
           -4.81619   , -23.07463111,  -0.20412443,   0.45549592]
C1 =  [-0.26560992,  0.5511927 , -0.00276721,  0.02014515]
D1 =  [ 0.12477345, 14.40882449, -6.32474782,  0.50078421,  7.75301779,
         -9.92800426, -7.53253637,  4.13179399, -5.83983628, -6.0130549 ,
           4.14011554, -2.4728966 ,  1.52993909, -1.28001379]

C0 = [-0.01354789, -0.11150584, -0.29602951, -0.22300478]
CX = [-0.90544989,  0.90350756, -3.09975025, -2.55448756]
D0 = [  -5.38791594,  -26.32023074,   84.02870901,   58.75312231,   17.20264244,
         -101.97867769, -126.52539177,  -52.52169173,  -29.47858027,   85.37279622,
            82.95178412,   68.05202336,   -4.75852749,    2.76839079]
DX = [-8.92792050e-02, -6.22937829e-01, -1.71282818e-03,  2.64396838e+00,
          6.56868844e+00, -2.14533522e+01, -8.49608251e+00, -1.02059650e+01,
            5.22521199e+01,  1.00051494e+01,  6.30031600e+00, -3.60742685e+01,
             -3.68163309e-01,  5.43989226e-01]
C1 =  [-0.52879875,  0.52855336, -0.04783615, -0.01584011]
D1 =  [-0.89239433,  5.3356237 ,  0.31806325, -2.02511857,  1.84127453,
         -2.00519748, -2.87840953,  0.26392891, -1.97027449, -2.36869895,
          -0.07705574, -1.52942195,  3.83846753, -3.42793328]

C0 = [-0.3354288 ,  0.00823391, -0.04016834, -0.03739124]
D0 = [ 0.87379432, -0.32382218,  1.57381194, -0.36202222]
C1 = [-0.38526515,  0.1543739 ,  0.00309856, -0.0008107 ]
D1 = [ 0.10065954, -0.11763847, -0.23123289,  0.47371275]
DX = [ 0.24444768, -0.10234676,  0.15287771, -0.31298206,  0.66436677,
         -0.61492233,  0.28235148, -0.06262296,  0.05777821,  0.37148324,
          -0.17916824,  0.33431956,  0.0331934 ,  0.57327276,  0.08151482,
           -0.0251695 ,  0.38206185,  0.04661124,  0.50096642]
CX = [-1.08777476, -0.02135043,  0.34136139, -0.19010579,  0.16444242,
          0.59732767, -0.37878256, -0.77262206, -0.3272416 , -0.30295938,
           -0.52918312, -0.36007932, -0.17528194, -0.21398146,  0.00502194]

C0 = [-0.75271089,  0.10959875, -0.03440979, -0.03600507]
D0 = [ 1.20846359,  0.83915766,  2.30938631, -2.58042136]
C1 = [-0.59282679,  0.53068725,  0.13090276,  0.09993011]
D1 = [-0.14510522, -0.64797943, -0.45476377,  1.3484906 ]
DX = [ 0.25249519, -0.09002764,  0.09059361, -0.283228  ,  0.55692011,
         -1.01693739,  0.39243504,  0.3359663 ,  0.15206114,  0.88673759,
          -0.46303391,  0.1175976 , -0.12285532,  0.70791151,  0.35443381,
           -0.09492766,  0.31915651, -0.10619349,  0.57643775]
CX = [-0.97689611,  0.02582968,  0.10913193, -0.11387742,  0.30327337,
          0.92849187, -0.50911983, -1.2058131 , -0.24816478, -0.14631313,
           -1.33539946, -0.7297475 , -0.20415085, -0.37376469,  0.14080384]

def setup_rks_calc2(mol, cx=CX, c0=C0, c1=C1, dx=DX, d0=D0, d1=D1,
                    vv10_coeff=None, fterm_scale=2.0):
    rks = dft.RKS(mol)
    rks.xc = 'SCAN'
    rks._numint = HFCNumInt2(cx, c0, c1, dx, d0, d1,
                             vv10_coeff=vv10_coeff,
                             fterm_scale=fterm_scale)
    rks = sgx_fit_corr(rks)
    rks.with_df.debug = False
    return rks

def setup_uks_calc2(mol, cx=CX, c0=C0, c1=C1, dx=DX, d0=D0, d1=D1,
                    vv10_coeff=None, fterm_scale=2.0):
    uks = dft.UKS(mol)
    uks.xc = 'SCAN'
    uks._numint = HFCNumInt2(cx, c0, c1, dx, d0, d1,
                             vv10_coeff=vv10_coeff,
                             fterm_scale=fterm_scale)
    uks = sgx_fit_corr(uks)
    uks.with_df.debug = True
    return uks
