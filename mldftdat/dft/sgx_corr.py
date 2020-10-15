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
from pyscf.sgx.sgx import _make_opt
from pyscf.dft.numint import eval_ao, eval_rho
from mldftdat.models.map_c6 import VSXCContribs
from pyscf import __config__
import pyscf.dft.numint as pyscf_numint

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

    return sgx_fit(mf, auxbasis=auxbasis, with_df=with_df)


def _eval_corr_uks(corr_model, rho_data, F):
    N = rho_data.shape[1]
    rhou = rho_data[0][0]
    g2u = np.einsum('ir,ir->r', rho_data[0][1:4], rho_data[0][1:4])
    tu = rho_data[0][5]
    rhod = rho_data[1][0]
    g2d = np.einsum('ir,ir->r', rho_data[1][1:4], rho_data[1][1:4])
    td = rho_data[1][5]
    ntup = (rhou, rhod)
    gtup = (g2u, g2d)
    ttup = (tu, td)
    rhot = rhou + rhod
    g2o = np.einsum('ir,ir->r', rho_data[0][1:4], rho_data[1][1:4])

    vtot = [np.zeros((N,2)), np.zeros((N,3)), np.zeros((N,2)),
            np.zeros((N,2)), np.zeros((N,2))]
        
    exc, vxc = corr_model.xefc(rhou, rhod, g2u, g2o, g2d,
                               tu, td, F[0], F[1],
                               include_baseline=False,
                               include_aug_sl=False,
                               include_aug_nl=True)

    vtot[0][:,:] += vxc[0]
    vtot[0][:,0] += vxc[3][:,0] * -4 * F[0] / (3 * rhou)
    vtot[0][:,1] += vxc[3][:,1] * -4 * F[1] / (3 * rhod)
    vtot[1][:,:] += vxc[1]
    vtot[3][:,:] += vxc[2]
    vtot[4][:,0] += vxc[3][:,0] / (LDA_FACTOR * rhou**(4.0/3))
    vtot[4][:,1] += vxc[3][:,1] / (LDA_FACTOR * rhod**(4.0/3))

    return exc / (rhot + 1e-20), vtot, None, None

def _eval_corr_rks(corr_model, rho_data, F):
    rho_data = np.stack([rho_data, rho_data])
    F = np.stack([F, F])
    exc, vxc = _eval_corr_uks(corr_model, rho_data, F)[:2]
    vxc = [vxc[0][:,0], 0.5 * vxc[1][:,0] + 0.25 * vxc[1][:,1],\
           vxc[2][:,0], vxc[3][:,0], vxc[4][:,0]]
    return exc, vxc, None, None

from pyscf.dft.numint import _rks_gga_wv0, _uks_gga_wv0

def _contract_corr_rks(vmat, mol, vxc, weight, ao, rho, mask):

    ngrid = weight.size
    shls_slice = (0, mol.nbas)
    ao_loc = mol.nao_loc_nr()
    aow = np.ndarray(ao[0].shape, order='F')
    vrho, vsigma, vlap, vtau = vxc[:4]
    den = rho[0]*weight
    excsum += np.dot(den, exc)

    wv = _rks_gga_wv0(rho, vxc, weight)
    aow = _scal_ao(ao[:4], wv)
    vmat += _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)

    wv = (0.5 * 0.5 * weight * vtau).reshape(-1,1)
    vmat += _dot_ao_ao(mol, ao[1], wv*ao[1], mask, shls_slice, ao_loc)
    vmat += _dot_ao_ao(mol, ao[2], wv*ao[2], mask, shls_slice, ao_loc)
    vmat += _dot_ao_ao(mol, ao[3], wv*ao[3], mask, shls_slice, ao_loc)

    vmat = vmat + vmat.T

    return excsum, vmat

def _contract_corr_uks(vmat, mol, vxc, weight, ao, rho, mask):

    ngrid = weight.size
    shls_slice = (0, mol.nbas)
    ao_loc = mol.nao_loc_nr()
    aow = np.ndarray(ao[0].shape, order='F')
    rho_a = rho[0]
    rho_b = rho[1]
    vrho, vsigma, vlpal, vtau = vxc[:4]
    den = rho_a[0]*weight
    excsum += np.dot(den, exc)
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
    vmat[1] += _dot_ao_ao(mol, ao[1], wv*ao[1], mask, shls_slice, ao_loc)
    vmat[1] += _dot_ao_ao(mol, ao[2], wv*ao[2], mask, shls_slice, ao_loc)
    vmat[1] += _dot_ao_ao(mol, ao[3], wv*ao[3], mask, shls_slice, ao_loc)

    vmat[0] = vmat[0] + vmat[0].T
    vmat[1] = vmat[1] + vmat[1].T

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
    if nset == 1:
        contract_corr = _contract_corr_rks
        eval_corr = _eval_corr_rks
    elif nset == 2:
        contract_corr = _contract_corr_uks
        eval_corr = _eval_corr_uks
    else:
        raise ValueError('Can only call sgx correlation model with nset=1,2')
    FXtmp = numpy.zeros(ngrids)
    Ec = 0
    tnuc = 0, 0
    for i0, i1 in lib.prange(0, ngrids, blksize):
        non0 = non0tab[ip0//BLKSIZE:]
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

        if with_j:
            rhog = numpy.einsum('xgu,gu->xg', fg, ao)
        else:
            rhog = None
        rhogs = numpy.einsum('xgu,gu->g', fg, ao)
        ex = numpy.zeros(rhogs.shape)
        FX = numpy.zeros(rhogs.shape)
        ao_data = eval_ao(mol, coords, deriv=1, non0tab=non0)
        # should make mask for rho_data in the future.
        rho_data = eval_rho(mol, ao_data, dm, non0tab=non0, xctype='MGGA')
        if rho_data.ndim == 2:
            rho_data = np.stack([rho_data, rho_data], axis=0)

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
            for i in range(nset):
                vk[i] += lib.einsum('gu,gv->uv', ao, gv[i])
                ex += lib.einsum('gu,gu->g', fg/wt, gv[i]/wt)
            FX = ex / (LDA_FACTOR * rhogs**(4.0/3))
            FXtmp[i0:i1] = FX
            # vctmp = (vrho, vsigma, vlapl, vtau, vxdens)
            ec, vctmp = eval_corr(sgx.corr_model, rho_data, FX)
            Ec += numpy.dot(ec * rhogs, grids.weights[i0:i1])
            contract_corr(vc, mol, vctmp[:-1], weights,
                          ao_data, rho_data, non0)
            for i in range(nset):
                vc[i] += lib.einsum('gu,gv->uv', ao, gv[i] * vx)

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
    logger.timer(mol, "vj and vk", *t0)

    sgx.current_fx = FXtmp
    sgx.current_ec = Ec
    sgx.current_vc = vc.reshape(dm_shape)
    return vj.reshape(dm_shape), vk.reshape(dm_shape)


class SGXCorr(SGX):

    def __init__(self, mol, auxbasis=None):
        super(SGXCorr, self).__init__(mol, auxbasis)
        self.grids_level_i = 1
        self.grids_level_f = 2

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
                 dss, dos, dx, dm, da, vv10_coeff = None):
        super(HFCNumInt, self).__init__()
        from mldftdat.models import map_c6
        self.corr_model = map_c6.VSXCContribs(
                                css, cos, cx, cm, ca,
                                dss, dos, dx, dm, da)

        if vv10_coeff is None:
            self.vv10 = False
        else:
            self.vv10 = True
            self.vv10_b, self.vv10_c = vv10_coeff

    def nr_rks(self, mol, grids, xc_code, dms, relativity=0, hermi=0,
               max_memory=2000, verbose=None):
        nelec, excsum, vmat = super(HFCNumInt, self).nr_rks(
                                mol, grids, xc_code, dms,
                                relativity, hermi,
                                max_memory, verbose)
        return nelec, excsum + self.sgx.current_ec, vmat + self.sgx.current_vc

    def nr_uks(self, mol, grids, xc_code, dms, relativity=0, hermi=0,
               max_memory=2000, verbose=None):
        nelec, excsum, vmat = super(HFCNumInt, self).nr_uks(
                                mol, grids, xc_code, dms,
                                relativity, hermi,
                                max_memory, verbose)
        return nelec, excsum + sgx.current_ec, vmat + sgx.current_vc

    def eval_xc(self, xc_code, rho, spin=0, relativity=0, deriv=1, omega=None,
                verbose=None):
        rho_data = rho
        N = rho_data.shape[1]
        rhou = rho_data[0][0]
        g2u = np.einsum('ir,ir->r', rho_data[0][1:4], rho_data[0][1:4])
        tu = rho_data[0][5]
        rhod = rho_data[1][0]
        g2d = np.einsum('ir,ir->r', rho_data[1][1:4], rho_data[1][1:4])
        td = rho_data[1][5]
        ntup = (rhou, rhod)
        gtup = (g2u, g2d)
        ttup = (tu, td)
        rhot = rhou + rhod
        g2o = np.einsum('ir,ir->r', rho_data[0][1:4], rho_data[1][1:4])

        vtot = [np.zeros((N,2)), np.zeros((N,3)), np.zeros((N,2)),
                np.zeros((N,2))]
            
        exc, vxc = self.corr_model.xefc(rhou, rhod, g2u, g2o, g2d,
                                   tu, td, None, None,
                                   include_baseline=True,
                                   include_aug_sl=True,
                                   include_aug_nl=False)

        vtot[0][:,:] += vxc[0]
        vtot[1][:,:] += vxc[1]
        vtot[3][:,:] += vxc[2]

        return exc / (rhot + 1e-20), vtot, None, None


DEFAULT_CSS = [0.23959239, 0.22204145, 0.20909579, 0.20122313]
DEFAULT_COS = [0.23959239, 0.22204145, 0.20909579, 0.20122313]
DEFAULT_CX = [0.14666512, 0.12489311, 0.10218985, 0.07728061]
DEFAULT_CM = [-0.02259418, -0.00977586,  0.00506951,  0.02281027]
DEFAULT_CA = [ 0.60325765,  0.19312293, -0.19796497, -0.57942448]
DEFAULT_DSS = [-0.00176003, -0.07177579,  0.10962605,\
               0.00168191, -0.00542834, 0.00951482]
DEFAULT_DOS = [-0.00176003, -0.07177579,  0.10962605,\
               0.00168191, -0.00542834, 0.00951482]
DEFAULT_DX = [0.00248014,  0.22934724, -0.30121907,\
              -0.00304773,  0.00757416,  -0.00774782]
DEFAULT_DM = [-0.00164101,  0.13828657,  0.1658308,\
              -0.00233602, -0.0014921,  0.00958348]
DEFAULT_DA = [5.63601128e-02,  1.09165704e-03,  1.45389821e-02,\
              -2.59986804e-05,  3.03803288e-04, -6.17802303e-04]

def setup_rks_calc(mol, css=DEFAULT_CSS, cos=DEFAULT_COS,
                   cx=DEFAULT_CX, cm=DEFAULT_CM, ca=DEFAULT_CA,
                   dss=DEFAULT_DSS, dos=DEFAULT_DOS, dx=DEFAULT_DX,
                   dm=DEFAULT_DM, da=DEFAULT_DA,
                   vv10_coeff = None):
    rks = dft.RKS(mol)
    rks.xc = None
    rks._numint = HFCNumInt(css, cos, cx, cm, ca,
                           dss, dos, dx, dm, da,
                           vv10_coeff)
    return sgx_fit_corr(rks)

def setup_uks_calc(mol, css=DEFAULT_CSS, cos=DEFAULT_COS,
                   cx=DEFAULT_CX, cm=DEFAULT_CM, ca=DEFAULT_CA,
                   dss=DEFAULT_DSS, dos=DEFAULT_DOS, dx=DEFAULT_DX,
                   dm=DEFAULT_DM, da=DEFAULT_DA,
                   vv10_coeff = None):
    uks = dft.UKS(mol)
    uks.xc = None
    uks._numint = HFCNumInt(css, cos, cx, cm, ca,
                           dss, dos, dx, dm, da,
                           vv10_coeff)
    return sgx_fit_corr(uks)
