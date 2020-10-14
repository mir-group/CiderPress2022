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
from pyscf.dft.numint import eval_ao, eval_rho
from mldftdat.models.map_c6 import VSXCContribs


def sgx_fit_corr(mf, xc_coeff, auxbasis=None, with_df=None):
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
        with_df = SGXCorr(mf.mol, xc_coeff)
        with_df.max_memory = mf.max_memory
        with_df.stdout = mf.stdout
        with_df.verbose = mf.verbose
        with_df.auxbasis = auxbasis

    return sgx_fit(mf, auxbasis=auxbasis, with_df=with_df)


def _eval_corr(corr_model, mol, rho_data, F):
    import time

    CF = 0.3 * (6 * np.pi**2)**(2.0/3)

    chkpt = time.monotonic()

    density = (np.einsum('npq,pq->n', mol.ao_to_aux, rdm1[0]),\
               np.einsum('npq,pq->n', mol.ao_to_aux, rdm1[1]))
    auxmol = mol.auxmol
    naux = auxmol.nao_nr()
    ao_to_aux = mol.ao_to_aux
    N = grid.weights.shape[0]

    desc = [0,0]
    dEddesc = [0, 0]

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

    exc = 0
        
    tot, vxc = mlfunc.corr_model.xefc(rhou, rhod, g2u, g2o, g2d,
                                      tu, td, F[0], F[1])

    exc += tot
    vtot[0][:,:] += vxc[0]
    vtot[0][:,0] += vxc[3][:,0] * -4 * F[0] / (3 * rhou)
    vtot[0][:,1] += vxc[3][:,1] * -4 * F[1] / (3 * rhod)
    vtot[1][:,:] += vxc[1]
    vtot[3][:,:] += vxc[2]
    vtot[4][:,:] += vxc[3]
    vtot[4][:,0] += vxc[3][:,0] / (LDA_FACTOR * rhou**(4.0/3))
    vtot[4][:,1] += vxc[3][:,1] / (LDA_FACTOR * rhod**(4.0/3))

    return exc / (rhot + 1e-20), vtot, None, None


def get_jkc(sgx, dm, hermi=1, with_j=True, with_k=True,
            direct_scf_tol=1e-13):
    """
    WARNING: Assumes dm.shape=(1,nao,nao) if restricted
    and dm.shape=(2,nao,nao) for unrestricted for correlation
    to be calculated correctly.
    """
    t0 = time.clock(), time.time()
    mol = sgx.mol
    grids = sgx.grids
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
    FXtmp = numpy.zeros(ngrids)
    Ec = 0
    tnuc = 0, 0
    for i0, i1 in lib.prange(0, ngrids, blksize):
        coords = grids.coords[i0:i1]
        ao = mol.eval_gto('GTOval', coords)
        wao = ao * grids.weights[i0:i1,None]

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
        ao_data = eval_ao(mol, coords, deriv=1)
        # should make mask for rho_data in the future.
        rho_data = eval_rho(mol, ao_data, dm, non0tab=None, xctype='MGGA')

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
            # vc = (vrho, vsigma, vlapl, vtau, vxdens)
            ec, vctmp = sgx.eval_corr(rho_data, FX)
            Ec += numpy.dot(ec * rhogs, grids.weights[i0:i1])
            for i in range(nset):
                vc[i] += sgx.contract_corr(vctmp[:-1], coords, weights,...)
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
        self.grids_level_i = 2
        self.grids_level_f = 3

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
