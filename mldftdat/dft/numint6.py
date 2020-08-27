import pyscf.dft.numint as pyscf_numint
from pyscf.dft.numint import _rks_gga_wv0, _scale_ao, _dot_ao_ao, _format_uks_dm
from pyscf.dft.libxc import eval_xc
from pyscf.dft.gen_grid import Grids
from pyscf import df, dft
import numpy as np
from mldftdat.density import get_x_helper_full, get_x_helper_full2, LDA_FACTOR,\
                             contract_exchange_descriptors,\
                             contract21_deriv, contract21
import scipy.linalg
from scipy.linalg.lapack import dgetrf, dgetri
from scipy.linalg.blas import dgemm, dgemv
from mldftdat.pyscf_utils import get_mgga_data, get_rho_second_deriv
from mldftdat.dft.utils import *
from mldftdat.dft.correlation import eval_custom_corr, nr_rks_vv10

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

def nr_rks(ni, mol, grids, xc_code, dms, relativity = 0, hermi = 0,
           max_memory = 2000, verbose = None):

    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

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
            print('dm shape', dms.shape)
            rho = make_rho(idm, ao, mask, 'MGGA')
            exc, vxc = ni.eval_xc(xc_code, mol, rho, grids, dms,
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

    if ni.vv10:
        if not hasattr(ni, 'nlcgrids'):
            nlcgrids = Grids(mol)
            nlcgrids.level = 1
            nlcgrids.build()
            ni.nlcgrids = nlcgrids
        _, excsum_vv10, vmat_vv10 = nr_rks_vv10(ni, mol, ni.nlcgrids, xc_code, dms, 
                relativity, hermi, max_memory, verbose, b=ni.vv10_b, c=ni.vv10_c)

    for i in range(nset):
        vmat[i] = vmat[i] + vmat[i].T
    if nset == 1:
        nelec = nelec[0]
        excsum = excsum[0]
        vmat = vmat.reshape(nao,nao)
    if ni.vv10:
        return nelec, excsum + excsum_vv10, vmat + vmat_vv10
    else:
        return nelec, excsum, vmat


def nr_uks(ni, mol, grids, xc_code, dms, relativity = 0, hermi = 0,
           max_memory = 2000, verbose = None):

    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    dma, dmb = _format_uks_dm(dms)
    nao = dma.shape[-1]
    make_rhoa, nset = ni._gen_rho_evaluator(mol, dma, hermi)[:2]
    make_rhob       = ni._gen_rho_evaluator(mol, dmb, hermi)[0]

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
            print('dm shape', dma.shape, dmb.shape)
            rho_a = make_rhoa(idm, ao, mask, 'MGGA')
            rho_b = make_rhob(idm, ao, mask, 'MGGA')
            exc, vxc = ni.eval_xc(xc_code, mol, (rho_a, rho_b),
                                  grids, (dma, dmb),
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

            vmat[0,idm] += 0.5 * vmol[0,:,:]
            vmat[1,idm] += 0.5 * vmol[1,:,:]

            rho_a = rho_b = exc = vxc = vrho = vsigma = wva = wvb = None

    if ni.vv10:
        if not hasattr(ni, 'nlcgrids'):
            nlcgrids = Grids(mol)
            nlcgrids.level = 1
            nlcgrids.build()
            ni.nlcgrids = nlcgrids
        _, excsum_vv10, vmat_vv10 = nr_rks_vv10(ni, mol, ni.nlcgrids, xc_code, dms[0] + dms[1],
                relativity, hermi, max_memory, verbose, b=ni.vv10_b, c=ni.vv10_c)
        vmat_vv10 = np.asarray([vmat_vv10, vmat_vv10])

    for i in range(nset):
        vmat[0,i] = vmat[0,i] + vmat[0,i].T
        vmat[1,i] = vmat[1,i] + vmat[1,i].T
    if isinstance(dma, np.ndarray) and dma.ndim == 2:
        vmat = vmat[:,0]
        nelec = nelec.reshape(2)
        excsum = excsum[0]
    if ni.vv10:
        return nelec, excsum + excsum_vv10, vmat + vmat_vv10
    else:
        return nelec, excsum, vmat


class NLNumInt(pyscf_numint.NumInt):

    nr_rks = nr_rks

    nr_uks = nr_uks

    def __init__(self, mlfunc_x, alpha,
                 dss, dos, vv10_coeff = None):
        super(NLNumInt, self).__init__()
        self.mlfunc_x = mlfunc_x
        from mldftdat.models import map_c1
        self.corr_model = map_c1.VSXCContribs(alpha, dss, dos)

        if vv10_coeff is None:
            self.vv10 = False
        else:
            self.vv10 = True
            self.vv10_b, self.vv10_c = vv10_coeff

    def eval_xc(self, xc_code, mol, rho_data, grid, rdm1, spin = 0,
                relativity = 0, deriv = 1, omega = None,
                verbose = None):
        """
        Args:
            mol (gto.Mole) should be assigned a few additional attributes:
                mlfunc (MLFunctional): The nonlocal functional object.
                auxmol (gto.Mole): auxiliary molecule containing the density basis.
                ao_to_aux(np.array): Matrix to convert atomic orbital basis to auxiliary
                    basis, shape (naux, nao, nao)
            rho_data (array (6, N)): The density, gradient, laplacian, and tau
            grid (Grids): The molecular grid
            rdm1: density matrix
        """
        if not (hasattr(mol, 'ao_to_aux') and hasattr(mol, 'auxmol')):
            mol.auxmol, mol.ao_to_aux = setup_aux(mol, self.beta)

        N = grid.weights.shape[0]
        print('XCCODE', xc_code)
        has_base_xc = (xc_code is not None) and (xc_code != '')
        if has_base_xc:
            exc0, vxc0, _, _ = eval_xc(xc_code, rho_data, spin, relativity,
                                       deriv, omega, verbose)

        if spin == 0:
            exc, vxc, _, _ = _eval_xc_0(self.mlfunc_c, mol,
                                      (rho_data / 2, rho_data / 2), grid,
                                      (rdm1, rdm1))
        else:
            exc, vxc, _, _ = _eval_xc_0(self.mlfunc_c, mol,
                                        (rho_data[0], rho_data[1]),
                                        grid, (2 * rdm1[0], 2 * rdm1[1]))
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


def _eval_xc_0(mlfunc, mol, rho_data, grid, rdm1, spin = 0):
    import time

    if spin == 0:
        spin = 1
    else:
        spin = 2

    chkpt = time.monotonic()

    density = (np.einsum('npq,pq->n', mol.ao_to_aux, rdm1[0]),\
               np.einsum('npq,pq->n', mol.ao_to_aux, rdm1[1]))
    auxmol = mol.auxmol
    naux = auxmol.nao_nr()
    ao_to_aux = mol.ao_to_aux
    N = grid.weights.shape[0]

    desc = [0,0]
    ddesc = [0,0]
    raw_desc = [0,0]
    raw_desc_r2 = [0,0]
    ovlps = [0,0]
    contracted_desc = [0,0]

    rhou = rho_data[0][0]
    rhod = rho_data[1][0]
    rhot = rhou + rhod

    co, v_lda_ud = eval_xc(',LDA_C_PW_MOD', (rhou, rhod), spin = 1)[:2]
    cu, v_lda_uu = eval_xc(',LDA_C_PW_MOD', (rhou, 0 * rhod), spin = 1)[:2]
    cd, v_lda_dd = eval_xc(',LDA_C_PW_MOD', (0 * rhou, rhod), spin = 1)[:2]
    cu *= rhou
    cd *= rhod
    co = co * rhot - cu - cd
    v_lda_uu = v_lda_uu[0][:,0]
    v_lda_dd = v_lda_dd[0][:,1]
    v_lda_ud = v_lda_ud[0]
    v_lda_ud[:,0] -= v_lda_uu
    v_lda_ud[:,1] -= v_lda_dd

    exc = 0

    for spin in range(2):
        pr2 = 2 * rho_data[spin] * np.linalg.norm(grid.coords, axis=1)**2
        print('r2', spin, np.dot(pr2, grid.weights))
        rho43 = rho_data[spin][0]**(4.0 / 3)
        desc[spin]  = np.zeros((N, len(mlfunc.desc_list)))
        raw_desc[spin], ovlps[spin] = get_x_helper_full2(
                                                auxmol, 2 * rho_data[spin], grid,
                                                density[spin], ao_to_aux,
                                                return_ovlp = True)
        raw_desc_r2[spin] = get_x_helper_full2(auxmol, 2 * rho_data[spin], grid,
                                               density[spin], ao_to_aux,
                                               integral_name = 'int1e_r2_origj')
        contracted_desc[spin] = contract_exchange_descriptors(raw_desc[spin])
        for i, d in enumerate(mlfunc.desc_list):
            desc[spin][:,i] = d.transform_descriptor(contracted_desc[spin])
        F[spin], dF[spin] = mlfunc.get_F_and_derivative(desc[spin])
        exc += 2**(1.0/3) * LDA_FACTOR * F[spin] * rho43
        dEddesc[spin] = 2**(1.0/3) * LDA_FACTOR * rho43 * dF[spin]
        Pc, dPc = self.corr_model.get_xeff_and_deriv(F[spin], use_cos = False)
        exc += 2**(1.0/3) * LDA_FACTOR * rho43 * Pc
        dEddesc[spin] += 2**(1.0/3) * LDA_FACTOR * rho43 * dPc * dF[spin]

    Qcuu, dQcuu = self.corr_model.get_xeff_and_deriv_ss(F[0])
    Qcdd, dQcdd = self.corr_model.get_xeff_and_deriv_ss(F[1])
    Qcud, dQcud = self.corr_model.get_xeff_and_deriv(
            (F[0] * rhou + F[1] * rhod) / (rhot + 1e-10),
            use_cos = True)
    exc += ldac_uu * Qcuu + ldac_ud * Qcud + ldac_dd * Qcdd
    dEddesc[0] += ldac_uu * dQuu * dF[0]
    dEddesc[0] += ldac_ud * dQud * rhou / (rhot + 1e-10) * dF[0]
    dEddesc[1] += ldac_dd * dQdd * dF[1]
    dEddesc[1] += ldac_ud * dQud * rhod / (rhot + 1e-10) * dF[1]
    # TODO: deriv wrt rhou, rhod above

    print('desc setup and run GP', time.monotonic() - chkpt)
    chkpt = time.monotonic()

    v_nst = [None, None]
    v_grad = [None, None]
    vmol = [None, None]

    for spin in range(2):
        v_nst[spin], v_grad[spin], vmol[spin] = \
            functional_derivative_loop(
                mol, mlfunc, dEddesc[spin],
                contracted_desc[spin],
                raw_desc[spin], raw_desc_r2[spin],
                2 * rho_data[spin], density[spin],
                ovlps[spin], grid)

    v_nst = np.stack(v_nst, axis=-1)
    v_grad = np.stack(v_grad, axis=-1)
    vmol = np.stack(vmol, axis=0)

    print('v_nonlocal', time.monotonic() - chkpt)
    chkpt = time.monotonic()

    g2u = np.einsum('ir,ir->i', rho_data[0][1:4], rho_data[0][1:4])
    g2d = np.einsum('ir,ir->i', rho_data[1][1:4], rho_data[1][1:4])
    en, dcu, dcd, dco, dnu, dnd, dg2u, dg2d, dtu, dtd = \
        self.corr_model.get_en_and_deriv_corr(cu, cd,
            co, rhou + 1e-10, rhod + 1e-10,
            g2u, g2d, rho_data[0][5] + 1e-10, rho_data[1][5] + 1e-10)
    exc += en

    # TODO v_nst[1] should have 3 dimensions on axis 0 not 2
    vtot = list(vref)
    if vtot[1] is None:
        vtot[1] = np.zeros((vtot[0].shape[0],3))
    if vtot[2] is None:
        vtot[2] = np.zeros((vtot[0].shape[0],2))
    if vtot[3] is None:
        vtot[3] = np.zeros((vtot[0].shape[0],2))
    vtot[0][:,0] += v_lda_uu * dcu + v_lda_ud[:,0] * dco
    vtot[0][:,1] += v_lda_dd * dcd + v_lda_ud[:,1] * dco
    vtot[0][:,0] += 2**(1.0/3) * LDA_FACTOR * rhou**(4.0/3) * F[0]
    vtot[0][:,1] += 2**(1.0/3) * LDA_FACTOR * rhod**(4.0/3) * F[1]
    vtot[0][:,0] += ldac_ud * dQud * (F[0] - F[1]) * rhod / (rhot + 1e-10)**2
    vtot[0][:,1] += ldac_ud * dQud * (F[1] - F[0]) * rhou / (rhot + 1e-10)**2
    vtot[0] += v_nst[0]
    vtot[1][:,0] += 2 * v_nst[1][:,0] + dg2u
    vtot[1][:,2] += 2 * v_nst[1][:,1] + dg2d
    vtot[2] += v_nst[2]
    vtot[3] += v_nst[3]
    vtot[3][:,0] += dtu
    vtot[3][:,1] += dtd

    for spin in range(2):
        fsl, dfsl_n, dfsl_g2, dfsl_t = \
            self.corr_model.get_f_and_deriv_ex(rho_data[spin][0] + 1e-10,
                np.einsum('ir,ir->i', rho_data[spin][1:4], rho_data[spin][1:4]),
                rho_data[spin][5] + 1e-10)
        exc += 2**(1.0/3) * LDA_FACTOR * fsl * rho_data[spin][0]**(4.0/3)
        vtot[0][:,spin] += 2**(1.0/3) * LDA_FACTOR * dfsl_n * rho_data[spin][0]**(4.0/3)
        vtot[0][:,spin] += 2**(1.0/3) * 4/3 * LDA_FACTOR * fsl * rho_data[spin][0]**(1.0/3)
        vtot[1][:,2 if spin == 1 else 0] += \
            2**(1.0/3) * LDA_FACTOR * dfsl_g2 * rho_data[spin][0]**(4.0/3)
        vtot[3][:,spin] += 2**(1.0/3) * LDA_FACTOR * dfsl_t * rho_data[spin][0]**(4.0/3)

    return exc / (rhot + 1e-10), (vtot[0], vtot[1], vtot[2], vtot[3], v_grad, vmol), None, None


def setup_aux(mol, beta):
    #auxbasis = df.aug_etb(mol, beta = beta)
    nao = mol.nao_nr()
    auxmol = df.make_auxmol(mol, 'weigend')
    #auxmol = df.make_auxmol(mol, auxbasis)
    naux = auxmol.nao_nr()
    # shape (naux, naux), symmetric
    aug_J = auxmol.intor('int2c2e')
    # shape (nao, nao, naux)
    aux_e2 = df.incore.aux_e2(mol, auxmol)
    #print(aux_e2.shape)
    # shape (naux, nao * nao)
    aux_e2 = aux_e2.reshape((-1, aux_e2.shape[-1])).transpose()
    aux_e2 = np.ascontiguousarray(aux_e2)
    lu, piv, info = dgetrf(aug_J, overwrite_a = True)
    inv_aug_J, info = dgetri(lu, piv, overwrite_lu = True)
    ao_to_aux = dgemm(1, inv_aug_J, aux_e2)
    ao_to_aux = ao_to_aux.reshape(naux, nao, nao)
    return auxmol, ao_to_aux


def setup_rks_calc(mol, mlfunc_x, mlfunc_c, vv10_coeff = None,
                   beta = 1.6, ss_terms = None, os_terms = None):
    rks = dft.RKS(mol)
    rks.xc = None
    rks._numint = NLNumInt(mlfunc_x, mlfunc_c, vv10_coeff,
                           beta, ss_terms, os_terms)
    return rks

def setup_uks_calc(mol, mlfunc_x, mlfunc_c, vv10_coeff = None,
                   beta = 1.6, ss_terms = None, os_terms = None):
    uks = dft.UKS(mol)
    uks.xc = None
    uks._numint = NLNumInt(mlfunc_x, mlfunc_c, vv10_coeff,
                           beta, ss_terms, os_terms)
    return uks
