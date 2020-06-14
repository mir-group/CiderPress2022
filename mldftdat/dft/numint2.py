import pyscf.dft.numint as pyscf_numint
from pyscf.dft.numint import _rks_gga_wv0, _scale_ao, _dot_ao_ao, _format_uks_dm
from pyscf.dft.libxc import eval_xc
from pyscf import df, dft
import numpy as np
from mldftdat.density import get_x_helper_full, LDA_FACTOR,\
                             contract_exchange_descriptors,\
                             contract21_deriv, contract21
import scipy.linalg
from scipy.linalg.lapack import dgetrf, dgetri
from scipy.linalg.blas import dgemm, dgemv
from mldftdat.pyscf_utils import get_mgga_data, get_rho_second_deriv
from mldftdat.dft.utils import *

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
    vxca = (vxc[0][:,0], vxc[1][:,0], vxc[2][:,0], vxc[3][:,0], vxc[4][:,:,0])
    vxcb = (vxc[0][:,1], vxc[1][:,2], vxc[2][:,1], vxc[3][:,1], vxc[4][:,:,1])
    wva = _rks_gga_wv0a(rho[0], vxca, weight)
    wvb = _rks_gga_wv0a(rho[1], vxcb, weight)
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
            vrho, vsigma, vlapl, vtau, vgrad = vxc[:5]
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

            rho = exc = vxc = vrho = vsigma = wv = None

    for i in range(nset):
        vmat[i] = vmat[i] + vmat[i].T
    if nset == 1:
        nelec = nelec[0]
        excsum = excsum[0]
        vmat = vmat.reshape(nao,nao)
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
            vrho, vsigma, vlapl, vtau, vgrad = vxc[:5]
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
            rho_a = rho_b = exc = vxc = vrho = vsigma = wva = wvb = None

    for i in range(nset):
        vmat[0,i] = vmat[0,i] + vmat[0,i].T
        vmat[1,i] = vmat[1,i] + vmat[1,i].T
    if isinstance(dma, np.ndarray) and dma.ndim == 2:
        vmat = vmat[:,0]
        nelec = nelec.reshape(2)
        excsum = excsum[0]
    return nelec, excsum, vmat


class NLNumInt(pyscf_numint.NumInt):

    nr_rks = nr_rks

    nr_uks = nr_uks

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
        N = grid.weights.shape[0]
        print('XCCODE', xc_code)
        has_base_xc = (xc_code is not None) and (xc_code != '')
        if has_base_xc:
            exc0, vxc0, _, _ = eval_xc(xc_code, rho_data, spin, relativity, deriv,
                                       omega, verbose)
        if spin == 0:
            exc, vxc, _, _ = _eval_xc_0(mol, rho_data, grid, rdm1)
        else:
            uterms = _eval_xc_0(mol, 2 * rho_data[0], grid, 2 * rdm1[0])
            dterms = _eval_xc_0(mol, 2 * rho_data[1], grid, 2 * rdm1[1])
            exc  = uterms[0] * rho_data[0][0,:]
            exc += dterms[0] * rho_data[1][0,:]
            exc /= (rho_data[0][0,:] + rho_data[1][0,:])
            vrho = np.zeros((N, 2))
            vsigma = np.zeros((N, 3))
            vlapl = np.zeros((N, 2))
            vtau = np.zeros((N, 2))
            vgrad = np.zeros((3, N, 2))

            vrho[:,0] = uterms[1][0]
            vrho[:,1] = dterms[1][0]

            vsigma[:,0] = uterms[1][1]
            vsigma[:,2] = dterms[1][1]

            vlapl[:,0] = uterms[1][2]
            vlapl[:,1] = dterms[1][2]

            vtau[:,0] = uterms[1][3]
            vtau[:,1] = dterms[1][3]

            vgrad[:,:,0] = uterms[1][4]
            vgrad[:,:,1] = dterms[1][4]

            vxc = (vrho, vsigma, vlapl, vtau, vgrad)
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


def _eval_xc_0(mol, rho_data, grid, rdm1):
    density = np.einsum('npq,pq->n', mol.ao_to_aux, rdm1)
    mlfunc = mol.mlfunc
    auxmol = mol.auxmol
    ao_to_aux = mol.ao_to_aux
    N = grid.weights.shape[0]
    desc  = np.zeros((N, len(mlfunc.desc_list)))
    ddesc = np.zeros((N, len(mlfunc.desc_list)))
    ao_data, rho_data = get_mgga_data(mol, grid, rdm1)
    ddrho = get_rho_second_deriv(mol, grid, rdm1, ao_data)
    raw_desc = get_x_helper_full(auxmol, rho_data, ddrho, grid,
                                 density, ao_to_aux)
    contracted_desc = contract_exchange_descriptors(raw_desc)
    for i, d in enumerate(mol.mlfunc.desc_list):
        desc[:,i], ddesc[:,i] = d.transform_descriptor(
                                  contracted_desc, deriv = 1)
    F = mol.mlfunc.get_F(desc)
    # shape (N, ndesc)
    dF = mol.mlfunc.get_derivative(desc)
    exc = LDA_FACTOR * F * rho_data[0]**(1.0/3)
    elda = LDA_FACTOR * rho_data[0]**(4.0/3)
    v_npa = np.zeros((4, N))
    dgpdp = np.zeros(rho_data.shape[1])
    dgpda = np.zeros(rho_data.shape[1])
    dFddesc = dF * ddesc

    sprefac = 2 * (3 * np.pi * np.pi)**(1.0/3)
    n43 = rho_data[0]**(4.0/3)
    svec = rho_data[1:4] / (sprefac * n43 + 1e-9)
    v_aniso = np.zeros((3,N))

    for i, d in enumerate(mlfunc.desc_list):
        if d.code == 0:
            continue
        elif d.code == 1:
            dgpdp += dFddesc[:,i]
        elif d.code == 2:
            dgpda += dFddesc[:,i]
        else:
            if d.code in [4, 15, 16]:
                g = contracted_desc[d.code]
                l = 0
            elif d.code == 5:
                g = raw_desc[13:16]
                l = 1
            elif d.code == 8:
                g = raw_desc[16:21]
                l = 2
            elif d.code == 6:
                g = raw_desc[13:16]
                dfmul = svec
                v_aniso += elda * dFddesc[:,i] * g
                l = -1
            elif d.code == 12:
                l = -2
                g = raw_desc[16:21]
                dfmul = contract21_deriv(svec)
                ddesc_dsvec = contract21(g, svec)
                v_aniso += elda * dFddesc[:,i] * 2 * ddesc_dsvec
            else:
                raise NotImplementedError('Cannot take derivative for code %d' % d.code)

            if d.code in [6, 12]:
                v_npa += v_nonlocal(rho_data, grid, dFddesc[:,i] * dfmul,
                                    density, mol.auxmol, g, l = l,
                                    mul = d.mul)
            else:
                v_npa += v_nonlocal(rho_data, grid, dFddesc[:,i],
                                    density, mol.auxmol, g, l = l,
                                    mul = d.mul)

    v_npa += v_semilocal(rho_data, F, dgpdp, dgpda)
    v_nst = v_basis_transform(rho_data, v_npa)
    v_nst[0] += np.einsum('ap,ap->p', -4.0 * svec / (3 * rho_data[0] + 1e-10), v_aniso)
    v_grad = v_aniso / (sprefac * n43 + 1e-10)
    return exc, (v_nst[0], v_nst[1], v_nst[2], v_nst[3], v_grad), None, None


def setup_aux(mol, beta):
    auxbasis = df.aug_etb(mol, beta = beta)
    nao = mol.nao_nr()
    auxmol = df.make_auxmol(mol, auxbasis)
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


def setup_rks_calc(mol, mlfunc, beta = 1.6):
    mol.build()
    mol.auxmol, mol.ao_to_aux = setup_aux(mol, beta)
    mol.mlfunc = mlfunc
    rks = dft.RKS(mol)
    rks.xc = None
    rks._numint = NLNumInt()
    return rks

def setup_uks_calc(mol, mlfunc, beta = 1.6):
    mol.build()
    mol.auxmol, mol.ao_to_aux = setup_aux(mol, beta)
    mol.mlfunc = mlfunc
    uks = dft.UKS(mol)
    uks.xc = None
    uks._numint = NLNumInt()
    return uks
