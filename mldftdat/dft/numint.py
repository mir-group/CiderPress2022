import pyscf.dft.numint as pyscf_numint
from pyscf.dft.numint import _rks_gga_wv0, _scale_ao, _dot_ao_ao
from pyscf import df, dft
import numpy as np
from mldftdat.density import get_x_helper_full, LDA_FACTOR, contract_exchange_descriptors
import scipy.linalg
from scipy.linalg.lapack import dgetrf, dgetri
from scipy.linalg.blas import dgemm, dgemv
from mldftdat.pyscf_utils import get_mgga_data, get_rho_second_deriv
from mldftdat.dft.utils import *

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
            exc, vxc = ni.eval_xc(mol, rho, grids, dms,
                                  0, relativity, 1,
                                  verbose=verbose)[:2]
            vrho, vsigma, vlapl, vtau = vxc[:4]
            den = rho[0] * weight
            nelec[idm] += den.sum()
            excsum[idm] += np.dot(den, exc)

            wv = _rks_gga_wv0(rho, vxc, weight)
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

class NLNumInt(pyscf_numint.NumInt):

    nr_rks = nr_rks

    def eval_xc(self, mol, rho_data, grid, rdm1, spin = 0,
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
        if spin == 0:
            return _eval_xc_0(mol, rho_data, grid, rdm1)
        else:
            uterms = _eval_xc_0(mol, rho_data[0], grid, rdm1[0])
            dterms = _eval_xc_0(mol, rho_data[1], grid, rdm1[1])
            exc  = uterms[0] * rho_data[0][0,:]
            exc += dterms[1] * rho_data[1][1,:]
            vrho = np.zeros(N, 2)
            vsigma = np.zeros(N, 3)
            vlapl = np.zeros(N, 2)
            vtau = np.zeros(N, 2)

            vrho[:,0] = uterms[1][0]
            vrho[:,1] = dterms[1][0]

            vsigma[:,0] = uterms[1][1]
            vsigma[:,2] = dterms[1][1]

            vlapl[:,0] = uterms[1][2]
            vlapl[:,1] = dterms[1][2]

            vtau[:,0] = uterms[1][3]
            vtau[:,1] = dterms[1][3]

            return exc, (vrho, vsigma, vlapl, vtau), None, None
            

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
    v_npa = np.zeros((4, N))
    dgpdp = np.zeros(rho_data.shape[1])
    dgpda = np.zeros(rho_data.shape[1])
    dFddesc = dF * ddesc
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
            #elif d.code == 6:
            #    g = raw_desc[(1,2,3,13,14,15)]
            #    l = 1
            #elif d.code == 12:
            #    l = 2
            #    g = raw_desc[(1,2,3,16,17,18,19,20)]
            else:
                raise NotImplementedError('Cannot take derivative for code %d' % d.code)
            v_npa += v_nonlocal(rho_data, grid, dFddesc[:,i],
                                density,
                                mol.auxmol, g, l = l,
                                mul = d.mul)
    v_npa += v_semilocal(rho_data, F, dgpdp, dgpda)
    v_nst = v_basis_transform(rho_data, v_npa)
    return exc, (v_nst[0], v_nst[1], v_nst[2], v_nst[3]), None, None


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
    rks._numint = NLNumInt()
    return rks

def setup_uks_calc(mol, mlfunc, beta = 1.6):
    mol.build()
    mol.auxmol, mol.ao_to_aux = setup_aux(mol, beta)
    mol.mlfunc = mlfunc
    uks = dft.UKS(mol)
    uks._numint = NLNumInt()
    return uks
