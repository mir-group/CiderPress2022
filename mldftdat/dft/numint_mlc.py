import pyscf.dft.numint as pyscf_numint
from pyscf.dft.numint import _rks_gga_wv0, _scale_ao, _dot_ao_ao, _format_uks_dm
from pyscf.dft.libxc import eval_xc
from pyscf.dft.gen_grid import Grids
from pyscf import df, dft
import numpy as np
from mldftdat.density import get_x_helper_full, get_x_helper_full, LDA_FACTOR,\
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

    def __init__(self, mlfunc_x, mlfunc_c, vv10_coeff = None,
                 beta = 1.6, ss_terms = None, os_terms = None):
        super(NLNumInt, self).__init__()
        self.beta = beta
        self.mlfunc_x = mlfunc_x
        self.mlfunc_c = mlfunc_c

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
            ex, vx, _, _ = _eval_x_0(self.mlfunc_x, mol, rho_data, grid, rdm1)
            ec, vc, _, _ = _eval_c_0(self.mlfunc_c, mol,
                                     (rho_data, rho_data), grid,
                                     (rdm1, rdm1))
            exc = ex + ec
            # Convert spin-polarized correlation to restricted spin
            vxc = [vx[0] + vc[0][:,0],\
                   vx[1] + 0.5 * vc[1][:,0] + 0.25 * vc[1][:,1],\
                   vx[2] + vc[2][:,0],\
                   vx[3] + vc[3][:,0],\
                   vx[4] + vc[4][:,:,0],\
                   vx[5] + vc[5][0,:,:]
            ]
        else:
            uterms = _eval_x_0(self.mlfunc_x, mol, 2 * rho_data[0], grid, 2 * rdm1[0])
            dterms = _eval_x_0(self.mlfunc_x, mol, 2 * rho_data[1], grid, 2 * rdm1[1])
            ec, vc, _, _ = _eval_c_0(self.mlfunc_c, mol, (2 * rho_data[0], 2 * rho_data[1]),
                                     grid, (2 * rdm1[0], 2 * rdm1[1]))
            exc  = uterms[0] * rho_data[0][0,:]
            exc += dterms[0] * rho_data[1][0,:]
            exc /= (rho_data[0][0,:] + rho_data[1][0,:] + 1e-20)
            exc += ec
            vrho = np.zeros((N, 2))
            vsigma = np.zeros((N, 3))
            vlapl = np.zeros((N, 2))
            vtau = np.zeros((N, 2))
            vgrad = np.zeros((3, N, 2))
            vmol = np.zeros((2, mol.nao_nr(), mol.nao_nr()))

            vrho[:,0] = uterms[1][0] + vc[0][:,0]
            vrho[:,1] = dterms[1][0] + vc[0][:,1]

            vsigma[:,0] = 2 * uterms[1][1] + vc[1][:,0]
            vsigma[:,1] = vc[1][:,1]
            vsigma[:,2] = 2 * dterms[1][1] + vc[1][:,2]

            vlapl[:,0] = uterms[1][2] + vc[2][:,0]
            vlapl[:,1] = dterms[1][2] + vc[2][:,1]

            vtau[:,0] = uterms[1][3] + vc[3][:,0]
            vtau[:,1] = dterms[1][3] + vc[3][:,1]

            vgrad[:,:,0] = uterms[1][4] + vc[4][:,:,0]
            vgrad[:,:,1] = dterms[1][4] + vc[4][:,:,1]

            vmol[0,:,:] = uterms[1][5] + vc[5][0,:,:]
            vmol[1,:,:] = dterms[1][5] + vc[5][1,:,:]

            vxc = (vrho, vsigma, vlapl, vtau, vgrad, vmol)
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


def _eval_x_0(mlfunc, mol, rho_data, grid, rdm1):
    import time

    chkpt = time.monotonic()

    density = np.einsum('npq,pq->n', mol.ao_to_aux, rdm1)
    auxmol = mol.auxmol
    naux = auxmol.nao_nr()
    ao_to_aux = mol.ao_to_aux
    N = grid.weights.shape[0]
    desc  = np.zeros((N, len(mlfunc.desc_list)))
    ddesc = np.zeros((N, len(mlfunc.desc_list)))
    ao_data, rho_data = get_mgga_data(mol, grid, rdm1)
    ddrho = get_rho_second_deriv(mol, grid, rdm1, ao_data)
    raw_desc, ovlps = get_x_helper_full(auxmol, rho_data, ddrho, grid,
                                 density, ao_to_aux, return_ovlp = True)
    raw_desc_r2 = get_x_helper_full(auxmol, rho_data, ddrho, grid,
                                    density, ao_to_aux,
                                    integral_name = 'int1e_r2_origj')
    contracted_desc = contract_exchange_descriptors(raw_desc)
    for i, d in enumerate(mlfunc.desc_list):
        desc[:,i], ddesc[:,i] = d.transform_descriptor(
                                  contracted_desc, deriv = 1)

    print('desc setup', time.monotonic() - chkpt)
    chkpt = time.monotonic()

    if mlfunc.y_to_f_mul is None:
        F, dF = mlfunc.get_F_and_derivative(desc)
    else:
        F = mlfunc.get_F(desc, s = contracted_desc[1])
    exc = LDA_FACTOR * F * rho_data[0]**(1.0/3)
    elda = LDA_FACTOR * rho_data[0]**(4.0/3)
    v_npa = np.zeros((4, N))
    dgpdp = np.zeros(rho_data.shape[1])
    dgpda = np.zeros(rho_data.shape[1])
    dFddesc = dF * ddesc

    print('run GP', time.monotonic() - chkpt)
    chkpt = time.monotonic()

    sprefac = 2 * (3 * np.pi * np.pi)**(1.0/3)
    n43 = rho_data[0]**(4.0/3)
    svec = rho_data[1:4] / (sprefac * n43 + 1e-20)
    v_aniso = np.zeros((3,N))
    v_aux = np.zeros(naux)

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
                if d.code == 4:
                    ovlp = ovlps[0]
                    gr2 = raw_desc_r2[12:13]
                elif d.code == 15:
                    ovlp = ovlps[3]
                    gr2 = raw_desc_r2[21:22]
                else:
                    ovlp = ovlps[4]
                    gr2 = raw_desc_r2[22:23]
                l = 0
            elif d.code == 5:
                g = raw_desc[13:16]
                gr2 = raw_desc_r2[13:16]
                ovlp = ovlps[1]
                l = 1
            elif d.code == 8:
                g = raw_desc[16:21]
                gr2 = raw_desc_r2[16:21]
                ovlp = ovlps[2]
                l = 2
            elif d.code == 6:
                g = raw_desc[13:16]
                gr2 = raw_desc_r2[13:16]
                ovlp = ovlps[1]
                dfmul = svec
                v_aniso += elda * dFddesc[:,i] * g
                l = -1
            elif d.code == 12:
                l = -2
                g = raw_desc[16:21]
                gr2 = raw_desc_r2[16:21]
                ovlp = ovlps[2]
                dfmul = contract21_deriv(svec)
                ddesc_dsvec = contract21(g, svec)
                v_aniso += elda * dFddesc[:,i] * 2 * ddesc_dsvec
            elif d.code == 13:
                g2 = raw_desc[16:21]
                g2r2 = raw_desc_r2[16:21]
                ovlp2 = ovlps[2]
                g1 = raw_desc[13:16]
                g1r2 = raw_desc_r2[13:16]
                ovlp1 = ovlps[1]
                dfmul = contract21_deriv(svec, g1)
                ddesc_dsvec = contract21(g2, g1)
                ddesc_dg1 = contract21(g2, svec)
                v_aniso += elda * dFddesc[:,i] * ddesc_dsvec
                vtmp1, dedaux1 = v_nonlocal_extra_fast(rho_data, grid, dFddesc[:,i] * ddesc_dg1,
                                         density, mol.auxmol, g1, g1r2, ovlp1, l = -1,
                                         mul = d.mul)
                vtmp2, dedaux2 = v_nonlocal_extra_fast(rho_data, grid, dFddesc[:,i] * dfmul,
                                         density, mol.auxmol, g2, g2r2, ovlp2, l = -2,
                                         mul = d.mul)
                vtmp = vtmp1 + vtmp2
                dedaux = dedaux1 + dedaux2
            else:
                raise NotImplementedError('Cannot take derivative for code %d' % d.code)

            if d.code in [6, 12]:
                vtmp, dedaux = v_nonlocal_extra_fast(rho_data, grid, dFddesc[:,i] * dfmul,
                                         density, mol.auxmol, g, gr2, ovlp, l = l,
                                         mul = d.mul)
            elif d.code == 13:
                pass
            else:
                vtmp, dedaux = v_nonlocal_extra_fast(rho_data, grid, dFddesc[:,i],
                                         density, mol.auxmol, g, gr2, ovlp, l = l,
                                         mul = d.mul)
            v_npa += vtmp
            v_aux += dedaux

    print('v_nonlocal', time.monotonic() - chkpt)
    chkpt = time.monotonic()

    vmol = np.einsum('a,aij->ij', v_aux, mol.ao_to_aux)
    v_npa += v_semilocal(rho_data, F, dgpdp, dgpda)
    v_nst = v_basis_transform(rho_data, v_npa)
    v_nst[0] += np.einsum('ap,ap->p', -4.0 * svec / (3 * rho_data[0] + 1e-20), v_aniso)
    v_grad = v_aniso / (sprefac * n43 + 1e-20)
    return exc, (v_nst[0], v_nst[1], v_nst[2], v_nst[3], v_grad, vmol), None, None


def _eval_c_0(mlfunc, mol, rho_data, grid, rdm1):
    import time

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

    for spin in range(2):
        pr2 = rho_data[spin] * np.linalg.norm(grid.coords, axis=1)**2
        print('r2', spin, np.dot(pr2, grid.weights))
        desc[spin]  = np.zeros((N, len(mlfunc.desc_list)))
        ddesc[spin] = np.zeros((N, len(mlfunc.desc_list)))
        raw_desc[spin], ovlps[spin] = get_x_helper_full(
                                                auxmol, rho_data[spin], grid,
                                                density[spin], ao_to_aux,
                                                return_ovlp = True)
        raw_desc_r2[spin] = get_x_helper_full(auxmol, rho_data[spin], grid,
                                               density[spin], ao_to_aux,
                                               integral_name = 'int1e_r2_origj')
        contracted_desc[spin] = contract_exchange_descriptors(raw_desc[spin])
        for i, d in enumerate(mlfunc.desc_list):
            desc[spin][:,i], ddesc[spin][:,i] = \
                d.transform_descriptor(contracted_desc[spin], deriv = 1)

    print('desc setup', time.monotonic() - chkpt)
    chkpt = time.monotonic()

    # get correlation energy density, vref, and derivative
    # of ec wrt GP descriptors
    # vref is the functional derivate of the energy density
    # wrt the inputs to pyscf eval_xc function. I.e. it should
    # apply to the reference functional.
    ec, vref, dec = mlfunc.get_F_and_derivative(desc,
                        (rho_data[0]/2, rho_data[1]/2), compare=contracted_desc)
    if type(dec) == tuple:
        dEddesc = (dec[0] * ddesc[0], dec[1] * ddesc[1])
    else:
        dEddesc = (dec * ddesc[0] * 0.5, dec * ddesc[1] * 0.5)

    print('run GP', time.monotonic() - chkpt)
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
                rho_data[spin], density[spin],
                ovlps[spin], grid)

    v_nst = np.stack(v_nst, axis=-1)
    v_grad = np.stack(v_grad, axis=-1)
    vmol = np.stack(vmol, axis=0)

    print('v_nonlocal', time.monotonic() - chkpt)
    chkpt = time.monotonic()

    # TODO v_nst[1] should have 3 dimensions on axis 0 not 2
    vtot = list(vref)
    if vtot[1] is None:
        vtot[1] = np.zeros((vtot[0].shape[0],3))
    if vtot[2] is None:
        vtot[2] = np.zeros((vtot[0].shape[0],2))
    if vtot[3] is None:
        vtot[3] = np.zeros((vtot[0].shape[0],2))
    vtot[0] += v_nst[0]
    vtot[1][:,0] += 2 * v_nst[1][:,0]
    vtot[1][:,2] += 2 * v_nst[1][:,1]
    vtot[2] += v_nst[2]
    vtot[3] += v_nst[3]
    return ec, (vtot[0], vtot[1], vtot[2], vtot[3], v_grad, vmol), None, None


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
