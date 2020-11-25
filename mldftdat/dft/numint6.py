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
from scipy.linalg import cho_factor, cho_solve
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

class QuickGrid():
    def __init__(self, coords, weights):
        self.coords = coords
        self.weights = weights

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
            exc, vxc = ni.eval_xc(xc_code, mol, rho, QuickGrid(coords, weight), dms,
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
                                  QuickGrid(coords, weight), (dma, dmb),
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
            print(np.max(np.abs(vmat[1,idm])))
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
            print(np.max(np.abs(vmat[1,idm])))
            vmat[0,idm] += 0.5 * vmol[0,:,:]
            vmat[1,idm] += 0.5 * vmol[1,:,:]
            print(np.max(np.abs(vmat[1,idm])))

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
        print("VMAT", np.max(np.abs(vmat[0,i])))
        print("VMAT", np.max(np.abs(vmat[1,i])))
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

    def __init__(self, mlfunc_x, css, cos, cx, cm, ca,
                 dss, dos, dx, dm, da, vv10_coeff = None):
        super(NLNumInt, self).__init__()
        self.mlfunc_x = mlfunc_x
        from mldftdat.models import map_c6
        self.mlfunc_x.corr_model = map_c6.VSXCContribs(
                                    css, cos, cx, cm, ca,
                                    dss, dos, dx, dm, da)

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
            mol.auxmol, mol.ao_to_aux = setup_aux(mol)

        N = grid.weights.shape[0]
        print('XCCODE', xc_code, spin)
        has_base_xc = (xc_code is not None) and (xc_code != '')
        if has_base_xc:
            exc0, vxc0, _, _ = eval_xc(xc_code, rho_data, spin, relativity,
                                       deriv, omega, verbose)

        if spin == 0:
            print('NO SPIN POL')
            exc, vxc, _, _ = _eval_xc_0(self.mlfunc_x, mol,
                                      (rho_data / 2, rho_data / 2), grid,
                                      (rdm1, rdm1))
            vxc = [vxc[0][:,1], 0.5 * vxc[1][:,2] + 0.25 * vxc[1][:,1],\
                   vxc[2][:,1], vxc[3][:,1], vxc[4][:,:,1], vxc[5][1,:,:]]
        else:
            print('YES SPIN POL')
            exc, vxc, _, _ = _eval_xc_0(self.mlfunc_x, mol,
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


class NLNumInt2(NLNumInt):

    def __init__(self, mlfunc_x, cx, c0, c1, dx, d0, d1,
                 vv10_coeff = None, fterm_scale=2.0):
        super(NLNumInt, self).__init__()
        self.mlfunc_x = mlfunc_x
        from mldftdat.models import map_c8
        self.mlfunc_x.corr_model = map_c8.VSXCContribs(
                                    cx, c0, c1, dx, d0, d1,
                                    fterm_scale=fterm_scale)

        if vv10_coeff is None:
            self.vv10 = False
        else:
            self.vv10 = True
            self.vv10_b, self.vv10_c = vv10_coeff


class NLNumInt3(NLNumInt):

    def __init__(self, mlfunc_x, d, dx, cx,
                 vv10_coeff = None, fterm_scale=2.0):
        super(NLNumInt, self).__init__()
        self.mlfunc_x = mlfunc_x
        from mldftdat.models import map_c9
        self.mlfunc_x.corr_model = map_c9.VSXCContribs(
                                    d, dx, cx,
                                    fterm_scale=fterm_scale)

        if vv10_coeff is None:
            self.vv10 = False
        else:
            self.vv10 = True
            self.vv10_b, self.vv10_c = vv10_coeff


def _eval_xc_0(mlfunc, mol, rho_data, grid, rdm1):
    import time

    #if spin == 0:
    #    spin = 1
    #else:
    #    spin = 2

    CF = 0.3 * (6 * np.pi**2)**(2.0/3)

    chkpt = time.monotonic()

    density = (np.einsum('npq,pq->n', mol.ao_to_aux, rdm1[0]),\
               np.einsum('npq,pq->n', mol.ao_to_aux, rdm1[1]))
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

    for spin in range(2):
        pr2 = 2 * rho_data[spin] * np.linalg.norm(grid.coords, axis=1)**2
        print('r2', spin, np.dot(pr2, grid.weights))
        rho43 = ntup[spin]**(4.0/3)
        rho13 = ntup[spin]**(1.0/3)
        desc[spin] = np.zeros((N, len(mlfunc.desc_list)))
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
        F[spin], dF[spin] = mlfunc.get_F_and_derivative(desc[spin], 2*ntup[spin])
        #F[spin][(ntup[spin]<1e-8)] = 0
        #dF[spin][(ntup[spin]<1e-8)] = 0
        exc += 2**(1.0/3) * LDA_FACTOR * rho43 * F[spin]
        vtot[0][:,spin] += 2**(1.0/3) * 4.0 / 3 * LDA_FACTOR * rho13 * F[spin]
        dEddesc[spin] = 2**(4.0/3) * LDA_FACTOR * rho43.reshape(-1,1) * dF[spin]
        
    tot, vxc = mlfunc.corr_model.xefc(rhou, rhod, g2u, g2o, g2d,
                                      tu, td, F[0], F[1],
                                      include_aug_sl=True,
                                      include_aug_nl=True)

    
    print('V ACTION')
    weights = grid.weights
    print(np.dot(vxc[0][:,0], rhou * weights))
    print(np.dot(vxc[0][:,1], rhod * weights))
    print(np.dot(vxc[1][:,0], g2u * weights))
    print(np.dot(vxc[1][:,1], g2o * weights))
    print(np.dot(vxc[1][:,2], g2d * weights))
    print(np.dot(vxc[2][:,0], tu * weights))
    print(np.dot(vxc[2][:,1], td * weights))
    
    exc += tot
    vtot[0][:,:] += vxc[0]
    vtot[1][:,:] += vxc[1]
    vtot[3][:,:] += vxc[2]
    dEddesc[0] += 2 * vxc[3][:,0,np.newaxis] * dF[0]
    dEddesc[1] += 2 * vxc[3][:,1,np.newaxis] * dF[1]
    

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

    vtot[0] += v_nst[0]
    vtot[1][:,0] += 2 * v_nst[1][:,0]
    vtot[1][:,2] += 2 * v_nst[1][:,1]
    vtot[2] += v_nst[2]
    vtot[3] += v_nst[3]

    return exc / (rhot + 1e-20), (vtot[0], vtot[1], vtot[2], vtot[3], v_grad, vmol), None, None


def setup_aux(mol):
    nao = mol.nao_nr()
    auxmol = df.make_auxmol(mol, 'weigend+etb')
    #auxmol = df.make_auxmol(mol, auxbasis)
    naux = auxmol.nao_nr()
    # shape (naux, naux), symmetric
    aug_J = auxmol.intor('int2c2e')
    # shape (nao, nao, naux)
    aux_e2 = df.incore.aux_e2(mol, auxmol)
    #print(aux_e2.shape)
    # shape (naux, nao * nao)
    """
    aux_e2 = aux_e2.reshape((-1, aux_e2.shape[-1])).transpose()
    aux_e2 = np.ascontiguousarray(aux_e2)
    lu, piv, info = dgetrf(aug_J, overwrite_a = True)
    inv_aug_J, info = dgetri(lu, piv, overwrite_lu = True)
    ao_to_aux = dgemm(1, inv_aug_J, aux_e2)
    """
    """
    aux_e2 = aux_e2.reshape((-1, aux_e2.shape[-1])).T
    inv_aug_J = np.linalg.inv(aug_J)
    ao_to_aux = np.dot(inv_aug_J, aux_e2)
    ao_to_aux = ao_to_aux.reshape(naux, nao, nao)
    """
    aux_e2 = aux_e2.reshape((-1, aux_e2.shape[-1])).T
    c_and_lower = cho_factor(aug_J)
    ao_to_aux = cho_solve(c_and_lower, aux_e2)
    ao_to_aux = ao_to_aux.reshape(naux, nao, nao)

    return auxmol, ao_to_aux

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

DEFAULT_COS = [-0.03569016,  0.00124141, -0.00101882,  0.00017673]
DEFAULT_CX = [ 0.01719458, -0.00963045, -0.00879279, -0.0057004 ]
DEFAULT_CA = [-0.71581393, -0.36932482, -0.19691875, -0.12507637]
DEFAULT_DOS = [-8.00966168e-04, -1.16338402e-02, -9.04350500e-02,  7.31694234e-04,
         -5.38843153e-04, -4.57654739e-05]
DEFAULT_DX = [-0.00363218,  0.00021466,  0.08919607,  0.00117978, -0.00894649,
          0.0114701 ]
DEFAULT_DA = [-7.59229970e-03,  2.63563397e-03,  7.46369471e-03, -3.94199066e-06,
          1.43406887e-04, -5.03764720e-04]

def setup_rks_calc(mol, mlfunc_x, css=DEFAULT_CSS, cos=DEFAULT_COS,
                   cx=DEFAULT_CX, cm=DEFAULT_CM, ca=DEFAULT_CA,
                   dss=DEFAULT_DSS, dos=DEFAULT_DOS, dx=DEFAULT_DX,
                   dm=DEFAULT_DM, da=DEFAULT_DA,
                   vv10_coeff = None, grid_level=3):
    rks = dft.RKS(mol)
    rks.xc = None
    rks._numint = NLNumInt(mlfunc_x, css, cos, cx, cm, ca,
                           dss, dos, dx, dm, da,
                           vv10_coeff)
    rks.grids.level = grid_level
    rks.grids.build()
    return rks

def setup_uks_calc(mol, mlfunc_x, css=DEFAULT_CSS, cos=DEFAULT_COS,
                   cx=DEFAULT_CX, cm=DEFAULT_CM, ca=DEFAULT_CA,
                   dss=DEFAULT_DSS, dos=DEFAULT_DOS, dx=DEFAULT_DX,
                   dm=DEFAULT_DM, da=DEFAULT_DA,
                   vv10_coeff = None, grid_level=3):
    uks = dft.UKS(mol)
    uks.xc = None
    uks._numint = NLNumInt(mlfunc_x, css, cos, cx, cm, ca,
                           dss, dos, dx, dm, da,
                           vv10_coeff)
    uks.grids.level = grid_level
    uks.grids.build()
    return uks

C0 = [0., 0., 0., 0.]
D0 = [0.92642313, 0.0612833 , 0.55086957, 0.49373008]
C1 = [0., 0., 0., 0.]
D1 = [-0.00482615, -0.10379149,  0.03454682,  0.09281949]
DX = [ 0.27384179, -0.09953398,  0.11309756, -0.35438573,  0.65416998,
         -0.06390127,  0.2808232 ,  0.06472575,  0.07132121]
CX = [ 0.01850537,  0.10110328,  0.08678594,  0.17720392,  0.22679325,
         -1.06448733, -0.24174785,  0.64852308,  0.06586449, -0.06078151,
          -0.73831308, -0.63320915, -1.10079207, -0.20164018, -0.59467933]

C0 = [0., 0., 0., 0.]
D0 = [ 0.96256403, -0.08511496,  0.16299247,  0.99210899]
C1 = [0., 0., 0., 0.]
D1 = [-0.00694123, -0.08840102,  0.01700766,  0.08447464]
DX = [ 0.28905389, -0.0998426 ,  0.12869457, -0.38328385,  0.64922815,
 -0.06751789,  0.277406  ,  0.06919556,  0.06402194, -0.00963708,
  0.10089947,  0.09044604,  0.19056558,  0.22981789,0,0,0,0,0]
CX = [-1.08582313, -0.24356876,  0.67803951,  0.04004844, -0.06792441,
 -0.70230199, -0.61922714, -1.08000744, -0.20070822, -0.59659058,0, 0,0,0,0]

def setup_rks_calc2(mol, mlfunc_x, cx=CX, c0=C0, c1=C1, dx=DX, d0=D0, d1=D1,
                    vv10_coeff=None, fterm_scale=2.0, grid_level=3):
    rks = dft.RKS(mol)
    rks.xc = None
    rks._numint = NLNumInt2(mlfunc_x, cx, c0, c1, dx, d0, d1,
                            vv10_coeff=vv10_coeff,
                            fterm_scale=fterm_scale)
    rks.grids.level = grid_level
    rks.grids.build()
    return rks

def setup_uks_calc2(mol, mlfunc_x, cx=CX, c0=C0, c1=C1, dx=DX, d0=D0, d1=D1,
                    vv10_coeff=None, fterm_scale=2.0, grid_level=3):
    uks = dft.UKS(mol)
    uks.xc = None
    uks._numint = NLNumInt2(mlfunc_x, cx, c0, c1, dx, d0, d1,
                            vv10_coeff=vv10_coeff,
                            fterm_scale=fterm_scale)
    uks.grids.level = grid_level
    uks.grids.build()
    return uks

V3_D = [0.01562573, 0.09629376, 0.09005671, 0.05556645, 0.05636563, 0.05542296]
V3_DX = [ 0.00876918,  0.1533378 , -0.07827145,  0.42990607,  0.81886636,
         -1.06456548,  0.55013466, -1.12063554, -0.22586826,  0.17390912,
           0.07342244, -0.55254566, -0.46900012,  0.01055503, -0.65602093]
V3_CX = [ 0.99435021,  0.90440151,  0.29465447, -0.14204387,  1.22866657,
         -2.55106949,  0.27385608, -0.1678324 , -0.51022371, -0.74233395,
           0.87083219,  0.35525901, -2.21769842]

V3_D = [0.101416  , 0.19702274, 0.04648554, 0.04476192, 0.00893325, 0.05131002]
V3_DX = [ 0.00303226,  0.16693523, -0.08922236,  0.36682272,  0.5517252 ,
         -0.4492544 ,  0.392798  , -0.7964105 , -0.0913046 ,  0.0764949 ,
           0.15904028, -0.45308748, -0.16027391,  0.01336907, -0.15479775]
V3_CX = [ 0.58162349,  0.65215009, -0.09015343, -0.20392164,  0.73793179,
         -1.47213268,  0.02184713, -0.43104224, -0.38730572, -0.64877241,
           0.22477249, -0.1231774 , -1.6799939 ]

def setup_rks_calc3(mol, mlfunc_x, d=V3_D, dx=V3_DX, cx=V3_CX,
                    vv10_coeff=None, fterm_scale=2.0, grid_level=3):
    rks = dft.RKS(mol)
    rks.xc = None
    rks._numint = NLNumInt3(mlfunc_x, d, dx, cx,
                            vv10_coeff=vv10_coeff,
                            fterm_scale=fterm_scale)
    rks.grids.level = grid_level
    rks.grids.build()
    return rks

def setup_uks_calc3(mol, mlfunc_x, d=V3_D, dx=V3_DX, cx=V3_CX,
                    vv10_coeff=None, fterm_scale=2.0, grid_level=3):
    print("UKS ARGS", gto.mole.pack(mol), d, dx, cx, vv10_coeff, fterm_scale, grid_level)
    uks = dft.UKS(mol)
    uks.xc = None
    uks._numint = NLNumInt3(mlfunc_x, d, dx, cx,
                            vv10_coeff=vv10_coeff,
                            fterm_scale=fterm_scale)
    uks.grids.level = grid_level
    uks.grids.build()
    return uks
