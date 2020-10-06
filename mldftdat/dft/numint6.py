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
        from mldftdat.models import map_c4
        self.mlfunc_x.corr_model = map_c4.VSXCContribs(
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
                                      tu, td, F[0], F[1])

    """
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
    """

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

DEFAULT_CSS = [ 1.85235182e-02,  1.59240884e-03,  1.34328251e-03, -9.60218877e-06]
DEFAULT_COS = [-0.00733269,  0.03943819, -0.01573257,  0.0124642 ]
DEFAULT_CX = [-0.01686982,  0.03515553, -0.01826475,  0.01427994]
DEFAULT_CM = [-0.00109183,  0.00171778, -0.00231616,  0.00269598]
DEFAULT_CA = [ 0.27852635,  0.10400466,  0.07101222, -0.0465479 ]
DEFAULT_DSS = [ 0.00067551, -0.11368569,  0.03248452,  0.00800122, -0.015655  ,
  0.00933268]
DEFAULT_DOS = [-0.0054529 , -0.04713826, -0.00382662,  0.00089633, -0.00106169,
  0.0052331 ]
DEFAULT_DX = [-0.00692729,  0.09775733, -0.09782589, -0.001562  ,  0.00353727,
 -0.00772637]
DEFAULT_DM = [-0.00058879, -0.05630044,  0.03688431,  0.00046997,  0.00375369,
  0.00430558]
DEFAULT_DA = [-0.01076074, -0.00428565,  0.02709608,  0.0001212 , -0.00057862,
  0.00045712]

DEFAULT_CSS = [0.01918448, 0.00564197, 0.00183687, 0.00268551]
DEFAULT_COS = [0.00467828, 0.03201976, 0.00147455, 0.00521976]
DEFAULT_CX = [-2.28728908e-03,  2.45566547e-02,  6.43830056e-05,  4.18863554e-03]
DEFAULT_CM = [ 0.00321239,  0.0009118 , -0.00111314,  0.00125196]
DEFAULT_CA = [ 0.36826917, -0.02012673,  0.22605946, -0.08914147]
DEFAULT_DSS = 1 * np.array([ 0.00057849, -0.13111775,  0.04027038,  0.00524443,  0.00534743,  0.01186421])
DEFAULT_DOS = [-0.00132567, -0.03299083,  0.01540183,  0.00264496, -0.00127063,
  0.00672478]
DEFAULT_DX = [-0.00112333,  0.04962348, -0.04250464, -0.00269584,  0.00104411,
 -0.00717285]
DEFAULT_DM = [-1.54949758e-05, -5.59163411e-02,  4.06422780e-02,  2.39106640e-03,
  6.97017989e-04,  2.28542889e-02]
DEFAULT_DA = [-4.12438660e-03,  2.98865297e-03, -4.97481560e-03,  3.79121735e-05,
 -1.95103682e-04,  6.94718714e-05]

DEFAULT_CSS = [0.01041255, 0.00285108, 0.00093685, 0.00232467]
DEFAULT_COS = [0.00798011, 0.01393721, 0.00694208, 0.00113179]
DEFAULT_CX = [ 0.00199167,  0.00943601,  0.00589237, -0.000264  ]
DEFAULT_CM = [ 0.00236945,  0.00032528, -0.00042179,  0.00015882]
DEFAULT_CA = [ 0.34570011, -0.08113403,  0.19654027, -0.05020341]
DEFAULT_DSS = [ 0.00038207, -0.11996255,  0.06394313,  0.00107678,  0.00508358,  0.02789195]
DEFAULT_DOS = [-0.0011635 , -0.02539043,  0.03105214,  0.00257946, -0.00234017,  0.00228939]
DEFAULT_DX = [-0.00127014, -0.00579732,  0.02582202, -0.00190989, -0.00024312, -0.00112326]
DEFAULT_DM = [ 0.00010496, -0.01948697,  0.04448069,  0.00139873,  0.0026852 ,  0.01342585]
DEFAULT_DA = [-0.00148732,  0.00837184, -0.01767974, -0.00011238,  0.0004853 , -0.00057994]


DEFAULT_CSS = [0.01041255, 0.00285108, 0.00093685, 0.00232467]
DEFAULT_COS = [0.00798011, 0.01393721, 0.00694208, 0.00113179]
DEFAULT_CX = [ 0.00199167,  0.00943601,  0.00589237, -0.000264  ]
DEFAULT_CM = [ 0.00236945,  0.00032528, -0.00042179,  0.00015882]
DEFAULT_CA = [ 0.34570011, -0.08113403,  0.19654027, -0.05020341]
DEFAULT_DSS = [ 0.00038207, -0.11996255,  0.06394313,  0.00107678,  0.00508358,  0.02789195]
DEFAULT_DOS = [-0.0011635 , -0.02539043,  0.03105214,  0.00257946, -0.00234017,  0.00228939]
DEFAULT_DX = [-0.00127014, -0.00579732,  0.02582202, -0.00190989, -0.00024312, -0.00112326]
DEFAULT_DM = [ 0.00010496, -0.01948697,  0.04448069,  0.00139873,  0.0026852 ,  0.01342585]
DEFAULT_DA = [-0.00148732,  0.00837184, -0.01767974, -0.00011238,  0.0004853 , -0.00057994]

DEFAULT_CSS = [0.01837658, 0.00388386, 0.00242569, 0.00380519]
DEFAULT_COS = [0.01551807, 0.01646528, 0.00762513, 0.00268281]
DEFAULT_CX = [0.00830597, 0.01434943, 0.00814918, 0.00087115]
DEFAULT_CM = [ 1.94963869e-04, -6.22115293e-05,  5.52116775e-04, -6.54518443e-04]
DEFAULT_CA = [ 0.34028604, -0.08169513,  0.23184562, -0.03471867]
DEFAULT_DSS = [ 0.00066699, -0.11118242,  0.02803618,  0.0005271 ,  0.00062375,
  0.02270012]
DEFAULT_DOS = [-0.00121522, -0.01997459,  0.02653492,  0.00177888, -0.00520799,
  0.00049216]
DEFAULT_DX = [-0.00176311,  0.00343492,  0.05508761, -0.00127211,  0.0020889 ,
  0.00053261]
DEFAULT_DM = [-0.00067015,  0.02179826, -0.01243287,  0.00053154,  0.00470197,
 -0.00124765]
DEFAULT_DA = [-0.003466  ,  0.00903202, -0.01741449, -0.00011676,  0.00044571,
 -0.00052964]

DEFAULT_CSS = [0.01373016, 0.00251136, 0.00220887, 0.00246698]
DEFAULT_COS = [ 1.78794146e-02,  5.95977706e-03,  8.52301194e-03, -2.82101675e-05]
DEFAULT_CX = [ 0.01373885,  0.00442742,  0.00903651, -0.0014393 ]
DEFAULT_CM = [ 0.00015336, -0.00034698,  0.00062514, -0.00073769]
DEFAULT_CA = [ 0.33977889, -0.08716333,  0.18935415, -0.04004029]
DEFAULT_DSS = [ 0.00044428, -0.09200185,  0.02296426, -0.0001243 ,  0.00140415,
          0.02371172]
DEFAULT_DOS = [-0.00050403, -0.02560208,  0.03280187,  0.00179605, -0.00385251,
         -0.00081374]
DEFAULT_DX = [-0.00069998,  0.00356027,  0.04754932, -0.00116589,  0.00060641,
          0.00171958]
DEFAULT_DM = [-3.04161931e-04,  1.34650093e-02, -4.41653097e-03,  3.22744169e-04,
          6.46549398e-03,  3.60830146e-05]
DEFAULT_DA = [-0.00198547,  0.00902299, -0.01699909, -0.00011945,  0.00043307,
         -0.00051764]

DEFAULT_CSS = [0.01975172, 0.02643192, 0.00684282, 0.01066147]
DEFAULT_COS = [0.01975172, 0.02643192, 0.00684282, 0.01066147]
DEFAULT_CX = [0.00577485, 0.01369322, 0.00759016, 0.00202679]
DEFAULT_CM = [ 0.00175147, -0.00161191,  0.00176254, -0.00193061]
DEFAULT_CA = [0]*4#[ 0.34794482, -0.05576447,  0.24719607, -0.03914316]
DEFAULT_DSS = [-0.00127952, -0.04815332, -0.01275823,  0.00101604,  0.00200155,
          0.00137506]
DEFAULT_DOS = [-0.00127952, -0.04815332, -0.01275823,  0.00101604,  0.00200155,
          0.00137506]
DEFAULT_DX = [-0.00285345,  0.04897107,  0.02013399, -0.00093622, -0.00325507,
         -0.00078262]
DEFAULT_DM = [-8.73440107e-05, -7.79228604e-03,  1.22301358e-02,  4.15236651e-04,
          8.89644694e-03,  1.33991076e-02]
DEFAULT_DA = [0]*6#[ 1.02182307e-03,  6.29811168e-03, -8.03489007e-03, -1.29607620e-06, -3.49977408e-04,  4.65553133e-04]

DEFAULT_CSS = [-0.05452783, -0.00076775, -0.01041533,  0.00035368]
DEFAULT_COS = [-0.06617057,  0.00878463, -0.01445688,  0.00632798]
DEFAULT_CX = [-0.07860136,  0.01874689, -0.02177897,  0.01350697]
DEFAULT_CM = [-0.01108625,  0.00712482, -0.00612263,  0.00535104]
DEFAULT_CA = [ 0.06855746, -0.08013932, -0.00688188,  0.0133931 ]
DEFAULT_DSS = [-0.00037777, -0.0408486 ,  0.04620767,  0.00957358, -0.01502667,
 -0.01146411]
DEFAULT_DOS = [-0.00204951,  0.00461263,  0.04031432, -0.00464195,  0.01852419,
 -0.01416855]
DEFAULT_DX = [-0.00413646,  0.04395188, -0.22308177,  0.00416432, -0.01603316,
  0.01326185]
DEFAULT_DM = [-0.00037919,  0.00480977, -0.01043522, -0.0071598 ,  0.03734132,
 -0.04103396]
DEFAULT_DA = [-0.01323424, -0.01298428,  0.06510855,  0.00024175, -0.0011504 ,
  0.00109085]

DEFAULT_CSS = [0]*4
DEFAULT_COS = [0]*4
DEFAULT_CX = [0]*4
DEFAULT_CM = [0]*4
DEFAULT_CA = [0]*4
DEFAULT_DSS = [ 0.00030139, -0.20149783,  0.03334822,  0.0065964 , -0.00139713,
  0.06582131]
DEFAULT_DOS = [-4.54440375e-03,  4.68927593e-02,  1.71085167e-01, -7.55834408e-05,
  1.43144853e-02, -2.47920272e-02]
DEFAULT_DX = [-0.00026006, -0.1129298 , -0.17554901,  0.00190763, -0.02059737,
  0.02543308]
DEFAULT_DM = [-0.00038683,  0.0128838 ,  0.06481756, -0.003573  ,  0.02111423,
  0.04788578]
DEFAULT_DA = [-5.06180839e-02,  1.85477868e-03,  1.12961979e-02, -8.91128820e-05,
  4.22394485e-04, -1.21530515e-03]

def setup_rks_calc(mol, mlfunc_x, css=DEFAULT_CSS, cos=DEFAULT_COS,
                   cx=DEFAULT_CX, cm=DEFAULT_CM, ca=DEFAULT_CA,
                   dss=DEFAULT_DSS, dos=DEFAULT_DOS, dx=DEFAULT_DX,
                   dm=DEFAULT_DM, da=DEFAULT_DA,
                   vv10_coeff = None):
    rks = dft.RKS(mol)
    rks.xc = None
    rks._numint = NLNumInt(mlfunc_x, css, cos, cx, cm, ca,
                           dss, dos, dx, dm, da,
                           vv10_coeff)
    return rks

def setup_uks_calc(mol, mlfunc_x, css=DEFAULT_CSS, cos=DEFAULT_COS,
                   cx=DEFAULT_CX, cm=DEFAULT_CM, ca=DEFAULT_CA,
                   dss=DEFAULT_DSS, dos=DEFAULT_DOS, dx=DEFAULT_DX,
                   dm=DEFAULT_DM, da=DEFAULT_DA,
                   vv10_coeff = None):
    uks = dft.UKS(mol)
    uks.xc = None
    uks._numint = NLNumInt(mlfunc_x, css, cos, cx, cm, ca,
                           dss, dos, dx, dm, da,
                           vv10_coeff)
    return uks
