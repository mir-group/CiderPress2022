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
            vxc = [vxc[0][:,0], 0.5 * vxc[1][:,0] + 0.25 * vxc[1][:,1],\
                   vxc[2][:,0], vxc[3][:,0], vxc[4][:,:,0], vxc[5][0,:,:]]
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

    rho_data_u_0 = rho_data[0].copy()
    rho_data_u_1 = rho_data[0].copy()
    rho_data_u_0[4] = 0
    rho_data_u_0[5] = g2u / (8 * rhou)
    rho_data_u_1[4] = 0
    rho_data_u_1[5] = CF * rhou**(5.0/3) + rho_data_u_0[5] + 1e-10

    rho_data_d_0 = rho_data[1].copy()
    rho_data_d_1 = rho_data[1].copy()
    rho_data_d_0[4] = 0
    rho_data_d_0[5] = g2d / (8 * rhod)
    rho_data_d_1[4] = 0
    rho_data_d_1[5] = CF * rhod**(5.0/3) + rho_data_d_0[5] + 1e-10

    co0, v_scan_ud_0 = eval_xc(',MGGA_C_SCAN', (rho_data_u_0, rho_data_d_0),
                               spin = 1)[:2]
    cu1, v_scan_uu_1 = eval_xc(',MGGA_C_SCAN', (rho_data_u_1, 0*rho_data_d_1),
                               spin = 1)[:2]
    cd1, v_scan_dd_1 = eval_xc(',MGGA_C_SCAN', (0*rho_data_u_1, rho_data_d_1),
                               spin = 1)[:2]
    co1, v_scan_ud_1 = eval_xc(',MGGA_C_SCAN', (rho_data_u_1, rho_data_d_1),
                               spin = 1)[:2]
    co0 *= rhot
    cu1 *= rhou
    cd1 *= rhod
    co1 = co1 * rhot - cu1 - cd1
    v_scan_ud_1[0][:,0] -= v_scan_uu_1[0][:,0]
    v_scan_ud_1[0][:,1] -= v_scan_dd_1[0][:,1]
    v_scan_ud_1[1][:,0] -= v_scan_uu_1[1][:,0]
    v_scan_ud_1[1][:,2] -= v_scan_dd_1[1][:,2]
    print('vtau mean', np.mean(v_scan_ud_0[3][:,0] * rhou))
    print('vtau mean', np.mean(v_scan_ud_1[3][:,0] * rhou))
    print('vtau mean', np.mean(v_scan_ud_0[3][:,1] * rhod))
    print('vtau mean', np.mean(v_scan_ud_1[3][:,1] * rhod))
    v_scan_uu_1[3][:] = 0
    v_scan_dd_1[3][:] = 0
    v_scan_ud_1[3][:] = 0
    v_scan_ud_0[3][:] = 0

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
        F[spin], dF[spin] = mlfunc.get_F_and_derivative(desc[spin])

        exc += 2**(1.0/3) * LDA_FACTOR * rho43 * F[spin]
        vtot[0][:,spin] += 2**(1.0/3) * 4.0 / 3 * LDA_FACTOR * rho13 * F[spin]
        dEddesc[spin] = 2**(4.0/3) * LDA_FACTOR * rho43.reshape(-1,1) * dF[spin]
        
    tot, vxc = mlfunc.corr_model.xefc(cu1, cd1, co1, co0,
                                      v_scan_uu_1, v_scan_dd_1,
                                      v_scan_ud_1, v_scan_ud_0,
                                      rhou, rhod, g2u, g2o, g2d,
                                      tu, td, F[0], F[1])

    exc += tot
    vtot[0][:,:] += vxc[0]
    vtot[1][:,:] += vxc[1]
    vtot[3][:,:] += vxc[2]
    dEddesc[0] += vxc[3][:,0,np.newaxis] * dF[0]
    dEddesc[1] += vxc[3][:,1,np.newaxis] * dF[1]

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

DEFAULT_CSS = [0.01694267, 0.01154786, 0.00019305, 0.00565762]
DEFAULT_COS = [ 0.02952912,  0.01509011,  0.03166647, -0.02339623]
DEFAULT_CX = [ 0.00205387,  0.03301437,  0.0105551 , -0.00348026]
DEFAULT_CM = [-0.01269282,  0.0148151 , -0.01662529,  0.01717711]
DEFAULT_CA = [ 0.34101643, -0.01378402,  0.27154121, -0.08137242]
DEFAULT_DSS = [ 0.00034126, -0.14140119,  0.06004581,  0.00776944, -0.01322139,
  0.01130723]
DEFAULT_DOS = [-0.00285467, -0.07683561, -0.05602746,  0.00107202, -0.00256635,
  0.00585838]
DEFAULT_DX = [-0.007279  ,  0.0572915 ,  0.06453063, -0.0008062 ,  0.00220696,
 -0.00468091]
DEFAULT_DM = [-0.00059425, -0.00892464, -0.02585809,  0.00067835, -0.0023319 ,
  0.00061676]
DEFAULT_DA = [ 1.41340827e-03,  7.02634508e-03, -5.76269940e-03, -2.49566825e-05,
 -2.46628527e-04,  1.64864944e-04]

DEFAULT_CSS = [0.0100458 , 0.0072586 , 0.00095743, 0.00394399]
DEFAULT_COS = [0.00632121, 0.02553949, 0.00487545, 0.00380487]
DEFAULT_CX = [-0.00129929,  0.0216167 ,  0.00976573, -0.00125441]
DEFAULT_CM = [-0.00147788, -0.00160216,  0.00335923, -0.00319578]
DEFAULT_CA = [ 0.34561037, -0.05442116,  0.25354443, -0.06530622]
DEFAULT_DSS = [ 0.0002054 , -0.0744199 ,  0.059358  , -0.00174137,  0.01088066,
  0.02379759]
DEFAULT_DOS = [-0.00122135, -0.00200719,  0.04211056,  0.00108703, -0.00208667,
 -0.00430475]
DEFAULT_DX = [-2.86962068e-03, -3.31386092e-02,  5.15970961e-02,  8.40754115e-05,
 -1.79613165e-03,  5.09152327e-03]
DEFAULT_DM = [-0.00303753, -0.02655579, -0.07728049, -0.00043326,  0.00671263,
 -0.01188554]
DEFAULT_DA = [-0.00448212,  0.00967764, -0.02234779, -0.00013738,  0.00058732,
 -0.00070051]

DEFAULT_CSS = [0.01921742, 0.00317641, 0.00299911, 0.00342114]
DEFAULT_COS = [0.01600721, 0.01645205, 0.00745459, 0.00270249]
DEFAULT_CX = [0.00840119, 0.01301878, 0.00770621, 0.00052342]
DEFAULT_CM = [ 0.00140456, -0.0003514 ,  0.00034346, -0.00052276]
DEFAULT_CA = [ 0.33589508, -0.08838812,  0.23104473, -0.03706446]
DEFAULT_DSS = [ 0.00073444, -0.11502326,  0.02273354,  0.00089824, -0.00060023,
  0.02836146]
DEFAULT_DOS = [-0.00040099, -0.02370804,  0.03368014,  0.00299498, -0.00830962,
 -0.00033488]
DEFAULT_DX = [-0.00046298, -0.00377196,  0.07380675, -0.00223734,  0.00437788,
  0.00201595]
DEFAULT_DM = [-0.00050864, -0.01457361, -0.01313629,  0.00150984,  0.00262915,
 -0.007714  ]
DEFAULT_DA = [-0.00216782,  0.0091289 , -0.01767703, -0.00014168,  0.00068019,
 -0.00086588]

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
