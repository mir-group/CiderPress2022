import numpy as np
import os
from mldftdat.dft.xc_models import NormGPFunctional
from mldftdat.density import contract_exchange_descriptors, contract21_deriv, contract21
from mldftdat.dft.utils import v_basis_transform
import traceback
from numba.experimental import jitclass
from numba import jit
from mldftdat.pyscf_utils import get_dft_input2

LDA_FACTOR = - 3.0 / 4.0 * (3.0 / np.pi)**(1.0/3)

# conversion for spherical harmonic orders
l1_qe2py = [1,2,0]
l1_py2qe = [2,0,1]
l2_qe2py = [4,2,0,1,3]
l2_py2qe = [2,3,1,4,0]

def test_permutations(grad, g1):
    import itertools
    for lst in itertools.permutations([0,1,2]):
        for i,j,k in [(1,1,1),(-1,1,1),(1,-1,1),(-1,-1,1),(1,1,-1),(-1,1,-1),(1,-1,-1),(-1,-1,-1)]:
            arr = np.array([i,j,k]).reshape(-1,1)
            print (lst, np.sum(np.einsum('ap,ap->p',grad[list(lst),:]*arr,g1)))

def init_pyfort():
    return TestPyFort()

class TestPyFort:

    def __init__(self, mlfunc=None):
        if mlfunc is None:
            dirname = os.path.dirname(os.path.abspath(__file__))
            fname = os.path.join(dirname, '../../functionals/B3LYP_CIDER.yaml')
            self.mlfunc = NormGPFunctional.load(fname)
            #from joblib import load
            #self.mlfunc = load(os.path.join(dirname, '../../test_files/agpr_spline_example_34.joblib'))
        else:
            self.mlfunc = mlfunc

    def get_xc_fortran(self, *args, **kwargs):
        #print ("calling Python")
        no_swap = kwargs.get('no_swap') or False
        try:
            xfac = args[9] * LDA_FACTOR
            #print(xfac)
            nspin = args[0].shape[-1]
            ngrid = args[0].shape[-2]
            raw_desc = np.concatenate([
                np.abs(args[0].reshape(1,ngrid,nspin)),
                args[1],
                args[2].reshape(1,ngrid,nspin) * 0,
                args[2].reshape(1,ngrid,nspin),
                args[3].transpose(1,0,2)
            ])
            #print('feat sum', raw_desc[6].sum(), np.abs(raw_desc[7:10]).sum(),
            #    np.abs(raw_desc)[10:15].sum())
            if not no_swap:
                raw_desc[7:10] = raw_desc[7:10][l1_qe2py]
                raw_desc[7:9] *= -1
                raw_desc[10:15] = raw_desc[10:15][l2_qe2py]
                raw_desc[11] *= -1
                raw_desc[13] *= -1
            contracted_desc = [None] * nspin
            F = [None] * nspin
            dF = [None] * nspin
            # TODO remove line below
            raw_desc[5,:,:] = np.linalg.norm(raw_desc[1:4,:,:], axis=0)**2 / (8*raw_desc[0,:,:])
            rho_data = raw_desc[:6]
            rho, g, alpha = get_dft_input2(rho_data)[:3]
            #test_permutations(raw_desc[1:4,:,0], raw_desc[7:10,:,0])
            #print (np.mean(1/(1+alpha**2)*rho)/np.mean(rho))
            for s in range(nspin):
                contracted_desc[s] = contract_exchange_descriptors(raw_desc[:,:,s])
                contracted_desc[s] = contracted_desc[s][self.mlfunc.desc_order]
                F[s], dF[s] = self.mlfunc.get_F_and_derivative(contracted_desc[s])
                args[4][:] += xfac * np.abs(args[0][:,s])**(4./3) * \
                              np.sign(args[0][:,s]) * F[s]
                dEddesc = (xfac * np.abs(args[0][:,s])**(4./3) * \
                          np.sign(args[0][:,s])).reshape(-1,1) * dF[s]
                vfeat, v_nst, v_grad = functional_derivative_loop(
                    self.mlfunc, dEddesc,
                    raw_desc[:,:,s], raw_desc[:6,:,s],
                )
                #print('feat sum', raw_desc[6].sum(), np.abs(raw_desc[7:10]).sum(), np.abs(raw_desc)[10:15].sum(), vfeat.shape)
                #print('vfeat sum', vfeat[0].sum(), np.abs(vfeat[1:4]).sum(), np.abs(vfeat[4:9]).sum(), vfeat.shape)
                if not no_swap:
                    vfeat[1:4] = vfeat[1:4][l1_py2qe]
                    vfeat[1:3] *= -1
                    vfeat[4:9] = vfeat[4:9][l2_py2qe]
                    vfeat[5] *= -1
                    vfeat[7] *= -1
                args[5][:,s] += xfac * 4.0/3 * np.abs(args[0][:,s])**(1./3) * F[s]
                args[8][:,:,s] += vfeat.T
                args[5][:,s] += v_nst[0]
                if no_swap:
                    args[6][:,s] += v_nst[1]
                else:
                    args[6][:,s] += v_nst[1] * 2
                #args[7][:,s] += v_nst[3] # TODO ADD THIS
                #print('tau sum', v_nst[3].sum(), v_nst.shape)
                
                args[10][:,:,s] += v_grad
        except Exception as e:
            print(str(e))
            traceback.print_exc()
            raise e

def functional_derivative_loop(mlfunc, dEddesc,
                               raw_desc, rho_data):
    """
    Core functional derivative loop for the CIDER features,
    called by NLNumInt.
    Args:
        mlfunc (MLFunctional): Exchange functional
        dEddesc (np.ndarray): ngrid x ndesc array of energy derivatives
            with respect to the descriptors.
        raw_desc (np.ndarray): raw CIDER descriptor vectors
        rho_data (np.ndarray): 6 x ngrid
    """
    sprefac = 2 * (3 * np.pi * np.pi)**(1.0/3)
    N = dEddesc.shape[-2]
    n43 = rho_data[0]**(4.0/3)
    svec = rho_data[1:4] / (sprefac * n43 + 1e-20)
    v_npa = np.zeros((4, N))
    v_aniso = np.zeros((3, N))
    vfeat = np.zeros((11,N))
    FSTART = 6

    for i, d in enumerate(mlfunc.desc_order):
        if d == 0:
            v_npa[0] += dEddesc[:,i]
        elif d == 1:
            v_npa[1] += dEddesc[:,i]
        elif d == 2:
            v_npa[3] += dEddesc[:,i]
        else:
            l_add = 0
            if d in [3, 10, 11]:
                if d == 3:
                    g = raw_desc[6]
                    vfeat[6-FSTART] += dEddesc[:,i]
                elif d == 10:
                    g = raw_desc[15]
                    vfeat[15-FSTART] += dEddesc[:,i]
                else:
                    g = raw_desc[16]
                    vfeat[16-FSTART] += dEddesc[:,i]
            elif d == 4:
                g = raw_desc[7:10]
                vfeat[7-FSTART:10-FSTART] += 2 * g * dEddesc[:,i]
            elif d == 6:
                g = raw_desc[10:15]
                vfeat[10-FSTART:15-FSTART] += 2 * g * dEddesc[:,i] / np.sqrt(5)
            elif d == 5:
                g = raw_desc[7:10]
                vfeat[7-FSTART:10-FSTART] += dEddesc[:,i] * svec
                v_aniso += dEddesc[:,i] * g
            elif d == 7:
                g = raw_desc[10:15]
                dfmul = contract21_deriv(svec)
                ddesc_dsvec = contract21(g, svec)
                v_aniso += dEddesc[:,i] * 2 * ddesc_dsvec
                vfeat[10-FSTART:15-FSTART] += dfmul * dEddesc[:,i]
            elif d == 8:
                g2 = raw_desc[10:15]
                g1 = raw_desc[7:10]
                dfmul = contract21_deriv(svec, g1)
                ddesc_dsvec = contract21(g2, g1)
                ddesc_dg1 = contract21(g2, svec)
                v_aniso += dEddesc[:,i] * ddesc_dsvec
                vfeat[7-FSTART:10-FSTART] += dEddesc[:,i] * ddesc_dg1
                vfeat[10-FSTART:15-FSTART] += dEddesc[:,i] * dfmul
            elif d == 9:
                g2 = raw_desc[10:15]
                g1 = raw_desc[7:10]
                dfmul = contract21_deriv(g1)
                ddesc_dg1 = 2 * contract21(g2, g1)
                vfeat[7-FSTART:10-FSTART] += dEddesc[:,i] * ddesc_dg1
                vfeat[10-FSTART:15-FSTART] += dEddesc[:,i] * dfmul
            else:
                raise NotImplementedError('Cannot take derivative for code %d' % d)
        g = g1 = g2 = None

    v_npa[-1,:] = 0.0 # TODO REMOVE
    v_nst = v_basis_transform(rho_data, v_npa)
    v_nst[0] += np.einsum('ap,ap->p', -4.0 * svec / (3 * rho_data[0] + 1e-20), v_aniso)
    v_grad = v_aniso / (sprefac * n43 + 1e-20)
    
    return vfeat, v_nst, v_grad
