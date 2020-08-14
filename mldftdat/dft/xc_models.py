from abc import ABC, abstractmethod, abstractproperty
import numpy as np
from pyscf.dft.libxc import eval_xc

"""
NORMALIZED_GRAD_CODE = -1
ALPHA_CODE = -2
L0_INTEGRAL_CODE = 0
L1_INTEGRAL_CODE = 1
L2_INTEGRAL_CODE = 2
L1_DOT_CODE = 3
L2_CONTRACT_CODE = 4
"""

def identity(x):
    return x.copy()

def square(x):
    return x**2

def single(x):
    return np.ones(x.shape)

def extract_kernel_components(kernel):
    """
    For an sklearn Kernel object composed of
    three kernels of the form:
    kernel = (const * comp) + (noise),
    extract the three components
    """
    return kernel.k1.k1, kernel.k1.k2, kernel.k2

# res
# 0:  rho
# 1:  s
# 2:  alpha
# 3:  nabla
# 4:  g0
# 5:  norm(g1)
# 6:  g1 dot svec
# 7:  norm(ddrho_{l=2})
# 8:  norm(g2)
# 9:  svec dot ddrho_{l=2} dot svec
# 10: g1 dot ddrho_{l=2} dot svec
# 11: g1 dot ddrho_{l=2} dot g1
# 12: svec dot g2 dot svec
# 13: g1 dot g2 dot svec
# 14: g1 dot g2 dot g1
# 15: g0-0.5
# 16: g0-2
class Descriptor():

    def __init__(self, code, transform = identity,
                 transform_deriv = single, mul = 1.0):
        self._transform = transform
        self._transform_deriv = transform_deriv
        self.code = code
        self.mul = mul

    def transform_descriptor(self, desc, deriv = 0):
        if deriv == 0:
            return self._transform(desc[self.code])
        else:
            return self._transform(desc[self.code]),\
                   self._transform_deriv(desc[self.code])

class MLFunctional(ABC):

    @abstractmethod
    def get_F(self, X):
        pass

    @abstractmethod
    def get_derivative(self, X):
        pass

kappa = 0.804
mu = 0.2195149727645171

class PBEFunctional(MLFunctional):

    def __init__(self):
        self.desc_list = [Descriptor(1)]
        self.y_to_f_mul = None

    def get_F(self, X):
        p = X.flatten()**2
        return 1 + kappa - kappa / (1 + mu * p / kappa)
        
    def get_derivative(self, X):
        p = X.flatten()**2
        return (mu / (1 + mu * p / kappa)**2).reshape(-1,1)


class SCANFunctional(MLFunctional):

    def __init__(self):
        self.desc_list = [Descriptor(1), Descriptor(2)]
        self.y_to_f_mul = None

    def get_F(self, X):
        p = X[:,0]**2
        s = X[:,0]
        alpha = X[:,1]
        muak = 10.0 / 81
        k1 = 0.065
        b2 = np.sqrt(5913 / 405000)
        b1 = (511 / 13500) / (2 * b2)
        b3 = 0.5
        b4 = muak**2 / k1 - 1606 / 18225 - b1**2
        h0 = 1.174
        a1 = 4.9479
        c1 = 0.667
        c2 = 0.8
        dx = 1.24
        tmp1 = muak * p
        tmp2 = 1 + b4 * p / muak * np.exp(-np.abs(b4) * p / muak)
        tmp3 = b1 * p + b2 * (1 - alpha) * np.exp(-b3 * (1 - alpha)**2)
        x = tmp1 * tmp2 + tmp3**2
        h1 = 1 + k1 - k1 / (1 + x / k1)
        gx = 1 - np.exp(-a1 / np.sqrt(s + 1e-9))
        dgdp = - a1 / 4 * (s + 1e-9)**(-2.5) * np.exp(-a1 / np.sqrt(s + 1e-9))
        fx = np.exp(-c1 * alpha / (1 - alpha)) * (alpha < 1)\
             - dx * np.exp(c2 / (1 - alpha)) * (alpha > 1)
        fx[np.isnan(fx)] = 0
        assert (not np.isnan(fx).any())
        Fscan = gx * (h1 + fx * (h0 - h1))
        return Fscan

    def get_derivative(self, X):
        p = X[:,0]**2
        s = X[:,0]
        alpha = X[:,1]
        muak = 10.0 / 81
        k1 = 0.065
        b2 = np.sqrt(5913 / 405000)
        b1 = (511 / 13500) / (2 * b2)
        b3 = 0.5
        b4 = muak**2 / k1 - 1606 / 18225 - b1**2
        h0 = 1.174
        a1 = 4.9479
        c1 = 0.667
        c2 = 0.8
        dx = 1.24
        tmp1 = muak * p
        tmp2 = 1 + b4 * p / muak * np.exp(-np.abs(b4) * p / muak)
        tmp3 = b1 * p + b2 * (1 - alpha) * np.exp(-b3 * (1 - alpha)**2)
        x = tmp1 * tmp2 + tmp3**2
        h1 = 1 + k1 - k1 / (1 + x / k1)
        gx = 1 - np.exp(-a1 / np.sqrt(s + 1e-9))
        dgdp = - a1 / 4 * (s + 1e-9)**(-2.5) * np.exp(-a1 / np.sqrt(s + 1e-9))
        fx = np.exp(-c1 * alpha / (1 - alpha)) * (alpha < 1)\
             - dx * np.exp(c2 / (1 - alpha)) * (alpha > 1)
        fx[np.isnan(fx)] = 0
        assert (not np.isnan(fx).any())
        Fscan = gx * (h1 + fx * (h0 - h1))
        dxdp = muak * tmp2 + tmp1 * (b4 / muak * np.exp(-np.abs(b4) * p / muak)\
               - b4 * np.abs(b4) * p / muak**2 * np.exp(-np.abs(b4) * p / muak))\
               + 2 * tmp3 * b1
        dxda = 2 * tmp3 * (-b2 * np.exp(-b3 * (1 - alpha)**2) \
                            + 2 * b2 * b3 * (1 - alpha)**2 * np.exp(-b3 * (1 - alpha)**2) )
        dhdx = 1 / (1 + x / k1)**2
        dhdp = dhdx * dxdp
        dhda = dhdx * dxda
        dfda = (-c1 * alpha / (1 - alpha)**2 - c1 / (1 - alpha))\
                * np.exp(-c1 * alpha / (1 - alpha)) * (alpha < 1)\
                - dx * c2 / (1 - alpha)**2 * np.exp(c2 / (1 - alpha)) * (alpha > 1)
        dfda[np.isnan(dfda)] = 0

        dFdp = dgdp * (h1 + fx * (h0 - h1)) + gx * (1 - fx) * dhdp
        dFda = gx * (dhda - fx * dhda + dfda * (h0 - h1))
        return np.array([dFdp, dFda]).T


class GPFunctional(MLFunctional):
    # TODO: This setup currently assumes that the gp directly
    # predict F_X - 1. This will not always be the case.

    def __init__(self, kernel, alpha, X_train, desc_list, y_to_f_mul = None):
        """
        desc_type_list should have the l value of each nonlocal
        descriptor, -1 for p, -2 for alpha
        """
        self.ndesc = len(kernel.length_scale)
        #self._y_train_mean = gpr.gp._y_train_mean
        #self._y_train_std = gpr.gp._y_train_std
        self.X_train_ = X_train
        self.alpha_ = alpha
        # assume that k1 is the covariance
        # and that k2 is the noise kernel
        # TODO: take into account the constant kernel
        # in front.
        self.kernel = kernel
        self.desc_list = desc_list
        if y_to_f_mul is not None:
            self.y_to_f_mul, self.y_to_f_mul_deriv = y_to_f_mul
        else:
            self.y_to_f_mul, self.y_to_f_mul_deriv = None, None

    def get_F(self, X, s = None):
        k = self.kernel(X, self.X_train_)
        y_mean = k.dot(self.alpha_)
        #y = y_mean * self._y_train_std + self._y_train_mean
        if self.y_to_f_mul is None:
            return y_mean + 1
        else:
            return (y_mean + 1) * self.y_to_f_mul(s)
        #F = self.y_to_f(y)

    def get_derivative(self, X, s = None, F = None):
        # shape n_test, n_train
        k = self.kernel(X, self.X_train_)
        # X has shape n_test, n_desc
        # X_train_ has shape n_train, n_desc
        ka = k * self.alpha_
        # shape n_test, n_desc
        kaxt = np.dot(ka, self.X_train_)
        kda = np.dot(k, self.alpha_)
        if self.y_to_f_mul is None:
            return (kaxt - X * kda.reshape(-1,1)) / self.kernel.length_scale**2
        else:
            term1 = (kaxt - X * kda.reshape(-1,1)) / self.kernel.length_scale**2
            term1 *= self.y_to_f_mul(s).reshape(-1,1)
            term2 = self.y_to_f_mul_deriv(s) * F
            term1[:,0] += term2
            return term1

    def get_F_and_derivative(self, X):
        return self.get_F(X), self.get_derivative(X)


def get_ref_corr(rho_data_u, rho_data_d, ref_functional):
    cu = eval_xc(ref_functional, (rho_data_u, 0), spin = 1)[0]
    cd = eval_xc(ref_functional, (rho_data_d, 0), spin = 1)[0]


class CorrGPFunctional(GPFunctional):

    def __init__(self, kernel, alpha, X_train, num_desc):
        self.ref_functional = ',MGGA_C_SCAN'
        self.desc_list = [
            Descriptor(1, square, single, mul = 1.0),\
            Descriptor(2, identity, single, mul = 1.0),\
            Descriptor(4, identity, single, mul = 1.0),\
            Descriptor(5, identity, single, mul = 1.0),\
            Descriptor(0, identity, single, mul = 1.0)
        ]
        cov_kernel = kernel.k1
        cov_ss = cov_kernel.k1
        cov_os = cov_kernel.k2
        const_ss = cov_ss.k1.constant_value
        const_os = cov_os.k1.constant_value
        self.alpha_up = const_ss * alpha * X_train[:,0]
        self.alpha_down = const_ss * alpha * X_train[:,num_desc]
        self.alpha_os = const_os * alpha * X_train[:,2*num_desc]
        self.rbf_os = cov_os.k2.k2
        self.rbf_ss = cov_ss.k2.k.k2
        self.X_train = X_train

    def get(self, Xup, Xdown):
        kup = self.rbf_ss(Xup, self.X_train[1:num_desc])
        kdown = self.rbf_ss(Xdown, self.X_train[num_desc+1:2*num_desc])
        kos = self.rbf_os((Xup+Xdown)/2, self.X_train[2*num_desc+1:3*num_desc])
        return np.dot(kup, self.alpha),\
               np.dot(kdown, self.alpha),\
               np.dot(kos, self.alpha)


#import mldftdat.models.map_v2 as mapper_corr

def density_mapper(x1, x2):
    matrix = np.zeros((2, x1.shape[0]))
    dmatrix = np.zeros((2, 2, x1.shape[0]))

    matrix[0] = 1.*np.log(1 + 32.97531959770354*(x1 + x2)**0.3333333333333333 + 53.15594987261972*(x1 + x2)**0.6666666666666666)
    matrix[1] = (x1 - x2)**2/(x1 + x2 + 0.5e-20)**2

    dmatrix[0,0] = (0.20678349696646658 + 0.6666666666666665*(x1 + x2)**0.3333333333333333)/((x1 + x2 + 1e-20)**0.6666666666666666*(0.01881256947522056 + 0.6203504908993999*(x1 + x2)**0.3333333333333333 + 1.*(x1 + x2)**0.6666666666666666))
    dmatrix[0,1] = (0.20678349696646658 + 0.6666666666666665*(x1 + x2)**0.3333333333333333)/((x1 + x2 + 1e-20)**0.6666666666666666*(0.01881256947522056 + 0.6203504908993999*(x1 + x2)**0.3333333333333333 + 1.*(x1 + x2)**0.6666666666666666))
    dmatrix[1,0] = (4*(x1 - x2)*x2)/(x1 + x2 + 0.5e-20)**3
    dmatrix[1,1] = (-4*x1*(x1 - x2))/(x1 + x2 + 0.5e-20)**3

    return matrix, dmatrix

class CorrGPFunctional2(GPFunctional):

    def __init__(self, evaluator):
        self.ref_functional = ',MGGA_C_SCAN'
        self.y_to_f_mul = None
        self.evaluator = evaluator
        self.desc_list = [
            Descriptor(1, square, single, mul = 1.0),\
            Descriptor(2, identity, single, mul = 1.0),\
            Descriptor(4, identity, single, mul = 1.0),\
            Descriptor(5, identity, single, mul = 1.0),\
            Descriptor(8, identity, single, mul = 1.0),\
            Descriptor(12, identity, single, mul = 1.00),\
            Descriptor(6, identity, single, mul = 1.00),\
            Descriptor(15, identity, single, mul = 0.25),\
            Descriptor(16, identity, single, mul = 4.00),\
        ]

    def get_F_and_derivative(self, X, rho_data, compare = None):
        # TODO: The derivative wrt p is incorrect, must take into account
        # spin polarization
        #tmp = ( np.sqrt(X[0][:,0]) + np.sqrt(X[1][:,0]) ) / 2
        X = (X[0] + X[1]) / 2
        #X[:,0] = tmp**2
        rmat, rdmat = density_mapper(rho_data[0][0], rho_data[1][0])
        mat, dmat = mapper.desc_and_ddesc_corr(X.T)
        #if compare is not None:
        #    print(np.linalg.norm(mat.T - compare[:,1:], axis=0))
        tmat = np.append(mat, rmat, axis=0)
        F, dF = self.evaluator.predict_from_desc(tmat.T, vec_eval = True, subind = 0)
        if False:#compare is not None:
            Xinit = compare
            test_desc = self.evaluator.get_descriptors(Xinit[0].T, Xinit[1].T, rho_data[0], rho_data[1], num = self.evaluator.num)
            print('COMPARE', test_desc.shape, np.linalg.norm(test_desc[:,2:] - tmat.T, axis=0))

        FUNCTIONAL = ',MGGA_C_SCAN'
        rho_data_u, rho_data_d = rho_data
        eu, vu = eval_xc(FUNCTIONAL, (rho_data_u, 0 * rho_data_u), spin = 1)[:2]
        ed, vd = eval_xc(FUNCTIONAL, (0 * rho_data_d, rho_data_d), spin = 1)[:2]
        eo, vo = eval_xc(FUNCTIONAL, (rho_data_u, rho_data_d), spin = 1)[:2]
        cu = eu * rho_data_u[0]
        cd = ed * rho_data_d[0]
        co = eo * (rho_data_u[0] + rho_data_d[0])
        co -= cu + cd
        E = (F * co + cu + cd) / (rho_data_u[0] + rho_data_d[0] + 1e-20)
        vo = list(vo)
        for i in range(4):
            j = 2 if i == 1 else 1
            vo[i][:,0] -= vu[i][:,0]
            vo[i][:,j] -= vd[i][:,j]
            vo[i] *= F.reshape(-1,1)
            vo[i][:,0] += vu[i][:,0]
            vo[i][:,j] += vd[i][:,j]

        dFddesc = np.einsum('ni,ijn->nj', dF[:,:-2], dmat)
        #dF[:,-1] = 0
        dFddesc_rho = np.einsum('ni,ijn->nj', dF[:,-2:], rdmat)
        vo[0][:,0] += co * dFddesc_rho[:,0]
        vo[0][:,1] += co * dFddesc_rho[:,1]

        return E, vo, co.reshape(-1,1) * dFddesc


class CorrGPFunctional4(GPFunctional):

    def __init__(self, evaluator):
        self.ref_functional = ',MGGA_C_SCAN'
        self.y_to_f_mul = None
        self.evaluator = evaluator
        self.desc_list = [
            Descriptor(1, square, single, mul = 1.0),\
            Descriptor(2, identity, single, mul = 1.0),\
            Descriptor(4, identity, single, mul = 1.0),\
            Descriptor(5, identity, single, mul = 1.0),\
            Descriptor(8, identity, single, mul = 1.0),\
            Descriptor(12, identity, single, mul = 1.00),\
            Descriptor(6, identity, single, mul = 1.00),\
            Descriptor(15, identity, single, mul = 0.25),\
            Descriptor(16, identity, single, mul = 4.00),\
            Descriptor(13, identity, single, mul = 1.00)
        ]

    def get_F_and_derivative(self, X, rho_data, compare = None):
        # TODO: The derivative wrt p is incorrect, must take into account
        # spin polarization
        #tmp = ( np.sqrt(X[0][:,0]) + np.sqrt(X[1][:,0]) ) / 2
        X = (X[0] + X[1]) / 2
        #X[:,0] = tmp**2
        rmat, rdmat = density_mapper(rho_data[0][0], rho_data[1][0])
        mat, dmat = mapper.desc_and_ddesc(X.T)
        #if compare is not None:
        #    print(np.linalg.norm(mat.T - compare[:,1:], axis=0))
        tmat = np.append(mat, rmat[:1], axis=0)
        F, dF = self.evaluator.predict_from_desc(tmat.T, vec_eval = True, subind = 0)
        if compare is not None:
            Xinit = compare
            test_desc = self.evaluator.get_descriptors(Xinit[0].T, Xinit[1].T, rho_data[0], rho_data[1], num = self.evaluator.num)
            print('COMPARE', test_desc.shape, np.linalg.norm(test_desc[:,2:] - tmat.T, axis=0))

        FUNCTIONAL = ',MGGA_C_SCAN'
        rho_data_u, rho_data_d = rho_data
        eu, vu = eval_xc(FUNCTIONAL, (rho_data_u, 0 * rho_data_u), spin = 1)[:2]
        ed, vd = eval_xc(FUNCTIONAL, (0 * rho_data_d, rho_data_d), spin = 1)[:2]
        eo, vo = eval_xc(FUNCTIONAL, (rho_data_u, rho_data_d), spin = 1)[:2]
        cu = eu * rho_data_u[0]
        cd = ed * rho_data_d[0]
        co = eo * (rho_data_u[0] + rho_data_d[0])
        co -= cu + cd
        E = (F * co + cu + cd) / (rho_data_u[0] + rho_data_d[0] + 1e-20)
        vo = list(vo)
        for i in range(4):
            j = 2 if i == 1 else 1
            vo[i][:,0] -= vu[i][:,0]
            vo[i][:,j] -= vd[i][:,j]
            vo[i] *= F.reshape(-1,1)
            vo[i][:,0] += vu[i][:,0]
            vo[i][:,j] += vd[i][:,j]

        dFddesc = np.einsum('ni,ijn->nj', dF[:,:-1], dmat)
        #dF[:,-1] = 0
        dFddesc_rho = np.einsum('ni,ijn->nj', dF[:,-1:], rdmat[:1,:])
        vo[0][:,0] += co * dFddesc_rho[:,0]
        vo[0][:,1] += co * dFddesc_rho[:,1]

        return E, vo, co.reshape(-1,1) * dFddesc


class CorrGPFunctional5(GPFunctional):

    def __init__(self, evaluator):
        self.ref_functional = ',MGGA_C_SCAN'
        self.y_to_f_mul = None
        self.evaluator = evaluator
        self.desc_list = [
            Descriptor(1, square, single, mul = 1.0),\
            Descriptor(2, identity, single, mul = 1.0),\
            Descriptor(4, identity, single, mul = 1.0),\
            Descriptor(5, identity, single, mul = 1.0),\
            Descriptor(8, identity, single, mul = 1.0),\
            Descriptor(12, identity, single, mul = 1.00),\
            Descriptor(6, identity, single, mul = 1.00),\
            Descriptor(15, identity, single, mul = 0.25),\
            Descriptor(16, identity, single, mul = 4.00),\
            Descriptor(13, identity, single, mul = 1.00)
        ]

    def get_F_and_derivative(self, X, rho_data, compare = None):
        # TODO: The derivative wrt p is incorrect, must take into account
        # spin polarization
        #tmp = ( np.sqrt(X[0][:,0]) + np.sqrt(X[1][:,0]) ) / 2
        amat, admat = mapper.desc_and_ddesc(X[0].T)
        bmat, bdmat = mapper.desc_and_ddesc(X[1].T)
        ramat, radmat = density_mapper(2 * rho_data[0][0], 0 * rho_data[0][0])
        rbmat, rbdmat = density_mapper(2 * rho_data[1][0], 0 * rho_data[1][0])
        ssind = np.array([0,2,6])
        amat, admat = amat[ssind], admat[ssind[:,None],ssind,:]
        ramat, radmat = ramat[:1], 2 * radmat[0,0]
        bmat, bdmat = bmat[ssind], bdmat[ssind[:,None],ssind,:]
        rbmat, rbdmat = rbmat[:1], 2 * rbdmat[0,0]

        X = (X[0] + X[1]) / 2
        #X[:,0] = tmp**2
        rmat, rdmat = density_mapper(rho_data[0][0], rho_data[1][0])
        mat, dmat = mapper.desc_and_ddesc(X.T)
        #if compare is not None:
        #    print(np.linalg.norm(mat.T - compare[:,1:], axis=0))
        tmat = np.concatenate([mat, rmat], axis=0)
        F, dF = self.evaluator.eval_os.predict_from_desc(tmat.T, vec_eval = True)
        tmat = np.concatenate([amat, ramat], axis=0)
        Fu, dFu = self.evaluator.eval_ss.predict_from_desc(tmat.T, vec_eval = True)
        tmat = np.concatenate([bmat, rbmat], axis=0)
        Fd, dFd = self.evaluator.eval_ss.predict_from_desc(tmat.T, vec_eval = True)
        #if compare is not None:
        #    Xinit = compare
        #    test_desc = self.evaluator.get_descriptors(Xinit[0].T, Xinit[1].T, rho_data[0], rho_data[1], num = self.evaluator.num)
        #    print('COMPARE', test_desc.shape, np.linalg.norm(test_desc[:,2:] - tmat.T, axis=0))

        FUNCTIONAL = ',MGGA_C_SCAN'
        rho_data_u, rho_data_d = rho_data
        eu, vu = eval_xc(FUNCTIONAL, (rho_data_u, 0 * rho_data_u), spin = 1)[:2]
        ed, vd = eval_xc(FUNCTIONAL, (0 * rho_data_d, rho_data_d), spin = 1)[:2]
        eo, vo = eval_xc(FUNCTIONAL, (rho_data_u, rho_data_d), spin = 1)[:2]
        cu = eu * rho_data_u[0]
        cd = ed * rho_data_d[0]
        co = eo * (rho_data_u[0] + rho_data_d[0])
        co -= cu + cd
        E = (F * co + Fu * cu + Fd * cd) / (rho_data_u[0] + rho_data_d[0] + 1e-20)
        vo = list(vo)
        for i in range(4):
            j = 2 if i == 1 else 1
            vo[i][:,0] -= vu[i][:,0]
            vo[i][:,j] -= vd[i][:,j]
            vo[i] *= F.reshape(-1,1)
            #print(vo[i][:,0].shape, vu[i][:,0].shape, Fu.shape)
            vo[i][:,0] += vu[i][:,0] * Fu
            vo[i][:,j] += vd[i][:,j] * Fd

        dEddesc = co.reshape(-1,1) * np.einsum('ni,ijn->nj', dF[:,:-2], dmat)
        print(admat.shape, bdmat.shape)
        dEddesc[:,ssind] += cu.reshape(-1,1) * np.einsum('ni,ijn->nj', dFu[:,:-1], admat)
        dEddesc[:,ssind] += cd.reshape(-1,1) * np.einsum('ni,ijn->nj', dFd[:,:-1], bdmat)
        #dF[:,-1] = 0
        dFddesc_rho = np.einsum('ni,ijn->nj', dF[:,-2:], rdmat)
        vo[0][:,0] += co * dFddesc_rho[:,0]
        vo[0][:,0] += cu * dF[:,-1] * radmat
        vo[0][:,1] += co * dFddesc_rho[:,1]
        vo[0][:,1] += cd * dF[:,-1] * rbdmat

        return E, vo, dEddesc


import mldftdat.models.map_v1 as mapper

class NormGPFunctional(GPFunctional):

    def __init__(self, evaluator, normp = True):
        self.evaluator = evaluator
        self.y_to_f_mul = None
        self.desc_list = [
            Descriptor(1, square, single, mul = 1.0),\
            Descriptor(2, identity, single, mul = 1.0),\
            Descriptor(4, identity, single, mul = 1.0),\
            Descriptor(5, identity, single, mul = 1.0),\
            Descriptor(8, identity, single, mul = 1.0),\
            Descriptor(12, identity, single, mul = 1.00),\
            Descriptor(6, identity, single, mul = 1.00),\
            Descriptor(15, identity, single, mul = 0.25),\
            Descriptor(16, identity, single, mul = 4.00),\
            Descriptor(13, identity, single, mul = 1.00),\
        ]
        self.normp = normp

    def get_F_and_derivative(self, X, compare = None):
        mat, dmat = mapper.desc_and_ddesc(X.T, normp = self.normp)
        if compare is not None:
            print(np.linalg.norm(mat.T - compare[:,1:], axis=0))
        F, dF = self.evaluator.predict_from_desc(mat.T, vec_eval = True, subind = 1)
        dFddesc = np.einsum('ni,ijn->nj', dF, dmat)
        return F + 1, dFddesc

    def get_F(self, X):
        return self.get_F_and_derivative(self, X)[0]

    def get_derivative(self, X):
        return self.get_F_and_derivative(self, X)[1]

