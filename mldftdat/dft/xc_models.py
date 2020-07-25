from abc import ABC, abstractmethod, abstractproperty
import numpy as np

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

import mldftdat.models.map_v1 as mapper

class NormGPFunctional(GPFunctional):

    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.y_to_f_mul = None
        self.desc_list = desc_list = [
            Descriptor(1, square, single, mul = 1.0),\
            Descriptor(2, identity, single, mul = 1.0),\
            Descriptor(4, identity, single, mul = 1.0),\
            Descriptor(5, identity, single, mul = 1.0),\
            Descriptor(8, identity, single, mul = 1.0),\
            Descriptor(6, identity, single, mul = 1.00),\
            Descriptor(12, identity, single, mul = 1.00),\
            Descriptor(15, identity, single, mul = 0.25),\
            Descriptor(16, identity, single, mul = 4.00),\
            Descriptor(13, identity, single, mul = 1.00),\
        ]

    def get_F_and_derivative(self, X):
        mat, dmat = mapper.desc_and_ddesc(X.T)
        F, dF = self.evaluator.predict_from_desc(mat.T, vec_eval = True, subind = 1)
        dFddesc = np.einsum('ni,ijn->nj', dF, dmat)
        return F, dFddesc

    def get_F(self, X):
        return self.get_F_and_derivative(self, X)[0]

    def get_derivative(self, X):
        return self.get_F_and_derivative(self, X)[1]
