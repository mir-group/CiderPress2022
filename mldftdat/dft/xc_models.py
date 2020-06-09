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
    return x

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
                 transform_deriv = identity, mul = 1.0):
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

class GPFunctional(MLFunctional):
    # TODO: This setup currently assumes that the gp directly
    # predict F_X - 1. This will not always be the case.

    def __init__(self, kernel, alpha, X_train, desc_list, y_to_f):
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
        self.y_to_f = y_to_f

    def get_F(self, X):
        k = self.kernel(X, self.X_train_)
        y_mean = k.dot(self.alpha_)
        #y = y_mean * self._y_train_std + self._y_train_mean
        return y_mean + 1
        #F = self.y_to_f(y)

    def get_derivative(self, X):
        # shape n_test, n_train
        k = self.kernel(X, self.X_train_)
        # X has shape n_test, n_desc
        # X_train_ has shape n_train, n_desc
        ka = k * self.alpha_
        # shape n_test, n_desc
        kaxt = np.dot(ka, self.X_train_)
        kda = np.dot(k, self.alpha_)
        print(kaxt.shape, kda.shape)
        return (kaxt - X * kda.reshape(-1,1)) / self.kernel.length_scale**2

    """
    def get_eps(self, X, rho_data):
        return LDA_FACTOR * self.get_F(X) * rho_data[0]**(1.0/3)

    def get_potential(self, X, gp_deriv, rho_data):
        v_npa = np.zeros(4, rho_data.shape[1])
        F = self.get_F(X)
        dgpdp = np.zeros(rho_data.shape[1])
        dgpda = np.zeros(rho_data.shape[1])
        for i, l in enumerate(self.desc_type_list):
            if l == -1:
                dgpdp += gp_deriv[:,i]
            elif l == -2:
                dgpda += gp_deriv[:,i]
            else:
                v_npa += v_nonlocal(rho_data, grid, gp_deriv[:,i],
                                    ao_to_aux, rdm1, auxmol, g, l = l,
                                    mul = self.muls[i])
        v_npa += v_semilocal(rho_data, F, dgpdp, dgpda)
        return v_basis_transform(rho_data, v_npa)
    """     

