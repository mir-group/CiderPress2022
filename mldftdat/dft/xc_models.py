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


class RBFFunctional(MLFunctional):
    # TODO: This setup currently assumes that the gp directly
    # predict F_X - 1. This will not always be the case.

    def __init__(self, gpr):
        # Assumes kernel_ is (const * rbf) + noise
        cov = gpr.gp.kernel_.k1
        self.alpha_ = cov.k1.constant_value * gpr.gp.alpha_
        self.kernel = RBF(length_scale=cov.ks.length_scale)
        self.X_train = gpr.gp.X_train_[:,1:]
        self.feature_list = gpr.feature_list
        self.nfeat = self.feature_list.nfeat
        self.fx_baseline = gpr.args.xed_y_converter[2]
        self.fxb_num = gpr.args.xed_y_converter[3]

    def get_F_and_derivative(self, X):
        mat = np.zeros((self.nfeat, X.shape[1]))
        self.feature_list.fill_vals_(mat, X)

        k = self.kernel(X, self.X_train_)
        F = k.dot(self.alpha_)

        # X has shape n_test, n_desc
        # X_train_ has shape n_train, n_desc
        ka = k * self.alpha_
        # shape n_test, n_desc
        kaxt = np.dot(ka, self.X_train_)
        kda = np.dot(k, self.alpha_)
        dF = (kaxt - X * kda.reshape(-1,1)) / self.kernel.length_scale**2

        dFddesc = np.zeros(X.shape)
        self.feature_list.fill_derivs_(dFddesc, dF, X)

        if rho is not None:
            highcut = 1e-3
            ecut = 1.0/2
            lowcut = 1e-6
            F[rho<highcut] *= 0.5 * (1 - np.cos(np.pi * (rho[rho<highcut] \
                                                / highcut)**ecut))
            dFddesc[rho<highcut,:] *= 0.5 * \
                (1 - np.cos(np.pi * (rho[rho<highcut,np.newaxis] / highcut)**ecut))
            dFddesc[rho<lowcut,:] = 0

        if self.fxb_num == 1:
            chfx = 1
        elif self.fxb_num == 2:
            chfx, dchfx = self.fx_baseline(X[:,1])
            dFddesc[:,1] += dchfx
        else:
            raise ValueError('Unsupported basline fx order.')
        F += chfx
    
        if rho is not None:
            F[rho<1e-9] = 0
            dFddesc[rho<1e-9,:] = 0

        return F, dFddesc

    def get_F(self, X):
        return self.get_F_and_derivative(X)[0]

    def get_derivative(self, X):
        return self.get_F_and_derivative(X)[1]


class NormGPFunctional(GPFunctional):

    def __init__(self, evaluator):
        # For use with evaluators generated using gp_to_spline.py
        self.evaluator = evaluator
        self.desc_order = evaluator.desc_order
        self.fxb_num = evaluator.fxb_num
        self.fx_baseline = evaluator.fx_baseline
        self.nfeat = evaluator.feature_list.nfeat
        self.feature_list = evaluator.feature_list

    def get_F_and_derivative(self, X):
        mat = np.zeros((self.nfeat, X.shape[1]))
        self.feature_list.fill_vals_(mat, X)
        F, dF = self.evaluator.predict_from_desc(mat.T, vec_eval=True)
        dFddesc = np.zeros(X.shape)
        self.feature_list.fill_derivs_(dFddesc, dF, X)

        if rho is not None:
            highcut = 1e-3
            ecut = 1.0/2
            lowcut = 1e-6
            F[rho<highcut] *= 0.5 * (1 - np.cos(np.pi * (rho[rho<highcut] \
                                                / highcut)**ecut))
            dFddesc[rho<highcut,:] *= 0.5 * \
                (1 - np.cos(np.pi * (rho[rho<highcut,np.newaxis] / highcut)**ecut))
            dFddesc[rho<lowcut,:] = 0

        if self.fxb_num == 1:
            chfx = 1
        elif self.fxb_num == 2:
            chfx, dchfx = self.fx_baseline(X[:,1])
            dFddesc[:,1] += dchfx
        else:
            raise ValueError('Unsupported basline fx order.')
        F += chfx
    
        if rho is not None:
            F[rho<1e-9] = 0
            dFddesc[rho<1e-9,:] = 0

        return F, dFddesc

    def get_F(self, X):
        return self.get_F_and_derivative(X)[0]

    def get_derivative(self, X):
        return self.get_F_and_derivative(X)[1]
