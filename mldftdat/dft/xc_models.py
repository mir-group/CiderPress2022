from abc import ABC, abstractmethod, abstractproperty
import numpy as np
from pyscf.dft.libxc import eval_xc
import numpy as np
import yaml
from interpolation.splines import UCGrid, CGrid, nodes
from interpolation.splines import filter_cubic, eval_cubic
from interpolation.splines.eval_cubic_numba import vec_eval_cubic_splines_G_1,\
                                                   vec_eval_cubic_splines_G_2,\
                                                   vec_eval_cubic_splines_G_3,\
                                                   vec_eval_cubic_splines_G_4



def get_vec_eval(grid, coeffs, X, N):
    coeffs = np.expand_dims(coeffs, coeffs.ndim)
    y = np.zeros((X.shape[0], 1))
    dy = np.zeros((X.shape[0], N, 1))
    a_, b_, orders = zip(*grid)
    if N == 1:
        vec_eval_cubic_splines_G_1(a_, b_, orders,
                                   coeffs, X, y, dy)
    elif N == 2:
        vec_eval_cubic_splines_G_2(a_, b_, orders,
                                   coeffs, X, y, dy)
    elif N == 3:
        vec_eval_cubic_splines_G_3(a_, b_, orders,
                                   coeffs, X, y, dy)
    elif N == 4:
        vec_eval_cubic_splines_G_4(a_, b_, orders,
                                   coeffs, X, y, dy)
    else:
        raise ValueError('invalid dimension N')
    return np.squeeze(y, -1), np.squeeze(dy, -1)


class Evaluator():

    def __init__(self, scale, ind_sets, spline_grids, coeff_sets,
                 xed_y_converter, feature_list, desc_order, const=0,
                 desc_version='c', a0=8.0, fac_mul=0.25, amin=0.05):
        self.scale = scale
        self.nterms = len(self.scale)
        self.ind_sets = ind_sets
        self.spline_grids = spline_grids
        self.coeff_sets = coeff_sets
        self.xed_y_converter = xed_y_converter
        self.feature_list = feature_list
        self.nfeat = feature_list.nfeat
        self.desc_order = desc_order
        self.const = const
        self.desc_version = desc_version
        self.a0 = a0
        self.fac_mul = fac_mul
        self.amin = amin

    def _xedy(self, y, x, code):
        if self.xed_y_converter[-1] == 1:
            return self.xed_y_converter[code](y, x[:,0])
        elif self.xed_y_converter[-1] == 2:
            return self.xed_y_converter[code](y, x[:,0], x[:,1])
        else:
            return self.xed_y_converter[code](y)

    def xed_to_y(self, y, x):
        return self._xedy(y, x, 0)

    def y_to_xed(self, y, x):
        return self._xedy(y, x, 1)

    def get_descriptors(self, x):
        return self.feature_list(x[:,self.desc_order])

    def predict_from_desc(self, X, vec_eval=False,
                          max_order=3, min_order=0):
        res = np.zeros(X.shape[0]) + self.const
        if vec_eval:
            dres = np.zeros(X.shape)
        for t in range(self.nterms):
            if type(self.ind_sets[t]) != int and len(self.ind_sets[t]) < min_order:
                continue
            if (type(self.ind_sets[t]) == int) or len(self.ind_sets[t]) <= max_order:
                ind_set = self.ind_sets[t]
                if vec_eval:
                    y, dy = get_vec_eval(self.spline_grids[t],
                                         self.coeff_sets[t],
                                         X[:,ind_set],
                                         len(ind_set))
                    res += y * self.scale[t]
                    dres[:,ind_set] += dy * self.scale[t]
                else:
                    res += eval_cubic(self.spline_grids[t],
                                      self.coeff_sets[t],
                                      X[:,ind_set])\
                           * self.scale[t]
        if vec_eval:
            return res, dres
        else:
            return res

    def predict(self, X, vec_eval=False):
        desc = self.get_descriptors(X)
        F = self.predict_from_desc(desc, vec_eval=vec_eval)
        if vec_eval:
            F = F[0]
        return self.y_to_xed(F, X)

    def dump(self, fname):
        state_dict = {
                        'scale': np.array(self.scale),
                        'ind_sets': self.ind_sets,
                        'spline_grids': self.spline_grids,
                        'coeff_sets': self.coeff_sets,
                        'xed_y_converter': self.xed_y_converter,
                        'feature_list': self.feature_list,
                        'desc_order': self.desc_order,
                        'const': self.const,
                        'desc_version': self.desc_version,
                        'a0': self.a0,
                        'fac_mul': self.fac_mul,
                        'amin': self.amin
                     }
        with open(fname, 'w') as f:
            yaml.dump(state_dict, f)

    @classmethod
    def load(cls, fname):
        with open(fname, 'r') as f:
            state_dict = yaml.load(f, Loader=yaml.Loader)
        return cls(state_dict['scale'],
                   state_dict['ind_sets'],
                   state_dict['spline_grids'],
                   state_dict['coeff_sets'],
                   state_dict['xed_y_converter'],
                   state_dict['feature_list'],
                   state_dict['desc_order'],
                   const=state_dict['const'],
                   desc_version=state_dict['desc_version'],
                   a0=state_dict['a0'],
                   fac_mul=state_dict['fac_mul'],
                   amin=state_dict['amin']
                  )




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

    def get_F(self, X):
        return self.get_F_and_derivative(X)[0]

    def get_derivative(self, X):
        return self.get_F_and_derivative(X)[1]

    @abstractmethod
    def get_F_and_derivative(self, X):
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
        
        """
        if rho is not None:
            highcut = 1e-3
            ecut = 1.0/2
            lowcut = 1e-6
            F[rho<highcut] *= 0.5 * (1 - np.cos(np.pi * (rho[rho<highcut] \
                                                / highcut)**ecut))
            dFddesc[rho<highcut,:] *= 0.5 * \
                (1 - np.cos(np.pi * (rho[rho<highcut,np.newaxis] / highcut)**ecut))
            dFddesc[rho<lowcut,:] = 0
        """

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


class NormGPFunctional(MLFunctional,Evaluator):

    def __init__(self, *args, **kwargs):
        Evaluator.__init__(self, *args, **kwargs)
        self.fxb_num = self.xed_y_converter[3]
        self.fx_baseline = self.xed_y_converter[2]

    def get_F_and_derivative(self, X):
        rho = X[0]
        mat = np.zeros((self.nfeat, X.shape[1]))
        self.feature_list.fill_vals_(mat, X)
        F, dF = self.predict_from_desc(mat.T, vec_eval=True)
        dFddesc = np.zeros(X.shape).T
        self.feature_list.fill_derivs_(dFddesc.T, dF.T, X)
        
        if rho is not None:
            highcut = 1e-3
            lowcut = 1e-6
            rhocut = np.maximum(rho[rho<highcut], lowcut)
            xcut = np.log(rhocut / lowcut) / np.log(highcut / lowcut)
            ##dFddesc[rho<highcut,0] = F[rho<highcut] * np.pi * np.sin(np.pi * xcut)\
            ##                         / (2 * rhocut * np.log(highcut / lowcut))
            F[rho<highcut] *= 0.5 * (1 - np.cos(np.pi * xcut))
            dFddesc[rho<highcut,:] *= 0.5 * (1 - np.cos(np.pi * xcut[:,np.newaxis]))
            dFddesc[rho<lowcut,:] = 0
        
        if self.fxb_num == 1:
            chfx = 1
        elif self.fxb_num == 2:
            chfx, dchfx = self.fx_baseline(X[1])
            dFddesc[:,1] += dchfx
        else:
            raise ValueError('Unsupported basline fx order.')
        F += chfx
    
        if rho is not None:
            F[rho<1e-9] = 0
            dFddesc[rho<1e-9,:] = 0

        return F, dFddesc


