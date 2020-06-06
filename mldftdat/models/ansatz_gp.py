from interpolation.splines import UCGrid, CGrid, nodes
from interpolation.splines import eval_linear
from mldftdat.gp import DFTGPR
from mldftdat.density import *
from mldftdat.data import *
#from mldftdat.models.matrix_rbf import *
import numpy as np
from pyscf.dft.libxc import eval_xc
from sklearn.gaussian_process.kernels import *
from sklearn.gaussian_process.kernels import _num_samples

A = 0.704 # maybe replace with sqrt(6/5)?
B = 2 * np.pi / 9 * np.sqrt(6.0/5)
FXP0 = 27 / 50 * 10 / 81
FXI = 1.0 / 3 * (4*np.pi**2 / 3)**(1.0/3)
#MU = 10/81
MU = 0.21
C1 = 1.0 / 3 * (4*np.pi**2 / 3)**(1.0/3)
C2 = 1 - C1
C3 = 0.19697 * np.sqrt(0.704)
C4 = (C3**2 - 0.09834 * MU) / C3**3

def edmgga_from_q(Q):
    x = A * Q + np.sqrt(1 + (A*Q)**2)
    FX = C1 + (C2 * x) / (1 + C3 * np.sqrt(x) * np.arcsinh(C4 * (x-1)))
    return FX

xref = np.linspace(1e-4, 8, 4000)
Qref = np.sinh(np.log(np.sinh(xref)))
Fref = edmgga_from_q(Qref)
grid = CGrid(Fref)

def f_to_x(f):
    f = np.maximum(f, 0.81)
    return eval_linear(grid, xref, f.reshape(-1,1))

def x_to_q(x):
    return np.sinh(np.log(np.sinh(x)))

def xed_to_y_q(xed, rho_data):
    F = get_y_from_xed(xed, rho_data[0]) + 1.0
    return x_to_q(f_to_x(F))

def y_to_xed_q(y, rho_data):
    F = edmgga_from_q(y)
    return get_xed_from_y(F - 1, rho_data[0])

def get_edmgga_descriptors(X, num=1):
    X[:,1] = X[:,1]**2
    return X[:,(1,2,3,4,5,8,6,12,15,16,13,14)[:num]]

def get_descriptors(X, num = 1):
    #X[:,1] = X[:,1]**2
    #return X[:,(1,2,3,4,5,8,6,12,15,16,13,14)[:num]]
    fac = (6 * np.pi**2)**(2.0/3) / (16 * np.pi)
    X[:,1] = X[:,1]**2
    p = X[:,1]
    alpha = X[:,2]
    nabla = X[:,3]
    scale = np.sqrt(1 + fac * p + 0.6 * fac * (alpha - 1)) # 4^(1/3) for 16, 1/(4)^(1/3) for 15
    desc = np.zeros((X.shape[0], 27))
    desc[:,0] = X[:,4] * scale
    desc[:,1] = X[:,4] * scale**3
    desc[:,2] = X[:,4] * scale**5
    desc[:,3] = X[:,5] * scale
    desc[:,4] = X[:,5] * scale**3
    desc[:,5] = X[:,5] * scale**5
    desc[:,6] = np.sqrt(X[:,8]) * scale
    desc[:,7] = np.sqrt(X[:,8]) * scale**3
    desc[:,8] = np.sqrt(X[:,8]) * scale**5
    desc[:,9] = X[:,15] * scale
    desc[:,10] = X[:,15] * scale**3
    desc[:,11] = X[:,15] * scale**5
    desc[:,12] = X[:,16] * scale
    desc[:,13] = X[:,16] * scale**3
    desc[:,14] = X[:,16] * scale**5
    desc[:,15] = p**2
    desc[:,16] = p * alpha
    desc[:,17] = alpha**2
    desc[:,18:21] = desc[:,0:3]**2
    desc[:,21:24] = desc[:,9:12]**2
    desc[:,24:27] = desc[:,12:15]**2
    return np.append(X[:,1:4], desc, axis=1)[:,:num]


def get_rho_and_edmgga_descriptors(X, rho_data, num=1, xed=None):
    X = get_descriptors(X, num)
    if xed is None:
        X = np.append(np.ones(X.shape[0]).reshape(-1,1), X, axis=1)
    else:
        X = np.append(get_y_from_xed(xed, rho_data[0]).reshape(-1,1), X, axis=1)
    return X


class PartialDot(DotProduct):

    def __init__(self, sigma_0=1.0, sigma_0_bounds=(1e-5, 1e5), start = 0):
        super(PartialDot, self).__init__(sigma_0, sigma_0_bounds)
        self.start = start

    def __call__(self, X, Y=None, eval_gradient=False):

        X = X[:,self.start:]
        if Y is not None:
            Y = Y[:,self.start:]
        return super(PartialDot, self).__call__(X, Y, eval_gradient)


class DerivNoise(StationaryKernelMixin, GenericKernelMixin,
                   Kernel):

    def __init__(self, interval = 0.006, index = 0):
        self.interval = interval
        self.index = index

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is not None and eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")

        if Y is None:
            K = np.diag(self.diag(X))
            if eval_gradient:
                return K, np.empty((_num_samples(X), _num_samples(X), 0))
            else:
                return K
        else:
            return np.zeros((_num_samples(X), _num_samples(Y)))

    def diag(self, X):
        F = X[:,self.index]
        return (x_to_q(f_to_x(F + self.interval)) - x_to_q(f_to_x(F)))**2

    def __repr__(self):
        return "{0}".format(self.__class__.__name__)


class FeatureNoise(StationaryKernelMixin, GenericKernelMixin,
                   Kernel):

    def __init__(self, index = 0):
        self.index = index

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is not None and eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")

        if Y is None:
            K = np.diag(self.diag(X))
            if eval_gradient:
                return K, np.empty((_num_samples(X), _num_samples(X), 0))
            else:
                return K
        else:
            return np.zeros((_num_samples(X), _num_samples(Y)))

    def diag(self, X):
        return x_to_q(f_to_x(X[:,self.index]))**2

    def __repr__(self):
        return "{0}".format(self.__class__.__name__)


class AnsatzGPR(DFTGPR):

    def __init__(self, num_desc, use_algpr = False):
        #dot = Exponentiation(PartialDot(start = 1), 2)
        dot = PartialDot(start = 1)
        dn = DerivNoise()
        fn = FeatureNoise()
        wk = WhiteKernel(noise_level=1.0e-4, noise_level_bounds=(1e-06, 1.0e5))
        wkf = WhiteKernel(noise_level = 0.004, noise_level_bounds=(1e-05, 0.1))
        wkd = WhiteKernel(noise_level = 1.0, noise_level_bounds=(1e-05, 1.0e5))
        cov_kernel = dot
        noise_kernel = wk + dn + fn * wkf
        init_kernel = cov_kernel + noise_kernel
        super(AnsatzGPR, self).__init__(num_desc,
                       descriptor_getter = get_rho_and_edmgga_descriptors,
                       xed_y_converter = (xed_to_y_q, y_to_xed_q),
                       init_kernel = init_kernel, use_algpr = use_algpr)

    def fit(self, xdesc, xed, rho_data, optimize_theta = True):
        if optimize_theta:
            optimizer = 'fmin_l_bfgs_b'
        else:
            optimizer = None
        self.gp.optimizer = optimizer
        self.X = self.get_descriptors(xdesc, rho_data, num=self.num, xed=xed)
        self.y = self.xed_to_y(xed, rho_data)
        print(np.isnan(self.X).sum(), np.isnan(self.y).sum())
        print(self.X.shape, self.y.shape)
        self.gp.fit(self.X, self.y)
        self.gp.kernel = self.gp.kernel_

    def scores(self, xdesc, xed_true, rho_data):
        # Returns
        # r^2 of the model itself
        # rmse of model
        # rmse of exchange energy density
        # relative rmse of exchange energy density
        # score of exchange energy density
        X_test = self.get_descriptors(xdesc, rho_data, num=self.num)
        y_true = self.xed_to_y(xed_true, rho_data)
        y_pred = self.gp.predict(X_test)
        if len(rho_data.shape) == 2:
            rho = rho_data[0]
        else:
            rho = rho_data
        xed_pred = self.y_to_xed(y_pred, rho_data)
        model_score = score(y_true, y_pred)
        model_rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        xed_rmse = np.sqrt(np.mean((xed_true - xed_pred)**2 / rho**2))
        xed_rel_rmse = np.sqrt(np.mean(((xed_true - xed_pred) / (xed_true + 1e-7))**2))
        xed_score = score(xed_true / rho, xed_pred / rho)
        return model_score, model_rmse, xed_rmse, xed_rel_rmse, xed_score

    def predict(self, X, rho_data, return_std = False):
        X = self.get_descriptors(X, rho_data, num=self.num)
        y = self.gp.predict(X, return_std = return_std)
        if return_std:
            return self.y_to_xed(y[0], rho_data), y[1] * ldax(rho_data[0])
        else:
            return self.y_to_xed(y, rho_data) 

    def add_point(self, xdesc, xed, rho_data, threshold_factor = 1.2):
        x = self.get_descriptors(xdesc, rho_data, num=self.num)
        y = self.xed_to_y(xed, rho_data)
        if self.is_uncertain(x, y, threshold_factor):
            self.X = np.append(self.X, x, axis=0)
            self.y = np.append(self.y, y)
            if self.al:
                self.gp.fit_single(x, y)
            else:
                prev_optimizer = self.gp.optimizer
                self.gp.optimizer = None
                self.gp.fit(self.X, self.y)
                self.gp.optimizer = prev_optimizer

    def is_uncertain(self, x, y, threshold_factor = 1.2, low_noise_bound = 0.002):
        threshold = max(low_noise_bound, np.sqrt(self.gp.kernel_.k2(x))) * threshold_factor
        y_pred = self.gp.predict(x)
        return np.abs(y - y_pred) > threshold
