from mldftdat.gp import DFTGPR
from mldftdat.density import *
from mldftdat.data import *
from mldftdat.models.matrix_rbf import *
import numpy as np
from pyscf.dft.libxc import eval_xc
from sklearn.gaussian_process.kernels import *

def xed_to_y_edmgga(xed, rho_data):
    y = xed / (ldax(rho_data[0]) - 1e-7)
    return y - edmgga(rho_data)

def y_to_xed_edmgga(y, rho_data):
    fx = np.array(y) + edmgga(rho_data)
    return fx * ldax(rho_data[0])

def xed_to_y_scan(xed, rho_data):
    pbex = eval_xc('SCAN,', rho_data)[0] * rho_data[0]
    return (xed - pbex) / (ldax(rho_data[0]) - 1e-7)

def y_to_xed_scan(y, rho_data):
    yp = y * ldax(rho_data[0])
    pbex = eval_xc('SCAN,', rho_data)[0] * rho_data[0]
    return yp + pbex

def xed_to_y_pbe(xed, rho_data):
    pbex = eval_xc('PBE,', rho_data)[0] * rho_data[0]
    return (xed - pbex) / (ldax(rho_data[0]) - 1e-7)

def y_to_xed_pbe(y, rho_data):
    yp = y * ldax(rho_data[0])
    pbex = eval_xc('PBE,', rho_data)[0] * rho_data[0]
    return yp + pbex

def xed_to_y_lda(xed, rho_data):
    return get_y_from_xed(xed, rho_data[0])

def y_to_xed_lda(y, rho_data):
    return get_xed_from_y(y, rho_data[0])

def get_edmgga_descriptors(X, rho_data, num=1):
    gradn = np.linalg.norm(rho_data[1:4], axis=0)
    tau0 = get_uniform_tau(rho_data[0]) + 1e-6
    tauw = get_single_orbital_tau(rho_data[0], gradn)
    QB = tau0 - rho_data[5] + tauw + 0.25 * rho_data[4]
    QB /= tau0
    x = A * QB + np.sqrt(1 + (A*QB)**2)
    x = np.arcsinh(x - 1)
    X = get_gp_x_descriptors(X, num = num)
    if num > 2:
        c = X[:,2]
        ndvh2 = rho_data[5] * c * 5e-4 / (1 - c + 1e-7)
        c2 = ndvh2 / (ndvh2 + rho_data[5] + 1e-7) - 0.5
        X[:,2] = c2
    X = np.append(x.reshape(-1,1), X, axis=1)
    return X
    #return X[:,(0,3,4)]

class PBEGPR(DFTGPR):

    def __init__(self, num_desc, init_kernel = None, use_algpr = False):
        super(PBEGPR, self).__init__(num_desc, descriptor_getter = None,
                       xed_y_converter = (xed_to_y_pbe, y_to_xed_pbe),
                       init_kernel = init_kernel, use_algpr = use_algpr)


class EDMGPR(DFTGPR):

    def __init__(self, num_desc, init_kernel = None, use_algpr = False):
        super(EDMGPR, self).__init__(num_desc, descriptor_getter = get_edmgga_descriptors,
                       xed_y_converter = (xed_to_y_pbe, y_to_xed_pbe),
                       init_kernel = init_kernel, use_algpr = use_algpr)

    def fit(self, xdesc, xed, rho_data, optimize_theta = True):
        if optimize_theta:
            optimizer = 'fmin_l_bfgs_b'
        else:
            optimizer = None
        self.gp.optimizer = optimizer
        self.X = self.get_descriptors(xdesc, rho_data, num=self.num)
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


def get_rho_and_edmgga_descriptors(X, rho_data, num=1):
    X = get_edmgga_descriptors(X, rho_data, num)
    X = np.append(rho_data[0].reshape(-1,1), X, axis=1)
    return X


class NoisyEDMGPR(EDMGPR):

    def __init__(self, num_desc, use_algpr = False):
        const = ConstantKernel(0.2)
        #rbf = PartialRBF([1.0] * (num_desc + 1),
        #rbf = PartialRBF([0.299, 0.224, 0.177, 0.257, 0.624][:num_desc+1],
        rbf = PartialRBF([0.395, 0.232, 0.297, 0.157, 0.468, 1.0][:num_desc+1],
                         length_scale_bounds=(1.0e-5, 1.0e5), start = 1)
        rhok1 = FittedDensityNoise(decay_rate = 20.0)
        rhok2 = FittedDensityNoise(decay_rate = 5.0)
        wk = WhiteKernel(noise_level=5.0e-6, noise_level_bounds=(1e-06, 1.0e5))
        wk1 = WhiteKernel(noise_level = 0.001, noise_level_bounds=(1e-05, 1.0e5))
        wk2 = WhiteKernel(noise_level = 0.0001, noise_level_bounds=(1e-05, 1.0e5))
        cov_kernel = const * rbf
        noise_kernel = wk + wk1 * rhok1 + wk2 * rhok2
        init_kernel = cov_kernel + noise_kernel
        super(EDMGPR, self).__init__(num_desc,
                       descriptor_getter = get_rho_and_edmgga_descriptors,
                       xed_y_converter = (xed_to_y_pbe, y_to_xed_pbe),
                       init_kernel = init_kernel, use_algpr = use_algpr)

    def is_uncertain(self, x, y, threshold_factor = 1.2, low_noise_bound = 0.002):
        threshold = max(low_noise_bound, np.sqrt(self.gp.kernel_.k2(x))) * threshold_factor
        y_pred = self.gp.predict(x)
        return np.abs(y - y_pred) > threshold
