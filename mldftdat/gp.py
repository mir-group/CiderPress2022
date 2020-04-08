from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from mldftdat.data import get_gp_x_descriptors, get_y_from_xed, get_xed_from_y,\
                          score, true_metric
from mldftdat import data
import numpy as np

from scipy.linalg import solve_triangular

"""
['L_', 'X_train_', '_K_inv', '__class__', '__delattr__', '__dict__',
'__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__',
'__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__',
'__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__',
'__repr__', '__setattr__', '__setstate__', '__sizeof__', '__str__',
'__subclasshook__', '__weakref__', '_constrained_optimization', '_estimator_type',
'_get_param_names', '_get_tags', '_more_tags', '_rng', '_y_train_mean',
'alpha', 'alpha_', 'copy_X_train', 'fit', 'get_params', 'kernel', 'kernel_',
'log_marginal_likelihood', 'log_marginal_likelihood_value_', 'n_restarts_optimizer',
'normalize_y', 'optimizer', 'predict', 'random_state',
'sample_y', 'score', 'set_params', 'y_train_']
"""

class ALGPR(GaussianProcessRegressor):

    def fit_single(self, x, y):
        # following Rasmussen A.12 (p. 201)
        if self._K_inv is None:
            # compute inverse K_inv of K based on its Cholesky
            # decomposition L and its inverse L_inv
            L_inv = solve_triangular(self.L_.T,
                                     np.eye(self.L_.shape[0]))
            self._K_inv = L_inv.dot(L_inv.T)

        Pinv = self._K_inv
        self.X_train_ = np.append(self.X_train_, x, axis=0)
        self.y_train_ = np.append(self.y_train_, y)
        newK = self.kernel_(x, self.X_train_).reshape(-1)
        newK[-1] += self.alpha
        R = newK[:-1]
        S = newK[-1]
        RPinv = np.dot(R, Pinv)
        PinvQRPinv = np.outer(RPinv, RPinv)
        M = 1 / (S - np.dot(R, np.dot(Pinv, R)))
        Ptilde = Pinv + M * PinvQRPinv
        Rtilde = - M * RPinv
        Stilde = M
        N_old = Pinv.shape[0]
        N_new = N_old + 1

        newKinv = np.zeros((N_new, N_new))
        newKinv[:-1,:-1] = Ptilde
        newKinv[-1,:-1] = Rtilde
        newKinv[:-1,-1] = Rtilde
        newKinv[-1,-1] = Stilde
        self._K_inv = newKinv

        newL = np.zeros((N_new, N_new))
        B = newK[:-1]
        R = solve_triangular(self.L_, B, lower=True)
        newL[:-1,:-1] = self.L_
        newL[:-1,-1] = 0
        newL[-1,:-1] = R
        newL[-1,-1] = np.sqrt(newK[-1] - np.dot(R, R))
        self.L_ = newL

        self.alpha_ = np.dot(self._K_inv, self.y_train_).reshape(-1)


class DFTGPR():

    def __init__(self, num_desc, descriptor_getter = None, xed_y_converter = None,
                 init_kernel = None, use_algpr = False, selection = None):
        if descriptor_getter is None:
            self.get_descriptors = data.get_gp_x_descriptors
        else:
            self.get_descriptors = descriptor_getter
        self.selection = selection
        if xed_y_converter is None:
            self.xed_to_y = data.get_y_from_xed
            self.y_to_xed = data.get_xed_from_y
        else:
            self.xed_to_y = xed_y_converter[0]
            self.y_to_xed = xed_y_converter[1]
        if init_kernel is None:
            rbf = RBF([1.0] * num_desc, length_scale_bounds=(1.0e-5, 1.0e5))
            wk = WhiteKernel(noise_level=1.0e-3, noise_level_bounds=(1e-04, 1.0e5))
            kernel = rbf + wk
        else:
            kernel = init_kernel
        self.X = None
        self.y = None
        if use_algpr:
            self.gp = ALGPR(kernel = kernel)
            self.al = True
        else:
            self.gp = GaussianProcessRegressor(kernel = kernel)
            self.al = False
        self.init_kernel = kernel
        self.num = num_desc

    def fit(self, xdesc, xed, rho_data, optimize_theta = True):
        if optimize_theta:
            optimizer = 'fmin_l_bfgs_b'
        else:
            optimizer = None
        self.gp.optimizer = optimizer
        self.X = self.get_descriptors(xdesc, num=self.num)
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
        X_test = self.get_descriptors(xdesc, num=self.num)
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

    def refit(self, optimize_theta = True):
        if optimize_theta:
            optimizer = 'fmin_l_bfgs_b'
        else:
            optimizer = None
        self.gp.optimizer = optimizer
        self.gp.fit(self.X, self.y)

    def predict(self, X, rho_data, return_std = False):
        X = self.get_descriptors(X, num=self.num)
        y = self.gp.predict(X, return_std = return_std)
        if return_std:
            return self.y_to_xed(y[0], rho_data), y[1] * ldax(rho_data[0])
        else:
            return self.y_to_xed(y, rho_data)

    def is_uncertain(self, x, y, threshold_factor = 1.2):
        threshold = self.noise * threshold_factor
        y_pred = self.gp.predict(x)
        return np.abs(y - y_pred) > threshold

    def add_point(self, xdesc, xed, rho_data, threshold_factor = 1.2):
        x = self.get_descriptors(xdesc, num=self.num)
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

    @property
    def noise(self):
        return max(0.01, np.sqrt(self.gp.kernel_.k2.noise_level))
