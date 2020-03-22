from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from mldftdat.data import get_gp_x_descriptors, get_y_from_xed, get_xed_from_y,\
                          score, true_metric
from mldftdat import data
import numpy as np

class DFTGPR():

    def __init__(self, num_desc, descriptor_getter = None, xed_y_converter = None):
        if descriptor_getter is None:
            self.get_descriptors = data.get_gp_x_descriptors
        else:
            self.get_descriptors = descriptor_getter
        if xed_y_converter is None:
            self.xed_to_y = data.get_y_from_xed
            self.y_to_xed = data.get_xed_from_y
        else:
            self.xed_to_y = xed_y_converter[0]
            self.y_to_xed = xed_y_converter[1]
        rbf = RBF([1.0] * num_desc, length_scale_bounds=(1.0e-5, 1.0e5))
        wk = WhiteKernel(noise_level=1.0e-3, noise_level_bounds=(1e-05, 1.0e5))
        kernel = rbf + wk
        self.X = None
        self.y = None
        self.gp = GaussianProcessRegressor(kernel = kernel)
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
        return self.y_to_xed(y, rho_data)

    def is_uncertain(self, x, y, threshold_factor = 1.2):
        threshold = self.noise * threshold_factor
        y_pred = self.gp.predict(x)
        return np.abs(y - y_pred) > threshold

    def fit_single(self, x, y):
        # NOTE: experimental
        raise NotImplementedError()

    def add_point(self, xdesc, xed, rho_data, threshold_factor = 1.2):
        x = self.get_descriptors(xdesc, num=self.num)
        y = self.xed_to_y(xed, rho_data)
        if self.is_uncertain(x, y, threshold_factor):
            self.X = np.append(self.X, x, axis=0)
            self.y = np.append(self.y, y)
            prev_optimizer = self.gp.optimizer
            self.gp.optimizer = None
            self.gp.fit(self.X, self.y)
            self.gp.optimizer = prev_optimizer

    @property
    def noise(self):
        return max(0.01, np.sqrt(self.gp.kernel_.k2.noise_level))
