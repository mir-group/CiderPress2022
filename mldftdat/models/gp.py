from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from mldftdat.models.kernels import *
from mldftdat.xcutil.transform_data import *
from mldftdat.data import score, true_metric
from mldftdat import density
import numpy as np

from scipy.linalg import solve_triangular


class ALGPR(GaussianProcessRegressor):
    # TODO this is WIP, DO NOT USE!!!
    # active learning GP with utility to add
    # one training point at a time efficiently.

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

    def __init__(self, feature_list, desc_order=None,
                 xed_y_converter=XED_Y_CONVERTERS['LDA'],
                 init_kernel=None, use_algpr=False):
        """
        Args:
            feature_list (e.g. xcutil.transform_data.FeatureList):
                An object containing an nfeat property which, when
                called, transforms the raw input descriptors to
                features for the GP to use.
        """
        num_desc = feature_list.nfeat
        if desc_order is None:
            desc_order = np.arange(num_desc)
        
        self.desc_order = desc_order
        self.feature_list = feature_list
        self.xed_y_converter = xed_y_converter

        if init_kernel is None:
            rbf = RBF([1.0] * num_desc, length_scale_bounds=(1.0e-5, 1.0e5))
            wk = WhiteKernel(noise_level=1.0e-3,
                             noise_level_bounds=(1e-05, 1.0e5))
            kernel = rbf + wk
        else:
            kernel = init_kernel
        self.X = None
        self.y = None
        if use_algpr:
            self.gp = ALGPR(kernel=kernel)
            self.al = True
        else:
            self.gp = GaussianProcessRegressor(kernel=kernel)
            self.al = False
        self.init_kernel = kernel

    def get_descriptors(self, x):
        return np.append(self.x[:,:1],
                         self.feature_list(x[:,self.desc_order]),
                         axis=1)

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

    def fit(self, xdesc, xed, optimize_theta=True):
        if optimize_theta:
            optimizer = 'fmin_l_bfgs_b'
        else:
            optimizer = None
        self.gp.optimizer = optimizer
        self.X = self.get_descriptors(xdesc)
        self.y = self.xed_to_y(xed, xdesc)
        print(np.isnan(self.X).sum(), np.isnan(self.y).sum())
        print(self.X.shape, self.y.shape)
        self.gp.fit(self.X, self.y)
        self.gp.set_params(kernel=self.gp.kernel_)

    def scores(self, xdesc, xed_true):
        # Returns
        # r^2 of the model itself
        # rmse of model
        # rmse of exchange energy density
        # relative rmse of exchange energy density
        # score of exchange energy density
        X_test = self.get_descriptors(xdesc)
        y_true = self.xed_to_y(xed_true, xdesc)
        y_pred = self.gp.predict(X_test)
        if len(rho_data.shape) == 2:
            rho = rho_data[0]
        else:
            rho = rho_data
        xed_pred = self.y_to_xed(y_pred, xdesc)
        model_score = score(y_true, y_pred)
        model_rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        xed_rmse = np.sqrt(np.mean((xed_true - xed_pred)**2 / rho**2))
        xed_rel_rmse = np.sqrt(np.mean(((xed_true - xed_pred) / (xed_true + 1e-7))**2))
        xed_score = score(xed_true / rho, xed_pred / rho)
        return model_score, model_rmse, xed_rmse, xed_rel_rmse, xed_score

    def refit(self, optimize_theta=True):
        if optimize_theta:
            optimizer = 'fmin_l_bfgs_b'
        else:
            optimizer = None
        self.gp.optimizer = optimizer
        self.gp.fit(self.X, self.y)
        self.gp.set_params(kernel = self.gp.kernel_)

    def predict(self, X, return_std=False):
        X = self.get_descriptors(X)
        y = self.gp.predict(X, return_std = return_std)
        if return_std:
            return self.y_to_xed(y[0], xdesc), y[1] * ldax(rho_data[0])
        else:
            return self.y_to_xed(y, xdesc)

    def is_inaccurate(self, x, y, threshold_factor=1.2):
        threshold = self.noise * threshold_factor
        y_pred = self.gp.predict(x)
        return np.abs(y - y_pred) > threshold

    def add_point(self, xdesc, xed, threshold_factor=1.2):
        # TODO WARNING: POORLY TESTED
        x = self.get_descriptors(xdesc)
        y = self.xed_to_y(xed, xdesc)
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

    # WARNING: Assumes gp.kernel_.k2 is noise kernel
    def is_uncertain(self, x, y, threshold_factor=2, low_noise_bound=0.002):
        threshold = max(low_noise_bound, np.sqrt(self.gp.kernel_.k2(x))) * threshold_factor
        y_pred, y_std = self.gp.predict(x, return_std=True)
        return (y_std > threshold).any()

    # WARNING: Assumes HEG represented by zero-vector
    def add_heg_limit(self):
        # set the feature vector to the HEG (all zeros).
        hegx = (0 * self.X[0])
        # set the density to be large -> low uncertainty.
        hegx[0] = 1e8
        # Assume heg y-value is zero.
        hegy = 0
        self.y = np.append([hegy], self.y)
        self.X = np.append([hegx], self.X, axis=0)
        self.gp.y_train_ = self.y
        self.gp.X_train_ = self.X
        K = self.gp.kernel_(self.gp.X_train_)
        K[np.diag_indices_from(K)] += self.gp.alpha
        # from sklearn gpr
        from scipy.linalg import cholesky, cho_solve
        try:
            self.gp.L_ = cholesky(K, lower=True)  # Line 2
            self.gp._K_inv = None
        except np.linalg.LinAlgError as exc:
            exc.args = ("The kernel, %s, is not returning a "
                        "positive definite matrix. Try gradually "
                        "increasing the 'alpha' parameter of your "
                        "GaussianProcessRegressor estimator."
                        % self.gp.kernel_,) + exc.args
            raise
        self.gp.alpha_ = cho_solve((self.gp.L_, True), self.gp.y_train_)  # Line 3

    @classmethod
    def from_settings(cls, X, feature_list, args):
        if args.desc_order is None:
            desc_order = np.arange(X.shape[1])
        else:
            desc_order = args.desc_order
        X = X[:,desc_order]

        if args.length_scale is None:
            XT = feature_list(X)
            length_scale = np.std(XT, axis=0) * args.length_scale_mul
        else:
            length_scale = args.length_scale

        if args.agpr:
            cov_kernel = get_agpr_kernel(length_scale, args.agpr_scale,
                                         args.agpr_order, args.agpr_nsingle)
        else:
            cov_kernel = get_rbf_kernel(length_scale)
        if args.optimize_noise:
            noise_kernel = get_exp_density_noise_kernel()
        else:
            noise_kernel = get_density_noise_kernel()
        init_kernel = cov_kernel + noise_kernel
        xed_y_converter = XED_Y_CONVERTERS[args.xed_y_code]

        return cls(feature_list, desc_order, xed_y_converter, init_kernel)
