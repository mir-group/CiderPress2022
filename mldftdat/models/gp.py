"""
Gaussian Process class for training CIDER functionals.
Serves as a wrapper for the GPR class of sklearn.
This module also contains utilities for constructing
covariance and noise kernels and baselines.
"""

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from mldftdat.models.kernels import *
from mldftdat.models.transform_data import *
from mldftdat.density import get_ldax_dens, get_ldax,\
                             get_xed_from_y, get_y_from_xed, get_x,\
                             tail_fx
import numpy as np

from scipy.linalg import solve_triangular

def true_metric(y_true, y_pred, rho):
    """
    Find relative and absolute mse, as well as r2
    score, for the exchange energy per particle (epsilon_x)
    from the true and predicted enhancement factor
    y_true and y_pred.
    """
    res_true = get_x(y_true, rho)
    res_pred = get_x(y_pred, rho)
    return np.sqrt(np.mean(((res_true - res_pred) / (1))**2)),\
           np.sqrt(np.mean(((res_true - res_pred) / (res_true + 1e-7))**2)),\
           score(res_true, res_pred)

def score(y_true, y_pred):
    """
    r2 score
    """
    #y_mean = np.mean(y_true)
    #return 1 - ((y_pred-y_true)**2).sum() / ((y_pred-y_mean)**2).sum()
    return r2_score(y_true, y_pred)


SCALE_FAC = (6 * np.pi**2)**(2.0/3) / (16 * np.pi)

def xed_to_y_tail(xed, rho_data):
    y = xed / (ldax(rho_data[0]) - 1e-10)
    return y / tail_fx(rho_data) - 1

def y_to_xed_tail(y, rho_data):
    return (y + 1) * tail_fx(rho_data) * ldax(rho_data[0])

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

def xed_to_y_lda(xed, rho):
    return get_y_from_xed(xed, rho)

def y_to_xed_lda(y, rho):
    return get_xed_from_y(y, rho)

def chachiyo_fx(s2):
    c = 4 * np.pi / 9
    x = c * np.sqrt(s2)
    dx = c / (2 * np.sqrt(s2))
    Pi = np.pi
    Log = np.log
    chfx = (3*x**2 + Pi**2*Log(1 + x))/((Pi**2 + 3*x)*Log(1 + x))
    dchfx = (-3*x**2*(Pi**2 + 3*x) + 3*x*(1 + x)*(2*Pi**2 + 3*x)*Log(1 + x) - 3*Pi**2*(1 + x)*Log(1 + x)**2)/((1 + x)*(Pi**2 + 3*x)**2*Log(1 + x)**2)
    dchfx *= dx
    chfx[s2<1e-8] = 1 + 8 * s2[s2<1e-8] / 27
    dchfx[s2<1e-8] = 8.0 / 27
    return chfx, dchfx

def xed_to_y_chachiyo(xed, rho, s2):
    return xed / get_ldax_dens(rho) - chachiyo_fx(s2)[0]

def y_to_xed_chachiyo(y, rho, s2):
    return (y + chachiyo_fx(s2)[0]) * get_ldax_dens(rho)

def get_unity():
    return 1

def get_identity(x):
    return x

XED_Y_CONVERTERS = {
    # method_name: (xed_to_y, y_to_xed, fx_baseline, nfeat--rho, s2, alpha...)
    'LDA': (xed_to_y_lda, y_to_xed_lda, get_unity, 1),
    'CHACHIYO': (xed_to_y_chachiyo, y_to_xed_chachiyo, chachiyo_fx, 2)
}

def get_rbf_kernel(length_scale):
    return ConstantKernel(1.0) * PartialRBF(length_scale=length_scale,
                                            start=1)

def get_agpr_kernel(length_scale, scale=None, order=2, nsingle=1):
    start=1
    if scale is None:
        scale = [1.0] * (order+1)
    if nsingle == 0:
        singles = None
    elif nsingle == 1:
        singles = SingleRBF(length_scale=length_scale[0], index=start)
    else:
        active_dims = np.arange(start,start+nsingle).tolist()
        singles = PartialRBF(length_scale=length_scale[:nsingle],
                             active_dims=active_dims)
    cov_kernel = PartialARBF(order=order, length_scale=length_scale[nsingle:],
                             scale=scale, length_scale_bounds=(0.01, 10),
                             start=start+nsingle)
    if singles is None:
        return cov_kernel
    else:
        return singles * cov_kernel

def get_density_noise_kernel(noise0=1e-5, noise1=1e-3):
    wk0 = WhiteKernel(noise_level=noise0, noise_level_bounds=(1e-6,1e-3))
    wk1 = WhiteKernel(noise_level=noise1)
    return wk0 + wk1 * DensityNoise()

def get_exp_density_noise_kernel(noise0=1e-5, noise1=1e-5):
    wk0 = WhiteKernel(noise_level=noise0, noise_level_bounds=(1e-6,1e-3))
    wk1 = WhiteKernel(noise_level=noise1, noise_level_bounds=(1e-7,1e-3))
    return wk0 + wk1 * ExponentialDensityNoise()

def get_fitted_density_noise_kernel(decay1=50.0, decay2=1e6, noise0=1e-6,
                                    noise1=0.0004, noise2=0.2):
    rhok1 = FittedDensityNoise(decay_rate=decay1, decay_rate_bounds='fixed')
    rhok2 = FittedDensityNoise(decay_rate=decay2, decay_rate_bounds='fixed')
    wk = WhiteKernel(noise_level=noise0, noise_level_bounds=(1e-6,1e-3))
    wk1 = WhiteKernel(noise_level=noise1)
    wk2 = WhiteKernel(noise_level=noise2)
    noise_kernel = wk + wk1 * rhok1 + wk2 * rhok2
    return noise_kernel


class DFTGPR():

    def __init__(self, feature_list, desc_order=None,
                 xed_y_converter=XED_Y_CONVERTERS['LDA'],
                 init_kernel=None):
        """
        Args:
            feature_list (e.g. models.transform_data.FeatureList):
                An object containing an nfeat property which, when
                called, transforms the raw input descriptors to
                features for the GP to use.
            desc_order (list): This list indexes the descriptors to
                pass to the feature_list
            xed_y_converter (tuple): baseline functional details,
                see XED_Y_CONVERTERS
                    (xed_to_y conversion function,
                     y_to_xed conversion function,
                     value and derivative of baseline
                     exchange enhancement factor,
                     1 if baseline is LDA, 2 if GGA)
                y is the target function and xed is the exchange
                energy density.
            init_kernel: Initial guess for kernel, if desired.
                Set to a simple RBF otherwise
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
        self.gp = GaussianProcessRegressor(kernel=kernel)
        self.al = False
        self.init_kernel = kernel

    def get_descriptors(self, x):
        return np.append(x[:,:1],
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

    def fit(self, xdesc, xed, optimize_theta=True,
            add_heg=False, add_tail=False):
        if optimize_theta:
            optimizer = 'fmin_l_bfgs_b'
        else:
            optimizer = None
        self.gp.optimizer = optimizer
        self.X = self.get_descriptors(xdesc)
        self.y = self.xed_to_y(xed, xdesc)
        print(np.isnan(self.X).sum(), np.isnan(self.y).sum())
        print(self.X.shape, self.y.shape)
        if add_heg:
            hegx = (0 * self.X[0])
            # set the density to be large -> low uncertainty.
            hegx[0] = 1e8
            # Assume heg y-value is zero.
            hegy = 0
            self.y = np.append([hegy], self.y)
            self.X = np.append([hegx], self.X, axis=0)
        if add_tail:
            tailx = (0 * self.X[0])
            tailx[0] = 1e8
            tailx[1] = 1.0
            tailx[2] = 1.0
            tailx[3:] = [b[0] for b in self.feature_list.bounds_list[2:]]
            tailx2 = tailx.copy()
            tailx2[2] = -1.0
            taily = 0
            self.y = np.append([taily, taily], self.y)
            self.X = np.append([tailx, tailx2], self.X, axis=0)
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

    def predict(self, xdesc, return_std=False):
        X = self.get_descriptors(xdesc)
        y = self.gp.predict(X, return_std=return_std)
        if return_std:
            return self.y_to_xed(y[0], xdesc), y[1] * get_ldax_dens(xdesc[:,0])
        else:
            return self.y_to_xed(y, xdesc)

    def is_inaccurate(self, x, y, threshold_factor=1.2):
        threshold = self.noise * threshold_factor
        y_pred = self.gp.predict(x)
        return np.abs(y - y_pred) > threshold

    # WARNING: POORLY TESTED
    def add_point(self, xdesc, xed, threshold_factor=1.2):
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
            noise_kernel = get_fitted_density_noise_kernel()
        else:
            noise_kernel = get_density_noise_kernel()
        init_kernel = cov_kernel + noise_kernel
        xed_y_converter = XED_Y_CONVERTERS[args.xed_y_code]

        gpr = cls(feature_list, desc_order, xed_y_converter, init_kernel)
        gpr.a0 = args.gg_a0
        gpr.fac_mul = args.gg_facmul
        gpr.amin = args.gg_amin
        gpr.desc_version = args.version

        return gpr
