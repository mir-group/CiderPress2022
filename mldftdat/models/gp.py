from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from mldftdat.models.kernels import *
from mldftdat.data import score, true_metric
from mldftdat import density
import numpy as np

from scipy.linalg import solve_triangular

from abc import ABC, abstractmethod, abstractproperty
import numpy as np


class FeatureNormalizer(ABC):

    @abstractproperty
    def num_arg(self):
        pass

    @abstractmethod
    def bounds(self):
        pass

    @abstractmethod
    def get_feat_(self, X):
        pass

    @abstractmethod
    def get_deriv_(self, X, dfdy):
        pass

    @classmethod
    def from_dict(cls, d):
        if d['code'] == 'L':
            return LMap.from_dict(d)
        elif d['code'] == 'U':
            return UMap.from_dict(d)
        elif d['code'] == 'V':
            return VMap.from_dict(d)
        elif d['code'] == 'W':
            return WMap.from_dict(d)
        elif d['code'] == 'X':
            return XMap.from_dict(d)
        elif d['code'] == 'Y':
            return YMap.from_dict(d)
        else:
            raise ValueError('Unrecognized code')


class LMap(FeatureNormalizer):

    def __init__(self, n, i):
        self.n = n
        self.i = i

    @property
    def bounds(self):
        return (-np.inf, np.inf)

    @property
    def num_arg(self):
        return 1

    def fill_feat_(self, y, x):
        n, i = self.n, self.i
        y[n] = x[i]

    def fill_deriv_(self, dfdx, dfdy, x):
        n, i, = self.n, self.i
        dfdx[i] += dfdy[n]

    def as_dict(self):
        return {
            'code': 'L',
            'n': self.n,
            'i': self.i
        }

    @classmethod
    def from_dict(cls, d):
        return LMap(d['n'], d['i'])


class UMap(FeatureNormalizer):

    def __init__(self, n, i, gamma):
        self.n = n
        self.i = i
        self.gamma = gamma

    @property
    def bounds(self):
        return (0, 1)

    @property
    def num_arg(self):
        return 1

    def fill_feat_(self, y, x):
        n, i = self.n, self.i
        y[n] = self.gamma * x[i] / (1 + self.gamma * x[i])

    def fill_deriv_(self, dfdx, dfdy, x):
        n, i = self.n, self.i
        dfdx[i] += dfdy[n] * self.gamma / (1 + self.gamma * x[i])**2

    def as_dict(self):
        return {
            'code': 'U',
            'n': self.n,
            'i': self.i,
            'gamma': self.gamma
        }

    @classmethod
    def from_dict(cls, d):
        return UMap(d['n'], d['i'], d['gamma'])


def get_vmap_heg_value(heg, gamma):
    return (heg * gamma) / (1 + heg * gamma)


class VMap(FeatureNormalizer):

    def __init__(self, n, i, gamma, scale=1.0, center=0.0):
        self.n = n
        self.i = i
        self.gamma = gamma
        self.scale = scale
        self.center = center

    @property
    def bounds(self):
        return (-center, 1-center)

    @property
    def num_arg(self):
        return 1

    def fill_feat_(self, y, x):
        n, i = self.n, self.i
        y[n] = -self.center + self.scale * self.gamma * x[i] / (1 + self.gamma * x[i])

    def fill_deriv_(self, dfdx, dfdy, x):
        n, i = self.n, self.i
        dfdx[i] += dfdy[n] * self.scale * self.gamma / (1 + self.gamma * x[i])**2

    def as_dict(self):
        return {
            'code': 'V',
            'n': self.n,
            'i': self.i,
            'gamma': self.gamma,
            'scale': self.scale,
            'center': self.center
        }

    @classmethod
    def from_dict(cls, d):
        return VMap(d['n'], d['i'], d['gamma'], d['scale'], d['center'])


class WMap(FeatureNormalizer):

    def __init__(self, n, i, j, k, gammai, gammaj):
        self.n = n
        self.i = i
        self.j = j
        self.k = k
        self.gammai = gammai
        self.gammaj = gammaj

    @property
    def bounds(self):
        return (-1, 1)

    @property
    def num_arg(self):
        return 3

    def fill_feat_(self, y, x):
        n, i, j, k = self.n, self.i, self.j, self.k
        gammai, gammaj = self.gammai, self.gammaj
        y[n] = (gammai*np.sqrt(gammaj/(1 + gammaj*x[j]))*x[k])/(1 + gammai*x[i])

    def fill_deriv_(self, dfdx, dfdy, x):
        n, i, j, k = self.n, self.i, self.j, self.k
        gammai, gammaj = self.gammai, self.gammaj
        dfdx[i] -= dfdy[n] * ((gammai**2*np.sqrt(gammaj/(1 + gammaj*x[j]))*x[k])/(1 + gammai*x[i])**2)
        dfdx[j] -= dfdy[n] * (gammai*(gammaj/(1 + gammaj*x[j]))**1.5*x[k])/(2.*(1 + gammai*x[i]))
        dfdx[k] += dfdy[n] * (gammai*np.sqrt(gammaj/(1 + gammaj*x[j])))/(1 + gammai*x[i])

    def as_dict(self):
        return {
            'code': 'W',
            'n': self.n,
            'i': self.i,
            'j': self.j,
            'k': self.k,
            'gammai': self.gammai,
            'gammaj': self.gammaj
        }

    @classmethod
    def from_dict(cls, d):
        return UMap(d['n'], d['i'], d['j'], d['k'],
                    d['gammai'], d['gammaj'])


class XMap(FeatureNormalizer):

    def __init__(self, n, i, j, k, gammai, gammaj):
        self.n = n
        self.i = i
        self.j = j
        self.k = k
        self.gammai = gammai
        self.gammaj = gammaj

    @property
    def bounds(self):
        return (-1, 1)

    @property
    def num_arg(self):
        return 3

    def fill_feat_(self, y, x):
        n, i, j, k = self.n, self.i, self.j, self.k
        gammai, gammaj = self.gammai, self.gammaj
        y[n] = np.sqrt(gammai/(1 + gammai*x[i]))*np.sqrt(gammaj/(1 + gammaj*x[j]))*x[k]

    def fill_deriv_(self, dfdx, dfdy, x):
        n, i, j, k = self.n, self.i, self.j, self.k
        gammai, gammaj = self.gammai, self.gammaj
        dfdx[i] -= dfdy[n] * ((gammai*np.sqrt(gammai/(1 + gammai*x[i]))*np.sqrt(gammaj/(1 + gammaj*x[j]))*x[k])/(2 + 2*gammai*x[i]))
        dfdx[j] -= dfdy[n] * ((gammaj*np.sqrt(gammai/(1 + gammai*x[i]))*np.sqrt(gammaj/(1 + gammaj*x[j]))*x[k])/(2 + 2*gammaj*x[j]))
        dfdx[k] += dfdy[n] * np.sqrt(gammai/(1 + gammai*x[i]))*np.sqrt(gammaj/(1 + gammaj*x[j]))

    def as_dict(self):
        return {
            'code': 'X',
            'n': self.n,
            'i': self.i,
            'j': self.j,
            'k': self.k,
            'gammai': self.gammai,
            'gammaj': self.gammaj
        }

    @classmethod
    def from_dict(cls, d):
        return UMap(d['n'], d['i'], d['j'], d['k'],
                    d['gammai'], d['gammaj'])


class YMap(FeatureNormalizer):

    def __init__(self, n, i, j, k, l, gammai, gammaj, gammak):
        self.n = n
        self.i = i
        self.j = j
        self.k = k
        self.l = l
        self.gammai = gammai
        self.gammaj = gammaj
        self.gammak = gammak

    @property
    def bounds(self):
        return (-1, 1)

    @property
    def num_arg(self):
        return 4

    def fill_feat_(self, y, x):
        n, i, j, k, l = self.n, self.i, self.j, self.k, self.l
        gammai, gammaj, gammak = self.gammai, self.gammaj, self.gammak
        y[n] = np.sqrt(gammai/(1 + gammai*x[i]))*x[l]*np.sqrt(gammaj/(1 + gammaj*x[j]))*np.sqrt(gammak/(1 + gammak*x[k]))

    def fill_deriv_(self, dfdx, dfdy, x):
        n, i, j, k, l = self.n, self.i, self.j, self.k, self.l
        gammai, gammaj, gammak = self.gammai, self.gammaj, self.gammak
        dfdx[i] -= dfdy[n] * ((gammai*np.sqrt(gammai/(1 + gammai*x[i]))*x[l]*np.sqrt(gammaj/(1 + gammaj*x[j]))*np.sqrt(gammak/(1 + gammak*x[k])))/(2 + 2*gammai*x[i]))
        dfdx[j] -= dfdy[n] * ((gammaj*np.sqrt(gammai/(1 + gammai*x[i]))*x[l]*np.sqrt(gammaj/(1 + gammaj*x[j]))*np.sqrt(gammak/(1 + gammak*x[k])))/(2 + 2*gammaj*x[j]))
        dfdx[k] -= dfdy[n] * ((gammak*np.sqrt(gammai/(1 + gammai*x[i]))*x[l]*np.sqrt(gammaj/(1 + gammaj*x[j]))*np.sqrt(gammak/(1 + gammak*x[k])))/(2 + 2*gammak*x[k]))
        dfdx[l] += dfdy[n] * np.sqrt(gammai/(1 + gammai*x[i]))*np.sqrt(gammaj/(1 + gammaj*x[j]))*np.sqrt(gammak/(1 + gammak*x[k]))

    def as_dict(self):
        return {
            'code': 'Y',
            'n': self.n,
            'i': self.i,
            'j': self.j,
            'k': self.k,
            'l': self.l,
            'gammai': self.gammai,
            'gammaj': self.gammaj,
            'gammak': self.gammak
        }

    @classmethod
    def from_dict(cls, d):
        return UMap(d['n'], d['i'], d['j'], d['k'], d['l']
                    d['gammai'], d['gammaj'], d['gammak'])


class FeatureList():

    def __init__(self, feat_list):
        self.feat_list = feat_list
        self.nfeat = len(self.feat_list)

    def __call__(self, xdesc):
        # xdesc (nsamp, ninp)
        xdesc = xdesc.T
        # now xdesc (ninp, nsamp)
        tdesc = np.zeros((self.nfeat, xdesc.shape[1]))
        for i in range(self.nfeat):
            self.feat_list[i].fill_feat(tdesc, xdesc)
        return tdesc

    def as_dict(self):
        d = {
            'nfeat': self.nfeat,
            'feat_list': [f.as_dict() for f in self.feat_list]
        }
        return d

    @classmethod
    def from_dict(cls, d):
        return cls([FeatureNormalizer.from_dict(d['feat_list'][i])\
                    for i in range(len(d['feat_list']))])


"""
Examples of FeatureList objects:
center0a = get_vmap_heg_value(2.0, gamma0a)
center0b = get_vmap_heg_value(8.0, gamma0b)
center0c = get_vmap_heg_value(0.5, gamma0c)
lst = [
        UMap(0, 0, gammax),
        VMap(1, 1, 1, scale=2.0, center=1.0),
        VMap(2, 2, gamma0a, scale=1.0, center=center0a),
        UMap(3, 3, gamma1),
        UMap(4, 4, gamma2),
        WMap(5, 0, 4, 5, gammax, gamma2),
        XMap(6, 0, 3, 6, gammax, gamma1),
        VMap(7, 7, gamma0b, scale=1.0, center=center0b),
        VMap(8, 8, gamma0c, scale=1.0, center=center0c),
        YMap(9, 0, 3, 4, 9, gammax, gamma1, gamma2),
]
flst = FeatureList(lst)

center0a = get_vmap_heg_value(2.0, gamma0a)
center0b = get_vmap_heg_value(3.0, gamma0b)
UMap(0, 0, gammax)
VMap(1, 1, 1, scale=2.0, center=1.0)
VMap(2, 2, gamma0a, scale=1.0, center=center0a)
UMap(3, 3, gamma1)
UMap(4, 4, gamma2)
WMap(5, 0, 4, 5, gammax, gamma2)
XMap(6, 0, 3, 6, gammax, gamma1)
VMap(7, 7, gamma0b, scale=1.0, center=center0b)
YMap(8, 0, 3, 4, 8, gammax, gamma1, gamma2)
"""


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

    def __init__(self, feature_list, xed_y_converter=None,
                 init_kernel=None, use_algpr=False, selection=None):
        """
        Args:
            feature_list (e.g. xgp.FeatureList): An object containing
                and nfeat property which, when called, transforms the raw
                input descriptors to features for the GP to use.
        """
        num_desc = feature_list.nfeat
        self.get_descriptors = feature_list
        self.selection = selection
        if xed_y_converter is None:
            self.xed_to_y = density.get_y_from_xed
            self.y_to_xed = density.get_xed_from_y
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
            self.gp = GaussianProcessRegressor(kernel=kernel)
            self.al = False
        self.init_kernel = kernel

    def fit(self, xdesc, xed, rho_data, optimize_theta=True):
        if optimize_theta:
            optimizer = 'fmin_l_bfgs_b'
        else:
            optimizer = None
        self.gp.optimizer = optimizer
        self.X = self.get_descriptors(xdesc)
        self.y = self.xed_to_y(xed, rho_data)
        print(np.isnan(self.X).sum(), np.isnan(self.y).sum())
        print(self.X.shape, self.y.shape)
        self.gp.fit(self.X, self.y)
        self.gp.set_params(kernel=self.gp.kernel_)

    def scores(self, xdesc, xed_true, rho_data):
        # Returns
        # r^2 of the model itself
        # rmse of model
        # rmse of exchange energy density
        # relative rmse of exchange energy density
        # score of exchange energy density
        X_test = self.get_descriptors(xdesc)
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

    def refit(self, optimize_theta=True):
        if optimize_theta:
            optimizer = 'fmin_l_bfgs_b'
        else:
            optimizer = None
        self.gp.optimizer = optimizer
        self.gp.fit(self.X, self.y)
        self.gp.set_params(kernel = self.gp.kernel_)

    def predict(self, X, rho_data, return_std=False):
        X = self.get_descriptors(X)
        y = self.gp.predict(X, return_std = return_std)
        if return_std:
            return self.y_to_xed(y[0], rho_data), y[1] * ldax(rho_data[0])
        else:
            return self.y_to_xed(y, rho_data)

    def is_inaccurate(self, x, y, threshold_factor=1.2):
        threshold = self.noise * threshold_factor
        y_pred = self.gp.predict(x)
        return np.abs(y - y_pred) > threshold

    def add_point(self, xdesc, xed, rho_data, threshold_factor=1.2):
        # TODO WARNING: POORLY TESTED
        x = self.get_descriptors(xdesc)
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

    # WARNING: Assumes gp.kernel_.k2 is noise kernel
    def is_uncertain(self, x, y, threshold_factor = 2, low_noise_bound = 0.002):
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
