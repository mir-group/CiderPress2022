from mldftdat.gp import DFTGPR
from mldftdat.pyscf_utils import *
from mldftdat.density import *
from mldftdat.data import *
from mldftdat.models.kernels import *
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

def get_gp_x_descriptors(X, num=1, selection=None):
    # X is initially rho, s, alpha, dvh/(dvh+tau), norm-int-dvh,
    # norm-int-ndn, norm-int-tau, int-n
    X = X[:,(0,1,2,3,4,5,7,6)]
    if selection is not None:
        num = 7
    rho, X = X[:,0], X[:,1:]
    X[:,0] = np.log(1+X[:,0])
    X[:,1] = np.log(0.5 * (1 + X[:,1]))
    #X[:,1] = 1 / (1 + X[:,1]**2) - 0.5
    X[:,3] = np.arcsinh(X[:,3])
    X[:,4] = np.arcsinh(X[:,4])
    X[:,6] = np.arcsinh(X[:,6])
    X[:,5] = np.log(X[:,5] / 6)
    if selection is None:
        X = X[:,(0,1,2,5,4,3,6)]
        return X[:,:num]
    else:
        return X[:,selection]

def get_edmgga_descriptors(X, rho_data, num=1):
    tau0 = get_uniform_tau(rho_data[0]) + 1e-6
    QB = rho_data[4] / tau0
    x = np.arcsinh(QB)
    X = get_gp_x_descriptors(X, num = num)
    if num > 2:
        c = X[:,2]
        ndvh2 = rho_data[5] * c * 1e-3 / (1 - c + 1e-7)
        c2 = ndvh2 / (ndvh2 + rho_data[5] + 1e-7) - 0.5
        X[:,2] = c2
    X = np.append(x.reshape(-1,1), X, axis=1)
    return X
    #return X[:,(0,3,4)]

def get_semilocal_suite(X, num = 3):
    # first 8 are normalized descriptors
    # 8:14 (next 6) are rho_data
    # 14:18 (next 4) are tau_data
    # 18:24 (last 6) are rho second derivatives xx, xy, xz, yy, yz, zz
    assert X.shape[-1] == 24
    rho_data = X[:,8:14]
    ws_radii = get_ws_radii(rho_data[:,0])
    tau_u = get_uniform_tau(rho_data[:,0])
    tau_data = X[:,14:18]
    ddrho = X[:,18:24]
    diag_ind = [0, 3, 5]
    #ddrho[:,diag_ind] -= rho_data[:,4].reshape(-1,1) / 3
    ddrho_mat = np.zeros((X.shape[0], 3, 3))
    inds = [[0, 1, 2], [1, 3, 4], [2, 4, 5]]
    for i in range(3):
        ddrho_mat[:,i,:] = ddrho[:,inds[i]]
    X = get_edmgga_descriptors(X, rho_data.transpose(), num=num)
    #if num > 2:
    #    c = X[:,2]
    #    ndvh2 = rho_data[:,5] * c * 1e-2 / (1 - c + 1e-7)
    #    c2 = ndvh2 / (ndvh2 + rho_data[:,5] + 1e-7) - 0.5
    #    X[:,2] = c2
    d1 = np.linalg.norm(tau_data[:,1:4], axis=1) * ws_radii / (tau_u + 1e-5)
    d2 = np.einsum('pi,pi->p', rho_data[:,1:4], tau_data[:,1:4])
    d2 /= np.linalg.norm(rho_data[:,1:4], axis=1) + 1e-6
    d2 /= np.linalg.norm(tau_data[:,1:4], axis=1) + 1e-6
    d3 = np.einsum('pi,pij,pj->p', rho_data[:,1:4], ddrho_mat, rho_data[:,1:4])
    d4 = np.einsum('pi,pij,pj->p', tau_data[:,1:4], ddrho_mat, tau_data[:,1:4])
    d5 = np.einsum('pi,pij,pj->p', rho_data[:,1:4], ddrho_mat, tau_data[:,1:4])
    d3 /= rho_data[:,0]**(13.0/3) + 1e-5
    d4 /= rho_data[:,0]**(15.0/3) + 1e-5
    d5 /= rho_data[:,0]**(14.0/3) + 1e-5
    return np.append(X, np.array([d1, d2, d3, d4, d5]).T, axis=1)

class EDMGPR(DFTGPR):

    def __init__(self, num_desc, init_kernel = None, use_algpr = False):
        super(EDMGPR, self).__init__(num_desc, descriptor_getter = get_edmgga_descriptors,
                       xed_y_converter = (xed_to_y_edmgga, y_to_xed_edmgga),
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
        rhok1 = FittedDensityNoise(decay_rate = 4.0)
        wk = WhiteKernel(noise_level=1.0e-5, noise_level_bounds=(1e-06, 1.0e5))
        wk1 = WhiteKernel(noise_level = 0.001, noise_level_bounds=(1e-05, 1.0e5))
        cov_kernel = const * rbf
        noise_kernel = wk + wk1 * rhok1# + wk2 * rhok2
        init_kernel = cov_kernel + noise_kernel
        super(EDMGPR, self).__init__(num_desc,
                       descriptor_getter = get_rho_and_edmgga_descriptors,
                       xed_y_converter = (xed_to_y_edmgga, y_to_xed_edmgga),
                       init_kernel = init_kernel, use_algpr = use_algpr)

    def is_uncertain(self, x, y, threshold_factor = 1.2, low_noise_bound = 0.002):
        threshold = max(low_noise_bound, np.sqrt(self.gp.kernel_.k2(x))) * threshold_factor
        y_pred = self.gp.predict(x)
        return np.abs(y - y_pred) > threshold
