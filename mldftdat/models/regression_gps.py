from mldftdat.gp import DFTGPR
from mldftdat.density import *
from mldftdat.data import *
from mldftdat.models.kernels import *
import numpy as np
from pyscf.dft.libxc import eval_xc
from sklearn.gaussian_process.kernels import *

def xed_to_y_tail(xed, rho_data):
    y = xed / (ldax(rho_data[0]) - 1e-10)
    return y / tail_fx(rho_data) - 1

def y_to_xed_tail(y, rho_data):
    return (y + 1) * tail_fx(rho_data) * ldax(rho_data[0])

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

def xed_to_y_b88(xed, rho_data):
    b88x = eval_xc('B88,', rho_data)[0] * rho_data[0]
    return xed / b88x - 1

def y_to_xed_b88(y, rho_data):
    b88x = eval_xc('B88,', rho_data)[0] * rho_data[0]
    return (y + 1) * b88x

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
        #X = self.get_descriptors(X, rho_data, num=self.num)
        #y = self.gp.predict(X, return_std = return_std)
        #return self.y_to_xed(y, rho_data)
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


def get_edmgga_descriptors(X, rho_data, num=1):
    return np.arcsinh(X[:,(1,2,4,5,8,6,12,15,16,13,14)[:num]])

def get_rho_and_edmgga_descriptors(X, rho_data, num=1):
    X = get_edmgga_descriptors(X, rho_data, num)
    X = np.append(rho_data[0].reshape(-1,1), X, axis=1)
    return X


class NoisyEDMGPR(EDMGPR):

    def __init__(self, num_desc, use_algpr = False):
        const = ConstantKernel(0.2)
        #rbf = PartialRBF([1.0] * (num_desc + 1),
        #rbf = PartialRBF([0.299, 0.224, 0.177, 0.257, 0.624][:num_desc+1],
        #rbf = PartialRBF([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0][:num_desc],
        #rbf = PartialRBF([0.321, 1.12, 0.239, 0.487, 1.0, 1.0, 1.0, 1.0][:num_desc],
        #rbf = PartialRBF([0.221, 0.468, 0.4696, 0.4829, 0.5, 0.5, 1.0, 1.0][:num_desc],
        # BELOW INIT WORKS WELL (gpr7_beta_v18b/c)
        #rbf = PartialRBF([0.321, 0.468, 0.6696, 0.6829, 0.6, 0.6, 1.0, 1.0][:num_desc],
        #                 length_scale_bounds=(1.0e-5, 1.0e5), start = 1)
        #rbf = PartialRBF([0.3, 0.321, 0.468, 0.6696, 0.6829, 0.6, 0.6, 1.0, 1.0][:num_desc+1],
        rbf = PartialRBF([0.3, 0.4, 0.6696, 0.6829, 0.6, 0.6, 1.0, 1.0, 1.0, 1.0][:num_desc],
                         length_scale_bounds=(1.0e-5, 1.0e5), start = 1)
        rhok1 = FittedDensityNoise(decay_rate = 2.0)
        rhok2 = FittedDensityNoise(decay_rate = 600.0)
        wk = WhiteKernel(noise_level=3.0e-5, noise_level_bounds=(1e-06, 1.0e5))
        wk1 = WhiteKernel(noise_level = 0.002, noise_level_bounds=(1e-05, 1.0e5))
        wk2 = WhiteKernel(noise_level = 0.02, noise_level_bounds=(1e-05, 1.0e5))
        cov_kernel = const * rbf
        noise_kernel = wk + wk1 * rhok1 + wk2 * Exponentiation(rhok2, 2)
        init_kernel = cov_kernel + noise_kernel
        super(EDMGPR, self).__init__(num_desc,
                       descriptor_getter = get_rho_and_edmgga_descriptors10,
                       xed_y_converter = (xed_to_y_lda, y_to_xed_lda),
                       init_kernel = init_kernel, use_algpr = use_algpr)

    #def is_uncertain(self, x, y, threshold_factor = 1.2, low_noise_bound = 0.002):
    #    threshold = max(low_noise_bound, np.sqrt(self.gp.kernel_.k2(x))) * threshold_factor
    #    y_pred = self.gp.predict(x)
    #    return np.abs(y - y_pred) > threshold

    def is_uncertain(self, x, y, threshold_factor = 2, low_noise_bound = 0.002):
        threshold = max(low_noise_bound, np.sqrt(self.gp.kernel_.k2(x))) * threshold_factor
        y_pred, y_std = self.gp.predict(x, return_std=True)
        return (y_std > threshold).any()


def get_edmgga_descriptors2(X, rho_data, num=1):
    X[:,1] = X[:,1]**2
    X[:,2] = 1 / (1 + X[:,2]**2)
    return X[:,(0,2,1,4,5,8,15,16,6,12,13,14)[:num+1]]

def get_big_desc(X, rho_data, num = 1):
    sprefac = 2 * (3 * np.pi * np.pi)**(1.0/3)
    gammax = 0.004
    ssigma = 2**(1.0/3) * sprefac * X[:,1]
    p, alpha = X[:,1]**2, X[:,2]
    fac = (6 * np.pi**2)**(2.0/3) / (16 * np.pi)
    scale = np.sqrt(1 + fac * p + 0.6 * fac * (alpha - 1))
    u = gammax * ssigma**2 / (1 + gammax * ssigma**2)
    t = get_uniform_tau(rho_data[0]) / (rho_data[5] + 1e-9)
    w = (t - 1) / (t + 1)
    desc = np.zeros((X.shape[0], 71))
    desc[:,0] = X[:,0]
    # c_ij, i->w, j->u
    desc[:,1]  = np.arcsinh(ssigma)
    desc[:,2]  = np.arcsinh(alpha)
    desc[:,3]  = w * u
    desc[:,4]  = u**2
    desc[:,5]  = (X[:,4]  - 2.0) / (1 + gammax * ssigma**2)
    desc[:,6]  = (X[:,4]  - 2.0) * scale**3 / (1 + gammax * ssigma**2)
    desc[:,7]  = (X[:,15] - 8.0) / (1 + gammax * ssigma**2)
    desc[:,8]  = (X[:,15] - 8.0) * scale**3 / (1 + gammax * ssigma**2)
    desc[:,9]  = (X[:,16] - 0.5) / (1 + gammax * ssigma**2)
    desc[:,10] = (X[:,16] - 0.5) * scale**3 / (1 + gammax * ssigma**2)
    desc[:,11] = (X[:,5]) / (1 + gammax * ssigma**2)
    desc[:,12] = (X[:,5]) * scale**3 / (1 + gammax * ssigma**2)
    desc[:,13] = (X[:,8]) / (1 + gammax * ssigma**2)
    desc[:,14] = (X[:,8]) * scale**3 / (1 + gammax * ssigma**2)
    i = 15
    for indset in [[1, 2, 5, 7, 9, 11, 13], [1, 2, 6, 8, 10, 12, 14]]:
        for j, ind1 in enumerate(indset):
            for k in range(j+1, len(indset)):
                ind2 = indset[k]
                desc[:,i] = desc[:,ind1] * desc[:,ind2]
                i += 1
    return desc[:,:num+1]


class SmoothEDMGPR(EDMGPR):

    def __init__(self, num_desc, use_algpr = False):
        const = ConstantKernel(0.2)
        #rbf = PartialRBF([1.0] * (num_desc + 1),
        #rbf = PartialRBF([0.299, 0.224, 0.177, 0.257, 0.624][:num_desc+1],
        #rbf = PartialRBF([0.395, 0.232, 0.297, 0.157, 0.468, 1.0][:num_desc+1],
        #                 length_scale_bounds=(1.0e-5, 1.0e5), start = 1)
        dot = PartialDot(start = 1)
        rhok1 = FittedDensityNoise(decay_rate = 2.0)
        rhok2 = FittedDensityNoise(decay_rate = 600.0)
        wk = WhiteKernel(noise_level=1.0e-4, noise_level_bounds=(1e-06, 1.0e5))
        wk1 = WhiteKernel(noise_level = 0.002, noise_level_bounds=(1e-05, 1.0e5))
        wk2 = WhiteKernel(noise_level = 0.02, noise_level_bounds=(1e-05, 1.0e5))
        cov_kernel = const * dot
        noise_kernel = wk + wk1 * rhok1 + wk2 * Exponentiation(rhok2, 2)
        init_kernel = cov_kernel + noise_kernel
        super(EDMGPR, self).__init__(num_desc,
                       descriptor_getter = get_big_desc,
                       xed_y_converter = (xed_to_y_lda, y_to_xed_lda),
                       init_kernel = init_kernel, use_algpr = use_algpr)

    def is_uncertain(self, x, y, threshold_factor = 1.2, low_noise_bound = 0.002):
        threshold = max(low_noise_bound, np.sqrt(self.gp.kernel_.k2(x))) * threshold_factor
        y_pred = self.gp.predict(x)
        return np.abs(y - y_pred) > threshold


class SmoothEDMGPR2(EDMGPR):

    def __init__(self, num_desc, use_algpr = False):
        const = ConstantKernel(0.2)
        #rbf = PartialRBF([1.0] * (num_desc + 1),
        #rbf = PartialRBF([0.299, 0.224, 0.177, 0.257, 0.624][:num_desc+1],
        #rbf = PartialRBF([0.395, 0.232, 0.297, 0.157, 0.468, 1.0][:num_desc+1],
        #                 length_scale_bounds=(1.0e-5, 1.0e5), start = 1)
        rbf1 = SingleRBF(length_scale=0.4, index = 1)
        rbf2 = SingleRBF(length_scale=0.4, index = 2)
        dot = PartialDot(start = 1)
        rhok1 = FittedDensityNoise(decay_rate = 2.0)
        rhok2 = FittedDensityNoise(decay_rate = 600.0)
        wk = WhiteKernel(noise_level=1.0e-4, noise_level_bounds=(1e-06, 1.0e5))
        wk1 = WhiteKernel(noise_level = 0.002, noise_level_bounds=(1e-05, 1.0e5))
        wk2 = WhiteKernel(noise_level = 0.02, noise_level_bounds=(1e-05, 1.0e5))
        #cov_kernel = const * rbf1 * rbf2 * Exponentiation(dot, 3)
        cov_kernel = const * rbf1 * rbf2 * dot
        noise_kernel = wk + wk1 * rhok1 + wk2 * Exponentiation(rhok2, 2)
        init_kernel = cov_kernel + noise_kernel
        super(EDMGPR, self).__init__(num_desc,
                       descriptor_getter = get_big_desc,
                       xed_y_converter = (xed_to_y_b88, y_to_xed_b88),
                       init_kernel = init_kernel, use_algpr = use_algpr)

    def is_uncertain(self, x, y, threshold_factor = 1.2, low_noise_bound = 0.002):
        threshold = max(low_noise_bound, np.sqrt(self.gp.kernel_.k2(x))) * threshold_factor
        y_pred = self.gp.predict(x)
        return np.abs(y - y_pred) > threshold


def get_big_desc2(X, rho_data, num = 1):
    sprefac = 2 * (3 * np.pi * np.pi)**(1.0/3)
    gammax = 0.004
    ssigma = 2**(1.0/3) * sprefac * X[:,1]
    p, alpha = X[:,1]**2, X[:,2]
    fac = (6 * np.pi**2)**(2.0/3) / (16 * np.pi)
    scale = np.sqrt(1 + fac * p + 0.6 * fac * (alpha - 1)).reshape(-1,1)
    u = gammax * ssigma**2 / (1 + gammax * ssigma**2)
    t = get_uniform_tau(rho_data[0]) / (rho_data[5] + 1e-9)
    w = (t - 1) / (t + 1)
    desc = np.zeros((X.shape[0], 71))
    desc = X.copy()
    desc[:,(1,2)] = np.arcsinh(desc[:,(1,2)])
    desc[:,1] = u
    desc[:,2] = 1 / (1 + alpha**2)
    desc[:,(3,4)] /= scale**3
    desc[:,5:10] /= scale**6
    desc[:,10:16] /= scale**9
    desc[:,16:] /= scale**6
    desc[:,3:] = np.arcsinh(desc[:,3:])
    #desc[:,3] -= 2
    #desc[:,4] -= 3.939
    # c_ij, i->w, j->u
    return desc[:,[0,1,2,3,4,5,7,10,9,6,8,11,12,13,14,15,16][:num+1]] / (1 + gammax * ssigma**2).reshape(-1,1)

def get_big_desc3(X, rho_data, num = 1):
    sprefac = 2 * (3 * np.pi * np.pi)**(1.0/3)
    gammax = 0.004
    ssigma = 2**(1.0/3) * sprefac * X[:,1]
    p, alpha = X[:,1]**2, X[:,2]
    fac = (6 * np.pi**2)**(2.0/3) / (16 * np.pi)
    scale = np.sqrt(1 + fac * p + 0.6 * fac * (alpha - 1)).reshape(-1,1)
    desc = np.zeros((X.shape[0], 71))
    desc = X.copy()
    #desc[:,(1,2)] = np.arcsinh(desc[:,(1,2)])
    desc[:,1] = tail_fx(rho_data)
    desc[:,2] = 1 / (1 + alpha**2)
    desc[:,(3,4)] /= scale**3
    desc[:,(5,6,9)] /= scale**8
    desc[:,(7,8,16)] /= scale**10
    desc[:,10:16] /= scale**13
    #desc[:,16:] /= scale**6
    #desc[:,3:] = np.arcsinh(desc[:,3:])
    #desc[:,3] -= 2
    #desc[:,4] -= 3.939
    # c_ij, i->w, j->u
    #desc[:,3:] /= (1 + gammax * ssigma**2).reshape(-1,1)
    desc[:,3:] /= np.array([[  0.98705586,   1.86278409,  15.84986931,  59.38764836,  15.85273698,
  65.28117628,  30.80285635, 108.30579585, 205.16873174, 388.54096809,
 216.71798354, 410.54856388, 777.91829647,  32.26364096]])
    desc[:,3:] = np.arcsinh(desc[:,3:])
    #return desc
    return desc[:,[0,1,2,3,5,7,4,10,9,6,8,11,12,13,14,15,16][:num+1]]


class SmoothEDMGPR3(EDMGPR):

    def __init__(self, num_desc, use_algpr = False):
        const = ConstantKernel(0.282**2)
        #rbf = PartialRBF([1.0] * (num_desc + 1),
        #rbf = PartialRBF([0.299, 0.224, 0.177, 0.257, 0.624][:num_desc+1],
        #rbf = PartialRBF([0.395, 0.232, 0.297, 0.157, 0.468, 1.0][:num_desc+1],
        #                 length_scale_bounds=(1.0e-5, 1.0e5), start = 1)
        rbf1 = SingleRBF(length_scale=0.5, index = 1)
        rbf2 = SingleRBF(length_scale=0.3, index = 2)
        rbf3 = SingleRBF(length_scale=0.3, index = 3)
        dot = PartialDot(start = 1)
        rhok1 = FittedDensityNoise(decay_rate = 2.0)
        rhok2 = FittedDensityNoise(decay_rate = 600.0)
        wk = WhiteKernel(noise_level=1.0e-4, noise_level_bounds=(1e-06, 1.0e5))
        wk1 = WhiteKernel(noise_level = 0.003, noise_level_bounds=(1e-05, 1.0e5))
        wk2 = WhiteKernel(noise_level = 0.02, noise_level_bounds=(1e-05, 1.0e5))
        #cov_kernel = const * rbf1 * rbf2 * Exponentiation(dot, 3)
        cov_kernel = const * rbf1 * rbf2 * rbf3 * dot
        noise_kernel = wk + wk1 * rhok1 #+ wk2 * Exponentiation(rhok2, 2)
        init_kernel = cov_kernel + noise_kernel
        super(EDMGPR, self).__init__(num_desc,
                       descriptor_getter = get_big_desc3,
                       xed_y_converter = (xed_to_y_lda, y_to_xed_lda),
                       init_kernel = init_kernel, use_algpr = use_algpr)

    def is_uncertain(self, x, y, threshold_factor = 1.2, low_noise_bound = 0.002):
        threshold = max(low_noise_bound, np.sqrt(self.gp.kernel_.k2(x))) * threshold_factor
        y_pred = self.gp.predict(x)
        return np.abs(y - y_pred) > threshold


class NoisyEDMGPR(EDMGPR):

    def __init__(self, num_desc, use_algpr = False):
        const = ConstantKernel(0.01)
        #rbf = PartialRBF([1.0] * (num_desc + 1),
        #rbf = PartialRBF([0.299, 0.224, 0.177, 0.257, 0.624][:num_desc+1],
        #rbf = PartialRBF([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0][:num_desc],
        #rbf = PartialRBF([0.321, 1.12, 0.239, 0.487, 1.0, 1.0, 1.0, 1.0][:num_desc],
        #rbf = PartialRBF([0.221, 0.468, 0.4696, 0.4829, 0.5, 0.5, 1.0, 1.0][:num_desc],
        # BELOW INIT WORKS WELL (gpr7_beta_v18b/c)
        #rbf = PartialRBF([0.321, 0.468, 0.6696, 0.6829, 0.6, 0.6, 1.0, 1.0][:num_desc],
        #                 length_scale_bounds=(1.0e-5, 1.0e5), start = 1)
        #rbf = PartialRBF([0.3, 0.321, 0.468, 0.6696, 0.6829, 0.6, 0.6, 1.0, 1.0][:num_desc+1],
        #rbf = PartialRBF([0.5, 0.282, 0.124, 0.202, 0.379, 0.441, 0.707, 0.613, 0.567, 1.0][:num_desc],
        rbf = PartialRBF([0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0][:num_desc],
                         length_scale_bounds=(1.0e-5, 1.0e5), start = 1)
        rhok1 = FittedDensityNoise(decay_rate = 2.0)
        rhok2 = FittedDensityNoise(decay_rate = 600.0)
        wk = WhiteKernel(noise_level=3.0e-5, noise_level_bounds=(1e-06, 1.0e5))
        wk1 = WhiteKernel(noise_level = 0.002, noise_level_bounds=(1e-05, 1.0e5))
        wk2 = WhiteKernel(noise_level = 0.02, noise_level_bounds=(1e-05, 1.0e5))
        cov_kernel = const * rbf
        noise_kernel = wk + wk1 * rhok1 + wk2 * Exponentiation(rhok2, 2)
        init_kernel = cov_kernel + noise_kernel
        super(EDMGPR, self).__init__(num_desc,
                       descriptor_getter = get_big_desc3,
                       xed_y_converter = (xed_to_y_lda, y_to_xed_lda),
                       init_kernel = init_kernel, use_algpr = use_algpr)

    #def is_uncertain(self, x, y, threshold_factor = 1.2, low_noise_bound = 0.002):
    #    threshold = max(low_noise_bound, np.sqrt(self.gp.kernel_.k2(x))) * threshold_factor
    #    y_pred = self.gp.predict(x)
    #    return np.abs(y - y_pred) > threshold

    def is_uncertain(self, x, y, threshold_factor = 2, low_noise_bound = 0.002):
        threshold = max(low_noise_bound, np.sqrt(self.gp.kernel_.k2(x))) * threshold_factor
        y_pred, y_std = self.gp.predict(x, return_std=True)
        return (y_std > threshold).any()
