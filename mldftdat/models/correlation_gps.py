from mldftdat.gp import DFTGPR
from mldftdat.density import *
from mldftdat.data import *
from mldftdat.models.matrix_rbf import *
import numpy as np
from pyscf.dft.libxc import eval_xc
from sklearn.gaussian_process.kernels import *


def ced_to_y_lda(ced, rho_data):
    if rho_data.ndim == 2:
        ldac = eval_xc(',LDA_C_PW92_MOD', rho_data)[0] * rho_data[0]
    else:
        rhot = rho_data[0][0] + rho_data[1][0]
        ldac = eval_xc(',LDA_C_PW92_MOD', rho_data, spin = 1)[0] * rhot
    return ced / (ldac - 1e-12) - 1

def y_to_ced_lda(y, rho_data):
    if rho_data.ndim == 2:
        ldac = eval_xc(',LDA_C_PW92_MOD', rho_data)[0] * rho_data[0]
    else:
        rhot = rho_data[0][0] + rho_data[1][0]
        ldac = eval_xc(',LDA_C_PW92_MOD', rho_data, spin = 1)[0] * rhot
    return (y + 1) * ldac


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


def get_big_desc2(X, num):
    sprefac = 2 * (3 * np.pi * np.pi)**(1.0/3)

    gammax = 0.004 * (2**(1.0/3) * sprefac)**2
    gamma1 = 0.004
    gamma2 = 0.004

    s = X[:,1]
    p, alpha = X[:,1]**2, X[:,2]

    fac = (6 * np.pi**2)**(2.0/3) / (16 * np.pi)
    scale = np.sqrt(1 + fac * p + 0.6 * fac * (alpha - 1))

    desc = np.zeros((X.shape[0], 12))
    refs = gammax / (1 + gammax * s**2)
    #desc[:,(1,2)] = np.arcsinh(desc[:,(1,2)])
    ref0a = 0.5 / (1 + X[:,4] * scale**3 / 2)
    ref0b = 0.125 / (1 + X[:,15] * scale**3 / 8)
    ref0c = 2 / (1 + X[:,16] * scale**3 / 0.5)
    ref1 = gamma1 / (1 + gamma1 * X[:,5]**2 * scale**6)
    ref2 = gamma2 / (1 + gamma2 * X[:,8] * scale**6) 

    sprefac = 2 * (3 * np.pi * np.pi)**(1.0/3)
    hprefac = 1.0 / 3 * (4 * np.pi**2 / 3)**(1.0 / 3)
    sp = 0.5 * sprefac * s / np.pi**(1.0/3) * 2**(1.0/3)

    desc[:,0] = X[:,0]
    #desc[:,1] = hprefac * 2.0 / 3 * sp / np.arcsinh(0.5 * sp + 1.1752012)
    #desc[:,1] = np.arcsinh(0.5 * sp)
    #desc[:,1] = tail_fx_direct(s)# * s**2 * refs
    desc[:,1] = s**2 * refs
    desc[:,2] = 2 / (1 + alpha**2) - 1.0
    #desc[:,2] = np.arcsinh(alpha - 1)
    desc[:,3] = (X[:,4] * scale**3 - 2.0) * ref0a
    #desc[:,3] = X[:,4] - 2.0 / scale**3
    desc[:,4] = X[:,5]**2 * scale**6 * ref1
    desc[:,5] = X[:,8] * scale**6 * ref2
    desc[:,6] = X[:,12] * scale**3 * refs * np.sqrt(ref2)
    desc[:,7] = X[:,6] * scale**3 * np.sqrt(refs) * np.sqrt(ref1)
    desc[:,8] = (X[:,15] * scale**3 - 8.0) * ref0b
    desc[:,9] = (X[:,16] * scale**3 - 0.5) * ref0c
    #desc[:,9] = X[:,16] - 0.5 / scale**3
    desc[:,10] = (X[:,13] * scale**6) * np.sqrt(refs) * np.sqrt(ref1) * np.sqrt(ref2)
    desc[:,11] = (X[:,14] * scale**9) * np.sqrt(ref2) * ref1
    return desc[:,:num+1]

def get_big_desc3(X, num):
    sprefac = 2 * (3 * np.pi * np.pi)**(1.0/3)

    gammax = 0.682
    gamma1 = 0.01552
    gamma2 = 0.01617
    gamma0a = 0.64772
    gamma0b = 0.44065
    gamma0c = 0.6144

    s = X[:,1]
    p, alpha = X[:,1]**2, X[:,2]

    fac = (6 * np.pi**2)**(2.0/3) / (16 * np.pi)
    scale = np.sqrt(1 + fac * p + 0.6 * fac * (alpha - 1))

    desc = np.zeros((X.shape[0], 12))
    refs = gammax / (1 + gammax * s**2)
    #desc[:,(1,2)] = np.arcsinh(desc[:,(1,2)])
    ref0a = gamma0a / (1 + X[:,4] * scale**3 * gamma0a)
    ref0b = gamma0b / (1 + X[:,15] * scale**3 * gamma0b)
    ref0c = gamma0c / (1 + X[:,16] * scale**3 * gamma0c)
    ref1 = gamma1 / (1 + gamma1 * X[:,5]**2 * scale**6)
    ref2 = gamma2 / (1 + gamma2 * X[:,8] * scale**6)

    sprefac = 2 * (3 * np.pi * np.pi)**(1.0/3)
    hprefac = 1.0 / 3 * (4 * np.pi**2 / 3)**(1.0 / 3)
    sp = 0.5 * sprefac * s / np.pi**(1.0/3) * 2**(1.0/3)

    desc[:,0] = X[:,0]
    #desc[:,1] = hprefac * 2.0 / 3 * sp / np.arcsinh(0.5 * sp + 1.1752012)
    #desc[:,1] = np.arcsinh(0.5 * sp)
    #desc[:,1] = tail_fx_direct(s)# * s**2 * refs
    desc[:,1] = s**2 * refs
    desc[:,2] = 2 / (1 + alpha**2) - 1.0
    #desc[:,2] = np.arcsinh(alpha - 1)
    desc[:,3] = (X[:,4] * scale**3 - 2.0) * ref0a
    #desc[:,3] = X[:,4] - 2.0 / scale**3
    desc[:,4] = X[:,5]**2 * scale**6 * ref1
    desc[:,5] = X[:,8] * scale**6 * ref2
    desc[:,6] = X[:,12] * scale**3 * refs * np.sqrt(ref2)
    desc[:,7] = X[:,6] * scale**3 * np.sqrt(refs) * np.sqrt(ref1)
    desc[:,8] = (X[:,15] * scale**3 - 8.0) * ref0b
    desc[:,9] = (X[:,16] * scale**3 - 0.5) * ref0c
    #desc[:,9] = X[:,16] - 0.5 / scale**3
    desc[:,10] = (X[:,13] * scale**6) * np.sqrt(refs) * np.sqrt(ref1) * np.sqrt(ref2)
    desc[:,11] = (X[:,14] * scale**9) * np.sqrt(ref2) * ref1
    return desc[:,:num]

def get_rho_and_edmgga_descriptors13(X, rho_data, num=1):
    X = get_big_desc2(X, num)
    #X = np.append(rho_data[0].reshape(-1,1), X, axis=1)
    return X

def get_rho_and_edmgga_descriptors14(X, rho_data, num=1):
    X = get_big_desc3(X, num)
    #X = np.append(rho_data[0].reshape(-1,1), X, axis=1)
    return X

def get_desc_spinpol(Xu, Xd, rho_data_u, rho_data_d, num = 1):
    Xu = get_big_desc3(Xu, num)
    Xd = get_big_desc3(Xd, num)
    Xt = np.hstack((Xu, Xd, (Xd + Xu) / 2))
    Xt[0] = eval_xc(',LDA_C_PW92_MOD', (rho_data_u[0], 0 * rho_data_u[0]), spin = 1)[0]
    Xt[num] = eval_xc(',LDA_C_PW92_MOD', (rho_data_d[0], 0 * rho_data_d[0]), spin = 1)[0]
    Xt[2*num] = eval_xc(',LDA_C_PW92_MOD', (rho_data_u[0], rho_data_d[0]), spin = 1)[0]
    return Xt


class CorrGPR(EDMGPR):

    def __init__(self, num_desc, use_algpr = False):
        constss = ConstantKernel(0.2)
        constos = ConstantKernel(1.0)
        ind = np.arange(num_desc * 3)
        rbfu = PartialRBF([0.3] * num_desc, active_dims = ind[0:num_desc])
        rbfd = PartialRBF([0.3] * num_desc, active_dims = ind[num_desc:2*num_desc])
        rbft = PartialRBF([0.3] * num_desc, active_dims = ind[2*num_desc:3*num_desc])
        rhok1 = FittedDensityNoise(decay_rate = 2.0)
        rhok2 = FittedDensityNoise(decay_rate = 600.0)
        wk = WhiteKernel(noise_level=3.0e-5, noise_level_bounds=(1e-06, 1.0e5))
        wk1 = WhiteKernel(noise_level = 0.002, noise_level_bounds=(1e-05, 1.0e5))
        wk2 = WhiteKernel(noise_level = 0.02, noise_level_bounds=(1e-05, 1.0e5))
        cov_kernel = constss * (rbfu + rbfd) + constos * rbft
        noise_kernel = wk + wk1 * rhok1 + wk2 * Exponentiation(rhok2, 2)
        init_kernel = cov_kernel + noise_kernel
        super(EDMGPR, self).__init__(num_desc,
                       descriptor_getter = get_desc_spinpol,
                       xed_y_converter = (ced_to_y_lda, y_to_ced_lda),
                       init_kernel = init_kernel, use_algpr = use_algpr)

    def is_uncertain(self, x, y, threshold_factor = 2, low_noise_bound = 0.002):
        threshold = max(low_noise_bound, np.sqrt(self.gp.kernel_.k2(x))) * threshold_factor
        y_pred, y_std = self.gp.predict(x, return_std=True)
        return (y_std > threshold).any()

    def fit(self, xdesc, ced, rho_data, optimize_theta = True):
        if optimize_theta:
            optimizer = 'fmin_l_bfgs_b'
        else:
            optimizer = None
        xdescu, xdescd = xdesc[0], xdesc[1]
        rho_data_u, rho_data_u = rho_data[0], rho_data[1]
        self.gp.optimizer = optimizer
        self.X = self.get_descriptors(xdescu, xdescd, rho_data_u, rho_data_d, num=self.num)
        self.y = self.xed_to_y(ced, rho_data)
        print(np.isnan(self.X).sum(), np.isnan(self.y).sum())
        print(self.X.shape, self.y.shape)
        self.gp.fit(self.X, self.y)
        self.gp.set_params(kernel = self.gp.kernel_)

    def scores(self, xdesc, xed_true, rho_data):
        # Returns
        # r^2 of the model itself
        # rmse of model
        # rmse of exchange energy density
        # relative rmse of exchange energy density
        # score of exchange energy density
        X_test = self.get_descriptors(xdescu, xdescd, rho_data_u, rho_data_d, num=self.num)
        y_true = self.xed_to_y(xed_true, rho_data)
        y_pred = self.gp.predict(X_test)
        xdescu, xdescd = xdesc[0], xdesc[1]
        rho_data_u, rho_data_u = rho_data[0], rho_data[1]
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
        X = self.get_descriptors(X, num=self.num)
        y = self.gp.predict(X, return_std = return_std)
        if return_std:
            return self.y_to_xed(y[0], rho_data), y[1] * ldax(rho_data[0])
        else:
            return self.y_to_xed(y, rho_data)