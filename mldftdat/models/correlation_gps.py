from mldftdat.gp import DFTGPR
from mldftdat.density import *
from mldftdat.data import *
from mldftdat.models.matrix_rbf import *
import numpy as np
from pyscf.dft.libxc import eval_xc
from sklearn.gaussian_process.kernels import *


def ced_to_y_lda(ced, rho_data_u, rho_data_d):
    rhot = rho_data_u[0] + rho_data_d[0]
    ldac = eval_xc(',LDA_C_PW_MOD', (rho_data_u, rho_data_d), spin = 1)[0]
    return ced / (rhot * ldac + 1e-20)
    #return ced / (ldac - 1e-12) - 1

def y_to_ced_lda(y, rho_data_u, rho_data_d):
    rhot = rho_data_u[0] + rho_data_d[0]
    ldac = eval_xc(',LDA_C_PW_MOD', (rho_data_u, rho_data_d), spin = 1)[0]
    return (y) * (rhot * ldac)

def identity(y, rho_data_u, rho_data_d):
    return y


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
    invrs = (4 * np.pi * X[:,0] / 3)**(1.0/3)
    a = (np.log(2) - 1) / (2 * np.pi**2)
    b = 20.4562557
    desc[:,num-1] = a * np.log(1 + b * invrs + b * invrs**2)
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
    Xt[:,0] = eval_xc(',LDA_C_PW_MOD', (rho_data_u[0], 0 * rho_data_u[0]), spin = 1)[0]
    Xt[:,num] = eval_xc(',LDA_C_PW_MOD', (rho_data_d[0], 0 * rho_data_d[0]), spin = 1)[0]
    Xt[:,2*num] = eval_xc(',LDA_C_PW_MOD', (rho_data_u[0], rho_data_d[0]), spin = 1)[0]
    return Xt

def get_desc_density(Xu, Xd, rho_data_u, rho_data_d, num = 1):
    Xu = get_big_desc3(Xu, num)
    Xd = get_big_desc3(Xd, num)
    Xt = np.hstack((Xu, Xd, (Xd + Xu) / 2))
    rhot = rho_data_u[0] + rho_data_d[0]
    ldac = eval_xc(',LDA_C_PW_MOD', (rho_data_u, rho_data_d), spin = 1)[0] * rhot + 1e-20
    FUNCTIONAL = ',MGGA_C_SCAN'
    cu = eval_xc(FUNCTIONAL, (rho_data_u, 0 * rho_data_u), spin = 1)[0] * rho_data_u[0]
    cd = eval_xc(FUNCTIONAL, (rho_data_d, 0 * rho_data_d), spin = 1)[0] * rho_data_d[0]
    co = eval_xc(FUNCTIONAL, (rho_data_u, rho_data_d), spin = 1)[0] \
            * (rho_data_u[0] + rho_data_d[0])
    co -= cu + cd
    Xt[:,0] = cu / ldac
    Xt[:,num] = cd / ldac
    Xt[:,2*num] = co / ldac
    return Xt

def spinpol_data(data_arr):
    if data_arr.ndim == 2:
        return data_arr, data_arr
    else:
        return data_arr[0], data_arr[1]


class CorrGPR(DFTGPR):

    def __init__(self, num_desc):
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
        #cov_kernel = constss * (rbfu + rbfd) + constos * rbft
        cov_kernel = constos * rbft
        noise_kernel = wk + wk1 * rhok1 + wk2 * Exponentiation(rhok2, 2)
        init_kernel = cov_kernel + noise_kernel
        super(CorrGPR, self).__init__(num_desc,
                       descriptor_getter = get_desc_spinpol,
                       xed_y_converter = (ced_to_y_lda, y_to_ced_lda),
                       init_kernel = init_kernel)

    def is_uncertain(self, x, y, threshold_factor = 1.2, low_noise_bound = 0.002):
        threshold = max(low_noise_bound, np.sqrt(self.gp.kernel_.k2(x))) * threshold_factor
        y_pred, y_std = self.gp.predict(x, return_std=True)
        return (y_std > threshold).any()

    def fit(self, xdesc, ced, rho_data, optimize_theta = True):
        if optimize_theta:
            optimizer = 'fmin_l_bfgs_b'
        else:
            optimizer = None
        rho_data_u, rho_data_d = spinpol_data(rho_data)
        xdescu, xdescd = spinpol_data(xdesc)
        self.gp.optimizer = optimizer
        self.X = self.get_descriptors(xdescu, xdescd, rho_data_u, rho_data_d, num=self.num)
        self.y = self.xed_to_y(ced, rho_data_u, rho_data_d)
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
        rho_data_u, rho_data_d = spinpol_data(rho_data)
        xdescu, xdescd = spinpol_data(xdesc)
        X_test = self.get_descriptors(xdescu, xdescd, rho_data_u, rho_data_d, num=self.num)
        y_true = self.xed_to_y(xed_true, rho_data_u, rho_data_d)
        y_pred = self.gp.predict(X_test)
        xdescu, xdescd = xdesc[0], xdesc[1]
        rho_data_u, rho_data_u = rho_data[0], rho_data[1]
        if len(rho_data.shape) == 2:
            rho = rho_data[0]
        else:
            rho = rho_data
        xed_pred = self.y_to_xed(y_pred, rho_data_u, rho_data_d)
        model_score = score(y_true, y_pred)
        model_rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        xed_rmse = np.sqrt(np.mean((xed_true - xed_pred)**2 / rho**2))
        xed_rel_rmse = np.sqrt(np.mean(((xed_true - xed_pred) / (xed_true + 1e-7))**2))
        xed_score = score(xed_true / rho, xed_pred / rho)
        return model_score, model_rmse, xed_rmse, xed_rel_rmse, xed_score

    def predict(self, xdesc, rho_data, return_std = False):
        rho_data_u, rho_data_d = spinpol_data(rho_data)
        xdescu, xdescd = spinpol_data(xdesc)
        X = self.get_descriptors(xdescu, xdescd, rho_data_u, rho_data_d, num=self.num)
        y = self.gp.predict(X, return_std = return_std)
        if return_std:
            raise NotImplementedError('Uncertainty for correlationo not fully implemented')
        else:
            return self.y_to_xed(y, rho_data_u, rho_data_d)


class CorrGPR2(CorrGPR):

    def __init__(self, num_desc):
        constss = ConstantKernel(1.0)
        constos = ConstantKernel(1.0)
        ind = np.arange(num_desc * 3)
        rbfss = PartialRBF([0.3] * (num_desc), active_dims = ind[1:num_desc])
        rbfos = PartialRBF([0.3] * (num_desc), active_dims = ind[2*num_desc+1:3*num_desc])
        rhok1 = FittedDensityNoise(decay_rate = 2.0)
        rhok2 = FittedDensityNoise(decay_rate = 600.0)
        wk = WhiteKernel(noise_level=1.0e-5, noise_level_bounds=(1e-06, 1.0e5))
        wk1 = WhiteKernel(noise_level = 0.002, noise_level_bounds=(1e-05, 1.0e5))
        wk2 = WhiteKernel(noise_level = 0.02, noise_level_bounds=(1e-05, 1.0e5))
        covss = SingleDot(sigma_0=0.0, sigma_0_bounds='fixed', index = 0) * rbfss
        covss = SpinSymKernel(covss, ind[:num_desc], ind[num_desc:2*num_desc])
        covos = SingleDot(sigma_0=0.0, sigma_0_bounds='fixed', index = 2*num_desc) * rbfos
        cov_kernel = constss * covss + constos * covos
        #cov_kernel = constos * rbft
        noise_kernel = wk + wk1 * rhok1 + wk2 * Exponentiation(rhok2, 2)
        init_kernel = cov_kernel + noise_kernel
        super(CorrGPR, self).__init__(num_desc,
                       descriptor_getter = get_desc_density,
                       xed_y_converter = (ced_to_y_lda, y_to_ced_lda),
                       init_kernel = init_kernel)
