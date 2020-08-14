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
    return ced / (rhot * ldac - 1e-20)
    #return ced / (ldac - 1e-12) - 1

def y_to_ced_lda(y, rho_data_u, rho_data_d):
    rhot = rho_data_u[0] + rho_data_d[0]
    ldac = eval_xc(',LDA_C_PW_MOD', (rho_data_u, rho_data_d), spin = 1)[0]
    return (y) * (rhot * ldac)

def ced_to_y_scan(ced, rho_data_u, rho_data_d):
    rhot = rho_data_u[0] + rho_data_d[0]
    ldac = eval_xc(',LDA_C_PW_MOD', (rho_data_u, rho_data_d), spin = 1)[0]
    scanc = eval_xc(',MGGA_C_SCAN', (rho_data_u, rho_data_d), spin = 1)[0]
    return (ced - scanc * rhot) / (rhot * ldac - 1e-20)
    #return ced / (ldac - 1e-12) - 1

def y_to_ced_scan(y, rho_data_u, rho_data_d):
    rhot = rho_data_u[0] + rho_data_d[0]
    ldac = eval_xc(',LDA_C_PW_MOD', (rho_data_u, rho_data_d), spin = 1)[0]
    scanc = eval_xc(',MGGA_C_SCAN', (rho_data_u, rho_data_d), spin = 1)[0]
    return (y) * (rhot * ldac) + rhot * scanc

def ced_to_y_os(ced, rho_data_u, rho_data_d):
    rhot = rho_data_u[0] + rho_data_d[0]
    ldac = eval_xc(',LDA_C_PW_MOD', (rho_data_u, rho_data_d), spin = 1)[0]
    FUNCTIONAL = ',MGGA_C_SCAN'
    cu = eval_xc(FUNCTIONAL, (rho_data_u, 0 * rho_data_u), spin = 1)[0] * rho_data_u[0]
    cd = eval_xc(FUNCTIONAL, (rho_data_d, 0 * rho_data_d), spin = 1)[0] * rho_data_d[0]
    return (ced - cu - cd) / (rhot * ldac - 1e-20)
    #return ced / (ldac - 1e-12) - 1

def y_to_ced_os(y, rho_data_u, rho_data_d):
    rhot = rho_data_u[0] + rho_data_d[0]
    ldac = eval_xc(',LDA_C_PW_MOD', (rho_data_u, rho_data_d), spin = 1)[0]
    FUNCTIONAL = ',MGGA_C_SCAN'
    cu = eval_xc(FUNCTIONAL, (rho_data_u, 0 * rho_data_u), spin = 1)[0] * rho_data_u[0]
    cd = eval_xc(FUNCTIONAL, (rho_data_d, 0 * rho_data_d), spin = 1)[0] * rho_data_d[0]
    return (y) * (rhot * ldac) + cu + cd

def identity(y, rho_data_u, rho_data_d, plus_one = False):
    return y

def get_big_desc(X, num, plus_one):
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
    #a = (np.log(2) - 1) / (2 * np.pi**2)
    a = 1.0
    b = 20.4562557
    if plus_one:
        desc[:,num] = a * np.log(1 + b * invrs + b * invrs**2)
        return desc[:,:num+1]
    else:
        desc[:,num-1] = a * np.log(1 + b * invrs + b * invrs**2)
        return desc[:,:num]

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
        print('SHAPE', self.X.shape)
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
        print('SHAPE', X.shape)
        y = self.gp.predict(X, return_std = return_std)
        if return_std:
            raise NotImplementedError('Uncertainty for correlation not fully implemented')
        else:
            return self.y_to_xed(y, rho_data_u, rho_data_d)


def get_desc_tot(Xu, Xd, rho_data_u, rho_data_d, num = 1):
    tmp = (Xu[:,1]**2 + Xd[:,1]**2) / 2
    X = (Xu + Xd) / 2
    X[:,1] = np.sqrt(tmp)
    X = get_big_desc3(X, num)
    zeta = (Xu[:,0] - Xd[:,0]) / (Xu[:,0] + Xd[:,0] + 1e-20)
    zeta = zeta**2
    rhot = rho_data_u[0] + rho_data_d[0]
    X = np.hstack([rhot.reshape(-1,1), X, zeta.reshape(-1,1)])
    ldac = eval_xc(',LDA_C_PW_MOD', (rho_data_u, rho_data_d), spin = 1)[0] * rhot - 1e-20
    FUNCTIONAL = ',MGGA_C_SCAN'
    cu = eval_xc(FUNCTIONAL, (rho_data_u, 0 * rho_data_u), spin = 1)[0] * rho_data_u[0]
    cd = eval_xc(FUNCTIONAL, (rho_data_d, 0 * rho_data_d), spin = 1)[0] * rho_data_d[0]
    co = eval_xc(FUNCTIONAL, (rho_data_u, rho_data_d), spin = 1)[0] \
            * (rho_data_u[0] + rho_data_d[0])
    co -= cu + cd
    X[:,1] = co / ldac
    return X

def get_desc_tot3(Xu, Xd, rho_data_u, rho_data_d, num = 1):
    tmp = (Xu[:,1]**2 + Xd[:,1]**2) / 2
    X = (Xu + Xd) / 2
    X[:,1] = np.sqrt(tmp)
    X = get_big_desc3(X, num)
    rhot = rho_data_u[0] + rho_data_d[0]
    X = np.hstack([rhot.reshape(-1,1), X])
    ldac = eval_xc(',LDA_C_PW_MOD', (rho_data_u, rho_data_d), spin = 1)[0] * rhot - 1e-20
    FUNCTIONAL = ',MGGA_C_SCAN'
    cu = eval_xc(FUNCTIONAL, (rho_data_u, 0 * rho_data_u), spin = 1)[0] * rho_data_u[0]
    cd = eval_xc(FUNCTIONAL, (rho_data_d, 0 * rho_data_d), spin = 1)[0] * rho_data_d[0]
    co = eval_xc(FUNCTIONAL, (rho_data_u, rho_data_d), spin = 1)[0] \
            * (rho_data_u[0] + rho_data_d[0])
    co -= cu + cd
    X[:,1] = co / ldac
    return X

def get_desc_tot5(Xu, Xd, rho_data_u, rho_data_d, num = 1):
    tmp = (Xu[:,1]**2 + Xd[:,1]**2) / 2
    X = (Xu + Xd) / 2
    X[:,1] = np.sqrt(tmp)
    X = get_big_desc3(X, num)
    zeta = (Xu[:,0] - Xd[:,0]) / (Xu[:,0] + Xd[:,0] + 1e-20)
    zeta = zeta**2
    rhot = rho_data_u[0] + rho_data_d[0]
    X = np.hstack([rhot.reshape(-1,1), X, zeta.reshape(-1,1)])
    ldac = eval_xc(',LDA_C_PW_MOD', (rho_data_u, rho_data_d), spin = 1)[0] * rhot - 1e-20
    FUNCTIONAL = ',MGGA_C_SCAN'
    cu = eval_xc(FUNCTIONAL, (rho_data_u, 0 * rho_data_u), spin = 1)[0] * rho_data_u[0]
    cd = eval_xc(FUNCTIONAL, (rho_data_d, 0 * rho_data_d), spin = 1)[0] * rho_data_d[0]
    co = eval_xc(FUNCTIONAL, (rho_data_u, rho_data_d), spin = 1)[0] \
            * (rho_data_u[0] + rho_data_d[0])
    co -= cu + cd
    X[:,1] = co / ldac
    return X

def get_desc_tot6(Xu, Xd, rho_data_u, rho_data_d, num = 1):
    tmp = (Xu[:,1]**2 + Xd[:,1]**2) / 2
    X = (Xu + Xd) / 2
    X[:,1] = np.sqrt(tmp)
    X = get_big_desc3(X, num)
    zeta = (Xu[:,0] - Xd[:,0]) / (Xu[:,0] + Xd[:,0] + 1e-20)
    zeta = zeta**2
    rhot = rho_data_u[0] + rho_data_d[0]
    X = np.hstack([rhot.reshape(-1,1), X, zeta.reshape(-1,1)])
    ldac = eval_xc(',LDA_C_PW_MOD', (rho_data_u, rho_data_d), spin = 1)[0] * rhot - 1e-20
    FUNCTIONAL = ',MGGA_C_SCAN'
    cu = eval_xc(FUNCTIONAL, (rho_data_u, 0 * rho_data_u), spin = 1)[0] * rho_data_u[0]
    cd = eval_xc(FUNCTIONAL, (rho_data_d, 0 * rho_data_d), spin = 1)[0] * rho_data_d[0]
    co = eval_xc(FUNCTIONAL, (rho_data_u, rho_data_d), spin = 1)[0] \
            * (rho_data_u[0] + rho_data_d[0])
    #co -= cu + cd
    X[:,1] = co / ldac
    return X

def get_desc_tot4(Xu, Xd, rho_data_u, rho_data_d, num = 1):
    NS = 4
    descu = get_big_desc(Xu, NS, True)
    descd = get_big_desc(Xd, NS, True)
    tmp = (Xu[:,1]**2 + Xd[:,1]**2) / 2
    X = (Xu + Xd) / 2
    X[:,1] = np.sqrt(tmp)
    X = get_big_desc(X, num, True)
    zeta = (Xu[:,0] - Xd[:,0]) / (Xu[:,0] + Xd[:,0] + 1e-20)
    zeta = zeta**2
    rhot = rho_data_u[0] + rho_data_d[0]
    X = np.hstack([rhot.reshape(-1,1), X, descu, descd])
    ldac = eval_xc(',LDA_C_PW_MOD', (rho_data_u, rho_data_d), spin = 1)[0] * rhot - 1e-20
    FUNCTIONAL = ',MGGA_C_SCAN'
    cu = eval_xc(FUNCTIONAL, (rho_data_u, 0 * rho_data_u), spin = 1)[0] * rho_data_u[0]
    cd = eval_xc(FUNCTIONAL, (rho_data_d, 0 * rho_data_d), spin = 1)[0] * rho_data_d[0]
    co = eval_xc(FUNCTIONAL, (rho_data_u, rho_data_d), spin = 1)[0] \
            * (rho_data_u[0] + rho_data_d[0])
    co -= cu + cd
    X[:,1] = co / ldac
    X[:,num+2] = cu / ldac
    X[:,num+3+NS] = cd / ldac
    return X

def get_desc_tot2(Xu, Xd, rho_data_u, rho_data_d, num = 1):
    X = (Xu + Xd) / 2
    X = get_big_desc3(X, num)
    zeta = (Xu[:,0] - Xd[:,0]) / (Xu[:,0] + Xd[:,0] + 1e-20)
    zeta = zeta**2
    rhot = rho_data_u[0] + rho_data_d[0]
    X = np.hstack([rhot.reshape(-1,1), X, zeta.reshape(-1,1)])
    ldac = eval_xc(',LDA_C_PW_MOD', (rho_data_u, rho_data_d), spin = 1)[0] * rhot - 1e-20
    FUNCTIONAL = ',MGGA_C_SCAN'
    cu = eval_xc(FUNCTIONAL, (rho_data_u, 0 * rho_data_u), spin = 1)[0] * rho_data_u[0]
    cd = eval_xc(FUNCTIONAL, (rho_data_d, 0 * rho_data_d), spin = 1)[0] * rho_data_d[0]
    co = eval_xc(FUNCTIONAL, (rho_data_u, rho_data_d), spin = 1)[0] \
            * (rho_data_u[0] + rho_data_d[0])
    X[:,1] = co / ldac
    return X


class CorrGPR3(CorrGPR):

    def __init__(self, num_desc):
        const = ConstantKernel(1.0)
        ind = np.arange(num_desc + 2)
        #rbf = PartialARBF([0.3] * (num_desc), active_dims = ind[2:num_desc+2])
        rbf = PartialARBF(order = 2, length_scale = [0.3] * num_desc,
                scale_bounds="fixed",
                scale = [0.14, 0.01, 1.07], active_dims = ind[2:num_desc+2])
        #rbf = PartialARBF(order = 2, length_scale = [0.121, 0.94, 0.175, 0.92,\
        #        0.207, 0.299, 0.163, 0.594, 0.102, 0.527],
        #        scale = [1.0, 1.0, 1.0], active_dims = ind[[2,3,4,5,6,7,8,9,10,12]])
        #rbf *= SingleRBF(length_scale=0.2, index = 11)
        rhok1 = FittedDensityNoise(decay_rate = 2.0)
        rhok2 = FittedDensityNoise(decay_rate = 600.0)
        wk = WhiteKernel(noise_level=4.0e-4, noise_level_bounds=(1e-06, 1.0e5))
        wk1 = WhiteKernel(noise_level = 0.002, noise_level_bounds=(1e-05, 1.0e5))
        wk2 = WhiteKernel(noise_level = 0.02, noise_level_bounds=(1e-05, 1.0e5))
        noise_kernel = wk + wk1 * rhok1 + wk2 * Exponentiation(rhok2, 2)
        dot = SingleDot(sigma_0=0.0, sigma_0_bounds='fixed', index = 1)
        print ("CHECK FIXED", rbf.hyperparameter_length_scale.fixed, rbf.hyperparameter_scale.fixed, dot.hyperparameter_sigma_0.fixed)
        cov_kernel = dot * rbf
        #cov_kernel = dot + dot * rbf
        init_kernel = cov_kernel + noise_kernel
        print(init_kernel.theta, init_kernel.bounds)
        super(CorrGPR, self).__init__(num_desc,
                       descriptor_getter = get_desc_tot6,
                       xed_y_converter = (ced_to_y_lda, y_to_ced_lda),
                       init_kernel = init_kernel)


class CorrGPR4(CorrGPR):

    def __init__(self, num_desc):
        const = ConstantKernel(1.0)
        NS = 4
        ind = np.arange(num_desc + 2 + 2 * (NS + 1))

        #rbf = PartialARBF(order = 2, length_scale = [0.3] * num_desc,
        rbf = PartialARBF(order = 2, length_scale = [0.165, 0.606, 0.161,\
                0.198, 0.192, 0.277, 0.141, 0.319, 0.115, 0.156, 0.306][:num_desc-1] + [0.297],
                length_scale_bounds = 'fixed',
                scale = [0.14, 0.01, 1.07], active_dims=ind[2:num_desc+2])
        dot = SingleDot(sigma_0=0.0, sigma_0_bounds='fixed', index = 1)
        covos = dot * rbf

        rbfss = PartialARBF(order = 2, length_scale = [0.165, 0.161, 0.141, 0.297],
                length_scale_bounds='fixed',
                scale=[0.25, 0.01, 0.041], active_dims=ind[1:NS+1])
        dotss = SingleDot(sigma_0=0.0, sigma_0_bounds='fixed', index = 0)
        covss = dotss * rbfss
        covss = SpinSymKernel(covss, ind[num_desc+2:num_desc+3+NS],
                                     ind[num_desc+3+NS:num_desc+4+2*NS])

        cov_kernel = covos + covss

        rhok1 = FittedDensityNoise(decay_rate = 2.0)
        rhok2 = FittedDensityNoise(decay_rate = 600.0)
        wk = WhiteKernel(noise_level=4.0e-4, noise_level_bounds=(1e-06, 1.0e5))
        wk1 = WhiteKernel(noise_level = 0.002, noise_level_bounds=(1e-05, 1.0e5))
        wk2 = WhiteKernel(noise_level = 0.02, noise_level_bounds=(1e-05, 1.0e5))
        noise_kernel = wk + wk1 * rhok1 + wk2 * Exponentiation(rhok2, 2)

        init_kernel = cov_kernel + noise_kernel
        super(CorrGPR, self).__init__(num_desc,
                       descriptor_getter = get_desc_tot4,
                       xed_y_converter = (ced_to_y_scan, y_to_ced_scan),
                       init_kernel = init_kernel)


class CorrGPR5(CorrGPR):

    def __init__(self, num_desc):
        const = ConstantKernel(1.0)
        NS = 4
        ind = np.arange(num_desc + 2 + 2 * (NS + 1))

        #rbf = PartialARBF(order = 2, length_scale = [0.3] * num_desc,
        rbf = PartialARBF(order = 2, length_scale = [0.165, 0.606, 0.161,\
                0.198, 0.192, 0.277, 0.141, 0.319, 0.115, 0.156, 0.306][:num_desc-1],
                length_scale_bounds = 'fixed',
                scale = [0.14, 0.01, 1.07], active_dims=ind[2:num_desc+1])
        rbf *= SingleRBF(length_scale=0.3, index=num_desc+2)
        dot = SingleDot(sigma_0=0.0, sigma_0_bounds='fixed', index = 1)
        covos = dot * rbf

        rbfss = PartialARBF(order = 2, length_scale = [0.165, 0.161, 0.141],
                length_scale_bounds='fixed',
                scale=[0.25, 0.01, 0.041], active_dims=ind[1:NS])
        rbfss *= SingleRBF(length_scale=0.3, index = NS+1)
        dotss = SingleDot(sigma_0=0.0, sigma_0_bounds='fixed', index = 0)
        covss = dotss * rbfss
        covss = SpinSymKernel(covss, ind[num_desc+2:num_desc+3+NS],
                                     ind[num_desc+3+NS:num_desc+4+2*NS])

        cov_kernel = covos + covss

        rhok1 = FittedDensityNoise(decay_rate = 2.0)
        rhok2 = FittedDensityNoise(decay_rate = 600.0)
        wk = WhiteKernel(noise_level=4.0e-4, noise_level_bounds=(1e-06, 1.0e5))
        wk1 = WhiteKernel(noise_level = 0.002, noise_level_bounds=(1e-05, 1.0e5))
        wk2 = WhiteKernel(noise_level = 0.02, noise_level_bounds=(1e-05, 1.0e5))
        noise_kernel = wk + wk1 * rhok1 + wk2 * Exponentiation(rhok2, 2)

        init_kernel = cov_kernel + noise_kernel
        super(CorrGPR, self).__init__(num_desc,
                       descriptor_getter = get_desc_tot4,
                       xed_y_converter = (ced_to_y_scan, y_to_ced_scan),
                       init_kernel = init_kernel)
