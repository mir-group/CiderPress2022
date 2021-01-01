from mldftdat.models.gp import DFTGPR
from mldftdat.density import *
from mldftdat.data import *
from mldftdat.models.kernels import *
import numpy as np
from pyscf.dft.libxc import eval_xc
from sklearn.gaussian_process.kernels import *
from mldftdat.pyscf_utils import GG_SMUL, GG_AMUL, GG_AMIN

SCALE_FAC = (6 * np.pi**2)**(2.0/3) / (16 * np.pi)

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

def get_edmgga_descriptors(X, rho_data, num=1):
    X = X.copy()
    X[:,2] = np.sinh(1 / (1 + X[:,2]**2))
    return np.arcsinh(X[:,(1,2,4,5,8,6,12,15,16,13,14)[:num]])

def chachiyo_fx(s2):
    c = 4 * np.pi / 9
    x = c * np.sqrt(s2)
    dx = c / (2 * np.sqrt(s2))
    Pi = np.pi
    mu = 8.0 / 27
    Log = np.log
    const = 1 / (1 + mu * s2)
    chfx = (3*x**2 + Pi**2*Log(const + x))/((Pi**2 + 3*x)*Log(const + x))
    dchfx = (-3*x**2*(Pi**2 + 3*x) + 3*x*(1 + x)*(2*Pi**2 + 3*x)*Log(1 + x) - 3*Pi**2*(1 + x)*Log(1 + x)**2)/((1 + x)*(Pi**2 + 3*x)**2*Log(1 + x)**2)
    dchfx *= dx
    chfx[s2<1e-8] = 1 + 8 * s2[s2<1e-8] / 27
    dchfx[s2<1e-8] = 8.0 / 27
    return chfx, dchfx

def xed_to_y_chachiyo(xed, rho_data):
    rho, s, alpha = get_dft_input2(rho_data)[:3]
    fac = SCALE_FAC
    scale = 1 + GG_SMUL * fac * s**2 + GG_AMUL * 0.6 * fac * (alpha - 1)
    #pbex = eval_xc('GGA_X_CHACHIYO,', rho_data)[0] * rho_data[0]
    y = get_y_from_xed(xed, rho_data[0])
    y += 1 - chachiyo_fx(s**2)[0]
    return y / (1 + 1e-2 * (scale-1)**2)
    #return (xed - pbex) / (ldax(rho_data[0]) - 1e-12) / (1 + 1e-2 * (scale-1)**2)

def xed_to_y_chachiyo(xed, rho_data):
    pbex = eval_xc('GGA_X_CHACHIYO,', rho_data)[0] * rho_data[0]
    return (xed - pbex) / (ldax(rho_data[0]) - 1e-16)

def y_to_xed_chachiyo(y, rho_data):
    yp = y * ldax(rho_data[0])
    pbex = eval_xc('GGA_X_CHACHIYO,', rho_data)[0] * rho_data[0]
    return yp + pbex

def xed_to_y_chr(xed, rho_data):
    rho, s, alpha = get_dft_input2(rho_data)[:3]
    fac = SCALE_FAC
    scale = 1 + GG_SMUL * fac * s**2 + GG_AMUL * 0.6 * fac * (alpha - 1)
    pbex = eval_xc('GGA_X_CHACHIYO,', rho_data)[0] * rho_data[0]
    return (xed - pbex) / pbex / (1 + 1e-2 * (scale-1)**2)

def y_to_xed_chr(y, rho_data):
    pbex = eval_xc('GGA_X_CHACHIYO,', rho_data)[0] * rho_data[0]
    return (y + 1) * pbex

class PBEGPR(DFTGPR):

    def __init__(self, num_desc, init_kernel = None, use_algpr = False):
        super(PBEGPR, self).__init__(num_desc, descriptor_getter = None,
                       xed_y_converter = (xed_to_y_pbe, y_to_xed_pbe),
                       init_kernel = init_kernel, use_algpr = use_algpr)


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


def get_rho_and_edmgga_descriptors(X, rho_data, num=1):
    X = get_edmgga_descriptors(X, rho_data, num)
    X[:,0] = tail_fx(rho_data)
    X = np.append(rho_data[0].reshape(-1,1), X, axis=1)
    return X

def get_rho_and_edmgga_descriptors2(X, rho_data, num=1):
    X = get_edmgga_descriptors(X, rho_data, num)
    X = np.append(edmgga(rho_data).reshape(-1,1), X, axis=1)
    X = np.append(rho_data[0].reshape(-1,1), X, axis=1)
    return X

def get_edmgga_descriptors3(X, rho_data, num=1):
    return np.arcsinh(X[:,(1,2,4,5,8,15,16,6,12,13,14)[:num]])

def get_rho_and_edmgga_descriptors3(X, rho_data, num=1):
    X = get_edmgga_descriptors3(X, rho_data, num)
    X = np.append(edmgga(rho_data).reshape(-1,1), X, axis=1)
    X = np.append(rho_data[0].reshape(-1,1), X, axis=1)
    return X

def get_edmgga_descriptors4(X, rho_data, num=1):
    comp = np.array([[ 0.60390014,  0.57011819,  0.55701874],
        [-0.08667547, -0.64772484,  0.75692793],
        [-0.79233326,  0.50538874,  0.34174586]]).T
    X[:,(4,15,16)] /= np.array([2.28279105, 4.5451314, 0.57933925])
    X[:,(4,15,16)] = np.dot(X[:,(4,15,16)], comp)
    return np.arcsinh(X[:,(1,2,4,5,8,15,16,6,12,13,14)[:num]])

def get_rho_and_edmgga_descriptors4(X, rho_data, num=1):
    X = get_edmgga_descriptors4(X, rho_data, num)
    X = np.append(edmgga(rho_data).reshape(-1,1), X, axis=1)
    X = np.append(rho_data[0].reshape(-1,1), X, axis=1)
    return X

def get_edmgga_descriptors5(X, rho_data, num=1):
    return np.arcsinh(X[:,(1,2,4,5,8,6,12,16,15,6,12,13,14)[:num]])

def get_rho_and_edmgga_descriptors5(X, rho_data, num=1):
    X = get_edmgga_descriptors5(X, rho_data, num)
    X = np.append(rho_data[0].reshape(-1,1), X, axis=1)
    return X

def get_edmgga_descriptors6(X, rho_data, num=1):
    return np.arcsinh(X[:,(2,4,5,8,6,12,16,15,6,12,13,14)[:num]])

def get_rho_and_edmgga_descriptors6(X, rho_data, num=1):
    X = get_edmgga_descriptors6(X, rho_data, num-1)
    X = np.append(edmgga(rho_data).reshape(-1,1), X, axis=1)
    X = np.append(rho_data[0].reshape(-1,1), X, axis=1)
    return X

def get_edmgga_descriptors7(X, rho_data, num=1):
    X = np.arcsinh(X[:,(2,4,5,8,6,12,15,16,6,12,13,14)])
    X[:,2] -= X[:,1]
    X[:,6] -= X[:,1]
    X[:,7] -= X[:,1]
    return X[:,:num]

def get_rho_and_edmgga_descriptors7(X, rho_data, num=1):
    X = get_edmgga_descriptors7(X, rho_data, num-1)
    X = np.append(edmgga(rho_data).reshape(-1,1), X, axis=1)
    X = np.append(rho_data[0].reshape(-1,1), X, axis=1)
    return X

def get_edmgga_descriptors8(X, rho_data, num=1):
    return np.arcsinh(X[:,(1,2,4,5,8,15,16,12,6,12,13,14)[:num]])

def get_rho_and_edmgga_descriptors8(X, rho_data, num=1):
    X = get_edmgga_descriptors8(X, rho_data, num)
    X = np.append(rho_data[0].reshape(-1,1), X, axis=1)
    return X

def get_edmgga_descriptors9(X, rho_data, num=1):
    X[:,1] = X[:,1]**2
    #X[:,2] = np.sinh(1 / (1 + X[:,2]**2))
    X[:,5] -= X[:,4]
    X[:,15] -= X[:,4]
    X[:,16] -= X[:,4]
    return np.arcsinh(X[:,(1,2,4,5,8,15,16,6,12,13,14)[:num]])

def get_rho_and_edmgga_descriptors9(X, rho_data, num=1):
    X = get_edmgga_descriptors9(X, rho_data, num)
    X = np.append(rho_data[0].reshape(-1,1), X, axis=1)
    return X

def get_edmgga_descriptors10(X, rho_data, num=1):
    #X[:,1] = X[:,1]**2
    #X[:,2] = np.sinh(1 / (1 + X[:,2]**2))
    #X[:,5] -= X[:,4]
    #X[:,15] -= X[:,4]
    #X[:,16] -= X[:,4]
    return np.arcsinh(X[:,(1,2,4,5,8,15,16,14,12,13,14)[:num]])

def get_rho_and_edmgga_descriptors10(X, rho_data, num=1):
    X = get_edmgga_descriptors10(X, rho_data, num)
    X = np.append(rho_data[0].reshape(-1,1), X, axis=1)
    return X

def get_edmgga_descriptors11(X, rho_data, num=1):
    #X[:,1] = X[:,1]**2
    #X[:,2] = np.sinh(1 / (1 + X[:,2]**2))
    return np.arcsinh(X[:,(1,2,4,15,16,5,8,12,12,13,14)[:num]])

def get_rho_and_edmgga_descriptors11(X, rho_data, num=1):
    X = get_edmgga_descriptors11(X, rho_data, num)
    X = np.append(rho_data[0].reshape(-1,1), X, axis=1)
    return X

def get_edmgga_descriptors12(X, rho_data, num=1):
    return np.arcsinh(X[:,(1,2,4,5,8,15,12,16,6,13,14)[:num]])

def get_rho_and_edmgga_descriptors12(X, rho_data, num=1):
    X = get_edmgga_descriptors12(X, rho_data, num)
    X = np.append(rho_data[0].reshape(-1,1), X, axis=1)
    return X

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

    gammax = 0.004 * sprefac**2
    gamma1 = 0.01552
    gamma2 = 0.01617
    gamma0a = 0.5
    gamma0b = 0.125
    gamma0c = 2.0

    gammax = 0.0537
    gamma1 = 0.0542
    gamma2 = 0.0394
    gamma0a = 0.0807
    gamma0b = 0.8126
    gamma0c = 0.2545

    gammax = 0.5469
    gamma1 = 0.0635
    gamma2 = 0.0524
    gamma0a = 0.1687
    gamma0b = 1.0166
    gamma0c = 0.2072

    gammax = 0.4219
    gamma1 = 0.0566
    gamma2 = 0.0327
    gamma0a = 0.3668
    gamma0b = 0.9966
    gamma0c = 0.2516

    s = X[:,1]
    p, alpha = X[:,1]**2, X[:,2]

    fac = (6 * np.pi**2)**(2.0/3) / (16 * np.pi)
    scale = np.sqrt(1 + GG_SMUL * fac * p + GG_AMUL * 0.6 * fac * (alpha - 1))

    desc = np.zeros((X.shape[0], 12))
    refs = gammax / (1 + gammax * s**2)
    ref0a = gamma0a / (1 + X[:,4] * scale**3 * gamma0a)
    ref0b = gamma0b / (1 + X[:,15] * scale**3 * gamma0b)
    ref0c = gamma0c / (1 + X[:,16] * scale**3 * gamma0c)
    ref1 = gamma1 / (1 + gamma1 * X[:,5]**2 * scale**6)
    ref2 = gamma2 / (1 + gamma2 * X[:,8] * scale**6)

    desc[:,0] = X[:,0]
    desc[:,1] = s**2 * refs
    desc[:,2] = 2 / (1 + alpha**2) - 1.0
    desc[:,3] = (X[:,4] * scale**3 - 2.0) * ref0a
    desc[:,4] = X[:,5]**2 * scale**6 * ref1
    desc[:,5] = X[:,8] * scale**6 * ref2
    desc[:,6] = X[:,12] * scale**3 * refs * np.sqrt(ref2)
    desc[:,7] = X[:,6] * scale**3 * np.sqrt(refs) * np.sqrt(ref1)
    desc[:,8] = (X[:,15] * scale**3 - 8.0) * ref0b
    desc[:,9] = (X[:,16] * scale**3 - 0.5) * ref0c
    desc[:,10] = (X[:,13] * scale**6) * np.sqrt(refs) * np.sqrt(ref1) * np.sqrt(ref2)
    desc[:,11] = (X[:,14] * scale**9) * np.sqrt(ref2) * ref1
    return desc[:,:num+1]


def get_big_desc5(X, num):

    gammax = 0.1838
    gamma1 = 0.0440
    gamma2 = 0.0161
    gamma0a = 0.4658
    gamma0b = 0.8318
    gamma0c = 0.3535

    gammax =  0.023363456685874712
    gamma1 =  0.0419231549751321
    gamma2 =  0.01778256399976796
    gamma0a =  0.6751313483086911
    gamma0b =  1.6063456594287293
    gamma0c =  0.19782664430722932

    s = X[:,1]
    p, alpha = X[:,1]**2, X[:,2]

    fac = (6 * np.pi**2)**(2.0/3) / (16 * np.pi)
    scale = np.sqrt(1 + GG_SMUL * fac * p + GG_AMUL * 0.6 * fac * (alpha - 1))

    desc = np.zeros((X.shape[0], 12))
    refs = gammax / (1 + gammax * s**2)
    ref0a = gamma0a / (1 + X[:,4] * gamma0a)
    ref0b = gamma0b / (1 + X[:,15] * gamma0b)
    ref0c = gamma0c / (1 + X[:,16] * gamma0c)
    ref1 = gamma1 / (1 + gamma1 * X[:,5]**2)
    ref2 = gamma2 / (1 + gamma2 * X[:,8])

    desc[:,0] = X[:,0]
    desc[:,1] = s**2 * refs
    desc[:,2] = 2 / (1 + alpha**2) - 1.0
    desc[:,3] = (X[:,4] - 2.0 / scale**3) * ref0a
    desc[:,4] = X[:,5]**2 * ref1
    desc[:,5] = X[:,8] * ref2
    desc[:,6] = X[:,12] * refs * np.sqrt(ref2)
    desc[:,7] = X[:,6] * np.sqrt(refs) * np.sqrt(ref1)
    desc[:,8] = (X[:,15] - 8.0 / scale**3) * ref0b
    desc[:,9] = (X[:,16] - 0.5 / scale**3) * ref0c
    desc[:,10] = (X[:,13]) * np.sqrt(refs) * np.sqrt(ref1) * np.sqrt(ref2)
    desc[:,11] = (X[:,14]) * np.sqrt(ref2) * ref1
    return desc[:,:num+1]


def get_big_desc6(X, num):
    """
    gammax = 0.03396679161527282
    gamma1 = 0.025525996367805805
    gamma2 = 0.015353511288718948
    gamma0a = 0.47032113660384833
    gamma0b = 1.1014410669536636
    gamma0c = 0.37588448262415936
    center0a = 0.48470667994514244
    center0b = 0.8980790815244916
    center0c = 0.15820823165989775
    """
    gammax = 0.12011685392376696
    gamma1 = 0.025802574385367972
    gamma2 = 0.01654121930252892
    gamma0a = 0.4891146891376963
    gamma0b = 0.8342344450082123
    gamma0c = 0.41209749093646153
    center0a = 0.4944974475751677
    center0b = 0.8696877487558009
    center0c = 0.17084611732524574

    s = X[:,1]
    p, alpha = X[:,1]**2, X[:,2]

    fac = (6 * np.pi**2)**(2.0/3) / (16 * np.pi)

    desc = np.zeros((X.shape[0], 12))
    refs = gammax / (1 + gammax * s**2)
    ref0a = gamma0a / (1 + X[:,4] * gamma0a)
    ref0b = gamma0b / (1 + X[:,15] * gamma0b)
    ref0c = gamma0c / (1 + X[:,16] * gamma0c)
    ref1 = gamma1 / (1 + gamma1 * X[:,5]**2)
    ref2 = gamma2 / (1 + gamma2 * X[:,8])

    desc[:,0] = X[:,0]
    desc[:,1] = s**2 * refs
    desc[:,2] = 2 / (1 + alpha**2) - 1.0
    desc[:,3] = X[:,4] * ref0a - center0a
    desc[:,4] = X[:,5]**2 * ref1
    desc[:,5] = X[:,8] * ref2
    desc[:,6] = X[:,12] * refs * np.sqrt(ref2)
    desc[:,7] = X[:,6] * np.sqrt(refs) * np.sqrt(ref1)
    desc[:,8] = X[:,15] * ref0b - center0b
    desc[:,9] = X[:,16] * ref0c - center0c
    desc[:,10] = (X[:,13]) * np.sqrt(refs) * np.sqrt(ref1) * np.sqrt(ref2)
    desc[:,11] = (X[:,14]) * np.sqrt(ref2) * ref1
    return desc[:,:num+1]


def get_big_desc4(X, num):
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
    desc[:,1] = np.arcsinh(s**2)
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

def get_rho_and_edmgga_descriptors13(X, rho_data, num=1):
    X = get_big_desc2(X, num)
    #X = np.append(rho_data[0].reshape(-1,1), X, axis=1)
    return X

def get_rho_and_edmgga_descriptors14(X, rho_data, num=1):
    X = get_big_desc6(X, num)
    #X = np.append(rho_data[0].reshape(-1,1), X, axis=1)
    return X


def get_rho_and_edmgga_descriptors15(X, rho_data, num=1):
    X = get_big_desc4(X, num)
    #X = np.append(rho_data[0].reshape(-1,1), X, axis=1)
    return X

class NoisyEDMGPR(EDMGPR):

    def __init__(self, num_desc, use_algpr = False, norm_feat = False):
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
        if not norm_feat:
            rbf = PartialRBF([0.3, 0.4, 0.6696, 0.6829, 0.6, 0.6, 1.0, 1.0, 1.0, 1.0][:num_desc],
                         length_scale_bounds=(1.0e-5, 1.0e5), start = 1)
        else:
            #const = ConstantKernel(1.0)
            const = ConstantKernel(12.1696)
            #rbf = PartialRBF(([0.232, 1.02, 0.279, 0.337, 0.526, 0.34, 0.333, 0.235, 0.237, 1.0, 1.0, 1.0, 1.0])[:num_desc],
            #rbf = PartialRBF(([0.6, 1.02, 0.279, 0.337, 0.526, 0.34, 0.333, 0.235, 0.237, 1.0, 1.0, 1.0, 1.0])[:num_desc],
            rbf = PartialRBF([0.2961, 0.9890, 0.3719, 0.4484, 0.4733, 0.6631, 0.5936, 0.6224, 0.2370,\
                         0.5662][:num_desc],
                         #[0.3282, 1.9472, 0.2135, 0.3494, 0.2396, 0.3231, 0.3301, 0.1146, 0.2129, 0.2809][]
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
                       descriptor_getter = get_rho_and_edmgga_descriptors14 if norm_feat\
                               else get_rho_and_edmgga_descriptors,
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

class SmoothEDMGPR(EDMGPR):

    def __init__(self, num_desc, use_algpr = False):
        const = ConstantKernel(0.2)
        #rbf = PartialRBF([1.0] * (num_desc + 1),
        #rbf = PartialRBF([0.299, 0.224, 0.177, 0.257, 0.624][:num_desc+1],
        #rbf = PartialRBF([0.395, 0.232, 0.297, 0.157, 0.468, 1.0][:num_desc+1],
        #                 length_scale_bounds=(1.0e-5, 1.0e5), start = 1)
        rbf = SingleRBF(length_scale=0.15, index = 1)
        dot = PartialDot(start = 2)
        rhok1 = FittedDensityNoise(decay_rate = 2.0)
        rhok2 = FittedDensityNoise(decay_rate = 600.0)
        wk = WhiteKernel(noise_level=1.0e-4, noise_level_bounds=(1e-06, 1.0e5))
        wk1 = WhiteKernel(noise_level = 0.002, noise_level_bounds=(1e-05, 1.0e5))
        wk2 = WhiteKernel(noise_level = 0.02, noise_level_bounds=(1e-05, 1.0e5))
        cov_kernel = const * rbf * Exponentiation(dot, 3)
        noise_kernel = wk + wk1 * rhok1 + wk2 * Exponentiation(rhok2, 2)
        init_kernel = cov_kernel + noise_kernel
        super(EDMGPR, self).__init__(num_desc,
                       descriptor_getter = get_edmgga_descriptors2,
                       xed_y_converter = (xed_to_y_lda, y_to_xed_lda),
                       init_kernel = init_kernel, use_algpr = use_algpr)

    def is_uncertain(self, x, y, threshold_factor = 1.2, low_noise_bound = 0.002):
        threshold = max(low_noise_bound, np.sqrt(self.gp.kernel_.k2(x))) * threshold_factor
        y_pred = self.gp.predict(x)
        return np.abs(y - y_pred) > threshold


class AddEDMGPR(EDMGPR):

    def __init__(self, num_desc, use_algpr = False, norm_feat = False):
        const = ConstantKernel(0.2)
        order = 3
        if not norm_feat:
            rbf = PartialARBF([0.3, 0.4, 0.6696, 0.6829, 0.6, 0.6, 1.0, 1.0, 1.0, 1.0][:num_desc],
                         length_scale_bounds=(1.0e-5, 1.0e5), start = 1)
        else:
            const = ConstantKernel(0.527**2)
            rbf = PartialARBF(order = order, length_scale = [0.232, 0.436, 0.222, 0.1797, 0.193, 0.232,\
                         0.1976, 0.312, 0.1716, 0.1886][:num_desc],
                         scale = [2e-5, 0.536 * 0.527**2, 0.0428 * 0.527**2, 0.05 * 0.527**2],
                         length_scale_bounds=(1.0e-5, 1.0e5), start = 1)
        rhok1 = FittedDensityNoise(decay_rate = 2.0)
        rhok2 = FittedDensityNoise(decay_rate = 600.0)
        wk = WhiteKernel(noise_level=3.0e-5, noise_level_bounds=(1e-06, 1.0e5))
        wk1 = WhiteKernel(noise_level = 0.002, noise_level_bounds=(1e-05, 1.0e5))
        wk2 = WhiteKernel(noise_level = 0.02, noise_level_bounds=(1e-05, 1.0e5))
        cov_kernel = rbf
        noise_kernel = wk + wk1 * rhok1 + wk2 * Exponentiation(rhok2, 2)
        init_kernel = cov_kernel + noise_kernel
        super(EDMGPR, self).__init__(num_desc,
                       descriptor_getter = get_rho_and_edmgga_descriptors14 if norm_feat\
                               else get_rho_and_edmgga_descriptors,
                       xed_y_converter = (xed_to_y_lda, y_to_xed_lda),
                       init_kernel = init_kernel, use_algpr = use_algpr)

    def is_uncertain(self, x, y, threshold_factor = 2, low_noise_bound = 0.002):
        threshold = max(low_noise_bound, np.sqrt(self.gp.kernel_.k2(x))) * threshold_factor
        y_pred, y_std = self.gp.predict(x, return_std=True)
        return (y_std > threshold).any()


class AddEDMGPR2(EDMGPR):

    def __init__(self, num_desc, use_algpr = False, norm_feat = False):
        const = ConstantKernel(0.2)
        order = 2
        if not norm_feat:
            rbf = PartialARBF([0.3, 0.4, 0.6696, 0.6829, 0.6, 0.6, 1.0, 1.0, 1.0, 1.0][:num_desc],
                         length_scale_bounds=(1.0e-5, 1.0e5), start = 1)
        else:
            const = ConstantKernel(0.527**2)
            #rbf = PartialARBF(order = order, length_scale = [0.388, 0.159, 0.205, 0.138, 0.134, 0.12, 0.172, 0.103, 0.126][:num_desc-1],
            #rbf = PartialARBF(order = order, length_scale = [1.8597, 0.4975, 0.6506, \
            #             0.8821, 1.2929, 0.8559, 0.8274, 0.2809, 0.8953][:num_desc-1],
            #rbf = PartialARBF(order = order, length_scale = [0.3, 2*0.127, 0.123, 0.132, \
            #             0.147, 0.17, 2*0.305, 2*0.0609, 0.154][:num_desc-1],
            rbf = PartialARBF(order = order, length_scale = [0.5, 0.2040, 0.3108, \
                         0.2987, 0.5127, 0.4741, 0.1140, 0.2166, 0.4969][:num_desc-1],
                         #scale = [0.25100654, 0.01732103, 0.02348104],
                         scale = [1e-5, 1e-5, 1e-2],
                         #scale = [0.296135,  0.0289514, 0.1114619],
                         #length_scale_bounds='fixed', scale_bounds='fixed', start = 2)
                         #length_scale_bounds='fixed', start = 2)
                         #length_scale_bounds=(1.0e-2, 1.0e1), start = 2)
                         length_scale_bounds=(6e-2, 1.0e0), scale_bounds=(1e-5, 1.0e0),
                         active_dims=[2,3,4,5,6,7,8,9,10])
        rhok1 = FittedDensityNoise(decay_rate = 46.8, decay_rate_bounds='fixed')
        rhok2 = FittedDensityNoise(decay_rate = 1e6, decay_rate_bounds='fixed')
        wk = WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-06, 1.0e5))
        wk1 = WhiteKernel(noise_level = 0.000418)#, noise_level_bounds='fixed')#=(1e-05, 1.0e5))
        wk2 = WhiteKernel(noise_level = 0.168)#, noise_level_bounds='fixed')#(1e-05, 1.0e5))
        cov_kernel = rbf * SingleRBF(length_scale=0.3, index = 1)#, length_scale_bounds='fixed')
        noise_kernel = wk + wk1 * rhok1 + wk2 * rhok2# Exponentiation(rhok2, 2)
        init_kernel = cov_kernel + noise_kernel
        super(EDMGPR, self).__init__(num_desc,
                       descriptor_getter = get_rho_and_edmgga_descriptors14 if norm_feat\
                               else get_rho_and_edmgga_descriptors,
                       xed_y_converter = (xed_to_y_chachiyo, y_to_xed_chachiyo),
                       #xed_y_converter = (xed_to_y_lda, y_to_xed_lda),
                       init_kernel = init_kernel, use_algpr = use_algpr)

    def is_uncertain(self, x, y, threshold_factor = 2, low_noise_bound = 0.002):
        threshold = max(low_noise_bound, np.sqrt(self.gp.kernel_.k2(x))) * threshold_factor
        y_pred, y_std = self.gp.predict(x, return_std=True)
        return (y_std > threshold).any()

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

