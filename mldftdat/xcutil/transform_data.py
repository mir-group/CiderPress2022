from abc import ABC, abstractmethod, abstractproperty
import numpy as np
import yaml


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
        return UMap(d['n'], d['i'], d['j'], d['k'], d['l'],
                    d['gammai'], d['gammaj'], d['gammak'])


class FeatureList():

    def __init__(self, feat_list):
        self.feat_list = feat_list
        self.nfeat = len(self.feat_list)

    def __getitem__(self, index):
        return self.feat_list[index]

    def __call__(self, xdesc):
        # xdesc (nsamp, ninp)
        xdesc = xdesc.T
        # now xdesc (ninp, nsamp)
        tdesc = np.zeros((self.nfeat, xdesc.shape[1]))
        for i in range(self.nfeat):
            self.feat_list[i].fill_feat_(tdesc, xdesc)
        return tdesc.T

    def fill_vals_(self, tdesc, xdesc):
        for i in range(self.nfeat):
            self.feat_list[i].fill_feat_(tdesc, xdesc)

    def fill_derivs_(self, dfdx, dfdy, xdesc):
        for i in range(self.nfeat):
            self.feat_list[i].fill_deriv_(dfdx, dfdy, xdesc)

    def as_dict(self):
        d = {
            'nfeat': self.nfeat,
            'feat_list': [f.as_dict() for f in self.feat_list]
        }
        return d

    def dump(self, fname):
        d = self.as_dict()
        with open(fname, 'w') as f:
            yaml.dump(d, f)

    @classmethod
    def from_dict(cls, d):
        return cls([FeatureNormalizer.from_dict(d['feat_list'][i])\
                    for i in range(len(d['feat_list']))])

    @classmethod
    def load(cls, fname):
        with open(fname, 'r') as f:
            d = yaml.load(f, Loader=yaml.Loader)
        return cls.from_dict(d)


"""
Examples of FeatureList objects:
center0a = get_vmap_heg_value(2.0, gamma0a)
center0b = get_vmap_heg_value(8.0, gamma0b)
center0c = get_vmap_heg_value(0.5, gamma0c)
lst = [
        UMap(1, 1, gammax),
        VMap(2, 2, 1, scale=2.0, center=1.0),
        VMap(3, 3, gamma0a, scale=1.0, center=center0a),
        UMap(4, 4, gamma1),
        UMap(5, 5, gamma2),
        WMap(6, 1, 5, 6, gammax, gamma2),
        XMap(7, 1, 4, 7, gammax, gamma1),
        VMap(8, 8, gamma0b, scale=1.0, center=center0b),
        VMap(9, 9, gamma0c, scale=1.0, center=center0c),
        YMap(10, 1, 4, 5, 10, gammax, gamma1, gamma2),
]
flst = FeatureList(lst)

center0a = get_vmap_heg_value(2.0, gamma0a)
center0b = get_vmap_heg_value(3.0, gamma0b)
UMap(1, 1, gammax)
VMap(2, 2, 1, scale=2.0, center=1.0)
VMap(3, 3, gamma0a, scale=1.0, center=center0a)
UMap(4, 4, gamma1)
UMap(5, 5, gamma2)
WMap(6, 1, 5, 6, gammax, gamma2)
XMap(7, 1, 4, 7, gammax, gamma1)
VMap(8, 8, gamma0b, scale=1.0, center=center0b)
YMap(9, 1, 4, 5, 9, gammax, gamma1, gamma2)
"""

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

def xed_to_y_lda(xed, rho_data):
    return get_y_from_xed(xed, rho_data[0])

def y_to_xed_lda(y, rho_data):
    return get_xed_from_y(y, rho_data[0])

def chachiyo_fx(s2):
    c = 4 * np.pi / 9
    x = c * np.sqrt(s2)
    dx = c / (2 * np.sqrt(s2))
    Pi = np.pi
    Log = np.log
    chfx = (3*x**2 + Pi**2*Log(1 + x))/((Pi**2 + 3*x)*Log(1 + x))
    dchfx = (-3*x**2*(Pi**2 + 3*x) + 3*x*(1 + x)*(2*Pi**2 + 3*x)*Log(1 + x) - 3*Pi**2*(1 + x)*Log(1 + x)**2)/((1 + x)*(Pi**2 + 3*x)**2*Log(1 + x)**2)
    dchfx *= dx
    chfx[s2<1e-8] = 1 + 8 * s2[s2<1e-6] / 27
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
    cov_kernel = PartialARBF(order=order, length_scale=length_scale[n:],
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

def get_exp_density_noise_kernel(noise0=1e-5, noise1=1e-3):
    wk0 = WhiteKernel(noise_level=noise0, noise_level_bounds=(1e-6,1e-3))
    wk1 = WhiteKernel(noise_level=noise1)
    return wk0 + wk1 * ExponentialDensityNoise()

def get_fitted_density_noise_kernel(decay1=2.0, decay2=600.0, noise0=3e-5,
                                    noise1=0.002, noise2=0.02):
    rhok1 = FittedDensityNoise(decay_rate=decay1)
    rhok2 = FittedDensityNoise(decay_rate=decay2)
    wk = WhiteKernel(noise_level=noise0, noise_level_bounds=(1e-6,1e-3))
    wk1 = WhiteKernel(noise_level=noise1)
    wk2 = WhiteKernel(noise_level=noise2)
    noise_kernel = wk + wk1 * rhok1 + wk2 * Exponentiation(rhok2, 2)
    return noise_kernel
