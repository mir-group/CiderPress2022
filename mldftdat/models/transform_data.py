from abc import ABC, abstractmethod, abstractproperty
import numpy as np
import yaml


class FeatureNormalizer(ABC):
    """
    Abstract base class for nonlinear transformations
    on raw, scale-invariant CIDER features.
    Basically a simple, manual differentiation tool
    in place of autodiff.
    """

    @abstractproperty
    def num_arg(self):
        """
        Number of arguments to the feature.
        """
        pass

    @abstractmethod
    def bounds(self):
        """
        Returns tuple: lower and upper bound of transformed feature.
        """
        pass

    @abstractmethod
    def fill_feat_(self, y, x):
        """
        Fill the transformed feature vector y in place
        from the initial feature vector x.
        """
        pass

    @abstractmethod
    def fill_deriv_(self, dfdx, dfdy, x):
        """
        Fill the derivative dfdx with respect to initial feature vector
        in place, given the derivative with respect to the
        transformed feature vector dfdy and initial feature
        vector x.
        """
        pass

    @classmethod
    def from_dict(cls, d):
        """
        Initialize from dictionary d.
        """
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
        elif d['code'] == 'Z':
            return ZMap.from_dict(d)
        else:
            raise ValueError('Unrecognized code')


class LMap(FeatureNormalizer):

    def __init__(self, n, i):
        """
        n: Output index
        i: Input index
        """
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
        """
        n: Output index
        i: Input index
        gamma: parameter
        y[n] = gamma * x[i] / (1 + gamma * x[i])
        """
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
        """
        n: Output index
        i: Input index
        gamma, scale, center: parameters
        y[n] = scale * gamma * x[i] / (1 + gamma * x[i]) - center
        """
        self.n = n
        self.i = i
        self.gamma = gamma
        self.scale = scale
        self.center = center

    @property
    def bounds(self):
        return (-self.center, self.scale-self.center)

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
        """
        n: Output index
        i: Input index 1
        j: Input index 2
        k: Input index 3
        gammai, gammaj: parameters
        y[n] = gammai/(1+gammai*x[i]) * sqrt(gammaj/(1+gammaj*x[j])) * x[k]
        """
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
        return WMap(d['n'], d['i'], d['j'], d['k'],
                    d['gammai'], d['gammaj'])


class XMap(FeatureNormalizer):

    def __init__(self, n, i, j, k, gammai, gammaj):
        """
        n: Output index
        i: Input index 1
        j: Input index 2
        k: Input index 3
        gammai, gammaj: parameters
        y[n] = sqrt(gammai/(1+gammai*x[i])) * sqrt(gammaj/(1+gammaj*x[j])) * x[k]
        """
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
        return XMap(d['n'], d['i'], d['j'], d['k'],
                    d['gammai'], d['gammaj'])


class YMap(FeatureNormalizer):

    def __init__(self, n, i, j, k, l, gammai, gammaj, gammak):
        """
        n: Output index
        i: Input index 1
        j: Input index 2
        k: Input index 3
        l: Input index 4
        gammai, gammaj, gammak: parameters
        y[n] = sqrt(gammai/(1+gammai*x[i])) * sqrt(gammaj/(1+gammaj*x[j])) 
               * sqrt(gammak/(1+gammak*x[k]))  * x[l]
        """
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
        return YMap(d['n'], d['i'], d['j'], d['k'], d['l'],
                    d['gammai'], d['gammaj'], d['gammak'])


class ZMap(FeatureNormalizer):

    def __init__(self, n, i, gamma, scale=1.0, center=0.0):
        """
        n: Output index
        i: Input index 1
        gamma, scale, center: parameters
        y[n] = -center + scale / (1 + gamma * x[i])
        """
        self.n = n
        self.i = i
        self.gamma = gamma
        self.scale = scale
        self.center = center

    @property
    def bounds(self):
        return (-self.center, self.scale-self.center)

    @property
    def num_arg(self):
        return 1

    def fill_feat_(self, y, x):
        n, i = self.n, self.i
        y[n] = -self.center + self.scale / (1 + self.gamma * x[i]**2)

    def fill_deriv_(self, dfdx, dfdy, x):
        n, i = self.n, self.i
        dfdx[i] -= 2 * dfdy[n] * self.scale * self.gamma * x[i] / (1 + self.gamma * x[i]**2)**2

    def as_dict(self):
        return {
            'code': 'Z',
            'n': self.n,
            'i': self.i,
            'gamma': self.gamma,
            'scale': self.scale,
            'center': self.center
        }

    @classmethod
    def from_dict(cls, d):
        return ZMap(d['n'], d['i'], d['gamma'], d['scale'], d['center'])


class FeatureList():
    """
    A class containing a list of FeatureNormalizer objects.
    Used to construct transformed feature vectors
    for input into spline or GP models.
    """

    def __init__(self, feat_list):
        """
        feat_list: list(FeatureNormalizer)
        """
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

    @property
    def bounds_list(self):
        return [f.bounds for f in self.feat_list]

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
        UMap(0, 1, gammax),
        VMap(1, 2, 1, scale=2.0, center=1.0),
        VMap(2, 3, gamma0a, scale=1.0, center=center0a),
        UMap(3, 4, gamma1),
        UMap(4, 5, gamma2),
        WMap(5, 1, 5, 6, gammax, gamma2),
        XMap(6, 1, 4, 7, gammax, gamma1),
        VMap(7, 8, gamma0b, scale=1.0, center=center0b),
        VMap(8, 9, gamma0c, scale=1.0, center=center0c),
        YMap(9, 1, 4, 5, 10, gammax, gamma1, gamma2),
]
flst = FeatureList(lst)

center0a = get_vmap_heg_value(2.0, gamma0a)
center0b = get_vmap_heg_value(3.0, gamma0b)
UMap(0, 1, gammax)
VMap(1, 2, 1, scale=2.0, center=1.0)
VMap(2, 3, gamma0a, scale=1.0, center=center0a)
UMap(3, 4, gamma1)
UMap(4, 5, gamma2)
WMap(5, 1, 5, 6, gammax, gamma2)
XMap(6, 1, 4, 7, gammax, gamma1)
VMap(7, 8, gamma0b, scale=1.0, center=center0b)
YMap(8, 1, 4, 5, 9, gammax, gamma1, gamma2)
"""

