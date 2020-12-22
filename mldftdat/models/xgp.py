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


class LMap(FeatureNormalizer):

    def __init__(self, n, i):
        self.n = n
        self.i = i

    @property
    def bounds(self):
        raise ValueError('Unbounded Feature')

    @property
    def num_arg(self):
        return 1

    def fill_feat_(self, y, x):
        n, i = self.n, self.i
        y[n] = x[i]

    def fill_deriv_(self, dfdx, dfdy, x):
        n, i, = self.n, self.i
        dfdx[i] += dfdy[n]


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

"""
Examples:
UMap(0, 0, gammax)
VMap(1, 1, 1, scale=2.0, center=1.0)
VMap(2, 2, gamma0a, scale=1.0, center=center0a)
UMap(3, 3, gamma1)
UMap(4, 4, gamma2)
WMap(5, 0, 4, 5, gammax, gamma2)
XMap(6, 0, 3, 6, gammax, gamma1)
VMap(7, 7, gamma0b, scale=1.0, center=center0b)
VMap(8, 8, gamma0c, scale=1.0, center=center0c)
YMap(9, 0, 3, 4, 9, gammax, gamma1, gamma2)

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
