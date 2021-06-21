# Kernels for training and testing the CIDER model.
# Based on the sklearn kernel module:
# https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/gaussian_process/kernels.py
# License info for sklearn model below:

#   Authors: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
#   License: BSD 3 clause
#   Note: this module is strongly inspired by the kernel module of the george
#         package.

import numpy as np 
from sklearn.gaussian_process.kernels import StationaryKernelMixin,\
    NormalizedKernelMixin, Kernel, Hyperparameter, RBF,\
    GenericKernelMixin, _num_samples, DotProduct, ConstantKernel

from scipy.special import kv, gamma
from scipy.spatial.distance import pdist, cdist, squareform


class ARBF(RBF):
    """
    Additive RBF kernel of Duvenaud et al.
    """

    def __init__(self, order=1, length_scale=1.0, scale=None,
                 length_scale_bounds=(1e-5, 1e5),
                 scale_bounds=(1e-5, 1e5)):
        """
        Args:
            order (int): Order of kernel
            length_scale (float or array): length scale of kernel
            scale (array): coefficients of each order, starting with
                0 and ascending.
            length_scale_bounds: bounds of length_scale
            scale_bounds: bounds of scale
        """
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self.order = order
        self.scale = scale
        self.scale_bounds = scale_bounds

    @property
    def hyperparameter_scale(self):
        return Hyperparameter("scale", "numeric",
                              self.scale_bounds,
                              len(self.scale))

    def __call__(self, X, Y=None, eval_gradient=False, get_sub_kernels=False):
        if self.anisotropic:#np.iterable(self.length_scale):
            num_scale = len(self.length_scale)
        else:
            num_scale = 1
        if Y is None:
            Y = X
        diff = (X[:,np.newaxis,:] - Y[np.newaxis,:,:]) / self.length_scale
        k0 = np.exp(-0.5 * diff**2)
        sk = []
        scale_terms = []
        if eval_gradient:
            deriv_size = 0
            if not self.hyperparameter_length_scale.fixed:
                deriv_size += num_scale
            else:
                num_scale = 0
            if not self.hyperparameter_scale.fixed:
                deriv_size += len(self.scale)
            derivs = np.zeros((X.shape[0], Y.shape[0], deriv_size))
        for i in range(self.order):
            sk.append(np.sum(k0**(i+1), axis=-1))
        en = [1]
        for n in range(self.order):
            en.append(sk[n] * (-1)**n)
            for k in range(n):
                en[-1] += (-1)**k * en[n-k] * sk[k]
            en[-1] /= n + 1
        print('scale', self.scale, self.length_scale)
        #res = self.scale[0] * en[0]
        #if eval_gradient:
        #    derivs[:,:,num_scale] = self.scale[0] * en[0]
        res = 0
        for n in range(self.order + 1):
            res += self.scale[n] * en[n]
            if eval_gradient and not self.hyperparameter_scale.fixed:
                derivs[:,:,num_scale + n] = self.scale[n] * en[n]
        kernel = res
        if get_sub_kernels:
            return kernel, en
        en = None
        res = None

        if eval_gradient and not self.hyperparameter_length_scale.fixed:
            inds = np.arange(X.shape[1])
            for ind in inds:
                den = [np.zeros(X.shape)]
                if self.order > 0:
                    den.append(np.ones(k0[:,:,ind].shape))
                for n in range(1,self.order):
                    den.append(sk[n-1] - k0[:,:,ind]**(n) * (-1)**(n-1))
                    for k in range(n-1):
                        den[-1] += (-1)**k * den[n-k] * (sk[k] - k0[:,:,ind]**(k+1))
                res = 0
                if self.order > 0:
                    res += self.scale[1] * den[1]
                for n in range(2, self.order + 1):
                    res += self.scale[n] * den[n] / (n-1)
                #res *= dzdi(X, self.length_scale, ind)
                res *= diff[:,:,ind]**2 * k0[:,:,ind]
                if self.anisotropic:#np.iterable(self.length_scale):
                    derivs[:,:,ind] = res
                else:
                    derivs[:,:,0] += res
                den = None
            print(type(kernel), type(derivs))
            print(kernel.shape, derivs.shape)
        
        if eval_gradient:
            return kernel, derivs
        
        print(type(kernel))
        print(kernel.shape)
        return kernel


def qarbf_args(arbf_base):
    ndim = len(arbf_base.length_scale)
    length_scale = arbf_base.length_scale
    scale = [arbf_base.scale[0]]
    scale += [arbf_base.scale[1]] * (ndim)
    scale += [arbf_base.scale[2]] * (ndim * (ndim - 1) // 2)
    return ndim, np.array(length_scale), scale

def arbf_args(arbf_base):
    ndim = len(arbf_base.length_scale)
    length_scale = arbf_base.length_scale
    order = arbf_base.order
    scale = [arbf_base.scale[0]]
    if order > 0:
        scale += [arbf_base.scale[1]] * (ndim)
    if order > 1:
        scale += [arbf_base.scale[2]] * (ndim * (ndim - 1) // 2)
    if order > 2:
        scale += [arbf_base.scale[3]] * (ndim * (ndim - 1) * (ndim - 2) // 6)
    if order > 3:
        raise ValueError('Order too high for mapping')
    return ndim, np.array(length_scale), scale, order



###############################################################
# NOTE: Below is a series of "Partial" kernels, which are the #
# same as their base kernels except that only a subset of the #
# features are used to construct the kernel, as specified by  #
# the initializer arguments start and active_dims.            #
###############################################################


class PartialRBF(RBF):
    """
    Child class of sklearn RBF which only acts on the slice X[:,start:]
    (or X[:,active_dims] if and only if active_dims is supplied).
    start is ignored if active_dims is supplied.
    """

    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5), start = 0,
                 active_dims = None):
        super(PartialRBF, self).__init__(length_scale, length_scale_bounds)
        self.start = start
        self.active_dims = active_dims

    def __call__(self, X, Y=None, eval_gradient=False):
        if self.active_dims is None:
            X = X[:,self.start:]
            if Y is not None:
                Y = Y[:,self.start:]
        else:
            X = X[:,self.active_dims]
            if Y is not None:
                Y = Y[:,self.active_dims]
        return super(PartialRBF, self).__call__(X, Y, eval_gradient)


class PartialARBF(ARBF):
    """
    ARBF where subset of X is selected.
    """

    def __init__(self, order = 1, length_scale=1.0,
                 length_scale_bounds = (1e-5, 1e5), scale = 1.0,
                 scale_bounds = (1e-5, 1e5), start = 1,
                 active_dims = None):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self.order = order
        self.scale = scale# if (scale is None) else [1.0] * (order + 1)
        self.scale_bounds = scale_bounds
        self.start = start
        self.active_dims = active_dims

    def __call__(self, X, Y=None, eval_gradient=False, get_sub_kernels=False):
        # hasattr check for back-compatibility
        if not np.iterable(self.scale):
            self.scale = [self.scale] * (self.order + 1)
        if (not hasattr(self, 'active_dims')) or (self.active_dims is None):
            X = X[:,self.start:]
            if Y is not None:
                Y = Y[:,self.start:]
        else:
            X = X[:,self.active_dims]
            if Y is not None:
                Y = Y[:,self.active_dims]
        return super(PartialARBF, self).__call__(X, Y, eval_gradient, get_sub_kernels)


class PartialDot(DotProduct):
    """
    Dot product kernel where subset of X is selected.
    """

    def __init__(self, sigma_0=1.0, sigma_0_bounds=(1e-5, 1e5), start = 0):
        super(PartialDot, self).__init__(sigma_0, sigma_0_bounds)
        self.start = start

    def __call__(self, X, Y=None, eval_gradient=False):

        X = X[:,self.start:]
        if Y is not None:
            Y = Y[:,self.start:]
        return super(PartialDot, self).__call__(X, Y, eval_gradient)


class SingleRBF(RBF):
    """
    RBF kernel with single index of X selected.
    """

    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5), index = 1):
        super(SingleRBF, self).__init__(length_scale, length_scale_bounds)
        self.index = index

    def __call__(self, X, Y=None, eval_gradient=False):

        X = X[:,self.index:self.index+1]
        if Y is not None:
            Y = Y[:,self.index:self.index+1]
        return super(SingleRBF, self).__call__(X, Y, eval_gradient)    

class SingleDot(DotProduct):
    """
    DotProduct kernel with single index of X selected.
    """

    def __init__(self, sigma_0=1.0, sigma_0_bounds=(1e-05, 100000.0), index = 0):
        super(SingleDot, self).__init__(sigma_0, sigma_0_bounds)
        self.index = index

    def __call__(self, X, Y = None, eval_gradient = False):
        X = X[:,self.index:self.index+1]
        if Y is not None:
            Y = Y[:,self.index:self.index+1]
        return super(SingleDot, self).__call__(X, Y, eval_gradient)



#################################################################
# Below are kernels for constructing heteroskedastic noise that #
# is a function of the electron density at the reference point. #
#################################################################


class DensityNoise(StationaryKernelMixin, GenericKernelMixin, Kernel):
    # sigma = 1 / rho

    def __init__(self, index=0):
        self.index = index

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is not None and eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")

        if Y is None:
            K = np.diag(self.diag(X))
            if eval_gradient:
                grad = np.empty((_num_samples(X), _num_samples(X), 0))
                return K, grad
            else:
                return K
        else:
            return np.zeros((_num_samples(X), _num_samples(Y)))

    def diag(self, X):
        return 1 / X[:,self.index]


class ExponentialDensityNoise(StationaryKernelMixin, GenericKernelMixin,
                              Kernel):
    # sigma = 1 / rho^exponent

    def __init__(self, exponent=1.0, exponent_bounds=(0.1, 10)):
        self.exponent = exponent
        self.exponent_bounds = exponent_bounds

    @property
    def hyperparameter_exponent(self):
        return Hyperparameter("exponent", "numeric", self.exponent_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is not None and eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")

        if Y is None:
            K = np.diag(self.diag(X))
            if eval_gradient and not self.hyperparameter_exponent.fixed:
                rho = X[:,0]
                grad = np.empty((_num_samples(X), _num_samples(X), 1))
                grad[:,:,0] = np.diag(-self.exponent * np.log(rho) \
                                      / rho**self.exponent)
                return K, grad
            elif eval_gradient:
                grad = np.zeros((X.shape[0], X.shape[0], 0))
                return K, grad
            else:
                return K
        else:
            return np.zeros((_num_samples(X), _num_samples(Y)))

    def diag(self, X):
        return 1 / X[:,0]**self.exponent

    def __repr__(self):
        return "{0}(exponent={1:.3g})".format(self.__class__.__name__,
                                              self.exponent)


class FittedDensityNoise(StationaryKernelMixin, GenericKernelMixin,
                         Kernel):
    """
    Kernel to model the noise of the exchange enhancement factor based
    on the density. 1 / (1 + decay_rate * rho)
    """

    def __init__(self, decay_rate=4.0, decay_rate_bounds=(1e-5, 1e5)):
        self.decay_rate = decay_rate
        self.decay_rate_bounds = decay_rate_bounds

    @property
    def hyperparameter_decay_rate(self):
        return Hyperparameter(
            "decay_rate", "numeric", self.decay_rate_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is not None and eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")

        if Y is None:
            K = np.diag(self.diag(X))
            if eval_gradient and not self.hyperparameter_decay_rate.fixed:
                rho = X[:,0]
                grad = np.empty((_num_samples(X), _num_samples(X), 1))
                grad[:,:,0] = np.diag(- rho / (1 + self.decay_rate * rho)**2)
                return K, grad
            elif eval_gradient:
                grad = np.zeros((X.shape[0], X.shape[0], 0))
                return K, grad
            else:
                return K
        else:
            return np.zeros((_num_samples(X), _num_samples(Y)))

    def diag(self, X):
        rho = X[:,0]
        return 1 / (1 + self.decay_rate * rho)

    def __repr__(self):
        return "{0}(decay_rate={1:.3g})".format(self.__class__.__name__,
                                                self.decay_rate)



#####################################################
# Miscellaneous kernels for constructing XC models. #
#####################################################


class ADKernel(Kernel):
    """
    A general kernel that passes the active_dims indices
    of the feature vector to the base kernel k.

    The method docs are taken from the sklearn kernel module
    authored by Jan Hendrik Metzen.
    """

    def __init__(self, k, active_dims):
        self.k = k
        self.active_dims = active_dims

    def get_params(self, deep=True):
        params = dict(k=self.k, active_dims=self.active_dims)
        if deep:
            deep_items = self.k.get_params().items()
            params.update(('k__' + k, val) for k, val in deep_items)

        return params

    @property
    def hyperparameters(self):
        """Returns a list of all hyperparameter."""
        return [Hyperparameter("k__" + hyperparameter.name,
                            hyperparameter.value_type,
                            hyperparameter.bounds, hyperparameter.n_elements)
             for hyperparameter in self.k.hyperparameters]

    @property
    def theta(self):
        """Returns the (flattened, log-transformed) non-fixed hyperparameters.
        Note that theta are typically the log-transformed values of the
        kernel's hyperparameters as this representation of the search space
        is more amenable for hyperparameter search, as hyperparameters like
        length-scales naturally live on a log-scale.
        Returns
        -------
        theta : ndarray of shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        return self.k.theta

    @theta.setter
    def theta(self, theta):
        """Sets the (flattened, log-transformed) non-fixed hyperparameters.
        Parameters
        ----------
        theta : ndarray of shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        self.k.theta = theta

    @property
    def bounds(self):
        """Returns the log-transformed bounds on the theta.
        Returns
        -------
        bounds : ndarray of shape (n_dims, 2)
            The log-transformed bounds on the kernel's hyperparameters theta
        """
        return self.k.bounds

    def __eq__(self, b):
        return self.k == b.k and self.active_dims == b.active_dims

    def is_stationary(self):
        """Returns whether the kernel is stationary. """
        return self.k.is_stationary()

    @property
    def requires_vector_input(self):
        """Returns whether the kernel is stationary. """
        return self.k.requires_vector_input

    def __call__(self, X, Y = None, eval_gradient = False):
        X = X[:,self.active_dims]
        if Y is not None:
            Y = Y[:,self.active_dims]
        return self.k.__call__(X, Y, eval_gradient)

    def diag(self, X):
        return self.k.diag(X)

    def __repr__(self):
        return self.k.__repr__()


class SpinSymKernel(ADKernel):
    """
    A kernel that takes features for the up spin-channel,
    up_active_dims, and down spin channel, down_active_dims,
    and calls the base kernel k on each of them, then sums the
    result. This results in a kernel that is symmetric to
    flipping the spin.
    """

    def __init__(self, k, up_active_dims, down_active_dims):
        """
        Args:
            k (Kernel): base kernel
            up_active_dims (list): indices of feature vector
                corresponding to up spin, to pass to k
            down_active_dims (list): indices of feature vector
                corresponding to down spin, to pass to k
        """
        self.k = k
        self.up_active_dims = up_active_dims
        self.down_active_dims = down_active_dims

    def get_params(self, deep=True):
        params = dict(k=self.k, up_active_dims=self.up_active_dims,
                      down_active_dims=self.down_active_dims)
        if deep:
            deep_items = self.k.get_params().items()
            params.update(('k__' + k, val) for k, val in deep_items)

        return params

    def __call__(self, X, Y = None, eval_gradient = False):
        Xup = X[:,self.up_active_dims]
        if Y is not None:
            Yup = Y[:,self.up_active_dims]
        else:
            Yup = None
        kup = self.k.__call__(Xup, Yup, eval_gradient)
        Xdown = X[:,self.down_active_dims]
        if Y is not None:
            Ydown = Y[:,self.down_active_dims]
        else:
            Ydown = None
        kdown = self.k.__call__(Xdown, Ydown, eval_gradient)
        if eval_gradient:
            return kup[0] + kdown[0], kup[1] + kdown[1]
        else:
            return kup + kdown
