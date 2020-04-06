# Adaptation of the sklearn RBF kernel to support a matrix covariance
# between the inputs. License info for sklearn model below:

# Author: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
# License: BSD 3 clause

# Note: this module is strongly inspired by the kernel module of the george
#       package.

import numpy as np 
from sklearn.gaussian_process.kernels import StationaryKernelMixin,\
    NormalizedKernelMixin, Kernel, Hyperparameter, RBF

from scipy.special import kv, gamma
from scipy.spatial.distance import pdist, cdist, squareform

def _check_length_scale(X, length_scale):
    length_scale = np.squeeze(length_scale).astype(float)
    if np.ndim(length_scale) == 1 and X.shape[1] * (X.shape[1] + 1) // 2 != length_scale.shape[0]:
        raise ValueError("Anisotropic kernel must have the same number of "
                         "dimensions as data (%d!=%d)"
                         % (length_scale.shape[0], X.shape[1]))
    return length_scale

def vector_to_tril(size, vec):
    if vec.shape[0] != size * (size + 1) // 2:
        raise ValueError('wrong size vector')
    tril = np.zeros((size, size))
    ind = 0
    for i in range(size):
        tril[i,:i+1] = vec[ind:ind+i+1]
        ind += i + 1
    assert np.allclose(tril, np.tril(tril))
    return tril

def tril_to_vector(tril):
    inds = np.tril_indices(tril.shape[0])
    return tril[inds]

class MatrixRBF(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """Radial-basis function kernel (aka squared-exponential kernel).
    The RBF kernel is a stationary kernel. It is also known as the
    "squared exponential" kernel. It is parameterized by a lower-triangular
    square matrix L with the size of the input vector. The kernel is given by:
    k(x_i, x_j) = exp(-1 / 2 d(x_i dot L, x_j dot L)^2).
    This kernel is infinitely differentiable, which implies that GPs with this
    kernel as covariance function have mean square derivatives of all orders,
    and are thus very smooth.
    Parameters
    ----------
    L : float or array with shape (n_features,), default: 1.0
        The length scale of the kernel. If a float, an isotropic kernel is
        used. If an array, an anisotropic kernel is used where each dimension
        of l defines the length-scale of the respective feature dimension.
    L_bounds : pair of floats >= 0, default: (1e-5, 1e5)
        The lower and upper bound on length_scale
    """
    def __init__(self, size, L = None, L_bounds=(1e-5, 1e5), ):
        if L is None:
            self.L = tril_to_vector(np.identity(size))
        elif len(L.shape) == 1:
            if L.shape[0] != size * (size + 1) // 2:
                raise ValueError('Size must match L shape.')
            self.L = L
        else:
            if size != L.shape[0]:
                raise ValueError('Size must match L shape')
            if not np.allclose(L, np.tril(L)):
                raise ValueError('L must be lower-triangular')
            self.L = tril_to_vector(L)
        self.size = size
        self.L_bounds = L_bounds

    @property
    def anisotropic(self):
        return True

    @property
    def hyperparameter_length_scale(self):
        return Hyperparameter("L", "numeric",
                              self.L_bounds,
                              self.size * (self.size + 1) // 2)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.
        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)
        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.
        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.
        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)
        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        X = np.atleast_2d(X)
        L = _check_length_scale(X, self.L)
        L = vector_to_tril(self.size, L)
        if Y is None:
            dists = pdist(np.matmul(X, L), metric='sqeuclidean')
            K = np.exp(-.5 * dists)
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")
            dists = cdist(np.matmul(X, L), np.matmul(Y, L),
                          metric='sqeuclidean')
            K = np.exp(-.5 * dists)

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                return K, np.empty((X.shape[0], X.shape[0], 0))
            else:
                # We need to recompute the pairwise dimension-wise distances
                Ldiff = np.dot(X[:, np.newaxis, :] - X[np.newaxis, :, :], L)
                diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
                diffprod = np.zeros((diff.shape[0], diff.shape[1],\
                                    self.size * (self.size + 1) // 2))
                ind = 0
                for i in range(self.size):
                    for j in range(i + 1):
                        diffprod[:,:,ind] = Ldiff[:,:,i] * diff[:,:,j]
                        ind += 1
                K_gradient = - diffprod * K[..., np.newaxis]
                return K, K_gradient
        else:
            return K

    def __repr__(self):
        return "{0}(length_scale=[{1}])".format(
            self.__class__.__name__, ", ".join(map("{0:.3g}".format,
                                               self.L)))


class PartialRBF(RBF):

    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5), start = 0):
        super(PartialRBF, self).__init__(length_scale, length_scale_bounds)
        self.start = start

    def __call__(self, X, Y=None, eval_gradient=False):
        X = X[:,start:]
        if Y is not None:
            Y = Y[:,start:]
        return super(PartialRBF, self).__call__(X, Y, eval_gradient)


class PartialMatrixRBF(MatrixRBF):

    def __init__(self, size, L = None, L_bounds=(1e-5, 1e5), start = 0):
        super(PartialMatrixRBF, self).__init__(size, L, L_bounds)
        self.start = start

    def __call__(self, X, Y=None, eval_gradient=False):
        X = X[:,start:]
        if Y is not None:
            Y = Y[:,start:]
        return super(PartialRBF, self).__call__(X, Y, eval_gradient)
        

class DensityNoise(StationaryKernelMixin, GenericKernelMixin,
                   Kernel):

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is not None and eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")

        if Y is None:
            K = self.diag(X)
            if eval_gradient:
                return K, np.empty((_num_samples(X), _num_samples(X), 0))
            else:
                return K
        else:
            return np.zeros((_num_samples(X), _num_samples(Y)))

    def diag(self, X):
        rho = X[:,0]
        return (0.02 / (1 + 6 * rho))**2

    def __repr__(self):
        return "{0}".format(self.__class__.__name__)
