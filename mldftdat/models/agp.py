#!/usr/bin/env python3

from gpytorch.kernels.kernel import Kernel
from gpytorch.kernels import GridInterpolationKernel
import torch
import gpytorch

class NAdditiveStructureKernel(Kernel):
    r"""
    A Kernel decorator for kernels with additive structure. If a kernel decomposes
    additively, then this module will be much more computationally efficient.
    A kernel function `k` decomposes additively if it can be written as
    .. math::
       \begin{equation*}
          k(\mathbf{x_1}, \mathbf{x_2}) = k'(x_1^{(1)}, x_2^{(1)}) + \ldots + k'(x_1^{(d)}, x_2^{(d)})
       \end{equation*}
    for some kernel :math:`k'` that operates on a subset of dimensions.
    Given a `b x n x d` input, `AdditiveStructureKernel` computes `d` one-dimensional kernels
    (using the supplied base_kernel), and then adds the component kernels together.
    Unlike :class:`~gpytorch.kernels.AdditiveKernel`, `AdditiveStructureKernel` computes each
    of the additive terms in batch, making it very fast.
    Args:
        :attr:`base_kernel` (Kernel):
            The kernel to approximate with KISS-GP
        :attr:`num_dims` (int):
            The dimension of the input data.
        :attr:'order' (int):
            Maximum order of kernel
        :attr:`active_dims` (tuple of ints, optional):
            Passed down to the `base_kernel`.
    """

    @property
    def is_stationary(self) -> bool:
        """
        Kernel is stationary if the base kernel is stationary.
        """
        return self.base_kernel.is_stationary

    def __init__(self, base_kernel, num_dims, order = 1, active_dims=None):
        super(NAdditiveStructureKernel, self).__init__(active_dims=active_dims)
        self.base_kernel = base_kernel
        self.num_dims = num_dims
        self.order = order
        self.ew = torch.nn.Parameter(torch.tensor([0] * self.order, dtype=torch.float64))
        self.sk_kernels = torch.nn.ModuleList()
        for n in range(1, self.order + 1):
            self.sk_kernels.append(GridInterpolationKernel(base_kernel**n,
                           num_dims = 1, grid_size = 100))

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        if last_dim_is_batch:
            raise RuntimeError("AdditiveStructureKernel does not accept the last_dim_is_batch argument.")

        # b x k x n x m for full or b x k x n for diag
        res = self.base_kernel(x1, x2, diag=diag, last_dim_is_batch=True, **params)
        en = [1]
        sk = []
        for i in range(self.order):
            sk.append(self.sk_kernels[i](x1, x2, diag=diag,
                                         last_dim_is_batch=True,
                                         **params).sum(-2 if diag else -3))
        for n in range(1, self.order + 1):
            en.append(0)
            for k in range(1, n+1):
                en[-1] += (-1)**(k-1) * en[n-k] * sk[k-1]
            en[-1] /= n
        res = 0
        for n in range(1, self.order + 1):
            res += torch.exp(ew[n-1]) * en[n]
        return res

    def prediction_strategy(self, train_inputs, train_prior_dist, train_labels, likelihood):
        return self.base_kernel.prediction_strategy(train_inputs, train_prior_dist, train_labels, likelihood)

    def num_outputs_per_input(self, x1, x2):
        return self.base_kernel.num_outputs_per_input(x1, x2)

