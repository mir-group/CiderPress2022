#!/usr/bin/env python3

from gpytorch.kernels.kernel import Kernel, AdditiveKernel
from gpytorch.kernels import GridInterpolationKernel, ScaleKernel
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
        self.num_dims = num_dims
        self.base_kernel = RBFKernel(active_dims=[0])
        if order < 1:
            raise ValueError('order must be positive integer')
        self.order = order
        self.sk_kernels = torch.nn.ModuleList()
        base_kernel_prods = [[RBFKernel(active_dims=[i]) for i in range(num_dims)]]
        for n in range(self.order - 1):
            base_kernel_prods.append([RBFKernel(active_dims=[i]) * base_kernel_prods[-1][i]\
                                      for i in range(num_dims)])
        for n in range(self.order):
            gridk = ScaleKernel(AdditiveKernel(*(base_kernel_prods[n])))
            self.sk_kernels.append(gridk)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        if last_dim_is_batch:
            raise RuntimeError("AdditiveStructureKernel does not accept the last_dim_is_batch argument.")

        # b x k x n x m for full or b x k x n for diag
        sk = []
        for i in range(self.order):
            sk.append(self.sk_kernels[i](x1, x2, diag=diag))
        en = []
        for n in range(1, self.order + 1):
            en.append(sk[n-1])
            for k in range(1, n):
                #en[-1] += (-1)**(k-1) * en[n-k] * sk[k-1]
                if k % 2 == 1:
                    en[-1] += en[n-k-1] * sk[k-1]
                else:
                    en[-1] -= en[n-k-1] * sk[k-1]
            #en[-1] /= n
        res = en[0]
        for n in range(1, self.order):
            res += en[n]
        return res

    def prediction_strategy(self, train_inputs, train_prior_dist, train_labels, likelihood):
        return self.base_kernel.prediction_strategy(train_inputs, train_prior_dist, train_labels, likelihood)

    def num_outputs_per_input(self, x1, x2):
        return self.base_kernel.num_outputs_per_input(x1, x2)

