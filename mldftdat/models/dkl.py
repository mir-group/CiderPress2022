from mldftdat.models.nn import get_desc3
import gpytorch
import torch
import copy
import mldftdat.models.nn
from torch import nn
from math import pi
from mldftdat.models.agp import NAdditiveStructureKernel
from gpytorch.kernels.additive_structure_kernel import AdditiveStructureKernel

# based on https://github.com/cornellius-gp/gpytorch/blob/master/examples/06_PyTorch_NN_Integration_DKL/KISSGP_Deep_Kernel_Regression_CUDA.ipynb

class Predictor(mldftdat.models.nn.Predictor):

    def __init__(self, model, get_descriptors, y_to_xed):
        self.model = model
        self.get_descriptors = get_descriptors
        self.y_to_xed = y_to_xed
        self.model.eval()

    def predict(self, X, rho_data):
        self.model.eval()
        with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
            X = torch.tensor(self.get_descriptors(X))
            F = self.model(X).mean.numpy().flatten()
        return self.y_to_xed(F, rho_data[0])

class FeatureExtractor(torch.nn.Sequential):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(421, 400))
        self.add_module('celu1', torch.nn.CELU())
        self.add_module('linear2', torch.nn.Linear(400, 200))
        self.add_module('celu2', torch.nn.CELU())
        self.add_module('linear3', torch.nn.Linear(200, 50))
        self.add_module('celu3', torch.nn.CELU())
        self.add_module('linear4', torch.nn.Linear(50, 3))
        self.add_module('sigmoid4', torch.nn.Sigmoid())

class GPRModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.GridInterpolationKernel(
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=3)),
            num_dims = 3, grid_size = 20
        )
        self.feature_extractor = FeatureExtractor()

    def forward(self, x):
        projected_x = self.feature_extractor(x)
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def tail_fx_deriv_direct(s):
    sp = 0.5 * sprefac * s / pi**(1.0/3) * 2**(1.0/3)
    sfac = 0.5 * sprefac / pi**(1.0/3) * 2**(1.0/3)
    term1 = hprefac * 2.0 / 3 * sfac * (1.0 / np.arcsinh(0.5 * sp)\
            - sp / (2 * np.sqrt(1+sp**2/4) * np.arcsinh(sp/2)**2))
    term3 = sp / 6 - 17 * sp**3 / 720 + 367 * sp**5 / 80640 - 27859 * sp**7 / 29030400\
            + 1295803 * sp**9 / 6131220480
    term3 *= hprefac * 2.0 / 3 * sfac
    denom = (1 + (l*s/2)**4)
    term2 = 2 * b * s / denom - l**4 * s**3 * (a + b * s**2) / (4 * denom**2)
    f = term2 + term3
    f[s > 0.025] = term2[s > 0.025] + term1[s > 0.025]
    return f

class FeatureNormalizer(torch.nn.Module):
    def __init__(self, ndim = 9):
        super(FeatureNormalizer, self).__init__()
        sprefac = 2 * (3 * pi * pi)**(1.0/3)
        self.gammax = nn.Parameter(torch.log(torch.tensor(0.004 * (2**(1.0/3) * sprefac)**2, dtype=torch.float64)))
        self.gamma1 = nn.Parameter(torch.log(torch.tensor(0.004, dtype=torch.float64)))
        self.gamma2 = nn.Parameter(torch.log(torch.tensor(0.004, dtype=torch.float64)))
        self.gamma0a = nn.Parameter(torch.log(torch.tensor(0.5, dtype=torch.float64)))
        self.gamma0b = nn.Parameter(torch.log(torch.tensor(0.125, dtype=torch.float64)))
        self.gamma0c = nn.Parameter(torch.log(torch.tensor(2.0, dtype=torch.float64)))
        self.ndim = ndim

    def forward(self, X):
        p, alpha = X[:,1]**2, X[:,2]

        fac = (6 * pi**2)**(2.0/3) / (16 * pi)
        scale = torch.sqrt(1 + fac * p + 0.6 * fac * (alpha - 1))

        gammax = torch.exp(self.gammax)
        gamma1 = torch.exp(self.gamma1)
        gamma2 = torch.exp(self.gamma2)
        gamma0a = torch.exp(self.gamma0a)
        gamma0b = torch.exp(self.gamma0c)
        gamma0c = torch.exp(self.gamma0b)

        refs = gammax / (1 + gammax * p)
        ref0a = gamma0a / (1 + gamma0a * X[:,4] * scale**3)
        ref0b = gamma0b / (1 + gamma0b * X[:,15] * scale**3)
        ref0c = gamma0c / (1 + gamma0c * X[:,16] * scale**3)
        ref1 = gamma1 / (1 + gamma1 * X[:,5]**2 * scale**6)
        ref2 = gamma2 / (1 + gamma2 * X[:,8] * scale**6)

        #d0 = X[:,0]
        d1 = p * refs
        #d1 = X[:,0]
        d2 = 2 / (1 + alpha**2) - 1.0
        d3 = (X[:,4] * scale**3 - 2.0) * ref0a
        d4 = X[:,5]**2 * scale**6 * ref1
        d5 = X[:,8] * scale**6 * ref2
        d6 = X[:,12] * scale**3 * refs * torch.sqrt(ref2)
        d7 = X[:,6] * scale**3 * torch.sqrt(refs) * torch.sqrt(ref1)
        d8 = (X[:,15] * scale**3 - 8.0) * ref0b
        d9 = (X[:,16] * scale**3 - 0.5) * ref0c
        d10 = (X[:,13] * scale**6) * torch.sqrt(refs) * torch.sqrt(ref1) * torch.sqrt(ref2)
        #d11 = (X[:,14] * scale**9) * torch.sqrt(ref2) * ref1

        return torch.stack((d1, d2, d3, d4, d5, d6, d7, d8, d9, d10)[:self.ndim], dim = 1)


class BigGPR(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, ndim = 9):
        super(BigGPR, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=ndim))
        self.feature_extractor = FeatureNormalizer(ndim)
        self.covar_module.base_kernel.lengthscale = torch.tensor(
                [[0.7, 1.02, 0.279, 0.337, 0.526, 0.34, 0.333, 0.235, 0.237, 0.3, 0.3][:ndim]], dtype=torch.float64)
        self.covar_module.outputscale = 1.0

    def forward(self, x):
        projected_x = self.feature_extractor(x)
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class AddGPR(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood, order = 1, ndim = 9):
        super(AddGPR, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        base_module = gpytorch.kernels.GridInterpolationKernel(
            gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=1)
            ), num_dims = 1, grid_size = 20
            )
        self.feature_extractor = FeatureNormalizer(ndim)
        #base_module.base_kernel.lengthscale = torch.tensor(
        #        [[0.3, 1.02, 0.279, 0.337, 0.526, 0.34, 0.333, 0.235, 0.237, 0.3, 0.3][:ndim]],
        #        dtype=torch.float64)
        base_module.outputscale = 1.0
        #self.covar_module = AdditiveStructureKernel(base_module,
        #                        ndim)
        self.covar_module = NAdditiveStructureKernel(base_module,
                                ndim, order = order)

    def forward(self, x):
        projected_x = self.feature_extractor(x)
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train(train_x, train_y, model_type = 'DKL', fixed_noise = None, lfbgs = False):

    train_x = torch.tensor(train_x)
    train_y = torch.tensor(train_y).squeeze()

    if torch.cuda.is_available():
        train_x = train_x.cuda()
        train_y = train_y.cuda()

    print(train_x.size(), train_y.size())

    if fixed_noise is None:
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
    else:
        print('using fixed noise')
        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                torch.tensor(fixed_noise[::2], dtype=torch.float64))

    if model_type == 'DKL':
        model = GPRModel(train_x[::2], train_y[::2], likelihood)
    elif model_type == 'ADD':
        model = AddGPR(train_x[::2], train_y[::2], likelihood, order = 1)
    else:
        model = BigGPR(train_x[::2], train_y[::2], likelihood, ndim=10)

    model = model.double()
    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()

    if lfbgs:
        training_iterations = 100
    else:
        training_iterations = 1000

    model.train()
    likelihood.train()

    if not lfbgs:
        optimizer = torch.optim.RMSprop([
            {'params': model.feature_extractor.parameters()},
            {'params': model.covar_module.parameters(), 'lr': 1e-4},
            {'params': model.mean_module.parameters(), 'lr': 1e-4},
            {'params': model.likelihood.parameters()},
        ], lr=0.01)
    else:
        optimizer = torch.optim.LBFGS(model.parameters(), lr = 0.1, max_iter = 200, history_size=200)
   
    print(optimizer.state_dict())

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    min_loss = 2.00
    for i in range(training_iterations):
        if not lfbgs:
            optimizer.zero_grad()
            output = model(train_x[::2])
            loss = -mll(output, train_y[::2])
            loss.backward()
            optimizer.step()
        else:
            def closure():
                optimizer.zero_grad()
                output = model(train_x[::2])
                loss = -mll(output, train_y[::2])
                loss.backward()
                return loss
            optimizer.step(closure)
            loss = closure()
        print('cycle', i, loss.item())
        if loss.item() < min_loss:
            print('updating state', i, loss.item(), min_loss)
            min_loss = loss.item()
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
        preds = model(train_x[1::2])
    print('TEST MAE: {}'.format(torch.mean(torch.abs(preds.mean - train_y[1::2]))))

    return model, min_loss
