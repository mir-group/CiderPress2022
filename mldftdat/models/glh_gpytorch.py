import gpytorch
import torch
import copy
from torch import nn
from math import pi, log
from gpytorch.lazy import DiagLazyTensor
import numpy as np

class CovarianceMatrix(torch.nn.Module):
    def __init__(self, ndim, init_guess=None):
        super(CovarianceMatrix, self).__init__()
        if init_guess is None:
            self.mat = torch.nn.Parameter(torch.eye(ndim))
        else:
            mat = torch.tensor(init_guess, dtype=torch.float64)
            mat = torch.cholesky(mat)
            self.mat = torch.nn.Parameter(mat)

    def forward(self, x):
        return torch.matmul(x, torch.tril(self.mat))


class LinearModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(LinearModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.LinearKernel()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class CovLinearModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,
                 ndim, init_guess=None):
        super(CovLinearModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()#ConstantMean()
        self.covar_module = gpytorch.kernels.LinearKernel()
        self.cov_mat = CovarianceMatrix(ndim, init_guess)

    def get_noise(self):
        return torch.zeros_like(self.train_targets)#torch.tensor(0, dtype=torch.float64)

    def forward(self, x):
        x = self.cov_mat(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class NoiseCovLinearModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,
                 ndim, init_noise, bnoise=None, init_guess=None):
        super(NoiseCovLinearModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()#ConstantMean()
        self.covar_module = gpytorch.kernels.LinearKernel()
        self.cov_mat = CovarianceMatrix(ndim, init_guess)
        self.noise = torch.nn.Parameter(torch.tensor(init_noise, dtype=torch.float64))
        if bnoise is None:
            self.bnoise = torch.tensor(np.zeros_like(init_noise))
        else:
            self.bnoise = torch.tensor(bnoise**2)
        #self.noise = torch.nn.Parameter(torch.tensor(1e-5*np.ones_like(init_noise), dtype=torch.float64))

    def get_noise(self):
        return 2e-4 * (torch.sigmoid(self.noise**2 / 2e-4) - 0.5) + self.bnoise

    def forward(self, x):
        x = self.cov_mat(x)
        mean_x = self.mean_module(x)
        if self.training:
            noise = self.get_noise()
            covar_x = self.covar_module(x) + DiagLazyTensor(noise)
        else:
            covar_x = self.covar_module(x)#.evaluate() + torch.diag(self.noise**2)
            n_test = covar_x.size()[0] - self.noise.size()[0]
            noise = self.get_noise()
            noise = torch.cat((noise, torch.zeros(n_test, dtype=torch.float64)))
            #noise = torch.cat((torch.zeros(n_test, dtype=torch.float64), self.noise))
            covar_x = covar_x + DiagLazyTensor(noise)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class NoiseCovLinearModel2(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,
                 ndim, init_noise, init_guess=None):
        super(NoiseCovLinearModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()#ConstantMean()
        self.covar_module = gpytorch.kernels.LinearKernel()
        self.cov_mat = CovarianceMatrix(ndim, init_guess)
        self.noise = torch.tensor(init_noise, dtype=torch.float64)**2
        self.scale = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        #self.noise = torch.nn.Parameter(torch.tensor(1e-5*np.ones_like(init_noise), dtype=torch.float64))

    def get_noise(self):
        return self.scale**2 * self.noise

    def forward(self, x):
        x = self.cov_mat(x)
        mean_x = self.mean_module(x)
        if self.training:
            noise = self.get_noise()
            covar_x = self.covar_module(x) + DiagLazyTensor(noise)
        else:
            covar_x = self.covar_module(x)#.evaluate() + torch.diag(self.noise**2)
            n_test = covar_x.size()[0] - self.noise.size()[0]
            noise = self.get_noise()
            noise = torch.cat((noise, torch.zeros(n_test, dtype=torch.float64)))
            #noise = torch.cat((torch.zeros(n_test, dtype=torch.float64), self.noise))
            covar_x = covar_x + DiagLazyTensor(noise)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train(train_x, train_y, test_x, test_y, 
          fixed_noise=None, knoise=None, use_cov=False,
          lfbgs=False, cov_mat=None, lr=0.01, bnoise=None,
          load_state=False, save_state=True):

    nfeat = train_x.shape[-1]
    train_x = torch.tensor(train_x)
    train_y = torch.tensor(train_y).squeeze()
    test_x = torch.tensor(test_x)
    test_y = torch.tensor(test_y)

    if torch.cuda.is_available():
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        test_x = test_x.cuda()
        test_y = test_y.cuda()

    print(train_x.size(), train_y.size())

    if fixed_noise is None:
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
                    noise_constraint=gpytorch.constraints.Interval(7.5e-7,1e-3)
                )
        likelihood.noise = torch.tensor([0.0001])
    else:
        print('using fixed noise')
        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                torch.tensor(fixed_noise, dtype=torch.float64),
                learn_additional_noise=True,
                noise_constraint=gpytorch.constraints.Interval(1e-8,1e-3)
        )

    if use_cov:
        if cov_mat is None:
            model = CovLinearModel(train_x, train_y, likelihood, nfeat)
        else:
            if knoise is not None:
                model = NoiseCovLinearModel(train_x, train_y, likelihood, nfeat,
                                            knoise, init_guess=cov_mat, bnoise=bnoise)
            else:
                model = CovLinearModel(train_x, train_y, likelihood,
                                       nfeat, init_guess=cov_mat)
    else:
        model = LinearModel(train_x, train_y, likelihood)

    model = model.double()
    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = model.likelihood

    if lfbgs:
        training_iterations = 100
    else:
        training_iterations = 2000

    model.train()
    likelihood.train()
    if load_state:
        likelihood.load_state_dict(torch.load('gpy_likelihood.pth'))
        model.load_state_dict(torch.load('gpy_model.pth'))
        best_state = copy.deepcopy(model.state_dict())

    if not lfbgs:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.LBFGS(model.parameters(),
                                      lr=lr, max_iter=200,
                                      history_size=200)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    orig_train_y = train_y.clone()
    orig_mean = likelihood(model(train_x)).mean

    min_loss = 2.00
    for i in range(training_iterations):
        if not lfbgs:
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
        else:
            def closure():
                optimizer.zero_grad()
                output = model(train_x)
                loss = -mll(output, train_y)
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
    preds = model(train_x)
    wts = model(torch.eye(nfeat, dtype=torch.float64))
    print('TEST MAE: {}'.format(torch.mean(torch.abs(preds.mean - train_y))))
    print('COEFS: {}'.format((wts.mean.detach()).numpy().tolist()))
    print('NOISE: {}'.format(likelihood.noise, model.get_noise()))
    if save_state:
        torch.save(model.state_dict(), 'gpy_model.pth')
        torch.save(likelihood.state_dict(), 'gpy_likelihood.pth')

    return wts.mean.detach().numpy(), (likelihood.noise + model.get_noise()).detach().numpy(), min_loss
