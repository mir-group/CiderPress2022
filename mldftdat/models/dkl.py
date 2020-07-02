from mldftdat.models.nn import get_desc3
import gpytorch
import torch

# based on https://github.com/cornellius-gp/gpytorch/blob/master/examples/06_PyTorch_NN_Integration_DKL/KISSGP_Deep_Kernel_Regression_CUDA.ipynb

class FeatureExtractor(torch.nn.Sequential):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(420, 420))
        self.add_module('sigmoid1', torch.nn.Sigmoid())
        self.add_module('linear1', torch.nn.Linear(420, 200))
        self.add_module('sigmoid1', torch.nn.Sigmoid())
        self.add_module('linear1', torch.nn.Linear(200, 50))
        self.add_module('sigmoid1', torch.nn.Sigmoid())
        self.add_module('linear1', torch.nn.Linear(50, 3))
        self.add_module('sigmoid1', torch.nn.Sigmoid())


class GPRModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.GridInterpolationKernel(
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=3)))
        self.feature_extractor = FeatureExtractor()

    def forward(self, x):
        projected_x = self.feature_extractor(x)
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train(train_x, train_y):

    x_train = torch.tensor(x_train)
    y_train = torch.tensor(y_train)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPRegressionModel(train_x, train_y, likelihood)

    model = model.double()
    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()

    training_iterations = 60

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {'params': model.feature_extractor.parameters()},
        {'params': model.covar_module.parameters()},
        {'params': model.mean_module.parameters()},
        {'params': model.likelihood.parameters()},
    ], lr=0.01)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        print('cycle', i, loss.item())
        optimizer.step()

    return model, loss.item()
