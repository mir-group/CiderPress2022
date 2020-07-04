from mldftdat.models.nn import get_desc3
import gpytorch
import torch
import copy
import mldftdat.models.nn

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

def train(train_x, train_y):

    train_x = torch.tensor(train_x)
    train_y = torch.tensor(train_y).squeeze()

    if torch.cuda.is_available():
        train_x = train_x.cuda()
        train_y = train_y.cuda()

    print(train_x.size(), train_y.size())

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPRModel(train_x[::2], train_y[::2], likelihood)

    model = model.double()
    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()

    training_iterations = 1000

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {'params': model.feature_extractor.parameters()},
        {'params': model.covar_module.parameters()},
        {'params': model.mean_module.parameters()},
        {'params': model.likelihood.parameters()},
    ], lr=0.01)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    min_loss = -1.00
    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(train_x[::2])
        loss = -mll(output, train_y[::2])
        loss.backward()
        print('cycle', i, loss.item())
        if loss.item() < min_loss:
            print('updating state', i, loss.item(), min_loss)
            min_loss = loss.item()
            best_state = copy.deepcopy(model.state_dict())
        optimizer.step()

    model.load_state_dict(best_state)

    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
        preds = model(train_x[1::2])
    print('TEST MAE: {}'.format(torch.mean(torch.abs(preds.mean - train_y[1::2]))))

    return model, min_loss
