import torch
import torch.nn as nn
import numpy as np

A = torch.tensor(0.704) # maybe replace with sqrt(6/5)?
B = 2 * np.pi / 9 * np.sqrt(6.0/5)
FXP0 = 27 / 50 * 10 / 81
FXI = 1.0 / 3 * (4*np.pi**2 / 3)**(1.0/3)
#MU = 10/81
MU = 0.21
C1 = 1.0 / 3 * (4*np.pi**2 / 3)**(1.0/3)
C2 = 1 - C1
C3 = 0.19697 * np.sqrt(0.704)
C4 = (C3**2 - 0.09834 * MU) / C3**3

def get_desc_default(X, rho_data, num = 1):
    fac = (6 * np.pi**2)**(2.0/3) / (16 * np.pi)
    p = X[:,1]**2
    alpha = X[:,2]
    nabla = X[:,3]
    scale = np.sqrt(fac * p + 0.6 * fac * (alpha - 1)) # 4^(1/3) for 16, 1/(4)^(1/3) for 15
    desc = np.zeros((X.shape[0], 18))
    desc[:,0] = X[:,4] * scale
    desc[:,1] = X[:,4] * scale**3
    desc[:,2] = X[:,4] * scale**5
    desc[:,3] = X[:,5] * scale
    desc[:,4] = X[:,5] * scale**3
    desc[:,5] = X[:,5] * scale**5
    desc[:,6] = X[:,8] * scale
    desc[:,7] = X[:,8] * scale**3
    desc[:,8] = X[:,8] * scale**5
    desc[:,9] = X[:,15] * scale
    desc[:,10] = X[:,15] * scale**3
    desc[:,11] = X[:,15] * scale**5
    desc[:,12] = X[:,16] * scale
    desc[:,13] = X[:,16] * scale**3
    desc[:,14] = X[:,16] * scale**5
    desc[:,15] = p**2
    desc[:,16] = p * alpha
    desc[:,17] = alpha**2
    return np.append(X[:,1:4], desc, axis=1)[:,:num]

def asinh(x):
    return torch.log(x+(x**2+1)**0.5)

def edmgga_from_q(Q):
    x = A * Q + torch.sqrt(1 + (A*Q)**2)
    FX = C1 + (C2 * x) / (1 + C3 * torch.sqrt(x) * asinh(C4 * (x-1)))
    return FX

def edmgga_from_q_param(Q, C3param, C4param):
    x = A * Q + torch.sqrt(1 + (A*Q)**2)
    FX = C1 + (C2 * x) / (1 + C3param * torch.sqrt(x) * asinh(C4param * (x-1)))
    return FX

class Predictor():

    def __init__(self, model, get_descriptors, num, y_to_xed):
        self.model = model
        self.get_descriptors = get_descriptors
        self.num = num
        self.y_to_xed = y_to_xed
        self.model.eval()

    def predict(self, X, rho_data):
        self.model.eval()
        with torch.no_grad():
            X = torch.tensor(self.get_descriptors(X, rho_data, self.num))
            F = self.model(X).numpy()
        return self.y_to_xed(F, rho_data)

class PolyAnsatz(nn.Module):

    def __init__(self, ndesc, func = edmgga_from_q_param, quadratic = False,
                 init_weight = None, init_bias = 1.0, C3init = C3, C4init = C4):
        super(PolyAnsatz, self).__init__()
        self.func = func
        if quadratic:
            ndesc = ndesc + ndesc * (ndesc + 1) // 2
        self.linear = nn.Linear(ndesc, 1, bias = True)
        self.quadratic = quadratic
        with torch.no_grad():
            print(self.linear.weight.size())
            print(self.linear.bias.size())
            weight = np.zeros(ndesc)
            if init_weight is not None:
                init_weight = np.array(init_weight)
                size = init_weight.shape[0]
                weight[:size] = init_weight
            else:
                weight[0] = 0.0
                weight[1] = -1.0
                weight[2] = 0.25
            self.linear.weight = nn.Parameter(torch.tensor([weight],
                                                dtype=torch.float64))
            self.linear.bias = nn.Parameter(torch.tensor([init_bias], dtype=torch.float64))
            print(self.linear.weight.size())
            print(self.linear.bias.size())
        self.C3param = nn.Parameter(torch.tensor(C3init, dtype=torch.float64))
        self.C4param = nn.Parameter(torch.tensor(C4init, dtype=torch.float64))

    def forward(self, x):
        if self.quadratic:
            x2 = torch.einsum('bi,bj->bij', x, x)
            tind = torch.transpose(x2[0,:,:].triu().nonzero(), 0, 1)
            #print(tind.size())
            x = torch.cat([x, x2[:, tind[0], tind[1]]], dim=1)
        x = self.linear(x)
        return torch.squeeze(self.func(x, self.C3param, self.C4param), dim=1)


class BayesianLinearFeat(nn.Module):

    def __init__(self, ndesc, X_train, y_train, train_weights):
        super(BayesianLinearFeat, self).__init__()
        self.linear = nn.Linear(ndesc, ndesc, bias = False)
        self.X_train = torch.tensor(X_train, requires_grad = False)
        self.y_train = torch.tensor(y_train, requires_grad = False)
        self.sigmoid = nn.Sigmoid()
        self.noise = nn.Parameter(torch.tensor(1e-4, dtype=torch.float64))
        self.train_weights = torch.tensor(train_weights, requires_grad = False)

    def transform_descriptors(self, X):
        return self.sigmoid(self.linear(X)) - 0.5

    def compute_weights(self):
        X = self.transform_descriptors(self.X_train)
        y = self.y_train * self.train_weights
        #print(X.size(), y.size())
        A = torch.matmul(X.T, self.train_weights * X) + self.noise
        Xy = torch.matmul(X.T, y)
        #print(A.size(), Xy.size())
        return torch.matmul(torch.inverse(A), Xy)

    def forward(self, X):
        w = self.compute_weights()
        X = self.transform_descriptors(X)
        return torch.matmul(X, w)


def get_training_obj(model, lr = 0.005):
    criterion = nn.MSELoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr = lr)
    optimizer = torch.optim.LBFGS(model.parameters(), lr = lr, max_iter = 2000, history_size=2000)
    return criterion, optimizer

def train(x, y_true, criterion, optimizer, model):
    model.train()
    x = torch.tensor(x)
    y_true = torch.tensor(y_true)
    optimizer.zero_grad()
    y_pred = model(x)
    print(y_pred.size(), y_true.size())
    loss = criterion(y_pred, y_true)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    #optimizer.step()
    def closure():
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y_true)
        #print('loss:', loss.item())
        loss.backward()
        return loss
    optimizer.step(closure)
    #print('Current Train Loss', loss.item())
    return loss.item()

def validate(x, y_true, criterion, model):
    model.eval()
    x = torch.tensor(x)
    y_true = torch.tensor(y_true)
    y_pred = model(x)
    loss = criterion(y_pred, y_true)
    print('Current Train Loss', loss.item())
    return loss.item()

def save_nn(model, fname):
    torch.save(model.state_dict(), fname)

def load_nn(fname, ndesc, func = edmgga_from_q_param, quadratic = False):
    model = PolyAnsatz(ndesc, func, quadratic)
    model.load_state_dict(torch.load(fname))
    model.eval()
    return model
