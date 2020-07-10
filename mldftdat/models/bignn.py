import numpy as np
from math import pi
import torch
import torch.nn as nn

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

def partition_chi(x):
    y2 = 4 * x * (1-x)
    y2 = 1 - (1-y2)**3
    y = 0.5 - np.cos(2 * np.pi * x) / 2
    p1 = y**4
    p2 = 1-(1-y)**2 - y**4
    p3 = y2 - (1-(1-y)**2)
    p4 = 1 - y2
    p5 = p4.copy()
    p6 = p3.copy()
    p7 = p2.copy()
    p2[x > 0.5] = 0
    p3[x > 0.5] = 0
    p4[x > 0.5] = 0
    p5[x < 0.5] = 0
    p6[x < 0.5] = 0
    p7[x < 0.5] = 0
    return p1, p2, p3, p4, p5, p6, p7

def get_desc3(X):
    sprefac = 2 * (3 * np.pi * np.pi)**(1.0/3)
    fac = (6 * np.pi**2)**(2.0/3) / (16 * np.pi)
    p = X[:,1]**2
    gammax = 0.004 * (2**(1.0/3) * sprefac)**2
    u = gammax * p / (1 + gammax * p)
    alpha = X[:,2]
    chi = 1 / (1 + alpha**2)
    nabla = X[:,3]
    scale = np.sqrt(1 + fac * p + 0.6 * fac * (alpha - 1))
    desc = np.zeros((X.shape[0], 421))
    desc0 = np.zeros((X.shape[0], 10))
    afilter_mid = 0.5 - np.cos(2 * np.pi * chi) / 2
    afilter_low = 1 - afilter_mid
    afilter_low[chi > 0.5] = 0
    afilter_high = 1 - afilter_mid
    afilter_high[chi < 0.5] = 0
    u_partition = np.array([1-u, u-u**2, u**2-u**3, u**3-u**4, u**4-u**5, u**5])
    w_partition = np.array(partition_chi(chi))
    rho = X[:,0]
    heg_scale = (rho / 2) / (rho / 2 + 1e-6)
    desc0[:,0] = 1.0
    desc0[:,1] = X[:,4]  - 2.0 / scale**3 * heg_scale
    desc0[:,2] = X[:,15] - 8.0 / scale**3 * heg_scale
    desc0[:,3] = X[:,16] - 0.5 / scale**3 * heg_scale
    desc0[:,4] = X[:,5]
    desc0[:,5] = X[:,8]
    desc0[:,6] = X[:,6]
    desc0[:,7] = X[:,12]
    desc0[:,8] = X[:,13]
    desc0[:,9] = X[:,14]
    #print('std', np.std(desc0, axis=0))
    #desc0[:,1:] /= np.array([2.75509692, 6.88291279, 0.64614893, 4.87467219,\
    #    92.73161058, 14.27137322, 74.4786665, 225.88666535, 10.04826384])
    #0.           2.75509692   6.88291279   0.64614893   4.87467219
    #92.73161058  14.27137322  74.4786665  225.88666535  10.04826384
    for i in range(6):
        for j in range(7):
            if i ==0 and j == 0:
                desc[:,0] = np.maximum(X[:,4], 0)
                desc[:,1:10] = desc0[:,1:] \
                    * (u_partition[i] * w_partition[j]).reshape(-1,1)
            else:
                desc[:,(i*7+j)*10:(i*7+j+1)*10] = \
                    desc0 * (u_partition[i] * w_partition[j]).reshape(-1,1)
    desc[:,420] = scale
    return np.arcsinh(desc)

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

    def __init__(self, model, get_descriptors, y_to_xed):
        self.model = model
        self.get_descriptors = get_descriptors
        self.y_to_xed = y_to_xed
        self.model.eval()

    def predict(self, X, rho_data):
        self.model.eval()
        with torch.no_grad():
            X = torch.tensor(self.get_descriptors(X))
            F = self.model(X).numpy().flatten()
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
        self.add_module('linear4', torch.nn.Linear(50, 10))
        self.add_module('celu4', torch.nn.CELU())
        self.add_module('linear5', torch.nn.Linear(10, 1))


class BigFeatSimple(nn.Module):

    def __init__(self, X_train, y_train, train_weights, order = 1):
        super(LinearBigFeat2, self).__init__()
        self.X_train = torch.tensor(X_train, requires_grad = False)
        self.y_train = torch.tensor(y_train, requires_grad = False)
        self.sigmoid = nn.Sigmoid()
        self.noise = nn.Parameter(torch.tensor(-9.0, dtype=torch.float64))
        self.train_weights = torch.tensor(train_weights, requires_grad = False)
        self.isize = 9#self.X_train.size(1) - 2
        self.n_layer = 3
        self.gammax = nn.Parameter(torch.log(torch.tensor(0.004 * (2**(1.0/3) * sprefac)**2,
                                    dtype=torch.float64)))
        self.gamma1 = nn.Parameter(torch.log(torch.tensor(0.004, dtype=torch.float64)))
        self.gamma2 = nn.Parameter(torch.log(torch.tensor(0.004, dtype=torch.float64)))
        self.gamma0a = nn.Parameter(torch.log(torch.tensor(0.5, dtype=torch.float64)))
        self.gamma0b = nn.Parameter(torch.log(torch.tensor(0.125, dtype=torch.float64)))
        self.gamma0c = nn.Parameter(torch.log(torch.tensor(2.0, dtype=torch.float64)))
        self.w = None
        self.nw = 6
        self.nu = 7
        sprefac = 2 * (3 * pi * pi)**(1.0/3)
        self.wsize = self.nw * self.nu * (self.isize + 1) - 1
        self.wsize = 420

    def transform_nl_data(self, X):
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
        #d1 = p * refs
        d1 = torch.ones(alpha.size(), dtype=torch.float64)
        d2 = (X[:,4] * scale**3 - 2.0) * ref0a
        d3 = X[:,5]**2 * scale**6 * ref1
        d4 = X[:,8] * scale**6 * ref2
        d5 = X[:,12] * scale**3 * refs * torch.sqrt(ref2)
        d6 = X[:,6] * scale**3 * torch.sqrt(refs) * torch.sqrt(ref1)
        d7 = (X[:,15] * scale**3 - 8.0) * ref0b
        d8 = (X[:,16] * scale**3 - 0.5) * ref0c
        d9 = (X[:,13] * scale**6) * torch.sqrt(refs) * torch.sqrt(ref1) * torch.sqrt(ref2)
        d10 = (X[:,14] * scale**9) * torch.sqrt(ref2) * ref1

        return torch.stack((d1, d2, d3, d4, d5, d6, d7, d8, d9, d10), dim = 1)

    def get_u_partition(self, s):
        p = s**2
        u = self.gamma * p / (1 + self.gamma * p)
        return torch.cat([1-u, u-u**2, u**2-u**3, u**3-u**4,\
                          u**4-u**5, u**5], dim = 1)

    def get_w_partition(self, a):
        x = 1 / (1 + a**2)
        y2 = 4 * x * (1-x)
        y2 = 1 - (1-y2)**3
        y = 0.5 - torch.cos(2 * pi * x) / 2
        p1 = y**4
        p2 = 1-(1-y)**2 - y**4
        p3 = y2 - (1-(1-y)**2)
        p4 = 1 - y2
        p5 = p4.clone()
        p6 = p3.clone()
        p7 = p2.clone()
        p2[x > 0.5] = 0
        p3[x > 0.5] = 0
        p4[x > 0.5] = 0
        p5[x < 0.5] = 0
        p6[x < 0.5] = 0
        p7[x < 0.5] = 0
        return torch.cat([p1, p2, p3, p4, p5, p6, p7], dim = 1)

    def transform_descriptors(self, X):
        s = torch.index_select(X, 1, torch.tensor([1]))
        a = torch.index_select(X, 1, torch.tensor([2]))
        X = torch.einsum('ni,nj,nk->nijk', self.get_w_partition(s),
                self.get_u_partition(a), self.transform_nl_data(X))
        X = torch.reshape(X, (X.size(0), -1))
        #print('size', X.size())
        return torch.index_select(X, 1, torch.arange(1,self.wsize))

    def compute_weights(self):
        X = self.transform_descriptors(self.X_train)
        y = self.y_train * self.train_weights
        #print(X.size(), y.size())
        A = torch.matmul(X.T, self.train_weights * X)\
            + torch.exp(self.noise) * torch.eye(self.wsize + 1)
        Xy = torch.matmul(X.T, y)
        #print(A.size(), Xy.size())
        return torch.matmul(torch.inverse(A), Xy)

    def forward(self, X):
        if self.training or self.w is None:
            self.w = self.compute_weights()
        #print(torch.isnan(X).any(), X.size(), torch.isnan(self.w).any())
        X = self.transform_descriptors(X)
        #print(torch.isnan(X).any(), X.size(), torch.sum(torch.isnan(X)), torch.max(self.C), torch.max(self.A))
        return torch.matmul(X, self.w)


def get_training_obj(model, lr = 0.005):
    criterion = nn.MSELoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr = lr)
    optimizer = torch.optim.LBFGS(model.parameters(), lr = lr, max_iter = 200, history_size=200)
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

def save_nn(model, fname):
    torch.save(model, fname)

def load_nn(fname, ndesc, func = edmgga_from_q_param, quadratic = False):
    model = PolyAnsatz(ndesc, func, quadratic)
    model.load_state_dict(torch.load(fname))
    model.eval()
    return model

def load_nn(fname):
    model = torch.load(fname)
    model.eval()
    return model

