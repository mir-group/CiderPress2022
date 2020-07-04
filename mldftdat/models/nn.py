import torch
import torch.nn as nn
import numpy as np
from math import pi

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
    scale = np.sqrt(1 + fac * p + 0.6 * fac * (alpha - 1)) # 4^(1/3) for 16, 1/(4)^(1/3) for 15
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

def get_desc2(X):
    sprefac = 2 * (3 * np.pi * np.pi)**(1.0/3)
    fac = (6 * np.pi**2)**(2.0/3) / (16 * np.pi)
    p = X[:,1]**2
    gammax = 0.004 * (2**(1.0/3) * sprefac)**2
    u = gammax * p / (1 + gammax * p)
    alpha = X[:,2]
    chi = 1 / (1 + alpha**2)
    nabla = X[:,3]
    scale = np.sqrt(1 + fac * p + 0.6 * fac * (alpha - 1))
    desc = np.zeros((X.shape[0], 48))
    desc0 = np.zeros((X.shape[0], 8))
    afilter_mid = 0.5 - np.cos(2 * np.pi * chi) / 2
    afilter_low = 1 - afilter_mid
    afilter_low[chi > 0.5] = 0
    afilter_high = 1 - afilter_mid
    afilter_high[chi < 0.5] = 0
    desc0[:,0] = u
    desc0[:,1] = X[:,4]  - 2.0 / scale**3
    desc0[:,2] = X[:,15] - 8.0 / scale**3
    desc0[:,3] = X[:,16] - 0.5 / scale**3
    desc0[:,4] = X[:,5]
    desc0[:,5] = X[:,8]
    desc0[:,6] = X[:,6]
    desc0[:,7] = X[:,12]
    desc0[:,1:] /= np.array([2.45332986, 6.11010142, 0.56641113, 4.34577285, 75.42829791, 6.10420534, 10.65421971])
    #print('std', np.std(desc0, axis=0))
    desc[:,0:8]   = desc0 * (afilter_low * u).reshape(-1,1)
    desc[:,8:16]  = desc0 * (afilter_low * (1-u)).reshape(-1,1)
    desc[:,16:24] = desc0 * (afilter_mid * u).reshape(-1,1)
    desc[:,24:32] = desc0 * (afilter_mid * (1-u)).reshape(-1,1)
    desc[:,32:40] = desc0 * (afilter_high * u).reshape(-1,1)
    desc[:,40:48] = desc0 * (afilter_high * (1-u)).reshape(-1,1)
    return desc

"""
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
    desc = np.zeros((X.shape[0], 121))
    desc0 = np.zeros((X.shape[0], 8))
    afilter_mid = 0.5 - np.cos(2 * np.pi * chi) / 2
    afilter_low = 1 - afilter_mid
    afilter_low[chi > 0.5] = 0
    afilter_high = 1 - afilter_mid
    afilter_high[chi < 0.5] = 0
    u_partition = np.array([1-u, u-u**2, u**2-u**3, u**3-u**4, u**4])
    w_partition = np.array([afilter_mid, afilter_low, afilter_high])
    desc0[:,0] = 1.0
    desc0[:,1] = X[:,4]  - 2.0 / scale**3
    desc0[:,2] = X[:,15] - 8.0 / scale**3
    desc0[:,3] = X[:,16] - 0.5 / scale**3
    desc0[:,4] = X[:,5]
    desc0[:,5] = X[:,8]
    desc0[:,6] = X[:,6]
    desc0[:,7] = X[:,12]
    desc0[:,1:] /= np.array([2.45332986, 6.11010142, 0.56641113,
        4.34577285, 75.42829791, 6.10420534, 10.65421971])
    for i in range(5):
        for j in range(3):
            if i ==0 and j == 0:
                desc[:,0] = X[:,4]
                desc[:,1:8] = desc0[:,1:] \
                    * (u_partition[i] * w_partition[j]).reshape(-1,1)
            else:
                desc[:,(i*3+j)*8:(i*3+j+1)*8] = \
                    desc0 * (u_partition[i] * w_partition[j]).reshape(-1,1)
    desc[:,120] = scale
    return desc
"""

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
    desc = np.zeros((X.shape[0], 181))
    desc0 = np.zeros((X.shape[0], 10))
    afilter_mid = 0.5 - np.cos(2 * np.pi * chi) / 2
    afilter_low = 1 - afilter_mid
    afilter_low[chi > 0.5] = 0
    afilter_high = 1 - afilter_mid
    afilter_high[chi < 0.5] = 0
    u_partition = np.array([1-u, u-u**2, u**2-u**3, u**3-u**4, u**4-u**5, u**5])
    w_partition = np.array([afilter_mid, afilter_low, afilter_high])
    desc0[:,0] = 1.0
    desc0[:,1] = X[:,4]  - 2.0 / scale**3
    desc0[:,2] = X[:,15] - 8.0 / scale**3
    desc0[:,3] = X[:,16] - 0.5 / scale**3
    desc0[:,4] = X[:,5]
    desc0[:,5] = X[:,8]
    desc0[:,6] = X[:,6]
    desc0[:,7] = X[:,12]
    desc0[:,8] = X[:,13]
    desc0[:,9] = X[:,14]
    #print('std', np.std(desc0, axis=0))
    desc0[:,1:] /= np.array([2.75509692, 6.88291279, 0.64614893, 4.87467219,\
        92.73161058, 14.27137322, 74.4786665, 225.88666535, 10.04826384])
    #0.           2.75509692   6.88291279   0.64614893   4.87467219
    #92.73161058  14.27137322  74.4786665  225.88666535  10.04826384
    for i in range(6):
        for j in range(3):
            if i ==0 and j == 0:
                desc[:,0] = np.maximum(X[:,4], 0)
                desc[:,1:10] = desc0[:,1:] \
                    * (u_partition[i] * w_partition[j]).reshape(-1,1)
            else:
                desc[:,(i*3+j)*10:(i*3+j+1)*10] = \
                    desc0 * (u_partition[i] * w_partition[j]).reshape(-1,1)
    desc[:,180] = scale
    return desc

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

    def __init__(self, ndesc_in, ndesc_out, X_train, y_train, train_weights, order = 1):
        super(BayesianLinearFeat, self).__init__()
        self.linear = nn.Linear(ndesc_in, ndesc_out, bias = False)
        self.X_train = torch.tensor(X_train, requires_grad = False)
        self.y_train = torch.tensor(y_train, requires_grad = False)
        self.sigmoid = nn.Sigmoid()
        self.noise = nn.Parameter(torch.tensor(1e-4, dtype=torch.float64))
        self.train_weights = torch.tensor(train_weights, requires_grad = False)
        self.order = order
        weight = 0 * self.linear.weight
        print(weight.size())
        for i in range(8):
            for j in range(6):
                weight[i,6*j+i] = 1.0
        self.linear.weight = nn.Parameter(weight)
        self.w = None
        if order > 4:
            raise ValueError('order must not be higher than 3')
        if order > 1:
            order2_inds = []
            for i in range(ndesc_out):
                for j in range(i,ndesc_out):
                    order2_inds.append(i*ndesc_out+j)
            self.order2_inds = order2_inds
        if order > 2:
            order3_inds = []
            for i in range(ndesc_out):
                for j in range(i,ndesc_out):
                    for k in range(j,ndesc_out):
                        order3_inds.append(i*ndesc_out*ndesc_out+j*ndesc_out+k)
            self.order3_inds = order3_inds
        if order > 3:
            order4_inds = []
            for i in range(ndesc_out):
                for j in range(i,ndesc_out):
                    for k in range(j,ndesc_out):
                        for l in range(k,ndesc_out):
                            order4_inds.append(i*ndesc_out**3+j*ndesc_out**2+k*ndesc_out+l)
            self.order4_inds = order4_inds

    def transform_descriptors(self, X):
        X1 = self.sigmoid(self.linear(X)) - 0.5
        XT = 1 * X1
        if self.order > 1:
            X2 = torch.einsum('bi,bj->bij', X1, X1)
            XT = torch.cat((XT, X2.reshape(X2.size(0),-1)[:,self.order2_inds]), dim=1)
        if self.order > 2:
            X3 = torch.einsum('bij,bk->bijk', X2, X1)
            XT = torch.cat((XT, X3.reshape(X3.size(0),-1)[:,self.order3_inds]), dim=1)
        if self.order > 3:
            X4 = torch.einsum('bijk,bl->bijkl', X3, X1)
            XT = torch.cat((XT, X4.reshape(X4.size(0),-1)[:,self.order4_inds]), dim=1)
        return XT

    def compute_weights(self):
        X = self.transform_descriptors(self.X_train)
        y = self.y_train * self.train_weights
        #print(X.size(), y.size())
        A = torch.matmul(X.T, self.train_weights * X) + self.noise
        Xy = torch.matmul(X.T, y)
        #print(A.size(), Xy.size())
        return torch.matmul(torch.inverse(A), Xy)

    def forward(self, X):
        if self.training or self.w is None:
            self.w = self.compute_weights()
        print(torch.isnan(X).any(), torch.isnan(self.w).any())
        X = self.transform_descriptors(X)
        print(torch.isnan(X).any())
        return torch.matmul(X, self.w)

class LinearBigFeat(nn.Module):

    def __init__(self, X_train, y_train, train_weights, order = 1):
        super(LinearBigFeat, self).__init__()
        self.X_train = torch.tensor(X_train, requires_grad = False)
        self.y_train = torch.tensor(y_train, requires_grad = False)
        self.sigmoid = nn.Sigmoid()
        self.noise = nn.Parameter(torch.tensor(1e-4, dtype=torch.float64))
        self.train_weights = torch.tensor(train_weights, requires_grad = False)
        self.wsize = self.X_train.size(1) - 2
        self.C = nn.Parameter(torch.zeros(self.wsize, dtype=torch.float64))
        self.B = nn.Parameter(-5 * torch.ones(self.wsize, dtype=torch.float64))
        self.A = nn.Parameter(torch.zeros(self.wsize, dtype=torch.float64))
        self.w = None

    def transform_descriptors(self, X):
        N = torch.index_select(X, 1, torch.arange(1,self.wsize+1))
        S = torch.index_select(X, 1, torch.tensor([self.wsize+1]))
        D = torch.index_select(X, 1, torch.tensor([0]))
        #print('negative D', (D < 0).any(), torch.min(D))
        #R = self.sigmoid(self.C * N)
        R = N / (1 + torch.exp(self.C) * D**torch.exp(self.A))
        #R = self.sigmoid(self.C * N)
        return R
        #return X

    def compute_weights(self):
        X = self.transform_descriptors(self.X_train)
        y = self.y_train * self.train_weights
        #print(X.size(), y.size())
        A = torch.matmul(X.T, self.train_weights * X)\
            + self.noise * torch.eye(self.wsize)
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

class LinearBigFeat2(nn.Module):

    def __init__(self, X_train, y_train, train_weights, order = 1):
        super(LinearBigFeat2, self).__init__()
        self.X_train = torch.tensor(X_train, requires_grad = False)
        self.y_train = torch.tensor(y_train, requires_grad = False)
        self.sigmoid = nn.Sigmoid()
        self.noise = nn.Parameter(torch.tensor(1e-4, dtype=torch.float64))
        self.train_weights = torch.tensor(train_weights, requires_grad = False)
        self.isize = 9#self.X_train.size(1) - 2
        self.n_layer = 3
        self.W = nn.Parameter(torch.ones((self.n_layer, self.isize), dtype=torch.float64))
        self.A = nn.Parameter(torch.zeros((self.n_layer, self.isize), dtype=torch.float64))
        self.S = nn.Parameter(torch.zeros((self.n_layer, self.isize), dtype=torch.float64))
        self.B = nn.Parameter(torch.zeros((self.n_layer, self.isize), dtype=torch.float64))
        self.C = nn.Parameter(torch.ones((self.n_layer, self.isize), dtype=torch.float64))
        self.w = None
        self.nw = 6
        self.nu = 7
        sprefac = 2 * (3 * pi * pi)**(1.0/3)
        self.gamma = nn.Parameter(torch.tensor(0.004 * (2**(1.0/3) * sprefac)**2, dtype=torch.float64))
        self.wsize = self.nw * self.nu * (self.isize + 1) - 1
        self.std = torch.tensor([1.0, 1.0, 2.75509692, 6.88291279, 0.64614893, 4.87467219,\
                        92.73161058, 14.27137322, 74.4786665, 225.88666535, 10.04826384], dtype=torch.float64)

    def transform_nl_data(self, X, a, s):
        x = 1 / (1 + a**2)
        p = s**2
        u = p / (1 + self.gamma * p)
        for i in range(self.n_layer):
            X = self.C[i] * self.sigmoid(self.W[i] * X + self.A[i] * x\
                                         + self.S[i] * u + self.B[i])
        X = torch.cat([X, torch.ones(X.size(0), dtype=torch.float64).unsqueeze(1)], dim = 1)
        return X

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
        X = torch.index_select(X, 1, torch.tensor([1,2,4,15,16,5,8,6,12,13,14])) / self.std
        s = torch.index_select(X, 1, torch.tensor([1]))
        a = torch.index_select(X, 1, torch.tensor([2]))
        X = torch.index_select(X, 1, torch.arange(2,self.isize+2))
        X = torch.einsum('ni,nj,nk->nijk', self.get_w_partition(s),
                self.get_u_partition(a), self.transform_nl_data(X, a, s))
        X = torch.reshape(X, (X.size(0), -1))
        #print('size', X.size())
        return X

    def compute_weights(self):
        X = self.transform_descriptors(self.X_train)
        y = self.y_train * self.train_weights
        #print(X.size(), y.size())
        A = torch.matmul(X.T, self.train_weights * X)\
            + self.noise * torch.eye(self.wsize + 1)
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

