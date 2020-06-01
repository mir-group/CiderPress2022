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

def asinh(x):
    return torch.log(x+(x**2+1)**0.5)

def edmgga_from_q(Q):
    x = A * Q + torch.sqrt(1 + (A*Q)**2)
    FX = C1 + (C2 * x) / (1 + C3 * torch.sqrt(x) * asinh(C4 * (x-1)))
    return FX

class PolyAnsatz(nn.Module):

    def __init__(self, ndesc, func = edmgga_from_q, quadratic = False):
        super(PolyAnsatz, self).__init__()
        self.func = edmgga_from_q
        if quadratic:
            ndesc = ndesc**2
        self.linear = nn.Linear(ndesc, 1, bias = True)
        self.quadratic = quadratic
        with torch.no_grad():
            print(self.linear.weight.size())
            print(self.linear.bias.size())
            #self.linear.weight = nn.Parameter(torch.tensor([[0.0, -1.0, 0.25, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0][:ndesc]], dtype=torch.float64))
            self.linear.weight = nn.Parameter(torch.tensor([[-0.2765, -0.9667,  0.2315, -0.0504,\
                                                            0.0052, -0.0024,  0.0046,  0.0142, 0.0555,\
                                                            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0][:ndesc]],\
                                                            dtype=torch.float64))
            #self.linear.bias = nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
            self.linear.bias = nn.Parameter(torch.tensor([1.0372], dtype=torch.float64))
            print(self.linear.weight.size())
            print(self.linear.bias.size())

    def forward(self, x):
        if self.quadratic:
            x = torch.ger(x, x)
            tind = x.triu().nonzero().transpose()
            x = x[tind[0], tind[1]]
        x = self.linear(x)
        return torch.squeeze(self.func(x), dim=1)

def get_traing_obj(model, lr = 0.01):
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = lr)
    return criterion, optimizer

def train(x, y_true, criterion, optimizer, model):
    model.train()
    x = torch.tensor(x)
    y_true = torch.tensor(y_true)
    optimizer.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred, y_true)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
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

def load_nn(fname, ndesc, func = edmgga_from_q, quadratic = False):
    model = PolyAnsatz(ndesc, func, quadratic)
    model.load_state_dict(torch.load(fname))
    model.eval()
    return model
