import torch
import torch.nn as nn

A = 0.704 # maybe replace with sqrt(6/5)?
B = 2 * np.pi / 9 * np.sqrt(6.0/5)
FXP0 = 27 / 50 * 10 / 81
FXI = 1.0 / 3 * (4*np.pi**2 / 3)**(1.0/3)
#MU = 10/81
MU = 0.21
C1 = 1.0 / 3 * (4*np.pi**2 / 3)**(1.0/3)
C2 = 1 - C1
C3 = 0.19697 * np.sqrt(0.704)
C4 = (C3**2 - 0.09834 * MU) / C3**3

def edmgga_from_q(Q):
    x = A * Q + np.sqrt(1 + (A*Q)**2)
    FX = C1 + (C2 * x) / (1 + C3 * torch.sqrt(x) * torch.arcsinh(C4 * (x-1)))
    return FX

class PolyAnsatz(nn.Module):

    def __init__(self, ndesc, func = edmgga_from_q, quadratic = False):
        self.func = edmgga_from_q
        if quadratic:
            ndesc = ndesc**2
        self.linear = nn.Linear(ndesc, 1, bias = True)
        self.quadratic = quadratic

    def forward(self, x):
        if self.quadratic:
            x = torch.ger(x, x)
            tind = x.triu().nonzero().transpose()
            x = x[tind[0], tind[1]]
        x = self.linear(x)
        return self.func(x)

def get_traing_obj(lr = 0.01):
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = lr)
    return criterion, optimizer

def train(x, y_true, criterion, optimizer, model):
    model.train()
    x = torch.tensor(x)
    y = torch.tensor(y)
    optimizer.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred, y_true)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    print('Current Train Loss', loss.item())
    return loss.item()

def validate(x, y_true, criterion, model):
    model.eval()
    x = torch.tensor(x)
    y = torch.tensor(y)
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
