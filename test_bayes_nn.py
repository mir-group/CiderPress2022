from mldftdat.models.nn import *
import numpy as np 
import matplotlib.pyplot as plt
from scipy.special import erf
import torch

x = np.linspace(-4, 4, 100).reshape(-1, 1)
#y = 1.2 * erf(2.1 * x)
desc = (1 / (1 + np.exp(-0.8 * x)) - 0.5)
desc2 = (1 / (1 + np.exp(0.6 * x)) - 0.5)
y = 1.2 * desc + 0.4 * desc**2 + 0.2 * desc2 + 0.5 * desc * desc2

x_train, x_test = x[:50], x[50:]
y_train, y_test = y[:50], y[50:]

model = BayesianLinearFeat(1, 2, x_train, y_train, np.random.rand(y_train.shape[0],1), order=2)
model.double()

criterion, optimizer = get_training_obj(model, lr = 0.01)

for i in range(20):
    print(i)
    train(x_test, y_test, criterion, optimizer, model)
    validate(x_test, y_test, criterion, model)

print(model.compute_weights())

y_pred = model.forward(torch.tensor(x))

#plt.plot(x, y_pred)
#plt.plot(x, y)
