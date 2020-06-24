from mldftdat.models.nn import *
import numpy as np 
import matplotlib.pyplot as plt
from scipy.special import erf

x = np.linspace(-4, 4, 100).reshape(-1, 1)
y = 1.2 * erf(2.1 * x)

x_train, x_test = x[:50], x[50:]
y_train, y_test = y[:50], y[50:]

model = BayesianLinearFeat(1, x_train, y_train, np.ones(y_train.shape))
model.double()

criterion, optimizer = get_training_obj(model, lr = 0.005)

for i in range(10):
    print(i)
    train(x_test, y_test, criterion, optimizer, model)

print(model.compute_weights())

y_pred = model.forward(x)

plt.plot(x, y_pred)
plt.plot(x, y)
