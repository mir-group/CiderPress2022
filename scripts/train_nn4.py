#from train_gp2 import *
from mldftdat.data import *
from mldftdat.density import *
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
from mldftdat.models.nn import *
import sys

SAVE_ROOT = os.environ['MLDFTDB']

#dir1 = os.path.join(SAVE_ROOT, 'DATASETS', 'atoms_locx86')
#dir2 = os.path.join(SAVE_ROOT, 'DATASETS', 'augG2_locx86_30_50')
#dir3 = os.path.join(SAVE_ROOT, 'DATASETS', 'qm9_locx86')
#dir4 = os.path.join(SAVE_ROOT, 'DATASETS', 'spinpol_atoms_locx86')

dir1 = os.path.join(SAVE_ROOT, 'DATASETS', 'atoms_x2')
dir2 = os.path.join(SAVE_ROOT, 'DATASETS', 'augG2_x2_30_50')
dir3 = os.path.join(SAVE_ROOT, 'DATASETS', 'qm9_x2')
dir4 = os.path.join(SAVE_ROOT, 'DATASETS', 'spinpol_atoms_x2')

X_train, y_train, rho_data_train, wttrain = load_descriptors(dir1, load_wt = True, val_dirname = dir1.replace('x2', 'locx86'))
X_train2, y_train2, rho_data_train2, wttrain2 = load_descriptors(dir4, load_wt = True, val_dirname = dir4.replace('x2', 'locx86'))
print(y_train.shape, y_train2.shape, wttrain.shape, wttrain2.shape)
X_train = np.append(X_train, X_train2[::50], axis=0)
y_train = np.append(y_train, y_train2[::50], axis=0)
wttrain = np.append(wttrain, wttrain2[::50], axis=0)
rho_data_train = np.append(rho_data_train, rho_data_train2[:,::50], axis=1)
X_train, y_train, rho_train, rho_data_train, wttrain = filter_descriptors(
        X_train, y_train, rho_data_train, tol=3e-3, wt=wttrain)
X_test, y_test, rho_data_test, wttest = load_descriptors(dir2, load_wt = True, val_dirname = dir2.replace('x2', 'locx86'))
X_test2, y_test2, rho_data_test2, wttest2 = load_descriptors(dir2.replace('30_50', '80_100'), load_wt = True, val_dirname = dir2.replace('30_50', '80_100').replace('x2', 'locx86'))
X_test3, y_test3, rho_data_test3, wttest3 = load_descriptors(dir3, load_wt = True, val_dirname = dir3.replace('x2', 'locx86'))
X_test = np.append(X_test, X_test2, axis=0)
y_test = np.append(y_test, y_test2, axis=0)
X_test = np.append(X_test, X_test3, axis=0)
y_test = np.append(y_test, y_test3, axis=0)
wttest = np.append(wttest, wttest2, axis=0)
wttest = np.append(wttest, wttest3, axis=0)
rho_data_test = np.append(rho_data_test, rho_data_test2, axis=1)
rho_data_test = np.append(rho_data_test, rho_data_test3, axis=1)
X_test, y_test, rho_test, rho_data_test, wttest = filter_descriptors(
        X_test, y_test, rho_data_test, tol=3e-3, wt=wttest)

X_train = np.append(X_train, X_test[::100], axis=0)
y_train = np.append(y_train, y_test[::100], axis=0)
rho_train = np.append(rho_train, rho_test[::100], axis=0)
rho_data_train = np.append(rho_data_train, rho_data_test[:,::100], axis=1)
wttrain = np.append(wttrain, wttest[::100], axis=0)

train_weights = LDA_FACTOR * rho_train**(1.0 / 3) * wttrain

print(np.isnan(X_train).any(), np.isnan(y_train).any(), (y_train > 0).any(),
      np.isnan(rho_train).any(), (rho_train < 0).any())
X_train = get_desc2(X_train)
y_train = get_y_from_xed(y_train, rho_train)

print(X_train.shape, y_train.shape, train_weights.shape)
model = BayesianLinearFeat(48, 8, X_train[::2], y_train[::2].reshape(-1,1), train_weights[::2].reshape(-1,1), order = 3)
#model = PolyAnsatz(NUM)
model.double()
criterion, optimizer = get_training_obj(model, lr=0.1)
converged = False
i = 0
X_val = get_desc2(X_test[4::100])
y_val = get_y_from_xed(y_test[4::100], rho_test[4::100])
while not converged:
    loss = train(X_train[1::2], y_train[1::2].reshape(-1,1), criterion, optimizer, model)
    if i % 10 == 0:
        print('Current Train Loss', loss)
        print(validate(X_val, y_val.reshape(-1,1), criterion, model))
    i += 1
    if loss < 0.0001:
        converged = True
    if i > 300:
        break
print(converged, np.sqrt(loss))
print(np.sqrt(validate(X_val, y_val.reshape(-1,1), criterion, model)))
save_nn(model, 'nnmodel_cubic.pth')
#print(model.linear.weight)
#print(model.linear.bias)

