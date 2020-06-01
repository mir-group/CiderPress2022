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

def xed_to_y(xed, rho_data):
    return xed / (ldax(rho_data[0,:]) - 1e-7)

def get_descriptors(X, rho_data, num = 1):
    X[:,1] = X[:,1]**2
    return X[:,(1,2,3,4,5,8,6,12,15,16,13,14)[:num]]

dir1 = os.path.join(SAVE_ROOT, 'DATASETS', 'atoms_x2')
dir2 = os.path.join(SAVE_ROOT, 'DATASETS', 'augG2_x2_30_50')
dir3 = os.path.join(SAVE_ROOT, 'DATASETS', 'qm9_x2')
dir4 = os.path.join(SAVE_ROOT, 'DATASETS', 'spinpol_atoms_x2')

NUM = int(sys.argv[1])
print("NUMER OF FEATURES", NUM)

X_train, y_train, rho_data_train = load_descriptors(dir1)
X_train2, y_train2, rho_data_train2 = load_descriptors(dir4)
X_train = np.append(X_train, X_train2[::100], axis=0)
y_train = np.append(y_train, y_train2[::100], axis=0)
rho_data_train = np.append(rho_data_train, rho_data_train2[:,::100], axis=1)
X_train, y_train, rho_train, rho_data_train = filter_descriptors(X_train, y_train, rho_data_train, tol=3e-3)
X_test, y_test, rho_data_test = load_descriptors(dir2)
X_test2, y_test2, rho_data_test2 = load_descriptors(dir2.replace('30_50', '80_100'))
X_test3, y_test3, rho_data_test3 = load_descriptors(dir3)
X_test = np.append(X_test, X_test2, axis=0)
y_test = np.append(y_test, y_test2, axis=0)
X_test = np.append(X_test, X_test3, axis=0)
y_test = np.append(y_test, y_test3, axis=0)
rho_data_test = np.append(rho_data_test, rho_data_test2, axis=1)
rho_data_test = np.append(rho_data_test, rho_data_test3, axis=1)
X_test, y_test, rho_test, rho_data_test = filter_descriptors(X_test, y_test, rho_data_test, tol=3e-3)

X_train = np.append(X_train, X_test[::200], axis=0)
y_train = np.append(y_train, y_test[::200], axis=0)
rho_train = np.append(rho_train, rho_test[::200], axis=0)
rho_data_train = np.append(rho_data_train, rho_data_test[:,::200], axis=1)

model = PolyAnsatz(NUM)
model.double()
criterion, optimizer = get_traing_obj(model)
converged = False
i = 0
X_train = get_descriptors(X_train, rho_data_train, num = NUM)
X_val = get_descriptors(X_test[4::100], rho_data_test[:,4::100], num = NUM)
y_train = xed_to_y(y_train, rho_data_train)
y_val = xed_to_y(y_test[4::100], rho_data_test[:,4::100])
while not converged:
    loss = train(X_train, y_train, criterion, optimizer, model)
    if i % 1000 == 0:
        print('Current Train Loss', loss)
    i += 1
    if loss < 0.0001:
        converged = True
    if i > 100000:
        break
print(converged, np.sqrt(loss))
print(validate(X_val, y_val, criterion, model))
save_nn(model, 'nnmodel_%d.pth' % NUM)
print(model.linear.weight)
print(model.linear.bias)
