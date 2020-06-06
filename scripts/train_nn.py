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
    #X[:,1] = X[:,1]**2
    #return X[:,(1,2,3,4,5,8,6,12,15,16,13,14)[:num]]
    fac = (6 * np.pi**2)**(2.0/3) / (16 * np.pi)
    X[:,1] = X[:,1]**2
    p = X[:,1]
    alpha = X[:,2]
    nabla = X[:,3]
    scale = np.sqrt(1 + fac * p + 0.6 * fac * (alpha - 1)) # 4^(1/3) for 16, 1/(4)^(1/3) for 15
    desc = np.zeros((X.shape[0], 27))
    desc[:,0] = X[:,4] * scale
    desc[:,1] = X[:,4] * scale**3
    desc[:,2] = X[:,4] * scale**5
    desc[:,3] = X[:,5] * scale
    desc[:,4] = X[:,5] * scale**3
    desc[:,5] = X[:,5] * scale**5
    desc[:,6] = np.sqrt(X[:,8]) * scale
    desc[:,7] = np.sqrt(X[:,8]) * scale**3
    desc[:,8] = np.sqrt(X[:,8]) * scale**5
    desc[:,9] = X[:,15] * scale
    desc[:,10] = X[:,15] * scale**3
    desc[:,11] = X[:,15] * scale**5
    desc[:,12] = X[:,16] * scale
    desc[:,13] = X[:,16] * scale**3
    desc[:,14] = X[:,16] * scale**5
    desc[:,15] = p**2
    desc[:,16] = p * alpha
    desc[:,17] = alpha**2
    desc[:,18:21] = desc[:,0:3]**2
    desc[:,21:24] = desc[:,9:12]**2
    desc[:,24:27] = desc[:,12:15]**2
    return np.append(X[:,1:4], desc, axis=1)[:,:num] 

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

init_weight = np.array([-0.3878, -1.2680,  0.2871,  0.0489, -0.1944,  0.2042, -0.0493,  0.0099,\
         -0.0829,  0.1251, -0.1165,  0.0350, -0.0320,  0.1202, -0.0364,  0.1601,\
         -0.2346,  0.0449,  0.0521,  0.1546, -0.0187])

init_weight = np.array([-3.8867e-01, -1.3832e+00,  2.7172e-01,  7.8060e-02, -1.2923e-01,\
          2.8096e-01, -5.6653e-02,  3.9503e-02, -1.0942e-01,  1.2630e-01,\
         -8.1625e-02,  3.0215e-02, -7.3405e-02,  1.2056e-01, -4.1592e-02,\
          1.7762e-01, -2.0912e-01,  8.8963e-02,  2.5596e-02,  1.1125e-01,\
         -7.2691e-03, -1.5155e-02, -3.1665e-03,  6.5592e-05])
init_bias = 1.1956
init_bias = 1.1176
#init_bias = np.sqrt(3.7)
model = PolyAnsatz(NUM, quadratic = False, init_weight = init_weight, init_bias = init_bias,
                   C3init=0.2054, C4init=1.4396)
                   #C3init=0.2078, C4init=1.4731)
#model = PolyAnsatz(NUM)
model.double()
criterion, optimizer = get_traing_obj(model, lr=0.0002)
converged = False
i = 0
X_train = get_descriptors(X_train, rho_data_train, num = NUM)
X_val = get_descriptors(X_test[4::100], rho_data_test[:,4::100], num = NUM)
y_train = xed_to_y(y_train, rho_data_train)
y_val = xed_to_y(y_test[4::100], rho_data_test[:,4::100])
while not converged:
    loss = train(X_train, y_train, criterion, optimizer, model)
    if i % 100 == 0:
        print('Current Train Loss', loss)
        print(validate(X_val, y_val, criterion, model))
    i += 1
    if loss < 0.0001:
        converged = True
    if i > 3000:
        break
print(converged, np.sqrt(loss))
print(validate(X_val, y_val, criterion, model))
save_nn(model, 'nnmodel_%d.pth' % NUM)
print(model.linear.weight)
print(model.linear.bias)
print(model.C3param, model.C4param)
