from mldftdat.data import *
from mldftdat.density import *
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
from mldftdat.gp import DFTGPR
from mldftdat.models.exchange_gps import PBEGPR, EDMGPR

def get_threshold(rho):
    if rho < 1e-2:
        return 3.0
    elif rho < 5e-2:
        return 2.5
    elif rho < 1e-1:
        return 2.0
    elif rho < 5e-1:
        return 1.6
    elif rho < 1.0:
        return 1.5
    elif rho < 10:
        return 2.0
    else:
        return 3.0

SAVE_ROOT = os.environ['MLDFTDB']

dir1 = os.path.join(SAVE_ROOT, 'DATASETS', 'noble_gas_x_aug')
dir2 = os.path.join(SAVE_ROOT, 'DATASETS', 'augG2_x_aug')

NUM = 4
print("NUMER OF FEATURES", NUM)

X_train, y_train, rho_data_train = load_descriptors(dir1)
X_train, y_train, rho_train, rho_data_train = filter_descriptors(X_train, y_train, rho_data_train, tol=5e-3)
X_test, y_test, rho_data_test = load_descriptors(dir2)
X_test2, y_test2, rho_data_test2 = load_descriptors(dir2 + '_80_100')
X_test = np.append(X_test, X_test2, axis=0)
y_test = np.append(y_test, y_test2, axis=0)
rho_data_test = np.append(rho_data_test, rho_data_test2, axis=1)
X_test, y_test, rho_test, rho_data_test = filter_descriptors(X_test, y_test, rho_data_test, tol=5e-3)

X_train = np.append(X_train, X_test[::10], axis=0)
y_train = np.append(y_train, y_test[::10], axis=0)
rho_train = np.append(rho_train, rho_test[::10], axis=0)
rho_data_train = np.append(rho_data_train, rho_data_test[:,::10], axis=1)

const = ConstantKernel(0.26**2)
rbf = RBF([0.625, 0.308, 0.458, 0.00579, 1.56], length_scale_bounds=(1.0e-5, 1.0e5))
wk = WhiteKernel(noise_level=6e-4, noise_level_bounds=(4e-04, 1e-03))
kernel = const * rbf + wk

gpr = EDMGPR(NUM, init_kernel = kernel, use_algpr = False)
gpr.fit(X_train[::40], y_train[::40], rho_data_train[:,::40], optimize_theta = False)

inds = np.arange(X_train.shape[0])
np.random.seed(42)
np.random.shuffle(inds)
print('inds', inds[:20])

old_size = gpr.X.shape[0]
print(X_train.shape)
for j in range(X_train.shape[0]):
    i = inds[j]
    if j % 200 == 0:
        print('current iter', j, gpr.X.shape[0], i)
    if gpr.X.shape[0] < 10000:
        gpr.add_point(X_train[i:i+1,:], y_train[i], rho_data_train[:,i:i+1], threshold_factor=2)
    if gpr.X.shape[0] - old_size > 200 and gpr.X.shape[0] > 2000:
        print('refitting', old_size, gpr.X.shape[0], j)
        #gpr.refit()
        old_size = gpr.X.shape[0]

print(gpr.X.shape)
gpr.refit()

pred = gpr.xed_to_y(gpr.predict(X_test[4::100], rho_data_test[:,4::100]), rho_data_test[:,4::100])
abserr = np.abs(pred - gpr.xed_to_y(y_test[4::100], rho_data_test[:,4::100]))
print(gpr.gp.kernel_)
print(np.mean(abserr))

from joblib import dump, load
dump(gpr, 'gpr%d_v8.joblib' % NUM)
