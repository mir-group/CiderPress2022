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


SAVE_ROOT = os.environ['MLDFTDB']

dir1 = os.path.join(SAVE_ROOT, 'DATASETS', 'noble_gas_x_aug')
dir2 = os.path.join(SAVE_ROOT, 'DATASETS', 'augG2_x_aug')

NUM = 4
print("NUMER OF FEATURES", NUM)

X_train, y_train, rho_data_train = load_descriptors(dir1)
X_train, y_train, rho_train, rho_data_train = filter_descriptors(X_train, y_train, rho_data_train, tol=7e-3)
X_test, y_test, rho_data_test = load_descriptors(dir2)
X_test2, y_test2, rho_data_test2 = load_descriptors(dir2 + '_80_100')
X_test = np.append(X_test, X_test2, axis=0)
y_test = np.append(y_test, y_test2, axis=0)
rho_data_test = np.append(rho_data_test, rho_data_test2, axis=1)
X_test, y_test, rho_test, rho_data_test = filter_descriptors(X_test, y_test, rho_data_test, tol=7e-3)

X_train = np.append(X_train, X_test[::400], axis=0)
y_train = np.append(y_train, y_test[::400], axis=0)
rho_train = np.append(rho_train, rho_test[::400], axis=0)
rho_data_train = np.append(rho_data_train, rho_data_test[:,::400], axis=1)

const = ConstantKernel(0.253**2)
rbf = RBF([0.687, 0.355, 0.573, 1.7, 7.04], length_scale_bounds=(1.0e-5, 1.0e5))
wk = WhiteKernel(noise_level=2.5e-3, noise_level_bounds=(2.5e-03, 1.0e5))
kernel = const * rbf + wk

gpr = EDMGPR(NUM, init_kernel = kernel)
gpr.fit(X_train, y_train, rho_data_train, optimize_theta = False)

old_size = gpr.X.shape[0]
print(X_train.shape)
for j in range(X_train.shape[0] - 200):
    i = j + 200
    if i % 200 == 0:
        print('current iter', i, gpr.X.shape[0])
    if gpr.X.shape[0] < 8000:
        gpr.add_point(X_train[i:i+1,:], y_train[i], rho_train[i], threshold_factor=4)
    if gpr.X.shape[0] - old_size > int(0.3 * old_size) and gpr.X.shape[0] > 2000:
        print('refitting', old_size, gpr.X.shape[0], i)
        #gpr.refit()
        old_size = gpr.X.shape[0]

print(gpr.X.shape)
gpr.refit()

pred = gpr.xed_to_y(gpr.predict(X_test[4::100], rho_data_test[:,4::100]), rho_data_test[:,4::100])
abserr = np.abs(pred - gpr.xed_to_y(y_test[4::100], rho_data_test[:,4::100]))
print(gpr.gp.kernel_)
print(np.mean(abserr))

from joblib import dump, load
dump(gpr, 'gpr%d_v2.joblib' % NUM)