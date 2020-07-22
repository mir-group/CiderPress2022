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
from mldftdat.gp import DFTGPR
from mldftdat.models.correlation_gps import CorrGPR
from mldftdat.models.matrix_rbf import MatrixRBF
import sys

SAVE_ROOT = os.environ['MLDFTDB']

dir1 = os.path.join(SAVE_ROOT, 'DATASETS', 'atoms_corr')
dir2 = os.path.join(SAVE_ROOT, 'DATASETS', 'qm9_corr')
dir4 = os.path.join(SAVE_ROOT, 'DATASETS', 'spinpol_atoms_corr')

NUM = int(sys.argv[1])
print("NUMER OF FEATURES", NUM)

X_train, y_train, rho_data_train = load_descriptors(dir1)
X_train2, y_train2, rho_data_train2 = load_descriptors(dir4)
X_train = np.append(X_train, X_train2[::200], axis=0)
y_train = np.append(y_train, y_train2[::200], axis=0)
rho_data_train = np.append(rho_data_train, rho_data_train2[:,::250], axis=1)
X_train, y_train, rho_train, rho_data_train = filter_descriptors(X_train, y_train, rho_data_train, tol=1e-5)
X_test, y_test, rho_data_test = load_descriptors(dir2)
X_test, y_test, rho_test, rho_data_test = filter_descriptors(X_test, y_test, rho_data_test, tol=1e-5)

X_train = np.append(X_train, X_test[::300], axis=0)
y_train = np.append(y_train, y_test[::300], axis=0)
rho_train = np.append(rho_train, rho_test[::300], axis=0)
rho_data_train = np.append(rho_data_train, rho_data_test[:,::300], axis=1)

gpr = CorrGPR(NUM)
gpr.fit(X_train, y_train, rho_data_train)
pred = gpr.xed_to_y(gpr.predict(X_test[4::100], rho_data_test[:,4::100]), rho_data_test[:,4::100])
abserr = np.abs(pred - gpr.xed_to_y(y_test[4::100], rho_data_test[:,4::100]))
print(gpr.gp.kernel_)
print(np.mean(abserr))

from joblib import dump, load
dump(gpr, 'gpr%d_corr.joblib' % NUM) 

