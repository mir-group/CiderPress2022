from joblib import load, dump
from mldftdat.dft.xc_models import *
import numpy as np
from mldftdat.dft.utils import arcsinh_deriv

gpr = load(fname)
"""
desc_list = [
    Descriptor(1, np.arcsinh, arcsinh_deriv, mul = 1.0),\
    Descriptor(2, np.arcsinh, arcsinh_deriv, mul = 1.0),\
    Descriptor(4, np.arcsinh, arcsinh_deriv, mul = 1.0),\
    Descriptor(5, np.arcsinh, arcsinh_deriv, mul = 1.0),\
    Descriptor(8, np.arcsinh, arcsinh_deriv, mul = 1.0),\
    Descriptor(15, np.arcsinh, arcsinh_deriv, mul = 0.25),\
    Descriptor(16, np.arcsinh, arcsinh_deriv, mul = 4.00),\
]
"""
desc_list = [
    Descriptor(1,  np.arcsinh, arcsinh_deriv, mul = 1.00),\
    Descriptor(2,  np.arcsinh, arcsinh_deriv, mul = 1.00),\
    Descriptor(4,  np.arcsinh, arcsinh_deriv, mul = 1.00),\
    Descriptor(15, np.arcsinh, arcsinh_deriv, mul = 0.25),\
    Descriptor(16, np.arcsinh, arcsinh_deriv, mul = 4.00),\
]

mlfunc = GPFunctional(gpr, desc_list, None)

dump(mlfunc, 'mlfunc.joblib')

