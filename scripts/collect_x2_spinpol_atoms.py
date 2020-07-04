#from mldftdat.gp import DFTGP
from mldftdat.data import compile_dataset2
from mldftdat.density import get_exchange_descriptors
from mldftdat.lowmem_analyzers import UHFAnalyzer
from mldftdat.workflow_utils import get_save_dir
from setup_fireworks import SAVE_ROOT
from sklearn.model_selection import train_test_split
import numpy as np 
import os
import time

CALC_TYPE = 'UKS'
FUNCTIONAL = 'SCAN'
MOL_IDS = next(os.walk(get_save_dir(SAVE_ROOT, CALC_TYPE, 'aug-cc-pvtz', 'atoms', FUNCTIONAL)))[1]
MOL_IDS = ['atoms/{}'.format(s) for s in MOL_IDS]
BASIS = 'aug-cc-pvtz'
print(MOL_IDS)

compile_dataset2('spinpol_atoms_xscan', MOL_IDS, SAVE_ROOT, CALC_TYPE, FUNCTIONAL, BASIS, UHFAnalyzer, spherical_atom=False)

