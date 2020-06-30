#from mldftdat.gp import DFTGP
from mldftdat.data import compile_dataset2
from mldftdat.density import get_exchange_descriptors
from mldftdat.lowmem_analyzers import RHFAnalyzer
from mldftdat.workflow_utils import get_save_dir
from setup_fireworks import SAVE_ROOT
from sklearn.model_selection import train_test_split
import numpy as np 
import os
import time

CALC_TYPE = 'RKS'
FUNCTIONAL = 'PBE'
MOL_IDS = ['atoms/{}-0'.format(s) for s in ['2-He', '4-Be', '10-Ne', '12-Mg', '18-Ar', '30-Zn', '36-Kr']]
BASIS = 'aug-cc-pvtz'

compile_dataset2('atoms_x2', MOL_IDS, SAVE_ROOT, CALC_TYPE, FUNCTIONAL, BASIS, RHFAnalyzer, spherical_atom=True)

