#from mldftdat.gp import DFTGP
from mldftdat.data import get_unique_coord_indexes_spherical, compile_dataset2
from mldftdat.density import get_exchange_descriptors
from mldftdat.loc_analyzers import RHFAnalyzer
from mldftdat.workflow_utils import get_save_dir
from setup_fireworks import SAVE_ROOT
from sklearn.model_selection import train_test_split
import numpy as np 
import os

CALC_TYPE = 'RKS'
FUNCTIONAL = 'SCAN'
MOL_IDS = next(os.walk(get_save_dir(SAVE_ROOT, CALC_TYPE, 'aug-cc-pvtz', 'augG2', FUNCTIONAL)))[1]
MOL_IDS = ['augG2/{}'.format(s) for s in MOL_IDS[80:100]]
print(MOL_IDS)
#exit()
BASIS = 'aug-cc-pvtz'


compile_dataset2('augG2_locx86scan_80_100', MOL_IDS, SAVE_ROOT, CALC_TYPE, FUNCTIONAL,
                 BASIS, RHFAnalyzer, locx=True, lam=0.86)

