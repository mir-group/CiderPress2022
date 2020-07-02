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
MOL_IDS = next(os.walk(get_save_dir(SAVE_ROOT, CALC_TYPE, 'aug-cc-pvtz', 'qm9', FUNCTIONAL)))[1]
MOL_IDS = ['qm9/{}'.format(s) for s in MOL_IDS]
MOL_IDS = ['qm9/11-C2H4O', 'qm9/7-C2H6', 'qm9/20-CH4N2O', 'qm9/10-C2H3N', 'qm9/2-H3N', 'qm9/12-CH3NO', 'qm9/13-C3H8', 'qm9/3-H2O', 'qm9/1-CH4', 'qm9/8-CH4O', 'qm9/18-C3H6O', 'qm9/5-CHN', 'qm9/4-C2H2', 'qm9/15-C2H6O', 'qm9/14-C2H6O', 'qm9/17-C2H4O', 'qm9/16-C3H6', 'qm9/9-C3H4', 'qm9/19-C2H5NO', 'qm9/6-CH2O']
#exit()
BASIS = 'aug-cc-pvtz'


compile_dataset2('qm9_locx86scan', MOL_IDS, SAVE_ROOT, CALC_TYPE, FUNCTIONAL, BASIS, RHFAnalyzer, locx=True, lam=0.86)

