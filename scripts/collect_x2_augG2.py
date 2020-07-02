#from mldftdat.gp import DFTGP
from mldftdat.data import get_unique_coord_indexes_spherical, compile_dataset2
from mldftdat.density import get_exchange_descriptors
from mldftdat.lowmem_analyzers import RHFAnalyzer
from mldftdat.workflow_utils import get_save_dir
from setup_fireworks import SAVE_ROOT
from sklearn.model_selection import train_test_split
import numpy as np 
import os

CALC_TYPE = 'RKS'
FUNCTIONAL = 'SCAN'
MOL_IDS = next(os.walk(get_save_dir(SAVE_ROOT, CALC_TYPE, 'aug-cc-pvtz', 'augG2', FUNCTIONAL)))[1]
MOL_IDS = ['augG2/{}'.format(s) for s in MOL_IDS[30:50]]
MOL_IDS = ['augG2/C4H4O_s', 'augG2/CO2_s', 'augG2/C2H5NO2_s', 'augG2/BeS_s', 'augG2/F2O_s', 'augG2/BeO_s', 'augG2/NH3O_s', 'augG2/PH3_s', 'augG2/CH2NHCH2_s', 'augG2/C6H6_s', 'augG2/CH3CH2SH_s', 'augG2/HCN_s', 'augG2/CH2SCH2_s', 'augG2/C2N2_s', 'augG2/CH2CO_s', 'augG2/CS2_s', 'augG2/CH3CH2CH2CH3_s', 'augG2/LiOH_s', 'augG2/CH2CHCHCH2_s', 'augG2/Cl2_s']
print(MOL_IDS)
#exit()
BASIS = 'aug-cc-pvtz'


compile_dataset2('augG2_xscan_30_50', MOL_IDS, SAVE_ROOT, CALC_TYPE, FUNCTIONAL, BASIS, RHFAnalyzer)

