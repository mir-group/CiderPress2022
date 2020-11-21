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
MOL_IDS = ['augG2/{}'.format(s) for s in MOL_IDS[80:100]]
MOL_IDS = ['augG2/SiH4_s', 'augG2/NH3_s', 'augG2/NaF_s', 'augG2/HCl_s', 'augG2/CH3SCH3_s', 'augG2/H2CCl2_s', 'augG2/CH3CH-CH3-CH3_s', 'augG2/Na2_s', 'augG2/CF4_s', 'augG2/CH3CHO_s', 'augG2/Li2O_s', 'augG2/PN_s', 'augG2/C4H4S_s', 'augG2/H2O_s', 'augG2/MgS_s', 'augG2/H2_s', 'augG2/COF2_s', 'augG2/P2_s', 'augG2/CS_s', 'augG2/HCOOCH3_s']
MOL_IDS += ['augG2/PF5_s', 'augG2/SiCl4_s', 'augG2/SiH4_s', 'augG2/Ne2_s', 'augG2/F2_s']
print(MOL_IDS)
#exit()
BASIS = 'aug-cc-pvtz'


compile_dataset2('augG2_xscan_80_100', MOL_IDS, SAVE_ROOT, CALC_TYPE, FUNCTIONAL, BASIS, RHFAnalyzer)

