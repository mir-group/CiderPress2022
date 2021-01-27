#from mldftdat.gp import DFTGP
from mldftdat.workflow_utils import get_save_dir
from mldftdat.fw_setup import SAVE_ROOT
import os
from joblib import load
from mldftdat.dft.numint3 import NLNumInt
from mldftdat.models.xefps_correlation import *

VAL_SET = ['BeH2_s', 'CH2CCH2_s', 'CH3CH2OH_s', 'CH3NH2_s', 'H2CO_s', 'BF3_s', 'C2F4_s', 'C5H5N_s', 'CH3OCH3_s', 'C5H8_s', 'C4H6-bi_s', 'CH2CHCl_s', 'CH2OCH2_s', 'NaCl_s', 'SF6_s', 'SO2_s', 'SiS_s', 'CH3CH2NH2_s', 'CH3NO2_s', 'C2H6_s', 'NF3_s', 'CO_s', 'ClF3_s', 'CH3SH_s', 'C2H6CHOH_s', 'HCF3_s', 'N2_s', 'SH2_s', 'NaOH_s', 'N2O_s']

#VAL_SET = [val.split('_')[0] for val in VAL_SET]
VAL_SET = ['augG2/{}'.format(s) for s in VAL_SET]

BASIS = 'def2-qzvppd'
#BASIS = 'aug-cc-pvtz'
FUNCTIONAL = 'SCAN'
MOL_IDS = next(os.walk(get_save_dir(SAVE_ROOT, 'UCCSD', BASIS, 'atoms')))[1]
MOL_IDSP = next(os.walk(get_save_dir(SAVE_ROOT, 'UKS', BASIS, 'atoms', FUNCTIONAL)))[1]
MOL_IDS = ['atoms/{}'.format(s) for s in MOL_IDS if s in MOL_IDSP]
IS_RESTRICTED_LIST = [False] * len(MOL_IDS)

MOL_IDS2 = next(os.walk(get_save_dir(SAVE_ROOT, 'RKS', BASIS, 'atoms', FUNCTIONAL)))[1]
MOL_IDS2 = ['atoms/{}'.format(s) for s in MOL_IDS2]
IS_RESTRICTED_LIST2 = [True] * len(MOL_IDS2)

MOL_IDS3 = next(os.walk(get_save_dir(SAVE_ROOT, 'RKS', BASIS, 'qm9', FUNCTIONAL)))[1]
MOL_IDS3 = ['qm9/{}'.format(s) for s in MOL_IDS3]
IS_RESTRICTED_LIST3 = [True] * len(MOL_IDS3)

MOL_IDS4 = next(os.walk(get_save_dir(SAVE_ROOT, 'CCSD', BASIS, 'augG2')))[1]
MOL_IDS4 = ['augG2/{}'.format(s) for s in MOL_IDS4]
IS_RESTRICTED_LIST4 = [True] * len(MOL_IDS4)

MOL_IDS5 = next(os.walk(get_save_dir(SAVE_ROOT, 'UCCSD', BASIS, 'augG2')))[1]
MOL_IDS5 = ['augG2/{}'.format(s) for s in MOL_IDS5]
IS_RESTRICTED_LIST5 = [False] * len(MOL_IDS5)

MOL_IDS = MOL_IDS + MOL_IDS2 + MOL_IDS3 + MOL_IDS4 + MOL_IDS5
IS_RESTRICTED_LIST = IS_RESTRICTED_LIST + IS_RESTRICTED_LIST2 + IS_RESTRICTED_LIST3 + IS_RESTRICTED_LIST4 + IS_RESTRICTED_LIST5

print(MOL_IDS, IS_RESTRICTED_LIST)
#FNAME = os.path.join(SAVE_ROOT, 'DATASETS/xefps_corrq/mols.yaml')
#store_mols_in_order(FNAME, SAVE_ROOT, MOL_IDS, IS_RESTRICTED_LIST, VAL_SET)

#FNAME = os.path.join(SAVE_ROOT, 'DATASETS/xefps_corrq/etot')
#store_total_energies_dataset(FNAME, SAVE_ROOT, MOL_IDS, IS_RESTRICTED_LIST)

mlfunc = load('mlfunc10map_heg_v47_clean.joblib')
#FNAME = os.path.join(SAVE_ROOT, 'DATASETS/xefps_corr/mnsf2')
#store_mn_contribs_dataset(FNAME, SAVE_ROOT, MOL_IDS, IS_RESTRICTED_LIST, include_x=True, use_sf=True)
#FNAME = os.path.join(SAVE_ROOT, 'DATASETS/xefps_corr/mlx6sf2')
#store_mlx_contribs_dataset(FNAME, SAVE_ROOT, MOL_IDS, IS_RESTRICTED_LIST, mlfunc, include_x=True, use_sf=True)

#FNAME = os.path.join(SAVE_ROOT, 'DATASETS/xefps_corrq/descn1_ex')
#store_full_contribs_dataset(FNAME, SAVE_ROOT, MOL_IDS, IS_RESTRICTED_LIST, mlfunc, exact=True)
#FNAME = os.path.join(SAVE_ROOT, 'DATASETS/xefps_corrq/descn3_ml')
#store_full_contribs_dataset(FNAME, SAVE_ROOT, MOL_IDS, IS_RESTRICTED_LIST, mlfunc, exact=False)
FNAME = os.path.join(SAVE_ROOT, 'DATASETS/xefps_corrq/alpha6_ex')
store_new_contribs_dataset(FNAME, SAVE_ROOT, MOL_IDS, IS_RESTRICTED_LIST, mlfunc, exact=True)

#FNAME = os.path.join(SAVE_ROOT, 'DATASETS/xefps_corrq/vv10')
#store_vv10_contribs_dataset(FNAME, SAVE_ROOT, MOL_IDS, IS_RESTRICTED_LIST)

