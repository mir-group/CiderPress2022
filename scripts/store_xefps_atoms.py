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

BASIS = 'aug-cc-pwcv5z'
FUNCTIONAL = 'SCAN'
MOL_IDS = next(os.walk(get_save_dir(SAVE_ROOT, 'UCCSD', BASIS, 'atoms')))[1]
MOL_IDS = ['atoms/{}'.format(s) for s in MOL_IDS]
MOL_IDS_DFT = [get_save_dir(SAVE_ROOT, 'UKS', BASIS, mol_id, functional=FUNCTIONAL) for mol_id in MOL_IDS]
MOL_IDS_CC = [get_save_dir(SAVE_ROOT, 'UCCSD', BASIS, mol_id) for mol_id in MOL_IDS]
IS_RESTRICTED_LIST = [False] * len(MOL_IDS)

MOL_IDS2 = next(os.walk(get_save_dir(SAVE_ROOT, 'CCSD', BASIS, 'atoms')))[1]
MOL_IDS2 = ['atoms/{}'.format(s) for s in MOL_IDS2]
MOL_IDS2_DFT = [get_save_dir(SAVE_ROOT, 'RKS', BASIS, mol_id, functional=FUNCTIONAL) for mol_id in MOL_IDS2]
MOL_IDS2_CC = [get_save_dir(SAVE_ROOT, 'CCSD', BASIS, mol_id) for mol_id in MOL_IDS2]
IS_RESTRICTED_LIST2 = [True] * len(MOL_IDS2)

MOL_IDS_DFT = MOL_IDS_DFT + MOL_IDS2_DFT
MOL_IDS_CC = MOL_IDS_CC + MOL_IDS2_CC
IS_RESTRICTED_LIST = IS_RESTRICTED_LIST + IS_RESTRICTED_LIST2

MOL_IDS_DFT.append(get_save_dir(SAVE_ROOT, 'UKS', 'aug-cc-pv5z', 'atoms/1-H-1', functional=FUNCTIONAL))
MOL_IDS_DFT.append(get_save_dir(SAVE_ROOT, 'RKS', 'aug-cc-pv5z', 'atoms/2-He-0', functional=FUNCTIONAL))
MOL_IDS_CC.append(get_save_dir(SAVE_ROOT, 'UCCSD', 'aug-cc-pv5z', 'atoms/1-H-1'))
MOL_IDS_CC.append(get_save_dir(SAVE_ROOT, 'CCSD', 'aug-cc-pv5z', 'atoms/2-He-0'))
IS_RESTRICTED_LIST.append(False)
IS_RESTRICTED_LIST.append(True)
MOL_IDS_FULL = [pair for pair in zip(MOL_IDS_DFT, MOL_IDS_CC)]

print(MOL_IDS, IS_RESTRICTED_LIST)
#FNAME = os.path.join(SAVE_ROOT, 'DATASETS/xefps_corrq/atom_ref.yaml')
#store_mols_in_order(FNAME, SAVE_ROOT, MOL_IDS_DFT, IS_RESTRICTED_LIST, VAL_SET, mol_id_full=True)

#FNAME = os.path.join(SAVE_ROOT, 'DATASETS/xefps_corrq/atom_etot')
#store_total_energies_dataset(FNAME, SAVE_ROOT, MOL_IDS_FULL, IS_RESTRICTED_LIST, mol_id_full=True)

mlfunc = load('mlfunc10map_heg_v47_clean.joblib')
#FNAME = os.path.join(SAVE_ROOT, 'DATASETS/xefps_corrq/atom_descn1_ex')
#store_full_contribs_dataset(FNAME, SAVE_ROOT, MOL_IDS_DFT, IS_RESTRICTED_LIST, mlfunc, exact=True, mol_id_full=True)
#FNAME = os.path.join(SAVE_ROOT, 'DATASETS/xefps_corrq/atom_descn3_ml')
#store_full_contribs_dataset(FNAME, SAVE_ROOT, MOL_IDS_DFT, IS_RESTRICTED_LIST, mlfunc, exact=False, mol_id_full=True)
FNAME = os.path.join(SAVE_ROOT, 'DATASETS/xefps_corrq/ai_alpha6_ex')
store_new_contribs_dataset(FNAME, SAVE_ROOT, MOL_IDS_DFT, IS_RESTRICTED_LIST, mlfunc, exact=True, mol_id_full=True)

#FNAME = os.path.join(SAVE_ROOT, 'DATASETS/xefps_corrq/atom_vv10')
#store_vv10_contribs_dataset(FNAME, SAVE_ROOT, MOL_IDS_DFT, IS_RESTRICTED_LIST, mol_id_full=True)

