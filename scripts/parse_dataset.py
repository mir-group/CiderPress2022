import os
import json

MLDFTDB = os.environ['MLDFTDB']
CALC_TYPE = 'DFT'
BASIS = 'aug-cc-pvtz'
FUNCTIONAL = 'PBE'
DATASET = 'augG2'

def get_energies(calcdir):
    energies = {}
    mol_ids = filter(lambda x: os.path.isdir(x), os.listdir(calcdir))
    for mol_id in mol_ids:
        run_info_path = os.path.join(calcdir, mol_id, 'run_info.json')
        with open(run_info_path, 'r') as f:
            data_dict = json.load(f)
        energies[mol_id] = data_dict['e_tot']
    return energies

if CALC_TYPE == 'HF':
    rname = 'RHF'
    uname = 'UHF'
elif CALC_TYPE == 'DFT':
    rname = 'RKS/' + FUNCTIONAL
    uname = 'UKS/' + FUNCTIONAL
elif CALC_TYPE == 'CCSD':
    rname = 'CCSD'
    uname = 'UCCSD'

rdir = os.path.join(MLDFTDB, rname, BASIS, DATASET)
udir = os.path.join(MLDFTDB, uname, BASIS, DATASET)

data = {}
data.update(get_energies(rdir))
data.update(get_energies(udir))

print(data)
