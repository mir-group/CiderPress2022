import os
import json
import pandas as pd

def get_energies(calcdir):
    energies = {}
    mol_ids = list(filter(lambda x: os.path.isdir(os.path.join(calcdir, x)),
                            os.listdir(calcdir)))
    for mol_id in mol_ids:
        run_info_path = os.path.join(calcdir, mol_id, 'run_info.json')
        with open(run_info_path, 'r') as f:
            data_dict = json.load(f)
        energies[mol_id] = data_dict['e_tot']
    return energies

def parse_calc_type(CALC_TYPE, BASIS, FUNCTIONAL, DATASET):

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

    return data

def parse_pandas(dataset, basis, methods):
    df = None
    for method in methods:
        if type(method) == tuple:
            calc_type, functional = method
            name = functional
        else:
            calc_type = method
            functional = ''
            name = calc_type
        method_data = parse_calc_type(CALC_TYPE, basis, functional, dataset)
        if df is None:
            df = pd.DataFrame.from_dict(method_data,
                        orient='index', columns=[name])
        else:
            df[name]= pd.series(method_data)
    return df


if __name__ == '__main__':
    MLDFTDB = os.environ['MLDFTDB']
    CALC_TYPE = 'DFT'
    BASIS = 'aug-cc-pvtz'
    FUNCTIONAL = 'PBE'
    DATASET = 'augG2'
    print(parse_calc_type(CALC_TYPE, BASIS, FUNCTIONAL, DATASET))

    print('Make pandas df')
    df = parse_pandas(DATASET, BASIS, [('DFT', 'PBE'), 'HF', ('DFT', 'B3LYP')])
    print (df)
