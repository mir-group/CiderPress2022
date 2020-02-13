import time, psutil, os
from pyscf import gto, lib

def safe_mem_cap_mb():
    return int(psutil.virtual_memory().available // 16e6)

def time_func(func, *args):
    start_time = time.monotonic()
    res = func(*args)
    finish_time = time.monotonic()
    return res, finish_time - start_time

def get_functional_db_name(functional):
    functional = functional.replace(',', '_')
    functional = functional.replace(' ', '_')
    functional = functional.upper()
    return functional

def get_save_dir(root, calc_type, basis, mol_id, functional=None):
    if functional is not None:
        calc_type = calc_type + '/' + get_functional_db_name(functional)
    return os.path.join(root, calc_type, basis, mol_id)

def mol_from_dict(mol_dict):
    for item in ['charge', 'spin', 'symmetry', 'verbose']:
        if type(mol_dict[item]).__module__ == np.__name__:
            mol_dict[item] = mol_dict[item].item()
    mol = gto.mole.unpack(mol_dict)
    mol.build()
    return mol

def load_calc(fname):
    analyzer_dict = lib.chkfile.load(fname, 'analyzer')
    mol = mol_from_dict(analyzer_dict['mol'])
    calc_type = analyzer_dict['calc_type']
    calc = CALC_TYPES[calc_type](mol)
    calc.__dict__.update(analyzer_dict['calc'])
    return calc, calc_type

def load_analyzer_data(dirname):
    data_file = os.path.join(dirname, 'data.hdf5')
    return lib.chkfile.load(data_file, 'analyzer/data')
