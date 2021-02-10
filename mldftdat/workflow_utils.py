import time, psutil, os
from pyscf import gto, lib
import numpy as np
from mldftdat.pyscf_utils import CALC_TYPES
from ase import Atoms
import yaml

SAVE_ROOT = os.environ.get('MLDFTDB')
ACCDB_DIR = os.environ.get('ACCDB')

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

def get_save_dir(root, calc_type, basis, mol_id, functional=None, ks_to_hf=True):
    if functional is not None:
        calc_type = calc_type + '/' + get_functional_db_name(functional)
        if ks_to_hf:
            calc_type = calc_type.replace('KS/HF', 'HF')
    return os.path.join(root, calc_type, basis, mol_id)

def load_mol_ids(mol_id_file):
    if not mol_id_file.endswith('.yaml'):
        mol_id_file += '.yaml'
    with open(mol_id_file, 'r') as f:
        contents = yaml.load(f, Loader=yaml.Loader)
    return contents['calc_type'], contents['mols']

def read_accdb_structure(struct_id):
    fname = '{}.xyz'.format(os.path.join(ACCDB_DIR, 'Geometries', struct_id))
    with open(fname, 'r') as f:
        #print(fname)
        lines = f.readlines()
        natom = int(lines[0])
        charge_and_spin = lines[1].split()
        charge = int(charge_and_spin[0].strip().strip(','))
        spin = int(charge_and_spin[1].strip().strip(',')) - 1
        symbols = []
        coords = []
        for i in range(natom):
            line = lines[2+i]
            symbol, x, y, z = line.split()
            if symbol.isdigit():
                symbol = int(symbol)
            else:
                symbol = symbol[0] + symbol[1:].lower()
            symbols.append(symbol)
            coords.append([x,y,z])
        struct = Atoms(symbols, positions = coords)
        #print(charge, spin, struct)
    return struct, os.path.join('ACCDB', struct_id), spin, charge
