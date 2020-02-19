import time, psutil, os
from pyscf import gto, lib
import numpy as np
from mldftdat.pyscf_utils import CALC_TYPES

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
