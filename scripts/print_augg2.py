from ase import Atoms
import ase.io 
from io import StringIO
import random
from mldftdat.pyscf_utils import mol_from_ase
from collections import Counter
from ase.data import chemical_symbols, atomic_numbers, ground_state_magnetic_moments
from math import floor, ceil

random.seed(56)

structs = []

f = open('data_files/jensen_dataset.txt', 'r')
txt = f.read()
f.close()

struct_strs = txt.split('\n\n')

fw_lst = []

nc = ['He2_s', 'Be2_s', 'Ne2_s', 'Ar2_s', 'Kr2_s']
main_elems = ['Al', 'Be', 'B', 'Li', 'Mg', 'Na', 'P', 'Si', 'S', 'F', 'Cl']

mols = {'NC': [], 'other': []}
for elem in main_elems:
    mols[elem] = []

names = []
for struct_str in struct_strs:
    name, struct_str = struct_str.split('\n', 1)
    name = name[:-4]
    structio = StringIO(struct_str)
    struct = ase.io.read(structio, format='xyz')
    try:
        mol = mol_from_ase(struct, 'sto-3g')
    except:
        mol = mol_from_ase(struct, 'sto-3g', spin=1)
    atoms = [a[0] for a in mol._atom]
    formula = Counter(atoms)
    elems = list(formula.keys())
    if name in nc:
        mols['NC'].append(name)
    else:
        for elem in main_elems:
            if elem in elems:
                mols[elem].append(name)
                break
        else:
            mols['other'].append(name)
    names.append(name)
    print(name)

print(mols)
trset = ['Ne2_s']
valset = ['Be2_s']
tsset = ['Ar2_s']
for k in mols.keys():
    print(len(mols[k]))
    if k != 'NC':
        random.shuffle(mols[k])
        N = len(mols[k])
        l, m = floor(0.4 * N), round(0.2 * N)
        trset += mols[k][:l]
        valset += mols[k][l:l+m]
        tsset += mols[k][l+m:]

import yaml

for s, dset in zip(['train', 'validation', 'test'], [trset, valset, tsset]):
    nsp_set, sp_set = [], []
    for name in dset:
        if name.endswith('_s'):
            nsp_set.append('augG2/' + name)
        else:
            sp_set.append('augG2/' + name)
    print(s)
    print(len(nsp_set), nsp_set)
    print(len(sp_set), sp_set)
    with open('data_files/augg2_{}_nsp.yaml'.format(s), 'w') as f:
        yaml.dump({'calc_type': 'RKS', 'mols': nsp_set}, f)
    with open('data_files/augg2_{}_sp.yaml'.format(s), 'w') as f:
        yaml.dump({'calc_type': 'UKS', 'mols': sp_set}, f)
