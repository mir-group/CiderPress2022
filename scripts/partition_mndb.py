from ase import Atoms
import ase.io 
from io import StringIO
import random
from mldftdat.pyscf_utils import mol_from_ase
from collections import Counter
from ase.data import chemical_symbols, atomic_numbers, ground_state_magnetic_moments
from math import floor, ceil
import os

random.seed(56)

trsets = ['AE17', '2pIsoE4', '4pIsoE4', 'pTC13']
valsets = ['SR-MGM-BE9', 'SR-MGN-BE107', 'ABDE13', 'MR-MGM-BE4',
          'MR-MGN-BE17', 'SR-TM-BE17', 'MR-TM-BE13', 'HTBH38',
          'NHTBH38', 'EA13', 'IP23']
tssets = ['MR-TMD-BE3', 'IsoL6', 'HC7']

trd = {}
vad = {}
tsd = {}
for s in trsets:
    trd[s] = []
for s in valsets:
    vad[s] = []
for s in tssets:
    tsd[s] = []

names = []
with open(os.path.join('DatasetEval.csv'), 'r') as f:
    lines = f.readlines()
    for line in lines:
        names.append(line.split(',')[0])
for name in names:
    for s in trsets:
        if name.startswith(s):
            trd[s].append(name)
            break
    else:
        for s in valsets:
            if name.startswith(s):
                vad[s].append(name)
                break
        else:
            for s in tssets:
                if name.startswith(s):
                    tsd[s].append(name)
                    break

vad_tr = vad.copy()
vad_ts = vad.copy()
for s in valsets:
    tmp = vad[s]
    random.shuffle(tmp)
    N = ceil(len(tmp)/2)
    vad_tr[s] = tmp[:N]
    vad_ts[s] = tmp[N:]

trsets = trsets + valsets
tssets = tssets + valsets
trd.update(vad_tr)
tsd.update(vad_ts)

import yaml

for s in trsets:
    with open('data_files/accdb_mn_train_{}.yaml'.format(s), 'w') as f:
        yaml.dump({'calc_type': 'UKS', 'set': trd[s]}, f)
for s in valsets:
    with open('data_files/accdb_mn_validation_{}.yaml'.format(s), 'w') as f:
        yaml.dump({'calc_type': 'UKS', 'set': vad[s]}, f)
for s in tssets:
    with open('data_files/accdb_mn_test_{}.yaml'.format(s), 'w') as f:
        yaml.dump({'calc_type': 'UKS', 'set': tsd[s]}, f)
