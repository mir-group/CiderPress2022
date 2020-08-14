from setup_fireworks import make_hf_firework, make_dft_firework,\
    make_ccsd_firework, LaunchPad, read_accdb_structure
from ase import Atoms
import ase.io

fw_lst = []

with open('data_files/ascdb_names.txt', 'r') as f:
    names = [name.strip() for name in f.readlines()]
struct_dat = [read_accdb_structure(name) for name in names]

functional_list = ['pbe']

for struct, mol_id, spin, charge in struct_dat:
    for basis in ['def2-tzvppd']:
        for functional in functional_list:
            fw_lst.append(make_dft_firework(struct, mol_id, basis,
                                            spin, functional = functional, charge = charge,
                                            name = mol_id + '_' + basis + '_' + functional,
                                            skip_analysis = True))

print(len(fw_lst))

launchpad = LaunchPad.auto_load()
for fw in fw_lst:
    launchpad.add_wf(fw)

