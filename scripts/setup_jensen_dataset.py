from setup_fireworks import make_hf_firework, make_dft_firework, make_ccsd_firework, LaunchPad
from ase import Atoms
import ase.io 
from io import StringIO

structs = []

f = open('data_files/jensen_dataset.txt', 'r')
txt = f.read()
f.close()

struct_strs = txt.split('\n\n')

fw_lst = []

for struct_str in struct_strs[:10]:
    name, struct_str = struct_str.split('\n', 1)
    name = name[:-4]
    if name.endswith('_s'):
        spin = 0
    elif name.endswith('_d'):
        spin = 1
    elif name.endswith('_t'):
        spin = 2
    elif name.endswith('_q'):
        spin = 3
    else:
        raise ValueError('Could not determine spin for %s' % name)
    structio = StringIO(struct_str)
    struct = ase.io.read(structio, format='xyz')
    print(name, spin, struct)
    mol_id = name
    for basis in ['aug-cc-pvtz', 'cc-pcvtz']:
        fw_lst.append(make_dft_firework(struct, mol_id, basis,
                                        spin, charge = 0,
                                        name = mol_id + '_' + basis + '_dft'))
        fw_lst.append(make_hf_firework(struct, mol_id, basis,
                                        spin, charge = 0,
                                        name = mol_id + '_' + basis + '_hf'))

launchpad = LaunchPad.auto_load()
for fw in fw_lst:
    launchpad.add_wf(fw)
