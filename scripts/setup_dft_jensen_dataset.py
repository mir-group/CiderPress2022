from mldftdat.fw_setup import make_hf_firework, make_dft_firework, make_ccsd_firework, LaunchPad
from ase import Atoms
import ase.io 
from io import StringIO

structs = []

f = open('data_files/jensen_dataset.txt', 'r')
txt = f.read()
f.close()

struct_strs = txt.split('\n\n')

fw_lst = []

functional_list = ['pbe', 'scan', 'b3lyp']
functional_list = ['MN15L']

names = []
for struct_str in struct_strs:
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
    ##print(name, spin, struct)
    if name in names:
        print(name)
    names.append(name)
    mol_id = 'augG2/' + name
    for basis in ['def2-qzvppd']:
        for functional in functional_list:
            fw_lst.append(make_dft_firework(struct, mol_id, basis,
                                        spin, functional = functional, charge = 0,
                                        name = mol_id + '_' + basis + '_' + functional))
exit()
launchpad = LaunchPad.auto_load()
for fw in fw_lst:
    launchpad.add_wf(fw)
