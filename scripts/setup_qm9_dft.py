from mldftdat.fw_setup import make_hf_firework, make_dft_firework, make_ccsd_firework, LaunchPad
from ase import Atoms
import ase.io

fw_lst = []

f = open('/n/holystore01/LABS/kozinsky_lab/Lab/Data/QM9/gdb9.sdf', 'r')
str_structs = f.read().split('$$$$\n', 300)
f.close()
from io import StringIO
structs = []
for str_struct in str_structs:
    obj = StringIO(str_struct)
    structs.append(ase.io.read(obj, format='sdf'))
structs = [(i+1, struct) for i, struct in enumerate(structs)]
structs = [struct for struct in structs if sum(struct[1].numbers) % 2 == 0]
structs = structs[:20]

functional_list = ['wB97M_V']

for i, struct in structs:
    for basis in ['def2-qzvppd']:
        mol_id = 'qm9/%d-%s' % (i, struct.get_chemical_formula())
        for functional in functional_list:
            fw_lst.append(make_dft_firework(struct, mol_id, basis,
                                            0, functional = functional, charge = 0,
                                            name = mol_id + '_' + basis + '_' + functional))

        
print(len(fw_lst))

launchpad = LaunchPad.auto_load()
for fw in fw_lst:
    launchpad.add_wf(fw)

