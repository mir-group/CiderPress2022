from setup_fireworks import make_hf_firework, make_dft_firework, make_ccsd_firework, LaunchPad
from ase import Atoms
import ase.io

fw_lst = []

f = open('/n/holylfs/LABS/kozinsky_lab/Data/QM9/gdb9.sdf', 'r')
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

for i, struct in structs:
    for basis in ['aug-cc-pvtz', 'cc-pcvtz']:
        mol_id = 'qm9/%d-%s' % (i, struct.get_chemical_formula())
        fw_lst.append(make_ccsd_firework(struct, mol_id,
                                        basis, 0, charge=0,
                                        name = mol_id + '-' + basis + '_hf-ccsd'))
        fw_lst.append(make_dft_firework(struct, mol_id,
                                        basis, 0, charge=0,
                                        name = mol_id + '-' + basis + '_dft'))
print(len(fw_lst))

launchpad = LaunchPad.auto_load()
for fw in fw_lst:
    launchpad.add_wf(fw)

