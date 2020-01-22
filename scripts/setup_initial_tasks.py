from setup_He import make_hf_firework, make_ccsd_firework, LaunchPad
from ase import Atoms
import ase.io

fw_lst = []

ids = [2, 3, 4, 5, 6, 7, 8, 9, 10]
spins = [0, 1, 0, 1, 2, 3, 2, 1, 0]
elements = ['He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne']

#for Z, spin, element in zip(ids, spins, elements):
#    struct = Atoms(element, positions=[(0,0,0)])
#    fw_lst.append(make_ccsd_firework(struct, 'small-atoms/%d' % Z, 'aug-cc-pvtz', spin))

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
structs = structs[1:15]

for i, struct in structs:
    fw_lst.append(make_hf_firework(struct, 'qm9/%d' % i, 'aug-cc-pvtz', 0))
print(len(fw_lst))

launchpad = LaunchPad.auto_load()
for fw in fw_lst:
    launchpad.add_wf(fw)

