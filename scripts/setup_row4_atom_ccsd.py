from setup_He import make_hf_firework, make_ccsd_firework, LaunchPad
from ase import Atoms
import ase.io

fw_lst = []

ids = [11, 12, 13, 14, 15, 16, 17, 18]
spins = [1, 0, 1, 2, 3, 2, 1, 0]
elements = ['K', 'Ca', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr']

for Z, spin, element in zip(ids, spins, elements):
    struct = Atoms(element, positions=[(0,0,0)])
    fw_lst.append(make_ccsd_firework(struct, 'row4-atoms/%d' % Z, 'aug-cc-pvtz', spin))

launchpad = LaunchPad.auto_load()
for fw in fw_lst:
    launchpad.add_wf(fw)

