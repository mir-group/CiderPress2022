from setup_fireworks import make_hf_firework, make_ccsd_firework, LaunchPad
from ase import Atoms
import ase.io

fw_lst = []

ids = [2, 3, 4, 5, 6, 7, 8, 9, 10]
spins = [0, 1, 0, 1, 2, 3, 2, 1, 0]
elements = ['He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne']

for Z, spin, element in zip(ids, spins, elements):
    struct = Atoms(element, positions=[(0,0,0)])
    fw_lst.append(make_ccsd_firework(struct, 'small-atoms/%d' % Z, 'ccpcvtz', spin))

launchpad = LaunchPad.auto_load()
for fw in fw_lst:
    launchpad.add_wf(fw)

