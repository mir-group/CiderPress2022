from setup_He import make_dft_firework, LaunchPad
from ase import Atoms
import ase.io

fw_lst = []

numbers1 = list(range(21, 31))
spins1 = [1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
numbers2 = [21, 22, 23, 24, 28]
spins2 = [3, 4, 5, 6, 0]

for Z, spin in zip(numbers1 + numbers2, spins1 + spins2):
    for basis in ['ccpcvtz', 'aug-cc-pvtz']:
        struct = Atoms([Z], positions=[(0,0,0)])
        print('dft fw')
        fw_lst.append(make_dft_firework(struct, 'row4-atoms/%d-%d' % (Z, spin), basis, spin, name='row4-%d-%d-%s' % (Z, spin, basis)))

launchpad = LaunchPad.auto_load()
for fw in fw_lst:
    launchpad.add_wf(fw)

