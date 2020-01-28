from setup_fireworks import make_hf_firework, make_ccsd_firework, LaunchPad
from ase import Atoms
import ase.io

fw_lst = []

numbers_r2 = [2, 3, 4, 5, 6, 7, 8, 9, 10]
spins_r2 = [0, 1, 0, 1, 2, 3, 2, 1, 0]

numbers_r3 = [11, 12, 13, 14, 15, 16, 17, 18]
spins_r3 = [1, 0, 1, 2, 3, 2, 1, 0]

numbers_r4 = [19, 20, 31, 32, 33, 34, 35, 36]
spins_r4 = [1, 0, 1, 2, 3, 2, 1, 0]

numbers1 = list(range(21, 31))
spins1 = [1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
numbers2 = [21, 22, 23, 24, 28]
spins2 = [3, 4, 5, 6, 0]

numbers = numbers_r2 + numbers_r3 + numbers_r4 + numbers1 + numbers2
spins = spins_r2 + spins_r3 + spins_r4 + spins1 + spins2

for Z, spin in zip(numbers, spins):
    for basis in ['aug-cc-pvtz', 'cc-pcvtz']:
        struct = Atoms([Z], positions=[(0,0,0)])
        element = struct.get_chemical_formula()
        mol_id = 'atoms/%d-%s-%d' % (Z, element, spin)
        fw_lst.append(make_ccsd_firework(struct, mol_id, basis,
                                        spin, charge = 0,
                                        name = mol_id + '_' + basis + '_hf-ccsd'))
        fw_lst.append(make_dft_firework(struct, mol_id, basis,
                                        spin, charge = 0,
                                        name = mol_id + '_' + basis + '_dft'))

launchpad = LaunchPad.auto_load()
for fw in fw_lst:
    launchpad.add_wf(fw)
