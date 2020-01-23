from mldftdat.pyscf_tasks import SCFCalc, CCSDCalc, TrainingDataCollector
from ase import Atoms
from fireworks import Firework, LaunchPad
import os

SAVE_ROOT = os.environ['MLDFTDB']
LAUNCH_DIR = os.environ['SCRATCHFW']

def get_hf_tasks(struct, mol_id, basis, spin, charge=0):
    calc_type = 'RHF' if spin == 0 else 'UHF'
    struct_dict = struct.todict()
    t1 = SCFCalc(struct=struct_dict, basis=basis, calc_type=calc_type, spin=spin, charge=charge)
    t2 = TrainingDataCollector(save_root_dir = SAVE_ROOT, mol_id=mol_id)
    return t1, t2

def get_dft_tasks(struct, mol_id, basis, spin, charge=0):
    calc_type = 'RKS' if spin == 0 else 'UKS'
    struct_dict = struct.todict()
    t1 = SCFCalc(struct=struct_dict, basis=basis, calc_type=calc_type, spin=spin, charge=charge)
    t2 = TrainingDataCollector(save_root_dir = SAVE_ROOT, mol_id=mol_id)
    return t1, t2

def make_hf_firework(struct, mol_id, basis, spin, charge=0, name=None):
    return Firework(get_hf_tasks(struct, mol_id, basis, spin, charge), name=name)

def make_dft_firework(struct, mol_id, basis, spin, charge=0, name=None):
    return Firework(get_dft_tasks(struct, mol_id, basis, spin, charge), name=name)

def make_ccsd_firework(struct, mol_id, basis, spin, charge=0, name=None):
    t1, t2 = get_hf_tasks(struct, mol_id, basis, spin, charge)
    t3 = CCSDCalc()
    t4 = TrainingDataCollector(save_root_dir = SAVE_ROOT, mol_id=mol_id)
    return Firework([t1, t2, t3, t4], name=name)

if __name__ == '__main__':
    fw = make_ccsd_firework(Atoms('He', positions=[(0,0,0)]), 'noble_gas/test', 'cc-pvdz', 0)
    launchpad = LaunchPad.auto_load()
    launchpad.add_wf(fw)
