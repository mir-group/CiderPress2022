from mldftdat.pyscf_tasks import SCFCalc, CCSDCalc, TrainingDataCollector,\
                                LoadCalcFromDB, DFTFromHF
from mldftdat.workflow_utils import get_save_dir
from ase import Atoms
from fireworks import Firework, LaunchPad
import os

SAVE_ROOT = os.environ['MLDFTDB']

def get_hf_tasks(struct, mol_id, basis, spin, charge=0, **kwargs):
    calc_type = 'RHF' if spin == 0 else 'UHF'
    struct_dict = struct.todict()
    t1 = SCFCalc(struct=struct_dict, basis=basis, calc_type=calc_type, spin=spin, charge=charge)
    t2 = TrainingDataCollector(save_root_dir = SAVE_ROOT, mol_id=mol_id, **kwargs)
    return t1, t2

def get_dft_tasks(struct, mol_id, basis, spin, functional=None, charge=0):
    calc_type = 'RKS' if spin == 0 else 'UKS'
    struct_dict = struct.todict()
    t1 = SCFCalc(struct=struct_dict, basis=basis, calc_type=calc_type,
                spin=spin, charge=charge, functional=functional)
    t2 = TrainingDataCollector(save_root_dir = SAVE_ROOT, mol_id=mol_id)
    return t1, t2

def make_dft_from_hf_firework(functional, hf_type, basis, mol_id):
    if hf_type == 'RHF':
        dft_type = 'RKS'
    elif hf_type == 'UHF':
        dft_type = 'UKS'
    else:
        raise ValueError('hf_type must be RHF or UHF, got {}'.format(hf_type))
    save_dir = get_save_dir(SAVE_ROOT, hf_type, basis, mol_id)
    t1 = LoadCalcFromDB(directory = save_dir)
    t2 = DFTFromHF(functional = functional)
    t3 = TrainingDataCollector(save_root_dir = SAVE_ROOT, mol_id=mol_id)
    name  = mol_id + '_' + basis + '_' + '_DFTfromHF_' + functional
    return Firework([t1, t2, t3], name=name)

def make_hf_firework(struct, mol_id, basis, spin, charge=0, name=None):
    return Firework(get_hf_tasks(struct, mol_id, basis, spin, charge), name=name)

def make_dft_firework(struct, mol_id, basis, spin, functional=None, charge=0, name=None):
    return Firework(get_dft_tasks(struct, mol_id, basis, spin,
                    functional=functional, charge=charge), name=name)

def make_ccsd_firework(struct, mol_id, basis, spin, charge=0, name=None, **kwargs):
    t1, t2 = get_hf_tasks(struct, mol_id, basis, spin, charge)
    t3 = CCSDCalc()
    t4 = TrainingDataCollector(save_root_dir = SAVE_ROOT, mol_id=mol_id, **kwargs)
    return Firework([t1, t2, t3, t4], name=name)

def make_ccsd_firework_no_hf(struct, mol_id, basis, spin, charge=0, name=None, **kwargs):
    if spin == 0:
        hf_type = 'RHF'
    else:
        hf_type = 'UHF'
    save_dir = get_save_dir(SAVE_ROOT, hf_type, basis, mol_id)
    t1 = LoadCalcFromDB(directory = save_dir)
    t2 = CCSDCalc()
    t3 = TrainingDataCollector(save_root_dir = SAVE_ROOT, mol_id=mol_id, **kwargs)
    return Firework([t1, t2, t3], name=name)

if __name__ == '__main__':
    fw1 = make_hf_firework(Atoms('He', positions=[(0,0,0)]), 'test/He', 'cc-pvdz', 0)
    fw2 = make_dft_firework(Atoms('He', positions=[(0,0,0)]), 'test/He', 'cc-pvdz', 0)
    fw3 = make_dft_firework(Atoms('He', positions=[(0,0,0)]), 'test/He', 'cc-pvdz', 0, functional='b3lyp')
    fw4 = make_ccsd_firework(Atoms('He', positions=[(0,0,0)]), 'test/He', 'cc-pvdz', 0)
    fw5 = make_ccsd_firework(Atoms('Li', positions=[(0,0,0)]), 'test/Li', 'cc-pvdz', 1)
    launchpad = LaunchPad.auto_load()
    for fw in [fw1, fw2, fw3, fw4, fw5]:
        launchpad.add_wf(fw)
