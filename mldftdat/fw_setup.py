from mldftdat.pyscf_tasks import SCFCalc, CCSDCalc, TrainingDataCollector,\
                                LoadCalcFromDB, DFTFromHF, MLSCFCalc, SGXCorrCalc,\
                                GridBenchmark
from mldftdat.workflow_utils import get_save_dir, SAVE_ROOT, ACCDB_DIR,\
                                    read_accdb_structure
from fireworks import Firework, LaunchPad

def get_hf_tasks(struct, mol_id, basis, spin, charge=0, **kwargs):
    calc_type = 'RHF' if spin == 0 else 'UHF'
    struct_dict = struct.todict()
    t1 = SCFCalc(struct=struct_dict, basis=basis, calc_type=calc_type,
                 spin=spin, charge=charge)
    t2 = TrainingDataCollector(save_root_dir = SAVE_ROOT, mol_id=mol_id,
                               **kwargs)
    return t1, t2

def get_dft_tasks(struct, mol_id, basis, spin, functional=None, charge=0,
                  skip_analysis = False):
    calc_type = 'RKS' if spin == 0 else 'UKS'
    struct_dict = struct.todict()
    t1 = SCFCalc(struct=struct_dict, basis=basis, calc_type=calc_type,
                spin=spin, charge=charge, functional=functional)
    t2 = TrainingDataCollector(save_root_dir = SAVE_ROOT, mol_id=mol_id,
                               skip_analysis = skip_analysis)
    return t1, t2

def get_ml_tasks(struct, mol_id, basis, spin, mlfunc_name, mlfunc_file,
                 mlfunc_settings_file, mlfunc_c_file = None, charge=0):
    calc_type = 'RKS' if spin == 0 else 'UKS'
    struct_dict = struct.todict()
    t1 = MLSCFCalc(struct=struct_dict, basis=basis, calc_type=calc_type,
                   spin=spin, charge=charge, mlfunc_name=mlfunc_name,
                   mlfunc_file=mlfunc_file,
                   mlfunc_settings_file=mlfunc_settings_file,
                   mlfunc_c_file=mlfunc_c_file)
    t2 = TrainingDataCollector(save_root_dir=SAVE_ROOT, mol_id=mol_id,
                               skip_analysis=True)
    return t1, t2

def get_sgx_tasks(struct, mol_id, basis, spin, mlfunc_name,
                  mlfunc_settings_file, charge=0):
    calc_type = 'RKS' if spin == 0 else 'UKS'
    struct_dict = struct.todict()
    t1 = SGXCorrCalc(struct=struct_dict, basis=basis, calc_type=calc_type,
                     spin=spin, charge=charge, mlfunc_name = mlfunc_name,
                     mlfunc_settings_file = mlfunc_settings_file)
    t2 = TrainingDataCollector(save_root_dir=SAVE_ROOT, mol_id=mol_id,
                               skip_analysis=False)
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
    return Firework(get_hf_tasks(struct, mol_id, basis, spin, charge),
                    name=name)

def make_dft_firework(struct, mol_id, basis, spin, functional=None, charge=0,
                      name=None, skip_analysis = False):
    return Firework(get_dft_tasks(struct, mol_id, basis, spin,
                    functional=functional, charge=charge,
                    skip_analysis=skip_analysis), name=name)

def make_ml_firework(struct, mol_id, basis, spin, mlfunc_name, mlfunc_file=None,
                     mlfunc_settings_file=None, mlfunc_c_file=None,
                     charge=0, name=None):
    return Firework(get_ml_tasks(struct, mol_id, basis, spin,
                    mlfunc_name, mlfunc_file, mlfunc_settings_file,
                    charge=charge, mlfunc_c_file = mlfunc_c_file), name=name)

def make_sgx_firework(struct, mol_id, basis, spin, mlfunc_name,
                      mlfunc_settings_file, charge=0, name=None):
    return Firework(get_sgx_tasks(struct, mol_id, basis, spin,
                    mlfunc_name, mlfunc_settings_file,
                    charge=charge), name=name)

def make_ccsd_firework(struct, mol_id, basis, spin, charge=0,
                       name=None, **kwargs):
    t1, t2 = get_hf_tasks(struct, mol_id, basis, spin, charge)
    t3 = CCSDCalc()
    t4 = TrainingDataCollector(save_root_dir=SAVE_ROOT, mol_id=mol_id,
                               **kwargs)
    return Firework([t1, t2, t3, t4], name=name)

def make_ccsd_firework_no_hf(struct, mol_id, basis, spin, charge=0,
                             name=None, **kwargs):
    if spin == 0:
        hf_type = 'RHF'
    else:
        hf_type = 'UHF'
    save_dir = get_save_dir(SAVE_ROOT, hf_type, basis, mol_id)
    t1 = LoadCalcFromDB(directory = save_dir)
    t2 = CCSDCalc()
    t3 = TrainingDataCollector(save_root_dir = SAVE_ROOT, mol_id=mol_id,
                               **kwargs)
    return Firework([t1, t2, t3], name=name)

def make_benchmark_firework(functional, radi_method, rad, ang,
                            prune, **kwargs):
    t = GridBenchmark(functional=functional, radi_method=radi_method,
                      rad=rad, ang=ang, prune=prune, **kwargs)
    return Firework([t], name='benchmark')
