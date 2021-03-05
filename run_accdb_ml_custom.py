from pyscf import dft, gto, scf
from mldftdat.workflow_utils import SAVE_ROOT, ACCDB_DIR, read_accdb_structure
from mldftdat.pyscf_tasks import TrainingDataCollector
from mldftdat.pyscf_utils import mol_from_ase
import sys
import os
import yaml
import joblib

BAS='def2-qzvppd'
struct_id = sys.argv[1]
functional = sys.argv[2]
struct, mol_id, spin, charge = read_accdb_structure(struct_id)
mol = mol_from_ase(struct, BAS, spin, charge)
mol.ecp = BAS
mol.verbose = 4
mol.build()

settings_fname = functional + '.yaml'
settings_fname = os.path.join(SAVE_ROOT, 'MLFUNCTIONALS',
                              settings_fname)

from mldftdat.dft import numint
with open(settings_fname, 'r') as f:
    settings = yaml.load(f, Loader=yaml.Loader)
mlfunc_file = os.path.join(SAVE_ROOT, 'MLFUNCTIONALS',
                           'CIDER', settings['mlfunc_file'])
if settings.get('corr_file') is not None:
    corr_file = os.path.join(SAVE_ROOT, 'MLFUNCTIONALS',
                           'GLH', settings['mlfunc_c_file'])
    corr_model = joblib.load(corr_file)
    settings.update({'corr_model': corr_model})
mlfunc = joblib.load(mlfunc_file)
ks = numint.setup_uks_calc(mol, mlfunc, **settings)

ks.conv_tol = 1e-7
ks.damp = 30
ks.DIIS = scf.diis.ADIIS
ks.level_shift = 0.1
ks.diis_start_cycle = 200
ks.max_cycle = 400
ks.diis_space = 20
ks.grids.level = 3
ks.grids.build()
ks.kernel()
dm = ks.make_rdm1()
ks.damp = 80
ks.level_shift = 0
ks.kernel(dm)

fw_spec = {}
fw_spec['calc'] = ks
fw_spec['calc_type'] = 'UKS'
fw_spec['mol'] = mol
fw_spec['struct'] = struct
fw_spec['cpu_count'] = 'NA'
fw_spec['functional'] = functional
fw_spec['wall_time'] = 'NA'

task = TrainingDataCollector(save_root_dir=SAVE_ROOT, mol_id=mol_id)
task.run_task(fw_spec)
