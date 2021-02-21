from pyscf import dft, gto, scf
from pyscf.scf.stability import uhf_internal
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
                           'GLH', settings['corr_file'])
    corr_model = joblib.load(corr_file)
    settings.update({'corr_model': corr_model})
mlfunc = joblib.load(mlfunc_file)

mol.verbose = 3
mol.build()
pbe = dft.UKS(mol)
pbe.xc = 'PBE'
pbe.max_cycle = 150
pbe.diis_start_cycle = 40
pbe.damp = 6
pbe.kernel()
mo = uhf_internal(pbe)
dm = pbe.make_rdm1(mo_coeff=mo)
pbe.kernel(dm)
mo = uhf_internal(pbe)
dm = pbe.make_rdm1(mo_coeff=mo)
pbe.kernel(dm)
uhf_internal(pbe)

print('END PBE')

mol.verbose=4
mol.build()

ks = numint.setup_uks_calc(mol, mlfunc, **settings)
ks.build()
print('ETOT', ks.energy_tot(pbe.make_rdm1()))

ks.conv_tol = 1e-7
ks.damp = 10
ks.level_shift = (0, 0.02)
ks.diis_start_cycle = 30
ks.max_cycle = 200
#ks.diis_space = 10
ks.grids.level = 3
ks.grids.build()
ks.conv_check = False
ks.kernel(pbe.make_rdm1())

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

