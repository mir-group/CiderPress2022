from fireworks import FiretaskBase, FWAction
from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.utilities.fw_serializers import recursive_dict

from ase import Atoms
from pyscf import scf, dft, cc

import json

from mldftdat.pyscf_utils import *
from mldftdat.analyzers import CALC_TYPES
import mldftdat.analyzers
import mldftdat.lowmem_analyzers

import os, psutil, multiprocessing, time, datetime
from itertools import product
from mldftdat.workflow_utils import safe_mem_cap_mb, time_func,\
                                    get_functional_db_name, get_save_dir


@explicit_serialize
class SCFCalc(FiretaskBase):

    required_params = ['struct', 'basis', 'calc_type']
    optional_params = ['spin', 'charge', 'max_conv_tol', 'functional']

    DEFAULT_MAX_CONV_TOL = 1e-7

    def run_task(self, fw_spec):
        atoms = Atoms.fromdict(self['struct'])
        kwargs = {}
        if self.get('spin') is not None:
            kwargs['spin'] = self['spin']
        if self.get('charge') is not None:
            kwargs['charge'] = self['charge']
        max_conv_tol = self.get('max_conv_tol') or self.DEFAULT_MAX_CONV_TOL
        mol = mol_from_ase(atoms, self['basis'], **kwargs)
        calc_type = self['calc_type']

        start_time = time.monotonic()
        calc = run_scf(mol, calc_type, self.get('functional'), remove_ld = True)
        stop_time = time.monotonic()

        max_iter = 50 # extra safety catch
        iter_step = 0
        while not calc.converged and calc.conv_tol < max_conv_tol\
                and iter_step < max_iter:
            iter_step += 1
            print ("Did not converge SCF, increasing conv_tol.")
            calc.conv_tol *= 10

            start_time = time.monotonic()
            calc.kernel()
            stop_time = time.monotonic()

        assert calc.converged, "SCF calculation did not converge!"
        update_spec={
            'calc'      : calc,
            'calc_type' : calc_type,
            'conv_tol'  : calc.conv_tol,
            'cpu_count' : multiprocessing.cpu_count(),
            'mol'       : mol,
            'struct'    : atoms,
            'wall_time' : stop_time - start_time
        }
        if 'KS' in calc_type:
            functional = self.get('functional')
            if functional is None:
                update_spec['functional'] = 'LDA_VWN'
            else:
                update_spec['functional'] = get_functional_db_name(functional)

        return FWAction(update_spec = update_spec)


@explicit_serialize
class SCFFromDB(FiretaskBase):
    """
    Rerun calculation in new basis.
    """

    required_params = ['struct', 'basis', 'oldbasis', 'calc_type', 'mol_id']
    optional_params = ['functional']

    DEFAULT_MAX_CONV_TOL = 1e-7

    def run_task(self, fw_spec):

        if self.get('functional') is not None:
            functional = get_functional_db_name(self['functional'])
        else:
            functional = None
        adir = get_save_dir(os.environ['MLDFTDB'], self['calc_type'],
                            self['oldbasis'], self['mol_id'],
                            functional=functional)
        if 'U' in calc_type:
            analyzer = UHFAnalyzer.load(adir)
        else:
            analyzer = RHFAnalyzer.load(adir)
        init_mol = analyzer.mol
        init_mol.build()
        mol = init_mol.copy()
        mol.basis = self['basis']
        mol.build()
        s = mol.get_ovlp()
        mo = analyzer.mo_coeff
        mo_occ = analyzer.mo_occ
        def fproj(mo):
            mo = addons.project_mo_nr2nr(init_mol, mo, mol)
            norm = numpy.einsum('pi,pi->i', mo.conj(), s.dot(mo))
            mo /= np.sqrt(norm)
            return mo
        if 'U' in calc_type:
            dm = analyzer.calc.make_rdm1([fproj(mo[0]), fproj(mo[1])], mo_occ)
        else:
            dm = analyzer.calc.make_rdm1(fproj(mo), mo_occ)

        start_time = time.monotonic()
        calc = run_scf(mol, calc_type, functional = functional,
                       remove_ld = True, dm0 = dm)
        stop_time = time.monotonic()

        max_iter = 50 # extra safety catch
        iter_step = 0
        while not calc.converged and calc.conv_tol < max_conv_tol\
                and iter_step < max_iter:
            iter_step += 1
            print ("Did not converge SCF, increasing conv_tol.")
            calc.conv_tol *= 10

            start_time = time.monotonic()
            calc.kernel(dm0 = dm)
            stop_time = time.monotonic()

        assert calc.converged, "SCF calculation did not converge!"
        update_spec={
            'calc'      : calc,
            'calc_type' : calc_type,
            'conv_tol'  : calc.conv_tol,
            'cpu_count' : multiprocessing.cpu_count(),
            'mol'       : mol,
            'struct'    : atoms,
            'wall_time' : stop_time - start_time
        }
        if 'KS' in calc_type:
            functional = self.get('functional')
            if functional is None:
                update_spec['functional'] = 'LDA_VWN'
            else:
                update_spec['functional'] = get_functional_db_name(functional)

        return FWAction(update_spec = update_spec)


@explicit_serialize
class MLSCFCalc(FiretaskBase):

    required_params = ['struct', 'basis', 'calc_type',\
                       'mlfunc_name', 'mlfunc_file', 'mlfunc_settings_file']
    optional_params = ['spin', 'charge', 'max_conv_tol',\
                       'mlfunc_c_file']

    DEFAULT_MAX_CONV_TOL = 1e-6

    def run_task(self, fw_spec):
        atoms = Atoms.fromdict(self['struct'])
        kwargs = {}
        if self.get('spin') is not None:
            kwargs['spin'] = self['spin']
        if self.get('charge') is not None:
            kwargs['charge'] = self['charge']
        max_conv_tol = self.get('max_conv_tol') or self.DEFAULT_MAX_CONV_TOL
        mol = mol_from_ase(atoms, self['basis'], **kwargs)
        calc_type = self['calc_type']

        import joblib
        if self.get('mlfunc_c_file') is None:
            from mldftdat.dft import numint6 as numint
            mlfunc = joblib.load(self['mlfunc_file'])
        else:
            from mldftdat.dft import numint5 as numint
            mlfunc = joblib.load(self['mlfunc_file'])
            mlfunc_c = joblib.load(self['mlfunc_c_file'])
        import yaml

        #if not hasattr(mlfunc, 'y_to_f_mul'):
        #    mlfunc.y_to_f_mul = None
        with open(self['mlfunc_settings_file'], 'r') as f:
            settings = yaml.load(f, Loader = yaml.Loader)
            if settings is None:
                settings = {}
        if self.get('mlfunc_c_file') is None:
            if calc_type == 'RKS':
                calc = numint.setup_rks_calc(mol, mlfunc, **settings)
            else:
                calc = numint.setup_uks_calc(mol, mlfunc, **settings)
        else:
            if calc_type == 'RKS':
                calc = numint.setup_rks_calc(mol, mlfunc, mlfunc_c, **settings)
            else:
                calc = numint.setup_uks_calc(mol, mlfunc, mlfunc_c, **settings)

        start_time = time.monotonic()
        calc.kernel()
        stop_time = time.monotonic()

        max_iter = 50 # extra safety catch
        iter_step = 0
        while not calc.converged and calc.conv_tol < max_conv_tol\
                and iter_step < max_iter:
            iter_step += 1
            print ("Did not converge SCF, increasing conv_tol.")
            calc.conv_tol *= 10

            start_time = time.monotonic()
            calc.kernel()
            stop_time = time.monotonic()

        assert calc.converged, "SCF calculation did not converge!"
        update_spec={
            'calc'      : calc,
            'calc_type' : calc_type,
            'conv_tol'  : calc.conv_tol,
            'cpu_count' : multiprocessing.cpu_count(),
            'mol'       : mol,
            'struct'    : atoms,
            'wall_time' : stop_time - start_time
        }
        update_spec['functional'] = get_functional_db_name(self['mlfunc_name'])

        return FWAction(update_spec = update_spec)


@explicit_serialize
class SGXCorrCalc(FiretaskBase):

    required_params = ['struct', 'basis', 'calc_type',\
                       'mlfunc_name', 'mlfunc_settings_file']
    optional_params = ['spin', 'charge', 'max_conv_tol']

    DEFAULT_MAX_CONV_TOL = 1e-8

    def run_task(self, fw_spec):
        atoms = Atoms.fromdict(self['struct'])
        kwargs = {}
        if self.get('spin') is not None:
            kwargs['spin'] = self['spin']
        if self.get('charge') is not None:
            kwargs['charge'] = self['charge']
        max_conv_tol = self.get('max_conv_tol') or self.DEFAULT_MAX_CONV_TOL
        mol = mol_from_ase(atoms, self['basis'], **kwargs)
        calc_type = self['calc_type']

        from mldftdat.dft import sgx_corr as numint
        import yaml

        with open(self['mlfunc_settings_file'], 'r') as f:
            settings = yaml.load(f, Loader = yaml.Loader)
            if settings is None:
                settings = {}
        if calc_type == 'RKS':
            calc = numint.setup_rks_calc(mol, **settings)
        else:
            calc = numint.setup_uks_calc(mol, **settings)

        start_time = time.monotonic()
        calc.kernel()
        stop_time = time.monotonic()

        max_iter = 50 # extra safety catch
        iter_step = 0
        while not calc.converged and calc.conv_tol < max_conv_tol\
                and iter_step < max_iter:
            iter_step += 1
            print ("Did not converge SCF, increasing conv_tol.")
            calc.conv_tol *= 10

            start_time = time.monotonic()
            calc.kernel()
            stop_time = time.monotonic()

        assert calc.converged, "SCF calculation did not converge!"
        update_spec={
            'calc'      : calc,
            'calc_type' : calc_type,
            'conv_tol'  : calc.conv_tol,
            'cpu_count' : multiprocessing.cpu_count(),
            'mol'       : mol,
            'struct'    : atoms,
            'wall_time' : stop_time - start_time
        }
        update_spec['functional'] = get_functional_db_name(self['mlfunc_name'])

        return FWAction(update_spec = update_spec)


@explicit_serialize
class LoadCalcFromDB(FiretaskBase):

    required_params = ['directory']

    def run_task(self, fw_spec):

        with open(os.path.join(self['directory'], 'run_info.json'), 'r') as f:
            info_dict = json.load(f)
        atoms = Atoms.fromdict(info_dict['struct'])
        data_file = os.path.join(self['directory'], 'data.hdf5')
        calc, calc_type = load_calc(data_file)

        update_spec={
            'calc'      : calc,
            'calc_type' : calc_type,
            'conv_tol'  : calc.conv_tol,
            'cpu_count' : multiprocessing.cpu_count(),
            'load_dir'  : self['directory'],
            'mol'       : calc.mol,
            'struct'    : atoms,
            'wall_time' : 'NA'
        }
        if 'KS' in calc_type:
            functional = calc.xc
            update_spec['functional'] = get_functional_db_name(functional)

        return FWAction(update_spec = update_spec)


@explicit_serialize
class DFTFromHF(FiretaskBase):
    # can run this after loading HF with LoadCalcFromDB

    required_params = ['functional']

    def run_task(self, fw_spec):
        hf_calc_type = fw_spec['calc_type']
        hf_calc = fw_spec['calc']
        if hf_calc_type == 'RHF':
            dft_type = 'RKS'
        elif hf_calc_type == 'UHF':
            dft_type = 'UKS'
        else:
            raise ValueError('HF calc_type {} not supported.'.format(hf_calc_type))

        print("RUNNING A DFT FROM HF OF TYPE", dft_type)

        init_rdm1 = hf_calc.make_rdm1()
        calc = run_scf(hf_calc.mol, dft_type, functional = self.get('functional'),
                        dm0 = init_rdm1)

        max_iter = 50 # extra safety catch
        iter_step = 0
        while not calc.converged and calc.conv_tol < 1e-7\
                and iter_step < max_iter:
            iter_step += 1
            print ("Did not converge SCF, increasing conv_tol.")
            calc.conv_tol *= 10

            start_time = time.monotonic()
            calc.kernel(dm0 = init_rdm1)
            stop_time = time.monotonic()

        assert calc.converged, "SCF calculation did not converge!"
        update_spec={
            'calc'      : calc,
            'calc_type' : dft_type,
            'conv_tol'  : calc.conv_tol,
            'cpu_count' : multiprocessing.cpu_count(),
            'mol'       : calc.mol,
            'wall_time' : stop_time - start_time
        }
        if 'KS' in calc_type:
            functional = self.get('functional')
            if functional is None:
                update_spec['functional'] = 'LDA_VWN'
            else:
                update_spec['functional'] = get_functional_db_name(functional)

        return FWAction(update_spec = update_spec)


# old name so that Fireworks can be rerun
@explicit_serialize
class HFCalc(SCFCalc):
    pass

        
@explicit_serialize
class CCSDCalc(FiretaskBase):

    optional_params = ['max_conv_tol']

    DEFAULT_MAX_CONV_TOL = 1e-5

    def run_task(self, fw_spec):
        mol = fw_spec['mol']
        hfcalc = fw_spec['calc']

        start_time = time.monotonic()
        calc = run_cc(hfcalc)
        stop_time = time.monotonic()
        if not calc.converged:
            calc.diis_start_cycle = 4
            calc.diis_space = 10
            start_time = time.monotonic()
            calc.kernel()
            stop_time = time.monotonic()

        max_iter = 50 # extra safety catch
        iter_step = 0
        max_conv_tol = self.get('max_conv_tol') or self.DEFAULT_MAX_CONV_TOL
        while not calc.converged and calc.conv_tol < max_conv_tol\
                and iter_step < max_iter:
            iter_step += 1
            print ("Did not converge CCSD, increasing conv_tol.")
            calc.conv_tol *= 10

            start_time = time.monotonic()
            calc.kernel()
            stop_time = time.monotonic()

        assert calc.converged, "CCSD calculation did not converge!"
        calc_type = 'CCSD' if type(calc) == cc.ccsd.CCSD else 'UCCSD'
        return FWAction(update_spec={
                'calc'      : calc,
                'calc_type' : calc_type,
                'conv_tol'  : calc.conv_tol,
                'cpu_count' : multiprocessing.cpu_count(),
                'hfcalc'    : hfcalc,
                'wall_time' : stop_time - start_time
            })


@explicit_serialize
class TrainingDataCollector(FiretaskBase):

    required_params = ['save_root_dir', 'mol_id']
    optional_params = ['no_overwrite', 'skip_analysis', 'ccsd_t']
    implemented_calcs = ['RHF', 'UHF', 'RKS', 'UKS', 'CCSD', 'UCCSD']

    def run_task(self, fw_spec):

        print('STARTING TRAINING DATA TASK', psutil.virtual_memory().available // 1e6)
        calc = fw_spec['calc']
        assert calc.converged, "This training data is not converged!"
        calc_type = fw_spec['calc_type']
        mol = fw_spec['mol']
        struct = fw_spec['struct']
        mol_dat = {
            'calc_type'  : calc_type,
            'charge'     : mol.charge,
            'conv_tol'   : calc.conv_tol,
            'cpu_count'  : fw_spec['cpu_count'],
            'e_tot'      : calc.e_tot,
            'functional' : fw_spec.get('functional'),
            'max_cycle'  : calc.max_cycle,
            'nelectron'  : mol.nelectron,
            'spin'       : mol.spin,
            'struct'     : struct.todict(),
            'task_run'   : str(datetime.datetime.now()),
            'wall_time'  : fw_spec['wall_time']
        }

        if calc.mo_coeff.shape[-1] < 200:
            print ("USING STD ANALYZER MODULE")
            analyzer_module = mldftdat.analyzers
        else:
            print("USING LOWMEM ANALYZER MODULE")
            analyzer_module = mldftdat.lowmem_analyzers

        if type(calc) == scf.hf.RHF:
            Analyzer = analyzer_module.RHFAnalyzer
        elif type(calc) == dft.rks.RKS:
            Analyzer = analyzer_module.RHFAnalyzer
        elif type(calc) == scf.uhf.UHF:
            Analyzer = analyzer_module.UHFAnalyzer
        elif type(calc) == dft.uks.UKS:
            Analyzer = analyzer_module.UHFAnalyzer
        elif type(calc) == cc.ccsd.CCSD:
            Analyzer = analyzer_module.CCSDAnalyzer
        elif type(calc) == cc.uccsd.UCCSD:
            Analyzer = analyzer_module.UCCSDAnalyzer
        else:
            raise NotImplementedError(
                'Training data collection not supported for {}'.format(type(calc)))

        if self.get('no_overwrite'):
            exist_ok = False
        else:
            exist_ok = True
        save_dir = get_save_dir(self['save_root_dir'], calc_type,
                                mol.basis, self['mol_id'],
                                functional = fw_spec.get('functional'))
        os.makedirs(save_dir, exist_ok=exist_ok)

        analyzer = Analyzer(calc, max_mem=safe_mem_cap_mb())
        print('MADE ANALYZER', psutil.virtual_memory().available // 1e6)
        if not self.get('skip_analysis'):
            analyzer.perform_full_analysis()
        if calc_type in ['CCSD', 'UCCSD'] and self.get('ccsd_t'):
            analyzer.calc_pert_triples()
        print('FINISHED ANALYSIS', psutil.virtual_memory().available // 1e6)
        analyzer.dump(os.path.join(save_dir, 'data.hdf5'))
        print('DUMP ANALYSIS', psutil.virtual_memory().available // 1e6)

        mol_dat['grid_size'] = analyzer.grid.weights.shape[0]
        mol_dat['basis_size'] = analyzer.rdm1.shape[-1]

        info_file = os.path.join(save_dir, 'run_info.json')
        f = open(info_file, 'w')
        json.dump(recursive_dict(mol_dat), f, indent=4, sort_keys=True)
        f.close()

        return FWAction(stored_data={'save_dir': save_dir})


@explicit_serialize
class SCFCalcConvergenceFixer(FiretaskBase):

    required_params = ['struct', 'basis', 'calc_type']
    optional_params = ['spin', 'charge', 'max_conv_tol', 'functional']

    DEFAULT_MAX_CONV_TOL = 1e-7

    def run_task(self, fw_spec):
        atoms = Atoms.fromdict(self['struct'])
        print('Running SCF for {}'.format(atoms.get_chemical_formula()))
        kwargs = {}
        if self.get('spin') is not None:
            kwargs['spin'] = self['spin']
        if self.get('charge') is not None:
            kwargs['charge'] = self['charge']
        max_conv_tol = self.get('max_conv_tol') or self.DEFAULT_MAX_CONV_TOL
        mol = mol_from_ase(atoms, self['basis'], **kwargs)
        calc_type = self['calc_type']

        start_time = time.monotonic()
        calc = run_scf(mol, calc_type, self.get('functional'), remove_ld=True)
        stop_time = time.monotonic()

        max_iter = 50 # extra safety catch
        iter_step = 0
        #diis_types = [scf.diis.SCF_DIIS, scf.diis.ADIIS, scf.diis.EDIIS, None]
        diis_types = [scf.diis.ADIIS, scf.diis.EDIIS, scf.diis.SCF_DIIS, None]
        init_guess_types = ['minao', 'atom', '1e']
        diis_options_list = [(8, 1), (14, 4)]
        while not calc.converged and calc.conv_tol <= max_conv_tol\
                and iter_step < max_iter:
            iter_step += 1
            print ("Did not converge SCF, changing params.")
            
            calc.max_cycle = 200
            calc.direct_scf = False
            for DIIS, init_guess, diis_opts in product(diis_types,
                                                       init_guess_types,
                                                       diis_options_list):

                if DIIS is None:
                    calc.diis = False
                else:
                    calc.DIIS = DIIS
                calc.init_guess = init_guess
                calc.diis_space = diis_opts[0]
                calc.diis_start_cycle = diis_opts[1]

                start_time = time.monotonic()
                calc.kernel()
                stop_time = time.monotonic()

                if calc.converged:
                    print('Fixed convergence issues! {} {} {}'.format(
                            init_guess, diis_opts, DIIS))
                    break
            else:
                print("Increasing convergence tolerance to %e" % (calc.conv_tol * 10))
                calc.conv_tol *= 10

        assert calc.converged, "SCF calculation did not converge!"
        update_spec={
            'calc'      : calc,
            'calc_type' : calc_type,
            'conv_tol'  : calc.conv_tol,
            'cpu_count' : multiprocessing.cpu_count(),
            'mol'       : mol,
            'struct'    : atoms,
            'wall_time' : stop_time - start_time
        }
        if 'KS' in calc_type:
            functional = self.get('functional')
            if functional is None:
                update_spec['functional'] = 'LDA_VWN'
            else:
                update_spec['functional'] = get_functional_db_name(functional)

        return FWAction(update_spec = update_spec)


class MLSCFCalcConvergenceFixer(FiretaskBase):

    required_params = ['struct', 'basis', 'calc_type',\
                       'mlfunc_name', 'mlfunc_file', 'mlfunc_settings_file']
    optional_params = ['spin', 'charge', 'max_conv_tol',\
                       'mlfunc_c_file']

    DEFAULT_MAX_CONV_TOL = 1e-8

    def run_task(self, fw_spec):
        atoms = Atoms.fromdict(self['struct'])
        kwargs = {}
        if self.get('spin') is not None:
            kwargs['spin'] = self['spin']
        if self.get('charge') is not None:
            kwargs['charge'] = self['charge']
        max_conv_tol = self.get('max_conv_tol') or self.DEFAULT_MAX_CONV_TOL
        mol = mol_from_ase(atoms, self['basis'], **kwargs)
        calc_type = self['calc_type']

        import joblib
        if self.get('mlfunc_c_file') is None:
            from mldftdat.dft import numint6 as numint
            mlfunc = joblib.load(self['mlfunc_file'])
        else:
            from mldftdat.dft import numint5 as numint
            mlfunc = joblib.load(self['mlfunc_file'])
            mlfunc_c = joblib.load(self['mlfunc_c_file'])
        import yaml

        #if not hasattr(mlfunc, 'y_to_f_mul'):
        #    mlfunc.y_to_f_mul = None
        with open(self['mlfunc_settings_file'], 'r') as f:
            settings = yaml.load(f, Loader = yaml.Loader)
            if settings is None:
                settings = {}
        if self.get('mlfunc_c_file') is None:
            if calc_type == 'RKS':
                calc = numint.setup_rks_calc(mol, mlfunc, **settings)
            else:
                calc = numint.setup_uks_calc(mol, mlfunc, **settings)
        else:
            if calc_type == 'RKS':
                calc = numint.setup_rks_calc(mol, mlfunc, mlfunc_c, **settings)
            else:
                calc = numint.setup_uks_calc(mol, mlfunc, mlfunc_c, **settings)

        start_time = time.monotonic()
        calc.kernel()
        stop_time = time.monotonic()

        max_iter = 50 # extra safety catch
        iter_step = 0
        #diis_types = [scf.diis.SCF_DIIS, scf.diis.ADIIS, scf.diis.EDIIS, None]
        diis_types = [scf.diis.ADIIS, scf.diis.EDIIS, scf.diis.SCF_DIIS, None]
        init_guess_types = ['minao', 'atom', '1e']
        diis_options_list = [(8, 1), (14, 4)]
        while not calc.converged and calc.conv_tol <= max_conv_tol\
                and iter_step < max_iter:
            iter_step += 1
            print ("Did not converge SCF, changing params.")

            calc.max_cycle = 100
            calc.direct_scf = False
            for DIIS, init_guess, diis_opts in product(diis_types,
                                                       init_guess_types,
                                                       diis_options_list):

                if DIIS is None:
                    calc.diis = False
                else:
                    calc.DIIS = DIIS
                calc.init_guess = init_guess
                calc.diis_space = diis_opts[0]
                calc.diis_start_cycle = diis_opts[1]

                start_time = time.monotonic()
                calc.kernel()
                stop_time = time.monotonic()

                if calc.converged:
                    print('Fixed convergence issues! {} {} {}'.format(
                            init_guess, diis_opts, DIIS))
                    break
            else:
                print("Increasing convergence tolerance to %e" % (calc.conv_tol * 10))
                calc.conv_tol *= 10

        assert calc.converged, "SCF calculation did not converge!"
        update_spec={
            'calc'      : calc,
            'calc_type' : calc_type,
            'conv_tol'  : calc.conv_tol,
            'cpu_count' : multiprocessing.cpu_count(),
            'mol'       : mol,
            'struct'    : atoms,
            'wall_time' : stop_time - start_time
        }
        update_spec['functional'] = get_functional_db_name(self['mlfunc_name'])

        return FWAction(update_spec = update_spec)
