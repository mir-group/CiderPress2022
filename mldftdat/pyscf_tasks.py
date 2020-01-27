from fireworks import FiretaskBase, FWAction
from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.utilities.fw_serializers import recursive_dict
from ase import Atoms
from mldftdat.pyscf_utils import *
import json
import datetime
from datetime import date
from mldftdat.analyzers import RHFAnalyzer, UHFAnalyzer, CCSDAnalyzer, UCCSDAnalyzer, RKSAnalyzer, UKSAnalyzer
import os
import psutil

from pyscf import gto, scf, dft, cc, fci


@explicit_serialize
class SCFCalc(FiretaskBase):

    required_params = ['struct', 'basis', 'calc_type']
    optional_params = ['spin', 'charge', 'max_conv_tol']

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
        calc = run_scf(mol, calc_type)
        max_iter = 50 # extra safety catch
        iter_step = 0
        while not calc.converged and calc.conv_tol < max_conv_tol\
                and iter_step < max_iter:
            iter_step += 1
            print ("Did not converge SCF, increasing conv_tol.")
            calc.conv_tol *= 10
            calc.kernel()
        assert calc.converged, "SCF calculation did not converge!"
        return FWAction(update_spec={
                'calc_type' :  calc_type,
                'struct'    :  atoms,
                'mol'       :  mol,
                'calc'      :  calc,
                'conv_tol'  :  calc.conv_tol
            })


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
        calc = run_cc(hfcalc)
        max_iter = 50 # extra safety catch
        iter_step = 0
        max_conv_tol = self.get('max_conv_tol') or self.DEFAULT_MAX_CONV_TOL
        while not calc.converged and calc.conv_tol < max_conv_tol\
                and iter_step < max_iter:
            iter_step += 1
            print ("Did not converge CCSD, increasing conv_tol.")
            calc.conv_tol *= 10
            calc.kernel()
        assert calc.converged, "CCSD calculation did not converge!"
        calc_type = 'CCSD' if type(calc) == cc.ccsd.CCSD else 'UCCSD'
        return FWAction(update_spec={
                'calc'      : calc,
                'hfcalc'    : hfcalc,
                'calc_type' : calc_type,
                'conv_tol'  : calc.conv_tol
            })


def get_general_data(analyzer):
    ao_data, rho_data = get_mgga_data(analyzer.mol, analyzer.grid,
                                        analyzer.rdm1)
    return {
        'coords': analyzer.grid.coords,
        'weights': analyzer.grid.weights,
        'mo_coeff': analyzer.mo_coeff,
        'mo_occ': analyzer.mo_occ,
        'ao_vals': analyzer.ao_vals,
        'mo_vals': analyzer.mo_vals,
        #'ao_vele_mat': analyzer.ao_vele_mat,
        #'mo_vele_mat': analyzer.mo_vele_mat,
        'ha_energy_density': analyzer.get_ha_energy_density(),
        'ee_energy_density': analyzer.get_ee_energy_density(),
        'ao_data': ao_data,
        'rho_data': rho_data
    }

def safe_mem_cap_mb():
    return int(psutil.virtual_memory().available // 4e6)

def analyze_rhf(calc):
    analyzer = RHFAnalyzer(calc, max_mem=safe_mem_cap_mb())
    data_dict = get_general_data(analyzer)
    data_dict['mo_energy'] = analyzer.mo_energy
    data_dict['fx_energy_density'] = analyzer.get_fx_energy_density()
    data_dict['rdm1'] = analyzer.rdm1
    return data_dict

def analyze_rks(calc):
    analyzer = RKSAnalyzer(calc, max_mem=safe_mem_cap_mb())
    data_dict = get_general_data(analyzer)
    data_dict['mo_energy'] = analyzer.mo_energy
    data_dict['fx_energy_density'] = analyzer.get_fx_energy_density()
    data_dict['rdm1'] = analyzer.rdm1
    return data_dict

def analyze_uhf(calc):
    analyzer = UHFAnalyzer(calc, max_mem=safe_mem_cap_mb())
    data_dict = get_general_data(analyzer)
    data_dict['mo_energy'] = analyzer.mo_energy
    data_dict['rdm1'] = analyzer.rdm1
    data_dict['fx_energy_density'] = analyzer.get_fx_energy_density()
    data_dict['fx_energy_density_u'] = analyzer.fx_energy_density_u
    data_dict['fx_energy_density_d'] = analyzer.fx_energy_density_d
    return data_dict

def analyze_uks(calc):
    analyzer = UKSAnalyzer(calc, max_mem=safe_mem_cap_mb())
    data_dict = get_general_data(analyzer)
    data_dict['mo_energy'] = analyzer.mo_energy
    data_dict['rdm1'] = analyzer.rdm1
    data_dict['fx_energy_density'] = analyzer.get_fx_energy_density()
    data_dict['fx_energy_density_u'] = analyzer.fx_energy_density_u
    data_dict['fx_energy_density_d'] = analyzer.fx_energy_density_d
    return data_dict

def analyze_ccsd(calc):
    analyzer = CCSDAnalyzer(calc, max_mem=safe_mem_cap_mb())
    data_dict = get_general_data(analyzer)
    data_dict['ao_rdm1'] = analyzer.ao_rdm1
    data_dict['ao_rdm2'] = analyzer.ao_rdm2
    data_dict['mo_rdm1'] = analyzer.mo_rdm1
    data_dict['mo_rdm2'] = analyzer.mo_rdm2
    data_dict['xc_energy_density'] = data_dict['ee_energy_density']\
                                        - data_dict['ha_energy_density']
    return data_dict

def analyze_uccsd(calc):
    analyzer = UCCSDAnalyzer(calc, max_mem=safe_mem_cap_mb())
    data_dict = get_general_data(analyzer)
    data_dict['ao_rdm1'] = analyzer.ao_rdm1
    data_dict['ao_rdm2'] = analyzer.ao_rdm2
    data_dict['mo_rdm1'] = analyzer.mo_rdm1
    data_dict['mo_rdm2'] = analyzer.mo_rdm2
    data_dict['xc_energy_density'] = data_dict['ee_energy_density']\
                                        - data_dict['ha_energy_density']
    data_dict['ee_energy_density_uu'] = analyzer.ee_energy_density_uu
    data_dict['ee_energy_density_ud'] = analyzer.ee_energy_density_ud
    data_dict['ee_energy_density_dd'] = analyzer.ee_energy_density_dd
    return data_dict


@explicit_serialize
class TrainingDataCollector(FiretaskBase):

    required_params = ['save_root_dir', 'mol_id']
    optional_params = ['overwrite']
    implemented_calcs = ['RHF', 'UHF', 'CCSD', 'UCCSD']

    def run_task(self, fw_spec):

        calc = fw_spec['calc']
        assert calc.converged, "This training data is not converged!"
        calc_type = fw_spec['calc_type']
        mol = fw_spec['mol']
        struct = fw_spec['struct']
        mol_dat = {
            'mol': gto.mole.pack(mol), # recover with gto.mole.unpack(dict)
            'calc_type': calc_type,
            'struct': struct.todict(),
            'task_run': str(datetime.datetime.now()),
            'conv_tol': calc.conv_tol,
            'max_cycle': calc.max_cycle,
            'e_tot': calc.e_tot
        }
        if type(calc) == scf.hf.RHF:
            arrays = analyze_rhf(calc)
        elif type(calc) == dft.rks.RKS:
            arrays = analyze_rks(calc)
        elif type(calc) == scf.uhf.UHF:
            arrays = analyze_uhf(calc)
        elif type(calc) == dft.uks.UKS:
            arrays = analyze_uks(calc)
        elif type(calc) == cc.ccsd.CCSD:
            arrays = analyze_ccsd(calc)
        elif type(calc) == cc.uccsd.UCCSD:
            arrays = analyze_uccsd(calc)
        else:
            raise NotImplementedError(
                'Training data collection not supported for {}'.format(type(calc)))

        if not self.get('overwrite'):
            exist_ok = False
        else:
            exist_ok = True
        save_dir = os.path.join(self['save_root_dir'], calc_type,
                mol.basis, self['mol_id'] + '-%s' % struct.get_chemical_formula())
        os.makedirs(save_dir, exist_ok=True)

        mol_dat_file = os.path.join(save_dir, 'mol_dat.json')
        f = open(mol_dat_file, 'w')
        json.dump(recursive_dict(mol_dat), f, indent=4, sort_keys=True)
        f.close()

        for dat_name, dat_arr in arrays.items():
            arr_file = os.path.join(save_dir, dat_name)
            np.save(arr_file, np.array(dat_arr, copy=False))

        return FWAction(stored_data={'save_dir': save_dir})
