from fireworks import FiretaskBase, FWAction
from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.utilities.fw_serializers import recursive_dict
from ase import Atoms
from mldftdat.pyscf_utils import *
import json
from datetime import date
from mldftdat.analyzers import RHFAnalyzer, UHFAnalyzer, CCSDAnalyzer, UCCSDAnalyzer
import os

from pyscf import gto, scf, dft, cc, fci


@explicit_serialize
class HFCalc(FiretaskBase):

    required_params = ['struct', 'basis', 'calc_type']
    optional_params = ['spin', 'charge']

    calc_opts = {
                'RHF': scf.RHF,
                'UHF': scf.UHF,
            }

    def run_task(self, fw_spec):
        atoms = Atoms.fromdict(self['struct'])
        kwargs = {}
        if self.get('spin'):
            kwargs['spin'] = spin
        if self.get('charge'):
            kwargs['charge'] = charge
        mol = mol_from_ase(atoms, self['basis'], **kwargs)
        calc_type = self['calc_type']
        calc = self.calc_opts[calc_type](mol)
        calc.kernel()
        return FWAction(update_spec={
                'calc_type' :  calc_type,
                'struct'    :  atoms,
                'mol'       :  mol,
                'calc'      :  calc,
                'rdm1'      :  calc.make_rdm1()
            })

        
@explicit_serialize
class CCSDCalc(FiretaskBase):

    def run_task(self, fw_spec):
        mol = fw_spec['mol']
        hfcalc = fw_spec['calc']
        calc = run_cc(hfcalc)
        calc_type = 'CCSD' if type(calc) == cc.ccsd.CCSD else 'UCCSD'
        return FWAction(update_spec={
                'calc'      : calc,
                'hfcalc'    : hfcalc,
                'calc_type' : calc_type,
            })


@explicit_serialize
class DFTCalc(FiretaskBase):

    required_params = ['struct', 'basis']
    optional_params = ['calc_type']

    calc_opts = {
        'RKS': dft.RKS,
        'UKS': dft.UKS,
    }


@explicit_serialize
class TrainingDataCollector(FiretaskBase):

    implemented_calcs = ['RHF', 'CCSD']

    def run_task(self, fw_spec):

        calc = fw_spec['calc']
        calc_type = fw_spec['calc_type']
        mol = fw_spec['mol']
        mol_dat = {
            'atom': mol.atom,
            'calc_type': calc_type,
            'basis': mol.basis,
        }

        rdm1 = fw_spec['rdm1']
        grid = get_grid['mol']
        ao_data = eval_ao(mol, grid.coords, deriv=3)
        rho_data = eval_rho(mol, ao_data, rdm1, xctype='mGGA')
        ao_vals = ao_data[0,:,:]
        vele_mat = get_vele_mat(mol, grid.coords)
        arrays = {
            'rdm1': rdm1,
            'coords': grid.coords,
            'weights': grid.weights,
            'ao_data': ao_data,
            'rho_data': rho_data
        }

        eha = get_ha_energy_density(mol, rdm1, vele_mat, ao_vals)
        if calc_type == 'RHF':
            exc = get_fx_energy_density(mol, rdm1, vele_mat, ao_vals)
            eee = eha + exc
            arrays['mo_coeff'] = calc.mo_coeff

        elif calc_type == 'UHF':
            exc = get_fx_energy_density_unrestricted(mol, rdm1, vele_mat, ao_vals)
            eee = eha + exc
            arrays['mo_coeff'] = calc.mo_coeff

        elif calc_type == 'CCSD':
            eee = get_ee_energy_density(mol, fw_spec['rdm2'], vele_mat, ao_vals)
            exc = eee - eha
            arrays['rdm2'] = fw_spec['rdm2']
            arrays['mo_coeff'] = fw_spec['hfcalc'].mo_coeff

        else:
            raise NotImplementedError(
                'Training data collection not supported for {}'.format(calc_type))

        arrays.update({'eha' : eha, 'exc' : exc, 'eee' : eee})

        mol_dat['arrays'] = arrays

        return FWAction(update_spec = {'mol_dat': mol_dat})

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
                'ao_vele_mat': analyzer.ao_vele_mat,
                'mo_vele_mat': analyzer.mo_vele_mat,
                'ha_energy_density': analyzer.get_ha_energy_density(),
                'ee_energy_density': analyzer.get_ee_energy_density(),
                'ao_data': ao_data,
                'rho_data': rho_data
            }

def analyze_rhf(calc):
    analyzer = RHFAnalyzer(calc)
    data_dict = get_general_data(analyzer)
    data_dict['mo_energy'] = analyzer.mo_energy
    data_dict['fx_energy_density'] = analyzer.get_fx_energy_density()
    data_dict['rdm1'] = analyzer.rdm1
    return data_dict

def analyze_uhf(calc):
    analyzer = UHFAnalyzer(calc)
    data_dict = get_general_data(analyzer)
    data_dict['mo_energy'] = analyzer.mo_energy
    data_dict['rdm1'] = analyzer.rdm1
    data_dict['fx_energy_density'] = analyzer.get_fx_energy_density()
    data_dict['fx_energy_density_u'] = analyzer.fx_energy_density_u
    data_dict['fx_energy_density_d'] = analyzer.fx_energy_density_d
    return data_dict

def analyze_ccsd(calc):
    analyzer = CCSDAnalyzer(calc)
    data_dict = get_general_data(analyzer)
    data_dict['ao_rdm1'] = analyzer.ao_rdm1
    data_dict['ao_rdm2'] = analyzer.ao_rdm2
    data_dict['mo_rdm1'] = analyzer.mo_rdm1
    data_dict['mo_rdm2'] = analyzer.mo_rdm2
    data_dict['xc_energy_density'] = data_dict['ee_energy_density']\
                                        - data_dict['ha_energy_density']
    return data_dict

def analyze_uccsd(calc):
    analyzer = UCCSDAnalyzer(calc)
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
        calc_type = fw_spec['calc_type']
        mol = fw_spec['mol']
        struct = fw_spec['struct']
        mol_dat = {
            'atom': mol.atom,
            'calc_type': calc_type,
            'basis': mol.basis,
            'struct': struct.todict(),
            'task_run': date.today()
        }
        if type(calc) == scf.hf.RHF:
            arrays = analyze_rhf(calc)
        elif type(calc) == scf.uhf.UHF:
            arrays = analyze_uhf(calc)
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
            np.save(arr_file, dat_arr)
