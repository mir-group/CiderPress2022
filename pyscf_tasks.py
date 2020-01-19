from fireworks import FiretaskBase
from fireworks.utilities.fw_utilities import explicit_serialize
import ase.io
from pyscf_utils import *
import json

from pyscf import gto, scf, dft, cc, fci


@explicit_serialize
class HFCalc(FiretaskBase):

    required_params = ['name', 'struct', 'basis']
    optional_params = ['calc_type']

    calc_opts = {
                'RHF': scf.RHF,
                'UHF': scf.UHF,
            }

    def run_task(self, fw_spec):
        atoms = ase.io.read(self['struct'], format='xyz')
        mol = pyscf_utils.mol_from_ase(atoms, self['basis'])
        if self['calc_type'] == None:
            calc_type = 'RHF'
        else:
            calc_type = self['calc_type']
        calc = self.calc_opts[calc_type](mol)
        calc.kernel()
        return FWAction(update_spec={
                'calc_type' :  calc_type,
                'struct'    :  struct,
                'mol'       :  mol,
                'calc'      :  calc,
                'rdm1'      :  calc.make_rdm1()
            })

        
@explicit_serialize
class CCSDCalc(FiretaskBase):

    def run_task(self, fw_spec):
        mol = fw_spec['mol']
        hfcalc = fw_spec['calc']
        calc = run_cc(mol, hfcalc)
        return FWAction(update_spec={
                'calc'      : calc,
                'hfcalc'    : hfcalc,
                'calc_type' : 'CCSD',
            })


@explicit_serialize
class DFTCalc(FiretaskBase):

    required_params = ['name', 'struct', 'basis']
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
            'calc_type': calc_type
            'basis': mol.basis
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
        

@explicit_serialize
class TrainingDataSaver(FiretaskBase):

    required_params = ['save_root_dir', 'id']
    optional_params = ['overwrite']

    def run_task(self, fw_spec):

        mol_dat = fw_spec['mol_dat']
        struct = fw_spec['struct']

        calc_type = mol_dat['calc_type']
        basis = mol_dat['basis']

        mol = fw_spec['mol']
        calc = fw_spec['calc']

        if not self.get('overwrite'):
            exist_ok = False
        else:
            exist_ok = True
        save_dir = os.path.join(self['save_root_dir'], calc_type,
                basis, self['id'] + '-%s' % struct.get_chemical_formula())
        os.makedirs(save_dir, exist_ok=exist_ok)

        arrays = mol_dat.pop('arrays')
        mol_dat_file = os.path.join(save_dir, 'mol_dat.json')
        f = open(save_dir, 'w')
        json.dump(mol_dat, f, indent=4, sort_keys=True)
        f.close()

        for dat_name, dat_arr in arrays.items():
            arr_file = os.path.join(save_dir, dat_name)
            np.save(arr_file, dat_arr)
