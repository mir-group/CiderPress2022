from pyscf import scf, dft, gto, ao2mo, df, lib, fci, cc
from pyscf.dft.numint import eval_ao, eval_rho, eval_rho2
from pyscf.dft.gen_grid import Grids
from pyscf.pbc.tools.pyscf_ase import atoms_from_ase
from mldftdat.pyscf_utils import *
import numpy as np
from abc import ABC, abstractmethod, abstractproperty
from io import BytesIO
import psutil
from pyscf.scf.hf import get_jk

"""
ElectronAnalyzer classes for loading, storing, and analyzing DFT
calculations. For thorough documentation, see the lowmem_analyzers module,
which is equivalent but more efficient for systems larger than a couple atoms.
"""

def recursive_remove_none(obj):
    if type(obj) == dict:
        return {k: recursive_remove_none(v) for k, v in obj.items() if v is not None}
    else:
        return obj


class ElectronAnalyzer(ABC):

    calc_class = None
    calc_type = None

    def __init__(self, calc, require_converged=True, max_mem=None, grid_level = 3):
        # max_mem in MB
        if not isinstance(calc, self.calc_class):
            raise ValueError('Calculation must be instance of {}.'.format(self.calc_class))
        if calc.e_tot is None:
            raise ValueError('{} calculation must be complete.'.format(self.calc_type))
        if require_converged and not calc.converged:
            raise ValueError('{} calculation must be converged.'.format(self.calc_type))
        self.calc = calc
        self.mol = calc.mol.build()
        self.conv_tol = self.calc.conv_tol
        self.converged = calc.converged
        self.max_mem = max_mem
        self.grid_level = grid_level
        print('PRIOR TO POST PROCESS', psutil.virtual_memory().available // 1e6)
        self.post_process()
        print('FINISHED POST PROCESS', psutil.virtual_memory().available // 1e6)

    def as_dict(self):
        calc_props = {
            'conv_tol' : self.conv_tol,
            'converged' : self.converged,
            'e_tot' : self.e_tot,
            'mo_coeff' : self.mo_coeff,
            'mo_occ' : self.mo_occ,
        }
        data = {
            'coords' : self.grid.coords,
            'weights' : self.grid.weights,
            'ha_total' : self.ha_total,
            'fx_total' : self.fx_total,
            'ha_energy_density' : self.ha_energy_density,
            'fx_energy_density' : self.fx_energy_density,
            'xc_energy_density' : self.xc_energy_density,
            'ee_energy_density' : self.ee_energy_density,
            'rho_data' : self.rho_data,
            'tau_data' : self.tau_data
        }
        return {
            'mol' : gto.mole.pack(self.mol),
            'calc_type' : self.calc_type,
            'calc' : calc_props,
            'data' : data
        }

    def dump(self, fname):
        h5dict = recursive_remove_none(self.as_dict())
        lib.chkfile.dump(fname, 'analyzer', h5dict)

    @classmethod
    def from_dict(cls, analyzer_dict, max_mem = None):
        if analyzer_dict['calc_type'] != cls.calc_type:
            raise ValueError('Dict is from wrong type of calc, {} vs {}!'.format(
                                analyzer_dict['calc_type'], cls.calc_type))
        mol = mol_from_dict(analyzer_dict['mol'])
        calc = get_scf(analyzer_dict['calc_type'], mol, analyzer_dict['calc'])
        analyzer = cls(calc, require_converged = False, max_mem = max_mem)
        analyzer_dict['data'].pop('coords')
        analyzer_dict['data'].pop('weights')
        analyzer.__dict__.update(analyzer_dict['data'])
        return analyzer

    @classmethod
    def load(cls, fname, max_mem = None):
        analyzer_dict = lib.chkfile.load(fname, 'analyzer')
        return cls.from_dict(analyzer_dict, max_mem)

    def assign_num_chunks(self, ao_vals_shape, ao_vals_dtype):
        if self.max_mem == None:
            self.num_chunks = 1
            return self.num_chunks

        if ao_vals_dtype == np.float32:
            nbytes = 4
        elif ao_vals_dtype == np.float64:
            nbytes = 8
        else:
            raise ValueError('Wrong dtype for ao_vals')
        num_mbytes = nbytes * ao_vals_shape[0] * ao_vals_shape[1]**2 // 1000000
        self.num_chunks = int(num_mbytes // self.max_mem) + 1
        return self.num_chunks

    def post_process(self):
        # The child post process function must set up the RDMs
        self.grid = get_grid(self.mol)
        if self.grid_level != 3:
            self.grid.level = self.grid_level
            self.grid.kernel()

        self.e_tot = self.calc.e_tot
        self.mo_coeff = self.calc.mo_coeff
        self.mo_occ = self.calc.mo_occ
        ###self.mo_energy = self.calc.mo_energy
        self.ao_vals = get_ao_vals(self.mol, self.grid.coords)
        self.mo_vals = get_mo_vals(self.ao_vals, self.mo_coeff)

        self.assign_num_chunks(self.ao_vals.shape, self.ao_vals.dtype)
        print("NUMBER OF CHUNKS", self.calc_type, self.num_chunks, self.ao_vals.dtype, psutil.virtual_memory().available // 1e6)

        if self.num_chunks > 1:
            self.ao_vele_mat = get_vele_mat_generator(self.mol, self.grid.coords,
                                                self.num_chunks)
        else:
            self.ao_vele_mat = get_vele_mat(self.mol, self.grid.coords)
            print('AO VELE MAT', self.ao_vele_mat.nbytes, self.ao_vele_mat.shape)

        print("MEM NOW", psutil.virtual_memory().available // 1e6)

        self.rdm1 = None
        self.rdm2 = None
        self.ao_data = None
        self.rho_data = None
        self.tau_data = None

        self.ha_total = None
        self.fx_total = None
        self.ee_total = None
        self.ha_energy_density = None
        self.fx_energy_density = None
        self.xc_energy_density = None
        self.ee_energy_density = None

    def get_ao_rho_data(self):
        if self.rho_data is None or self.tau_data is None:
            ao_data, self.rho_data = get_mgga_data(
                                        self.mol, self.grid, self.rdm1)
            self.tau_data = get_tau_and_grad(self.mol, self.grid,
                                            self.rdm1, ao_data)
        return self.rho_data, self.tau_data

    def perform_full_analysis(self):
        self.get_ao_rho_data()
        self.get_ha_energy_density()
        self.get_ee_energy_density()


class RHFAnalyzer(ElectronAnalyzer):

    calc_class = scf.hf.RHF
    calc_type = 'RHF'

    def as_dict(self):
        analyzer_dict = super(RHFAnalyzer, self).as_dict()
        analyzer_dict['calc']['mo_energy'] = self.mo_energy
        return analyzer_dict

    def post_process(self):
        super(RHFAnalyzer, self).post_process()
        self.rdm1 = np.array(self.calc.make_rdm1())
        self.mo_energy = self.calc.mo_energy
        self.jmat, self.kmat = scf.hf.get_jk(self.mol, self.rdm1)
        self.ha_total, self.fx_total = get_hf_coul_ex_total2(self.rdm1,
                                                    self.jmat, self.kmat)
        if self.num_chunks > 1:
            self.mo_vele_mat = get_vele_mat_generator(self.mol, self.grid.coords,
                                                self.num_chunks, self.mo_coeff)
        else:
            self.mo_vele_mat = get_mo_vele_mat(self.ao_vele_mat, self.mo_coeff)
            print("MO VELE MAT", self.mo_vele_mat.nbytes, psutil.virtual_memory().available // 1e6)

    def get_ha_energy_density(self):
        if self.ha_energy_density is None:
            self.ha_energy_density = get_ha_energy_density2(
                                    self.mol, self.rdm1,
                                    self.ao_vele_mat, self.ao_vals
                                    )
        return self.ha_energy_density

    def get_fx_energy_density(self):
        if self.fx_energy_density is None:
            self.fx_energy_density = get_fx_energy_density2(
                                    self.mol, self.mo_occ,
                                    self.mo_vele_mat, self.mo_vals
                                    )
        return self.fx_energy_density

    def get_ee_energy_density(self):
        if self.ee_energy_density is None:
            self.ee_energy_density = self.get_ha_energy_density()\
                                     + self.get_fx_energy_density()
        return self.ee_energy_density

    def _get_rdm2(self):
        if self.rdm2 is None:
            self.eri_ao = self.mol.intor('int2e')
            self.rdm2 = make_rdm2_from_rdm1(self.rdm1)
        return self.rdm2


class UHFAnalyzer(ElectronAnalyzer):

    calc_class = scf.uhf.UHF
    calc_type = 'UHF'

    def as_dict(self):
        analyzer_dict = super(UHFAnalyzer, self).as_dict()
        analyzer_dict['calc']['mo_energy'] = self.mo_energy
        analyzer_dict['data']['fx_energy_density_u'] = self.fx_energy_density_u
        analyzer_dict['data']['fx_energy_density_d'] = self.fx_energy_density_d
        return analyzer_dict

    def post_process(self):
        super(UHFAnalyzer, self).post_process()
        self.rdm1 = np.array(self.calc.make_rdm1())
        self.mo_energy = self.calc.mo_energy
        self.fx_energy_density_u = None
        self.fx_energy_density_d = None
        self.jmat, self.kmat = scf.hf.get_jk(self.mol, self.rdm1)
        self.ha_total, self.fx_total = get_hf_coul_ex_total2(self.rdm1,
                                                    self.jmat, self.kmat)
        if self.num_chunks > 1:
            self.mo_vele_mat = [get_vele_mat_generator(self.mol, self.grid.coords,
                                                self.num_chunks, self.mo_coeff[0]),\
                                get_vele_mat_generator(self.mol, self.grid.coords,
                                                self.num_chunks, self.mo_coeff[1])]
        else:
            self.mo_vele_mat = get_mo_vele_mat(self.ao_vele_mat, self.mo_coeff)

    def get_ha_energy_density(self):
        if self.ha_energy_density is None:
            self.ha_energy_density = get_ha_energy_density2(
                                    self.mol, np.sum(self.rdm1, axis=0),
                                    self.ao_vele_mat, self.ao_vals
                                    )
        return self.ha_energy_density

    def get_fx_energy_density(self):
        if self.fx_energy_density is None:
            self.fx_energy_density_u = 0.5 * get_fx_energy_density2(
                                        self.mol, 2 * self.mo_occ[0],
                                        self.mo_vele_mat[0], self.mo_vals[0]
                                        )
            self.fx_energy_density_d = 0.5 * get_fx_energy_density2(
                                        self.mol, 2 * self.mo_occ[1],
                                        self.mo_vele_mat[1], self.mo_vals[1]
                                        )
            self.fx_energy_density = self.fx_energy_density_u + self.fx_energy_density_d
        return self.fx_energy_density

    def get_ee_energy_density(self):
        if self.ee_energy_density is None:
            self.ee_energy_density = self.get_ha_energy_density()\
                                     + self.get_fx_energy_density()
        return self.ee_energy_density

    def _get_rdm2(self):
        if self.rdm2 is None:
            self.eri_ao = self.mol.intor('int2e')
            self.rdm2 = make_rdm2_from_rdm1_unrestricted(self.rdm1)
        return self.rdm2


class RKSAnalyzer(RHFAnalyzer):

    def __init__(self, calc, require_converged=True, max_mem=None):
        if type(calc) != dft.rks.RKS:
            raise ValueError('Calculation must be RKS.')
        self.dft = calc
        hf = scf.RHF(self.dft.mol)
        hf.e_tot = self.dft.e_tot
        hf.mo_coeff = self.dft.mo_coeff
        hf.mo_occ = self.dft.mo_occ
        hf.mo_energy = self.dft.mo_energy
        hf.converged = self.dft.converged
        super(RKSAnalyzer, self).__init__(hf, require_converged, max_mem)


class UKSAnalyzer(UHFAnalyzer):

    def __init__(self, calc, require_converged=True, max_mem=None):
        if type(calc) != dft.uks.UKS:
            raise ValueError('Calculation must be UKS.')
        self.dft = calc
        hf = scf.UHF(self.dft.mol)
        hf.e_tot = self.dft.e_tot
        hf.mo_coeff = self.dft.mo_coeff
        hf.mo_occ = self.dft.mo_occ
        hf.mo_energy = self.dft.mo_energy
        hf.converged = self.dft.converged
        super(UKSAnalyzer, self).__init__(hf, require_converged, max_mem)
