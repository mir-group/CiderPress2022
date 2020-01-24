from pyscf import scf, dft, gto, ao2mo, df, lib, fci, cc
from pyscf.dft.numint import eval_ao, eval_rho
from pyscf.dft.gen_grid import Grids
from pyscf.pbc.tools.pyscf_ase import atoms_from_ase
from mldftdat.pyscf_utils import *
import numpy as np
from abc import ABC, abstractmethod, abstractproperty


class ElectronAnalyzer(ABC):

    def __init__(self, calc, require_converged=True, max_mem=None):
        # max_mem in MB
        if type(calc) != self.calc_class:
            raise ValueError('Calculation must be type {}'.format(self.calc_class))
        if calc.e_tot is None:
            raise ValueError('{} calculation must be complete.'.format(self.calc_type))
        if require_converged and not calc.converged:
            raise ValueError('{} calculation must be converged.'.format(self.calc_type))
        self.calc = calc
        self.mol = calc.mol
        self.converged = calc.converged
        self.max_mem = max_mem
        self.post_process()

    @abstractproperty
    def calc_class(self):
        return None

    @abstractproperty
    def calc_type(self):
        return None

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
        self.rdm1 = self.calc.make_rdm1()
        self.grid = get_grid(self.mol)
        self.jmat, self.kmat = scf.hf.get_jk(self.mol, self.rdm1)
        self.eri_ao = self.mol.intor('int2e')

        self.e_tot = self.calc.e_tot
        self.mo_coeff = self.calc.mo_coeff
        self.mo_occ = self.calc.mo_occ
        ###self.mo_energy = self.calc.mo_energy
        self.ao_vals = get_ao_vals(self.mol, self.grid.coords)
        self.mo_vals = get_mo_vals(self.ao_vals, self.mo_coeff)

        self.assign_num_chunks(self.ao_vals.shape, self.ao_vals.dtype)
        print("NUMBER OF CHUNKS", self.num_chunks, self.ao_vals.dtype)

        if self.num_chunks > 1:
            self.ao_vele_mat = get_vele_mat_generator(self.mol, self.grid.coords,
                                                self.num_chunks, self.ao_vals)
        else:
            self.ao_vele_mat = get_vele_mat(self.mol, self.grid.coords)

        self.ha_total = np.sum(self.jmat * self.rdm1) / 2
        self.fx_total = np.sum(self.kmat * self.rdm1) / 2

        self.rdm2 = None
        self.ha_energy_density = None
        self.fx_energy_density = None
        self.xc_energy_density = None
        self.ee_energy_density = None


class RHFAnalyzer(ElectronAnalyzer):

    @property
    def calc_class(self):
        return scf.hf.RHF

    @property
    def calc_type(self):
        return 'RHF'

    def post_process(self):
        super(RHFAnalyzer, self).post_process()
        self.mo_energy = self.calc.mo_energy
        if self.num_chunks > 1:
            self.mo_vele_mat = get_vele_mat_generator(self.mol, self.grid.coords,
                                                self.num_chunks, self.mo_vals,
                                                self.mo_coeff)
        else:
            self.mo_vele_mat = get_mo_vele_mat(self.ao_vele_mat, self.mo_coeff)

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
            self.rdm2 = make_rdm2_from_rdm1(self.rdm1)
        return self.rdm2

    def _get_ee_energy_density_slow(self):
        rdm2 = self._get_rdm2()
        return get_ee_energy_density2(self.mol, self.rdm2,
                                        self.ao_vele_mat, self.ao_vals)


class UHFAnalyzer(ElectronAnalyzer):

    @property
    def calc_class(self):
        return scf.uhf.UHF

    @property
    def calc_type(self):
        return 'UHF'

    def post_process(self):
        super(UHFAnalyzer, self).post_process()
        self.mo_energy = self.calc.mo_energy
        if self.num_chunks > 1:
            self.mo_vele_mat = [get_vele_mat_generator(self.mol, self.grid.coords,
                                                self.num_chunks, self.mo_vals[0],
                                                self.mo_coeff[0]),\
                                get_vele_mat_generator(self.mol, self.grid.coords,
                                                self.num_chunks, self.mo_vals[1],
                                                self.mo_coeff[1])]
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
            self.rdm2 = make_rdm2_from_rdm1_unrestricted(self.rdm1)
        return self.rdm2

    def _get_ee_energy_density_slow(self):
        rdm2 = self._get_rdm2()
        euu = get_ee_energy_density2(self.mol, self.rdm2[0],
                                    self.ao_vele_mat, self.ao_vals)
        eud = get_ee_energy_density2(self.mol, self.rdm2[1],
                                    self.ao_vele_mat, self.ao_vals)
        edd = get_ee_energy_density2(self.mol, self.rdm2[2],
                                    self.ao_vele_mat, self.ao_vals)
        return euu + 2 * eud + edd


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


class CCSDAnalyzer(ElectronAnalyzer):

    @property
    def calc_class(self):
        return cc.ccsd.CCSD

    @property
    def calc_type(self):
        return 'CCSD'

    def post_process(self):
        super(CCSDAnalyzer, self).post_process()
        self.mo_rdm1 = self.calc.make_rdm1()
        self.mo_rdm2 = self.calc.make_rdm2()

        self.ao_rdm1 = transform_basis_1e(self.mo_rdm1, self.mo_coeff.transpose())
        self.ao_rdm2 = transform_basis_2e(self.mo_rdm2, self.mo_coeff.transpose())
        self.eri_mo = transform_basis_2e(self.eri_ao, self.mo_coeff)
        self.rdm1 = self.ao_rdm1
        self.rdm2 = self.ao_rdm2

    def get_ha_energy_density(self):
        if self.ha_energy_density is None:
            self.ha_energy_density = get_ha_energy_density2(
                                    self.mol, self.ao_rdm1,
                                    self.ao_vele_mat, self.ao_vals
                                    )
        return self.ha_energy_density

    def get_ee_energy_density(self):
        if self.ee_energy_density is None:
            self.ee_energy_density = get_ee_energy_density2(
                                    self.mol, self.ao_rdm2,
                                    self.ao_vele_mat, self.ao_vals)
        return self.ee_energy_density


class UCCSDAnalyzer(ElectronAnalyzer):

    @property
    def calc_class(self):
        return cc.uccsd.UCCSD

    @property
    def calc_type(self):
        return 'UCCSD'

    def post_process(self):
        super(UCCSDAnalyzer, self).post_process()
        self.mo_rdm1 = self.calc.make_rdm1()
        self.mo_rdm2 = self.calc.make_rdm2()

        # These are all three-tuples
        trans_mo_coeff = np.transpose(self.mo_coeff, axes=(0,2,1))
        self.ao_rdm1 = transform_basis_1e(self.mo_rdm1, trans_mo_coeff)
        self.ao_rdm2 = transform_basis_2e(self.mo_rdm2, trans_mo_coeff)
        eri_ao_lst = [self.eri_ao] * 3
        self.eri_mo = transform_basis_2e(eri_ao_lst, self.mo_coeff)
        self.rdm1 = self.ao_rdm1
        self.rdm2 = self.ao_rdm2

        self.ha_energy_density = None
        self.ee_energy_density = None
        self.xc_energy_density = None

    def get_ha_energy_density(self):
        if self.ha_energy_density is None:
            self.ha_energy_density = get_ha_energy_density2(
                                    self.mol, np.sum(self.ao_rdm1, axis=0),
                                    self.ao_vele_mat, self.ao_vals
                                    )
        return self.ha_energy_density

    def get_ee_energy_density(self):
        if self.ee_energy_density is None:
            self.ee_energy_density_uu = get_ee_energy_density2(
                                    self.mol, self.ao_rdm2[0],
                                    self.ao_vele_mat, self.ao_vals)
            self.ee_energy_density_ud = get_ee_energy_density2(
                                    self.mol, self.ao_rdm2[1],
                                    self.ao_vele_mat, self.ao_vals)
            self.ee_energy_density_dd = get_ee_energy_density2(
                                    self.mol, self.ao_rdm2[2],
                                    self.ao_vele_mat, self.ao_vals)
            self.ee_energy_density = self.ee_energy_density_uu\
                                    + 2 * self.ee_energy_density_ud\
                                    + self.ee_energy_density_dd
        return self.ee_energy_density
