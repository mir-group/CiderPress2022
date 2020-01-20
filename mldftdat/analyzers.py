from pyscf import scf, dft, gto, ao2mo, df, lib, fci, cc
from pyscf.dft.numint import eval_ao, eval_rho
from pyscf.dft.gen_grid import Grids
from pyscf.pbc.tools.pyscf_ase import atoms_from_ase
from mldftdat.pyscf_utils import *
import numpy as np

class RHFAnalyzer():

    def __init__(self, calc):
        if type(calc) != scf.hf.RHF:
            raise ValueError('Calculation must be RHF.')
        if calc.mo_coeff is None:
            raise ValueError('Calculation must be complete before initializing.')
        self.calc = calc
        self.mol = calc.mol
        self.post_process()

    def post_process(self):
        self.rdm1 = self.calc.make_rdm1()
        self.grid = get_grid(self.mol)
        self.jmat, self.kmat = scf.hf.get_jk(self.mol, self.rdm1)
        self.eri_ao = self.mol.intor('int2e')

        self.mo_coeff = self.calc.mo_coeff
        self.mo_occ = self.calc.mo_occ
        self.mo_energy = self.calc.mo_energy
        self.ao_vals = get_ao_vals(self.mol, self.grid.coords)
        self.mo_vals = get_mo_vals(self.ao_vals, self.mo_coeff)
        self.ao_vele_mat = get_vele_mat(self.mol, self.grid.coords)
        self.mo_vele_mat = get_mo_vele_mat(self.ao_vele_mat, self.mo_coeff)

        self.ha_total = np.sum(self.jmat * self.rdm1) / 2
        self.fx_total = np.sum(self.kmat * self.rdm1) / 2

        self.rdm2 = None
        self.ha_energy_density = None
        self.fx_energy_density = None
        self.ee_energy_density = None

    def get_ha_energy_density(self):
        if self.ha_energy_density is None:
            self.ha_energy_density = get_ha_energy_density(
                                    self.mol, self.rdm1,
                                    self.ao_vele_mat, self.ao_vals
                                    )
        return self.ha_energy_density

    def get_fx_energy_density(self):
        if self.fx_energy_density is None:
            self.fx_energy_density = get_fx_energy_density(
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
        return get_ee_energy_density(self.mol, self.rdm2,
                                        self.ao_vele_mat, self.ao_vals)



class UHFAnalyzer():

    def __init__(self, calc):
        if type(calc) != scf.uhf.UHF:
            raise ValueError('Calculation must be UHF.')
        if calc.mo_coeff is None:
            raise ValueError('Calculation must be complete before initializing.')
        self.calc = calc
        self.mol = calc.mol
        self.post_process()

    def post_process(self):
        self.rdm1 = self.calc.make_rdm1()
        self.grid = get_grid(self.mol)
        self.jmat, self.kmat = scf.hf.get_jk(self.mol, self.rdm1)
        self.eri_ao = self.mol.intor('int2e')

        self.mo_coeff = self.calc.mo_coeff
        self.mo_occ = self.calc.mo_occ
        self.mo_energy = self.calc.mo_energy
        self.ao_vals = get_ao_vals(self.mol, self.grid.coords)
        self.mo_vals = get_mo_vals(self.ao_vals, self.mo_coeff)
        self.ao_vele_mat = get_vele_mat(self.mol, self.grid.coords)
        self.mo_vele_mat = get_mo_vele_mat_unrestricted(
                            self.ao_vele_mat, self.mo_coeff)

        self.ha_total = np.sum(self.jmat * self.rdm1) / 2
        self.fx_total = np.sum(self.kmat * self.rdm1) / 2

        self.rdm2 = None
        self.ha_energy_density = None
        self.fx_energy_density = None
        self.ee_energy_density = None

    def get_ha_energy_density(self):
        if self.ha_energy_density is None:
            self.ha_energy_density = get_ha_energy_density(
                                    self.mol, np.sum(self.rdm1, axis=0),
                                    self.ao_vele_mat, self.ao_vals
                                    )
        return self.ha_energy_density

    def get_fx_energy_density(self):
        if self.fx_energy_density is None:
            self.fx_energy_density_u = 0.5 * get_fx_energy_density(
                                        self.mol, 2 * self.mo_occ[0],
                                        self.mo_vele_mat[0], self.mo_vals[0]
                                        )
            self.fx_energy_density_d = 0.5 * get_fx_energy_density(
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
        euu = get_ee_energy_density(self.mol, self.rdm2[0],
                                    self.ao_vele_mat, self.ao_vals)
        eud = get_ee_energy_density(self.mol, self.rdm2[1],
                                    self.ao_vele_mat, self.ao_vals)
        edd = get_ee_energy_density(self.mol, self.rdm2[2],
                                    self.ao_vele_mat, self.ao_vals)
        return euu + 2 * eud + edd
