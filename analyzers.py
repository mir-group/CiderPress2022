from pyscf import scf, dft, gto, ao2mo, df, lib, fci, cc
from pyscf.dft.numint import eval_ao, eval_rho
from pyscf.dft.gen_grid import Grids
from pyscf.pbc.tools.pyscf_ase import atoms_from_ase
import numpy as np

class RHFAnalyzer():

    def __init__(self, calc):
        self.calc = calc
        self.mol = calc.mol
        if self.calc.mo_coeff is None:
            raise ValueError('Calculation must be complete before initializing')
        self.post_process()

    def post_process(self):
        self.rdm1 = self.calc.make_rdm1()
        self.grid = get_grid(self.mol)
        self.jmat, self.kmat = scf.hf.get_jk(self.mol, self.rdm1)
        self.eri_ao = self.mol.intor('int2e')

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
        return self.get_ee_energy_density(self.mol, self.rdm2,
                                        self.ao_vele_mat, self.ao_vals)


