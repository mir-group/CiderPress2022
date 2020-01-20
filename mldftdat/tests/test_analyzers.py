from pyscf import scf, gto
from nose import SkipTest
from nose.tools import nottest
from nose.plugins.skip import Skip
from numpy.testing import assert_almost_equal, assert_equal

from mldftdat.pyscf_utils import get_hf_coul_ex_total, get_hf_coul_ex_total_unrestricted,\
                                run_scf, integrate_on_grid
from mldftdat.analyzers import RHFAnalyzer, UHFAnalyzer
import numpy as np

class TestRHFAnalyzer():

    @classmethod
    def setup_class(cls):
        cls.mol = gto.Mole(atom='H 0 0 0; F 0 0 1.1', basis = '631g')
        cls.mol.build()
        cls.rhf = run_scf(cls.mol, 'RHF')
        cls.analyzer = RHFAnalyzer(cls.rhf)
        cls.ha_tot_ref, cls.fx_tot_ref = get_hf_coul_ex_total(cls.mol, cls.rhf)

    def test_post_process(self):
        # This is tested in the rest of the module
        assert_almost_equal(self.ha_tot_ref + self.fx_tot_ref, self.mol.energy_elec()[1])

    def test_get_ha_energy_density(self):
        ha_density = self.analyzer.get_ha_energy_density()
        ha_tot = integrate_on_grid(ha_density, self.analyzer.grid.weights)
        assert_almost_equal(ha_tot, self.ha_tot_ref, 5)

    def test_get_fx_energy_density(self):
        fx_density = self.analyzer.get_fx_energy_density()
        fx_tot = integrate_on_grid(fx_density, self.analyzer.grid.weights)
        assert_almost_equal(fx_tot, self.fx_tot_ref, 5)

    def test_get_ee_energy_density(self):
        ee_density = self.analyzer.get_ee_energy_density()
        ee_tot = integrate_on_grid(ee_density, self.analyzer.grid.weights)
        assert_almost_equal(ee_tot, self.mol.energy_elec()[1], 5)

    def test__get_rdm2(self):
        # Tested by next test
        pass

    def test__get_ee_energy_density_slow(self):
        ee_density = self.analyzer._get_ee_energy_density_slow()
        ee_tot = integrate_on_grid(ee_density, self.analyzer.grid.weights)
        assert_almost_equal(ee_tot, self.mol.energy_elec()[1], 5)


class TestUHFAnalyzer():

    @classmethod
    def setup_class(cls):
        cls.mol = gto.Mole(atom='N 0 0 0; O 0 0 1.15', basis = '631g', spin = 1)
        cls.mol.build()
        cls.uhf = run_scf(cls.mol, 'UHF')
        cls.analyzer = UHFAnalyzer(cls.uhf)
        cls.ha_tot_ref, cls.fx_tot_ref = get_hf_coul_ex_total_unrestricted(cls.mol, cls.uhf)

    def test_post_process(self):
        # This is tested in the rest of the module
        assert_almost_equal(self.ha_tot_ref + self.fx_tot_ref, self.mol.energy_elec()[1])

    def test_get_ha_energy_density(self):
        ha_density = self.analyzer.get_ha_energy_density()
        ha_tot = integrate_on_grid(ha_density, self.analyzer.grid.weights)
        assert_almost_equal(ha_tot, self.ha_tot_ref, 5)

    def test_get_fx_energy_density(self):
        fx_density = self.analyzer.get_fx_energy_density()
        fx_tot = integrate_on_grid(fx_density, self.analyzer.grid.weights)
        assert_almost_equal(fx_tot, self.fx_tot_ref, 5)

    def test_get_ee_energy_density(self):
        ee_density = self.analyzer.get_ee_energy_density()
        ee_tot = integrate_on_grid(ee_density, self.analyzer.grid.weights)
        assert_almost_equal(ee_tot, self.mol.energy_elec()[1], 5)

    def test__get_rdm2(self):
        # Tested by next test
        pass

    def test__get_ee_energy_density_slow(self):
        ee_density = self.analyzer._get_ee_energy_density_slow()
        ee_tot = integrate_on_grid(ee_density, self.analyzer.grid.weights)
        assert_almost_equal(ee_tot, self.mol.energy_elec()[1], 5)
