from pyscf import scf, gto, lib
from nose import SkipTest
from nose.tools import nottest
from nose.plugins.skip import Skip
from numpy.testing import assert_almost_equal, assert_equal

from mldftdat.pyscf_utils import get_hf_coul_ex_total, get_hf_coul_ex_total_unrestricted,\
                                run_scf, run_cc, integrate_on_grid, get_ccsd_ee_total,\
                                transform_basis_2e, transform_basis_1e, get_ccsd_ee
from mldftdat.loc_analyzers import RHFAnalyzer, UHFAnalyzer
import numpy as np
import numbers
import os

TMP_TEST = 'test_files/tmp'


class TestRHFAnalyzer():

    @classmethod
    def setup_class(cls):
        cls.mol = gto.Mole(atom='H 0 0 0; F 0 0 1.1', basis='def2-svp')
        cls.mol.build()
        cls.rhf = run_scf(cls.mol, 'RHF')
        cls.analyzer = RHFAnalyzer(cls.rhf)
        cls.ha_tot_ref, cls.fx_tot_ref = get_hf_coul_ex_total(cls.mol, cls.rhf)

    def test_get_loc_fx_energy_density(self):
        fx_density = self.analyzer.get_loc_fx_energy_density()
        fx_tot = integrate_on_grid(fx_density, self.analyzer.grid.weights)
        assert_almost_equal(fx_tot, self.fx_tot_ref, 4)
        assert_almost_equal(self.analyzer.fx_total, self.fx_tot_ref)


class TestUHFAnalyzer():

    @classmethod
    def setup_class(cls):
        cls.mol = gto.Mole(atom='N 0 0 0; O 0 0 1.15', basis='def2-svp', spin=1)
        cls.mol.build()
        cls.uhf = run_scf(cls.mol, 'UHF')
        cls.analyzer = UHFAnalyzer(cls.uhf)
        cls.ha_tot_ref, cls.fx_tot_ref = get_hf_coul_ex_total_unrestricted(cls.mol, cls.uhf)

    def test_get_loc_fx_energy_density(self):
        fx_density = self.analyzer.get_loc_fx_energy_density()
        fx_tot = integrate_on_grid(fx_density, self.analyzer.grid.weights)
        # Precision is about 2e-4 in practice, a bit higher than 4-digit thresh
        assert_almost_equal(fx_tot, self.fx_tot_ref, 3)
        assert_almost_equal(self.analyzer.fx_total, self.fx_tot_ref)
