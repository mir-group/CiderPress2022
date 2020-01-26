from pyscf import scf, gto, lib
from nose import SkipTest
from nose.tools import nottest
from nose.plugins.skip import Skip
from numpy.testing import assert_almost_equal, assert_equal

from mldftdat.pyscf_utils import get_hf_coul_ex_total, get_hf_coul_ex_total_unrestricted,\
                                run_scf, run_cc, integrate_on_grid, get_ccsd_ee_total,\
                                transform_basis_2e, transform_basis_1e, get_ccsd_ee
from mldftdat.analyzers import RHFAnalyzer, UHFAnalyzer, CCSDAnalyzer, UCCSDAnalyzer,\
                                RKSAnalyzer, UKSAnalyzer
import numpy as np
import numbers
import os

TMP_TEST = 'test_files/tmp'

class TestRHFAnalyzer():

    @classmethod
    def setup_class(cls):
        cls.mol = gto.Mole(atom='H 0 0 0; F 0 0 1.1', basis = 'sto-3g')
        cls.mol.build()
        cls.rhf = run_scf(cls.mol, 'RHF')
        cls.analyzer = RHFAnalyzer(cls.rhf)
        cls.ha_tot_ref, cls.fx_tot_ref = get_hf_coul_ex_total(cls.mol, cls.rhf)

    def test_post_process(self):
        # This is tested in the rest of the module
        assert_almost_equal(self.ha_tot_ref + self.fx_tot_ref, self.rhf.energy_elec()[1])

    def test_get_ha_energy_density(self):
        ha_density = self.analyzer.get_ha_energy_density()
        ha_tot = integrate_on_grid(ha_density, self.analyzer.grid.weights)
        assert_almost_equal(ha_tot, self.ha_tot_ref, 5)
        assert_almost_equal(self.analyzer.ha_total, self.ha_tot_ref)

    def test_get_fx_energy_density(self):
        fx_density = self.analyzer.get_fx_energy_density()
        fx_tot = integrate_on_grid(fx_density, self.analyzer.grid.weights)
        assert_almost_equal(fx_tot, self.fx_tot_ref, 5)
        assert_almost_equal(self.analyzer.fx_total, self.fx_tot_ref)

    def test_get_ee_energy_density(self):
        ee_density = self.analyzer.get_ee_energy_density()
        ee_tot = integrate_on_grid(ee_density, self.analyzer.grid.weights)
        assert_almost_equal(ee_tot, self.rhf.energy_elec()[1], 5)

    def test__get_rdm2(self):
        ee_tot = get_ccsd_ee(self.analyzer._get_rdm2(), self.analyzer.eri_ao)
        assert_almost_equal(ee_tot, self.rhf.energy_elec()[1])

    def test__get_ee_energy_density_slow(self):
        ee_density = self.analyzer._get_ee_energy_density_slow()
        ee_tot = integrate_on_grid(ee_density, self.analyzer.grid.weights)
        assert_almost_equal(ee_tot, self.rhf.energy_elec()[1], 5)

    def test_as_dict_from_dict(self):
        analyzer1 = RHFAnalyzer(self.rhf)
        dict1 = analyzer1.as_dict()
        analyzer1.get_ha_energy_density()
        analyzer1.get_fx_energy_density()
        analyzer1.get_ee_energy_density()
        dict2 = analyzer1.as_dict()
        analyzer2 = RHFAnalyzer.from_dict(dict1)
        analyzer2.get_ha_energy_density()
        analyzer2.get_fx_energy_density()
        analyzer2.get_ee_energy_density()
        assert_almost_equal(analyzer1.ha_energy_density, analyzer2.ha_energy_density)
        assert_almost_equal(analyzer1.fx_energy_density, analyzer2.fx_energy_density)
        assert_almost_equal(analyzer1.ee_energy_density, analyzer2.ee_energy_density)
        analyzer3 = RHFAnalyzer.from_dict(dict2)
        assert_almost_equal(analyzer1.ha_energy_density, analyzer3.ha_energy_density)
        assert_almost_equal(analyzer1.fx_energy_density, analyzer3.fx_energy_density)
        assert_almost_equal(analyzer1.ee_energy_density, analyzer3.ee_energy_density)

    def test_dump_load(self):
        analyzer1 = RHFAnalyzer(self.rhf)
        analyzer1.get_ee_energy_density()
        analyzer1.dump(TMP_TEST)
        analyzer2 = RHFAnalyzer.load(TMP_TEST)
        for key in analyzer1.__dict__:
            if isinstance(analyzer1.__getattribute__(key), numbers.Number)\
                or isinstance(analyzer1.__getattribute__(key), np.ndarray):
                assert_equal(analyzer1.__getattribute__(key),
                    analyzer2.__getattribute__(key))
        data_dict = lib.chkfile.load(TMP_TEST, 'analyzer/data')
        assert_equal(analyzer1.grid.coords, data_dict['coords'])
        assert_equal(analyzer1.grid.weights, data_dict['weights'])
        assert_equal(analyzer1.ha_energy_density, data_dict['ha_energy_density'])
        assert_equal(analyzer1.ee_energy_density, data_dict['ee_energy_density'])
        os.remove(TMP_TEST)


class TestRHFAnalyzerChunks(TestRHFAnalyzer):

    @classmethod
    def setup_class(cls):
        cls.mol = gto.Mole(atom='H 0 0 0; F 0 0 1.1', basis = 'sto-3g')
        cls.mol.build()
        cls.rhf = run_scf(cls.mol, 'RHF')
        cls.analyzer = RHFAnalyzer(cls.rhf, max_mem=5)
        cls.ha_tot_ref, cls.fx_tot_ref = get_hf_coul_ex_total(cls.mol, cls.rhf)


class TestRKSAnalyzer():

    @classmethod
    def setup_class(cls):
        cls.mol = gto.Mole(atom='H 0 0 0; F 0 0 1.1', basis = 'sto-3g')
        cls.mol.build()
        cls.rks = run_scf(cls.mol, 'RHF')
        cls.analyzer = RKSAnalyzer(cls.rks)
        cls.rhf = cls.analyzer.calc
        cls.ha_tot_ref, cls.fx_tot_ref = get_hf_coul_ex_total(cls.mol, cls.rhf)


class TestUHFAnalyzer():

    @classmethod
    def setup_class(cls):
        cls.mol = gto.Mole(atom='N 0 0 0; O 0 0 1.15', basis = 'sto-3g', spin = 1)
        cls.mol.build()
        cls.uhf = run_scf(cls.mol, 'UHF')
        cls.analyzer = UHFAnalyzer(cls.uhf)
        cls.ha_tot_ref, cls.fx_tot_ref = get_hf_coul_ex_total_unrestricted(cls.mol, cls.uhf)

    def test_post_process(self):
        # This is tested in the rest of the module
        assert_almost_equal(self.ha_tot_ref + self.fx_tot_ref, self.uhf.energy_elec()[1])

    def test_get_ha_energy_density(self):
        ha_density = self.analyzer.get_ha_energy_density()
        ha_tot = integrate_on_grid(ha_density, self.analyzer.grid.weights)
        assert_almost_equal(ha_tot, self.ha_tot_ref, 5)
        assert_almost_equal(self.analyzer.ha_total, self.ha_tot_ref)

    def test_get_fx_energy_density(self):
        fx_density = self.analyzer.get_fx_energy_density()
        fx_tot = integrate_on_grid(fx_density, self.analyzer.grid.weights)
        assert_almost_equal(fx_tot, self.fx_tot_ref, 5)
        assert_almost_equal(self.analyzer.fx_total, self.fx_tot_ref)

    def test_get_ee_energy_density(self):
        ee_density = self.analyzer.get_ee_energy_density()
        ee_tot = integrate_on_grid(ee_density, self.analyzer.grid.weights)
        assert_almost_equal(ee_tot, self.uhf.energy_elec()[1], 5)

    def test__get_rdm2(self):
        ee_tot = get_ccsd_ee(self.analyzer._get_rdm2(), self.analyzer.eri_ao)
        assert_almost_equal(ee_tot, self.uhf.energy_elec()[1])

    def test__get_ee_energy_density_slow(self):
        ee_density = self.analyzer._get_ee_energy_density_slow()
        ee_tot = integrate_on_grid(ee_density, self.analyzer.grid.weights)
        assert_almost_equal(ee_tot, self.uhf.energy_elec()[1], 5)

    def test_as_dict_from_dict(self):
        analyzer1 = UHFAnalyzer(self.uhf)
        dict1 = analyzer1.as_dict()
        analyzer1.get_ha_energy_density()
        analyzer1.get_fx_energy_density()
        analyzer1.get_ee_energy_density()
        dict2 = analyzer1.as_dict()
        analyzer2 = UHFAnalyzer.from_dict(dict1)
        analyzer2.get_ha_energy_density()
        analyzer2.get_fx_energy_density()
        analyzer2.get_ee_energy_density()
        assert_almost_equal(analyzer1.ha_energy_density, analyzer2.ha_energy_density)
        assert_almost_equal(analyzer1.fx_energy_density, analyzer2.fx_energy_density)
        assert_almost_equal(analyzer1.fx_energy_density_u, analyzer2.fx_energy_density_u)
        assert_almost_equal(analyzer1.fx_energy_density_d, analyzer2.fx_energy_density_d)
        assert_almost_equal(analyzer1.ee_energy_density, analyzer2.ee_energy_density)
        analyzer3 = UHFAnalyzer.from_dict(dict2)
        assert_almost_equal(analyzer1.ha_energy_density, analyzer3.ha_energy_density)
        assert_almost_equal(analyzer1.fx_energy_density, analyzer3.fx_energy_density)
        assert_almost_equal(analyzer1.fx_energy_density_u, analyzer3.fx_energy_density_u)
        assert_almost_equal(analyzer1.fx_energy_density_d, analyzer3.fx_energy_density_d)
        assert_almost_equal(analyzer1.ee_energy_density, analyzer3.ee_energy_density)

    def test_dump_load(self):
        analyzer1 = UHFAnalyzer(self.uhf, require_converged=False)
        analyzer1.get_ee_energy_density()
        analyzer1.dump(TMP_TEST)
        analyzer2 = UHFAnalyzer.load(TMP_TEST)
        for key in analyzer1.__dict__:
            if isinstance(analyzer1.__getattribute__(key), numbers.Number)\
                or isinstance(analyzer1.__getattribute__(key), np.ndarray):
                assert_equal(analyzer1.__getattribute__(key),
                    analyzer2.__getattribute__(key))
        data_dict = lib.chkfile.load(TMP_TEST, 'analyzer/data')
        assert_equal(analyzer1.grid.coords, data_dict['coords'])
        assert_equal(analyzer1.grid.weights, data_dict['weights'])
        assert_equal(analyzer1.ha_energy_density, data_dict['ha_energy_density'])
        assert_equal(analyzer1.ee_energy_density, data_dict['ee_energy_density'])
        os.remove(TMP_TEST)


class TestUHFAnalyzerChunks(TestUHFAnalyzer):

    @classmethod
    def setup_class(cls):
        cls.mol = gto.Mole(atom='N 0 0 0; O 0 0 1.15', basis = 'sto-3g', spin = 1)
        cls.mol.build()
        cls.uhf = run_scf(cls.mol, 'UHF')
        cls.analyzer = UHFAnalyzer(cls.uhf, max_mem=5)
        cls.ha_tot_ref, cls.fx_tot_ref = get_hf_coul_ex_total_unrestricted(cls.mol, cls.uhf)


class TestUKSAnalyzer(TestUHFAnalyzer):

    @classmethod
    def setup_class(cls):
        cls.mol = gto.Mole(atom='N 0 0 0; O 0 0 1.15', basis = 'sto-3g', spin = 1)
        cls.mol.build()
        cls.uks = run_scf(cls.mol, 'UKS')
        cls.analyzer = UKSAnalyzer(cls.uks, require_converged=False)
        cls.uhf = cls.analyzer.calc
        cls.ha_tot_ref, cls.fx_tot_ref = get_hf_coul_ex_total_unrestricted(cls.mol, cls.uhf)

    def test_as_dict_from_dict(self):
        analyzer1 = UHFAnalyzer(self.uhf, require_converged=False)
        dict1 = analyzer1.as_dict()
        analyzer1.get_ha_energy_density()
        analyzer1.get_fx_energy_density()
        analyzer1.get_ee_energy_density()
        dict2 = analyzer1.as_dict()
        analyzer2 = UHFAnalyzer.from_dict(dict1)
        analyzer2.get_ha_energy_density()
        analyzer2.get_fx_energy_density()
        analyzer2.get_ee_energy_density()
        assert_almost_equal(analyzer1.ha_energy_density, analyzer2.ha_energy_density)
        assert_almost_equal(analyzer1.fx_energy_density, analyzer2.fx_energy_density)
        assert_almost_equal(analyzer1.fx_energy_density_u, analyzer2.fx_energy_density_u)
        assert_almost_equal(analyzer1.fx_energy_density_d, analyzer2.fx_energy_density_d)
        assert_almost_equal(analyzer1.ee_energy_density, analyzer2.ee_energy_density)
        analyzer3 = UHFAnalyzer.from_dict(dict2)
        assert_almost_equal(analyzer1.ha_energy_density, analyzer3.ha_energy_density)
        assert_almost_equal(analyzer1.fx_energy_density, analyzer3.fx_energy_density)
        assert_almost_equal(analyzer1.fx_energy_density_u, analyzer3.fx_energy_density_u)
        assert_almost_equal(analyzer1.fx_energy_density_d, analyzer3.fx_energy_density_d)
        assert_almost_equal(analyzer1.ee_energy_density, analyzer3.ee_energy_density)


class TestCCSDAnalyzer():

    @classmethod
    def setup_class(cls):
        cls.mol = gto.Mole(atom='He 0 0 0', basis = 'cc-pvdz')
        cls.mol.build()
        cls.hf = run_scf(cls.mol, 'RHF')
        cls.cc = run_cc(cls.hf)
        cls.analyzer = CCSDAnalyzer(cls.cc)
        cls.ee_tot_ref = get_ccsd_ee_total(cls.mol, cls.cc, cls.hf)

    def test_post_process(self):
        pass

    def test_get_ha_energy_density(self):
        eri = self.mol.intor('int2e')
        eri = transform_basis_2e(eri, self.hf.mo_coeff)
        rdm1 = self.cc.make_rdm1()
        ha_tot_ref = np.sum(np.sum(eri * rdm1, axis=(2,3)) * rdm1) / 2
        ha_density = self.analyzer.get_ha_energy_density()
        ha_tot = integrate_on_grid(ha_density, self.analyzer.grid.weights)
        assert_almost_equal(ha_tot, ha_tot_ref, 5)
        assert_almost_equal(self.analyzer.ha_total, ha_tot_ref)

    def test_get_ee_energy_density(self):
        ee_density = self.analyzer.get_ee_energy_density()
        ee_tot = integrate_on_grid(ee_density, self.analyzer.grid.weights)
        assert_almost_equal(ee_tot, self.ee_tot_ref)
        # ee repulsion should be similar to HF case
        # Note this case may not pass for all systems, but hsould pass for He and Li
        assert_almost_equal(ee_tot, self.hf.energy_elec()[1], 1)
        assert_almost_equal(self.analyzer.ee_total, self.ee_tot_ref)

    def test_as_dict_from_dict(self):
        analyzer1 = CCSDAnalyzer(self.cc)
        dict1 = analyzer1.as_dict()
        analyzer1.get_ha_energy_density()
        analyzer1.get_ee_energy_density()
        dict2 = analyzer1.as_dict()
        analyzer2 = CCSDAnalyzer.from_dict(dict1)
        analyzer2.get_ha_energy_density()
        analyzer2.get_ee_energy_density()
        assert_almost_equal(analyzer1.ha_energy_density, analyzer2.ha_energy_density)
        assert_almost_equal(analyzer1.ee_energy_density, analyzer2.ee_energy_density)
        analyzer3 = CCSDAnalyzer.from_dict(dict2)
        assert_almost_equal(analyzer1.ha_energy_density, analyzer3.ha_energy_density)
        assert_almost_equal(analyzer1.ee_energy_density, analyzer3.ee_energy_density)

    def test_dump_load(self):
        analyzer1 = CCSDAnalyzer(self.cc)
        analyzer1.get_ee_energy_density()
        analyzer1.get_ha_energy_density()
        analyzer1.dump(TMP_TEST)
        analyzer2 = CCSDAnalyzer.load(TMP_TEST)
        for key in analyzer1.__dict__:
            if isinstance(analyzer1.__getattribute__(key), numbers.Number)\
                or isinstance(analyzer1.__getattribute__(key), np.ndarray):
                assert_equal(analyzer1.__getattribute__(key),
                    analyzer2.__getattribute__(key))
        data_dict = lib.chkfile.load(TMP_TEST, 'analyzer/data')
        assert_equal(analyzer1.grid.coords, data_dict['coords'])
        assert_equal(analyzer1.grid.weights, data_dict['weights'])
        assert_equal(analyzer1.ha_energy_density, data_dict['ha_energy_density'])
        assert_equal(analyzer1.ee_energy_density, data_dict['ee_energy_density'])
        os.remove(TMP_TEST)


class TestCCSDAnalyzerChunks(TestCCSDAnalyzer):

    @classmethod
    def setup_class(cls):
        cls.mol = gto.Mole(atom='He 0 0 0', basis = 'cc-pvdz')
        cls.mol.build()
        cls.hf = run_scf(cls.mol, 'RHF')
        cls.cc = run_cc(cls.hf)
        cls.analyzer = CCSDAnalyzer(cls.cc, max_mem=5)
        cls.ee_tot_ref = get_ccsd_ee_total(cls.mol, cls.cc, cls.hf)


class TestUCCSDAnalyzer():

    @classmethod
    def setup_class(cls):
        cls.mol = gto.Mole(atom='Li 0 0 0', basis = 'cc-pvdz', spin=1)
        cls.mol.build()
        cls.hf = run_scf(cls.mol, 'UHF')
        cls.cc = run_cc(cls.hf)
        cls.analyzer = UCCSDAnalyzer(cls.cc)
        cls.ee_tot_ref = get_ccsd_ee_total(cls.mol, cls.cc, cls.hf)

    def test_post_process(self):
        pass

    def test_get_ha_energy_density(self):
        eri = self.mol.intor('int2e')
        rdm1 = self.cc.make_rdm1()
        rdm1 = transform_basis_1e(rdm1, np.transpose(self.hf.mo_coeff, axes=(0,2,1)))
        rdm1 = np.sum(rdm1, axis=0)
        ha_tot_ref = np.sum(np.sum(eri * rdm1, axis=(2,3)) * rdm1) / 2
        ha_density = self.analyzer.get_ha_energy_density()
        ha_tot = integrate_on_grid(ha_density, self.analyzer.grid.weights)
        assert_almost_equal(ha_tot, ha_tot_ref)
        assert_almost_equal(self.analyzer.ha_total, ha_tot_ref)

    def test_get_ee_energy_density(self):
        ee_density = self.analyzer.get_ee_energy_density()
        ee_tot = integrate_on_grid(ee_density, self.analyzer.grid.weights)
        assert_almost_equal(ee_tot, self.ee_tot_ref)
        # ee repulsion should be similar to HF case
        # Note this case may not pass for all systems, but hsould pass for He and Li
        assert_almost_equal(ee_tot, self.hf.energy_elec()[1], 1)
        assert_almost_equal(self.analyzer.ee_total, self.ee_tot_ref)

    def test_as_dict_from_dict(self):
        analyzer1 = UCCSDAnalyzer(self.cc)
        dict1 = analyzer1.as_dict()
        analyzer1.get_ha_energy_density()
        analyzer1.get_ee_energy_density()
        dict2 = analyzer1.as_dict()
        analyzer2 = UCCSDAnalyzer.from_dict(dict1)
        analyzer2.get_ha_energy_density()
        analyzer2.get_ee_energy_density()
        assert_almost_equal(analyzer1.ha_energy_density, analyzer2.ha_energy_density)
        assert_almost_equal(analyzer1.ee_energy_density, analyzer2.ee_energy_density)
        assert_almost_equal(analyzer1.ee_energy_density_uu, analyzer2.ee_energy_density_uu)
        assert_almost_equal(analyzer1.ee_energy_density_ud, analyzer2.ee_energy_density_ud)
        assert_almost_equal(analyzer1.ee_energy_density_dd, analyzer2.ee_energy_density_dd)
        analyzer3 = UCCSDAnalyzer.from_dict(dict2)
        assert_almost_equal(analyzer1.ha_energy_density, analyzer3.ha_energy_density)
        assert_almost_equal(analyzer1.ee_energy_density, analyzer3.ee_energy_density)
        assert_almost_equal(analyzer1.ee_energy_density_uu, analyzer3.ee_energy_density_uu)
        assert_almost_equal(analyzer1.ee_energy_density_ud, analyzer3.ee_energy_density_ud)
        assert_almost_equal(analyzer1.ee_energy_density_dd, analyzer3.ee_energy_density_dd)

    def test_dump_load(self):
        analyzer1 = UCCSDAnalyzer(self.cc)
        analyzer1.get_ee_energy_density()
        analyzer1.get_ha_energy_density()
        analyzer1.dump(TMP_TEST)
        analyzer2 = UCCSDAnalyzer.load(TMP_TEST)
        for key in analyzer1.__dict__:
            if isinstance(analyzer1.__getattribute__(key), numbers.Number)\
                or isinstance(analyzer1.__getattribute__(key), np.ndarray):
                assert_equal(analyzer1.__getattribute__(key),
                    analyzer2.__getattribute__(key))
        data_dict = lib.chkfile.load(TMP_TEST, 'analyzer/data')
        assert_equal(analyzer1.grid.coords, data_dict['coords'])
        assert_equal(analyzer1.grid.weights, data_dict['weights'])
        assert_equal(analyzer1.ha_energy_density, data_dict['ha_energy_density'])
        assert_equal(analyzer1.ee_energy_density, data_dict['ee_energy_density'])
        os.remove(TMP_TEST)


class TestUCCSDAnalyzerChunks(TestUCCSDAnalyzer):

    @classmethod
    def setup_class(cls):
        cls.mol = gto.Mole(atom='Li 0 0 0', basis = 'cc-pvdz', spin=1)
        cls.mol.build()
        cls.hf = run_scf(cls.mol, 'UHF')
        cls.cc = run_cc(cls.hf)
        cls.analyzer = UCCSDAnalyzer(cls.cc, max_mem=5)
        cls.ee_tot_ref = get_ccsd_ee_total(cls.mol, cls.cc, cls.hf)
