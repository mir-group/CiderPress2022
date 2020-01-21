from mldftdat.pyscf_utils import *

from nose import SkipTest
from nose.tools import nottest
from nose.plugins.skip import Skip
from numpy.testing import assert_almost_equal, assert_equal


class TestPyscfUtils():

    @classmethod
    def setUpClass(cls):

        cls.FH = gto.Mole(atom='H 0 0 0; F 0 0 1.1', basis = '631g')
        cls.FH.build()
        cls.rhf = run_scf(cls.FH, 'RHF')
        cls.rhf_grid = get_grid(cls.FH)
        cls.rhf_ao_vals = eval_ao(cls.FH, cls.rhf_grid.coords)
        cls.rhf_rdm1 = cls.rhf.make_rdm1()
        cls.rhf_vele_mat = get_vele_mat(cls.FH, cls.rhf_grid.coords)

        cls.NO = gto.Mole(atom='N 0 0 0; O 0 0 1.15', basis = '631g', spin = 1)
        cls.NO.build()
        cls.uhf = run_scf(cls.NO, 'UHF')
        cls.uhf_grid = get_grid(cls.NO)
        cls.uhf_ao_vals = eval_ao(cls.NO, cls.uhf_grid.coords)
        cls.uhf_rdm1 = cls.uhf.make_rdm1()
        cls.uhf_vele_mat = get_vele_mat(cls.NO, cls.uhf_grid.coords)

        cls.He = gto.Mole(atom='He 0 0 0', basis = 'cc-pvdz')
        cls.He.build()
        cls.hf_He = run_scf(cls.He, 'RHF')
        cls.cc_He = run_cc(cls.hf_He)
        cls.He_grid = get_grid(cls.He)
        cls.He_ao_vals = eval_ao(cls.He, cls.He_grid.coords)
        cls.He_rdm1, cls.He_rdm2 = get_cc_rdms(cls.cc_He)
        cls.He_vele_mat = get_vele_mat(cls.He, cls.He_grid.coords)
        cls.He_mo_vals = get_mo_vals(cls.He_ao_vals, cls.hf_He.mo_coeff)
        cls.He_mo_vele_mat = get_mo_vele_mat(cls.He_vele_mat, cls.hf_He.mo_coeff)

        cls.Li = gto.Mole(atom='Li 0 0 0', basis = 'cc-pvdz', spin = 1)
        cls.Li.build()
        cls.hf_Li = run_scf(cls.Li, 'UHF')
        cls.cc_Li = run_cc(cls.hf_Li)
        cls.Li_grid = get_grid(cls.Li)
        cls.Li_ao_vals = eval_ao(cls.Li, cls.Li_grid.coords)
        cls.Li_rdm1, cls.Li_rdm2 = get_cc_rdms(cls.cc_Li)
        cls.Li_vele_mat = get_vele_mat(cls.Li, cls.Li_grid.coords)
        cls.Li_mo_vals = get_mo_vals(cls.Li_ao_vals, cls.hf_Li.mo_coeff)
        cls.Li_mo_vele_mat = get_mo_vele_mat_unrestricted(cls.Li_vele_mat, cls.hf_Li.mo_coeff)

        cls.rtot_ref_h, cls.rtot_ref_x = get_hf_coul_ex_total(cls.FH, cls.rhf)
        cls.utot_ref_h, cls.utot_ref_x = get_hf_coul_ex_total(cls.NO, cls.uhf)
        cls.He_ref_ee = get_ccsd_ee_total(cls.He, cls.cc_He, cls.hf_He)
        cls.Li_ref_ee = get_ccsd_ee_total(cls.Li, cls.cc_Li, cls.hf_Li)

    def test_mol_from_ase(self):
        pass

    def test_run_scf(self):
        # covered in setup
        pass

    def test_run_cc(self):
        # covered in setup
        pass

    def test_get_cc_rdms(self):
        # covered in setup
        pass

    def test_get_grid(self):
        # covered in setup
        pass

    def test_get_ha_total(self):
        # covered in setup
        pass

    def test_get_hf_coul_ex_total(self):
        # covered in setup
        pass

    def test_get_ccsd_ee_total(self):
        # covered in setup
        pass

    def test_make_rdm2_from_rdm1(self):
        rhf_rdm2 = make_rdm2_from_rdm1(self.rhf_rdm1)
        ree = get_ee_energy_density(self.FH, rhf_rdm2,
                                    self.rhf_vele_mat, self.rhf_ao_vals)
        rtot = integrate_on_grid(ree, self.rhf_grid.weights)
        assert_almost_equal(rtot, self.rtot_ref_h + self.rtot_ref_x, 5)

    def test_make_rdm2_from_rdm1_unrestricted(self):
        uhf_rdm2 = make_rdm2_from_rdm1_unrestricted(self.uhf_rdm1)
        uee = get_ee_energy_density(self.NO, uhf_rdm2,
                                    self.uhf_vele_mat, self.uhf_ao_vals)
        utot = integrate_on_grid(uee, self.uhf_grid.weights)
        assert_almost_equal(utot, self.utot_ref_h + self.utot_ref_x, 5)

    def test_get_vele_mat(self):
        # covered in setup
        pass

    def test_get_ha_energy_density(self):
        rha = get_ha_energy_density(self.FH, self.rhf_rdm1,
                                    self.rhf_vele_mat, self.rhf_ao_vals)
        print(self.uhf_rdm1.shape)
        uha = get_ha_energy_density(self.NO, self.uhf_rdm1,
                                    self.uhf_vele_mat, self.uhf_ao_vals)

        rtot = integrate_on_grid(rha, self.rhf_grid.weights)
        utot = integrate_on_grid(uha, self.uhf_grid.weights)

        assert_almost_equal(rtot, self.rtot_ref_h, 5)
        assert_almost_equal(utot, self.utot_ref_h, 5)

    def test_get_fx_energy_density(self):
        rfx = get_fx_energy_density(self.FH, self.rhf.mo_occ,
                                    get_mo_vele_mat(
                                        self.rhf_vele_mat, self.rhf.mo_coeff),
                                    get_mo_vals(self.rhf_ao_vals,
                                        self.rhf.mo_coeff))
        uhf_mo_vele_mat = get_mo_vele_mat_unrestricted(
                                self.uhf_vele_mat, self.uhf.mo_coeff)
        uhf_mo_vals = get_mo_vals(self.uhf_ao_vals, self.uhf.mo_coeff)
        ufx, ufx_parts = get_fx_energy_density_unrestricted(self.NO, self.uhf.mo_occ,
                                    uhf_mo_vele_mat, uhf_mo_vals)

        rtot = integrate_on_grid(rfx, self.rhf_grid.weights)
        utot = integrate_on_grid(ufx, self.uhf_grid.weights)

        assert_almost_equal(rtot, self.rtot_ref_x, 5)
        assert_almost_equal(utot, self.utot_ref_x, 5)

    def test_get_ee_energy_density(self):
        ree = get_ee_energy_density(self.He, self.He_rdm2,
                                    self.He_mo_vele_mat, self.He_mo_vals)
        rtot = integrate_on_grid(ree, self.He_grid.weights)
        assert_almost_equal(rtot, self.He_ref_ee)

        uee, uee_parts = get_ee_energy_density_unrestricted(
                                    self.Li, self.Li_rdm2,
                                    self.Li_mo_vele_mat, self.Li_mo_vals
                                    )
        utot = integrate_on_grid(uee, self.Li_grid.weights)
        assert_almost_equal(utot, self.Li_ref_ee)
