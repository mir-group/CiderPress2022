from mldftdat.pyscf_utils import *

from nose import SkipTest
from nose.tools import nottest
from nose.plugins.skip import Skip
from numpy.testing import assert_almost_equal, assert_equal


class TestPyscfUtils():

    @classmethod
    def setUpClass(cls):

        self.FH = gto.Mole(atom='H 0 0 0; F 0 0 1.1', basis = '631g')
        self.FH.build()
        self.rhf = run_scf(self.FH, 'RHF')
        self.rhf_grid = get_grid(self.FH)
        self.rhf_ao_vals = eval_ao(self.FH, self.rhf_grid.coords)
        self.rhf_rdm1 = self.rhf.make_rdm1()
        self.rhf_vele_mat = get_vele_mat(self.FH, self.rhf_grid.coords)

        self.NO = gto.Mole(atom='N 0 0 0; O 0 0 1.15', basis = '631g')
        self.NO.build()
        self.uhf = run_scf(self.NO, 'UHF')
        self.uhf_grid = get_grid(self.NO)
        self.uhf_ao_vals = eval_ao(self.NO, self.uhf_grid.coords)
        self.uhf_rdm1 = self.uhf.make_rdm1()
        self.uhf_vele_mat = get_vele_mat(self.NO, self.uhf_grid.coords)

        self.He = gto.Mole(atom='He 0 0 0', basis = 'cc-pvdz')
        self.He.build()
        self.hf_He = run_scf(self.He, 'RHF')
        self.cc_He = run_cc(self.He, self.hf_He)
        self.He_grid = get_grid(self.He)
        self.He_ao_vals = eval_ao(self.He, self.He_grid.coords)
        self.He_rdm1, self.He_rdm2 = get_cc_rdms(self.cc_He)
        self.He_aordm1, self.He_aordm2 = get_cc_rdms(self.cc_He,
                                                    self.hf_He.mo_coeff)
        self.He_vele_mat = get_vele_mat(self.He, self.He_grid.coords)

        self.rtot_ref_h, self.rtot_ref_x = get_hf_coul_ex_total(self.FH, self.rhf)
        self.utot_ref_h, self.utot_ref_x = get_hf_coul_ex_total(self.NO, self.uhf)
        self.He_ref_ee = get_ccsd_ee_total(self.He, self.cc_he, self.hf_He)

    def test_mol_from_ase(self):
        pass

    def test_run_scf(self):
        # covered in setup
        pass

    def test_run_cc(self):
        # covered in setup
        pass

    def test_get_cc_rdms(self):
        eeint = mol.intor('int2e', aosym='s1')
        eee = np.sum(eeint * self.He_aordm2) / 2
        ha = get_ha_total(self.He_aordm1, eeint)
        assert_almost_equal(eee, self.He_ref_ee)

        eeint = ao2mo.incore.full(eeint, self.hf_He.mo_coeff)
        ha_ref = get_ha_total(self.He_rdm1, eeint)
        ee_ref = np.sum(eeint * self.He_rdm2) / 2
        assert_almost_equal(ee_ref, self.He_ref_ee)
        assert_almost_equal(ha, ha_ref)

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
        uhf_rdm2 = make_rdm2_from_rdm1(self.uhf_rdm1, restricted = False)

        ree = get_ee_energy_density(self.FH, rhf_rdm2,
                                    self.rhf_vele_mat, self.rhf_ao_vals)
        uee = get_ee_energy_density(self.NO, uhf_rdm2,
                                    self.uhf_vele_mat, self.uhf_ao_vals)

        rtot = integrate_on_grid(ree, self.rhf_grid.weights)
        utot = integrate_on_grid(uee, self.uhf_grid.weights)

        assert_almost_equal(rtot, self.rtot_ref_h + self.rtot_ref_x, 5)
        assert_almost_equal(utot, self.utot_ref_h + self.utot_ref_x, 5)

    def test_get_vele_mat(self):
        # covered in setup
        pass

    def test_get_ha_energy_density(self):
        rha = get_fx_energy_density(self.FH, self.rhf_rdm1,
                                    self.rhf_vele_mat, self.rhf_ao_vals)
        uha = get_fx_energy_density(self.NO, self.uhf_rdm1,
                                    self.uhf_vele_mat, self.uhf_ao_vals)

        rtot = integrate_on_grid(rha, self.rhf_grid.weights)
        utot = integrate_on_grid(uha, self.uhf_grid.weights)

        assert_almost_equal(rtot, self.rtot_ref_h, 5)
        assert_almost_equal(utot, self.utot_ref_h, 5)

    def test_get_fx_energy_density(self):
        rfx = get_fx_energy_density(self.FH, self.rhf_rdm1,
                                    self.rhf_vele_mat, self.rhf_ao_vals)
        ufx = get_fx_energy_density(self.NO, self.uhf_rdm1,
                                    self.uhf_vele_mat, self.uhf_ao_vals,
                                    restricted = False)

        rtot = integrate_on_grid(rfx, self.rhf_grid.weights)
        utot = integrate_on_grid(ufx, self.uhf_grid.weights)

        assert_almost_equal(rtot, self.rtot_ref_x, 5)
        assert_almost_equal(utot, self.utot_ref_x, 5)

    def test_get_ee_energy_density(self):
        eee = get_ee_energy_density(self.He, self.He_aordm2,
                                    self.He_vele_mat, self.He_ao_vals)
        tot = integrate_on_grid(eee, self.He_grid.weights)
        assert_almost_equal(tot, self.He_ref_ee)


        ree = get_ee_energy_density(self.FH, self.rhf_rdm1,
                                    self.rhf_vele_mat, self.rhf_ao_vals)
        uee = get_ee_energy_density(self.NO, self.uhf_rdm1,
                                    self.uhf_vele_mat, self.uhf_ao_vals)
        rtot = integrate_on_grid(ree, self.rhf_grid.weights)
        utot = integrate_on_grid(uee, self.uhf_grid.weights)
