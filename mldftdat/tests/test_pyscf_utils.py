from mldftdat.pyscf_utils import *

import ase.io
import unittest
from nose import SkipTest
from nose.tools import nottest
from nose.plugins.skip import Skip
from numpy.testing import assert_almost_equal, assert_equal, assert_raises


class TestPyscfUtils(unittest.TestCase):

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
        cls.He_rdm1, cls.He_rdm2 = cls.cc_He.make_rdm1(), cls.cc_He.make_rdm2()
        cls.He_vele_mat = get_vele_mat(cls.He, cls.He_grid.coords)
        cls.He_mo_vals = get_mo_vals(cls.He_ao_vals, cls.hf_He.mo_coeff)
        cls.He_mo_vele_mat = get_mo_vele_mat(cls.He_vele_mat, cls.hf_He.mo_coeff)

        cls.rtot_ref_h, cls.rtot_ref_x = get_hf_coul_ex_total(cls.FH, cls.rhf)
        cls.utot_ref_h, cls.utot_ref_x = get_hf_coul_ex_total_unrestricted(cls.NO, cls.uhf)
        cls.He_ref_ee = get_ccsd_ee_total(cls.He, cls.cc_He, cls.hf_He)

    def test_matrix_understanding(self):
        b = self.FH.get_ovlp()
        v = self.rhf.mo_coeff
        e = self.rhf.mo_energy
        f = self.rhf.get_fock()
        assert_almost_equal(np.dot(v.transpose(), np.dot(b, v)),
                            np.identity(b.shape[0]), 6)
        assert_almost_equal(np.dot(f, v[:,0]), e[0] * np.dot(b, v[:,0]), 6)

    def test_mol_from_ase(self):
        water = ase.io.read('test_files/water.sdf')
        mol = mol_from_ase(water, 'cc-pvdz')
        print(mol.atom, type(mol.atom))
        mol_atom_ref = [['C', np.array([-0.0127,  1.0858,  0.008 ])],\
                        ['H', np.array([ 0.0022, -0.006 ,  0.002 ])],\
                        ['H', np.array([1.0117e+00, 1.4638e+00, 3.0000e-04])],\
                        ['H', np.array([-0.5408,  1.4475, -0.8766])],\
                        ['H', np.array([-0.5238,  1.4379,  0.9064])]]
        assert_equal(len(mol.atom), len(mol_atom_ref))
        for item, ref in zip(mol.atom, mol_atom_ref):
            assert_equal(len(item), len(ref))
            self.assertEqual(item[0], ref[0])
            assert_equal(item[1], ref[1])
        self.assertEqual(mol.basis, 'cc-pvdz')

    def test_get_hf_coul_ex_total(self):
        # covered in setup
        jtot, ktot = get_hf_coul_ex_total(self.FH, self.rhf)
        assert_almost_equal(jtot + ktot, self.rhf.energy_elec()[1])

    def test_transform_basis_1e_and_2e(self):
        # currently just tests restricted case in a basic way
        mo_rdm_ref = np.diag(self.rhf.mo_occ)
        mo_rdm = transform_basis_1e(self.rhf_rdm1, np.linalg.inv(self.rhf.mo_coeff.transpose()))
        assert_almost_equal(mo_rdm, mo_rdm_ref)
        trdm1 = transform_basis_1e(self.He_rdm1, self.hf_He.mo_coeff.transpose())
        rdm1 = transform_basis_1e(trdm1, np.linalg.inv(self.hf_He.mo_coeff.transpose()))
        assert_almost_equal(rdm1, self.He_rdm1)

    def test_get_mgga_data(self):
        ao_data, rho_data = get_mgga_data(self.FH, self.rhf_grid, self.rhf_rdm1)
        desired_shape_ao = (20, self.rhf_grid.coords.shape[0], self.rhf_rdm1.shape[0])
        desired_shape_rho = (6, self.rhf_grid.coords.shape[0])
        assert_equal(ao_data.shape, desired_shape_ao)
        assert_equal(rho_data.shape, desired_shape_rho)

        ao_data, rho_data = get_mgga_data(self.NO, self.uhf_grid, self.uhf_rdm1)
        desired_shape_ao = (20, self.uhf_grid.coords.shape[0], self.uhf_rdm1[0].shape[0])
        desired_shape_rho = (6, self.uhf_grid.coords.shape[0])
        assert_equal(ao_data.shape, desired_shape_ao)
        assert_equal(len(rho_data), 2)
        for spin_data in rho_data:
            assert_equal(spin_data.shape, desired_shape_rho)
        assert_raises(AssertionError, assert_almost_equal, rho_data[0], rho_data[1])

    def test_make_rdm2_from_rdm1(self):
        rhf_rdm2 = make_rdm2_from_rdm1(self.rhf_rdm1)
        ree = get_ee_energy_density(self.FH, rhf_rdm2,
                                    self.rhf_vele_mat, self.rhf_ao_vals)
        rtot = integrate_on_grid(ree, self.rhf_grid.weights)
        assert_almost_equal(rtot, self.rtot_ref_h + self.rtot_ref_x, 5)

    def test_make_rdm2_from_rdm1_unrestricted(self):
        uhf_rdm2 = make_rdm2_from_rdm1_unrestricted(self.uhf_rdm1)
        euu = get_ee_energy_density(
                self.NO, uhf_rdm2[0],
                self.uhf_vele_mat, self.uhf_ao_vals)
        eud = get_ee_energy_density(
                self.NO, uhf_rdm2[1],
                self.uhf_vele_mat, self.uhf_ao_vals)
        edd = get_ee_energy_density(
                self.NO, uhf_rdm2[2],
                self.uhf_vele_mat, self.uhf_ao_vals)
        eee = euu + 2 * eud + edd
        etot = integrate_on_grid(eee, self.uhf_grid.weights)
        assert_almost_equal(etot, self.utot_ref_h + self.utot_ref_x, 5)

    def test_get_vele_mat(self):
        # covered in setup
        pass

    def test_get_ha_energy_density(self):
        rha = get_ha_energy_density(self.FH, self.rhf_rdm1,
                                    self.rhf_vele_mat, self.rhf_ao_vals)

        rtot = integrate_on_grid(rha, self.rhf_grid.weights)

        assert_almost_equal(rtot, self.rtot_ref_h, 5)

    def test_get_fx_energy_density(self):
        rfx = get_fx_energy_density(self.FH, self.rhf.mo_occ,
                                    get_mo_vele_mat(
                                        self.rhf_vele_mat, self.rhf.mo_coeff),
                                    get_mo_vals(self.rhf_ao_vals,
                                        self.rhf.mo_coeff))

        rtot = integrate_on_grid(rfx, self.rhf_grid.weights)

        assert_almost_equal(rtot, self.rtot_ref_x, 5)

    def test_get_ee_energy_density(self):
        ree = get_ee_energy_density(self.He, self.He_rdm2,
                                    self.He_mo_vele_mat, self.He_mo_vals)
        rtot = integrate_on_grid(ree, self.He_grid.weights)
        assert_almost_equal(rtot, self.He_ref_ee)
