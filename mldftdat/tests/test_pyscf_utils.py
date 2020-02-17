from mldftdat.pyscf_utils import *

import ase.io
import unittest
from scipy.special import erf
from pyscf.dft.gen_grid import Grids
from nose import SkipTest
from nose.tools import nottest
from nose.plugins.skip import Skip
from numpy.testing import assert_almost_equal, assert_equal, assert_raises,\
                        assert_array_almost_equal


class TestPyscfUtils(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.FH = gto.Mole(atom='H 0 0 0; F 0 0 1.1', basis = 'sto-3g')
        cls.FH.build()
        cls.rhf = run_scf(cls.FH, 'RHF')
        cls.rhf_grid = get_grid(cls.FH)
        cls.rhf_ao_vals = eval_ao(cls.FH, cls.rhf_grid.coords)
        cls.rhf_rdm1 = cls.rhf.make_rdm1()
        cls.rhf_vele_mat = get_vele_mat(cls.FH, cls.rhf_grid.coords)

        cls.NO = gto.Mole(atom='N 0 0 0; O 0 0 1.15', basis = 'sto-3g', spin = 1)
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

        cls.Li = gto.Mole(atom='Li 0 0 0', basis = 'cc-pvdz', spin=1)
        cls.Li.build()
        cls.hf_Li = run_scf(cls.Li, 'UHF')
        cls.cc_Li = run_cc(cls.hf_Li)

        """
        zs, wts = np.polynomial.legendre.leggauss(12)
        NUMPHI = 24
        phis = np.linspace(0, 2*np.pi, num=NUMPHI, endpoint=False)
        dphi = 2 * np.pi / NUMPHI
        dphi_lst = dphi * np.ones(phis.shape)

        rs = np.linspace(0.001, 5, 10)
        rsw0 = np.append([0], rs)
        drs = rsw0[1:] - rsw0[:-1]
        rwts = rs**2 * drs

        points = np.vstack(np.meshgrid(rs, zs, phis, indexing='ij')).reshape(3,-1)
        weights3d = np.vstack(np.meshgrid(rwts, wts, dphi_lst, indexing='ij')).reshape(3,-1)
        rhos = weights3d[1]
        weights = np.cumprod(weights3d, axis=0)[-1,:]
        xs = points[0] * np.sqrt(1-points[1]**2) * np.cos(points[2])
        ys = points[0] * np.sqrt(1-points[1]**2) * np.sin(points[2])
        zs = points[0] * points[1]
        coords = np.array([xs, ys, zs]).T
        """

        Hatom = gto.Mole(atom='H')
        Hatom.basis = 'sto-3g'
        Hatom.spin = 1
        Hatom.build()
        grid = Grids(Hatom)
        grid.level = 1
        grid.kernel()
        coords = grid.coords

        rs = np.linalg.norm(coords, axis=1)
        MAXR = 4.5
        coords = coords[rs < MAXR, :]
        weights = grid.weights[rs < MAXR]
        rs = rs[rs < MAXR]
        func = (1 / (4*np.pi)) * np.exp(-rs**2)
        cls.tst_rs = rs
        cls.tst_dens = func
        cls.tst_weights = weights
        cls.tst_coords = coords
        cls.tst_grad = (-2 * coords.T * func)
        cls.tst_grad2 = -2 * func + 4 * rs**2 * func
        cls.tst_tau = rs**2/2 * func
        cls.tst_grad_tau = (coords.T * func * (1 - rs**2))
        #cls.tst_tau = get_single_orbital_tau(rho, np.linalg.norm(cls.tst_grad))
        cls.tst_vh = np.sqrt(np.pi) * erf(rs) / (4 * rs)
        cls.tst_grad_vh = - np.sqrt(np.pi) * erf(rs) / (4 * rs**2) \
                     + np.exp(-rs**2) / (2 * rs) 

    def test_tst_grid_setup(self):
        N = np.sqrt(np.pi) / (4)
        assert_almost_equal(np.dot(self.tst_dens, self.tst_weights), N, 5)

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

    def test_run_scf(self):
        lda_He = run_scf(self.He, 'RKS')
        assert_equal(lda_He.xc, 'LDA,VWN')
        b3lyp_He = run_scf(self.He, 'RKS', functional='B3LYP')
        assert_equal(b3lyp_He.xc, 'B3LYP')
        assert_raises(AssertionError, assert_almost_equal, lda_He.e_tot, b3lyp_He.e_tot)
        assert_raises(AssertionError, assert_almost_equal, self.hf_He.e_tot, b3lyp_He.e_tot)

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

    def test_get_tau_and_grad(self):
        ao_data, rho_data = get_mgga_data(self.FH, self.rhf_grid, self.rhf_rdm1)
        tau_data = get_tau_and_grad(self.FH, self.rhf_grid, self.rhf_rdm1, ao_data)
        desired_shape = (4, self.rhf_grid.coords.shape[0])
        assert_equal(tau_data.shape, desired_shape)
        assert_almost_equal(tau_data[0], rho_data[5])
        zero = integrate_on_grid(tau_data[1:], self.rhf_grid.weights)
        assert_almost_equal(np.linalg.norm(zero), 0, 4)

        ao_data, rho_data = get_mgga_data(self.NO, self.uhf_grid, self.uhf_rdm1)
        tau_data = get_tau_and_grad(self.NO, self.uhf_grid, self.uhf_rdm1, ao_data)
        desired_shape = (2, 4, self.uhf_grid.coords.shape[0])
        assert_almost_equal(tau_data[:,0,:], rho_data[:,5,:])
        zero = integrate_on_grid(tau_data[:,1:,:], self.uhf_grid.weights)
        assert_almost_equal(np.linalg.norm(zero), 0, 4)

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

    def test_get_vele_mat_chunks(self):
        vele_mat = None
        for vele_chunk, ao_chunk in get_vele_mat_chunks(self.FH, self.rhf_grid.coords,
                                                        13, self.rhf_ao_vals):
            if vele_mat is None:
                vele_mat = vele_chunk
                ao_vals = ao_chunk
            else:
                vele_mat = np.append(vele_mat, vele_chunk, axis=0)
                ao_vals = np.append(ao_vals, ao_chunk, axis=0)
        assert_almost_equal(vele_mat, self.rhf_vele_mat)

        vele_mat = None
        for vele_chunk, ao_chunk in get_vele_mat_chunks(self.He, self.He_grid.coords,
                                        13, self.He_ao_vals, self.hf_He.mo_coeff):
            if vele_mat is None:
                vele_mat = vele_chunk
                ao_vals = ao_chunk
            else:
                vele_mat = np.append(vele_mat, vele_chunk, axis=0)
                ao_vals = np.append(ao_vals, ao_chunk, axis=0)
        assert_almost_equal(vele_mat, self.He_mo_vele_mat)

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

    def test_get_ha_energy_density2(self):
        rha_ref = get_ha_energy_density(self.FH, self.rhf_rdm1,
                                    self.rhf_vele_mat, self.rhf_ao_vals)

        vele_mat_gen1 = get_vele_mat_generator(self.FH, self.rhf_grid.coords,
                                               2, self.rhf_ao_vals)
        vele_mat_gen2 = get_vele_mat_generator(self.FH, self.rhf_grid.coords,
                                               13, self.rhf_ao_vals)

        rha1 = get_ha_energy_density2(self.FH, self.rhf_rdm1,
                                    self.rhf_vele_mat, self.rhf_ao_vals)
        rha2 = get_ha_energy_density2(self.FH, self.rhf_rdm1,
                                    vele_mat_gen1, self.rhf_ao_vals)
        rha3 = get_ha_energy_density2(self.FH, self.rhf_rdm1,
                                    vele_mat_gen2, self.rhf_ao_vals)
        assert_almost_equal(rha1, rha_ref)
        assert_almost_equal(rha2, rha_ref)
        assert_almost_equal(rha3, rha_ref)

    def test_load_calc(self):
        # also tests get_scf and get_ccsd
        rhf_test, calc_type = load_calc('test_files/RHF_HF.hdf5')
        assert calc_type == 'RHF'
        assert_almost_equal(rhf_test.e_tot, self.rhf.e_tot)
        assert_almost_equal(rhf_test.mo_energy, self.rhf.mo_energy)
        assert_almost_equal(rhf_test.mo_coeff, self.rhf.mo_coeff)

        uhf_test, calc_type = load_calc('test_files/UHF_NO.hdf5')
        assert calc_type == 'UHF'
        assert_almost_equal(uhf_test.e_tot, self.uhf.e_tot)
        assert_almost_equal(uhf_test.mo_energy, self.uhf.mo_energy)
        assert_almost_equal(uhf_test.mo_coeff, self.uhf.mo_coeff)

        ccsd_test, calc_type = load_calc('test_files/CCSD_He.hdf5')
        assert calc_type == 'CCSD'
        assert_almost_equal(ccsd_test.e_tot, self.cc_He.e_tot)
        assert_almost_equal(ccsd_test.t1, self.cc_He.t1)
        assert_almost_equal(ccsd_test.t2, self.cc_He.t2)
        assert_almost_equal(ccsd_test.l1, self.cc_He.l1)
        assert_almost_equal(ccsd_test.l2, self.cc_He.l2)

        uccsd_test, calc_type = load_calc('test_files/UCCSD_Li.hdf5')
        assert calc_type == 'UCCSD'
        assert_array_almost_equal(uccsd_test.e_tot, self.cc_Li.e_tot)
        assert_array_almost_equal(uccsd_test.t1[0], self.cc_Li.t1[0])
        assert_array_almost_equal(uccsd_test.t1[1], self.cc_Li.t1[1])
        assert_array_almost_equal(uccsd_test.t2[0], self.cc_Li.t2[0])
        assert_array_almost_equal(uccsd_test.t2[1], self.cc_Li.t2[1])
        assert_array_almost_equal(uccsd_test.t2[2], self.cc_Li.t2[2])

    def test_squish_density(self):
        # just check that the method is nondestructive
        ALPHA = 1.2
        coords, weights = self.rhf_grid.coords, self.rhf_grid.weights
        rhf_ao_data = eval_ao(self.FH, coords, deriv=2)
        rho_data = dft.numint.eval_rho(self.FH, rhf_ao_data, self.rhf_rdm1,
                                        xctype='mGGA')
        old_coords, old_weights, old_rho_data = coords.copy(), weights.copy(), rho_data.copy()
        new_coords, new_weights, new_rho_data = squish_density(rho_data, coords, weights, ALPHA)
        assert_equal(old_coords, coords)
        assert_equal(old_rho_data, rho_data)
        assert_almost_equal(new_coords, coords / ALPHA)
        assert_almost_equal(new_rho_data[0], rho_data[0] * ALPHA**3)
        assert_almost_equal(new_rho_data[1:4], rho_data[1:4] * ALPHA**4)
        assert_almost_equal(new_rho_data[4:6], rho_data[4:6] * ALPHA**5)

    def test_get_dft_input(self):
        # NOTE: might need to work on the precision here, though
        # not certain how to accomplish that.
        # At the very least, I need to att some controls for the uncertainty
        # at low densities.
        ALPHA = 1.2
        MAX_R = 2
        coords, weights = self.rhf_grid.coords, self.rhf_grid.weights
        rs = np.linalg.norm(coords, axis=1)
        coords = coords[rs<MAX_R,:]
        weights = weights[rs<MAX_R]
        rhf_ao_data = eval_ao(self.FH, coords, deriv=2)
        rho_data = dft.numint.eval_rho(self.FH, rhf_ao_data, self.rhf_rdm1,
                                        xctype='mGGA')
        coords, weights = self.rhf_grid.coords, self.rhf_grid.weights
        rho, s, alpha, tau_w, tau_unif = get_dft_input(rho_data)
        squish_coords, squish_weights, squish_rho_data = \
            squish_density(rho_data, coords, weights, ALPHA)
        squish_rho, squish_s, squish_alpha, squish_tau_w, squish_tau_unif = \
            get_dft_input(squish_rho_data)

        print(np.flip(np.sort(alpha)))
        assert (alpha >= -1e-16).all()

        assert_almost_equal(rho * ALPHA**3, squish_rho)
        assert_almost_equal(s, squish_s, 4)
        assert_almost_equal(alpha, squish_alpha, 2)
        assert_almost_equal(tau_w * ALPHA**5, squish_tau_w, 4)
        assert_almost_equal(tau_unif * ALPHA**5, squish_tau_unif, 4)

    def test_get_vh(self):
        i = 200
        coords = self.tst_coords - self.tst_coords[i]
        weights = self.tst_weights
        rs = np.linalg.norm(coords, axis=1)
        rs[i] = (2.0/3) * (3 * weights[i] / (4 * np.pi))**(1.0 / 3)
        vh = get_vh(self.tst_dens, rs, weights)
        assert_almost_equal(vh, self.tst_vh[i], 6)

    
    def test_get_nonlocal_data(self):
        # should check that the values make sense when passed
        # the dummy density
        rho_data = np.append(self.tst_dens.reshape((1, self.tst_dens.shape[0])),
                                self.tst_grad, axis=0)
        tau_data = np.append(self.tst_tau.reshape((1, self.tst_dens.shape[0])),
                                self.tst_grad_tau, axis=0)
        ws_radii = get_ws_radii(rho_data[0])
        nldat = get_nonlocal_data(rho_data, tau_data, ws_radii,
                                self.tst_coords, self.tst_weights)

        import matplotlib.pyplot as plt
        plt.scatter(self.tst_rs, np.abs(self.tst_grad_vh))
        plt.scatter(self.tst_rs, nldat[0])
        plt.show()
        assert_almost_equal(nldat[0], np.abs(self.tst_grad_vh), 2)
    

    def test_regularize_nonlocal_data(self):
        # should just test that the resulting values
        # obey density scaling relations
        pass
