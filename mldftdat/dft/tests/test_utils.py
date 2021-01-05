from pyscf import scf, gto, lib
from nose import SkipTest
from nose.tools import nottest
from nose.plugins.skip import Skip
from numpy.testing import assert_almost_equal, assert_equal
from mldftdat.pyscf_utils import run_scf
from pyscf.dft.libxc import eval_xc
from pyscf.dft.numint import eval_rho, eval_ao
from pyscf import scf, dft, gto
from pyscf.dft.gen_grid import Grids

from mldftdat.pyscf_utils import get_gradient_magnitude, get_dft_input2
from mldftdat.analyzers import RHFAnalyzer
from mldftdat.density import get_gaussian_grid, get_gaussian_grid_c
from mldftdat.dft.utils import *
import numpy as np
import numbers
import os

TMP_TEST = 'test_files/tmp'

LDA_FACTOR = - 3.0 / 4.0 * (3.0 / np.pi)**(1.0/3)

class TestFunctionalDerivatives():

    @classmethod
    def setup_class(cls):
        cls.mol = gto.Mole(atom='H 0 0 0; F 0 0 1.1', basis = 'sto-3g')
        cls.mol.build()
        cls.rhf = run_scf(cls.mol, 'RHF')
        cls.analyzer = RHFAnalyzer(cls.rhf)
        cls.analyzer.get_ao_rho_data()

    def test_semilocal(self):
        """
        Tests v_semilocal and v_basis_transform by checking it can
        reproduce the PBE and SCAN potentials.
        """
        kappa = 0.804
        mu = 0.21951
        rho_data = self.analyzer.rho_data
        mag_grad = get_gradient_magnitude(rho_data)
        rho, s, alpha, _, _ = get_dft_input2(rho_data)
        p = s**2
        fpbe = 1 + kappa - kappa / (1 + mu * p / kappa)
        dpbe = mu / (1 + mu * p / kappa)**2
        v_npa = v_semilocal(rho_data, fpbe, dpbe, 0)
        v_nst = v_basis_transform(rho_data, v_npa)
        eps_ref, v_ref, _, _ = eval_xc('PBE,', rho_data)
        cond1 = rho > 1e-4
        cond2 = rho > 1e-2
        for i in [0, 1, 3]:
            if v_ref[i] is None:
                continue
            assert (not np.isnan(v_nst[i]).any())
            # some small numerical differences here, might want to take a look later
            assert_almost_equal(v_nst[i][cond1], v_ref[i][cond1], 4)
            assert_almost_equal(v_nst[i][cond2], v_ref[i][cond2], 5)

        muak = 10.0 / 81
        k1 = 0.065
        b2 = np.sqrt(5913 / 405000)
        b1 = (511 / 13500) / (2 * b2)
        b3 = 0.5
        b4 = muak**2 / k1 - 1606 / 18225 - b1**2
        h0 = 1.174
        a1 = 4.9479
        c1 = 0.667
        c2 = 0.8
        dx = 1.24
        tmp1 = muak * p
        tmp2 = 1 + b4 * p / muak * np.exp(-np.abs(b4) * p / muak)
        tmp3 = b1 * p + b2 * (1 - alpha) * np.exp(-b3 * (1 - alpha)**2)
        x = tmp1 * tmp2 + tmp3**2
        h1 = 1 + k1 - k1 / (1 + x / k1)
        gx = 1 - np.exp(-a1 / np.sqrt(s + 1e-9))
        dgdp = - a1 / 4 * (s + 1e-9)**(-2.5) * np.exp(-a1 / np.sqrt(s + 1e-9))
        fx = np.exp(-c1 * alpha / (1 - alpha)) * (alpha < 1)\
             - dx * np.exp(c2 / (1 - alpha)) * (alpha > 1)
        fx[np.isnan(fx)] = 0
        assert (not np.isnan(fx).any())
        Fscan = gx * (h1 + fx * (h0 - h1))
        dxdp = muak * tmp2 + tmp1 * (b4 / muak * np.exp(-np.abs(b4) * p / muak)\
               - b4 * np.abs(b4) * p / muak**2 * np.exp(-np.abs(b4) * p / muak))\
               + 2 * tmp3 * b1
        dxda = 2 * tmp3 * (-b2 * np.exp(-b3 * (1 - alpha)**2) \
                            + 2 * b2 * b3 * (1 - alpha)**2 * np.exp(-b3 * (1 - alpha)**2) )
        dhdx = 1 / (1 + x / k1)**2
        dhdp = dhdx * dxdp
        dhda = dhdx * dxda
        dfda = (-c1 * alpha / (1 - alpha)**2 - c1 / (1 - alpha))\
                * np.exp(-c1 * alpha / (1 - alpha)) * (alpha < 1)\
                - dx * c2 / (1 - alpha)**2 * np.exp(c2 / (1 - alpha)) * (alpha > 1)
        dfda[np.isnan(dfda)] = 0

        dFdp = dgdp * (h1 + fx * (h0 - h1)) + gx * (1 - fx) * dhdp
        dFda = gx * (dhda - fx * dhda + dfda * (h0 - h1))

        v_npa = v_semilocal(rho_data, Fscan, dFdp, dFda)
        v_nst = v_basis_transform(rho_data, v_npa)
        eps_ref, v_ref, _, _ = eval_xc('SCAN,', rho_data)
        cond1 = rho > 1e-4
        cond2 = rho > 1e-2
        eps = Fscan * LDA_FACTOR * rho_data[0,:]**(1.0/3)
        assert_almost_equal(eps[cond1], eps_ref[cond1])
        for i in [0, 1, 3]:
            print(i)
            if v_ref[i] is None:
                continue
            assert (not np.isnan(v_nst[i]).any())
            # some small numerical differences here, might want to take a look later
            assert_almost_equal(v_nst[i][cond1], v_ref[i][cond1], 4)
            assert_almost_equal(v_nst[i][cond2], v_ref[i][cond2], 5)

    def eval_integration(self, func):
        LDA_FACTOR = - 3.0 / 4.0 * (3.0 / np.pi)**(1.0/3)

        mol = gto.Mole(atom = 'He')
        norm = 1.0 / (2 / np.pi)**(0.75) * 1e12
        print(norm)
        mol.basis = {'He' : gto.basis.parse('''
        BASIS "ao basis" PRINT
        He    S
              1.0000E-16     1.000000
        END
        '''.format(norm))}
        print(mol.basis)
        mol.build()
        mol._env[mol._bas[:,6]] = np.sqrt(4 * np.pi)
        grid = Grids(mol)
        grid.build()
        r = np.linalg.norm(grid.coords, axis = 1)
        print(np.max(r), r.shape)
        rho = np.ones(grid.weights.shape)
        rho_data = np.zeros((6, rho.shape[0]))
        # HEG
        rho_data[0] = rho

        rho, s, alpha, tau_w, tau_unif = get_dft_input2(rho_data)
        alpha += 1
        rho_data[5] = tau_unif
        atm, bas, env = func(grid.coords, rho[0], l=0, s=s, alpha=alpha)
        gridmol = gto.Mole(_atm = atm, _bas = bas, _env = env)
        a = gridmol._env[gridmol._bas[:,5]]
        norm = mol.intor('int1e_ovlp')
        print(norm**0.5)
        g = a[:,np.newaxis] * gto.mole.intor_cross('int1e_r2_origj', mol, gridmol).T
        assert_almost_equal(g.flatten(), 3)
        g = a[:,np.newaxis]**2 * gto.mole.intor_cross('int1e_r4_origj', mol, gridmol).T
        assert_almost_equal(g.flatten(), 15./2)
        g = gto.mole.intor_cross('int1e_ovlp', gridmol, mol)
        assert_almost_equal(g, 2)

        dfdg = 3.2 * np.ones(r.shape)
        print('shape', dfdg.shape)
        density = np.ones(1)

        fac = (6 * np.pi**2)**(2.0/3) / (16 * np.pi)
        ref_val = 2 * LDA_FACTOR * dfdg
        ref_dgda = -3 * 2**(2.0/3) / np.pi
        ref_dedb = LDA_FACTOR * dfdg
        dadn = 2 * a / (3 * rho_data[0])
        dadp = a * fac 
        print('dadp 2', dadn, ref_dgda, ref_dedb)
        dadalpha = a * fac * 0.6
        ref_dn = ref_dedb * ref_dgda * dadn
        ref_dp = ref_dedb * ref_dgda * dadp
        ref_dalpha = ref_dedb * ref_dgda * dadalpha

        """
        print('hi', a[0] * fac, np.pi * fac * (rho_data[0] / 2)**(2.0/3))
        vbas = project_xc_basis(ref_dedb, gridmol, grid, l=0)
        print(np.linalg.norm(vbas - ref_val))
        print(np.mean(vbas[r < 3]), np.max(vbas[r < 3]), np.min(vbas[r < 3]), np.min(ref_val))
        assert_almost_equal(vbas[r < 3], ref_val[r < 3], 3)
        """

        """
        v_npa = v_nonlocal(rho_data, grid, dfdg, density, mol, g, l = 0, mul = 1.0)
        print(np.mean(v_npa[0][r < 3] - vbas[r < 3]), np.max(ref_dn))
        print(np.mean(v_npa[0][r < 3]), np.max(ref_dn))
        assert_almost_equal(v_npa[0][r < 3] - vbas[r < 3], ref_dn[r < 3], 3)
        #assert_almost_equal(v_npa[1][r < 3], ref_dp[r < 3], 2)
        assert_almost_equal(v_npa[3][r < 3], ref_dalpha[r < 3], 3)
        """

    def test_integration(self):
        self.eval_integration(get_gaussian_grid)

    def test_integration_c(self):
        self.eval_integration(get_gaussian_grid_c)
