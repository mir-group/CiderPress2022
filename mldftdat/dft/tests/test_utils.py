from pyscf import scf, gto, lib
from nose import SkipTest
from nose.tools import nottest
from nose.plugins.skip import Skip
from numpy.testing import assert_almost_equal, assert_equal
from mldftdat.pyscf_utils import run_scf
from pyscf.dft.libxc import eval_xc
from pyscf.dft.numint import eval_rho, eval_ao

from mldftdat.pyscf_utils import get_gradient_magnitude, get_dft_input2
from mldftdat.analyzers import RHFAnalyzer
from mldftdat.dft.utils import v_semilocal, v_basis_transform
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

