from pyscf import scf, gto, lib
from nose import SkipTest
from nose.tools import nottest
from nose.plugins.skip import Skip
from numpy.testing import assert_almost_equal

from mldftdat.xcutil.cdesc import *
from mldftdat.lowmem_analyzers import UHFAnalyzer
import numpy as np

def make_mesh(*args):
    return [a.flatten() for a in\
            np.meshgrid(*args)]

class TestCDesc():

    @classmethod
    def setup_class(cls):
        cls.analyzer = UHFAnalyzer.load('test_files/UHF_NO.hdf5')
        cls.rhou = cls.analyzer.rho_data[0,0,:]
        cls.rhod = cls.analyzer.rho_data[1,0,:]
        cls.rho = cls.rhou + cls.rhod
        cls.rhou = cls.rhou[cls.rho>1e-3]
        cls.rhod = cls.rhod[cls.rho>1e-3]
        cls.rho = cls.rho[cls.rho>1e-6]
        cls.delta = 1e-8
        cls.rho = np.exp(np.linspace(-6, 6, 10, dtype=np.float64))
        cls.rho2 = np.exp(np.linspace(-3, 3, 10, dtype=np.float64))
        cls.zeta = np.linspace(-0.99, 0.99, 11, dtype=np.float64)
        cls.chi = np.linspace(-1, 1, 11, dtype=np.float64)
        cls.x2 = np.exp(np.linspace(-7, 7, 10, dtype=np.float64))
        cls.g2 = np.exp(np.linspace(-3, 3, 10, dtype=np.float64))
        cls.rs = get_rs(cls.rho)[0]

    def test_get_rs(self):
        rs0, drs0 = get_rs(self.rho)
        rs1, drs1 = get_rs(self.rho*(1+self.delta))
        assert_almost_equal((rs1-rs0)/(self.rho*self.delta), drs0)

    def test_get_zeta(self):
        d = 1e-8
        z0, dzu0, dzd0 = get_zeta(self.rhou, self.rhod)
        z1, dzu1, dzd1 = get_zeta(self.rhou+d, self.rhod)
        z2, dzu2, dzd2 = get_zeta(self.rhou, self.rhod+d)
        assert_almost_equal((z1-z0)/d, dzu0)
        assert_almost_equal((z2-z0)/d, dzd0)

    def test_get_pw92term(self):
        rs, _ = get_rs(self.rhou)
        for code in [0,1,2]:
            e0, de0 = get_pw92term(rs, code)
            e1, de1 = get_pw92term(rs+1e-8, code)
            assert_almost_equal((e1-e0)/self.delta, de0)

    def test_get_pw92(self):
        rs, _ = get_rs(self.rho)
        d = 1e-8
        grid = make_mesh(rs, self.zeta)
        gridr = make_mesh(rs+d, self.zeta)
        gridz = make_mesh(rs, self.zeta+d)
        f, dfdr, dfdz = get_pw92(*grid)
        fr, _, _ = get_pw92(*gridr)
        fz, _, _ = get_pw92(*gridz)
        assert_almost_equal((fr-f)/d, dfdr)
        assert_almost_equal((fz-f)/d, dfdz)
        zeta = np.linspace(0,0.99,100)
        rs = np.ones(zeta.shape)
        fp, dfdrp, dfdzp = get_pw92(rs, zeta)
        fm, dfdrm, dfdzm = get_pw92(rs, -zeta)
        assert_almost_equal(fp, fm)
        assert_almost_equal(dfdrp, dfdrm)
        assert_almost_equal(dfdzp, -dfdzm)

    def test_get_phi0(self):
        zeta = np.linspace(-0.999999, 0.999999, 100)
        p0, dp0 = get_phi0(zeta)
        p1, dp1 = get_phi0(zeta+1e-8)
        assert_almost_equal((p1-p0)/self.delta, dp0)

    def test_get_phi1(self):
        d = 1e-8
        zeta = np.linspace(-0.99, 0.99, 100)
        p0, dp0 = get_phi1(zeta)
        p1, dp1 = get_phi1(zeta+d)
        assert_almost_equal((p1-p0)/d, dp0, 6)

    def test_get_amix_schmidt2(self):
        d = 1e-10
        grid = make_mesh(self.rho, self.zeta, self.x2, self.chi)
        gridr = make_mesh(self.rho+d, self.zeta, self.x2, self.chi)
        gridz = make_mesh(self.rho, self.zeta+d, self.x2, self.chi)
        gridx = make_mesh(self.rho, self.zeta, self.x2+d, self.chi)
        gridc = make_mesh(self.rho, self.zeta, self.x2, self.chi+d)

        f, dfdn, dfdz, dfdx2, dfdchi = get_amix_schmidt2(*grid)
        fr, _, _, _, _ = get_amix_schmidt2(*gridr)
        fz, _, _, _, _ = get_amix_schmidt2(*gridz)
        fx, _, _, _, _ = get_amix_schmidt2(*gridx)
        fc, _, _, _, _ = get_amix_schmidt2(*gridc)

        print(dfdn[np.abs((fr-f)/d-dfdn)>1e-5].tolist())
        print(grid[0][np.abs((fr-f)/d-dfdn)>1e-5].tolist())
        print(grid[1][np.abs((fr-f)/d-dfdn)>1e-5].tolist())
        print(grid[2][np.abs((fr-f)/d-dfdn)>1e-5].tolist())
        print(grid[3][np.abs((fr-f)/d-dfdn)>1e-5].tolist())
        print(np.max(np.abs(dfdn)))

        assert_almost_equal((fr-f)/d, dfdn, 5)
        assert_almost_equal((fz-f)/d, dfdz, 5)
        assert_almost_equal((fx-f)/d, dfdx2, 5)
        assert_almost_equal((fc-f)/d, dfdchi, 5)

    def test_get_baseline1(self):
        d=1e-8
        alle = []
        for z in [0.9, 0.7, 0.0, -0.7, -0.9]:
            e, dedlda, dedrs, dedzeta, deds2 = get_baseline1(1, 1, z, 1)
            en, _, _, _, _ = get_baseline1(1+d, 1, z, 1)
            er, _, _, _, _ = get_baseline1(1, 1+d, z, 1)
            ez, _, _, _, _ = get_baseline1(1, 1, z+d, 1)
            es, _, _, _, _ = get_baseline1(1, 1, z, 1+d)
            assert_almost_equal((en-e)/d, dedlda)
            assert_almost_equal((er-e)/d, dedrs)
            assert_almost_equal((ez-e)/d, dedzeta)
            assert_almost_equal((es-e)/d, deds2)
            alle.append(e)
        assert_almost_equal(alle[0], alle[-1])
        assert_almost_equal(alle[1], alle[-2])

    def test_get_baseline0(self):
        d=1e-8
        grid = make_mesh(self.rs, self.zeta, self.x2)
        gridr = make_mesh(self.rs+d, self.zeta, self.x2)
        gridz = make_mesh(self.rs, self.zeta+d, self.x2)
        gridx = make_mesh(self.rs, self.zeta, self.x2+d)

        e, dedrs, dedzeta, deds2 = get_baseline0(*grid)
        er, _, _, _ = get_baseline0(*gridr)
        ez, _, _, _ = get_baseline0(*gridz)
        es, _, _, _ = get_baseline0(*gridx)
        assert_almost_equal((er-e)/d, dedrs, 6)
        assert_almost_equal((ez-e)/d, dedzeta, 6)
        assert_almost_equal((es-e)/d, deds2, 6)

    def test_get_baseline1(self):
        d=1e-8
        grid = make_mesh(self.rs, self.zeta, self.x2)
        gridr = make_mesh(self.rs+d, self.zeta, self.x2)
        gridz = make_mesh(self.rs, self.zeta+d, self.x2)
        gridx = make_mesh(self.rs, self.zeta, self.x2+d)

        e, dedrs, dedzeta, deds2 = get_baseline1b(*grid)
        er = get_baseline1b(*gridr)[0]
        ez = get_baseline1b(*gridz)[0]
        es = get_baseline1b(*gridx)[0]
        assert_almost_equal((er-e)/d, dedrs, 6)
        assert_almost_equal((ez-e)/d, dedzeta, 6)
        assert_almost_equal((es-e)/d, deds2, 6)

    def test_get_os_baseline(self):
        d = 1e-8
        grid = make_mesh(self.rho2, self.rho2, self.x2)
        gridu = make_mesh(self.rho2+d, self.rho2, self.x2)
        gridd = make_mesh(self.rho2, self.rho2+d, self.x2)
        gridg = make_mesh(self.rho2, self.rho2, self.x2+d)
        grid[2][:] *= (grid[0][:] + grid[1][:])**(4./3)
        gridu[2][:] *= (gridu[0][:] + gridu[1][:])**(4./3)
        gridd[2][:] *= (gridd[0][:] + gridd[1][:])**(4./3)
        gridg[2][:] *= (gridg[0][:] + gridg[1][:])**(4./3)

        for t in [0, 1]:
            print('TYPE', t)
            e, v = get_os_baseline(*grid, type=t)
            eu, _ = get_os_baseline(*gridu, type=t)
            ed, _ = get_os_baseline(*gridd, type=t)
            eg, _ = get_os_baseline(*gridg, type=t)
            e *= (grid[0][:] + grid[1][:])
            eu *= (gridu[0][:] + gridu[1][:])
            ed *= (gridg[0][:] + gridg[1][:])

            assert_almost_equal((eu-e)/d, v[0][:,0], 5)
            assert_almost_equal((ed-e)/d, v[0][:,1], 5)
            #assert_almost_equal((eg-e)/d, v[1][:,0], 5)

    def test_get_os_baseline2(self):
        d = 1e-8
        grid = make_mesh(self.rho, self.zeta, self.x2)
        gridu = make_mesh(self.rho+d, self.zeta, self.x2)
        gridd = make_mesh(self.rho, self.zeta+d, self.x2)
        gridg = make_mesh(self.rho, self.zeta, self.x2+d)

        for t in [0, 1]:
            print('TYPE', t)
            e, dedn, dedzeta, dedx2 = get_os_baseline2(*grid, type=t)
            eu, _, _, _ = get_os_baseline2(*gridu, type=t)
            ed, _, _, _ = get_os_baseline2(*gridd, type=t)
            eg, _, _, _ = get_os_baseline2(*gridg, type=t)
            e *= grid[0][:]
            eu *= gridu[0][:]
            ed *= gridd[0][:]
            eg *= gridg[0][:]

            assert_almost_equal((eu-e)/d, dedn, 5)
            assert_almost_equal((ed-e)/d, dedzeta, 5)
            assert_almost_equal((eg-e)/d, dedx2, 5)

    def test_get_separate_xef_terms(self):
        f = np.linspace(0.7, 5, 200)
        res, dres = get_separate_xef_terms(f)
        res2, _ = get_separate_xef_terms(f+1e-8)
        assert_almost_equal((res2-res)/1e-8, dres)

    def test_get_chidesc_small(self):
        res, dres = get_chidesc_small(self.chi)
        res2, _ = get_chidesc_small(self.chi+1e-8)
        assert_almost_equal((res2-res)/1e-8, dres)

    def test_get_chi_full_deriv(self):
        N = self.zeta.shape[0]
        rho = np.ones(N)
        g2 = np.ones(N)
        tauw = g2 / (8 * rho)
        d = 1e-8
        for tau in [tauw, tauw+1, tauw+2]:
            chi, dchidrho, dchidzeta, dchidg2, dchidtau \
                = get_chi_full_deriv(rho, self.zeta, g2, tau)
            chirho, _, _, _, _ = get_chi_full_deriv(rho+d, self.zeta, g2, tau)
            chiz, _, _, _, _ = get_chi_full_deriv(rho, self.zeta+d, g2, tau)
            chig2, _, _, _, _ = get_chi_full_deriv(rho, self.zeta, g2+d, tau)
            chitau, _, _, _, _ = get_chi_full_deriv(rho, self.zeta, g2, tau+d)

            assert_almost_equal((chirho-chi)/d, dchidrho)
            assert_almost_equal((chiz-chi)/d, dchidzeta)
            assert_almost_equal((chig2-chi)/d, dchidg2)
            assert_almost_equal((chitau-chi)/d, dchidtau)
