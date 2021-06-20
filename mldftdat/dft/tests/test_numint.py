from mldftdat.dft.numint import setup_rks_calc, setup_uks_calc
from mldftdat.dft.xc_models import NormGPFunctional
from pyscf import gto, dft
from numpy.testing import assert_almost_equal

B3LYP_SL_PART = '.08*SLATER + .72*B88, .81*LYP + .19*VWN'
SCAN0_SL_PART = '0.75*MGGA_X_SCAN + MGGA_C_SCAN_VV10'
BASIS = 'def2-tzvp'

h2o = gto.M(
    atom='''O    0.   0.       0.
            H    0.   -0.757   0.587
            H    0.   0.757    0.587
    ''',
    basis = BASIS,
    verbose=4
)
no = gto.M(atom='N 0 0 0; O 0 0 1.15', basis='def2-tzvp', spin=1)

print('loading CIDER')
mlfunc = NormGPFunctional.load('functionals/B3LYP_CIDER.yaml')
print('loaded CIDER')

class TestCIDER():

    def test_rks_uks(self):
        ks = dft.RKS(h2o)
        ks.xc = 'B3LYP'
        ks.grids.level = 1
        eref = ks.kernel()
        ks = setup_rks_calc(h2o, mlfunc, grid_level=1,
                            xc=B3LYP_SL_PART, xmix=0.2)
        etot = ks.kernel()
        ks = setup_uks_calc(h2o, mlfunc, grid_level=1,
                            xc=B3LYP_SL_PART, xmix=0.2)
        euks = ks.kernel()
        assert_almost_equal(etot, eref, 2)
        assert_almost_equal(euks, eref, 2)
        assert_almost_equal(euks, etot)

    def test_uks(self):
        ks = dft.UKS(no)
        ks.xc = 'B3LYP'
        ks.grids.level = 1
        eref = ks.kernel()
        ks = setup_uks_calc(no, mlfunc, grid_level=1,
                            xc=B3LYP_SL_PART, xmix=0.2)
        euks = ks.kernel()
        assert_almost_equal(euks, eref, 2)

    def test_vv10(self):
        ks = dft.RKS(h2o)
        ks.xc = SCAN0_SL_PART + ' + 0.25*HF'
        ks.xc = 'MGGA_X_SCAN,MGGA_C_SCAN_VV10'
        ks.nlc = 'VV10'
        ks.grids.level = 3
        ks.nlcgrids.level = 1
        eref = ks.kernel()
        ks.xc = '0.25*HF + 0.75*MGGA_X_SCAN + MGGA_C_SCAN_VV10'
        erefhyb = ks.kernel()
        ks = setup_rks_calc(h2o, mlfunc,
                            grid_level=3,
                            xc='MGGA_X_SCAN,MGGA_C_SCAN_VV10', xmix=0.00)
        ks.nlc = 'VV10'
        ks.nlcgrids.level = 1
        etot = ks.kernel()
        ks.xc = '0.75*MGGA_X_SCAN + MGGA_C_SCAN_VV10'
        ks._numint.mlfunc_x.xmix = 0.25
        ehyb = ks.kernel()
        ks = setup_uks_calc(h2o, mlfunc,
                            grid_level=3,
                            xc='0.75*MGGA_X_SCAN + MGGA_C_SCAN_VV10', xmix=0.25)
        ks.nlc = 'VV10'
        ks.nlcgrids.level = 1
        euhyb = ks.kernel()
        assert_almost_equal(etot, eref)
        assert_almost_equal(ehyb, erefhyb, 2)
        assert_almost_equal(euhyb, ehyb)



if __name__ == '__main__':
    ks = dft.RKS(h2o)
    ks.xc = 'B3LYP'
    ks.grids.level = 1
    eref = ks.kernel()
    ks = setup_rks_calc(h2o, mlfunc, grid_level=1,
                        xc=B3LYP_SL_PART, xmix=0.2)
    etot = ks.kernel()
    assert_almost_equal(etot, eref, 2)
