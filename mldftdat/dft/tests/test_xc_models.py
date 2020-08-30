from sklearn.gaussian_process import GaussianProcessRegressor as GPR 
from sklearn.gaussian_process.kernels import RBF
from mldftdat.dft.xc_models import Descriptor, GPFunctional
import numpy as np
from numpy.testing import assert_almost_equal

class TestGPFunctional():

    @classmethod
    def setup_class(cls):
        x = np.linspace(0, 2 * np.pi, 40)
        x_samp = np.linspace(0.0, 2 * np.pi, 1000)
        y = np.sin(x)
        y_samp = np.sin(x_samp)
        dy_samp = np.cos(x_samp)
        gpr = GPR()
        gpr.fit(x.reshape(-1,1), y)
        desc = Descriptor(0)
        alpha = gpr.alpha_ * gpr.kernel_.k1.constant_value
        X_train = x.reshape(-1, 1)
        kernel = RBF(np.array([gpr.kernel_.k2.length_scale]))
        gpf = GPFunctional(kernel, alpha, X_train, [desc], None)
        cls.x_samp = x_samp
        cls.y_samp = y_samp
        cls.dy_samp = dy_samp
        cls.gpf = gpf

    def test_get_F(self):
        y_pred = self.gpf.get_F(self.x_samp.reshape(-1,1)) - 1
        assert_almost_equal(y_pred, self.y_samp, 5)

    def test_get_derivative(self):
        dy_pred = self.gpf.get_derivative(self.x_samp.reshape(-1,1))
        assert_almost_equal(dy_pred.reshape(-1), self.dy_samp, 4)

