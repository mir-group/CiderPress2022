from mldftdat.utilf import utils as utilf
import numpy as np 
from scipy.special import sph_harm, genlaguerre
from numpy.testing import assert_almost_equal
from sympy.physics.hydrogen import Psi_nlm

theta = [0, np.pi /2, np.pi, 0.2, 1.1]
phi = [0, np.pi/2, np.pi/4, np.pi/3, np.pi * 1.123, np.pi * 0.764]

for t in theta:
    for p in phi:
        for l in range(5):
            for m in range(l+1):
                ref = sph_harm(m, l, p, t)
                if m == 0:
                    refp = ref
                    refm = ref
                else:
                    refp = np.sqrt(2) * (-1)**m * np.real(ref)
                    refm = np.sqrt(2) * (-1)**m * np.imag(ref)
                actp = utilf.ylm(l, m, np.cos(t), p)
                actm = utilf.ylm(l, -m, np.cos(t), p)
                #print(refp-actp, refm-actm)
                #print(l, m, t, p)
                assert_almost_equal(actp, refp)
                assert_almost_equal(actm, refm)

xs = np.linspace(0, 4)
act = 0 * xs
for n in range(5):
    for a in range(4):
        for i in range(xs.shape[0]):
            act[i] = utilf.laguerre(n, a, xs[i])
        ref = genlaguerre(n, a)(xs)
        assert_almost_equal(act, ref)

