from mldftdat.analyzers import UHFAnalyzer
from pyscf.dft.libxc import eval_xc

analyzer = UHFAnalyzer.load('test_files/UHF_NO.hdf5')

exc, vxc, _, _ = eval_xc('SCAN', analyzer.rho_data, spin = 1, deriv = 1)
print(exc.shape)
print(vxc[0].shape, vxc[1].shape, vxc[3].shape)
