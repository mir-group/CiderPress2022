from pyscf import scf, dft, gto
from pyscf.dft.gen_grid import Grids
import numpy as np
from mldftdat.pyscf_utils import *
from mldftdat.dft.utils import *
from numpy.testing import assert_almost_equal

LDA_FACTOR = - 3.0 / 4.0 * (3.0 / np.pi)**(1.0/3)

mol = gto.Mole(atom = 'He')
norm = 1.0 / (2 / np.pi)**(0.75) * 1e12
print(norm)
"""
mol.basis = {'He' : gto.basis.parse('''
He   S
     1.00000000     1.00000000
''')}
"""
mol.basis = {'He' : gto.basis.parse('''
BASIS "ao basis" PRINT
He    S
      1.0000E-16     1.000000
END
'''.format(norm))}
print(mol.basis)
mol.build()
rho_fac = 1
mol._env[mol._bas[:,6]] = rho_fac * np.sqrt(4 * np.pi)
grid = Grids(mol)
grid.build()
r = np.linalg.norm(grid.coords, axis = 1)
print(np.max(r), r.shape)
rho = rho_fac * np.ones(grid.weights.shape)
rho_data = np.zeros((6, rho.shape[0]))
# HEG
rho_data[0] = rho

rho, s, alpha, tau_w, tau_unif = get_dft_input2(rho_data)
alpha += 1
rho_data[5] = tau_unif
atm, bas, env, inv_rs, _ = get_gaussian_grid_b(grid.coords, rho[0] * 1.00, l = 0, s = s, alpha = alpha)
gridmol = gto.Mole(_atm = atm, _bas = bas, _env = env)
a = gridmol._env[gridmol._bas[:,5]]
norm = mol.intor('int1e_ovlp')
print(norm**0.5)
#ovlp = gto.mole.intor_cross('int1e_ovlp', mol, gridmol).transpose() 
ovlp = gto.mole.intor_cross('int1e_r2_origj', mol, gridmol).transpose() * inv_rs**2
proj = ovlp
print('proj', np.max(proj), np.max(inv_rs), np.max(a))

dfdg = 3.2 * np.ones(r.shape)
print('shape', dfdg.shape)
g = proj
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

print('hi', a[0] * fac, np.pi * fac * (rho_data[0] / 2)**(2.0/3))
vbas = project_xc_basis(ref_dedb, gridmol, grid, l=0)
print(np.linalg.norm(vbas - ref_val))
print(np.mean(vbas[r < 3]), np.max(vbas[r < 3]), np.min(vbas[r < 3]), np.min(ref_val))
assert_almost_equal(vbas[r < 3], ref_val[r < 3], 3)

v_npa = v_nonlocal(rho_data, grid, dfdg, density, mol, g, l = 0, mul = 1.0)
print(np.mean(v_npa[0][r < 3] - vbas[r < 3]), np.max(ref_dn))
print(np.mean(v_npa[0][r < 3]), np.max(ref_dn))
assert_almost_equal(v_npa[0][r < 3] - vbas[r < 3], ref_dn[r < 3], 2)
#assert_almost_equal(v_npa[1][r < 3], ref_dp[r < 3], 2)
assert_almost_equal(v_npa[3][r < 3], ref_dalpha[r < 3], 2)
