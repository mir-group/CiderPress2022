from pyscf import scf, gto, ao2mo, df, lib, fci, cc
from pyscf.dft.numint import eval_ao, eval_rho
import numpy as np

def get_hartree_energy_density(mol, rdm1, points):
    auxmol = gto.fakemol_for_charges(points)
    Vele_mat = df.incore.aux_e2(mol, auxmol)
    Vele = np.einsum('ijp,ij->p', df.incore.aux_e2(mol, auxmol), rdm1)
    ao_vals = eval_ao(mol, points)
    rho = eval_rho(mol, ao_vals, rdm1)
    return 0.5 * Vele * rho

def get_ee_energy_density(mol, rdm2, points):
    auxmol = gto.fakemol_for_charges(points)
    Vele_mat = df.incore.aux_e2(mol, auxmol)
    Vele_tmp = 0 * Vele_mat
    ao_vals = eval_ao(mol, points).transpose()
    for i in range(rdm2.shape[0]):
        for j in range(rdm2.shape[1]):
            Vele_tmp[i,j,:] = np.einsum('ijp,ij->p', df.incore.aux_e2(mol, auxmol), rdm2[i,j,:,:])
    tmp = np.sum(Vele_tmp * ao_vals, axis=1)
    Vele = np.sum(tmp * ao_vals, axis=0)
    return 0.5 * Vele

mol = gto.Mole(atom='H 0 0 0; F 0 0 1.1', basis = 'ccpvdz')
mol.build()
myhf = scf.RHF(mol)
myhf.kernel()
#print(dir(myhf))
#print(dir(mol))
eri_tst = mol.intor('int2e', aosym='s4')
eri_tst2 = mol.intor('int2e_sph')
orb = myhf.mo_coeff
eri_4fold = ao2mo.kernel(mol, orb)
eri_tst3 = ao2mo.incore.full(eri_tst, orb)
print(eri_4fold.shape, myhf.mo_coeff.shape)
print(eri_tst.shape, eri_tst2.shape)
print(eri_tst3.shape)
print(mol._atm)
print(mol._bas)
print(mol._env)

he = gto.Mole(atom='He 0 0 0', basis='ccpvdz')
mol.build()
hf = scf.RHF(he).run()
ci = cc.CCSD(hf)
ci.kernel()
rdm1 = ci.make_rdm1()
rdm2 = ci.make_rdm2()
eeint = he.intor('int2e', aosym='s1')
print(np.trace(rdm1), np.sum(np.sum(eeint*rdm1, axis=(2,3)) * rdm1) / 2, np.sum(eeint*rdm2) / 2)
xs = np.linspace(-1.5, 1.5, 120)
dx = xs[1] - xs[0]
points = lib.cartesian_prod([xs, xs, xs])
ha = get_hartree_energy_density(he, rdm1, points)
ee = get_ee_energy_density(he, rdm2, points)
print(np.sum(ha) * dx**3, np.sum(ee) * dx**3)
#print(rdm2.shape)

xs = np.linspace(-1.5, 1.5, 30)
zs = np.linspace(-1.5, 4.0, 55)
points = lib.cartesian_prod([xs, xs, zs])
auxmol = gto.fakemol_for_charges(points)
Vele_mat = df.incore.aux_e2(mol, auxmol)
print(Vele_mat.shape)
Vele = points * 0
ao_vals = eval_ao(mol, points).transpose()
print(ao_vals.shape)
tmp = np.sum(Vele_mat * ao_vals, axis=1)
Vele = np.sum(tmp * ao_vals, axis=0)
print(Vele.shape)

#for i in range(Vele_mat.shape[2]):
#    for j in range(Vele_mat.shape[3]):
#        Vele[i,j] = np.sum(Vele_mat[:,:,i,j] * )

