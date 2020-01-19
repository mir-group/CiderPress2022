from pyscf import scf, gto, ao2mo, df, lib, fci, cc
from pyscf.dft.numint import eval_ao, eval_rho
from pyscf.dft.gen_grid import Grids
import numpy as np

print("SCF DIR", scf.__file__)

def make_rdm2_from_rdm1(rdm1):
    rdm1copy = rdm1.copy()
    part1 = np.einsum('ij,kl->ijkl', rdm1, rdm1copy)
    part2 = np.einsum('lj,ki->ijkl', rdm1, rdm1copy)
    return part1 - 0.5 * part2

def get_hartree_energy_density(mol, rdm1, points):
    auxmol = gto.fakemol_for_charges(points)
    Vele_mat = df.incore.aux_e2(mol, auxmol)
    Vele = np.einsum('ijp,ij->p', Vele_mat, rdm1)
    ao_vals = eval_ao(mol, points)
    rho = eval_rho(mol, ao_vals, rdm1)
    return 0.5 * Vele * rho

def get_ee_energy_density(mol, rdm2, points):
    #mu,nu,lambda,sigma->i,j,k,l; r->p
    auxmol = gto.fakemol_for_charges(points)
    Vele_mat = np.ascontiguousarray(np.transpose(df.incore.aux_e2(mol, auxmol),axes=(2,0,1)))
    #Vele_mat = df.incore.aux_e2(mol, auxmol)
    ao_vals = eval_ao(mol, points)
    Vele_tmp = np.einsum('ijkl,pkl->pij', rdm2, Vele_mat)
    tmp = np.einsum('pij,pj->pi', Vele_tmp, ao_vals)
    Vele = np.einsum('pi,pi->p', tmp, ao_vals)
    return 0.5 * Vele

mol = gto.Mole(atom='H 0 0 0; F 0 0 1.1', basis = 'cc-pvtz')
#mol = gto.Mole(atom='H 0 0 0; F 0 0 1.1', basis = 'aug-cc-pvtz')
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
overlap = scf.hf.get_ovlp(mol)
mol_rdm1 = myhf.make_rdm1()
mol_rdm2 = make_rdm2_from_rdm1(mol_rdm1)
jmat, kmat = scf.hf.get_jk(mol, mol_rdm1)
print(np.sum(jmat * mol_rdm1), np.sum(kmat * mol_rdm1) / 2)
print(myhf.e_tot)

he = gto.Mole(atom='He 0 0 0', basis='cc-pvtz')
#he = gto.Mole(atom='He 0 0 0', basis='aug-cc-pvtz')
mol.build()
hf = scf.RHF(he).run()
ci = cc.CCSD(hf)
ci.kernel()
rdm1 = ci.make_rdm1()
orb = hf.mo_coeff
rdm1 = np.matmul(orb, np.matmul(rdm1, np.transpose(orb)))
rdm2 = ci.make_rdm2()
rdm2 = ao2mo.incore.full(rdm2, np.transpose(orb))
eeint = he.intor('int2e', aosym='s1')
#eeint = ao2mo.incore.full(eeint, hf.mo_coeff)
print("MOs")
print(np.trace(rdm1), np.sum(np.sum(eeint*rdm1, axis=(2,3)) * rdm1) / 2, np.sum(np.conj(np.transpose(eeint))*rdm2) / 2)

grid = Grids(he)
grid.kernel()
points = grid.coords
ha = get_hartree_energy_density(he, rdm1, points)
ee = get_ee_energy_density(he, rdm2, points)
print(np.dot(ha, grid.weights), np.dot(ee, grid.weights))
#print(rdm2.shape)

print("Starting rdm")
grid = Grids(mol)
grid.kernel()
points = grid.coords
print("NUM POINTS", points.shape)
ha = get_hartree_energy_density(mol, mol_rdm1, points)
ee = get_ee_energy_density(mol, mol_rdm2, points)
#print(np.sum(ha) * dx**3, np.sum(ee) * dx**3)
mat = np.matmul(overlap, mol_rdm1)
print(np.dot(myhf.mo_energy, myhf.mo_occ))
print(myhf.mo_occ, myhf.mo_energy, myhf.energy_elec(), myhf.energy_tot())
print(np.trace(mol_rdm1), np.trace(mat), np.trace(np.trace(mol_rdm2)), np.dot(ha, grid.weights), np.dot(ee, grid.weights))

