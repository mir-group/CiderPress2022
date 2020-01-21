from pyscf import scf, dft, gto, ao2mo, df, lib, fci, cc
from pyscf.dft.numint import eval_ao, eval_rho
from pyscf.dft.gen_grid import Grids
from pyscf.pbc.tools.pyscf_ase import atoms_from_ase
import numpy as np

SCF_TYPES = {
    'RHF': scf.hf.RHF,
    'UHF': scf.uhf.UHF,
    'RKS': dft.RKS,
    'UKS': dft.UKS
}

def mol_from_ase(atoms, basis):
    """
    Get a pyscf gto.Mole object from an ase Atoms object (atoms).
    Assign it the atomic basis set (basis).
    Return the Mole object.
    """
    mol = gto.Mole()
    mol.atom = atoms_from_ase(atoms)
    mol.basis = basis
    mol.build()
    return mol

def run_scf(mol, calc_type):
    """
    Run an SCF calculation on a gto.Mole object (Mole)
    of a given calc_type in SCF_TYPES. Return the calc object.
    """
    if not calc_type in SCF_TYPES:
        raise ValueError('Calculation type must be in {}'.format(list(SCF_TYPES.keys())))
    calc = SCF_TYPES[calc_type](mol)
    calc.kernel()
    return calc

def run_cc(hf):
    """
    Run and return a restricted CCSD calculation on mol,
    with HF molecular orbital coefficients in the RHF object hf.
    """
    if type(hf) == SCF_TYPES['RHF']:
        calc_cls = cc.CCSD
    elif type(hf) == SCF_TYPES['UHF']:
        calc_cls = cc.UCCSD
    else:
        raise NotImplementedError('HF type {} not supported'.format(type(hf)) +\
            '\nSupported Types: {}'.format(SCF_TYPES['RHF'], SCF_TYPES['UHF']))
    calc = calc_cls(hf)
    calc.kernel()
    return calc

def get_grid(mol):
    """
    Get the real-space grid of a molecule for numerical integration.
    """
    grid = Grids(mol)
    grid.kernel()
    return grid

def get_ha_total(rdm1, eeint):
    return np.sum(np.sum(eeint * rdm1, axis=(2,3)) * rdm1)

def get_hf_coul_ex_total(mol, hf):
    rdm1 = hf.make_rdm1()
    jmat, kmat = hf.get_jk(mol, rdm1)
    return np.sum(jmat * rdm1) / 2, -np.sum(kmat * rdm1) / 4

def get_hf_coul_ex_total_unrestricted(mol, hf):
    rdm1 = hf.make_rdm1()
    jmat, kmat = hf.get_jk(mol, rdm1)
    return np.sum(jmat * np.sum(rdm1, axis=0)) / 2, -np.sum(kmat * rdm1) / 2

def transform_basis_1e(mat, coeff):
    """
    Transforms the 1-electron matrix mat into the basis
    described by coeff (with the basis vectors being the columns).
    To transform AO operator to MO operator, pass mo_coeff.
    To transform MO operator to AO operator, pass inv(mo_coeff).
    To transform AO density matrix to MO density matrix, pass inv(transpose(mo_coeff)).
    To transform MO density matrix to AO density matrix, pass transpose(mo_coeff).
    """
    if len(coeff.shape) == 2:
        return np.matmul(coeff.transpose(), np.matmul(mat, coeff))
    else:
        if len(coeff) != 2 or len(mat) != 2:
            raise ValueError('Need two sets of orbitals, two mats for unrestricted case.')
        part0 = np.matmul(coeff[0].transpose(), np.matmul(mat[0], coeff[0]))
        part1 = np.matmul(coeff[1].transpose(), np.matmul(mat[1], coeff[1]))
        return (part0, part1)

def transform_basis_2e(eri, coeff):
    """
    Transforms the 2-electron matrix eri into the basis
    described by coeff (with the basis vectors being the columns).
    See transform_basis_1e for how to do different transformations.
    """
    if len(coeff.shape) == 2:
        return ao2mo.incore.full(eri, coeff)
    else:
        if len(coeff) != 2 or len(eri) != 3:
            raise ValueError('Need two sets of orbitals, three eri tensors for unrestricted case.')
        set00 = [coeff[0]] * 4
        set11 = [coeff[1]] * 4
        set01 = set00[:2] + set11[:2]
        part00 = ao2mo.incore.general(eri[0], set00)
        part01 = ao2mo.incore.general(eri[1], set01)
        part11 = ao2mo.incore.general(eri[2], set11)
        return (part00, part01, part11)

def get_ccsd_ee_total(mol, cccalc, hfcalc):
    rdm2 = cccalc.make_rdm2()
    eeint = mol.intor('int2e', aosym='s1')
    if len(hfcalc.mo_coeff.shape) == 2:
        eeint = transform_basis_2e(eeint, hfcalc.mo_coeff)
        return np.sum(eeint * rdm2) / 2
    else:
        eeint = transform_basis_2e([eeint] * 3, hfcalc.mo_coeff)
        return 0.5 * np.sum(eeint[0] * rdm2[0])\
                + np.sum(eeint[1] * rdm2[1])\
                + 0.5 * np.sum(eeint[2] * rdm2[2])

integrate_on_grid = np.dot

def make_rdm2_from_rdm1(rdm1):
    """
    For an RHF calculation, return the 2-RDM from
    a given 1-RDM. Given D2(ijkl)=<psi| i+ k+ l j |psi>,
    and D(ij)=<psi| i+ j |psi>, then
    D2(ijkl) = D(ij) * D(kl) - 0.5 * D(lj) * D(ki)
    """
    rdm1copy = rdm1.copy()
    part1 = np.einsum('ij,kl->ijkl', rdm1, rdm1copy)
    part2 = np.einsum('lj,ki->ijkl', rdm1, rdm1copy)
    return part1 - 0.5 * part2

def make_rdm2_from_rdm1_unrestricted(rdm1):
    spinparts = []
    rdm1copy = rdm1.copy()
    for s in [0,1]:
        part1 = np.einsum('ij,kl->ijkl', rdm1[s], rdm1copy[s])
        part2 = np.einsum('lj,ki->ijkl', rdm1[s], rdm1copy[s])
        spinparts.append(part1 - part2)
    mixspinpart = np.einsum('ij,kl->ijkl', rdm1[0], rdm1copy[1])
    return spinparts[0], mixspinpart, spinparts[1]

def get_ao_vals(mol, points):
    return eval_ao(mol, points)

def get_vele_mat(mol, points):
    """
    Return shape (N, nao, nao)
    """
    auxmol = gto.fakemol_for_charges(points)
    vele_mat = df.incore.aux_e2(mol, auxmol)
    return np.ascontiguousarray(np.transpose(vele_mat, axes=(2,0,1)))

def get_mo_vals(ao_vals, mo_coeff):
    """
    Args:
        ao_vals shape (N,nao)
        mo_coeff shape (nao,nao)
    Returns
        shape (N,nao)
    """
    return np.matmul(ao_vals, mo_coeff)

def get_mo_vele_mat(vele_mat, mo_coeff):
    return np.matmul(mo_coeff.transpose(),
            np.matmul(vele_mat, mo_coeff))

def get_mo_vele_mat_unrestricted(vele_mat, mo_coeff):
    """
    Args:
        ao_vals shape (N,nao,nao)
        mo_coeff shape (nao,nao)
    Returns
        shape (N,nao,nao)
    """
    tmp = np.einsum('puv,svj->spuj', vele_mat, mo_coeff)
    return np.einsum('sui,spuj->spij', mo_coeff, tmp)

def get_ha_energy_density(mol, rdm1, vele_mat, ao_vals):
    """
    Get the classical Hartree energy density on a real-space grid,
    for a given molecular structure with basis set (mol),
    for a given 1-electron reduced density matrix (rdm1).
    Returns the Hartree energy density.
    """
    if len(rdm1.shape) == 2:
        Vele = np.einsum('pij,ij->p', vele_mat, rdm1)
    else:
        rdm1 = np.array(rdm1)
        Vele = np.einsum('pij,sij->p', vele_mat, rdm1)
    rho = eval_rho(mol, ao_vals, rdm1)
    return 0.5 * Vele * rho

def get_fx_energy_density(mol, mo_occ, mo_vele_mat, mo_vals):
    """
    Get the Hartree Fock exchange energy density on a real-space grid,
    for a given molecular structure with basis set (mol),
    for a given atomic orbital (AO) 1-electron reduced density matrix (rdm1).
    Returns the exchange energy density, which is negative.
    """
    A = mo_occ * mo_vals
    tmp = np.einsum('pi,pij->pj', A, mo_vele_mat)
    return -0.25 * np.sum(A * tmp, axis=1)

"""
def get_ee_energy_density(mol, rdm2, vele_mat, ao_vals):
    vele_mat = np.ascontiguousarray(np.transpose(vele_mat, axes=(2,0,1)))
    vele_mat = vele_mat.view()
    shape = vele_mat.shape
    vele_mat.shape = (shape[0] * shape[1], shape[2] * shape[3])
    rdm2 = rdm2.view()
    shape = rdm2.shape
    rdm2.shape = (shape[0], shape[1] * shape[2])
    tmp = np.dot(vele_mat, rdm2)

    Vele_tmp = np.einsum('ij,pkl->pij', rdm2, vele_mat)
    tmp = np.einsum('pij,pj->pi', Vele_tmp, ao_vals)
    Vele = np.einsum('pi,pi->p', tmp, ao_vals)
    return 0.5 * Vele
"""

def get_ee_energy_density(mol, rdm2, vele_mat, orb_vals):
    """
    Get the electron-electron repulsion energy density for a system and basis set (mol),
    for a given molecular structure with basis set (mol).
    Returns the electron-electron repulsion energy.
    NOTE: vele_mat, rdm2, and orb_vals must be in the same basis! (AO or MO)
    Args:
        mol (gto.Mole)
        rdm2 (4-dimensional array shape (nao, nao, nao, nao))
        vele_mat (3-dimensional array shape (nao, nao, N))
        orb_vals (2D array shape (N, nao))
    """
    #mu,nu,lambda,sigma->i,j,k,l; r->p
    Vele_tmp = np.einsum('ijkl,pkl->pij', rdm2, vele_mat)
    tmp = np.einsum('pij,pj->pi', Vele_tmp, orb_vals)
    Vele = np.einsum('pi,pi->p', tmp, orb_vals)
    return 0.5 * Vele
