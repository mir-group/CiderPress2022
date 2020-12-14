from pyscf import scf, dft, gto, ao2mo, df, lib, fci, cc
from pyscf.dft.numint import eval_ao, eval_rho
from pyscf.dft.gen_grid import Grids
from pyscf.pbc.tools.pyscf_ase import atoms_from_ase
from scipy.linalg.blas import dgemm
import numpy as np
from mldftdat.utilf import utils as utilf

CALC_TYPES = {
    'RHF'   : scf.hf.RHF,
    'UHF'   : scf.uhf.UHF,
    'RKS'   : dft.rks.RKS,
    'UKS'   : dft.uks.UKS,
    'CCSD'  : cc.ccsd.CCSD,
    'UCCSD' : cc.uccsd.UCCSD
}

SCF_TYPES = {
    'RHF'  : scf.hf.RHF,
    'ROHF' : scf.rohf.ROHF,
    'UHF'  : scf.uhf.UHF,
    'RKS'  : dft.RKS,
    'UKS'  : dft.UKS
}

GG_SMUL = 1.0
GG_AMUL = 1.0
GG_AMIN = 1.0 / 18

def mol_from_ase(atoms, basis, spin = 0, charge = 0):
    """
    Get a pyscf gto.Mole object from an ase Atoms object (atoms).
    Assign it the atomic basis set (basis).
    Return the Mole object.
    """
    mol = gto.Mole()
    mol.atom = atoms_from_ase(atoms)
    mol.basis = basis
    mol.spin = spin
    mol.charge = charge
    mol.build()
    return mol

def run_scf(mol, calc_type, functional = None, remove_ld = False, dm0 = None):
    """
    Run an SCF calculation on a gto.Mole object (Mole)
    of a given calc_type in SCF_TYPES. Return the calc object.
    Note, if RKS or UKS is the calc_type, default functional is used.
    """
    if not calc_type in SCF_TYPES:
        raise ValueError('Calculation type must be in {}'.format(list(SCF_TYPES.keys())))

    calc = SCF_TYPES[calc_type](mol)
    if remove_ld:
        print("Removing linear dependence from overlap matrix")
        calc = scf.addons.remove_linear_dep_(calc)
    if 'KS' in calc_type and functional is not None:
        calc.xc = functional
        if functional == 'wB97M_V':
            print ('Specialized wB97M-V params')
            calc.nlc = 'VV10'
            calc.grids.prune = None
            calc.grids.level = 4
            if np.array([gto.charge(mol.atom_symbol(i)) <= 18 for i in range(mol.natm)]).all():
                calc.nlcgrids.prune = dft.gen_grid.sg1_prune
            else:
                calc.nlcgrids.prune = None
            calc.nlcgrids.level = 1

    calc.kernel(dm0 = dm0)
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

def get_gaussian_grid(coords, rho, l = 0, s = None, alpha = None):
    N = coords.shape[0]
    auxmol = gto.fakemol_for_charges(coords)
    atm = auxmol._atm.copy()
    bas = auxmol._bas.copy()
    start = auxmol._env.shape[0] - 2
    env = np.zeros(start + 2 * N)
    env[:start] = auxmol._env[:-2]
    bas[:,5] = start + np.arange(N)
    bas[:,6] = start + N + np.arange(N)

    a = np.pi * (rho / 2 + 1e-16)**(2.0 / 3)
    scale = 1
    fac = (6 * np.pi**2)**(2.0/3) / (16 * np.pi)
    if s is not None:
        scale += GG_SMUL * fac * s**2
    if alpha is not None:
        scale += GG_AMUL * 0.6 * fac * (alpha - 1)
    bas[:,1] = l
    ascale = a * scale
    cond = ascale < GG_AMIN
    ascale[cond] = GG_AMIN * np.exp(ascale[cond] / GG_AMIN - 1)
    env[bas[:,5]] = ascale
    print(np.sqrt(np.min(env[bas[:,5]])))
    #env[bas[:,6]] = np.sqrt(4 * np.pi) * (4 * np.pi * rho / 3)**(l / 3.0) * np.sqrt(scale)**l
    env[bas[:,6]] = np.sqrt(4 * np.pi**(1-l)) * (8 * np.pi / 3)**(l/3.0) * ascale**(l/2.0)

    return atm, bas, env

def get_gaussian_grid_c(coords, rho, l = 0, s = None, alpha = None):
    N = coords.shape[0]
    auxmol = gto.fakemol_for_charges(coords)
    atm = auxmol._atm.copy()
    bas = auxmol._bas.copy()
    start = auxmol._env.shape[0] - 2
    env = np.zeros(start + 2 * N)
    env[:start] = auxmol._env[:-2]
    bas[:,5] = start + np.arange(N)
    bas[:,6] = start + N + np.arange(N)
    ratio = alpha + 5./3 * s**2

    fac = 0.6 * (6 * np.pi**2)**(2.0/3) / (2 * np.pi)
    a = np.pi * (rho / 2 + 1e-16)**(2.0 / 3)
    #rp = (fac*ratio / (ratio + 1))
    amix = 0#1 / (ratio + 1)
    scale = 1 + ratio * fac
    #scale = amix + (1-amix) * ratio
    bas[:,1] = l
    ascale = a * scale
    cond = ascale < GG_AMIN
    ascale[cond] = GG_AMIN * np.exp(ascale[cond] / GG_AMIN - 1)
    env[bas[:,5]] = ascale
    print(np.sqrt(np.min(env[bas[:,5]])))
    #env[bas[:,6]] = np.sqrt(4 * np.pi) * (4 * np.pi * rho / 3)**(l / 3.0) * np.sqrt(scale)**l
    env[bas[:,6]] = fac**1.5 * np.sqrt(4 * np.pi**(1-l)) * (8 * np.pi / 3)**(l/3.0) * ascale**(l/2.0)

    return atm, bas, env

def get_gaussian_grid_b(coords, rho, l = 0, s = None, alpha = None):
    N = coords.shape[0]
    auxmol = gto.fakemol_for_charges(coords)
    atm = auxmol._atm.copy()
    bas = auxmol._bas.copy()
    start = auxmol._env.shape[0] - 2
    env = np.zeros(start + 2 * N)
    env[:start] = auxmol._env[:-2]
    bas[:,5] = start + np.arange(N)
    bas[:,6] = start + N + np.arange(N)

    a = np.pi * (rho / 2 + 1e-6)**(2.0 / 3)
    scale = 1
    #fac = (6 * np.pi**2)**(2.0/3) / (16 * np.pi)
    fac = (6 * np.pi**2)**(2.0/3) / (16 * np.pi)
    if s is not None:
        scale += fac * s**2
    if alpha is not None:
        scale += 3.0 / 5 * fac * (alpha - 1)
    bas[:,1] = l
    env[bas[:,5]] = a * scale
    env[bas[rho<1e-8,5]] = 1e16
    print(np.sqrt(np.min(env[bas[:,5]])))
    env[bas[:,6]] = np.sqrt(4 * np.pi) * (4 * np.pi * rho / 3)**(l / 3.0) * np.sqrt(scale)**l

    return atm, bas, env, (4 * np.pi * rho / 3)**(1.0 / 3), scale

def get_ha_total(rdm1, eeint):
    return np.sum(np.sum(eeint * rdm1, axis=(2,3)) * rdm1)

def get_hf_coul_ex_total(mol, hf):
    rdm1 = hf.make_rdm1()
    jmat, kmat = hf.get_jk(mol, rdm1)
    return np.sum(jmat * rdm1) / 2, -np.sum(kmat * rdm1) / 4

def get_hf_coul_ex_total2(rdm1, jmat, kmat):
    if len(rdm1.shape) == 2:
        return np.sum(jmat * rdm1) / 2, -np.sum(kmat * rdm1) / 4
    else:
        return np.sum(jmat * np.sum(rdm1, axis=0)) / 2, -np.sum(kmat * rdm1) / 2

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
        return np.array([part0, part1])

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
        return np.array([part00, part01, part11])

def get_ccsd_ee_total(mol, cccalc, hfcalc):
    """
    Get the total CCSD electron-electron repulsion energy.
    """
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

def get_ccsd_ee(rdm2, eeint):
    """
    Get the total CCSD electron-electron repulsion energy.
    """
    if len(rdm2.shape) == 4:
        return np.sum(eeint * rdm2) / 2
    else:
        if len(eeint.shape) == 4:
            eeint = [eeint] * 3
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
    """
    For a UHF calculation, return the 2-RDM from
    a given 1-RDM. Given D2(ijkl)=<psi| i+ k+ l j |psi>,
    and D(ij)=<psi| i+ j |psi>, then:
    For like spin, D2(ijkl) = D(ij) * D(kl) - D(lj) * D(ki).
    For opposite spin, D2(ijkl) = D(ij) * D(kl)
    Return D(uu,ijkl), D(ud,ijkl), D(dd,ijkl)
    """
    spinparts = []
    rdm1copy = rdm1.copy()
    for s in [0,1]:
        part1 = np.einsum('ij,kl->ijkl', rdm1[s], rdm1copy[s])
        part2 = np.einsum('lj,ki->ijkl', rdm1[s], rdm1copy[s])
        spinparts.append(part1 - part2)
    mixspinpart = np.einsum('ij,kl->ijkl', rdm1[0], rdm1copy[1])
    return np.array([spinparts[0], mixspinpart, spinparts[1]])

def get_ao_vals(mol, points):
    return eval_ao(mol, points)

def get_mgga_data(mol, grid, rdm1):
    """
    Get atomic orbital and density data.
    See eval_ao and eval_rho docs for details.
    Briefly, returns 0-3 derivatives of the atomic orbitals
    in ao_data;
    and the density, first derivatives of density,
    Laplacian of density, and kinetic energy density
    in rho_data.
    """
    ao_data = eval_ao(mol, grid.coords, deriv=3)
    if len(rdm1.shape) == 2:
        rho_data = eval_rho(mol, ao_data, rdm1, xctype='mGGA')
    else:
        part0 = eval_rho(mol, ao_data, rdm1[0], xctype='mGGA')
        part1 = eval_rho(mol, ao_data, rdm1[1], xctype='mGGA')
        rho_data = np.array([part0, part1])
    return ao_data, rho_data

def get_tau_and_grad_helper(mol, grid, rdm1, ao_data):
    """
    Passes the derivatives of the atomic orbitals
    to eval_rho to get the kinetic energy density and its
    derivatives. Not sure if this works.
    """
    # 0 1 2 3 4  5  6  7  8  9
    # 0 x y z xx xy xz yy yz zz
    aox = ao_data[[1, 4, 5, 6]]
    aoy = ao_data[[2, 5, 7, 8]]
    aoz = ao_data[[3, 6, 8, 9]]
    tau  = eval_rho(mol, aox, rdm1, xctype='GGA')
    tau += eval_rho(mol, aoy, rdm1, xctype='GGA')
    tau += eval_rho(mol, aoz, rdm1, xctype='GGA')
    return 0.5 * tau

def get_tau_and_grad(mol, grid, rdm1, ao_data):
    if len(rdm1.shape) == 2:
        return get_tau_and_grad_helper(mol, grid, rdm1, ao_data)
    else:
        return np.array([get_tau_and_grad_helper(mol, grid, rdm1[0], ao_data),\
                        get_tau_and_grad_helper(mol, grid, rdm1[1], ao_data)])

def get_rho_second_deriv_helper(mol, grid, dm, ao):
    from pyscf.dft.numint import _contract_rho, _dot_ao_dm
    from pyscf.dft.gen_grid import make_mask, BLKSIZE

    nao = mol.nao_nr()
    N = grid.weights.shape[0]
    non0tab = np.ones(((N+BLKSIZE-1)//BLKSIZE, mol.nbas),
                         dtype=np.uint8)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    c0 = _dot_ao_dm(mol, ao[0], dm, non0tab, shls_slice, ao_loc)
    c1 = np.zeros((3, N, nao))
    # 0 1 2 3 4  5  6  7  8  9
    # 0 x y z xx xy xz yy yz zz
    # - - - - 0  1  2  3  4  5
    # - - - - 11 12 13 22 23 33
    ddrho = np.zeros((6, N))
    alphas = [0, 0, 0, 1, 1, 2]
    betas =  [0, 1, 2, 1, 2, 2]
    for i in range(3):
        c1[i] = _dot_ao_dm(mol, ao[i+1], dm.T, non0tab, shls_slice, ao_loc)
    for i in range(6):
        term1 = _contract_rho(c0, ao[i + 4])
        term2 = _contract_rho(c1[alphas[i]], ao[betas[i]+1])
        total = term1 + term2
        ddrho[i] = total + total.conj()
    return ddrho

def get_rho_second_deriv(mol, grid, rdm1, ao_data):
    if len(rdm1.shape) == 2:
        return get_rho_second_deriv_helper(mol, grid, rdm1, ao_data)
    else:
        return np.array([get_rho_second_deriv_helper(mol, grid, rdm1[0], ao_data),\
                        get_rho_second_deriv_helper(mol, grid, rdm1[1], ao_data)])

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
    """
    Convert the return value of get_vele_mat to the MO basis.
    """
    if len(mo_coeff.shape) == 2:
        return np.matmul(mo_coeff.transpose(),
            np.matmul(vele_mat, mo_coeff))
    else:
        tmp = np.einsum('puv,svj->spuj', vele_mat, mo_coeff)
        return np.einsum('sui,spuj->spij', mo_coeff, tmp)

def get_vele_mat_chunks(mol, points, num_chunks, orb_vals, mo_coeff=None):
    """
    Generate chunks of vele_mat on the fly to reduce memory load.
    """
    num_pts = points.shape[0]
    for i in range(num_chunks):
        start = (i * num_pts) // num_chunks
        end = ((i+1) * num_pts) // num_chunks
        auxmol = gto.fakemol_for_charges(points[start:end])
        orb_vals_chunk = orb_vals[start:end]
        vele_mat_chunk = df.incore.aux_e2(mol, auxmol)
        vele_mat_chunk = np.ascontiguousarray(np.transpose(
                                vele_mat_chunk, axes=(2,0,1)))
        if mo_coeff is not None:
            vele_mat_chunk = get_mo_vele_mat(vele_mat_chunk, mo_coeff)
        yield vele_mat_chunk, orb_vals_chunk

def get_vele_mat_generator(mol, points, num_chunks, mo_coeff=None):
    get_generator = lambda orb_vals: get_vele_mat_chunks(mol, points,
                                num_chunks, orb_vals, mo_coeff)
    return get_generator

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

def get_ee_energy_density(mol, rdm2, vele_mat, orb_vals):
    """
    Get the electron-electron repulsion energy density for a system and basis set (mol),
    for a given molecular structure with basis set (mol).
    Returns the electron-electron repulsion energy.
    NOTE: vele_mat, rdm2, and orb_vals must be in the same basis! (AO or MO)
    Args:
        mol (gto.Mole)
        rdm2 (4-dimensional array shape (nao, nao, nao, nao))
        vele_mat (3-dimensional array shape (N, nao, nao))
        orb_vals (2D array shape (N, nao))

    The following script is equivalent and easier to read (but slower):

    Vele_tmp = np.einsum('ijkl,pkl->pij', rdm2, vele_mat)
    tmp = np.einsum('pij,pj->pi', Vele_tmp, orb_vals)
    Vele = np.einsum('pi,pi->p', tmp, orb_vals)
    return 0.5 * Vele
    """
    #mu,nu,lambda,sigma->i,j,k,l; r->p
    rdm2, shape = np.ascontiguousarray(rdm2).view(), rdm2.shape
    rdm2.shape = (shape[0] * shape[1], shape[2] * shape[3])
    vele_mat, shape = vele_mat.view(), vele_mat.shape
    vele_mat.shape = (shape[0], shape[1] * shape[2])
    vele_tmp = dgemm(1, vele_mat, rdm2, trans_b=True)
    vele_tmp.shape = (shape[0], orb_vals.shape[1], orb_vals.shape[1])
    tmp = np.einsum('pij,pj->pi', vele_tmp, orb_vals)
    Vele = np.einsum('pi,pi->p', tmp, orb_vals)
    return 0.5 * Vele

def get_ee_energy_density_split(rdm2, vele_mat, orb_vals1, orb_vals2):
    """
    Get the electron-electron repulsion energy density for a system.
    Returns the electron-electron repulsion energy.
    NOTE: vele_mat, rdm2, and orb_vals must be in the same basis! (AO or MO)
    This variant allows one to split the calculation into pieces to save
    memory
    Args:
        rdm2 (d1,d2,d3,d4)
        vele_mat (N,d3,d4)
        orb_vals1 (N,d1)
        orb_vals2 (N,d2)

    The following script is equivalent and easier to read (but slower):

    Vele_tmp = np.einsum('ijkl,pkl->pij', rdm2, vele_mat)
    tmp = np.einsum('pij,pj->pi', Vele_tmp, orb_vals)
    Vele = np.einsum('pi,pi->p', tmp, orb_vals)
    return 0.5 * Vele

    return \sum_{pqrs} dm2[p,q,r,s] * vele_mat[:,r,s] * mo_vals[:,p] * mo_vals[:,q]
    return \sum_{pqrs} < p^\dagger r^\dagger s q > * < r | |x-x'|^{-1} | s > 
                          * < x | p > * < x | q >
    Note: Assumes real input
    """
    #mu,nu,lambda,sigma->i,j,k,l; r->p
    rdm2, shape = np.ascontiguousarray(rdm2).view(), rdm2.shape
    rdm2.shape = (shape[0] * shape[1], shape[2] * shape[3])
    vele_mat, shape = np.ascontiguousarray(vele_mat).view(), vele_mat.shape
    vele_mat.shape = (shape[0], shape[1] * shape[2])
    vele_tmp = np.zeros((shape[0], orb_vals1.shape[1] * orb_vals2.shape[1]),
                        dtype=vele_mat.dtype)
    vele_tmp = dgemm(1, vele_mat, rdm2, c=vele_tmp,
                        overwrite_c=True, trans_b=True)
    vele_tmp.shape = (shape[0], orb_vals1.shape[1], orb_vals2.shape[1])
    tmp = np.einsum('pij,pj->pi', vele_tmp, orb_vals2)
    Vele = np.einsum('pi,pi->p', tmp, orb_vals1)
    return 0.5 * Vele

def get_lowmem_ee_energy(mycc, vele_mat, mo_vals, dm1 = None):
    """
    return \sum_{pqrs} dm2[p,q,r,s] * vele_mat[:,r,s] * mo_vals[:,p] * mo_vals[:,q]
    return \sum_{pqrs} < p^\dagger r^\dagger s q > * < r | |x-x'|^{-1} | s > 
                          * < x | p > * < x | q >
    Note: Assumes real input
    """
    from mldftdat.external.pyscf_ccsd_rdm import lowmem_ee_energy
    if isinstance(vele_mat, np.ndarray):
        return lowmem_ee_energy(mycc, mycc.t1, mycc.t2, mycc.l1, mycc.l2,
                                vele_mat, mo_vals, dm1=dm1)
    else:
        ee_energy_density = np.array([])
        for vele_mat_chunk, mo_vals_chunk in vele_mat(mo_vals):
            ee_energy_density = np.append(ee_energy_density,
                                    lowmem_ee_energy(mycc, mycc.t1, mycc.t2,
                                                    mycc.l1, mycc.l2,
                                                    vele_mat_chunk, mo_vals_chunk,
                                                    dm1 = dm1))
        return ee_energy_density

def get_corr_energy_density(mol, tau, vele_mat_ov, orbvals_occ,
                            orbvals_vir, direct = True):
    """
    Get the coupled-cluster correlation energy density for a system
    and basis set (mol).
    Args:
        tau (nocc1,nocc2,nvir1,nvir2)
        vele_mat_ov(gridsize,nocc2,nvir2)
        orbvals_occ (gridsize,nocc1)
        orbvals_occ (gridsize,nvir1)
    """
    if direct:
        vele_tmp = np.einsum('ijab,pjb->pia', tau, vele_mat_ov)
    else:
        vele_tmp = np.einsum('jiab,pjb->pia', tau, vele_mat_ov)
    vele_tmp = np.einsum('pjb,pb->pj', vele_tmp, orbvals_vir)
    vele = np.einsum('pj,pj->p', vele_tmp, orbvals_occ)

    return vele

    #e += 2 * numpy.einsum('ijab,iabj', tau, eris_ovvo)
    #e -=     numpy.einsum('jiab,iabj', tau, eris_ovvo)

def get_corr_density(tau, mo_to_aux_ov, direct = True):
    if direct:
        vele_tmp = np.einsum('ijab,pjb->pia', tau, mo_to_aux_ov)
    else:
        vele_tmp = np.einsum('jiab,pjb->pia', tau, mo_to_aux_ov)
    vele = np.einsum('pjb,qjb->pq', vele_tmp, mo_to_aux_ov)
    return vele


# The following functions are helpers that check whether vele_mat
# is a numpy array or a generator before passing to the methods
# above. This allows one to integrate the memory-saving (but slower)
# chunk-generating approach smoothly.

def get_ha_energy_density2(mol, rdm1, vele_mat, ao_vals):
    if isinstance(vele_mat, np.ndarray):
        return get_ha_energy_density(mol, rdm1, vele_mat, ao_vals)
    else:
        ha_energy_density = np.array([])
        for vele_mat_chunk, orb_vals_chunk in vele_mat(ao_vals):
            ha_energy_density = np.append(ha_energy_density,
                                    get_ha_energy_density(mol, rdm1,
                                        vele_mat_chunk, orb_vals_chunk))
        return ha_energy_density

def get_fx_energy_density2(mol, mo_occ, mo_vele_mat, mo_vals):
    # make sure to test that the grids end up the same
    if isinstance(mo_vele_mat, np.ndarray):
        return get_fx_energy_density(mol, mo_occ, mo_vele_mat, mo_vals)
    else:
        fx_energy_density = np.array([])
        for vele_mat_chunk, orb_vals_chunk in mo_vele_mat(mo_vals):
            fx_energy_density = np.append(fx_energy_density,
                                    get_fx_energy_density(mol, mo_occ,
                                        vele_mat_chunk, orb_vals_chunk))
        return fx_energy_density

def get_ee_energy_density2(mol, rdm2, vele_mat, orb_vals):
    if isinstance(vele_mat, np.ndarray):
        return get_ee_energy_density(mol, rdm2, vele_mat, orb_vals)
    else:
        ee_energy_density = np.array([])
        for vele_mat_chunk, orb_vals_chunk in vele_mat(orb_vals):
            ee_energy_density = np.append(ee_energy_density,
                                    get_ee_energy_density(mol, rdm2,
                                        vele_mat_chunk, orb_vals_chunk))
        return ee_energy_density


def mol_from_dict(mol_dict):
    for item in ['charge', 'spin', 'symmetry', 'verbose']:
        if type(mol_dict[item]).__module__ == np.__name__:
            mol_dict[item] = mol_dict[item].item()
    mol = gto.mole.unpack(mol_dict)
    mol.build()
    return mol

def get_scf(calc_type, mol, calc_data = None):
    calc = CALC_TYPES[calc_type](mol)
    calc.__dict__.update(calc_data)
    return calc

def get_ccsd(calc_type, mol, calc_data = None):
    if calc_type == 'CCSD':
        hf = scf.hf.RHF(mol)
    else:
        hf = scf.uhf.UHF(mol)
    hf.e_tot = calc_data.pop('e_tot') - calc_data['e_corr']
    calc = CALC_TYPES[calc_type](hf)
    calc.__dict__.update(calc_data)
    return calc

def load_calc(fname):
    analyzer_dict = lib.chkfile.load(fname, 'analyzer')
    mol = mol_from_dict(analyzer_dict['mol'])
    calc_type = analyzer_dict['calc_type']
    if 'CCSD' in calc_type:
        return get_ccsd(calc_type, mol, analyzer_dict['calc']), calc_type
    else:
        return get_scf(calc_type, mol, analyzer_dict['calc']), calc_type

def load_analyzer_data(fname):
    data_file = os.path.join(dirname, fname)
    return lib.chkfile.load(data_file, 'analyzer/data')

def get_ws_radii(rho):
    return (3.0 / (4 * np.pi * rho + 1e-16))**(1.0/3)

def get_gradient_magnitude(rho_data):
    return np.linalg.norm(rho_data[1:4,:], axis=0)

def get_normalized_grad(rho, mag_grad):
    sprefac = 2 * (3 * np.pi * np.pi)**(1.0/3)
    n43 = rho**(4.0/3)
    s = mag_grad / (sprefac * n43 + 1e-16)
    return s

def get_single_orbital_tau(rho, mag_grad):
    return mag_grad**2 / (8 * rho + 1e-16)

def get_uniform_tau(rho):
    return (3.0/10) * (3*np.pi**2)**(2.0/3) * rho**(5.0/3)

def get_normalized_tau(tau, tau_w, tau_unif):
    alpha = (tau - tau_w) / (tau_unif + 1e-4)
    return alpha**3 / (alpha**2 + 1e-3)

def get_dft_input(rho_data):
    rho = rho_data[0,:]
    r_s = get_ws_radii(rho)
    mag_grad = get_gradient_magnitude(rho_data)
    s = get_normalized_grad(rho, mag_grad)
    tau_w = get_single_orbital_tau(rho, mag_grad)
    tau_unif = get_uniform_tau(rho)
    alpha = get_normalized_tau(rho_data[5], tau_w, tau_unif)
    return rho, s, alpha, tau_w, tau_unif

def get_dft_input2(rho_data):
    rho = rho_data[0,:]
    r_s = get_ws_radii(rho)
    mag_grad = get_gradient_magnitude(rho_data)
    s = get_normalized_grad(rho, mag_grad)
    tau_w = get_single_orbital_tau(rho, mag_grad)
    tau_unif = get_uniform_tau(rho)
    alpha = (rho_data[5] - tau_w) / (tau_unif + 1e-16)
    return rho, s, alpha, tau_w, tau_unif

def get_vh(rho, rs, weights):
    return np.dot(rho / rs, weights)

def get_dvh(drho, rs, weights):
    return np.dot(drho / rs, weights)

def get_hartree_potential(rho_data, coords, weights):
    init_shape = rho_data.shape
    if len(init_shape) == 1:
        rho_data = rho_data.reshape((1, rho_data.shape[0]))
    print('getting hartree potential')
    return utilf.hartree_potential(rho_data, coords.transpose(),
                                   weights)[1].reshape(init_shape)

def get_nonlocal_data(rho_data, tau_data, ws_radii, coords, weights):
    coords = coords.transpose()
    vh_data = utilf.hartree_potential(rho_data, coords, weights)[1]
    if np.isnan(vh_data).any():
        raise ValueError('Part of vh_data is nan %d' % np.count_nonzero(np.isnan(vh_data)))
    print('getting nonlocal_data')
    return utilf.nonlocal_dft_data(rho_data[:4], tau_data[1:4],
                                   vh_data[1:4], ws_radii,
                                   coords, weights)[1]

def get_nonlocal_data_slow(rho_data, tau_data, ws_radii, coords, weights):
    vals = []
    rho = rho_data[0,:]
    drho = rho_data[1:4,:]
    vals = np.zeros((5, rho.shape[0]))
    dvh = np.zeros((rho.shape[0], 3))
    for i in range(weights.shape[0]):
        vecs = coords - coords[i]
        rs = np.linalg.norm(vecs, axis=1)
        rs[i] = (2.0/3) * (3 * weights[i] / (4 * np.pi))**(1.0 / 3)
        dvh[i,:] = get_dvh(drho, rs, weights)
    for i in range(weights.shape[0]):
        ws_radius = ws_radii[i]
        vecs = coords - coords[i]
        rs = np.linalg.norm(vecs, axis=1)
        exp_weights = np.exp(- rs / ws_radius) * weights
        # r dot nabla rho
        rddrho = np.einsum('pu,up->p', vecs, drho)
        # r dot nabla v_ha
        rddvh = np.einsum('pu,pu->p', vecs, dvh)
        rddvh_int = np.dot(exp_weights, rho * rddvh)
        rddrho_int = np.dot(exp_weights, rho * rddrho)
        dtau = tau_data[1:4,:]
        # r dot nabla tau
        rddtau = np.einsum('pu,up->p', vecs, dtau)
        rddtau_int = np.dot(exp_weights, rho * rddtau)
        rho_int = np.dot(exp_weights, rho)

        vals[:,i] = np.array([np.linalg.norm(dvh[i,:]), rddvh_int, rddrho_int,\
                              rddtau_int, rho_int])

    return vals

def squish_density(rho_data, coords, weights, alpha):
    new_coords = coords / alpha
    new_weights = weights / alpha**3
    rho_data = rho_data.copy()
    rho_data[0,:] *= alpha**3
    rho_data[1:4,:] *= alpha**4
    rho_data[4:6,:] *= alpha**5
    return new_coords, new_weights, rho_data

def squish_tau(tau_data, alpha):
    tau_data = tau_data.copy()
    tau_data[0,:] *= alpha**5
    tau_data[1:4] *= alpha**6
    return tau_data

def get_regularized_nonlocal_data(nonlocal_data, rho_data):
    """
    INPUT:
        0 : | nabla v_H |
        1 : \int exp(r'/r_s) n(r+r') r' dot nabla' v_H(r+r')
        2 : \int exp(r'/r_s) n(r+r') r' dot nabla' n(r+r')
        3 : \int exp(r'/r_s) n(r+r') r' dot nabla' tau(r+r')
        4 : \int exp(r'/r_s) n(r+r')
        5 : \int exp(r'/r_s) n(r+r')^(4/3)
        6 : \int exp(r'/r_s) n(r+r')^(5/3)
        7 : \int exp(r'/r_s) n(r+r')^2
    OUTPUT:
        0 : INPUT[0] * RHO / (INPUT[0] * RHO + TAU)
        1 : INPUT[1] / INPUT[5]
        2 : INPUT[2] / INPUT[7]
        3 : INPUT[3] / INPUT[6]
        4 : INPUT[4]
    """
    nonlocal_data = nonlocal_data.copy()
    rho = rho_data[0,:]
    ws_radii = get_ws_radii(rho)
    sprefac = 2 * (3 * np.pi * np.pi)**(1.0/3)
    n43 = rho**(4.0/3)
    tau_unif = get_uniform_tau(rho)
    mag_grad = get_gradient_magnitude(rho_data)
    tau_w = get_single_orbital_tau(rho, mag_grad)
    #nonlocal_data[0,:] /= np.sqrt(n43) + 1e-6
    ndvh = rho_data[0,:] * nonlocal_data[0,:]
    nonlocal_data[0,:] = ndvh / (ndvh + rho_data[5,:] + 1e-6)
    nonlocal_data[1,:] /= nonlocal_data[5,:] + 1e-6
    nonlocal_data[2,:] /= nonlocal_data[7,:] + 1e-6
    # TODO: below value is not normalized properly
    nonlocal_data[3,:] /= nonlocal_data[6,:] + 1e-6
    return nonlocal_data[:5,:]

import scipy
def get_proj(mol, grids):
    nao = mol.nao_nr()
    sn = np.zeros((nao, nao))
    ngrids = grids.coords.shape[0]
    blksize = dft.gen_grid.BLKSIZE
    for i0, i1 in lib.prange(0, ngrids, blksize):
        coords = grids.coords[i0:i1]
        ao = mol.eval_gto('GTOval', coords)
        wao = ao * grids.weights[i0:i1,None]
        sn += lib.dot(ao.T, wao)
    ovlp = mol.intor_symmetric('int1e_ovlp')
    proj = scipy.linalg.solve(sn, ovlp)
    return proj

def get_normalized_rho_integration(ks):
    T = type(ks._numint)
    class NormNumint(T):
        def __init__(self):
            super(T, self).__init__()
            self.proj = get_proj(ks.mol, ks.grids)
            self.omega = ks._numint.omega

        def _gen_rho_evaluator(self, mol, dms, hermi=0):
            if isinstance(dms, np.ndarray) and dms.ndim == 2:
                dms = [dms]
            if not hermi:
                # For eval_rho when xctype==GGA, which requires hermitian DMs
                dms = [(dm+dm.conj().T)*.5 for dm in dms]
            nao = dms[0].shape[0]
            ndms = len(dms)
            def make_rho(idm, ao, non0tab, xctype):
                return self.eval_rho(mol, ao, dms[idm], non0tab, xctype, hermi=1)
            return make_rho, ndms, nao

        def eval_rho(self, mol, ao, dm, non0tab=None, xctype='LDA', hermi=0, verbose=None):
            rho_exact = eval_rho(mol, ao, dm, non0tab, xctype, hermi, verbose)
            proj_dm = lib.einsum('ki,ij->kj', self.proj, dm)
            rho_scale = eval_rho(mol, ao, proj_dm, non0tab, xctype, hermi, verbose)
            return lib.tag_array(rho_scale, rho_exact=rho_exact)

        def eval_xc(self, xc_code, rho, spin=0, relativity=0, deriv=1, omega=None,
                    verbose=None):
            if omega is None: omega = self.omega
            return self.libxc.eval_xc(xc_code, rho.rho_exact, spin, relativity, deriv,
                                      omega, verbose)
    ks._numint = NormNumint()
    return ks

