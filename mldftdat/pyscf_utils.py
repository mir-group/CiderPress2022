from pyscf import scf, dft, gto, ao2mo, df, lib, cc
from pyscf.dft.numint import eval_ao, eval_rho
from pyscf.dft.gen_grid import Grids
from pyscf.pbc.tools.pyscf_ase import atoms_from_ase
import numpy as np
import logging


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

########################################################
# BASIC HELPER ROUTINES FOR RUNNING PYSCF CALCULATIONS #
########################################################

def mol_from_ase(atoms, basis, spin=0, charge=0):
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

def setup_rks_calc(mol, xc, grid_level=3, vv10=False, **kwargs):
    """
    Set up a PySCF RKS calculation with sensible defaults.
    """
    rks = dft.RKS(mol)
    rks.xc = xc
    rks.grids.level = grid_level
    rks.grids.build()
    logging.warning('xc: {}, grid level: {}'.format(xc, grid_level))
    if vv10:
        logging.warning('Using VV10 in UKS setup')
        rks.nlc = 'VV10'
        if np.array([gto.charge(mol.atom_symbol(i)) <= 18 for i in range(mol.natm)]).all():
            rks.nlcgrids.prune = dft.gen_grid.sg1_prune
        else:
            rks.nlcgrids.prune = None
        rks.nlcgrids.level = 1
    return rks

def setup_uks_calc(mol, xc, grid_level=3, vv10=False, **kwargs):
    """
    Set up a PySCF UKS calculation with sensible defaults.
    """
    uks = dft.UKS(mol)
    uks.xc = xc
    uks.grids.level = grid_level
    uks.grids.build()
    logging.warning('xc: {}, grid level: {}'.format(xc, grid_level))
    if vv10:
        logging.warning('Using VV10 in UKS setup')
        uks.nlc = 'VV10'
        if np.array([gto.charge(mol.atom_symbol(i)) <= 18 for i in range(mol.natm)]).all():
            uks.nlcgrids.prune = dft.gen_grid.sg1_prune
        else:
            uks.nlcgrids.prune = None
        uks.nlcgrids.level = 1
    return uks

def run_scf(mol, calc_type, functional=None, remove_ld=False, dm0=None):
    """
    Run an SCF calculation on a gto.Mole object (Mole)
    of a given calc_type in SCF_TYPES. Return the calc object.
    Note, if RKS or UKS is the calc_type, default functional is used.
    """
    if not calc_type in SCF_TYPES:
        raise ValueError('Calculation type must be in {}'.format(list(SCF_TYPES.keys())))

    calc = SCF_TYPES[calc_type](mol)
    if remove_ld:
        logging.info("Removing linear dependence from overlap matrix")
        calc = scf.addons.remove_linear_dep_(calc)
    if 'KS' in calc_type and functional is not None:
        calc.xc = functional
        if 'MN' in functional:
            logging.info('MN grid level 4')
            calc.grids.level = 4
        if functional == 'wB97M_V':
            logging.info('Using Specialized wB97M-V params')
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



#############################################
# HELPER FUNCTIONS FOR THE analyzers MODULE #
#############################################


def get_grid(mol, level=3):
    """
    Get the real-space grid of a molecule for numerical integration.
    """
    grid = Grids(mol)
    grid.level = level
    grid.kernel()
    return grid

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




##################################################
# HELPER FUNCTIONS FOR COMPUTING DFT INGREDIENTS #
##################################################


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

def get_regularized_tau(tau, tau_w, tau_unif):
    alpha = (tau - tau_w) / (tau_unif + 1e-4)
    return alpha**3 / (alpha**2 + 1e-3)

def get_normalized_tau(tau, tau_w, tau_unif):
    return (tau - tau_w) / (tau_unif + 1e-16)

def get_dft_input(rho_data):
    rho = rho_data[0,:]
    r_s = get_ws_radii(rho)
    mag_grad = get_gradient_magnitude(rho_data)
    s = get_normalized_grad(rho, mag_grad)
    tau_w = get_single_orbital_tau(rho, mag_grad)
    tau_unif = get_uniform_tau(rho)
    alpha = get_regularized_tau(rho_data[5], tau_w, tau_unif)
    return rho, s, alpha, tau_w, tau_unif

def get_dft_input2(rho_data):
    rho = rho_data[0,:]
    r_s = get_ws_radii(rho)
    mag_grad = get_gradient_magnitude(rho_data)
    s = get_normalized_grad(rho, mag_grad)
    tau_w = get_single_orbital_tau(rho, mag_grad)
    tau_unif = get_uniform_tau(rho)
    alpha = get_normalized_tau(rho_data[5], tau_w, tau_unif)
    return rho, s, alpha, tau_w, tau_unif

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
