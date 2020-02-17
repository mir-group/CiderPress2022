from pyscf import scf, dft, gto, ao2mo, df, lib, fci, cc
from pyscf.dft.numint import eval_ao, eval_rho
from pyscf.dft.gen_grid import Grids
from pyscf.pbc.tools.pyscf_ase import atoms_from_ase
from scipy.linalg.blas import dgemm
import numpy as np

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

def get_vele_mat_generator(mol, points, num_chunks, orb_vals, mo_coeff=None):
    get_generator = lambda: get_vele_mat_chunks(mol, points,
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
        vele_mat (3-dimensional array shape (nao, nao, N))
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
    vele_tmp.shape = (shape[0], shape[1], shape[2])
    tmp = np.einsum('pij,pj->pi', vele_tmp, orb_vals)
    Vele = np.einsum('pi,pi->p', tmp, orb_vals)
    return 0.5 * Vele


# The following functions are helpers that check whether vele_mat
# is a numpy array or a generator before passing to the methods
# above. This allows one to integrate the memory-saving (but slower)
# chunk-generating approach smoothly.

def get_ha_energy_density2(mol, rdm1, vele_mat, ao_vals):
    if isinstance(vele_mat, np.ndarray):
        return get_ha_energy_density(mol, rdm1, vele_mat, ao_vals)
    else:
        ha_energy_density = np.array([])
        for vele_mat_chunk, orb_vals_chunk in vele_mat():
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
        for vele_mat_chunk, orb_vals_chunk in mo_vele_mat():
            fx_energy_density = np.append(fx_energy_density,
                                    get_fx_energy_density(mol, mo_occ,
                                        vele_mat_chunk, orb_vals_chunk))
        return fx_energy_density

def get_ee_energy_density2(mol, rdm2, vele_mat, orb_vals):
    if isinstance(vele_mat, np.ndarray):
        return get_ee_energy_density(mol, rdm2, vele_mat, orb_vals)
    else:
        ee_energy_density = np.array([])
        for vele_mat_chunk, orb_vals_chunk in vele_mat():
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
    return (3.0 / (4 * np.pi * rho + 1e-7))**(1.0/3)

def get_gradient_magnitude(rho_data):
    return np.linalg.norm(rho_data[1:4,:], axis=0)

def get_normalized_grad(rho, mag_grad):
    sprefac = 2 * (3 * np.pi * np.pi)**(1.0/3)
    n43 = rho**(4.0/3)
    s = mag_grad / (sprefac * n43 + 1e-7)
    return s

def get_single_orbital_tau(rho, mag_grad):
    return mag_grad**2 / (8 * rho + 1e-7)

def get_uniform_tau(rho):
    return (3.0/10) * (3*np.pi**2)**(2.0/3) * rho**(5.0/3)

def get_normalized_tau(tau, tau_w, tau_unif):
    return (tau - tau_w) / (tau_unif + 1e-6)

def get_dft_input(rho_data):
    rho = rho_data[0,:]
    r_s = get_ws_radii(rho)
    mag_grad = get_gradient_magnitude(rho_data)
    s = get_normalized_grad(rho, mag_grad)
    tau_w = get_single_orbital_tau(rho, mag_grad)
    tau_unif = get_uniform_tau(rho)
    alpha = get_normalized_tau(rho_data[5], tau_w, tau_unif)
    return rho, s, alpha, tau_w, tau_unif

def get_vh(rho, rs, weights):
    return np.dot(rho / rs, weights)

def get_dvh(drho, rs, weights):
    return np.dot(drho / rs, weights)

def get_nonlocal_data(rho_data, tau_data, ws_radii, coords, weights):
    vals = []

    indexes = np.arange(weights.shape[0])
    for i in range(weights.shape[0]):
        ws_radius = ws_radii[i]
        weightsp = weights[indexes != i]
        vecs = coords - coords[i]
        vecsp = vecs[indexes != i]
        rs = np.linalg.norm(vecsp, axis=1)
        exp_weights = np.exp(- rs / ws_radius)
        drho = rho_data[1:4,indexes != i]
        dvh = get_dvh(drho, rs, weightsp)
        # r dot nabla rho
        rddrho = np.dot(vecsp, drho)
        # r dot nabla v_ha
        rddvh = np.dot(vecsp, dvh)
        rddvh_int = np.dot(weightsp, rddvh)
        rddrho_int = np.dot(weightsp, rddrho)
        dtau = tau_data[1:4,:]
        # r dot nabla tau
        rddtau = np.dot(vecs, dtau)
        rddtau_int = np.dot(weights, rddtau)

        vals.append([np.linalg.norm(dvh, axis=0), rddvh_int, rddrho_int, rddtau_int])

    return np.array(vals).transpose()

def squish_density(rho_data, coords, weights, alpha):
    new_coords = coords / alpha
    new_weights = weights / alpha**3
    rho_data = rho_data.copy()
    rho_data[0,:] *= alpha**3
    rho_data[1:4,:] *= alpha**4
    rho_data[4:6,:] *= alpha**5
    return new_coords, new_weights, rho_data

def regularize_nonlocal_data(nonlocal_data, rho, s, alpha):
    sprefac = 2 * (3 * np.pi * np.pi)**(1.0/3)
    n43 = rho**(4.0/3)
    nonlocal_data[0,:] /= tau_unif
    nonlocal_data[1,:] /= (sprefac * n43)
    nonlocal_data[3,:] /= tau_unif
    return nonlocal_data
