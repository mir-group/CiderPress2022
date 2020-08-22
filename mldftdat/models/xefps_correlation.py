from mldftdat.lowmem_analyzers import RHFAnalyzer, UHFAnalyzer, CCSDAnalyzer, UCCSDAnalyzer
from mldftdat.dft.numint3 import setup_uks_calc, setup_rks_calc
from mldftdat.dft.numint5 import _eval_x_0, setup_aux
from pyscf.dft.libxc import eval_xc
from mldftdat.dft.correlation import *
from mldftdat.workflow_utils import get_save_dir
from sklearn.linear_model import LinearRegression
from pyscf.dft.numint import NumInt
import os
import numpy as np

LDA_FACTOR = - 3.0 / 4.0 * (3.0 / np.pi)**(1.0/3)

DEFAULT_FUNCTIONAL = 'SCAN'
DEFAULT_BASIS = 'aug-cc-pvtz'

def get_mlx_contribs(dft_dir, restricted, mlfunc):

    if restricted:
        dft_analyzer = RHFAnalyzer.load(dft_dir + '/data.hdf5')
        rhot = dft_analyzer.rho_data[0]
    else:
        dft_analyzer = UHFAnalyzer.load(dft_dir + '/data.hdf5')
        rhot = dft_analyzer.rho_data[0][0] + dft_analyzer.rho_data[1][0]

    rho_data = dft_analyzer.rho_data
    weights = dft_analyzer.grid.weights
    grid = dft_analyzer.grid
    spin = dft_analyzer.mol.spin
    mol = dft_analyzer.mol
    rdm1 = dft_analyzer.rdm1
    E_pbe = dft_analyzer.e_tot

    auxmol, ao_to_aux = setup_aux(mol, 0)
    mol.ao_to_aux = ao_to_aux
    mol.auxmol = auxmol

    if restricted:
        rho_data_u, rho_data_d = rho_data, rho_data
    else:
        rho_data_u, rho_data_d = rho_data[0], rho_data[1]

    FUNCTIONAL = ',MGGA_C_SCAN'
    cu = eval_xc(FUNCTIONAL, (rho_data_u, 0 * rho_data_u), spin = 1)[0] * rho_data_u[0]
    cd = eval_xc(FUNCTIONAL, (rho_data_d, 0 * rho_data_d), spin = 1)[0] * rho_data_d[0]
    co = eval_xc(FUNCTIONAL, (rho_data_u, rho_data_d), spin = 1)[0] \
            * (rho_data_u[0] + rho_data_d[0])
    co -= cu + cd

    numint0 = ProjNumInt(xterms = [], ssterms = [], osterms = [])
    if restricted:
        ex = _eval_x_0(mlfunc, mol, rho_data, grid, rdm1)[0]
        exu = ex
        exd = ex
        exo = ex
        rhou = rho_data[0]
        rhod = rho_data[0]
        rhot = rho_data[0]
        Ex = np.dot(exo * rhot, weights)
    else:
        exu = _eval_x_0(mlfunc, mol, 2 * rho_data[0], grid, 2 * rdm1[0])[0]
        exd = _eval_x_0(mlfunc, mol, 2 * rho_data[1], grid, 2 * rdm1[1])[0]
        rhou = 2 * rho_data[0][0]
        rhod = 2 * rho_data[1][0]
        rhot = rho_data[0][0] + rho_data[1][0]
        exo = (exu * rho_data[0][0] + exd * rho_data[1][0])
        Ex = np.dot(exo, weights)
        exo /= (rhot + 1e-20)

    Eterms = np.array([Ex])

    for rho, ex, c in zip([rhou, rhod, rhot], [exu, exd, exo], [cu, cd, co]):
        elda = LDA_FACTOR * rho[0]**(1.0/3) - 1e-20
        Fx = ex / elda
        Etmp = np.zeros(5)
        x1 = 1 / (1 + Fx) - 0.5
        for i in range(5):
            Etmp[i] = np.dot(c * x1**i, weights)
        Eterms = np.append(Eterms, Etmp)

    print(Eterms.shape)

    # Eterms = Eu, Ed, Eo
    return Eterms

def store_mlx_contribs_dataset(FNAME, ROOT, MOL_IDS, IS_RESTRICTED_LIST, MLFUNC):

    X = np.zeros([0,16])

    for mol_id, is_restricted in zip(MOL_IDS, IS_RESTRICTED_LIST):

        print(mol_id)

        if is_restricted:
            dft_dir = get_save_dir(ROOT, 'RKS', DEFAULT_BASIS,
                mol_id, functional = DEFAULT_FUNCTIONAL)
        else:
            dft_dir = get_save_dir(ROOT, 'UKS', DEFAULT_BASIS,
                mol_id, functional = DEFAULT_FUNCTIONAL)

        sl_contribs = get_mlx_contribs(dft_dir, is_restricted, MLFUNC)

        X = np.vstack([X, sl_contribs])

    np.save(FNAME, X)


def get_etot_contribs(dft_dir, ccsd_dir, restricted):

    if restricted:
        dft_analyzer = RHFAnalyzer.load(dft_dir + '/data.hdf5')
        ccsd_analyzer = CCSDAnalyzer.load(ccsd_dir + '/data.hdf5')
    else:
        dft_analyzer = UHFAnalyzer.load(dft_dir + '/data.hdf5')
        ccsd_analyzer = UCCSDAnalyzer.load(ccsd_dir + '/data.hdf5')

    E_pbe = dft_analyzer.e_tot
    E_ccsd = ccsd_analyzer.e_tot + ccsd_analyzer.e_tri

    return np.array([E_pbe, E_ccsd])

def store_total_energies_dataset(FNAME, ROOT, MOL_IDS, IS_RESTRICTED_LIST):

    # PBE, CCSD
    y = np.zeros([0, 2])

    for mol_id, is_restricted in zip(MOL_IDS, IS_RESTRICTED_LIST):

        if is_restricted:
            dft_dir = get_save_dir(ROOT, 'RKS', DEFAULT_BASIS,
                                   mol_id, functional = DEFAULT_FUNCTIONAL)
            ccsd_dir = get_save_dir(ROOT, 'CCSD', DEFAULT_BASIS, mol_id)
        else:
            dft_dir = get_save_dir(ROOT, 'UKS', DEFAULT_BASIS,
                                   mol_id, functional = DEFAULT_FUNCTIONAL)
            ccsd_dir = get_save_dir(ROOT, 'UCCSD', DEFAULT_BASIS, mol_id)

        dft_ccsd = get_etot_contribs(dft_dir, ccsd_dir, is_restricted)

        y = np.vstack([y, dft_ccsd])

    np.save(FNAME, y)


def get_vv10_contribs(dft_dir, restricted, NLC_COEFS):

    if restricted:
        dft_analyzer = RHFAnalyzer.load(dft_dir + '/data.hdf5')
        rhot = dft_analyzer.rho_data[0]
        rdm1_nsp = dft_analyzer.rdm1
    else:
        dft_analyzer = UHFAnalyzer.load(dft_dir + '/data.hdf5')
        rhot = dft_analyzer.rho_data[0][0] + dft_analyzer.rho_data[1][0]
        rdm1_nsp = dft_analyzer.rdm1[0] + dft_analyzer.rdm1[1]

    rho_data = dft_analyzer.rho_data
    weights = dft_analyzer.grid.weights
    grid = dft_analyzer.grid
    spin = dft_analyzer.mol.spin
    mol = dft_analyzer.mol
    rdm1 = dft_analyzer.rdm1
    E_pbe = dft_analyzer.e_tot
    numint = NumInt()

    grid.level = 2
    grid.buil()

    vv10_contribs = []

    for b_test, c_test in NLC_COEFS:

        _, Evv10, _ = nr_rks_vv10(numint, mol, grid, None, rdm1_nsp, b = b_test, c = c_test)

        vv10_contribs.append(Evv10)

    return np.array(vv10_contribs)

DEFAULT_NLC_COEFS = [[5.9, 0.0093], [6.0, 0.01], [6.3, 0.0089],\
                     [9.8, 0.0093], [14.0, 0.0093], [15.7, 0.0093]]

def store_vv10_contribs_dataset(FNAME, ROOT, MOL_IDS, IS_RESTRICTED_LIST,
                                NLC_COEFS=DEFAULT_NLC_COEFS):

    X = np.zeros([0, len(NLC_COEFS)])

    for mol_id, is_restricted in zip(MOL_IDS, IS_RESTRICTED_LIST):

        print(mol_id)

        if is_restricted:
            dft_dir = get_save_dir(ROOT, 'RKS', DEFAULT_BASIS,
                                   mol_id, functional = DEFAULT_FUNCTIONAL)
            ccsd_dir = get_save_dir(ROOT, 'CCSD', DEFAULT_BASIS, mol_id)
        else:
            dft_dir = get_save_dir(ROOT, 'UKS', DEFAULT_BASIS,
                                   mol_id, functional = DEFAULT_FUNCTIONAL)
            ccsd_dir = get_save_dir(ROOT, 'UCCSD', DEFAULT_BASIS, mol_id)

        vv10_contribs = get_vv10_contribs(dft_dir, is_restricted, NLC_COEFS)

        X = np.vstack([X, vv10_contribs])

    np.save(FNAME, X)


def solve_from_stored_ae(DATA_ROOT, v2 = False):

    import yaml
    from collections import Counter
    from ase.data import chemical_symbols, atomic_numbers, ground_state_magnetic_moments
    from sklearn.metrics import r2_score
    from pyscf import gto

    coef_sets = []
    scores = []

    etot = np.load(os.path.join(DATA_ROOT, 'etot.npy'))
    mlx = np.load(os.path.join(DATA_ROOT, 'mlx.npy'))
    vv10 = np.load(os.path.join(DATA_ROOT, 'vv10.npy'))
    f = open(os.path.join(DATA_ROOT, 'mols.yaml'), 'r')
    mols = yaml.load(f, Loader = yaml.Loader)
    f.close()
    mols = [gto.mole.unpack(mol) for mol in mols]
    for mol in mols:
        mol.build()

    Z_to_ind = {}
    formulas = {}
    ecounts = []
    for i, mol in enumerate(mols):
        print(i)
        ecounts.append(mol.nelectron)
        if len(mol._atom) == 1:
            Z_to_ind[atomic_numbers[mol._atom[0][0]]] = i
        else:
            atoms = [atomic_numbers[a[0]] for a in mol._atom]
            formulas[i] = Counter(atoms)
    ecounts = np.array(ecounts)

    N = etot.shape[0]
    num_vv10 = vv10.shape[-1]

    print(formulas, Z_to_ind)

    for i in range(num_vv10):
        E_vv10 = vv10[:,i]
        E_dft = etot[:,0]
        E_ccsd = etot[:,1]
        E_x = mlx[:,0]
        E_c = np.append(mlx[:,2:6] + mlx[:,7:11], mlx[:,12:], axis=1)

        diff = nlx + pw92 - pbexc
        #E_dft - E_scan + E_x + E_c + E_vv10
        #E_ccsd - E_scan
        diff = E_ccsd - (E_dft + E_x + E_vv10)

        # E_{tot,PBE} + diff + Evv10 + dot(c, sl_contribs) = E_{tot,CCSD(T)}
        # dot(c, sl_contribs) = E_{tot,CCSD(T)} - E_{tot,PBE} - diff - Evv10
        # not an exact relationship, but should give a decent fit
        X = E_c.copy()
        y = diff.copy()
        weights = []
        for i in range(len(mols)):
            if i in formulas.keys():
                weights.append(1.0)
                formula = formulas[i]
                for Z in list(formula.keys()):
                    X[i,:] -= formula[Z] * E_c[Z_to_ind[Z],:]
                    y[i] -= formula[Z] * y[Z_to_ind[Z]]
                print(mols[i], y[i])
            else:
                weights.append(1.0 / mols[i].nelectron if mols[i].nelectron <= 18 else 0)

        weights = np.array(weights)

        noise = 1e-3
        A = np.linalg.inv(np.dot(X.T * weights, X) + noise * np.identity(X.shape[1]))
        B = np.dot(X.T, weights * y)
        coef = np.dot(A, B)

        score = r2_score(y, np.dot(X, coef))

        coef_sets.append(coef)
        scores.append(score)

    return coef_sets, scores
