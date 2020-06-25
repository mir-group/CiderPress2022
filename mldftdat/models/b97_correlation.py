from mldftdat.lowmem_analyzers import RHFAnalyzer, UHFAnalyzer, CCSDAnalyzer, UCCSDAnalyzer
from mldftdat.dft.numint3 import setup_uks_calc, setup_rks_calc
from pyscf.dft.libxc import eval_xc
from mldftdat.dft.correlation import *
from mldftdat.workflow_utils import get_save_dir
from sklearn.linear_model import LinearRegression

def get_sl_contribs(pbe_dir, restricted):

    if restricted:
        pbe_analyzer = RHFAnalyzer.load(pbe_dir + '/data.hdf5')
        rhot = pbe_analyzer.rho_data[0]
        rdm1_nsp = pbe_analyzer.rdm1
    else:
        pbe_analyzer = UHFAnalyzer.load(pbe_dir + '/data.hdf5')
        rhot = pbe_analyzer.rho_data[0][0] + pbe_analyzer.rho_data[1][0]
        rdm1_nsp = pbe_analyzer.rdm1[0] + pbe_analyzer.rdm1[1]

    rho_data = pbe_analyzer.rho_data
    weights = pbe_analyzer.grid.weights
    grid = pbe_analyzer.grid
    spin = pbe_analyzer.mol.spin
    mol = pbe_analyzer.mol
    rdm1 = pbe_analyzer.rdm1
    E_pbe = pbe_analyzer.e_tot

    numint0 = ProjNumInt(xterms = [], ssterms = [], osterms = [])
    exc0 = numint0.eval_xc(None, rho_data, spin = spin)[0]
    Exc0 = np.dot(exc0 * rhot, weights)

    ss_contribs = []
    for term in default_ss_terms:
        numint = ProjNumInt(xterms = [], ssterms = [term], osterms = [])
        eterm, vterm = numint.eval_xc(None, rho_data, spin = spin)[:2]
        ss_contribs.append(np.dot(eterm * rhot, weights) - Exc0)

    os_contribs = []
    for term in default_os_terms:
        numint = ProjNumInt(xterms = [], ssterms = [], osterms = [term])
        eterm, vterm = numint.eval_xc(None, rho_data, spin = spin)[:2]
        os_contribs.append(np.dot(eterm * rhot, weights) - Exc0)

    sl_contribs = np.array(ss_contribs + os_contribs)

    return sl_contribs

def get_vv10_contribs(pbe_dir, restricted, NLC_COEFS):

    if restricted:
        pbe_analyzer = RHFAnalyzer.load(pbe_dir + '/data.hdf5')
        rhot = pbe_analyzer.rho_data[0]
        rdm1_nsp = pbe_analyzer.rdm1
    else:
        pbe_analyzer = UHFAnalyzer.load(pbe_dir + '/data.hdf5')
        rhot = pbe_analyzer.rho_data[0][0] + pbe_analyzer.rho_data[1][0]
        rdm1_nsp = pbe_analyzer.rdm1[0] + pbe_analyzer.rdm1[1]

    rho_data = pbe_analyzer.rho_data
    weights = pbe_analyzer.grid.weights
    grid = pbe_analyzer.grid
    spin = pbe_analyzer.mol.spin
    mol = pbe_analyzer.mol
    rdm1 = pbe_analyzer.rdm1
    E_pbe = pbe_analyzer.e_tot

    vv10_contribs = []

    for b_test, c_test in NLC_COEFS:

        _, Evv10, _ = nr_rks_vv10(numint, mol, grid, None, rdm1_nsp, b = b_test, c = c_test)

        vv10_contribs.append(Evv10)

    return np.array(vv10_contribs)

def get_nlx_contribs(pbe_dir, restricted, mlfunc):

    if restricted:
        pbe_analyzer = RHFAnalyzer.load(pbe_dir + '/data.hdf5')
        rhot = pbe_analyzer.rho_data[0]
        rdm1_nsp = pbe_analyzer.rdm1
        ml_numint = setup_rks_calc(pbe_analyzer.mol, mlfunc)._numint
    else:
        pbe_analyzer = UHFAnalyzer.load(pbe_dir + '/data.hdf5')
        rhot = pbe_analyzer.rho_data[0][0] + pbe_analyzer.rho_data[1][0]
        rdm1_nsp = pbe_analyzer.rdm1[0] + pbe_analyzer.rdm1[1]
        ml_numint = setup_uks_calc(pbe_analyzer.mol, mlfunc)._numint

    rho_data = pbe_analyzer.rho_data
    weights = pbe_analyzer.grid.weights
    grid = pbe_analyzer.grid
    spin = pbe_analyzer.mol.spin
    mol = pbe_analyzer.mol
    rdm1 = pbe_analyzer.rdm1
    E_pbe = pbe_analyzer.e_tot

    return ml_numint.eval_xc(None, mol, rho_data, grid, rdm1, spin = spin)[0]

def get_nlx_contribs(pbe_dir, restricted):

    if restricted:
        pbe_analyzer = RHFAnalyzer.load(pbe_dir + '/data.hdf5')
        ccsd_analyzer = CCSDAnalyzer.load(ccsd_dir + '/data.hdf5')
    else:
        pbe_analyzer = UHFAnalyzer.load(pbe_dir + '/data.hdf5')
        ccsd_analyzer = UCCSDAnalyzer.load(ccsd_dir + '/data.hdf5')

    E_pbe = pbe_analyzer.e_tot
    E_ccsd = ccsd_analyzer.e_tot

    return np.array([E_pbe, E_ccsd])

def get_b97_data_and_targets(pbe_dir, ccsd_dir, restricted,
                             b_test, c_test, mlfunc):

    if restricted:
        pbe_analyzer = RHFAnalyzer.load(pbe_dir + '/data.hdf5')
        ccsd_analyzer = CCSDAnalyzer.load(ccsd_dir + '/data.hdf5')
        rhot = pbe_analyzer.rho_data[0]
        rdm1_nsp = pbe_analyzer.rdm1
        ml_numint = setup_rks_calc(pbe_analyzer.mol, mlfunc)._numint
    else:
        pbe_analyzer = UHFAnalyzer.load(pbe_dir + '/data.hdf5')
        ccsd_analyzer = UCCSDAnalyzer.load(ccsd_dir + '/data.hdf5')
        rhot = pbe_analyzer.rho_data[0][0] + pbe_analyzer.rho_data[1][0]
        rdm1_nsp = pbe_analyzer.rdm1[0] + pbe_analyzer.rdm1[1]
        ml_numint = setup_uks_calc(pbe_analyzer.mol, mlfunc)._numint

    rho_data = pbe_analyzer.rho_data
    weights = pbe_analyzer.grid.weights
    grid = pbe_analyzer.grid
    spin = pbe_analyzer.mol.spin
    mol = pbe_analyzer.mol
    rdm1 = pbe_analyzer.rdm1
    E_pbe = pbe_analyzer.e_tot
    E_ccsd = ccsd_analyzer.e_tot

    numint0 = ProjNumInt(xterms = [], ssterms = [], osterms = [])
    exc0 = numint0.eval_xc(None, rho_data, spin = spin)[0]
    Exc0 = np.dot(exc0 * rhot, weights)

    ss_contribs = []
    for term in default_ss_terms:
        numint = ProjNumInt(xterms = [], ssterms = [term], osterms = [])
        eterm, vterm = numint.eval_xc(None, rho_data, spin = spin)[:2]
        ss_contribs.append(np.dot(eterm * rhot, weights) - Exc0)

    os_contribs = []
    for term in default_os_terms:
        numint = ProjNumInt(xterms = [], ssterms = [], osterms = [term])
        eterm, vterm = numint.eval_xc(None, rho_data, spin = spin)[:2]
        os_contribs.append(np.dot(eterm * rhot, weights) - Exc0)

    sl_contribs = np.array(ss_contribs + os_contribs)

    _, Evv10, _ = nr_rks_vv10(numint, mol, grid, None, rdm1_nsp, b = b_test, c = c_test)

    epbe = eval_xc('GGA_X_PBE,GGA_C_PBE', rho_data, spin = spin)[0]
    eml = ml_numint.eval_xc(None, mol, rho_data, grid, rdm1, spin = spin)[0]

    diff = np.dot(eml - epbe, rhot * weights)

    # E_{tot,PBE} + diff + Evv10 + dot(c, sl_contribs) = E_{tot,CCSD(T)}
    # dot(c, sl_contribs) = E_{tot,CCSD(T)} - E_{tot,PBE} - diff - Evv10
    # not an exact relationship, but should give a decent fit

    target = E_ccsd - E_pbe - diff - Evv10

    return sl_contribs, target


def store_sl_contribs_dataset(ROOT, MOL_IDS, IS_RESTRICTED_LIST):

    X = np.zeros([0,8])

    for mol_id, is_restricted in zip(MOL_IDS, IS_RESTRICTED_LIST):

        if is_restricted:
            pbe_dir = get_save_dir(ROOT, 'RKS', 'aug-cc-pvtz', mol_id, functional = 'PBE')
            ccsd_dir = get_save_dir(ROOT, 'CCSD', 'aug-cc-pvtz', mol_id)
        else:
            pbe_dir = get_save_dir(ROOT, 'UKS', 'aug-cc-pvtz', mol_id, functional = 'PBE')
            ccsd_dir = get_save_dir(ROOT, 'UCCSD', 'aug-cc-pvtz', mol_id)

        sl_contribs = get_sl_contribs(pbe_dir, is_restricted, NLC_COEFS)

        X = np.vstack([X, sl_contribs])

    return X

def store_nlx_contribs_dataset(ROOT, MOL_IDS, IS_RESTRICTED_LIST, MLFUNC):

    x = []

    for mol_id, is_restricted in zip(MOL_IDS, IS_RESTRICTED_LIST):

        if is_restricted:
            pbe_dir = get_save_dir(ROOT, 'RKS', 'aug-cc-pvtz', mol_id, functional = 'PBE')
            ccsd_dir = get_save_dir(ROOT, 'CCSD', 'aug-cc-pvtz', mol_id)
        else:
            pbe_dir = get_save_dir(ROOT, 'UKS', 'aug-cc-pvtz', mol_id, functional = 'PBE')
            ccsd_dir = get_save_dir(ROOT, 'UCCSD', 'aug-cc-pvtz', mol_id)

        x.append(get_nlx_contribs(pbe_dir, is_restricted))

    return x

def store_total_energies_dataset(ROOT, MOL_IDS, IS_RESTRICTED_LIST):

    # PBE, CCSD
    y = np.zeros([0, 2])

    for mol_id, is_restricted in zip(MOL_IDS, IS_RESTRICTED_LIST):

        if is_restricted:
            pbe_dir = get_save_dir(ROOT, 'RKS', 'aug-cc-pvtz', mol_id, functional = 'PBE')
            ccsd_dir = get_save_dir(ROOT, 'CCSD', 'aug-cc-pvtz', mol_id)
        else:
            pbe_dir = get_save_dir(ROOT, 'UKS', 'aug-cc-pvtz', mol_id, functional = 'PBE')
            ccsd_dir = get_save_dir(ROOT, 'UCCSD', 'aug-cc-pvtz', mol_id)

        pbe_ccsd = get_etot_contribs(pbe_dir, is_restricted)

        X = np.vstack([X, pbe_ccsd])

    return X

def store_vv10_contribs_dataset(ROOT, MOL_IDS, IS_RESTRICTED_LIST, NLC_COEFS):

    X = np.zeros([0, len(NLC_COEFS)])

    for mol_id, is_restricted in zip(MOL_IDS, IS_RESTRICTED_LIST):

        if is_restricted:
            pbe_dir = get_save_dir(ROOT, 'RKS', 'aug-cc-pvtz', mol_id, functional = 'PBE')
            ccsd_dir = get_save_dir(ROOT, 'CCSD', 'aug-cc-pvtz', mol_id)
        else:
            pbe_dir = get_save_dir(ROOT, 'UKS', 'aug-cc-pvtz', mol_id, functional = 'PBE')
            ccsd_dir = get_save_dir(ROOT, 'UCCSD', 'aug-cc-pvtz', mol_id)

        vv10_contribs = get_vv10_contribs(pbe_dir, is_restricted, NLC_COEFS)

        X = np.vstack([X, vv10_contribs])

    return X

def solve_b97_coef(ROOT, MOL_IDS, IS_RESTRICTED_LIST, NLC_COEFS, MLFUNC):

    coef_sets = []
    scores = []

    for b_test, c_test in NLC_COEFS:

        X = np.zeros([0,8])
        y = []

        for mol_id, is_restricted in zip(MOL_IDS, IS_RESTRICTED_LIST):

            if is_restricted:
                pbe_dir = get_save_dir(ROOT, 'RKS', 'aug-cc-pvtz', mol_id, functional = 'PBE')
                ccsd_dir = get_save_dir(ROOT, 'CCSD', 'aug-cc-pvtz', mol_id)
            else:
                pbe_dir = get_save_dir(ROOT, 'UKS', 'aug-cc-pvtz', mol_id, functional = 'PBE')
                ccsd_dir = get_save_dir(ROOT, 'UCCSD', 'aug-cc-pvtz', mol_id)

            sl_contribs, target = get_b97_data_and_targets(pbe_dir, ccsd_dir,
                                                           is_restricted,
                                                           b_test, c_test, MLFUNC)

            X = np.vstack([X, sl_contribs])
            y.append(target)

        lr = LinearRegression(fit_intercept = False)
        lr.fit(X, y)

        score = lr.score(X, y)

        coef_sets.append(lr.coef_)
        scores.append(score)

    return coef_sets, scores
