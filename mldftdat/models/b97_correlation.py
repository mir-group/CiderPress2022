from pyscf.lowmem_analyzers import RHFAnalyzer, UHFAnalyzer, CCSDAnalyzer, UCCSDAnalyzer

def get_b97_data_and_targets(pbe_dir, ccsd_dir, restricted,
                             b_test, c_test, ml_numint):

    if restricted:
        pbe_analyzer = RHFAnalyzer.load(pbe_dir)
        ccsd_analyzer = CCSDAnalyzer.load(ccsd_dir)
        rhot = pbe_analyzer.rho_data[0]
        rdm1_nsp = pbe_analyzer.rdm1
    else:
        pbe_analyzer = UHFAnalyzer.load(pbe_dir)
        ccsd_analyzer = UCCSDAnalyzer.load(ccsd_dir)
        rhot = pbe_analyzer.rho_data[0][0] + pbe_analyzer.rho_data[1][0]
        rdm1_nsp = pbe_analyzer.rdm1[0] + pbe_analyzer.rdm1[1]

    rho_data = pbe_analyzer.rho_data
    weights = pbe_analyzer.grid.weights
    spin = pbe_analyzer.mol.spin
    mol = pbe_analyzer.mol
    rdm1 = pbe_analyzer.rdm1

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


def solve_b97_coeff(ROOT, MOL_IDS, IS_RESTRICTED_LIST, NLC_COEFS, ML_NUMINT):

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
                                                           is_restricted, ML_NUMINT)

            X = np.vstack([X, sl_contribs])
            y.append(target)

        lr = LinearRegression(fit_intercept = False)
        lr.fit(X, y)

        score = lr.score(X, y)

        coef_sets.append(lr.coef_)
        scores.append(score)

    return coef_sets, scores
