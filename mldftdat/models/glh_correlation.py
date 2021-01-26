from mldftdat.lowmem_analyzers import RHFAnalyzer, UHFAnalyzer, CCSDAnalyzer, UCCSDAnalyzer
from pyscf.dft.libxc import eval_xc
from mldftdat.xcutil.cdesc import *
from mldftdat.workflow_utils import get_save_dir, SAVE_ROOT
from sklearn.linear_model import LinearRegression
from pyscf.dft.numint import NumInt
from mldftdat.density import get_exchange_descriptors2
import os
import numpy as np
import yaml
import logging

LDA_FACTOR = - 3.0 / 4.0 * (3.0 / np.pi)**(1.0/3)

DEFAULT_FUNCTIONAL = 'PBE'
DEFAULT_BASIS = 'def2-qzvppd'

CF = 0.3 * (6 * np.pi**2)**(2.0/3)

def default_desc_getter(weights, rhou, rhod, g2u, g2o, g2d, tu, td, exu, exd):
    rhot = rhou + rhod
    g2 = g2u + 2* g2o + g2d
    exo = exu + exd

    co0, vo0 = get_os_baseline(rhou, rhod, g2, type=0)[:2]
    co1, vo1 = get_os_baseline(rhou, rhod, g2, type=1)[:2]
    co0 *= rhot
    co1 *= rhot
    cx = co0
    co = co1

    nu, nd = rhou, rhod

    zeta = (rhou - rhod) / (rhot)
    ds = ((1-zeta)**(5.0/3) + (1+zeta)**(5.0/3))/2
    CU = 0.3 * (3 * np.pi**2)**(2.0/3)

    ldaxu = 2**(1.0/3) * LDA_FACTOR * rhou**(4.0/3) - 1e-20
    ldaxd = 2**(1.0/3) * LDA_FACTOR * rhod**(4.0/3) - 1e-20
    ldaxt = ldaxu + ldaxd

    gamma = 2**(2./3) * 0.004
    gammass = 0.004
    chi = get_chi_full_deriv(rhot + 1e-16, zeta, g2, tu + td)[0]
    chiu = get_chi_full_deriv(rhou + 1e-16, 1, g2u, tu)[0]
    chid = get_chi_full_deriv(rhod + 1e-16, 1, g2d, td)[0]
    x2 = get_x2(nu+nd, g2)[0]
    x2u = get_x2(nu, g2u)[0]
    x2d = get_x2(nd, g2d)[0]
    amix = get_amix_schmidt2(rhot, zeta, x2, chi)[0]
    #amix2 = get_amix_schmidt2(rhot, zeta, x2, chi, order=0)[0]
    amix3 = get_amix_schmidt2(rhot, zeta, x2, chi, mixer='n')[0]
    #amix4 = get_amix_schmidt2(rhot, zeta, x2, chi, zorder=0)[0]
    chidesc = np.array(get_chi_desc(chi)[:4])
    chidescu = np.array(get_chi_desc(chiu)[:4])
    chidescd = np.array(get_chi_desc(chid)[:4])
    Fx = exo / ldaxt
    Fxu = exu / ldaxu
    Fxd = exd / ldaxd
    corrterms = np.append(get_separate_xef_terms(Fx, return_deriv=False),
                          chidesc, axis=0)
    extermsu = np.append(get_separate_sl_terms(x2u, chiu, gammass)[0],
                         get_separate_xefa_terms(Fxu, chiu)[0], axis=0)
    extermsd = np.append(get_separate_sl_terms(x2d, chid, gammass)[0],
                         get_separate_xefa_terms(Fxd, chid)[0], axis=0)

    cmscale = 17.0 / 3
    cmix = cmscale * (1 - chi) / (cmscale - chi)
    #cmix_terms = np.array([cmix * (1-cmix), cmix**2 * (1-cmix),
    #                       cmix * (1-cmix)**2, cmix**2 * (1-cmix)**2])
    cmix_terms0 = np.array([chi**2-chi, chi**3-chi, chi**4-chi**2, chi**5-chi**3,
                           chi**6-chi**4, chi**7-chi**5, chi**8-chi**6])
    cmix_terms = np.array([chi, chi**2, chi**3-chi, chi**4-chi**2, chi**5-chi**3,
                           chi**6-chi**4, chi**7-chi**5, chi**8-chi**6])
    Ecscan = np.dot(co * cmix + cx * (1-cmix), weights)
    Eterms = np.dot(cmix_terms0 * (cx-co), weights)
    Eterms2 = np.dot(cmix_terms * cx, weights)
    Eterms2[0] = np.dot(co * amix * (Fx-1), weights)
    Eterms2[1] = np.dot(cx * amix * (Fx-1), weights)
    Eterms3 = np.append(np.dot(corrterms[1:5] * cx, weights),
                        np.dot(corrterms[1:5] * (1-chi**2) * (co), weights))
    #Eterms3[-1] = np.dot((Fx-1) * cx, weights)
    #Eterms3[-2] = np.dot((Fx-1) * (co * cmix + cx * (1-cmix)), weights)
    Fterms = np.dot(extermsu * ldaxu * amix, weights)
    Fterms += np.dot(extermsd * ldaxd * amix, weights)

    return Ecscan, np.concatenate([Eterms, Eterms2, Eterms3, Fterms],
                                   axis=0)

def default_desc_noise():
    n0 = 1e-3 * np.ones(15)
    n1 = 1e-4 * np.ones(36)
    return np.append(n0, n1)


def get_corr_contribs(dft_dir, restricted, mlfunc,
                      desc_getter=default_desc_getter):

    exact = (mlfunc is None)

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

    if restricted:
        rho_data_u, rho_data_d = rho_data / 2, rho_data / 2
    else:
        rho_data_u, rho_data_d = rho_data[0], rho_data[1]

    rhou = rho_data_u[0] + 1e-20
    g2u = np.einsum('ir,ir->r', rho_data_u[1:4], rho_data_u[1:4])
    tu = rho_data_u[5] + 1e-20
    rhod = rho_data_d[0] + 1e-20
    g2d = np.einsum('ir,ir->r', rho_data_d[1:4], rho_data_d[1:4])
    td = rho_data_d[5] + 1e-20
    ntup = (rhou, rhod)
    gtup = (g2u, g2d)
    ttup = (tu, td)
    rhot = rhou + rhod
    g2o = np.einsum('ir,ir->r', rho_data_u[1:4], rho_data_d[1:4])
    g2 = g2u + 2 * g2o + g2d

    N = dft_analyzer.grid.weights.shape[0]
    if restricted:
        if exact:
            ex = dft_analyzer.fx_energy_density / (rho_data[0] + 1e-20)
        else:
            desc  = np.zeros((N, len(mlfunc.desc_list)))
            ddesc = np.zeros((N, len(mlfunc.desc_list)))
            xdesc = get_exchange_descriptors2(dft_analyzer, restricted=True)
            for i, d in enumerate(mlfunc.desc_list):
                desc[:,i], ddesc[:,i] = d.transform_descriptor(xdesc, deriv = 1)
            xef = mlfunc.get_F(desc)
            ex = LDA_FACTOR * xef * rho_data[0]**(1.0/3)
        exu = ex
        exd = ex
        exo = ex
        rhou = rho_data[0] / 2
        rhod = rho_data[0] / 2
        rhot = rho_data[0]
        Ex = np.dot(exo * rhot, weights)
    else:
        if exact:
            exu = dft_analyzer.fx_energy_density_u / (rho_data[0][0] + 1e-20)
            exd = dft_analyzer.fx_energy_density_d / (rho_data[1][0] + 1e-20)
        else:
            desc  = np.zeros((N, len(mlfunc.desc_list)))
            ddesc = np.zeros((N, len(mlfunc.desc_list)))
            xdesc_u, xdesc_d = get_exchange_descriptors2(dft_analyzer, restricted=False)
            for i, d in enumerate(mlfunc.desc_list):
                desc[:,i], ddesc[:,i] = d.transform_descriptor(xdesc_u, deriv = 1)
            xef = mlfunc.get_F(desc)
            exu = 2**(1.0/3) * LDA_FACTOR * xef * rho_data[0][0]**(1.0/3)
            for i, d in enumerate(mlfunc.desc_list):
                desc[:,i], ddesc[:,i] = d.transform_descriptor(xdesc_d, deriv = 1)
            xef = mlfunc.get_F(desc)
            exd = 2**(1.0/3) * LDA_FACTOR * xef * rho_data[1][0]**(1.0/3)
        rhou = rho_data[0][0]
        rhod = rho_data[1][0]
        rhot = rho_data[0][0] + rho_data[1][0]
        exo = (exu * rho_data[0][0] + exd * rho_data[1][0])
        Ex = np.dot(exo, weights)
        exo /= (rhot + 1e-20)

    exu = exu * rhou
    exd = exd * rhod
    exo = exo * rhot

    Excbas = dft_analyzer.e_tot - (dft_analyzer.calc.energy_tot() - dft_analyzer.fx_total)

    logging.info('EX ERROR {} {} {}'.format(Ex - dft_analyzer.fx_total, Ex, dft_analyzer.fx_total))
    if (np.abs(Ex - dft_analyzer.fx_total) > 1e-7):
        logging.warn('LARGE ERROR')

    Ecbas, desc = desc_getter(weights, rhou, rhod, g2u, g2o, g2d, tu, td, exu, exd)

    return np.concatenate([[Ex, Excbas], desc,
                          [Ecbas, dft_analyzer.fx_total]], axis=0)


def store_corr_contribs_dataset(FNAME, MOL_FNAME, MLFUNC=None,
                                desc_getter=default_desc_getter,
                                functional=DEFAULT_FUNCTIONAL,
                                basis=DEFAULT_BASIS):

    with open(os.path.join(MOL_FNAME), 'r') as f:
        data = yaml.load(f, Loader = yaml.Loader)
        dft_dirs = data['dft_dirs']
        is_restricted_list = data['is_restricted_list']

    #SIZE = ndesc + 4
    #X = np.zeros([0,SIZE])
    X = None

    for dft_dir, is_restricted in zip(dft_dirs, is_restricted_list):

        logging.info('Corr contribs in {}'.format(dft_dir))

        sl_contribs = get_corr_contribs(dft_dir, is_restricted,
                                        MLFUNC, desc_getter)
        assert (not np.isnan(sl_contribs).any())
        if X is None:
            X = sl_contribs.copy().reshape(1,-1)
        else:
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
    if ccsd_analyzer.mol.nelectron < 3:
        E_ccsd = ccsd_analyzer.e_tot
    else:
        E_ccsd = ccsd_analyzer.e_tot + ccsd_analyzer.e_tri

    return np.array([E_pbe, E_ccsd])

def get_etot_contribs(dft_dir, ccsd_dir, restricted):
    with open(os.path.join(dft_dir, 'run_info.json'), 'r') as f:
        dft_dat = json.load(f)
    with open(os.path.join(ccsd_dir, 'run_info.json'), 'r') as f:
        ccsd_dat = json.load(f)
    if ccsd_dat['nelectron'] < 3:
        E_ccsd = ccsd_dat['e_tot']
    else:
        E_ccsd = ccsd_dat['e_tot'] + ccsd_dat['e_tri']
    return np.arra([dft_dat['e_tot'], E_ccsd])

def store_total_energies_dataset(FNAME, MOL_FNAME,
                                 functional=DEFAULT_FUNCTIONAL,
                                 basis=DEFAULT_BASIS):

    # DFT, CCSD
    y = np.zeros([0, 2])

    with open(os.path.join(MOL_FNAME), 'r') as f:
        data = yaml.load(f, Loader = yaml.Loader)
        dft_dirs = data['dft_dirs']
        ccsd_dirs = data['ccsd_dirs']
        is_restricted_list = data['is_restricted_list']

    for dft_dir, ccsd_dir, is_restricted in zip(dft_dirs, ccsd_dirs,
                                                is_restricted_list):
        logging.info('Storing total energies from {} {}'.format(dft_dir, ccsd_dir))

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
    grid.build()

    vv10_contribs = []

    for b_test, c_test in NLC_COEFS:

        _, Evv10, _ = nr_rks_vv10(numint, mol, grid, None, rdm1_nsp, b = b_test, c = c_test)

        vv10_contribs.append(Evv10)

    return np.array(vv10_contribs)

DEFAULT_NLC_COEFS = [[5.9, 0.0093], [6.0, 0.01], [6.3, 0.0089],\
                     [9.8, 0.0093], [14.0, 0.0093], [15.7, 0.0093]]

def store_vv10_contribs_dataset(FNAME, MOL_FNAME,
                                NLC_COEFS=DEFAULT_NLC_COEFS,
                                functional=DEFAULT_FUNCTIONAL,
                                basis=DEFAULT_BASIS):
    with open(os.path.join(MOL_FNAME), 'r') as f:
        data = yaml.load(f, Loader = yaml.Loader)
        dft_dirs = data['dft_dirs']
        is_restricted_list = data['is_restricted_list']

    X = np.zeros([0, len(NLC_COEFS)])

    for dft_dir, is_restricted in zip(dft_dirs, is_restricted_list):

        logging.info('Calculate VV10 contribs for {}'.format(mol_id))

        vv10_contribs = get_vv10_contribs(dft_dir, is_restricted, NLC_COEFS)

        X = np.vstack([X, vv10_contribs])

    np.save(FNAME, X)


def solve_from_stored_ae(AE_DIR, ATOM_DIR, DESC_NAME, noise=1e-3,
                         use_vv10=False, regression_method='weighted_llsr'):
    """
    regression_method options:
        weighted_lrr: weighted linear ridge regression
        weighted_lasso: weighted lasso regression
    """
    import yaml
    from collections import Counter
    from ase.data import chemical_symbols, atomic_numbers, ground_state_magnetic_moments
    from sklearn.metrics import r2_score
    from pyscf import gto

    coef_sets = []
    scores = []

    etot = np.load(os.path.join(AE_DIR, 'etot.npy'))
    mlx = np.load(os.path.join(AE_DIR, DESC_NAME))
    if use_vv10:
        vv10 = np.load(os.path.join(AE_DIR, 'vv10.npy'))
    with open(os.path.join(AE_DIR, 'mols.yaml'), 'r') as f:
        mols = yaml.load(f, Loader=yaml.Loader)['mols']

    aetot = np.load(os.path.join(ATOM_DIR, 'etot.npy'))
    amlx = np.load(os.path.join(ATOM_DIR, DESC_NAME))
    if use_vv10:
        atom_vv10 = np.load(os.path.join(ATOM_DIR, 'vv10.npy'))
    with open(os.path.join(ATOM_DIR, 'mols.yaml'), 'r') as f:
        amols = yaml.load(f, Loader=yaml.Loader)['mols']

    logging.debug("SHAPES {} {} {} {}".format(mlx.shape, etot.shape, amlx.shape, aetot.shape))

    valset_bools_init = np.array([mol['valset'] for mol in mols])
    valset_bools_init = np.append(valset_bools_init,
                        np.zeros(len(amols), valset_bools_init.dtype))
    mols = [gto.mole.unpack(mol) for mol in mols]
    for mol in mols:
        mol.build()
    amols = [gto.mole.unpack(mol) for mol in amols]
    for mol in amols:
        mol.build()

    Z_to_ind = {}
    Z_to_ind_bsl = {}
    ind_to_Z_ion = {}
    formulas = {}
    ecounts = []
    for i, mol in enumerate(mols):
        ecounts.append(mol.nelectron)
        if len(mol._atom) == 1:
            Z = atomic_numbers[mol._atom[0][0]]
            Z_to_ind[Z] = i
        else:
            atoms = [atomic_numbers[a[0]] for a in mol._atom]
            formulas[i] = Counter(atoms)

    for i, mol in enumerate(amols):
        Z = atomic_numbers[mol._atom[0][0]]
        if mol.spin == ground_state_magnetic_moments[Z]:
            Z_to_ind_bsl[Z] = i
        else:
            ind_to_Z_ion[i] = Z

    ecounts = np.array(ecounts)

    N = etot.shape[0]
    if use_vv10:
        num_vv10 = vv10.shape[-1]
    else:
        num_vv10 = 1

    for i in range(num_vv10):

        def get_terms(etot, mlx, vv10=None):
            if vv10 is not None:
                E_vv10 = vv10[:,i]
            E_dft = etot[:,0]
            E_ccsd = etot[:,1]
            E_x = mlx[:,0]
            E_xcbas = mlx[:,1]
            E_cbas = mlx[:,-2]
            E_c = mlx[:,2:-2]
            """
            E_c[:,3:7] = 0
            E_c[:,9:15] = 0
            #E_c[:,17:19] = 0
            E_c[:,15:23] = 0
            E_c[:,23:38:4] = 0
            E_c[:,34:38] =0
            E_c[:,38:-1:4] = 0
            E_c[:,40:-1:4] = 0
            E_c[:,42:-1:4] = 0
            """
            diff = E_ccsd - (E_dft - E_xcbas + E_x + E_cbas)
            if use_vv10:
                diff -= E_vv10

            stds = np.std(E_c, axis=0)
            print(len(stds[stds!=0]))
            print(stds.tolist())
            print(E_dft.shape, E_ccsd.shape, E_x.shape, E_xcbas.shape, E_c.shape, diff.shape)
            return E_c, diff, E_ccsd, E_dft, E_xcbas, E_x, E_cbas

        E_c, diff, E_ccsd, E_dft, E_xscan, E_x, E_cscan = \
            get_terms(etot, mlx)
        E_c2, diff2, E_ccsd2, E_dft2, E_xscan2, E_x2, E_cscan2 = \
            get_terms(aetot, amlx)
        if type(noise) is not float:
            noise = noise[:E_c.shape[1]]
        E_c = np.append(E_c, E_c2, axis=0)
        diff = np.append(diff, diff2)
        E_ccsd = np.append(E_ccsd, E_ccsd2)
        E_dft = np.append(E_dft, E_dft2)
        E_xscan = np.append(E_xscan, E_xscan2)
        E_x = np.append(E_x, E_x2)
        E_cscan = np.append(E_cscan, E_cscan2)

        # E_{tot,PBE} + diff + Evv10 + dot(c, sl_contribs) = E_{tot,CCSD(T)}
        # dot(c, sl_contribs) = E_{tot,CCSD(T)} - E_{tot,PBE} - diff - Evv10
        # not an exact relationship, but should give a decent fit
        X = E_c.copy()
        y = diff.copy()
        Ecc = E_ccsd.copy()
        Edf = E_dft.copy()
        weights = []
        for i in range(len(mols)):
            if i in formulas.keys():
                #weights.append(1.0)
                weights.append(1.0 / (len(mols[i]._atom) - 1))
                formula = formulas[i]
                if formula.get(1) == 2 and formula.get(8) == 1 and len(list(formula.keys()))==2:
                    waterind = i
                    logging.info("{} {} {}".format(formula, E_ccsd[i], E_dft[i]))
                for Z in list(formula.keys()):
                    X[i,:] -= formula[Z] * X[Z_to_ind[Z],:]
                    y[i] -= formula[Z] * y[Z_to_ind[Z]]
                    Ecc[i] -= formula[Z] * Ecc[Z_to_ind[Z]]
                    Edf[i] -= formula[Z] * Edf[Z_to_ind[Z]]
                logging.debug("{} {} {} {} {}".format(formulas[i], y[i], Ecc[i],
                                                      Edf[i], E_x[i] - E_xscan[i]))
            else:
                if mols[i].nelectron == 1:
                    hind = i
                if mols[i].nelectron == 8:
                    oind = i
                    logging.debug("{} {} {}".format(mols[i], E_ccsd[i], E_dft[i]))
                if mols[i].nelectron == 3:
                    weights.append(1e-8 / 3)
                else:
                    weights.append(1e-8 / mols[i].nelectron if mols[i].nelectron <= 10 else 0)
        for i in range(len(amols)):
            weights.append(8 / amols[i].nelectron)
            if i in ind_to_Z_ion.keys():
                j = len(mols) + i
                k = len(mols) + Z_to_ind_bsl[ind_to_Z_ion[i]]
                X[j,:] -= X[k,:]
                y[j] -= y[k]
                Ecc[j] -= Ecc[k]
                Edf[j] -= Edf[k]
                weights[-1] = 4.0

        weights = np.array(weights)
        
        logging.info("{}".format(E_xscan[[hind,oind,waterind]]))
        logging.info('ASSESS MEAN DIFF')
        logging.info("{}".format(np.mean(np.abs(Ecc-Edf)[weights > 0])))
        logging.info("{}".format(np.mean(np.abs(diff)[weights > 0])))

        inds = np.arange(len(y))
        valset_bools = valset_bools_init[weights > 0]
        X = X[weights > 0, :]
        y = y[weights > 0]
        Ecc = Ecc[weights > 0]
        Edf = Edf[weights > 0]
        inds = inds[weights > 0]
        indd = {}
        for i in range(inds.shape[0]):
            indd[inds[i]] = i
        weights = weights[weights > 0]

        logging.info("{} {}".format(E_ccsd[waterind], E_dft[waterind]))

        oind = indd[oind]
        hind = indd[hind]
        waterind = indd[waterind]

        trset_bools = np.logical_not(valset_bools)
        Xtr = X[trset_bools]
        Xts = X[valset_bools]
        ytr = y[trset_bools]
        yts = y[valset_bools]
        wtr = weights[trset_bools]
        if regression_method == 'weighted_lrr':
            if type(noise) == float:
                noise = np.ones(Xtr.shape[1]) * noise
            A = np.linalg.inv(np.dot(Xtr.T * wtr, Xtr) + np.diag(noise))
            B = np.dot(Xtr.T, wtr * ytr)
            coef = np.dot(A, B)
        elif regression_method == 'weighted_lasso':
            from sklearn.linear_model import Lasso
            model = Lasso(alpha=noise, fit_intercept=False)
            model.fit(Xtr * wtr[:,np.newaxis], ytr * wtr)
            coef = model.coef_
        else:
            raise ValueError('Model choice not recognized')

        score = r2_score(yts, np.dot(Xts, coef))
        score0 = r2_score(yts, np.dot(Xts, 0 * coef))
        logging.info("{} {}".format(Xts.shape, yts.shape))
        logging.info("{} {}".format(score, score0))
        logging.info("{} {} {} {} {}".format((Ecc)[[hind,oind,waterind]], Ecc[oind],
                     Edf[oind], Ecc[waterind], Edf[waterind]))
        logging.info("{} {} {} {} {}".format((y - Ecc - np.dot(X, coef))[[hind,oind,waterind]],
                     Ecc[oind], Edf[oind], Ecc[waterind], Edf[waterind]))
        print('SCAN ALL', np.mean(np.abs(Ecc-Edf)),
                     np.mean((Ecc-Edf)), np.std(Ecc-Edf))
        print('SCAN VAL', np.mean(np.abs(Ecc-Edf)[valset_bools]),
                     np.mean((Ecc-Edf)[valset_bools]),
                     np.std((Ecc-Edf)[valset_bools]))
        print('ML ALL', np.mean(np.abs(y - np.dot(X, coef))),
                     np.mean(y - np.dot(X, coef)),
                     np.std(y - np.dot(X,coef)))
        print('ML VAL', np.mean(np.abs(yts - np.dot(Xts, coef))),
                     np.mean(yts - np.dot(Xts, coef)),
                     np.std(yts-np.dot(Xts,coef)))
        print(np.max(np.abs(y - np.dot(X, coef))),
                     np.max(np.abs(Ecc - Edf)))
        print(np.max(np.abs(yts - np.dot(Xts, coef))),
                     np.max(np.abs(Ecc - Edf)[valset_bools]))
        print(coef)

        coef_sets.append(coef)
        scores.append(score)

    return coef_sets, scores


def solve_from_stored_accdb(AE_DIR, ATOM_DIR, DESC_NAME, noise=1e-3,
                            use_vv10=False, regression_method='weighted_llsr'):
    """
    regression_method options:
        weighted_lrr: weighted linear ridge regression
        weighted_lasso: weighted lasso regression
    """
    import yaml
    from collections import Counter
    from ase.data import chemical_symbols, atomic_numbers, ground_state_magnetic_moments
    from sklearn.metrics import r2_score
    from pyscf import gto

    with open(TRAIN_FILE, 'r') as f:
        d = yaml.load(f, Loader=yaml.Loader)
        weights = np.array(d['WEIGHTS'])
        bond_counts = np.array(d['BOND_COUNTS'])
    # TODO get formulas as (entry_num count)
    # TODO get ref_etot

    etot = np.load(os.path.join(AE_DIR, 'etot.npy'))
    feat = np.load(os.path.join(AE_DIR, DESC_NAME))
    if use_vv10:
        vv10 = np.load(os.path.join(AE_DIR, 'vv10.npy'))
    with open(os.path.join(AE_DIR, 'mols.yaml'), 'r') as f:
        mols = yaml.load(f, Loader=yaml.Loader)['mols']

    logging.debug("SHAPES {} {} {} {}".format(mlx.shape, etot.shape))

    mols = [gto.mole.unpack(mol) for mol in mols]
    for mol in mols:
        mol.build()

    N = etot.shape[0]
    if use_vv10:
        num_vv10 = vv10.shape[-1]
    else:
        num_vv10 = 1

    np.random.seed(42)
    lst = np.arange(len(formulas))
    np.random.shuffle(lst)

    for i in range(num_vv10):

        if use_vv10:
            E_vv10 = vv10[:,i]
        E_dft = etot
        E_x = mlx[:,0]
        E_xcbas = mlx[:,1]
        E_cbas = mlx[:,-2]
        E_c = mlx[:,2:-2]
        E_bas = E_dft - E_xcbas + E_x + E_cbas
        if use_vv10:
            diff -= E_vv10
        stds = np.std(E_c, axis=0)

        if type(noise) is not float:
            noise = noise[:E_c.shape[1]]

        X = np.zeros((len(formulas), E_c.shape[1]))
        y = diff.copy()
        for i in range(len(formulas)):
            for count, entry_num in formulas[i]:
                X[i,:] += count * E_c[entry_num,:]
                y[i] -= count * E_bas[entry_num]

        Xtr = X[lst[NVAL:]]
        ytr = y[lst[NVAL:]]
        wtr = weights[lst[NVAL:]]

        Xts = X[lst[:NVAL]]
        yts = y[lst[:NVAL]]

        if regression_method == 'weighted_lrr':
            if type(noise) == float:
                noise = np.ones(Xtr.shape[1]) * noise
            A = np.linalg.inv(np.dot(Xtr.T * wtr, Xtr) + np.diag(noise))
            B = np.dot(Xtr.T, wtr * ytr)
            coef = np.dot(A, B)
        elif regression_method == 'weighted_lasso':
            from sklearn.linear_model import Lasso
            model = Lasso(alpha=noise, fit_intercept=False)
            model.fit(Xtr * wtr[:,np.newaxis], ytr * wtr)
            coef = model.coef_
        else:
            raise ValueError('Model choice not recognized')

        score = r2_score(yts, np.dot(Xts, coef))
        score0 = r2_score(yts, np.dot(Xts, 0 * coef))
        logging.info("{} {}".format(Xts.shape, yts.shape))
        logging.info("{} {}".format(score, score0))
        print('SCAN ALL', np.mean(np.abs(Ecc-Edf)),
                     np.mean((Ecc-Edf)), np.std(Ecc-Edf))
        print('SCAN VAL', np.mean(np.abs(Ecc-Edf)[valset_bools]),
                     np.mean((Ecc-Edf)[valset_bools]),
                     np.std((Ecc-Edf)[valset_bools]))
        print('ML ALL', np.mean(np.abs(y - np.dot(X, coef))),
                     np.mean(y - np.dot(X, coef)),
                     np.std(y - np.dot(X,coef)))
        print('ML VAL', np.mean(np.abs(yts - np.dot(Xts, coef))),
                     np.mean(yts - np.dot(Xts, coef)),
                     np.std(yts-np.dot(Xts,coef)))
        print(np.max(np.abs(y - np.dot(X, coef))),
                     np.max(np.abs(Ecc - Edf)))
        print(np.max(np.abs(yts - np.dot(Xts, coef))),
                     np.max(np.abs(Ecc - Edf)[valset_bools]))
        print(coef)

        coef_sets.append(coef)
        scores.append(score)

    return coef_sets, scores


def store_mols_in_order(FNAME, ROOT, MOL_IDS, IS_RESTRICTED_LIST,
                        VAL_SET=None, mol_id_full=False,
                        functional=DEFAULT_FUNCTIONAL,
                        basis=DEFAULT_BASIS):
    from pyscf import gto
    import yaml

    dft_dirs = []
    ccsd_dirs = []
    mol_dicts = []

    for mol_id, is_restricted in zip(MOL_IDS, IS_RESTRICTED_LIST):
        print(mol_id)
        if mol_id_full:
            dft_dir = mol_id[0].replace('SCAN', functional).replace('KS/HF', 'HF')
            ccsd_dir = mol_id[1]
        else:
            if is_restricted:
                dft_dir = get_save_dir(ROOT, 'RKS', basis, mol_id,
                                       functional=functional)
                ccsd_dir = get_save_dir(ROOT, 'CCSD', basis, mol_id)
            else:
                dft_dir = get_save_dir(ROOT, 'UKS', basis, mol_id,
                                       functional=functional)
                ccsd_dir = get_save_dir(ROOT, 'UCCSD', basis, mol_id)
        if is_restricted:
            dft_analyzer = RHFAnalyzer.load(dft_dir + '/data.hdf5')
        else:
            dft_analyzer = UHFAnalyzer.load(dft_dir + '/data.hdf5')

        print(dft_dir, ccsd_dir)
        mol_dicts.append(gto.mole.pack(dft_analyzer.mol))
        dft_dirs.append(dft_dir)
        ccsd_dirs.append(ccsd_dir)
        if VAL_SET is not None:
            mol_dicts[-1].update({'valset': mol_id in VAL_SET})

    all_data = {
        'mols': mol_dicts,
        'dft_dirs': dft_dirs,
        'ccsd_dirs': ccsd_dirs,
        'is_restricted_list': IS_RESTRICTED_LIST
    }

    with open(FNAME, 'w') as f:
        yaml.dump(all_data, f)
