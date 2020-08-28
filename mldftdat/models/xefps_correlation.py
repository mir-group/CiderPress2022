from mldftdat.lowmem_analyzers import RHFAnalyzer, UHFAnalyzer, CCSDAnalyzer, UCCSDAnalyzer
from mldftdat.dft.numint5 import _eval_x_0, setup_aux
from pyscf.dft.libxc import eval_xc
from mldftdat.dft.correlation import *
from mldftdat.workflow_utils import get_save_dir
from sklearn.linear_model import LinearRegression
from pyscf.dft.numint import NumInt
from mldftdat.models.map_c2 import VSXCContribs
import os
import numpy as np

LDA_FACTOR = - 3.0 / 4.0 * (3.0 / np.pi)**(1.0/3)

DEFAULT_FUNCTIONAL = 'SCAN'
DEFAULT_BASIS = 'aug-cc-pvtz'

def get_mlx_contribs(dft_dir, restricted, mlfunc,
                     include_x = False, scanx = False):

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
        rho_data_u, rho_data_d = rho_data / 2, rho_data / 2
    else:
        rho_data_u, rho_data_d = rho_data[0], rho_data[1]

    xu = np.linalg.norm(rho_data_u[1:4], axis=0) / (rho_data_u[0]**(4.0/3) + 1e-20)
    xd = np.linalg.norm(rho_data_d[1:4], axis=0) / (rho_data_d[0]**(4.0/3) + 1e-20)
    CF = 0.3 * (6 * np.pi**2)**(2.0/3)
    zu = rho_data_u[5] / (rho_data_u[0]**(5.0/3) + 1e-20) - CF
    zd = rho_data_d[5] / (rho_data_d[0]**(5.0/3) + 1e-20) - CF
    Du = 1 - 0.125 * xu**2 / (zu + CF + 1e-20)
    Dd = 1 - 0.125 * xd**2 / (zd + CF + 1e-20)
    print(np.mean(Du), np.mean(Dd))

    FUNCTIONAL = ',LDA_C_PW_MOD'
    cu = eval_xc(FUNCTIONAL, (rho_data_u, 0 * rho_data_u), spin = 1)[0] * rho_data_u[0]
    cd = eval_xc(FUNCTIONAL, (rho_data_d, 0 * rho_data_d), spin = 1)[0] * rho_data_d[0]
    co = eval_xc(FUNCTIONAL, (rho_data_u, rho_data_d), spin = 1)[0] \
            * (rho_data_u[0] + rho_data_d[0])
    co -= cu + cd
    cu *= Du
    cd *= Dd
    FUNCTIONAL = DEFAULT_FUNCTIONAL
    Exscan = eval_xc(FUNCTIONAL, (rho_data_u, rho_data_d), spin = 1)[0] \
             * (rho_data_u[0] + rho_data_d[0])
    Exscan = np.dot(Exscan, weights)

    numint0 = ProjNumInt(xterms = [], ssterms = [], osterms = [])
    if restricted:
        if scanx:
            ex = eval_xc(',MGGA_C_SCAN', rho_data)[0]
        else:
            ex = _eval_x_0(mlfunc, mol, rho_data, grid, rdm1)[0]
        exu = ex
        exd = ex
        exo = ex
        rhou = rho_data[0]
        rhod = rho_data[0]
        rhot = rho_data[0]
        Ex = np.dot(exo * rhot, weights)
    else:
        if scanx:
            exu = eval_xc(',MGGA_C_SCAN', (rho_data[0], 0 * rho_data[0]), spin = 1)[0]
            exd = eval_xc(',MGGA_C_SCAN', (rho_data[1], 0 * rho_data[1]), spin = 1)[0]
        else:
            exu = _eval_x_0(mlfunc, mol, 2 * rho_data[0], grid, 2 * rdm1[0])[0]
            exd = _eval_x_0(mlfunc, mol, 2 * rho_data[1], grid, 2 * rdm1[1])[0]
        rhou = 2 * rho_data[0][0]
        rhod = 2 * rho_data[1][0]
        rhot = rho_data[0][0] + rho_data[1][0]
        exo = (exu * rho_data[0][0] + exd * rho_data[1][0])
        Ex = np.dot(exo, weights)
        exo /= (rhot + 1e-20)

    Eterms = np.array([Ex, Exscan])

    for rho, ex, c in zip([rhou, rhod, rhot], [exu, exd, exo], [cu, cd, co]):
        elda = LDA_FACTOR * rho**(1.0/3) - 1e-20
        Fx = ex / elda
        Etmp = np.zeros(5)
        x1 = (1 - Fx**6) / (1 + Fx**6)
        for i in range(5):
            Etmp[i] = np.dot(c * x1**i, weights)
        Eterms = np.append(Eterms, Etmp)

    if include_x:
        for rho, ex in zip([rhou, rhod], [exu, exd]):
            elda = LDA_FACTOR * rho**(1.0/3) - 1e-20
            Fx = ex / elda
            Etmp = np.zeros(5)
            x1 = (1 - Fx**6) / (1 + Fx**6)
            for i in range(5):
                Etmp[i] = np.dot(elda * rho / 2 * x1**i, weights)
            Eterms = np.append(Eterms, Etmp)

    print(Eterms.shape)

    # Eterms = Eu, Ed, Eo
    return Eterms

def store_mlx_contribs_dataset(FNAME, ROOT, MOL_IDS, IS_RESTRICTED_LIST,
                               MLFUNC, include_x = False, scanx = False):

    SIZE = 27 if include_x else 17
    X = np.zeros([0,SIZE])

    for mol_id, is_restricted in zip(MOL_IDS, IS_RESTRICTED_LIST):

        print(mol_id)

        if is_restricted:
            dft_dir = get_save_dir(ROOT, 'RKS', DEFAULT_BASIS,
                mol_id, functional = DEFAULT_FUNCTIONAL)
        else:
            dft_dir = get_save_dir(ROOT, 'UKS', DEFAULT_BASIS,
                mol_id, functional = DEFAULT_FUNCTIONAL)

        sl_contribs = get_mlx_contribs(dft_dir, is_restricted,
                                       MLFUNC, include_x = include_x,
                                       scanx = scanx)

        X = np.vstack([X, sl_contribs])

    np.save(FNAME, X)


def get_mn_contribs(dft_dir, restricted, include_x = False):

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
        rho_data_u, rho_data_d = rho_data / 2, rho_data / 2
    else:
        rho_data_u, rho_data_d = rho_data[0], rho_data[1]
    rho_data_t = rho_data_u + rho_data_d

    xu = np.linalg.norm(rho_data_u[1:4], axis=0) / (rho_data_u[0]**(4.0/3) + 1e-10)
    xd = np.linalg.norm(rho_data_d[1:4], axis=0) / (rho_data_d[0]**(4.0/3) + 1e-10)
    xo = np.linalg.norm(rho_data_t[1:4], axis=0) / (rho_data_t[0]**(4.0/3) + 1e-10)
    CF = 0.3 * (6 * np.pi**2)**(2.0/3)
    zu = rho_data_u[5] / (rho_data_u[0]**(5.0/3) + 1e-10) - CF
    zd = rho_data_d[5] / (rho_data_d[0]**(5.0/3) + 1e-10) - CF
    zo = rho_data_t[5] / (rho_data_t[0]**(5.0/3) + 1e-10) - CF / 2**(2.0/3)
    zu *= 2
    zd *= 2
    #alpha_ss, alpha_os = 0.005151, 0.003050 * 2**(2.0/3)
    alpha_x = 0.001867
    alpha_ss, alpha_os = 0.00515088, 0.00304966
    gamma_ss, gamma_os = 0.06, 0.0031
    Du = 1 - 0.25 * xu**2 / (zu + 2 * CF + 1e-10)
    Dd = 1 - 0.25 * xd**2 / (zd + 2 * CF + 1e-10)
    #Du = rho_data_u[5] * rho_data_u[0] - np.linalg.norm(rho_data_u[1:4], axis=0)**2 / 8
    #Dd = rho_data_d[5] * rho_data_d[0] - np.linalg.norm(rho_data_d[1:4], axis=0)**2 / 8
    #Du /= rho_data_u[5] * rho_data_u[0] + 1e-20
    #Dd /= rho_data_d[5] * rho_data_d[0] + 1e-20
    print(np.std(zu - zd), np.std(xu - xd))
    print(np.mean(Du), np.mean(Dd))
    dvals = np.zeros((18, weights.shape[0]))
    start = 0
    def gamma_func(x2, z, alpha):
        return 1 + alpha * (x2 + z)
    for x2, z, alpha in [(xu**2, zu, alpha_ss), (xd**2, zd, alpha_ss),\
                        ((xu**2+xd**2), (zu+zd), alpha_os)]:
        gamma = gamma_func(x2, z, alpha)
        dvals[start+0] = 1 / gamma - 1
        dvals[start+1] = x2 / gamma**2
        dvals[start+2] = z / gamma**2
        dvals[start+3] = x2**2 / gamma**3
        dvals[start+4] = x2 * z / gamma**3
        dvals[start+5] = z**2 / gamma**3
        start += 6
    cvals = np.zeros((15, weights.shape[0]))
    start = 0
    for x2, gamma in [(xu**2, gamma_ss), (xd**2, gamma_ss), ((xu**2+xd**2), gamma_os)]:
        u = gamma * x2 / (1 + gamma * x2)
        for i in range(5):
            cvals[start+i] = u**i
        start += 5

    FUNCTIONAL = ',LDA_C_PW_MOD'
    cu, vu = eval_xc(FUNCTIONAL, (rho_data_u, 0 * rho_data_u), spin = 1)[:2]
    cd, vd = eval_xc(FUNCTIONAL, (rho_data_d, 0 * rho_data_d), spin = 1)[:2]
    co, vo = eval_xc(FUNCTIONAL, (rho_data_u, rho_data_d), spin = 1)[:2]
    cu *= rho_data_u[0]
    cd *= rho_data_d[0]
    co *= (rho_data_u[0] + rho_data_d[0])
    vuu = vu[0][:,0]
    vdd = vd[0][:,0]
    vou = vo[0][:,0] - vuu
    vod = vo[0][:,1] - vdd
    co -= cu + cd
    cutmp = cu.copy()
    cdtmp = cd.copy()
    cu *= Du
    cd *= Dd
    ctst, vtst = eval_xc(',MGGA_C_M06_L', (rho_data_u, rho_data_d), spin = 1)[:2] 
    ctst *= rho_data_t[0]
    
    dvals[:6] *= cu
    dvals[6:12] *= cd
    dvals[12:] *= co
    cvals[:5] *= cu
    cvals[5:10] *= cd
    cvals[10:] *= co
    ccoef = [1.0, 5.396620e-1, -3.161217e+1, 5.149592e+1, -2.919613e+1,\
             1.0, 1.776783e+2, -2.513252e+2, 7.635173e+1, -1.255699e+1]
    dcoef = [4.650534e-1, 1.617589e-1, 1.833657e-1, 4.692100e-4, -4.990573e-3, 0,\
             3.957626e-1, -5.614546e-1, 1.403963e-2, 9.831442e-4, -3.577176e-3, 0]
    #coef = [5.349466e-01,  5.396620e-01, -3.161217e+01,  5.149592e+01, -2.919613e+01,
    #6.042374e-01,  1.776783e+02, -2.513252e+02,  7.635173e+01, -1.255699e+01,
    #4.650534e-01,  1.617589e-01,  1.833657e-01,  4.692100e-04, -4.990573e-03,  0.000000e+00,
    #3.957626e-01, -5.614546e-01,  1.403963e-02,  9.831442e-04, -3.577176e-03,  0.000000e+00] 
    #ccoef, dcoef = coef[:10], coef[10:]
    ccoef = ccoef[:5] * 2 + ccoef[5:]
    dcoef = dcoef[:6] * 2 + dcoef[6:]
    Epw0 = np.dot(cu+cd+co, weights)

    dvals = np.dot(dvals, weights)
    cvals = np.dot(cvals, weights)
    print(np.dot(dvals, dcoef) + np.dot(cvals, ccoef), np.dot(ctst, weights))
    #print(np.dot(dvals[6:12], dcoef[6:12]) + np.dot(cvals[5:10], ccoef[5:10]), np.dot(ctst, weights))

    g2u = np.einsum('ir,ir->r', rho_data_u[1:4], rho_data_u[1:4])
    g2d = np.einsum('ir,ir->r', rho_data_d[1:4], rho_data_d[1:4])
    corr_model = VSXCContribs(None, None, None, None, dcoef[:6], dcoef[6:])
    tot, uderiv, dderiv = corr_model.corr_mn(cutmp, cdtmp, co, vuu, vdd, vou, vod,
                                             rho_data_u[0], rho_data_d[0],
                                             g2u, g2d, rho_data_u[5], rho_data_d[5])
    print('TEST VSXC CONTRIBS')
    print(np.dot(tot, weights), np.dot(ctst, weights))
    print(np.dot(uderiv[0], rho_data_u[0] * weights),
        np.dot(vtst[0][:,0], rho_data_u[0] * weights))
    print(np.dot(dderiv[0], rho_data_d[0] * weights),
        np.dot(vtst[0][:,1], rho_data_d[0] * weights))

    if include_x:
        xvals = np.zeros((12,weights.shape[0]))
        start = 0
        for rho, x2, z, alpha in [(rho_data_u[0], xu**2, zu, alpha_x),\
                                  (rho_data_d[0], xd**2, zd, alpha_x)]:
            gamma = gamma_func(x2, z, alpha)
            xvals[start+0] = 1 / gamma - 1
            xvals[start+1] = x2 / gamma**2
            xvals[start+2] = z / gamma**2
            xvals[start+3] = x2**2 / gamma**3
            xvals[start+4] = x2 * z / gamma**3
            xvals[start+5] = z**2 / gamma**3
            xvals[start:start+6] *= LDA_FACTOR * 2**(1.0/3) * rho**(4.0/3)
            start += 6
        xvals = np.dot(xvals, weights)
        return np.concatenate([dvals[:6] + dvals[6:12], dvals[12:],\
                               cvals[:5] + cvals[5:10], cvals[10:],\
                               xvals[:6] + xvals[6:],\
                               [dft_analyzer.fx_total]], axis=0)
    else:
        return np.concatenate([dvals[:6] + dvals[6:12], dvals[12:],\
                               cvals[:5] + cvals[5:10], cvals[10:],\
                               [dft_analyzer.fx_total]], axis=0)

def store_mn_contribs_dataset(FNAME, ROOT, MOL_IDS, IS_RESTRICTED_LIST,
                              include_x = False):

    SIZE = 29 if include_x else 23
    X = np.zeros([0,SIZE])

    for mol_id, is_restricted in zip(MOL_IDS, IS_RESTRICTED_LIST):

        print(mol_id)

        if is_restricted:
            dft_dir = get_save_dir(ROOT, 'RKS', DEFAULT_BASIS,
                mol_id, functional = DEFAULT_FUNCTIONAL)
        else:
            dft_dir = get_save_dir(ROOT, 'UKS', DEFAULT_BASIS,
                mol_id, functional = DEFAULT_FUNCTIONAL)

        sl_contribs = get_mn_contribs(dft_dir, is_restricted,
                                      include_x = include_x)

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

def store_total_energies_dataset(FNAME, ROOT, MOL_IDS, IS_RESTRICTED_LIST):

    # PBE, CCSD
    y = np.zeros([0, 2])

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
    mlx = np.load(os.path.join(DATA_ROOT, 'mlx6c.npy'))
    mnc = np.load(os.path.join(DATA_ROOT, 'mnc.npy'))
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
        #E_x = mnc[:,-1]
        E_xscan = mlx[:,1]
        #print(E_x)
        #print(E_xscan)
        #print(E_x - E_xscan)
        #print(E_ccsd - E_dft)
        #print(E_vv10)
        E_c = np.append(mlx[:,3:7] + mlx[:,8:12], mlx[:,13:17], axis=1)
        E_c = np.append(E_c, mlx[:,18:22] + mlx[:,23:27], axis=1)
        E_c = np.append(E_c, mnc[:,:12], axis=1)
        E_c = np.append(E_c, mnc[:,-7:-1], axis=1)
        #E_c = E_c[:,-18:]
        #E_c = mnc[:,:12]
        print("SHAPE", E_c.shape)

        #diff = E_ccsd - (E_dft - E_xscan + E_x + E_vv10 + mlx[:,2] + mlx[:,7] + mlx[:,12])
        diff = E_ccsd - (E_dft - E_xscan + E_x + mlx[:,2] + mlx[:,7] + mlx[:,12])

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
                weights.append(1.0)
                formula = formulas[i]
                for Z in list(formula.keys()):
                    X[i,:] -= formula[Z] * X[Z_to_ind[Z],:]
                    y[i] -= formula[Z] * y[Z_to_ind[Z]]
                    Ecc[i] -= formula[Z] * Ecc[Z_to_ind[Z]]
                    Edf[i] -= formula[Z] * Edf[Z_to_ind[Z]]
                print(formulas[i], y[i], Ecc[i], Edf[i], E_x[i] - E_xscan[i])
            else:
                weights.append(1.0 / mols[i].nelectron if mols[i].nelectron <= 18 else 0)
                #weights.append(0.0)

        weights = np.array(weights)

        print(np.mean(np.abs(Ecc-Edf)[weights > 0]))

        X = X[weights > 0, :]
        y = y[weights > 0]
        Ecc = Ecc[weights > 0]
        Edf = Edf[weights > 0]
        weights = weights[weights > 0]

        noise = 1e-3
        A = np.linalg.inv(np.dot(X.T * weights, X) + noise * np.identity(X.shape[1]))
        B = np.dot(X.T, weights * y)
        coef = np.dot(A, B)

        score = r2_score(y, np.dot(X, coef))
        score0 = r2_score(y, np.dot(X, 0 * coef))
        print(score, score0)
        print(y - np.dot(X, coef))
        print('SCAN', np.mean(np.abs(Ecc-Edf)[weights > 0]), np.mean((Ecc-Edf)[weights > 0]))
        print('ML', np.mean(np.abs(y - np.dot(X, coef))), np.mean(y - np.dot(X, coef)))
        print(np.max(np.abs(y - np.dot(X, coef))), np.max(np.abs(Ecc - Edf)))

        coef_sets.append(coef)
        scores.append(score)

    return coef_sets, scores


def store_mols_in_order(FNAME, ROOT, MOL_IDS, IS_RESTRICTED_LIST):
    from pyscf import gto
    import yaml

    mol_dicts = []

    for mol_id, is_restricted in zip(MOL_IDS, IS_RESTRICTED_LIST):

        if is_restricted:
            pbe_dir = get_save_dir(ROOT, 'RKS', 'aug-cc-pvtz', mol_id, functional = 'PBE')
            pbe_analyzer = RHFAnalyzer.load(pbe_dir + '/data.hdf5')
        else:
            pbe_dir = get_save_dir(ROOT, 'UKS', 'aug-cc-pvtz', mol_id, functional = 'PBE')
            pbe_analyzer = UHFAnalyzer.load(pbe_dir + '/data.hdf5')

        mol_dicts.append(gto.mole.pack(pbe_analyzer.mol))

    with open(FNAME, 'w') as f:
        yaml.dump(mol_dicts, f)
