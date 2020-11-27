from mldftdat.lowmem_analyzers import RHFAnalyzer, UHFAnalyzer, CCSDAnalyzer, UCCSDAnalyzer
from mldftdat.dft.numint5 import _eval_x_0, setup_aux
from mldftdat.dft.numint6 import _eval_xc_0
from pyscf.dft.libxc import eval_xc
from mldftdat.dft.correlation import *
from mldftdat.workflow_utils import get_save_dir, SAVE_ROOT
from sklearn.linear_model import LinearRegression
from pyscf.dft.numint import NumInt
from mldftdat.models.map_c2 import VSXCContribs
from mldftdat.density import get_exchange_descriptors2
import os
import numpy as np
import yaml

LDA_FACTOR = - 3.0 / 4.0 * (3.0 / np.pi)**(1.0/3)

DEFAULT_FUNCTIONAL = 'SCAN'
#DEFAULT_FUNCTIONAL = 'SGXCORR_ALPHA3'
#DEFAULT_BASIS = 'aug-cc-pvtz'
DEFAULT_BASIS = 'def2-qzvppd'

CF = 0.3 * (6 * np.pi**2)**(2.0/3)

def get_mlx_contribs(dft_dir, restricted, mlfunc,
                     include_x = False, scanx = False, use_sf = False):

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

    corr_model = VSXCContribs(None, None, None, None, None, None)

    if include_x:
        if use_sf:
            g2u = np.einsum('ir,ir->r', rho_data_u[1:4], rho_data_u[1:4])
            g2d = np.einsum('ir,ir->r', rho_data_d[1:4], rho_data_d[1:4])
            g2o = np.einsum('ir,ir->r', rho_data_u[1:4], rho_data_d[1:4])
            sf, dsfdnu, dsfdnd, dsfdg2, dsfdt = \
                corr_model.spinpol_factor(rho_data_u[0] + 1e-20, rho_data_d[0] + 1e-20,
                                          g2u + 2 * g2o + g2d,
                                          rho_data_u[5] + rho_data_d[5] + 1e-20)
        for rho, ex in zip([rhou, rhod], [exu, exd]):
            elda = LDA_FACTOR * rho**(1.0/3) - 1e-20
            Fx = ex / elda
            Etmp = np.zeros(5)
            x1 = (1 - Fx**6) / (1 + Fx**6)
            if use_sf:
                for i in range(5):
                    Etmp[i] = np.dot(elda * sf * rho / 2 * x1**i, weights)
            else:
                for i in range(5):
                    Etmp[i] = np.dot(elda * rho / 2 * x1**i, weights)
            Eterms = np.append(Eterms, Etmp)

    print(Eterms.shape)

    # Eterms = Eu, Ed, Eo
    return Eterms

def store_mlx_contribs_dataset(FNAME, ROOT, MOL_IDS, IS_RESTRICTED_LIST,
                               MLFUNC, include_x = False, scanx = False,
                               use_sf = False):

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
                                       scanx = scanx, use_sf = use_sf)

        X = np.vstack([X, sl_contribs])

    np.save(FNAME, X)


def get_mn_contribs(dft_dir, restricted, include_x = False, use_sf = False):

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

    xu = np.linalg.norm(rho_data_u[1:4], axis=0) / (rho_data_u[0]**(4.0/3) + 1e-20)
    xd = np.linalg.norm(rho_data_d[1:4], axis=0) / (rho_data_d[0]**(4.0/3) + 1e-20)
    CF = 0.3 * (6 * np.pi**2)**(2.0/3)
    zu = rho_data_u[5] / (rho_data_u[0]**(5.0/3) + 1e-20) - CF
    zd = rho_data_d[5] / (rho_data_d[0]**(5.0/3) + 1e-20) - CF
    zu *= 2
    zd *= 2
    Du = 1 - np.linalg.norm(rho_data_u[1:4], axis=0)**2 / (8 * rho_data_u[0] * rho_data_u[5] + 1e-20)
    Dd = 1 - np.linalg.norm(rho_data_d[1:4], axis=0)**2 / (8 * rho_data_d[0] * rho_data_d[5] + 1e-20)
    #alpha_ss, alpha_os = 0.005151, 0.003050 * 2**(2.0/3)
    alpha_x = 0.001867
    alpha_ss, alpha_os = 0.00515088, 0.00304966
    gamma_ss, gamma_os = 0.06, 0.0031
    #Du = 1 - 0.25 * xu**2 / (zu + 2 * CF + 1e-10)
    #Dd = 1 - 0.25 * xd**2 / (zd + 2 * CF + 1e-10)
    dvals = np.zeros((18, weights.shape[0]))
    start = 0
    def gamma_func(x2, z, alpha):
        return 1 + alpha * (x2 + z)
    for x2, z, alpha in [(xu**2, zu, alpha_ss), (xd**2, zd, alpha_ss),\
                        (xu**2+xd**2, zu+zd, alpha_os)]:
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
    for x2, gamma in [(xu**2, gamma_ss), (xd**2, gamma_ss), (xu**2+xd**2, gamma_os)]:
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
    g2o = np.einsum('ir,ir->r', rho_data_u[1:4], rho_data_d[1:4])
    corr_model = VSXCContribs(None, None, None, None, dcoef[:6], dcoef[12:],
            None, ccoef[1:5], ccoef[-4:])
    tot, uderiv, dderiv = corr_model.corr_mn(cutmp, cdtmp, co, vuu, vdd, vou, vod,
                                             rho_data_u[0] + 1e-20, rho_data_d[0] + 1e-20,
                                             g2u, g2d, rho_data_u[5] + 1e-20, rho_data_d[5] + 1e-20)
    totb, uderivb, dderivb = corr_model.corr_mnexp(cutmp, cdtmp, co, vuu, vdd, vou, vod,
                                             rho_data_u[0] + 1e-20, rho_data_d[0] + 1e-20,
                                             g2u, g2d, rho_data_u[5] + 1e-20, rho_data_d[5] + 1e-20)
    print('TEST VSXC CONTRIBS')
    nu = rho_data_u[0]
    nd = rho_data_d[0]
    print(np.mean(zu*nu), np.mean(zd*nd), np.mean(xu**2*nu), np.mean(xd**2*nd), np.mean(Du*nu), np.mean(Dd*nd))
    print(np.dot(tot + totb, weights), np.dot(ctst, weights))
    for i in range(3):
        j = 2 if i == 1 else 1
        k = 3 if i == 2 else i
        print(i)
        print(np.dot(uderiv[i] + uderivb[i], rho_data_u[0] * weights),
            np.dot(vtst[k][:,0], rho_data_u[0] * weights))
        print(np.dot(dderiv[i] + dderivb[i], rho_data_d[0] * weights),
            np.dot(vtst[k][:,j], rho_data_d[0] * weights))

    if include_x:
        if use_sf:
            sf, dsfdnu, dsfdnd, dsfdg2, dsfdt = \
                corr_model.spinpol_factor(rho_data_u[0] + 1e-20, rho_data_d[0] + 1e-20,
                                          g2u + 2 * g2o + g2d,
                                          rho_data_u[5] + rho_data_d[5] + 1e-20)
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
            if use_sf:
                xvals[start:start+6] *= sf
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
                              include_x = False, use_sf = False):

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
                                      include_x = include_x,
                                      use_sf = use_sf)

        X = np.vstack([X, sl_contribs])

    np.save(FNAME, X)


def get_full_contribs(dft_dir, restricted, mlfunc, exact=True):

    from mldftdat.models import map_c6

    corr_model = map_c6.VSXCContribs(None, None, None, None, None,
                                     None, None, None, None, None,
                                     fterm_scale=2.0)

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
    rsu = corr_model.get_rs(rhou)[0]
    rsd = corr_model.get_rs(rhod)[0]
    rs = corr_model.get_rs(rhot)[0]

    zeta = (rhou - rhod) / (rhot)
    ds = ((1-zeta)**(5.0/3) + (1+zeta)**(5.0/3))/2
    CU = 0.3 * (3 * np.pi**2)**(2.0/3)

    rho_data_u_0 = rho_data_u.copy()
    rho_data_u_1 = rho_data_u.copy()
    rho_data_u_0[4] = 0
    rho_data_u_0[5] = g2 / (8 * rhot)
    rho_data_u_1[4] = 0
    rho_data_u_1[5] = CU * ds * rhot**(5.0/3) + rho_data_u_0[5]

    rho_data_d_0 = rho_data_d.copy()
    rho_data_d_1 = rho_data_d.copy()
    rho_data_d_0[4] = 0
    rho_data_d_0[5] = 0#g2d / (8 * rhod)
    rho_data_d_1[4] = 0
    rho_data_d_1[5] = 0#CF * rhod**(5.0/3) + rho_data_d_0[5]

    tstldau = corr_model.pw92term(rsu, 1)[0]
    tstldad = corr_model.pw92term(rsd, 1)[0]
    tstldao = corr_model.pw92(rs, zeta)[0]
    co0, vo0 = corr_model.os_baseline(rhou, rhod, g2, type=0)[:2]
    co1, vo1 = corr_model.os_baseline(rhou, rhod, g2, type=1)[:2]
    cu1, vu1 = corr_model.ss_baseline(rhou, g2u)[:2]
    cd1, vd1 = corr_model.ss_baseline(rhod, g2d)[:2]
    ecldau = eval_xc('LDA_C_PW_MOD', (rhou, 0*rhou), spin=1)[0]
    ecldad = eval_xc('LDA_C_PW_MOD', (rhod, 0*rhod), spin=1)[0]
    ecldao = eval_xc('LDA_C_PW_MOD', (rhou, rhod), spin=1)[0]
    escan0 = eval_xc('MGGA_C_SCAN', (rho_data_u_0, rho_data_d_0), spin=1)[0]
    escan1 = eval_xc('MGGA_C_SCAN', (rho_data_u_1, rho_data_d_1), spin=1)[0]
    print(np.dot(ecldau, rhou*weights), np.dot(tstldau, rhou*weights))
    print(np.dot(ecldad, rhod*weights), np.dot(tstldad, rhod*weights))
    print(np.dot(ecldao, rhot*weights), np.dot(tstldao, rhot*weights))
    print(np.dot(escan0, rhot*weights), np.dot(co0, rhot*weights))
    print(np.dot(escan1, rhot*weights), np.dot(co1, rhot*weights))
    co0 *= rhot
    cu1 *= rhou
    cd1 *= rhod
    co1 = co1 * rhot - cu1 - cd1
    cu = cu1
    cd = cd1
    co = co1
    cx = co0
    ct = cu1 + cd1 + co1
    A = 2.74
    B = 132
    sprefac = 2 * (3 * np.pi**2)**(1.0/3)

    nu, nd = rhou, rhod
    x2u = corr_model.get_x2(nu, g2u)[0]
    x2d = corr_model.get_x2(nd, g2d)[0]
    x2o = corr_model.get_x2((nu+nd)/2**(1.0/3), g2)[0]
    zu = corr_model.get_z(nu, tu)[0]
    zd = corr_model.get_z(nd, td)[0]
    zo = corr_model.get_z((nu+nd)/2**(2.0/3), tu+td)[0]
    s2 = x2o / sprefac**2
    Du = corr_model.get_D(rhou, g2u, tu)[0]
    Dd = corr_model.get_D(rhod, g2d, td)[0]
    Do = corr_model.get_D(rhot, g2u + 2 * g2o + g2d, tu+td)
    fDo = Do
    Do = fDo[0]
    amix = corr_model.get_amix(rhou, rhod, g2, fDo)[0]

    alpha_x = 0.001867
    alpha_ss, alpha_os = 0.00515088, 0.00304966
    dvals = np.zeros((30, weights.shape[0]))
    start = 0

    def gamma_func(x2, z, alpha):
        return 1 + alpha * (x2 + z)
    for x2, z, alpha in [(x2u, zu, alpha_ss), (x2d, zd, alpha_ss),\
                        (x2o, zo, alpha_os),\
                        (x2o, zo, alpha_os),\
                        (x2o, zo, alpha_os)]:
        gamma = gamma_func(x2, z, alpha)
        dvals[start:start+6] = corr_model.get_separate_corrfunc_terms(x2, z, gamma)
        start += 6
    dvals[:6] *= cu * Do
    dvals[6:12] *= cd * Do
    dvals[12:18] *= ct * Do
    dvals[18:24] *= cx
    dvals[24:30] *= cx * (1 - Do)
    dvals = np.dot(dvals, weights)
    dvals = np.append(dvals[:6] + dvals[6:12], dvals[12:])
    cvals = np.zeros((25, weights.shape[0]))
    start = 0
    gamma_ss, gamma_os = 0.06, 0.0031
    for x2, gamma in [(x2u, gamma_ss), (x2d, gamma_ss),\
                      (x2o, gamma_os),\
                      (x2o, gamma_os),\
                      (x2o, gamma_os)]:
        u = gamma * x2 / (1 + gamma * x2)
        for i in range(5):
            cvals[start+i] = u**i
        start += 5
    cvals[:5] *= cu * Du
    cvals[5:10] *= cd * Dd
    cvals[15:20] *= cx
    cvals[10:15] *= ct * Do
    cvals[20:25] *= cx * (1 - Do)
    cvals = np.dot(cvals, weights)
    cvals = np.append(cvals[:5] + cvals[5:10], cvals[10:])

    numint0 = ProjNumInt(xterms = [], ssterms = [], osterms = [])
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
        rhou = rho_data[0]
        rhod = rho_data[0]
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
        rhou = 2 * rho_data[0][0]
        rhod = 2 * rho_data[1][0]
        rhot = rho_data[0][0] + rho_data[1][0]
        exo = (exu * rho_data[0][0] + exd * rho_data[1][0])
        Ex = np.dot(exo, weights)
        exo /= (rhot + 1e-20)

    FUNCTIONAL = DEFAULT_FUNCTIONAL
    Exscan = eval_xc(FUNCTIONAL, (rho_data_u, rho_data_d), spin = 1)[0] \
             * (rho_data_u[0] + rho_data_d[0])
    Exscan = np.dot(Exscan, weights)

    print('EX ERROR', Ex - dft_analyzer.fx_total, Ex, dft_analyzer.fx_total)
    if (np.abs(Ex - dft_analyzer.fx_total) > 1e-7):
        print('LARGE ERROR')
    #assert np.abs(Ex - dft_analyzer.fx_total) < 1e-4

    Eterms = np.array([Ex, Exscan])

    for rho, ex, c in zip([rhou, rhod, rhot, rhot, rhot], [exu, exd, exo, exo, exo],
                          [cu * Du, cd * Dd, ct * Do, cx, cx * (1 - Do)]):
        elda = LDA_FACTOR * rho**(1.0/3) - 1e-20
        Fx = ex / elda
        E_tmp = corr_model.get_separate_xef_terms(Fx)
        E_tmp *= c
        E_tmp = np.dot(E_tmp, weights)
        Eterms = np.append(Eterms, E_tmp)

    Fterms = np.array([])

    for rho, ex in zip([rhou, rhod], [exu, exd]):
            elda = LDA_FACTOR * rho**(1.0/3) - 1e-20
            Fx = ex / elda
            E_tmp = corr_model.get_separate_xef_terms(Fx)
            E_tmp *= elda * amix * rho / 2
            E_tmp = np.dot(E_tmp, weights)
            Fterms = np.append(Fterms, E_tmp)
    print("FTERMS", Fterms)

    Fterms2 = np.array([])

    for rho, ex in zip([rhou, rhod], [exu, exd]):
            elda = LDA_FACTOR * rho**(1.0/3) - 1e-20
            Fx = ex / elda
            E_tmp = corr_model.get_separate_xef_terms(Fx)
            E_tmp *= elda * rho / 2
            E_tmp = np.dot(E_tmp, weights)
            Fterms2 = np.append(Fterms2, E_tmp)

    xvals = np.zeros((12,weights.shape[0]))
    start = 0
    for rho, x2, z, alpha in [(rho_data_u[0], x2u, zu, alpha_x),\
                              (rho_data_d[0], x2d, zd, alpha_x)]:
        gamma = gamma_func(x2, z, alpha)
        xvals[start:start+6] = corr_model.get_separate_corrfunc_terms(x2, z, gamma)
        xvals[start:start+6] *= LDA_FACTOR * 2**(1.0/3) * amix * rho**(4.0/3)
        start += 6
    xvals = np.dot(xvals, weights)

    #                      25      10      24     12,    10
    return np.concatenate([Eterms, Fterms, dvals, xvals, Fterms2,
                          [dft_analyzer.fx_total]], axis=0)

def get_full_contribs2(dft_dir, restricted, mlfunc, exact=True):

    from mldftdat.models import map_c6

    corr_model = map_c6.VSXCContribs(None, None, None, None, None,
                                     None, None, None, None, None,
                                     fterm_scale=1.0)

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
    rsu = corr_model.get_rs(rhou)[0]
    rsd = corr_model.get_rs(rhod)[0]
    rs = corr_model.get_rs(rhot)[0]

    zeta = (rhou - rhod) / (rhot)
    ds = ((1-zeta)**(5.0/3) + (1+zeta)**(5.0/3))/2
    CU = 0.3 * (3 * np.pi**2)**(2.0/3)

    rho_data_u_0 = rho_data_u.copy()
    rho_data_u_1 = rho_data_u.copy()
    rho_data_u_0[4] = 0
    rho_data_u_0[5] = g2 / (8 * rhot)
    rho_data_u_1[4] = 0
    rho_data_u_1[5] = CU * ds * rhot**(5.0/3) + rho_data_u_0[5]

    rho_data_d_0 = rho_data_d.copy()
    rho_data_d_1 = rho_data_d.copy()
    rho_data_d_0[4] = 0
    rho_data_d_0[5] = 0#g2d / (8 * rhod)
    rho_data_d_1[4] = 0
    rho_data_d_1[5] = 0#CF * rhod**(5.0/3) + rho_data_d_0[5]

    tstldau = corr_model.pw92term(rsu, 1)[0]
    tstldad = corr_model.pw92term(rsd, 1)[0]
    tstldao = corr_model.pw92(rs, zeta)[0]
    co0, vo0 = corr_model.os_baseline(rhou, rhod, g2, type=0)[:2]
    co1, vo1 = corr_model.os_baseline(rhou, rhod, g2, type=1)[:2]
    cu1, vu1 = corr_model.ss_baseline(rhou, g2u)[:2]
    cd1, vd1 = corr_model.ss_baseline(rhod, g2d)[:2]
    ecldau = eval_xc('LDA_C_PW_MOD', (rhou, 0*rhou), spin=1)[0]
    ecldad = eval_xc('LDA_C_PW_MOD', (rhod, 0*rhod), spin=1)[0]
    ecldao = eval_xc('LDA_C_PW_MOD', (rhou, rhod), spin=1)[0]
    escan0 = eval_xc('MGGA_C_SCAN', (rho_data_u_0, rho_data_d_0), spin=1)[0]
    escan1 = eval_xc('MGGA_C_SCAN', (rho_data_u_1, rho_data_d_1), spin=1)[0]
    print(np.dot(ecldau, rhou*weights), np.dot(tstldau, rhou*weights))
    print(np.dot(ecldad, rhod*weights), np.dot(tstldad, rhod*weights))
    print(np.dot(ecldao, rhot*weights), np.dot(tstldao, rhot*weights))
    print(np.dot(escan0, rhot*weights), np.dot(co0, rhot*weights))
    print(np.dot(escan1, rhot*weights), np.dot(co1, rhot*weights))
    co0 *= rhot
    cu1 *= rhou
    cd1 *= rhod
    co1 = co1 * rhot - cu1 - cd1
    cu = cu1
    cd = cd1
    co = co1
    cx = co0
    ct = cu1 + cd1 + co1
    A = 2.74
    B = 132
    sprefac = 2 * (3 * np.pi**2)**(1.0/3)

    nu, nd = rhou, rhod
    x2u = corr_model.get_x2(nu, g2u)[0]
    x2d = corr_model.get_x2(nd, g2d)[0]
    x2o = corr_model.get_x2((nu+nd)/2**(1.0/3), g2)[0]
    zu = corr_model.get_z(nu, tu)[0]
    zd = corr_model.get_z(nd, td)[0]
    zo = corr_model.get_z((nu+nd)/2**(2.0/3), tu+td)[0]
    s2 = x2o / sprefac**2
    Du = corr_model.get_D(rhou, g2u, tu)[0]
    Dd = corr_model.get_D(rhod, g2d, td)[0]
    Do = corr_model.get_D(rhot, g2u + 2 * g2o + g2d, tu+td)
    fDo = Do
    Do = fDo[0]
    amix = corr_model.get_amix(rhou, rhod, g2, fDo)[0]

    alpha_x = 0.001867
    alpha_ss, alpha_os = 0.00515088, 0.00304966
    dvals = np.zeros((30, weights.shape[0]))
    start = 0

    def gamma_func(x2, z, alpha):
        return 1 + alpha * (x2 + z)
    for x2, z, alpha in [(x2u, zu, alpha_ss), (x2d, zd, alpha_ss),\
                        (x2o, zo, alpha_os),\
                        (x2o, zo, alpha_os),\
                        (x2o, zo, alpha_os)]:
        gamma = gamma_func(x2, z, alpha)
        dvals[start:start+6] = corr_model.get_separate_corrfunc_terms(x2, z, gamma)
        start += 6
    dvals[:6] *= cu * Do
    dvals[6:12] *= cd * Do
    dvals[12:18] *= ct * Do
    dvals[18:24] *= cx
    dvals[24:30] *= cx * (1 - Do)
    dvals = np.dot(dvals, weights)
    dvals = np.append(dvals[:6] + dvals[6:12], dvals[12:])
    cvals = np.zeros((25, weights.shape[0]))
    start = 0
    gamma_ss, gamma_os = 0.06, 0.0031
    for x2, gamma in [(x2u, gamma_ss), (x2d, gamma_ss),\
                      (x2o, gamma_os),\
                      (x2o, gamma_os),\
                      (x2o, gamma_os)]:
        u = gamma * x2 / (1 + gamma * x2)
        for i in range(5):
            cvals[start+i] = u**i
        start += 5
    cvals[:5] *= cu * Du
    cvals[5:10] *= cd * Dd
    cvals[15:20] *= cx
    cvals[10:15] *= ct * Do
    cvals[20:25] *= cx * (1 - Do)
    cvals = np.dot(cvals, weights)
    cvals = np.append(cvals[:5] + cvals[5:10], cvals[10:])

    numint0 = ProjNumInt(xterms = [], ssterms = [], osterms = [])
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

    FUNCTIONAL = DEFAULT_FUNCTIONAL
    Exscan = eval_xc(FUNCTIONAL, (rho_data_u, rho_data_d), spin = 1)[0] \
             * (rho_data_u[0] + rho_data_d[0])
    Exscan = np.dot(Exscan, weights)
    Ecscan = eval_xc(',MGGA_C_REVSCAN', (rho_data_u, rho_data_d), spin = 1)[0] \
                     * (rho_data_u[0] + rho_data_d[0])
    Ecscan = np.dot(Ecscan, weights)

    print('EX ERROR', Ex - dft_analyzer.fx_total, Ex, dft_analyzer.fx_total)
    if (np.abs(Ex - dft_analyzer.fx_total) > 1e-7):
        print('LARGE ERROR')
    #assert np.abs(Ex - dft_analyzer.fx_total) < 1e-4

    Eterms = np.array([Ex, Exscan])

    exu = exu * rhou
    exd = exd * rhod
    exo = exo * rhot
    ldaxu = 2**(1.0/3) * LDA_FACTOR * rhou**(4.0/3) - 1e-20
    ldaxd = 2**(1.0/3) * LDA_FACTOR * rhod**(4.0/3) - 1e-20
    ldaxt = ldaxu + ldaxd

    for elda, ex, c in zip([ldaxu, ldaxd, ldaxt, ldaxt, ldaxt],
                           [exu, exd, exo, exo, exo],
                           [cu * Du, cd * Dd, ct * Do, cx, cx * (1 - Do)]):
        Fx = ex / elda
        E_tmp = corr_model.get_separate_xef_terms(Fx)
        E_tmp *= c
        E_tmp = np.dot(E_tmp, weights)
        Eterms = np.append(Eterms, E_tmp)

    Fterms = np.array([])

    for elda, ex in zip([ldaxu, ldaxd], [exu, exd]):
        Fx = ex / elda
        E_tmp = corr_model.get_separate_xef_terms(Fx)
        E_tmp *= elda * amix
        E_tmp = np.dot(E_tmp, weights)
        Fterms = np.append(Fterms, E_tmp)
    print("FTERMS", Fterms)

    Fterms2 = np.array([])

    for elda, ex in zip([ldaxu, ldaxd], [exu, exd]):
        Fx = ex / elda
        E_tmp = corr_model.get_separate_xef_terms(Fx)
        E_tmp *= elda
        E_tmp = np.dot(E_tmp, weights)
        Fterms2 = np.append(Fterms2, E_tmp)

    xvals = np.zeros((12,weights.shape[0]))
    start = 0
    for rho, x2, z, alpha in [(rho_data_u[0], x2u, zu, alpha_x),\
                              (rho_data_d[0], x2d, zd, alpha_x)]:
        gamma = gamma_func(x2, z, alpha)
        xvals[start:start+6] = corr_model.get_separate_corrfunc_terms(x2, z, gamma)
        xvals[start:start+6] *= LDA_FACTOR * 2**(1.0/3) * amix * rho**(4.0/3)
        start += 6
    xvals = np.dot(xvals, weights)

    #                      25      10      24     12,    10
    return np.concatenate([Eterms, Fterms, dvals, xvals, Fterms2,
                          [Ecscan, dft_analyzer.fx_total]], axis=0)

def get_new_contribs(dft_dir, restricted, mlfunc, exact=True):

    from mldftdat.models import map_c7

    corr_model = map_c7.VSXCContribs(None, None, None, None, None,
                                     None, fterm_scale=2.0)

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

    zeta = (rhou - rhod) / (rhot)
    ds = ((1-zeta)**(5.0/3) + (1+zeta)**(5.0/3))/2
    CU = 0.3 * (3 * np.pi**2)**(2.0/3)

    co0, vo0 = corr_model.os_baseline(rhou, rhod, g2, type=0)[:2]
    co1, vo1 = corr_model.os_baseline(rhou, rhod, g2, type=1)[:2]
    co0 *= rhot
    co1 *= rhot
    cx = co0
    co = co1

    #D = corr_model.get_D(rhou+rhot, g2, tu+td)
    #amix = corr_model.get_amix(rhou, rhod, g2, D)[0]

    nu, nd = rhou, rhod

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

    FUNCTIONAL = DEFAULT_FUNCTIONAL
    Exscan = eval_xc(FUNCTIONAL, (rho_data_u, rho_data_d), spin = 1)[0] \
             * (rho_data_u[0] + rho_data_d[0])
    Exscan = np.dot(Exscan, weights)
    Ecscan = eval_xc(',MGGA_C_REVSCAN', (rho_data_u, rho_data_d), spin = 1)[0] \
                     * (rho_data_u[0] + rho_data_d[0])
    Ecscan = np.dot(Ecscan, weights)

    print('EX ERROR', Ex - dft_analyzer.fx_total, Ex, dft_analyzer.fx_total)
    if (np.abs(Ex - dft_analyzer.fx_total) > 1e-7):
        print('LARGE ERROR')
    #assert np.abs(Ex - dft_analyzer.fx_total) < 1e-4

    exu = exu * rhou
    exd = exd * rhod
    exo = exo * rhot
    ldaxu = 2**(1.0/3) * LDA_FACTOR * rhou**(4.0/3) - 1e-20
    ldaxd = 2**(1.0/3) * LDA_FACTOR * rhod**(4.0/3) - 1e-20
    ldaxt = ldaxu + ldaxd

    gamma = 2**(2./3) * 0.05
    gammass = 0.05
    alpha = corr_model.get_alpha(rhot + 1e-16, zeta, g2, tu + td)[0]
    alphau = corr_model.get_alpha(rhou + 1e-16, 1, g2u, tu)[0]
    alphad = corr_model.get_alpha(rhod + 1e-16, 1, g2d, td)[0]
    chi = corr_model.get_chi(alpha)[0]
    #co *= (1-chi)*chi*4
    Ecscan = np.dot(co * (1 - chi**4), weights)
    co *= 1 - chi**4
    chiu = corr_model.get_chi(alphau)[0]
    chid = corr_model.get_chi(alphad)[0]
    x2 = corr_model.get_x2(nu+nd, g2)[0]
    x2u = corr_model.get_x2(nu, g2u)[0]
    x2d = corr_model.get_x2(nd, g2d)[0]
    amix = corr_model.get_amix(rhot, zeta, x2, chi)[0]
    #ratio = 0.5 * (1 - g2 / (8 * rhot * (tu + td) + 1e-16))
    #amix = corr_model.get_amix(rhou, rhod, g2, ratio)[0]
    Fx = exo / ldaxt
    Fxu = exu / ldaxu
    Fxd = exd / ldaxd
    E_tmp = corr_model.get_separate_xef_terms(Fx) * cx
    Eterms = np.dot(E_tmp, weights)
    E_tmp = corr_model.get_separate_xef_terms(Fx) * co
    Etermso = np.dot(E_tmp, weights)
    E_tmp = corr_model.get_separate_xef_terms(Fxu) * ldaxu
    E_tmp += corr_model.get_separate_xef_terms(Fxd) * ldaxd
    Fterms2 = np.dot(E_tmp, weights)
    E_tmp *= amix
    Fterms = np.dot(E_tmp, weights)
    E_tmp = corr_model.get_separate_sl_terms(x2, chi, gamma)[0] * cx
    dvals = np.dot(E_tmp, weights)
    E_tmp = corr_model.get_separate_sl_terms(x2, chi, gamma)[0] * co
    dvalso = np.dot(E_tmp, weights)
    E_tmp = corr_model.get_separate_sl_terms(x2u, chiu, gammass)[0] * ldaxu
    E_tmp += corr_model.get_separate_sl_terms(x2d, chid, gammass)[0] * ldaxd
    xvals2 = np.dot(E_tmp, weights)
    E_tmp *= amix
    xvals = np.dot(E_tmp, weights)

    #                                    5       5       11     11,    5,       11,
    return np.concatenate([[Ex, Exscan], Eterms, Fterms, dvals, xvals, Etermso, dvalso,
                          [Ecscan, dft_analyzer.fx_total]], axis=0)

def get_new_contribs2(dft_dir, restricted, mlfunc, exact=True):

    from mldftdat.models import map_c8

    corr_model = map_c8.VSXCContribs(None, None, None, None, None,
                                     None, fterm_scale=2.0)

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

    zeta = (rhou - rhod) / (rhot)
    ds = ((1-zeta)**(5.0/3) + (1+zeta)**(5.0/3))/2
    CU = 0.3 * (3 * np.pi**2)**(2.0/3)

    co0, vo0 = corr_model.os_baseline(rhou, rhod, g2, type=0)[:2]
    co1, vo1 = corr_model.os_baseline(rhou, rhod, g2, type=1)[:2]
    co0 *= rhot
    co1 *= rhot
    cx = co0
    co = co1

    nu, nd = rhou, rhod

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

    FUNCTIONAL = DEFAULT_FUNCTIONAL
    Exscan = eval_xc(FUNCTIONAL, (rho_data_u, rho_data_d), spin = 1)[0] \
             * (rho_data_u[0] + rho_data_d[0])
    Exscan = np.dot(Exscan, weights)
    Ecscan = eval_xc(',MGGA_C_REVSCAN', (rho_data_u, rho_data_d), spin = 1)[0] \
                     * (rho_data_u[0] + rho_data_d[0])
    Ecscan = np.dot(Ecscan, weights)

    print('EX ERROR', Ex - dft_analyzer.fx_total, Ex, dft_analyzer.fx_total)
    if (np.abs(Ex - dft_analyzer.fx_total) > 1e-7):
        print('LARGE ERROR')
    #assert np.abs(Ex - dft_analyzer.fx_total) < 1e-4

    exu = exu * rhou
    exd = exd * rhod
    exo = exo * rhot
    ldaxu = 2**(1.0/3) * LDA_FACTOR * rhou**(4.0/3) - 1e-20
    ldaxd = 2**(1.0/3) * LDA_FACTOR * rhod**(4.0/3) - 1e-20
    ldaxt = ldaxu + ldaxd

    gamma = 2**(2./3) * 0.004
    gammass = 0.004
    chi = corr_model.get_chi_full_deriv(rhot + 1e-16, zeta, g2, tu + td)[0]
    chiu = corr_model.get_chi_full_deriv(rhou + 1e-16, 1, g2u, tu)[0]
    chid = corr_model.get_chi_full_deriv(rhod + 1e-16, 1, g2d, td)[0]
    x2 = corr_model.get_x2(nu+nd, g2)[0]
    x2u = corr_model.get_x2(nu, g2u)[0]
    x2d = corr_model.get_x2(nd, g2d)[0]
    amix = corr_model.get_amix(rhot, zeta, x2, chi)[0]
    chidesc = np.array(corr_model.get_chi_desc(chi)[:4])
    chidescu = np.array(corr_model.get_chi_desc(chiu)[:4])
    chidescd = np.array(corr_model.get_chi_desc(chid)[:4])
    Fx = exo / ldaxt
    Fxu = exu / ldaxu
    Fxd = exd / ldaxd
    corrterms = np.append(corr_model.get_separate_xef_terms(Fx),
                          chidesc, axis=0)
    extermsu = np.append(corr_model.get_separate_sl_terms(x2u, chiu, gammass)[0],
                         corr_model.get_separate_xefa_terms(Fxu, chiu)[0], axis=0)
    extermsd = np.append(corr_model.get_separate_sl_terms(x2d, chid, gammass)[0],
                         corr_model.get_separate_xefa_terms(Fxd, chid)[0], axis=0)

    co *= 1 - chi**6
    Ecscan = np.dot(co, weights)
    Eterms = np.dot(corrterms * cx, weights)
    Etermso = np.dot(corrterms * co, weights)
    Fterms = np.dot(extermsu * ldaxu * amix, weights)
    Fterms += np.dot(extermsd * ldaxd * amix, weights)

    #                                    9,      9,       34
    return np.concatenate([[Ex, Exscan], Eterms, Etermso, Fterms,
                          [Ecscan, dft_analyzer.fx_total]], axis=0)

def get_new_contribs3(dft_dir, restricted, mlfunc, exact=True):

    from mldftdat.models import map_c9

    corr_model = map_c9.VSXCContribs(None, None, None, fterm_scale=2.0)

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

    zeta = (rhou - rhod) / (rhot)
    ds = ((1-zeta)**(5.0/3) + (1+zeta)**(5.0/3))/2
    CU = 0.3 * (3 * np.pi**2)**(2.0/3)

    co0, vo0 = corr_model.os_baseline(rhou, rhod, g2, type=0)[:2]
    co1, vo1 = corr_model.os_baseline(rhou, rhod, g2, type=1)[:2]
    co0 *= rhot
    co1 *= rhot
    cx = co0
    co = co1

    nu, nd = rhou, rhod

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

    FUNCTIONAL = DEFAULT_FUNCTIONAL
    try:
        Exscan = eval_xc(FUNCTIONAL, (rho_data_u, rho_data_d), spin = 1)[0] \
                 * (rho_data_u[0] + rho_data_d[0])
        Exscan = np.dot(Exscan, weights)
    except:
        Exscan = dft_analyzer.fx_total
        with open(os.path.join(SAVE_ROOT, 'MLFUNCTIONALS', FUNCTIONAL, 'settings.yaml'), 'r') as f:
            settings = yaml.load(f, Loader = yaml.Loader)
        corr_model_iter = map_c9.VSXCContribs(settings['d'], settings['dx'],
                                              settings['cx'],
                                              fterm_scale=settings['fterm_scale'])
        exc, vxc = corr_model_iter.xefc2(rhou, rhod, g2u, g2o, g2d,
                                         tu, td, exu, exd)
        Exscan += np.dot(dft_analyzer.grid.weights, exc)
    Ecscan = eval_xc(',MGGA_C_REVSCAN', (rho_data_u, rho_data_d), spin = 1)[0] \
                    * (rho_data_u[0] + rho_data_d[0])
    Ecscan = np.dot(Ecscan, weights)

    print('EX ERROR', Ex - dft_analyzer.fx_total, Ex, dft_analyzer.fx_total)
    if (np.abs(Ex - dft_analyzer.fx_total) > 1e-7):
        print('LARGE ERROR')
    #assert np.abs(Ex - dft_analyzer.fx_total) < 1e-4

    ldaxu = 2**(1.0/3) * LDA_FACTOR * rhou**(4.0/3) - 1e-20
    ldaxd = 2**(1.0/3) * LDA_FACTOR * rhod**(4.0/3) - 1e-20
    ldaxt = ldaxu + ldaxd

    gamma = 2**(2./3) * 0.004
    gammass = 0.004
    chi = corr_model.get_chi_full_deriv(rhot + 1e-16, zeta, g2, tu + td)[0]
    chiu = corr_model.get_chi_full_deriv(rhou + 1e-16, 1, g2u, tu)[0]
    chid = corr_model.get_chi_full_deriv(rhod + 1e-16, 1, g2d, td)[0]
    x2 = corr_model.get_x2(nu+nd, g2)[0]
    x2u = corr_model.get_x2(nu, g2u)[0]
    x2d = corr_model.get_x2(nd, g2d)[0]
    amix = corr_model.get_amix(rhot, zeta, x2, chi)[0]
    chidesc = np.array(corr_model.get_chi_desc(chi)[:4])
    chidescu = np.array(corr_model.get_chi_desc(chiu)[:4])
    chidescd = np.array(corr_model.get_chi_desc(chid)[:4])
    Fx = exo / ldaxt
    Fxu = exu / ldaxu
    Fxd = exd / ldaxd
    corrterms = np.append(corr_model.get_separate_xef_terms(Fx),
                          chidesc, axis=0)
    extermsu = np.append(corr_model.get_separate_sl_terms(x2u, chiu, gammass)[0],
                         corr_model.get_separate_xefa_terms(Fxu, chiu)[0], axis=0)
    extermsd = np.append(corr_model.get_separate_sl_terms(x2d, chid, gammass)[0],
                         corr_model.get_separate_xefa_terms(Fxd, chid)[0], axis=0)

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
    Eterms3 = np.dot(cmix_terms * (Fx-1) * cx, weights)
    Fterms = np.dot(extermsu * ldaxu * amix, weights)
    Fterms += np.dot(extermsd * ldaxd * amix, weights)

    #                                    7,      8,       8,       28
    return np.concatenate([[Ex, Exscan], Eterms, Eterms2, Eterms3, Fterms,
                          [Ecscan, dft_analyzer.fx_total]], axis=0)


def store_full_contribs_dataset(FNAME, ROOT, MOL_IDS,
                                IS_RESTRICTED_LIST, MLFUNC,
                                exact=True, BASIS=DEFAULT_BASIS,
                                mol_id_full=False):

    #SIZE = 25+10+24+12+10+3
    SIZE = 25+10+24+12+10+4
    X = np.zeros([0,SIZE])

    for mol_id, is_restricted in zip(MOL_IDS, IS_RESTRICTED_LIST):

        print(mol_id)

        if mol_id_full:
            dft_dir = mol_id
        elif is_restricted:
            dft_dir = get_save_dir(ROOT, 'RKS', BASIS,
                mol_id, functional = DEFAULT_FUNCTIONAL)
        else:
            dft_dir = get_save_dir(ROOT, 'UKS', BASIS,
                mol_id, functional = DEFAULT_FUNCTIONAL)

        sl_contribs = get_full_contribs2(dft_dir, is_restricted,
                                         MLFUNC, exact=exact)

        X = np.vstack([X, sl_contribs])

    np.save(FNAME, X)


def store_new_contribs_dataset(FNAME, ROOT, MOL_IDS,
                               IS_RESTRICTED_LIST, MLFUNC,
                               exact=True, BASIS=DEFAULT_BASIS,
                               mol_id_full=False):

    XSIZE = 14
    SIZE = 2+5+5+3*XSIZE+5+2
    SIZE = 2+9+9+34+2
    SIZE = 2+7+8+8+28+2
    X = np.zeros([0,SIZE])

    for mol_id, is_restricted in zip(MOL_IDS, IS_RESTRICTED_LIST):

        print(mol_id)

        if mol_id_full:
            dft_dir = mol_id.replace('SCAN', DEFAULT_FUNCTIONAL)
        elif is_restricted:
            dft_dir = get_save_dir(ROOT, 'RKS', BASIS,
                mol_id, functional = DEFAULT_FUNCTIONAL)
        else:
            dft_dir = get_save_dir(ROOT, 'UKS', BASIS,
                mol_id, functional = DEFAULT_FUNCTIONAL)

        sl_contribs = get_new_contribs3(dft_dir, is_restricted,
                                        MLFUNC, exact=exact)
        print(sl_contribs)
        assert (not np.isnan(sl_contribs).any())
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

def store_total_energies_dataset(FNAME, ROOT, MOL_IDS, IS_RESTRICTED_LIST,
                                 mol_id_full=False):

    # PBE, CCSD
    y = np.zeros([0, 2])

    for mol_id, is_restricted in zip(MOL_IDS, IS_RESTRICTED_LIST):
        print(mol_id)

        if mol_id_full:
            dft_dir = mol_id[0]
            ccsd_dir = mol_id[1]
        elif is_restricted:
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
                                NLC_COEFS=DEFAULT_NLC_COEFS, mol_id_full=False):

    X = np.zeros([0, len(NLC_COEFS)])

    for mol_id, is_restricted in zip(MOL_IDS, IS_RESTRICTED_LIST):

        print(mol_id)

        if mol_id_full:
            dft_dir = mol_id
        elif is_restricted:
            dft_dir = get_save_dir(ROOT, 'RKS', DEFAULT_BASIS,
                                   mol_id, functional = DEFAULT_FUNCTIONAL)
        else:
            dft_dir = get_save_dir(ROOT, 'UKS', DEFAULT_BASIS,
                                   mol_id, functional = DEFAULT_FUNCTIONAL)

        vv10_contribs = get_vv10_contribs(dft_dir, is_restricted, NLC_COEFS)

        X = np.vstack([X, vv10_contribs])

    np.save(FNAME, X)


def solve_from_stored_ae(DATA_ROOT, version='a'):

    import yaml
    from collections import Counter
    from ase.data import chemical_symbols, atomic_numbers, ground_state_magnetic_moments
    from sklearn.metrics import r2_score
    from pyscf import gto

    coef_sets = []
    scores = []

    etot = np.load(os.path.join(DATA_ROOT, 'etot.npy'))
    mlx = np.load(os.path.join(DATA_ROOT, 'alpha4_ex.npy'))
    #mlx = np.load(os.path.join(DATA_ROOT, 'descn_ex.npy'))
    vv10 = np.load(os.path.join(DATA_ROOT, 'vv10.npy'))
    f = open(os.path.join(DATA_ROOT, 'mols.yaml'), 'r')
    mols = yaml.load(f, Loader = yaml.Loader)
    f.close()

    aetot = np.load(os.path.join(DATA_ROOT, 'atom_etot.npy'))
    amlx = np.load(os.path.join(DATA_ROOT, 'atom_alpha4_ex.npy'))
    #amlx = np.load(os.path.join(DATA_ROOT, 'atom_descn_ex.npy'))
    atom_vv10 = np.load(os.path.join(DATA_ROOT, 'atom_vv10.npy'))
    f = open(os.path.join(DATA_ROOT, 'atom_ref.yaml'), 'r')
    amols = yaml.load(f, Loader = yaml.Loader)
    f.close()

    print ("SHAPES", mlx.shape, etot.shape, amlx.shape, aetot.shape) 


    valset_bools_init = np.array([mol['valset'] for mol in mols])
    valset_bools_init = np.append(valset_bools_init,
                        np.zeros(len(amols), valset_bools_init.dtype))
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
    #        if formulas[i]['C'] == 4 and formulas[i]['H'] == 9 and len(mol._atom) == 13:
    #            badind = i
    #mlx = np.append(mlx[:i], mlx[i+1:], axis=0)
    #mols = mols[:i] + mols[i+1:]
    #ecounts = ecounts[:i] + ecounts[i+1:]
    #valset_bools_init = np.append(valset_bools_init[:i], valset_bools_init[i+1:])

    ecounts = np.array(ecounts)

    N = etot.shape[0]
    num_vv10 = vv10.shape[-1]
    #num_vv10 = 1

    #print(formulas, Z_to_ind)

    for i in range(num_vv10):

        def get_terms(etot, mlx, vv10=None):
            if vv10 is not None:
                E_vv10 = vv10[:,i]
            if version == 'a':
                E_dft = etot[:,0]
                E_ccsd = etot[:,1]
                E_x = mlx[:,0]
                E_xscan = mlx[:,1]
                E_cscan = mlx[:,-2]
                # 0, 1 -- Ex pred and Exscan
                # 2:27 -- Eterms
                # 27:37 -- Fterms
                # 37:61 -- dvals
                # 61:73 -- xvals
                # 73 -- Ex exact
                E_c = mlx[:,13:17]
                #E_c = mlx[:,18:22]
                E_c = np.append(E_c, mlx[:,18:22], axis=1)
                ###E_c = np.append(E_c, mlx[:,23:27], axis=1)
                E_c = np.append(E_c, mlx[:,28:32] + mlx[:,33:37], axis=1)
                E_c = np.append(E_c, mlx[:,43:49], axis=1)
                E_c = np.append(E_c, mlx[:,49:55], axis=1)
                ###E_c = np.append(E_c, mlx[:,55:61], axis=1)
                E_c = np.append(E_c, mlx[:,61:67] + mlx[:,67:73], axis=1)
                #E_c = np.append(E_c, mlx[:,74:78] + mlx[:,79:83], axis=1)
                print("SHAPE", E_c.shape)

                #diff = E_ccsd - (E_dft - E_xscan + E_x + E_vv10 + mlx[:,2] + mlx[:,7] + mlx[:,12])
                #diff = E_ccsd - (E_dft - E_xscan + E_x + mlx[:,2] + mlx[:,9] + mlx[:,16] + mlx[:,30])
                #diff = E_ccsd - (E_dft - E_xscan + E_x + mlx[:,12] + mlx[:,22])
                diff = E_ccsd - (E_dft - E_xscan + E_x + E_cscan)
                #diff = E_ccsd - (E_dft - E_xscan + E_x + mlx[:,2] + mlx[:,7] + mlx[:,12] + mlx[:,22])
                #diff = E_ccsd - (E_dft - E_xscan + E_x + mlx[:,2] + mlx[:,7] + mlx[:,12])
            elif version == 'b':
                E_dft = etot[:,0]
                E_ccsd = etot[:,1]
                E_x = mlx[:,0]
                E_xscan = mlx[:,1]
                E_cscan = mlx[:,-2]
                # 0, 1 -- Ex pred and EXscan
                # 2:7 -- Eterms
                # 7:12 -- Fterms
                # 12:23 -- dvals
                # 23:34 -- xvals
                # 34:39 -- Etermso
                # 39:50 -- dvalso
                E_c = np.append(mlx[:,3:7], mlx[:,8:12], axis=1)
                E_c = np.append(E_c, mlx[:,12:40], axis=1)
                E_c = np.append(E_c, mlx[:,41:45], axis=1)
                E_c = np.append(E_c, mlx[:,45:59], axis=1)
                diff = E_ccsd - (E_dft - E_xscan + E_x + E_cscan)
            elif version == 'c':
                E_dft = etot[:,0]
                E_ccsd = etot[:,1]
                E_x = mlx[:,0]
                E_xscan = mlx[:,1]
                E_cscan = mlx[:,-2]
                # 0, 1 -- Ex pred and EXscan
                # 2:7 -- xef terms alpha=0
                # 7:11 -- chi terms alpha=0
                # 11:16 -- xef terms alpha=1
                # 16:20 -- chi terms alpha=1
                # 20:39 -- sl ex terms
                # 39:54 -- nl ex terms
                E_c = np.append(mlx[:,3:11], mlx[:,12:20], axis=1)
                #E_c = np.append(mlx[:,7:11], mlx[:,16:20], axis=1)
                E_c = np.append(E_c, mlx[:,20:39], axis=1)
                E_c = np.append(E_c, mlx[:,39:54], axis=1)
                diff = E_ccsd - (E_dft - E_xscan + E_x + E_cscan)# + E_c[:,6])
                noise = np.ones(E_c.shape[1]) * 1e-5
                noise[:4] /= 1000
            else:
                E_dft = etot[:,0]
                E_ccsd = etot[:,1]
                E_x = mlx[:,0]
                E_xscan = mlx[:,1]
                E_cscan = mlx[:,-2]
                E_c = mlx[:,4:38]
                E_c = np.append(mlx[:,4:10], mlx[:,10:26], axis=1)
                E_c = np.append(E_c, mlx[:,26:54], axis=1)
                print(E_c.shape)
                E_c = np.append(E_c, mlx[:,3:4]-mlx[:,2:3], axis=1)
                diff = E_ccsd - (E_dft - E_xscan + E_x + E_cscan + E_vv10)# - mlx[:,37])
                means = np.mean(np.abs(E_c), axis=0)
                noise = np.ones(E_c.shape[1]) * 1e-3
                noise[:16] *= 1
                noise[-1] *= 1
                #noise[:2] *= 10
                #noise[8:10] *= 100
                #noise = np.ones(E_c.shape[1]) * 1e-3

            return E_c, diff, E_ccsd, E_dft, E_xscan, E_x, E_cscan, noise

        E_c, diff, E_ccsd, E_dft, E_xscan, E_x, E_cscan, noise = get_terms(etot, mlx)
        E_c2, diff2, E_ccsd2, E_dft2, E_xscan2, E_x2, E_cscan2, noise = get_terms(aetot, amlx)
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
                weights.append(1.0)
                formula = formulas[i]
                if formula.get(1) == 2 and formula.get(8) == 1 and len(list(formula.keys()))==2:
                    waterind = i
                    print(formula, E_ccsd[i], E_dft[i])
                for Z in list(formula.keys()):
                    X[i,:] -= formula[Z] * X[Z_to_ind[Z],:]
                    y[i] -= formula[Z] * y[Z_to_ind[Z]]
                    Ecc[i] -= formula[Z] * Ecc[Z_to_ind[Z]]
                    Edf[i] -= formula[Z] * Edf[Z_to_ind[Z]]
               # print(formulas[i], y[i], Ecc[i], Edf[i], E_x[i] - E_xscan[i])
            else:
                if mols[i].nelectron == 1:
                    hind = i
                if mols[i].nelectron == 8:
                    oind = i
                    print(mols[i], E_ccsd[i], E_dft[i])
                #weights.append(1.0 / mols[i].nelectron if mols[i].nelectron <= 10 else 0)
                if mols[i].nelectron == 3:
                    weights.append(1e-8 / 3)
                else:
                    weights.append(1e-8 / mols[i].nelectron if mols[i].nelectron <= 10 else 0)
                #weights.append(0.0)
        for i in range(len(amols)):
            weights.append(8 / mols[i].nelectron)

        weights = np.array(weights)
        
        print(E_xscan[[hind,oind,waterind]])
        print('ASSESS MEAN DIFF')
        print(np.mean(np.abs(Ecc-Edf)[weights > 0]))
        print(np.mean(np.abs(diff)[weights > 0]))

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

        print(E_ccsd[waterind], E_dft[waterind])

        oind = indd[oind]
        hind = indd[hind]
        waterind = indd[waterind]

        #noise = 1e-5
        trset_bools = np.logical_not(valset_bools)
        Xtr = X[trset_bools]
        Xts = X[valset_bools]
        ytr = y[trset_bools]
        yts = y[valset_bools]
        wtr = weights[trset_bools]
        #A = np.linalg.inv(np.dot(Xtr.T * wtr, Xtr) + noise * np.identity(Xtr.shape[1]))
        A = np.linalg.inv(np.dot(Xtr.T * wtr, Xtr) + np.diag(noise))
        B = np.dot(Xtr.T, wtr * ytr)
        coef = np.dot(A, B)
        #coef *= 0

        mlxtmp = np.append(mlx[:,12] + mlx[:,22], amlx[:,12] + amlx[:,22])
        E0 = E_x[inds] + mlxtmp[inds]
        #E0 = E_x[inds] + E_cscan[inds]

        score = r2_score(yts, np.dot(Xts, coef))
        score0 = r2_score(yts, np.dot(Xts, 0 * coef))
        print(Xts.shape, yts.shape)
        print(score, score0)
        print((Ecc)[[hind,oind,waterind]], Ecc[oind], Edf[oind], Ecc[waterind], Edf[waterind])
        print((y - Ecc - np.dot(X, coef))[[hind,oind,waterind]], Ecc[oind], Edf[oind], Ecc[waterind], Edf[waterind])
        print('20', (np.dot(E_c[inds,:20], coef[:20]) + E0)[[hind, oind, waterind]])
        print('26', (np.dot(E_c[inds,:26], coef[:26]) + E0)[[hind, oind, waterind]])
        print('32', (np.dot(E_c[inds,:32], coef[:32]) + E0)[[hind, oind, waterind]])
        print('38', (np.dot(E_c[inds,:38], coef[:38]) + E0)[[hind, oind, waterind]])
        print('44', (np.dot(E_c[inds,:44], coef[:44]) + E0)[[hind, oind, waterind]])
        print('working', (np.dot(E_c[inds,:20], coef[:20]) + np.dot(E_c[inds,32:38], coef[32:38]) + np.dot(E_c[inds,44:], coef[44:]) + E0)[[hind, oind, waterind]])
        print('50', (np.dot(E_c[inds], coef) + E0)[[hind, oind, waterind]])
        print(coef[20:32], coef[38:44])
        print((E_x[inds])[[hind, oind, waterind]])
        print('SCAN ALL', np.mean(np.abs(Ecc-Edf)), np.mean((Ecc-Edf)), np.std(Ecc-Edf))
        print('SCAN VAL', np.mean(np.abs(Ecc-Edf)[valset_bools]), np.mean((Ecc-Edf)[valset_bools]),
                np.std((Ecc-Edf)[valset_bools]))
        print('ML ALL', np.mean(np.abs(y - np.dot(X, coef))), np.mean(y - np.dot(X, coef)),
                np.std(y - np.dot(X,coef)))
        print('ML VAL', np.mean(np.abs(yts - np.dot(Xts, coef))), np.mean(yts - np.dot(Xts, coef)),
                np.std(yts-np.dot(Xts,coef)))
        print(np.max(np.abs(y - np.dot(X, coef))), np.max(np.abs(Ecc - Edf)))
        print(np.max(np.abs(yts - np.dot(Xts, coef))), np.max(np.abs(Ecc - Edf)[valset_bools]))

        coef_sets.append(coef)
        scores.append(score)

        break

    return coef_sets, scores


def solve_from_stored_ae_ml(DATA_ROOT):

    import yaml
    from collections import Counter
    from ase.data import chemical_symbols, atomic_numbers, ground_state_magnetic_moments
    from sklearn.metrics import r2_score
    from pyscf import gto

    coef_sets = []
    scores = []

    etot = np.load(os.path.join(DATA_ROOT, 'etot.npy'))
    mlx = np.load(os.path.join(DATA_ROOT, 'desc_ml.npy'))
    #vv10 = np.load(os.path.join(DATA_ROOT, 'vv10.npy'))
    f = open(os.path.join(DATA_ROOT, 'mols.yaml'), 'r')
    mols = yaml.load(f, Loader = yaml.Loader)
    f.close()

    aetot = np.load(os.path.join(DATA_ROOT, 'atom_etot.npy'))
    amlx = np.load(os.path.join(DATA_ROOT, 'atom_desc_ml.npy'))
    #vv10 = np.load(os.path.join(DATA_ROOT, 'atom_vv10.npy'))
    f = open(os.path.join(DATA_ROOT, 'atom_ref.yaml'), 'r')
    amols = yaml.load(f, Loader = yaml.Loader)
    f.close()

    print ("SHAPES", mlx.shape, etot.shape, amlx.shape, aetot.shape) 


    valset_bools_init = np.array([mol['valset'] for mol in mols])
    valset_bools_init = np.append(valset_bools_init,
                        np.zeros(len(amols), valset_bools_init.dtype))
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
    #num_vv10 = vv10.shape[-1]
    num_vv10 = 1

    #print(formulas, Z_to_ind)

    for i in range(num_vv10):

        def get_terms(etot, mlx, vv10=None):
            if vv10 is not None:
                E_vv10 = vv10[:,i]
            E_dft = etot[:,0]
            E_ccsd = etot[:,1]
            E_x = mlx[:,0]
            E_xscan = mlx[:,1]
            # 0, 1 -- Ex pred and Exscan
            # 2:27 -- Eterms
            # 27:37 -- Fterms
            # 37:61 -- dvals
            # 61:73 -- xvals
            # 73:83 -- exchange contribs
            # 73 -- Ex exact
            E_c = mlx[:,13:17]
            E_c = np.append(E_c, mlx[:,18:22], axis=1)
            E_c = np.append(E_c, mlx[:,28:32] + mlx[:,33:37], axis=1)
            E_c = np.append(E_c, mlx[:,43:49], axis=1)
            E_c = np.append(E_c, mlx[:,49:55], axis=1)
            E_c = np.append(E_c, mlx[:,61:67] + mlx[:,67:73], axis=1)
            E_c = np.append(E_c, mlx[:,74:78] + mlx[:,79:83], axis=1)
            print("SHAPE", E_c.shape)

            diff = E_ccsd - (E_dft - E_xscan + E_x + mlx[:,12] + mlx[:,22])

            return E_c, diff, E_ccsd, E_dft, E_xscan, E_x

        def get_ex_terms(etot, mlx, vv10=None):
            if vv10 is not None:
                E_vv10 = vv10[:,i]
            E_dft = etot[:,0]
            E_ccsd = etot[:,1]
            E_x = mlx[:,0]
            E_xscan = mlx[:,1]

            E_c = mlx[:,13:17]
            E_c = np.append(E_c, mlx[:,18:22], axis=1)
            E_c = np.append(E_c, mlx[:,23:27], axis=1)
            E_c = np.append(E_c, mlx[:,28:32] + mlx[:,33:37], axis=1)
            E_c = np.append(E_c, mlx[:,43:49], axis=1)
            E_c = np.append(E_c, mlx[:,49:55], axis=1)
            E_c = np.append(E_c, mlx[:,55:61], axis=1)
            E_c = np.append(E_c, mlx[:,61:67] + mlx[:,67:73], axis=1)
            E_c *= 0
            E_c = np.append(E_c, mlx[:,74:78] + mlx[:,79:83], axis=1)

            diff = mlx[:,-1] - E_x # exact exchange - ml exchange

            return E_c, diff, E_ccsd, E_dft, E_xscan, E_x

        E_c, diff, E_ccsd, E_dft, E_xscan, E_x = get_terms(etot, mlx)
        E_c2, diff2, E_ccsd2, E_dft2, E_xscan2, E_x2 = get_terms(aetot, amlx)
        E_c = np.append(E_c, E_c2, axis=0)
        diff = np.append(diff, diff2)
        E_ccsd = np.append(E_ccsd, E_ccsd2)
        E_dft = np.append(E_dft, E_dft2)
        E_xscan = np.append(E_xscan, E_xscan2)
        E_x = np.append(E_x, E_x2)

        xdiff = np.append(mlx[:,-1], amlx[:,-1]) - np.append(mlx[:,0], amlx[:,0])
        E_cx = E_c.copy()
        E_cx[:,:-8] = 0

        E_c = np.append(E_c, E_cx, axis=0)
        diff = np.append(diff, xdiff)

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
                if formula.get(1) == 2 and formula.get(8) == 1 and len(list(formula.keys()))==2:
                    waterind = i
                    print(formula, E_ccsd[i], E_dft[i])
                for Z in list(formula.keys()):
                    X[i,:] -= formula[Z] * X[Z_to_ind[Z],:]
                    y[i] -= formula[Z] * y[Z_to_ind[Z]]
                    Ecc[i] -= formula[Z] * Ecc[Z_to_ind[Z]]
                    Edf[i] -= formula[Z] * Edf[Z_to_ind[Z]]
               # print(formulas[i], y[i], Ecc[i], Edf[i], E_x[i] - E_xscan[i])
            else:
                if mols[i].nelectron == 1:
                    hind = i
                if mols[i].nelectron == 8:
                    oind = i
                    print(mols[i], E_ccsd[i], E_dft[i])
                #weights.append(1.0 / mols[i].nelectron if mols[i].nelectron <= 10 else 0)
                weights.append(0.0001 / mols[i].nelectron if mols[i].nelectron <= 10 else 0)
                #weights.append(0.0)
        for i in range(len(amols)):
            weights.append(1 / mols[i].nelectron)

        weights = np.array(weights)

        inds = np.arange(len(y))
        cond1 = weights > 0
        cond2 = np.append(cond1, cond1)
        valset_bools = valset_bools_init[weights > 0]
        valset_bools = np.append(valset_bools, valset_bools)
        X = X[cond2, :]
        y = y[cond2 > 0]
        Ecc = Ecc[cond1]
        Edf = Edf[cond1]
        weights = weights[cond1]

        noise = 1e-3
        trset_bools = np.logical_not(valset_bools)
        valset_bools = valset_bools[:valset_bools.shape[0]//2]
        Xtr = X[trset_bools]
        Xts = X[:X.shape[0]//2][valset_bools]
        ytr = y[trset_bools]
        yts = y[:y.shape[0]//2][valset_bools]
        weights = np.append(weights, weights)
        wtr = weights[trset_bools]
        A = np.linalg.inv(np.dot(Xtr.T * wtr, Xtr) + noise * np.identity(Xtr.shape[1]))
        B = np.dot(Xtr.T, wtr * ytr)
        coef = np.dot(A, B)

        score = r2_score(yts, np.dot(Xts, coef))
        score0 = r2_score(yts, np.dot(Xts, 0 * coef))
        print(Xts.shape, yts.shape)
        print(score, score0)
        print('SCAN ALL', np.mean(np.abs(Ecc-Edf)), np.mean((Ecc-Edf)))
        print('SCAN VAL', np.mean(np.abs(Ecc-Edf)[valset_bools]),
                          np.mean((Ecc-Edf)[valset_bools]))
        print('ML ALL', np.mean(np.abs(y - np.dot(X, coef))),
                        np.mean(y - np.dot(X, coef)))
        print('ML VAL', np.mean(np.abs(yts - np.dot(Xts, coef))),
                        np.mean(yts - np.dot(Xts, coef)))
        print(np.max(np.abs(y - np.dot(X, coef))), np.max(np.abs(Ecc - Edf)))
        print(np.max(np.abs(yts - np.dot(Xts, coef))), np.max(np.abs(Ecc - Edf)[valset_bools]))

        coef_sets.append(coef)
        scores.append(score)

        #break

    return coef_sets, scores


def store_mols_in_order(FNAME, ROOT, MOL_IDS, IS_RESTRICTED_LIST, VAL_SET=None, mol_id_full=False):
    from pyscf import gto
    import yaml

    mol_dicts = []

    for mol_id, is_restricted in zip(MOL_IDS, IS_RESTRICTED_LIST):

        if mol_id_full:
            if is_restricted:
                pbe_analyzer = RHFAnalyzer.load(mol_id+'/data.hdf5')
            else:
                pbe_analyzer = UHFAnalyzer.load(mol_id+'/data.hdf5')
        elif is_restricted:
            pbe_dir = get_save_dir(ROOT, 'RKS', DEFAULT_BASIS, mol_id, functional='SCAN')#'SGXCORR_ALPHA3')
            pbe_analyzer = RHFAnalyzer.load(pbe_dir + '/data.hdf5')
        else:
            pbe_dir = get_save_dir(ROOT, 'UKS', DEFAULT_BASIS, mol_id, functional='SCAN')#'SGXCORR_ALPHA3')
            pbe_analyzer = UHFAnalyzer.load(pbe_dir + '/data.hdf5')

        mol_dicts.append(gto.mole.pack(pbe_analyzer.mol))
        if VAL_SET is not None:
            mol_dicts[-1].update({'valset': mol_id in VAL_SET})

    with open(FNAME, 'w') as f:
        yaml.dump(mol_dicts, f)
