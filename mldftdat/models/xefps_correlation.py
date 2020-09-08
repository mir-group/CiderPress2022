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


def get_full_contribs(dft_dir, restricted, mlfunc, exact = False):

    from mldftdat.models import map_c4

    corr_model = map_c4.VSXCContribs(None, None, None, None, None,
                                     None, None, None, None, None)

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

    co0_, vo0_ = eval_xc(',MGGA_C_SCAN', (rho_data_u_0, rho_data_d_0), spin = 1)[:2]
    rho_data_u_1[5] = CU * ds * rhou**(5.0/3) + g2u / (8 * rhou)
    cu1_, vu1_ = eval_xc(',MGGA_C_SCAN', (rho_data_u_1, 0*rho_data_d_1), spin = 1)[:2]
    rho_data_d_1[5] = CU * ds * rhod**(5.0/3) + g2d / (8 * rhod)
    cd1_, vd1_ = eval_xc(',MGGA_C_SCAN', (0*rho_data_u_1, rho_data_d_1), spin = 1)[:2]
    rho_data_u_1[5] = CU * ds * rhot**(5.0/3) + g2 / (8 * rhot)
    rho_data_d_1[5] = 0
    co1_, vo1_ = eval_xc(',MGGA_C_SCAN', (rho_data_u_1, rho_data_d_1), spin = 1)[:2]
    co0, vo0 = corr_model.os_baseline(rhou, rhod, g2, type=0)[:2]
    co1, vo1 = corr_model.os_baseline(rhou, rhod, g2, type=1)[:2]
    cu1, vu1 = corr_model.ss_baseline(rhou, g2)[:2]
    cd1, vd1 = corr_model.ss_baseline(rhod, g2)[:2]
    print(np.dot(vu1[0], rhou*weights), np.dot(vu1_[0][:,0], rhou*weights))
    print(np.dot(vd1[0], rhod*weights), np.dot(vd1_[0][:,1], rhod*weights))
    print(np.dot(vo1[0][:,0], rhot*weights), np.dot(vo1_[0][:,0], rhot*weights))
    print(np.dot(vo0[0][:,0], rhot*weights), np.dot(vo0_[0][:,0], rhot*weights))
    print(np.dot(co0, rhot*weights), np.dot(co0_, rhot*weights))
    print(np.dot(co1, rhot*weights), np.dot(co1_, rhot*weights))
    print(np.dot(cu1, rhot*weights), np.dot(cu1_, rhot*weights))
    print(np.dot(cd1, rhot*weights), np.dot(cd1_, rhot*weights))
    co0 *= rhot
    cu1 *= rhou
    cd1 *= rhod
    co1 = co1 * rhot - cu1 - cd1
    cu = cu1
    cd = cd1
    co = co1
    cx = co0
    A = 2.74
    B = 132
    sprefac = 2 * (3 * np.pi**2)**(1.0/3)

    xu = np.linalg.norm(rho_data_u[1:4], axis=0) / (rho_data_u[0]**(4.0/3) + 1e-20)
    xd = np.linalg.norm(rho_data_d[1:4], axis=0) / (rho_data_d[0]**(4.0/3) + 1e-20)
    s2 = (g2u + 2 * g2o + g2d) / (sprefac**2 * rhot**(8.0/3) + 1e-20)
    zu = rho_data_u[5] / (rho_data_u[0]**(5.0/3) + 1e-20) - CF
    zd = rho_data_d[5] / (rho_data_d[0]**(5.0/3) + 1e-20) - CF
    zu *= 2
    zd *= 2
    #Du = 1 - np.linalg.norm(rho_data_u[1:4], axis=0)**2 / (8 * rho_data_u[0] * rho_data_u[5] + 1e-20)
    #Dd = 1 - np.linalg.norm(rho_data_d[1:4], axis=0)**2 / (8 * rho_data_d[0] * rho_data_d[5] + 1e-20)
    Du = 1 - g2u / (8 * rhou * tu)
    Dd = 1 - g2d / (8 * rhod * td)
    Do = 1 - (g2u + 2 * g2o + g2d) / (8 * (rhou + rhod) * (tu + td))
    #Du = np.sin(np.pi * 0.5 * Du)
    #Dd = np.sin(np.pi * 0.5 * Dd)
    oDu = 0.5 * (1 - np.cos(np.pi * Du))
    oDd = 0.5 * (1 - np.cos(np.pi * Dd))
    oDo = 0.5 * (1 - np.cos(np.pi * Do))
    Du = corr_model.get_D(rhou, g2u, tu)[0]
    Dd = corr_model.get_D(rhod, g2d, td)[0]
    Do = corr_model.get_D(rhot, g2u + 2 * g2o + g2d, tu+td)
    fDo = Do
    Do = fDo[0]

    zeta = (rhou - rhod) / (rhot)
    phi = ((1-zeta)**(2.0/3) + (1+zeta)**(2.0/3))/2
    phi43 = ((1-zeta)**(4.0/3) + (1+zeta)**(4.0/3))/2
    phi43 = (1 - 2.3631 * (phi43 - 1)) * (1-zeta**12)
    chi_inf = 0.128026
    chi = 0.72161
    #b1c = 0.0285764
    b1c = 0.0285764
    gamma_eps = 0.031091

    part1 = b1c * np.log(1 + (1-np.e)/np.e / (1 + 4 * chi_inf * s2)**(0.25))
    part1 *= phi43
    part2 = gamma_eps * phi**3 * np.log((1 - 1 / (1 + 4 * chi * s2)**(0.25)) + 1e-30)
    epslim = part1 * (1-Do) + part2 * Do
    exlda = 2**(1.0 / 3) * LDA_FACTOR * rhou**(4.0/3)
    exlda += 2**(1.0 / 3) * LDA_FACTOR * rhod**(4.0/3)
    exlda /= (rhot)
    amixo = 1 - 1 / (1 + A * np.log(1 + B * (epslim / exlda)))
    amix = corr_model.get_amix(rhou, rhod, g2u, g2o, g2d, fDo)[0]
    b0 = corr_model.baseline0inf(zeta, s2)[0]
    b1 = corr_model.baseline1inf(zeta, s2)[0]
    bt = corr_model.baseline_inf(rhou, rhod, g2u + 2 * g2o + g2d, fDo)[0]

    #print(np.mean(np.abs(part2 - b1) * rhot))
    #print(np.mean(np.abs(part1 - b0) * rhot))
    #print(np.mean(np.abs(epslim - bt) * rhot))
    print(np.min(epslim / exlda), np.max(epslim / exlda))
    print(np.mean(np.abs(amix - amixo) * rhot))
    #print(np.min(amix[rhot > 1e-5]), np.max(amix[rhot > 1e-5]), np.min(amixo[rhot > 1e-5]), np.max(amixo[rhot > 1e-5]))
    print('D RANGE', np.min(Du), np.min(Dd), np.min(Do))
    print('D RANGE', np.max(Du), np.max(Dd), np.max(Do))

    alpha_x = 0.001867
    alpha_ss, alpha_os = 0.00515088, 0.00304966
    dvals = np.zeros((30, weights.shape[0]))
    start = 0

    def gamma_func(x2, z, alpha):
        return 1 + alpha * (x2 + z)
    for x2, z, alpha in [(xu**2, zu, alpha_ss), (xd**2, zd, alpha_ss),\
                        (xu**2+xd**2, zu+zd, alpha_os),\
                        (xu**2+xd**2, zu+zd, alpha_os),\
                        (xu**2+xd**2, zu+zd, alpha_os)]:
        gamma = gamma_func(x2, z, alpha)
        dvals[start+0] = 1 / gamma - 1
        dvals[start+1] = x2 / gamma**2
        dvals[start+2] = z / gamma**2
        dvals[start+3] = x2**2 / gamma**3
        dvals[start+4] = x2 * z / gamma**3
        dvals[start+5] = z**2 / gamma**3
        start += 6
    dvals[:6] *= cu * Du
    dvals[6:12] *= cd * Dd
    dvals[12:18] *= co
    dvals[18:24] *= cx
    dvals[24:30] *= (cx - co) * (1 - Do)
    dvals = np.dot(dvals, weights)
    dvals = np.append(dvals[:6] + dvals[6:12], dvals[12:])
    cvals = np.zeros((20, weights.shape[0]))
    start = 0
    gamma_ss, gamma_os = 0.06, 0.0031
    for x2, gamma in [(xu**2, gamma_ss), (xd**2, gamma_ss),\
                      (xu**2+xd**2, gamma_os),\
                      (xu**2+xd**2, gamma_os)]:
        u = gamma * x2 / (1 + gamma * x2)
        for i in range(5):
            cvals[start+i] = u**i
        start += 5
    """
    for D, gamma in [(Du, gamma_ss), (Dd, gamma_ss),\
                     (Do, gamma_os),\
                     ((1-Do), gamma_os)]:
        for i in range(5):
            cvals[start+i] = D**i * (1 - D)
        start += 5
    """
    cvals[:5] *= cu * Du
    cvals[5:10] *= cd * Dd
    cvals[15:20] *= cx * (1 - Do)
    cvals[10:15] *= co * Do
    cvals = np.dot(cvals, weights)
    cvals = np.append(cvals[:5] + cvals[5:10], cvals[10:])

    numint0 = ProjNumInt(xterms = [], ssterms = [], osterms = [])
    if restricted:
        if exact:
            ex = dft_analyzer.fx_energy_density / (rho_data[0] + 1e-20)
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
        if exact:
            exu = dft_analyzer.fx_energy_density_u / (rho_data[0][0] + 1e-20)
            exd = dft_analyzer.fx_energy_density_d / (rho_data[1][0] + 1e-20)
        else:
            exu = _eval_x_0(mlfunc, mol, 2 * rho_data[0], grid, 2 * rdm1[0])[0]
            exd = _eval_x_0(mlfunc, mol, 2 * rho_data[1], grid, 2 * rdm1[1])[0]
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
                          [cu * Du, cd * Dd, co, cx, (cx - co) * (1 - Do)]):
        if isinstance(c, tuple):
            c, fac = c
        else:
            fac = None
        elda = LDA_FACTOR * rho**(1.0/3) - 1e-20
        Fx = ex / elda
        Etmp = np.zeros(5)
        x1 = (1 - Fx**6) / (1 + Fx**6)
        for i in range(5):
            if i == 0 and (fac is not None):
                Etmp[i] = np.dot(c * fac, weights)
            else:
                Etmp[i] = np.dot(c * x1**i, weights)
        Eterms = np.append(Eterms, Etmp)

    Fterms = np.array([])

    for rho, ex in zip([rhou, rhod], [exu, exd]):
            elda = LDA_FACTOR * rho**(1.0/3) - 1e-20
            Fx = ex / elda
            Etmp = np.zeros(5)
            x1 = (1 - Fx**6) / (1 + Fx**6)
            for i in range(5):
                Etmp[i] = np.dot(elda * amix * rho / 2 * x1**i, weights)
            Fterms = np.append(Fterms, Etmp)

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
        xvals[start:start+6] *= LDA_FACTOR * 2**(1.0/3) * amix * rho**(4.0/3)
        start += 6
    xvals = np.dot(xvals, weights)

    #                      25      10      24     12
    return np.concatenate([Eterms, Fterms, dvals, xvals,
                          [dft_analyzer.fx_total]], axis=0)

def store_full_contribs_dataset(FNAME, ROOT, MOL_IDS,
                                IS_RESTRICTED_LIST, MLFUNC):

    SIZE = 25+10+24+12+3
    X = np.zeros([0,SIZE])

    for mol_id, is_restricted in zip(MOL_IDS, IS_RESTRICTED_LIST):

        print(mol_id)

        if is_restricted:
            dft_dir = get_save_dir(ROOT, 'RKS', DEFAULT_BASIS,
                mol_id, functional = DEFAULT_FUNCTIONAL)
        else:
            dft_dir = get_save_dir(ROOT, 'UKS', DEFAULT_BASIS,
                mol_id, functional = DEFAULT_FUNCTIONAL)

        sl_contribs = get_full_contribs(dft_dir, is_restricted,
                                       MLFUNC)

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
    mlx = np.load(os.path.join(DATA_ROOT, 'lhlikeml3.npy'))
    mlx0 = np.load(os.path.join(DATA_ROOT, 'lhlike.npy'))
    mnc = np.load(os.path.join(DATA_ROOT, 'mnsf2.npy'))
    vv10 = np.load(os.path.join(DATA_ROOT, 'vv10.npy'))
    f = open(os.path.join(DATA_ROOT, 'mols.yaml'), 'r')
    mols = yaml.load(f, Loader = yaml.Loader)
    f.close()
    valset_bools_init = np.array([mol['valset'] for mol in mols])
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
        # 0, 1 -- Ex pred and Exscan
        # 2:27 -- Eterms
        # 27:37 -- Fterms
        # 37:61 -- dvals
        # 61:73 -- xvals
        # 73 -- Ex exact
        E_c = np.append(mlx[:,3:7] + mlx[:,8:12], mlx[:,13:17], axis=1)
        #E_c = np.zeros((mlx.shape[0],0))
        E_c = np.append(E_c, mlx[:,18:22], axis=1)
        E_c = np.append(E_c, mlx[:,23:27], axis=1)
        E_c = np.append(E_c, mlx[:,28:32] + mlx[:,33:37], axis=1)
        E_c = np.append(E_c, mlx[:,37:43], axis=1)
        E_c = np.append(E_c, mlx[:,43:49], axis=1)
        E_c = np.append(E_c, mlx[:,49:55], axis=1)
        E_c = np.append(E_c, mlx[:,55:61], axis=1)
        E_c = np.append(E_c, mlx[:,61:67] + mlx[:,67:73], axis=1)
        print("SHAPE", E_c.shape)

        #diff = E_ccsd - (E_dft - E_xscan + E_x + E_vv10 + mlx[:,2] + mlx[:,7] + mlx[:,12])
        #diff = E_ccsd - (E_dft - E_xscan + E_x + mlx[:,2] + mlx[:,9] + mlx[:,16] + mlx[:,30])
        diff = E_ccsd - (E_dft - E_xscan + E_x + mlx[:,2] + mlx[:,7] + mlx[:,12] + mlx[:,22])
        #diff = E_ccsd - (E_dft - E_xscan + E_x + mlx[:,2] + mlx[:,7] + mlx[:,12])

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
                weights.append(1.0 / mols[i].nelectron if mols[i].nelectron <= 10 else 0)
                #weights.append(0.0)


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

        noise = 7.5e-3
        trset_bools = np.logical_not(valset_bools)
        Xtr = X[trset_bools]
        Xts = X[valset_bools]
        ytr = y[trset_bools]
        yts = y[valset_bools]
        wtr = weights[trset_bools]
        A = np.linalg.inv(np.dot(Xtr.T * wtr, Xtr) + noise * np.identity(Xtr.shape[1]))
        B = np.dot(Xtr.T, wtr * ytr)
        coef = np.dot(A, B)
        #coef *= 0

        score = r2_score(yts, np.dot(Xts, coef))
        score0 = r2_score(yts, np.dot(Xts, 0 * coef))
        print(score, score0)
        print((y - np.dot(X, coef))[[hind,oind,waterind]], Ecc[oind], Edf[oind], Ecc[waterind], Edf[waterind])
        print((y - Ecc - np.dot(X, coef))[[hind,oind,waterind]], Ecc[oind], Edf[oind], Ecc[waterind], Edf[waterind])
        print('SCAN ALL', np.mean(np.abs(Ecc-Edf)), np.mean((Ecc-Edf)))
        print('SCAN VAL', np.mean(np.abs(Ecc-Edf)[valset_bools]), np.mean((Ecc-Edf)[valset_bools]))
        print('ML ALL', np.mean(np.abs(y - np.dot(X, coef))), np.mean(y - np.dot(X, coef)))
        print('ML VAL', np.mean(np.abs(yts - np.dot(Xts, coef))), np.mean(yts - np.dot(Xts, coef)))
        print(np.max(np.abs(y - np.dot(X, coef))), np.max(np.abs(Ecc - Edf)[weights > 0]))
        print(np.max(np.abs(yts - np.dot(Xts, coef))), np.max(np.abs(Ecc - Edf)[valset_bools]))

        coef_sets.append(coef)
        scores.append(score)

        break

    return coef_sets, scores


def store_mols_in_order(FNAME, ROOT, MOL_IDS, IS_RESTRICTED_LIST, VAL_SET=None):
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
        if VAL_SET is not None:
            mol_dicts[-1].update({'valset': mol_id in VAL_SET})

    with open(FNAME, 'w') as f:
        yaml.dump(mol_dicts, f)
