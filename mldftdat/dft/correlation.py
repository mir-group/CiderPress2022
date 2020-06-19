from pyscf.dft.numint import _vv10nlc
from pyscf.dft.libxc import eval_xc
from pyscf.dft import numint as pyscf_numint
import numpy as np
from pyscf.dft.numint import _vv10nlc, _rks_gga_wv0, _scale_ao, _dot_ao_ao

# (w exponent, u exponent)
# (kinetic, grad)
default_x_terms = [(0.416, 1, 0), (1.308, 0, 1), (3.070, 1, 1), (1.901, 0, 2)]
default_ss_terms = [(-5.668,1,0), (-1.855,0,2), (-20.497,3,2), (-20.364,4,2)]
default_os_terms = [(2.535,1,0), (1.573,0,1), (-6.427,3,2), (-6.298,0,3)]

class ProjNumInt(pyscf_numint.NumInt):

    def __init__(self, xterms = default_x_terms,
                       ssterms = default_ss_terms,
                       osterms = default_os_terms):
        super(ProjNumInt, self).__init__()
        self.xterms = xterms
        self.ssterms = ssterms
        self.osterms = osterms

    def eval_xc(self, xc_code, rho, spin=0, relativity=0, deriv=1, omega=None,
                verbose=None):
        deriv = 1
        relativity = 0
        omega = None

        if spin == 0:
            rhoa, rhob = rho / 2, rho / 2
        else:
            rhoa, rhob = rho[0], rho[1]

        ures = get_u(rhoa, rhob)
        wres = get_w(rhoa, rhob)
        zeros = 0 * rhob

        ex0a, vx0a, _, _ = eval_xc('LDA,', (rhoa[0], zeros), spin=1)
        ex0b, vx0b, _, _ = eval_xc('LDA,', (zeros, rhob[0]), spin=1)
        ec0t, vc0t, _, _ = eval_xc(',VWN', (rhoa[0], rhob[0]), spin=1)
        ec0a, vc0a, _, _ = eval_xc(',VWN', (rhoa[0], zeros), spin=1)
        ec0b, vc0b, _, _ = eval_xc(',VWN', (zeros, rhob[0]), spin=1)
        vx0a = vx0a[0][:,0]
        vx0b = vx0b[0][:,1]
        vc0t = vc0t[0]
        vc0a = vc0a[0][:,0]
        vc0b = vc0b[0][:,1]

        ec0os = ec0t * (rhoa[0] + rhob[0]) - ec0a * rhoa[0] - ec0b * rhob[0]
        ec0os /= rhoa[0] + rhob[0] + 1e-12
        vc0os = vc0t.copy()
        vc0os[:,0] -= vc0a
        vc0os[:,1] -= vc0b

        g = 1
        dgdn = 0
        dgdgrad = 0
        dgdtau = 0
        for c, i, j in self.xterms:
            u, dudn, dudgrad = ures[4]
            w, dwdn, dwdtau = wres[0]
            g += c * w**i * u**j
            if i > 0:
                dgdn += c * i * w**(i-1) * u**j * dwdn
                dgdtau += c * i * w**(i-1) * u**j * dwdtau
            if j > 0:
                dgdn += c * j * w**i * u**(j-1) * dudn
                dgdgrad += c * j * w**i * u**(j-1) * dudgrad
        Exa = ex0a * g * rhoa[0]
        vxa_rho = vx0a * g + ex0a * dgdn
        vxa_grad = ex0a * dgdgrad
        vxa_tau = ex0a * dgdtau

        g = 1
        dgdn = 0
        dgdgrad = 0
        dgdtau = 0
        for c, i, j in self.xterms:
            u, dudn, dudgrad = ures[5]
            w, dwdn, dwdtau = wres[1]
            g += c * w**i * u**j
            if i > 0:
                dgdn += c * i * w**(i-1) * u**j * dwdn
                dgdtau += c * i * w**(i-1) * u**j * dwdtau
            if j > 0:
                dgdn += c * j * w**i * u**(j-1) * dudn
                dgdgrad += c * j * w**i * u**(j-1) * dudgrad
        Exb = ex0b * g * rhob[0]
        vxb_rho = vx0b * g + ex0b * dgdn
        vxb_grad = ex0b * dgdgrad
        vxb_tau = ex0b * dgdtau

        g = 1
        dgdn = 0
        dgdtau = 0
        dgdgrad = 0
        for c, i, j in self.ssterms:
            u, dudn, dudgrad = ures[0]
            w, dwdn, dwdtau = wres[0]
            g += c * w**i * u**j
            if i > 0:
                dgdn += c * i * w**(i-1) * u**j * dwdn
                dgdtau += c * i * w**(i-1) * u**j * dwdtau
            if j > 0:
                dgdn += c * j * w**i * u**(j-1) * dudn
                dgdgrad += c * j * w**i * u**(j-1) * dudgrad
        Eca = ec0a * g * rhoa[0]
        vca_rho = vc0a * g + ec0a * dgdn
        vca_grad = ec0a * dgdgrad
        vca_tau = ec0a * dgdtau

        g = 1
        dgdn = 0
        dgdtau = 0
        dgdgrad = 0
        for c, i, j in self.ssterms:
            u, dudn, dudgrad = ures[1]
            w, dwdn, dwdtau = wres[1]
            g += c * w**i * u**j
            if i > 0:
                dgdn += c * i * w**(i-1) * u**j * dwdn
                dgdtau += c * i * w**(i-1) * u**j * dwdtau
            if j > 0:
                dgdn += c * j * w**i * u**(j-1) * dudn
                dgdgrad += c * j * w**i * u**(j-1) * dudgrad
        Ecb = ec0b * g * rhob[0]
        vcb_rho = vc0b * g + ec0b * dgdn
        vcb_grad = ec0b * dgdgrad
        vcb_tau = ec0b * dgdtau

        g = 1
        dgdna = 0
        dgdnb = 0
        dgdtaua = 0
        dgdtaub = 0
        dgdgrada = 0
        dgdgradb = 0
        for c, i, j in self.osterms:
            u, dudna, dudgrada = ures[2]
            u, dudnb, dudgradb = ures[3]
            w, dwdna, dwdtaua = wres[2]
            w, dwdnb, dwdtaub = wres[3]
            g += c * w**i * u**j
            if i > 0:
                dgdna += c * i * w**(i-1) * u**j * dwdna
                dgdnb += c * i * w**(i-1) * u**j * dwdnb
                dgdtaua += c * i * w**(i-1) * u**j * dwdtaua
                dgdtaub += c * i * w**(i-1) * u**j * dwdtaub
            if j > 0:
                dgdna += c * j * w**i * u**(j-1) * dudna
                dgdnb += c * j * w**i * u**(j-1) * dudnb
                dgdgrada += c * j * w**i * u**(j-1) * dudgrada
                dgdgradb += c * j * w**i * u**(j-1) * dudgradb

        Ecos = ec0os * g * (rhoa[0] + rhob[0])
        vca_rho += ec0os * dgdna + vc0os[:,0] * g
        vcb_rho += ec0os * dgdnb + vc0os[:,1] * g
        vca_grad += ec0os * dgdgrada
        vcb_grad += ec0os * dgdgradb
        vca_tau += ec0os * dgdtaua
        vcb_tau += ec0os * dgdtaub

        print(Eca, Ecb, Ecos, Exa, Exb)

        ec = (Eca + Ecb + Ecos + Exa + Exb) / (rhoa[0] + rhob[0] + 1e-12)
        ec = (Eca + Ecb + Ecos + Exa + Exb) / (rhoa[0] + rhob[0] + 1e-12)
        #vc_rho = vc0os * gos.reshape(-1,1) + vc0a * ga * vc0b * gb
        #vc_rho += ec0os * dgosdn + ec0a * dgadn + ec0b * dgbdn

        vc_rho = np.vstack((vca_rho + vxa_rho, vcb_rho + vxb_rho)).T
        vc_grad = np.vstack((vca_grad + vxa_grad,\
                            0 * vca_grad, vcb_grad + vxb_grad)).T
        vc_nabla = 0 * vcb_rho
        vc_tau = np.vstack((vca_tau + vxa_tau, vcb_tau + vxb_tau)).T

        return ec, (vc_rho, vc_grad, vc_nabla, vc_tau), None, None


def nr_rks_vv10(ni, mol, grids, xc_code, dms, relativity=0, hermi=0,
                max_memory=2000, verbose=None, b = 5.9, c = 0.0093):
    import numpy
    xctype = ni._xc_type(xc_code)
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi)

    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    nelec = numpy.zeros(nset)
    excsum = numpy.zeros(nset)
    vmat = numpy.zeros((nset,nao,nao))
    aow = None

    nlc_pars = (b, c)
    ao_deriv = 1
    vvrho=numpy.empty([nset,4,0])
    vvweight=numpy.empty([nset,0])
    vvcoords=numpy.empty([nset,0,3])
    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
        rhotmp = numpy.empty([0,4,weight.size])
        weighttmp = numpy.empty([0,weight.size])
        coordstmp = numpy.empty([0,weight.size,3])
        for idm in range(nset):
            rho = make_rho(idm, ao, mask, 'GGA')
            rho = numpy.expand_dims(rho,axis=0)
            rhotmp = numpy.concatenate((rhotmp,rho),axis=0)
            weighttmp = numpy.concatenate((weighttmp,numpy.expand_dims(weight,axis=0)),axis=0)
            coordstmp = numpy.concatenate((coordstmp,numpy.expand_dims(coords,axis=0)),axis=0)
            rho = None
        vvrho=numpy.concatenate((vvrho,rhotmp),axis=2)
        vvweight=numpy.concatenate((vvweight,weighttmp),axis=1)
        vvcoords=numpy.concatenate((vvcoords,coordstmp),axis=1)
        rhotmp = weighttmp = coordstmp = None
    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
        ngrid = weight.size
        aow = numpy.ndarray(ao[0].shape, order='F', buffer=aow)
        for idm in range(nset):
            rho = make_rho(idm, ao, mask, 'GGA')
            exc, vxc = _vv10nlc(rho,coords,vvrho[idm],vvweight[idm],vvcoords[idm],nlc_pars)
            den = rho[0] * weight
            nelec[idm] += den.sum()
            excsum[idm] += numpy.dot(den, exc)
# ref eval_mat function
            wv = _rks_gga_wv0(rho, vxc, weight)
            #:aow = numpy.einsum('npi,np->pi', ao, wv, out=aow)
            aow = _scale_ao(ao, wv, out=aow)
            vmat[idm] += _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
            rho = exc = vxc = wv = None
    vvrho = vvweight = vvcoords = None

    for i in range(nset):
        vmat[i] = vmat[i] + vmat[i].T
    if nset == 1:
        nelec = nelec[0]
        excsum = excsum[0]
        vmat = vmat.reshape(nao,nao)
    return nelec, excsum, vmat

def corr_term(u, w, du, dw, i, j):
    term = w**i * u**j
    if i > 0:
        dterm_i = i * w**(i-1) * u**j * dw
    else:
        dterm_i = 0 * term
    if j > 0:
        dterm_j = j * w**i * u**(j-1) * du
    else:
        dterm_j = 0 * term
    return term, dterm_i, dterm_j

def get_s2_ss(rho_data):
    rho83 = rho_data[0]**(8.0 / 3) + 1e-12
    rho113 = rho_data[0]**(11.0 / 3) + 1e-12
    gradn2 = np.linalg.norm(rho_data[1:4], axis=0)**2
    s2 = gradn2 / rho83
    ds2dgrad = 1 / rho83
    ds2dn = (-8.0/3) * gradn2 / rho113
    return s2, ds2dn, ds2dgrad

def get_u(rho_data_u, rho_data_d):
    # TODO spin-polarize the opposite-spin derivatives.
    su2, dsu2dn, dsu2dgrad = get_s2_ss(rho_data_u)
    sd2, dsd2dn, dsd2dgrad = get_s2_ss(rho_data_d)
    so2 = 0.5 * (su2 + sd2)
    dso2dnu = 0.5 * dsu2dn
    dso2dnd = 0.5 * dsd2dn
    dso2dgradu = 0.5 * dsu2dgrad
    dso2dgradd = 0.5 * dsd2dgrad
    gamma_css = 0.2
    gamma_cos = 0.006
    gamma_x = 0.004
    results = []
    for s2, ds2dn, ds2dgrad, gamma in\
                           [(su2, dsu2dn, dsu2dgrad, gamma_css),\
                            (sd2, dsd2dn, dsd2dgrad, gamma_css),\
                            (so2, dso2dnu, dso2dgradu, gamma_cos),\
                            (so2, dso2dnd, dso2dgradd, gamma_cos),\
                            (su2, dsu2dn, dsu2dgrad, gamma_x),\
                            (sd2, dsd2dn, dsd2dgrad, gamma_x)]:
        u = gamma * s2 / (1 + gamma * s2)
        duds2 = gamma / (1 + gamma * s2) - gamma**2 * s2 / (1 + gamma * s2)**2
        dudn = duds2 * ds2dn
        dudgrad = duds2 * ds2dgrad
        results.append((u, dudn, dudgrad))
    s2 = ds2dn = ds2dgrad = gamma = None

    #u = gamma_cos * so2 / (1 + gamma_cos * so2)
    #duds2 = gamma_cos / (1 + gamma_cos * so2) - gamma_cos**2 * so2 / (1 + gamma_cos * so2)**2
    #dudn = duds2 * dso2dn
    #dudgrad = duds2 * dso2dgrad
    return results

def get_t_ss(rho, tau):
    tau0 = 0.3 * (6 * np.pi**2)**(2.0/3) * rho**(5.0/3)
    t = tau0 / (tau + 1e-12)
    dtdn = 0.5 * (6 * np.pi**2)**(2.0/3) * rho**(2.0/3) / (tau + 1e-12)
    dtdtau = - tau0 / (tau**2 + 1e-12)
    return t, dtdn, dtdtau

def get_w(rho_data_u, rho_data_d):
    # TODO: B97M-V paper has 3/5 instead of 3/10,
    # but I think this is wrong. Should be factor
    # of 2^2/3 compared to non-spinpol, not 2^5/3
    tu, dtudn, dtudtau = get_t_ss(rho_data_u[0], rho_data_u[5])
    td, dtddn, dtddtau = get_t_ss(rho_data_d[0], rho_data_d[5])
    to = 0.5 * (tu + td)
    dtodnu = 0.5 * dtudn
    dtodnd = 0.5 * dtddn
    dtodtauu = 0.5 * dtudtau
    dtodtaud = 0.5 * dtddtau
    results = []
    for t, dtdn, dtdtau in [(tu, dtudn, dtudtau),\
                            (td, dtddn, dtddtau),\
                            (to, dtodnu, dtodtauu),\
                            (to, dtodnd, dtodtaud)]:
        w = (t - 1) / (t + 1)
        dwdt = 1 / (t + 1) - (t - 1) / (t + 1)**2
        dwdn = dwdt * dtdn
        dwdtau = dwdt * dtdtau
        results.append((w, dwdn, dwdtau))
    return results

def get_corr_terms(mol, rho_data, coords, weights):
    vint_term = -3 * rho_data[0] + np.dot(coords, rho_data[1:4])
    w = get_w(rho_data)
    u = get_u(rho_data)
    dw = get_dw(rho_data)
    du = get_du(rho_data)
    e_terms = []
    v_terms = []
    for i,j in default_ss_terms:
        term, dterm_i, dterm_j = corr_term(u, w, du, dw, i, j)
        etot = np.dot(term, weights)
        dterm_n = project_mgga_deriv_onto_basis(mol, dterm_i, dterm_j)
        vtot = np.dot(vint_term, dterm_n)
        e_terms.append(etot)
        v_terms.append(vtot)
    return e_terms, v_terms
