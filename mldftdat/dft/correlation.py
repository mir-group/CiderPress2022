from pyscf.dft.numint import _vv10nlc
from pyscf.dft.libxc import eval_xc

class ProjNumInt(pyscf_numint.NumInt):

    def __init__(self, exps, coefs):
        super(ProjNumInt, self).__init__()
        self.exps = exps
        self.coefs = coefs

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
        dres = get_w(rhoa, rhob)

        ex0, vx0 = eval_xc('LDA,', (rhoa, rhob), spin=1)
        ec0t, vc0t = eval_xc(',PW92', (rhoa, rhob), spin=1)
        ec0a, vc0a = eval_xc(',PW92', (rhoa, 0), spin=1)
        ec0b, vc0b = eval_xc(',PW92', (0, rhob), spin=1)
        vx0 = vx0[0]
        vc0t = vc0t[0]
        vc0a = vc0a[0]
        vc0b = vc0b[0]

        ec0os = ec0t - ec0a - ec0b
        vc0os = vc0t - vc0a - vc0b

        g = 1
        for exp, c in zip(self.exps, self.coefs):
            i, j = exp
            g += c * w**i * u**j
            dgdn += c * i * w**(i-1) * u**j * dwdn
            dgdtau += c * i * w**(i-1) * u**j * dwdtau
            dgdn += c * j * w**i * u**(j-1) * dudn
            dgdgrad += c * j * w**i * u**(j-1) * dudgrad

        ec = ec0os * gos + ec0a * ga + ec0b * gb
        vc_rho = vc0os * gos + vc0a * ga * vc0b * gb
        vc_rho += ec0os * dgosdn + ec0a * dgadn + ec0b * dgbdn




def nr_rks_vv10(ni, mol, grids, xc_code, dms, relativity=0, hermi=0,
                max_memory=2000, verbose=None, b = 5.9, c = 0.0093):
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

# (w exponent, u exponent)
# (kinetic, grad)
default_ss_terms = [(1,0), (0,2), (3,2), (4,2)]
default_os_terms = [(1,0), (0,1), (3,2), (0.3)]

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
    rho83 = rho_data[0]**(8.0 / 3) + 1e-10
    rho113 = rho_data[0]**(11.0 / 3) + 1e-10
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
    dso2dn = 0.5 * (dsu2dn + dsd2dn)
    dso2dgrad = 0.5 * (dsu2dgrad + dsd2dgrad)
    gamma_css = 0.2
    gamma_cos = 0.006
    results = []
    for s2, ds2dn, ds2dgrad, gamma in\
                           [(su2, dsu2dn, dsu2dgrad, gamma_css),\
                            (sd2, dsd2dn, dsd2dgrad, gamma_css),\
                            (so2, dso2dn, dso2dgrad, gamma_cos)]:
        u = gamma * s2 / (1 + gamma * s2)
        duds2 = gamma / (1 + gamma * s2) - gamma**2 * s2 / (1 + gamma * s2)**2
        dudn = duds2 * ds2dn
        dudgrad = duds2 * ds2dgrad
        results.append((u, dudn, dudgrad))
    return results

def get_t_ss(rho, tau):
    tau0 = 0.3 * (6 * np.pi**2)**(2.0/3) * rho**(5.0/3)
    t = tau0 / (tau + 1e-10)
    dtdn = 0.5 * (6 * np.pi**2)**(2.0/3) * rho**(2.0/3) / (tau + 1e-10)
    dtdtau = - tau0 / (tau**2 + 1e-10)
    return t, dtdn, dtdtau

def get_w(rho_data_u, rho_data_d):
    rho = rho_data[0]
    # TODO: B97M-V paper has 3/5 instead of 3/10,
    # but I think this is wrong. Should be factor
    # of 2^2/3 compared to non-spinpol, not 2^5/3
    tu, dtudn, dtudtau = get_t_ss(rho_data_u[0], rho_data_u[5])
    td, dtddn, dtddtau = get_t_ss(rho_data_d[0], rho_data_d[5])
    to = 0.5 * (tu + td)
    dtodn = 0.5 * (dtudn + dtddn) 
    dtodtau = 0.5 * (dtudtau + dtddtau) 
    results = []
    for t, dtdn, dtdtau in [(tu, dtudn, dtudtau),\
                            (td, dtddn, dtddtau),\
                            (to, dtodn, dtodtau)]:
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
