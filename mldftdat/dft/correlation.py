

def vv10(b = 5.9, c = 0.0093):
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
        dterm_n = projector_mgga_deriv_onto_basis(mol, dterm_i, dterm_j)
        vtot = np.dot(vint_term, dterm_n)
        e_terms.append(etot)
        v_terms.append(vtot)
    return e_terms, v_terms
