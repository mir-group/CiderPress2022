"""
Correlation energy descriptors.

Most functions are formatted as get_*. The return
values of these functions are tuples, with the first
element of the typle containing the requested quantity
and the other elements containing the derivatives wrt
the inputs in order. The exception is get_*_baseline,
which returns pyscf-like formatted list of derivatives
as its second return value.

Feature names:
n : density
nu : spin-alpha density
nd : spin-beta density
zeta : spin polarization (nu-nd)/(nu+nd)
g2 : |\nabla n|^2 (squared norm of density gradient)
g2u : \nabla nu \cdot \nabla nu
g2d : \nabla nd \cdot \nabla nd
g2o : \nabla nu \cdot \nabla nd
t : Kinetic energy density 0.5 \sum_i f_i |\nabla \phi_i|^2
tu : Spin-alpha kinetic energy
td : Spin-beta kinetic energy
f or F : Exchange enhancement factor (XEF) e_x n^{-4/3} / C_{LDA}
fu or Fu : spin-alpha XEF e_{x,u} nu^{-4/3} / (2^{1/3} C_{LDA})
fd or Fd : spin-beta XEF e_{x,d} nd^{-4/3} / (2^{1/3} C_{LDA})
"""

import numpy as np

LDA_FACTOR = - 3.0 / 4.0 * (3.0 / np.pi)**(1.0/3)
alphax = 0.001867
alphass, alphaos = 0.00515088, 0.00304966
CF = 0.3 * (6 * np.pi**2)**(2.0/3)
CFC = 0.3 * (3 * np.pi**2)**(2.0/3)

A = 2.74
B = 132
sprefac = 2 * (3 * np.pi**2)**(1.0/3)
chiinf = 0.128026
chi = 0.72161
b1c = 0.0285764
b2c = 0.0889
b3c = 0.125541
gamma = 0.031091

# from libxc https://gitlab.com/libxc/libxc/-/blob/master/maple/lda_exc/lda_c_pw.mpl
params_a_a      = [0.0310907, 0.01554535, 0.0168869]
params_a_alpha1 = [0.21370,  0.20548,  0.11125]
params_a_beta1  = [7.5957, 14.1189, 10.357]
params_a_beta2  = [3.5876, 6.1977, 3.6231]
params_a_beta3  = [1.6382, 3.3662,  0.88026]
params_a_beta4  = [0.49294, 0.62517, 0.49671]
FZ20            = np.array([1.709920934161365617563962776245])


#######################
# SEMILOCAL BASELINES #
#######################

def get_rs(n):
    """
    Wigner-Seitz radius:
    """
    rs = (4 * np.pi * n / 3)**(-1.0/3)
    return rs, -rs / (3 * n)

def get_zeta(nu, nd):
    """
    Spin polarization
    """
    nt = nd + nu
    zeta = (-nd + nu) / nt
    dzetau = (2*nd/nt)/nt
    dzetad = (-2*nu/nt)/nt
    return zeta, dzetau, dzetad

def get_pw92term(rs, code):
    """
    Helper function for pw92 correlation
     0 - ec0
     1 - ec1
     2 - alphac
    """
    A = params_a_a[code]
    alpha1 = params_a_alpha1[code]
    beta1 = params_a_beta1[code]
    beta2 = params_a_beta2[code]
    beta3 = params_a_beta3[code]
    beta4 = params_a_beta4[code]
    Sqrt = np.sqrt
    lda = -2*A*(1 + alpha1*rs)*np.log(1 + 1/(2.*A*(beta1*Sqrt(rs) \
        + beta2*rs + beta3*rs**1.5 + beta4*rs**2)))
    dlda = ((1 + alpha1*rs)*(beta2 + beta1/(2.*Sqrt(rs)) + (3*beta3*Sqrt(rs))/2. \
        + 2*beta4*rs))/((beta1*Sqrt(rs) + beta2*rs + beta3*rs**1.5 \
        + beta4*rs**2)**2*(1 + 1/(2.*A*(beta1*Sqrt(rs) + beta2*rs + beta3*rs**1.5 \
            + beta4*rs**2)))) - 2*A*alpha1*np.log(1 + 1/(2.*A*(beta1*Sqrt(rs) + beta2*rs \
        + beta3*rs**1.5 + beta4*rs**2)))
    return lda, dlda

def get_pw92(rs, zeta):
    """
    PW92 correlation functional
    """
    ec0, dec0 = get_pw92term(rs, 0)
    ec1, dec1 = get_pw92term(rs, 1)
    alphac, dalphac = get_pw92term(rs, 2)

    fzeta = (-2 + (1 - zeta)**1.3333333333333333 + (1 + zeta)**1.3333333333333333)/(-2 + 2*2**0.3333333333333333)
    dfz = ((-4*(1 - zeta)**0.3333333333333333)/3. + (4*(1 + zeta)**0.3333333333333333)/3.)/(-2 + 2*2**0.3333333333333333)
    e = ec0 + (-ec0 + ec1)*fzeta*zeta**4 - (alphac*fzeta*(1 - zeta**4))/FZ20
    dedec0 = 1 - fzeta*zeta**4
    dedec1 = fzeta*zeta**4
    dedac0 = -((fzeta*(1 - zeta**4))/FZ20)
    dedfz = (-ec0 + ec1)*zeta**4 - (alphac*(1 - zeta**4))/FZ20
    dedzeta = 4*(-ec0 + ec1)*fzeta*zeta**3 + (4*alphac*fzeta*zeta**3)/FZ20

    dedrs = dedec0 * dec0 + dedec1 * dec1 + dedac0 * dalphac
    dedzeta = dedfz * dfz + dedzeta

    return e, dedrs, dedzeta

def get_phi0(zeta):
    """
    Spin polarization function for alpha=0 limit of SCAN
    """
    G = (1 - zeta**12)*(1 - 2.363*(-1 + ((1 - zeta)**1.3333333333333333 + (1 + zeta)**1.3333333333333333)/2.))
    dG = -1.1815*(1 - zeta**12)*((-4*(1 - zeta)**0.3333333333333333)/3. + (4*(1 + zeta)**0.3333333333333333)/3.) - 12*zeta**11*(1 - 2.363*(-1 + ((1 - zeta)**1.3333333333333333 + (1 + zeta)**1.3333333333333333)/2.))
    return G, dG

def get_phi1(zeta):
    """
    Spin polarization function for alpha=1 limt of SCAN
    """
    thr = 1-1e-10
    zeta = np.minimum(zeta, thr)
    zeta = np.maximum(zeta, -thr)
    p = 2./3
    pp = 1./3
    phi = ((1 - zeta)**p + (1 + zeta)**p)/2.
    dphi = (-2/(3.*(1 - zeta)**pp) + 2/(3.*(1 + zeta)**pp))/2.
    return phi, dphi

def get_baseline0inf(zeta, s2):
    """
    Define n_{gamma} = gamma^3 n(gamma r)
    Returns lim_{gamma -> infty} epsilon_{SCAN}(alpha=0)
    """
    G, dG = get_phi0(zeta)
    elim = b1c*G*np.log(1 + (1 - np.e)/(np.e*(1 + 4*chiinf*s2)**0.25))
    dedG = b1c*np.log(1 + (1 - np.e)/(np.e*(1 + 4*chiinf*s2)**0.25))
    deds2 = -((b1c*chiinf*(1 - np.e)*G)/(np.e*(1 + 4*chiinf*s2)**1.25*(1 + (1 - np.e)/(np.e*(1 + 4*chiinf*s2)**0.25))))
    return elim, dedG * dG, deds2

def get_baseline1inf(zeta, s2, ss=False):
    """
    Define n_{gamma} = gamma^3 n(gamma r)
    Returns lim_{gamma -> infty} epsilon_{SCAN}(alpha=1)
    """
    if ss:
        phi, dphi = 2**(-1.0/3), 0
    else:
        phi, dphi = get_phi1(zeta)
    elim = gamma*phi**3*np.log(1.0000000001 - (1 + (4*chi*s2)/phi**2)**(-0.25))
    dedphi = (-2*chi*gamma*s2)/((1 + (4*chi*s2)/phi**2)**1.25*(1.0000000001 - (1 + (4*chi*s2)/phi**2)**(-0.25))) + 3*gamma*phi**2*np.log(1.0000000001 - (1 + (4*chi*s2)/phi**2)**(-0.25))
    deds2 = (chi*gamma*phi)/((1 + (4*chi*s2)/phi**2)**1.25*(1.0000000001 - (1 + (4*chi*s2)/phi**2)**(-0.25)))
    return elim, dedphi*dphi, deds2

def get_baseline_inf(zeta, x2, chi):
    """
    Mixes baseline0inf and baseline1inf using chi in a smooth
    way that matches SCAN for chi={-1,0,1}
    """
    a = 17. / 3
    D = a * (1 - chi) / (a - chi)
    dDdchi = (a - a**2) / (a - chi)**2
    s2 = x2 / sprefac**2
    e0lim, de0dzeta, de0ds2 = get_baseline0inf(zeta, s2)
    e1lim, de1dzeta, de1ds2 = get_baseline1inf(zeta, s2)
    elim = e1lim * D + e0lim * (1 - D)
    dedzeta = de1dzeta * D + de0dzeta * (1 - D)
    deds2 = de1ds2 * D + de0ds2 * (1 - D)
    dedchi = (e1lim - e0lim) * dDdchi
    return elim, dedzeta, deds2 / sprefac**2, dedchi

def get_baseline_inf_z(nu, nd, g2, D):
    """
    Mixes baseline0inf and baseline1inf using tau_w / tau
    in a smooth way
    """
    N = nu.shape[0]
    s2, ds2n, ds2g2 = get_s2(nu+nd, g2 + 1e-30)
    zeta, dzetau, dzetad = get_zeta(nu, nd)
    e0lim, de0dzeta, de0ds2 = baseline0inf(zeta, s2)
    e1lim, de1dzeta, de1ds2 = baseline1inf(zeta, s2)
    elim = e1lim * D[0] + e0lim * (1 - D[0])
    dedzeta = de1dzeta * D[0] + de0dzeta * (1 - D[0])
    deds2 = de1ds2 * D[0] + de0ds2 * (1 - D[0])
    tmp = e1lim - e0lim
    vxc = [np.zeros((N,2)), np.zeros((N,3)),
           np.zeros((N,2))]
    fill_vxc_os_(vxc, tmp * D[1],
                 tmp * D[2],
                 tmp * D[3])
    vxc[0][:,0] += dedzeta * dzetau + deds2 * ds2n
    vxc[0][:,1] += dedzeta * dzetad + deds2 * ds2n
    vxc[1][:,0] += deds2 * ds2g2
    vxc[1][:,1] += 2 * deds2 * ds2g2
    vxc[1][:,2] += deds2 * ds2g2
    return elim, vxc

def get_baseline0(rs, zeta, s2):
    """
    epsilon_{SCAN}(alpha=0)
    """
    lda = -(b1c/(1 + b2c*np.sqrt(rs) + b3c*rs))
    dlda = (b1c*(b3c + b2c/(2.*np.sqrt(rs))))/(1 + b2c*np.sqrt(rs) + b3c*rs)**2
    G, dG = get_phi0(zeta)
    EC = G*(lda + b1c*np.log(1 + (-1 + np.exp(-lda/b1c))*(1 - (1 + 4*chiinf*s2)**(-0.25))))
    dECdlda = G*(1 - (1 - (1 + 4*chiinf*s2)**(-0.25))/(np.exp(lda/b1c)*(1 + (-1 + np.exp(-lda/b1c))*(1 - (1 + 4*chiinf*s2)**(-0.25)))))
    dECds2 = (b1c*chiinf*(-1 + np.exp(-lda/b1c))*G)/((1 + 4*chiinf*s2)**1.25*(1 + (-1 + np.exp(-lda/b1c))*(1 - (1 + 4*chiinf*s2)**(-0.25))))
    dECdGZ = lda + b1c*np.log(1 + (-1 + np.exp(-lda/b1c))*(1 - (1 + 4*chiinf*s2)**(-0.25)))
    return EC, dECdlda * dlda, dECdGZ * dG, dECds2

def get_baseline1(lda, rs, zeta, s2, ss=False):
    """
    epsilon_{SCAN}(alpha=1)
    """
    Pi = np.pi
    Log = np.log
    Exp = np.exp
    if ss:
        phi, dphi = 2**(-1.0/3), 0
    else:
        phi, dphi = get_phi1(zeta)
    beta = (0.066725*(1 + 0.1*rs))/(1 + 0.1778*rs)
    dbetadrs = (-0.011863705000000002*(1 + 0.1*rs))/(1 + 0.1778*rs)**2 + 0.006672500000000001/(1 + 0.1778*rs)
    w1 = -1. + np.exp(-lda/(gamma*phi**3))
    dw1dlda = -(1/(np.exp(lda/(gamma*phi**3))*gamma*phi**3))
    dw1dphi = (3*lda)/(np.exp(lda/(gamma*phi**3))*gamma*phi**4)
    t2 = (1.5**0.6666666666666666*Pi**1.3333333333333333*s2)/(4.*phi**2*rs)
    dt2drs = -(1.5**0.6666666666666666*Pi**1.3333333333333333*s2)/(4.*phi**2*rs**2)
    dt2ds2 = (1.5**0.6666666666666666*Pi**1.3333333333333333)/(4.*phi**2*rs)
    dt2dphi = -(1.5**0.6666666666666666*Pi**1.3333333333333333*s2)/(2.*phi**3*rs)
    e = lda + gamma*phi**3*Log(1 + (1 - (1 + (4*beta*t2)/(gamma*w1))**(-0.25))*w1)
    dedlda = 1
    dedt2 = (beta*phi**3)/((1 + (4*beta*t2)/(gamma*w1))**1.25*(1 + (1 - (1 + (4*beta*t2)/(gamma*w1))**(-0.25))*w1))
    dedphi = 3*gamma*phi**2*Log(1 + (1 - (1 + (4*beta*t2)/(gamma*w1))**(-0.25))*w1)
    dedbeta = (phi**3*t2)/((1 + (4*beta*t2)/(gamma*w1))**1.25*(1 + (1 - (1 + (4*beta*t2)/(gamma*w1))**(-0.25))*w1))
    dedw1 = (gamma*phi**3*(1 - (1 + (4*beta*t2)/(gamma*w1))**(-0.25) - (beta*t2)/(gamma*(1 + (4*beta*t2)/(gamma*w1))**1.25*w1)))/(1 + (1 - (1 + (4*beta*t2)/(gamma*w1))**(-0.25))*w1)
    
    dedrs = dedbeta * dbetadrs + dedt2 * dt2drs
    dedlda += dedw1 * dw1dlda
    deds2 = dedt2 * dt2ds2
    dedphi += dedt2 * dt2dphi + dedw1 * dw1dphi

    return e, dedlda, dedrs, dedphi * dphi, deds2

def get_baseline1b(rs, zeta, s2, ss=False):
    """
    epsilon_{SCAN}(alpha=1)
    """
    lda, dldadrs, dldadzeta = get_pw92(rs, zeta)
    Pi = np.pi
    Log = np.log
    Exp = np.exp
    if ss:
        phi, dphi = 2**(-1.0/3), 0
    else:
        phi, dphi = get_phi1(zeta)
    beta = (0.066725*(1 + 0.1*rs))/(1 + 0.1778*rs)
    dbetadrs = (-0.011863705000000002*(1 + 0.1*rs))/(1 + 0.1778*rs)**2 + 0.006672500000000001/(1 + 0.1778*rs)
    w1 = -1. + np.exp(-lda/(gamma*phi**3))
    dw1dlda = -(1/(np.exp(lda/(gamma*phi**3))*gamma*phi**3))
    dw1dphi = (3*lda)/(np.exp(lda/(gamma*phi**3))*gamma*phi**4)
    t2 = (1.5**0.6666666666666666*Pi**1.3333333333333333*s2)/(4.*phi**2*rs)
    dt2drs = -(1.5**0.6666666666666666*Pi**1.3333333333333333*s2)/(4.*phi**2*rs**2)
    dt2ds2 = (1.5**0.6666666666666666*Pi**1.3333333333333333)/(4.*phi**2*rs)
    dt2dphi = -(1.5**0.6666666666666666*Pi**1.3333333333333333*s2)/(2.*phi**3*rs)
    e = lda + gamma*phi**3*Log(1 + (1 - (1 + (4*beta*t2)/(gamma*w1))**(-0.25))*w1)
    dedlda = 1
    dedt2 = (beta*phi**3)/((1 + (4*beta*t2)/(gamma*w1))**1.25*(1 + (1 - (1 + (4*beta*t2)/(gamma*w1))**(-0.25))*w1))
    dedphi = 3*gamma*phi**2*Log(1 + (1 - (1 + (4*beta*t2)/(gamma*w1))**(-0.25))*w1)
    dedbeta = (phi**3*t2)/((1 + (4*beta*t2)/(gamma*w1))**1.25*(1 + (1 - (1 + (4*beta*t2)/(gamma*w1))**(-0.25))*w1))
    dedw1 = (gamma*phi**3*(1 - (1 + (4*beta*t2)/(gamma*w1))**(-0.25) - (beta*t2)/(gamma*(1 + (4*beta*t2)/(gamma*w1))**1.25*w1)))/(1 + (1 - (1 + (4*beta*t2)/(gamma*w1))**(-0.25))*w1)
    
    dedrs = dedbeta * dbetadrs + dedt2 * dt2drs
    dedlda += dedw1 * dw1dlda
    deds2 = dedt2 * dt2ds2
    dedphi += dedt2 * dt2dphi + dedw1 * dw1dphi

    dedrs += dedlda * dldadrs
    dedzeta = dedphi * dphi + dedlda * dldadzeta

    return e, dedrs, dedzeta, deds2

def get_ss_baseline(n, g2):
    """
    epsilon_{SCAN}(alpha=1,zeta=1)
    """
    N = n.shape[0]
    rs, drs = get_rs(n)
    s2, ds2n, ds2g2 = get_s2(n, g2)
    #lda, dlda = eval_xc(',LDA_C_PW_MOD', (n, 0*n), spin=1)[:2]
    lda, dlda = get_pw92term(rs, 1)
    #dlda = (dlda[0][:,0] - lda) / n
    e, dedlda, dedrs, _, deds2 = get_baseline1(lda, rs, 1, s2, ss=True)
    vxc = [None, None, None, None]
    #vxc[0] = dedrs * drs + deds2 * ds2n + dedlda * dlda
    vxc[0] = (dedrs + dedlda * dlda) * drs + deds2 * ds2n
    vxc[1] = deds2 * ds2g2 * n
    vxc[0] = vxc[0] * n + e
    return e, vxc

def get_os_baseline(nu, nd, g2, type=0):
    """
    epsilon_{SCAN}(alpha=type)
    type=0 or 1
    """
    N = nu.shape[0]
    rs, drs = get_rs(nu+nd)
    zeta, dzetau, dzetad = get_zeta(nu, nd)
    s2, ds2n, ds2g2 = get_s2(nu+nd, g2)
    if type == 0:
        e, dedrs, dedzeta, deds2 = get_baseline0(rs, zeta, s2)
    else:
        lda, dldadrs, dldadzeta = get_pw92(rs, zeta)
        e, dedlda, dedrs, dedzeta, deds2 = get_baseline1(lda, rs, zeta, s2)
        dedrs += dedlda * dldadrs
        dedzeta += dedlda * dldadzeta
    vxc = [np.zeros((N,2)), np.zeros((N,3)), None, None]
    vxc[0][:,0] = dedrs * drs + dedzeta * dzetau + deds2 * ds2n
    vxc[0][:,1] = dedrs * drs + dedzeta * dzetad + deds2 * ds2n
    vxc[0] *= (nu + nd)[:,np.newaxis]
    vxc[0] += e[:,np.newaxis]
    vxc[1][:,0] = deds2 * ds2g2
    vxc[1][:,1] = deds2 * 2 * ds2g2
    vxc[1][:,2] = deds2 * ds2g2
    vxc[1] *= (nu + nd)[:,np.newaxis]
    return e, vxc

def get_os_baseline2(n, zeta, x2, type=0):
    """
    epsilon_{SCAN}(alpha=type)
    type=0 or 1
    """
    N = n.shape[0]
    rs, drs = get_rs(n)
    s2 = x2 / sprefac**2
    if type == 0:
        e, dedrs, dedzeta, deds2 = get_baseline0(rs, zeta, s2)
    else:
        e, dedrs, dedzeta, deds2 = get_baseline1b(rs, zeta, s2)
    dedn = -rs / 3 * dedrs
    return e, -rs / 3 * dedrs + e, dedzeta * n, n * deds2 / sprefac**2


##########################################
# SCALE-INVARIANT SEMI-LOCAL DESCRIPTORS #
##########################################

def get_x2(n, g2):
    """
    Takes n and g2 and returns the squared normalized gradient
    without the constant included in s^2
    x^2 = g2 / n^{8/3}
    """
    return g2/(n**2.6666666666666665+1e-20),\
           (-8*g2)/(3.*n**3.6666666666666665+1e-20),\
           1/(n**(2.6666666666666665)+1e-20)

def get_s2(n, g2):
    """
    Returns the normalized gradient (see get_x2) with the usual
    normalization constant
    s^2 = g2 / (sprefac^2 * n^{8/3})
    """
    a, b, c = get_x2(n, g2)
    return a / sprefac**2 + 1e-16, b / sprefac**2, c / sprefac**2

def get_alpha(n, zeta, g2, t):
    """
    The iso-orbital indicator alpha used by SCAN
    """
    d = 0.5 * ((1+zeta)**(5./3) + (1-zeta)**(5./3))
    dd = (5./6) * ((1+zeta)**(2./3) - (1-zeta)**(2./3))
    alpha = (t - g2/(8*n)) / (CFC*n**(5./3)*d)
    return alpha,\
           ((-5 * t + g2 / n) / n) / (3 * CFC * d * n**(5./3)),\
           -alpha / d * dd,\
           -1.0 / (8 * CFC * n**(8./3) * d),\
           1.0 / (CFC * n**(5./3) * d)
#-(5.0/3) * alpha / n + (g2 / (8*n**2)) / (CFC * n**(5./3) * d),\
#(-5 * t + g2 / n) / (3 * CFC * d * n**(8./3)),\

def get_chi(alpha):
    """
    Transformation of alpha to [-1,1], similar but not identical
    to the electron localization factor (ELF).
    chi = (1 - alpha) / (1 + alpha)
    single-orbital -> chi = 1
    HEG -> chi = 0
    atomic tail or non-covalent bond -> chi = -1
    """
    chi = (1 - alpha) / (1 + alpha)
    return chi, -2 / (1 + alpha)**2

def get_chi_full_deriv(n, zeta, g2, t):
    """
    Returns chi directly without first calculating alpha,
    for numerical stability.
    """
    d = 0.5 * ((1+zeta)**(5./3) + (1-zeta)**(5./3))
    dd = (5./6) * ((1+zeta)**(2./3) - (1-zeta)**(2./3))
    tau_u = CFC * d * n**(5./3)
    tau_w = g2/(8*n)
    tau = np.maximum(t, tau_w)
    D = (tau_u + tau - tau_w)
    chi = np.sin(np.pi/2 * (tau_u - tau + tau_w) / D)
    #chi = (tau_u - tau + tau_w) / D
    deriv = np.pi/2 * np.cos(np.pi/2 * (tau_u - tau + tau_w) / D)
    tmp1 = 2 * (tau - tau_w) / D
    tmp2 = CFC * (5./3) * n**(2./3) * d / D
    tmp3 = CFC * n**(5./3) * dd / D
    tmp4 = 2 * tau_u / D
    tmp5 = -(tau_w / n) / D
    tmp6 = 1 / (8 * n * D)
    return chi,\
           deriv*(tmp1 * tmp2 + tmp4 * tmp5),\
           deriv*(tmp1 * tmp3),\
           deriv*(tmp4 * tmp6),\
           deriv*(-tmp4 / D)




#####################
# Descriptor Arrays #
#####################

def get_chi_desc(chi):
    """
    Polynomial arrays and derivatives for chi
    TODO needs cleanup and return format more consistent with
    other descriptrs, e.g. get_chi_terms
    """
    return chi, chi**2, chi**3, chi**4, np.ones_like(chi), 2*chi, 3*chi**2, 4*chi**3

def get_separate_xef_terms(f, return_deriv=True, cf=2.0):
    """
    Returns (f-1)^n exp(-c * (f-1)^2),
    where f is the XEF and n is [0,1,2,3,4]
    """
    f = f - 1
    fterm0 = np.exp(-cf * f**2)
    res = np.array([fterm0 * f**i for i in range(5)])
    res[0,:] -= 1
    if return_deriv:
        dres = np.array([i * f**(i-1) - 2 * cf * f**(i+1) for i in range(5)])
        dres *= fterm0
        return res, dres
    return res

def get_separate_sl_terms(x2, chi, gamma):
    """
    Takes x2, chi, and a constant gamma and constructs
    semi-local descriptors
    """
    desc = np.zeros((15, x2.shape[0]))
    dx2 = np.zeros((15, x2.shape[0]))
    dchi = np.zeros((15, x2.shape[0]))
    u = gamma * x2 / (1 + gamma * x2)
    du = gamma / (1 + gamma * x2)**2
    a0, a1, a2, a3, da0, da1, da2, da3 = get_chi_desc(chi)
    ind = 0
    for a, da in zip([a0, a1, a2], [da0, da1, da2]):
        desc[ind] = a
        dchi[ind] = da
        ind += 1
    for i in range(3):
        diff = u**(i+1)
        ddiff = (i+1) * u**(i)
        ddiff *= du
        desc[ind] = diff
        dx2[ind] = ddiff
        ind += 1
        for a, da in zip([a0, a1, a2], [da0, da1, da2]):
            desc[ind] = diff * a
            dx2[ind] = ddiff * a
            dchi[ind] = diff * da
            ind += 1
    return desc, dx2, dchi

def get_separate_xefa_terms(F, chi):
    """
    Takes F and chi and constructs nonlocal descriptors.
    """
    x, dx = get_separate_xef_terms(F, return_deriv=True)
    x = x[1:4]
    dx = dx[1:4]
    desc = np.zeros((13, F.shape[0]))
    df = np.zeros((13, F.shape[0]))
    dchi = np.zeros((13, F.shape[0]))
    a0, a1, a2, a3, da0, da1, da2, da3 = get_chi_desc(chi)
    ind = 0
    for i in range(3):
        diff = x[i]
        ddiff = dx[i]
        desc[ind] = diff
        df[ind] = ddiff
        ind += 1
        for a, da in zip([a0, a1, a2], [da0, da1, da2]):
            desc[ind] = diff * a 
            df[ind] = ddiff * a 
            dchi[ind] = diff * da
            ind += 1
    desc[ind] = F-1
    df[ind] = 1
    return desc, df, dchi

def get_sl_small(x2, chi, gamma):
    """
    Takes x2, chi, and a constant gamma and constructs
    semi-local descriptors
    """
    desc = np.zeros((8, x2.shape[0]))
    dx2 = np.zeros((8, x2.shape[0]))
    dchi = np.zeros((8, x2.shape[0]))
    u = gamma * x2 / (1 + gamma * x2)
    du = gamma / (1 + gamma * x2)**2
    a0, a1, a2, a3, da0, da1, da2, da3 = get_chi_desc(chi)
    ind = 0
    for a, da in zip([a1, a2], [da1, da2]):
        desc[ind] = a
        dchi[ind] = da
        ind += 1
    for i in range(2):
        diff = u**(i+1)
        ddiff = (i+1) * u**(i)
        ddiff *= du
        desc[ind] = diff
        dx2[ind] = ddiff
        ind += 1
        for a, da in zip([a1, a2], [da1, da2]):
            desc[ind] = diff * a
            dx2[ind] = ddiff * a
            dchi[ind] = diff * da
            ind += 1
    return desc, dx2, dchi

def get_xefa_small(F, chi):
    """
    Takes F and chi and constructs nonlocal descriptors.
    """
    x, dx = get_separate_xef_terms(F, return_deriv=True)
    x = x[1:4]
    dx = dx[1:4]
    desc = np.zeros((7, F.shape[0]))
    df = np.zeros((7, F.shape[0]))
    dchi = np.zeros((7, F.shape[0]))
    a0, a1, a2, a3, da0, da1, da2, da3 = get_chi_desc(chi)
    ind = 0
    for i in range(3):
        diff = x[i]
        ddiff = dx[i]
        for a, da in zip([a0, a2], [da0, da2]):
            desc[ind] = diff * a
            df[ind] = ddiff * a
            dchi[ind] = diff * da
            ind += 1
    desc[ind] = F-1
    df[ind] = 1
    return desc, df, dchi

def get_mn15_rho_desc(n):
    den = (1 + 2.5 * n**(1./3))
    return 1 / den, -(5./6) / den**2 * n**(-2./3)

def get_t2(n, zeta, x2):
    t2 = (np.pi / 3)**(1./3) / (16 * phi**2) * x2 * n**(1./3)
    dt2dn = t2 / (3 * n)
    dt2dz = -2 * t2 / phi * (1./3) * ((1+zeta)**(-1./3) - (1-zeta)**(-1./3))
    dt2dx2 = (np.pi / 3)**(1./3) / (16 * phi**2) * n**(1./3)
    return t2, dt2dn, dt2dz, dt2dx2

def get_t2_desc(n, zeta, x2):
    t2, dt2dn, dt2dz, dt2dx2 = get_t2(n, zeta, x2)
    desc = 1 / (1 + 0.5 * t2)
    ddesc = -0.5 / (1 + 0.5 * t2)**2
    return desc, ddesc * dt2dn, ddesc * dt2dz, ddesc * dt2dx2

def get_t2_nsp(n, x2):
    t2 = (np.pi / 3)**(1./3) / 16 * x2 * n**(1./3)
    dt2dn = t2 / (3 * n)
    dt2dx2 = (np.pi / 3)**(1./3) / 16 * n**(1./3)
    return t2, dt2dn, dt2dx2

def get_t2_desc_nsp(n, x2):
    t2, dt2dn, dt2dx2 = get_t2(n, x2)
    desc = 1 / (1 + 0.5 * t2)
    ddesc = -0.5 / (1 + 0.5 * t2)**2
    return desc, ddesc * dt2dn, ddesc * dt2dx2

def get_combined_amix_desc(n, zeta, x2):
    a0, da0dn, da0dz, da0dx2 = get_t2_desc(n, zeta, x2)
    a1, da1dn = get_mn15_rho_desc(n)
    desc = np.zeros((5, n.shape[0]))
    ddescdn = np.zeros((5, n.shape[0]))
    ddescdz = np.zeros((5, n.shape[0]))
    ddescdx2 = np.zeros((5, n.shape[0]))

    desc[0] = a0
    ddescdn[0] = da0dn
    ddescdz[0] = da0dz
    ddescdx2[0] = da0dx2

    desc[1] = a1
    ddescdn[1] = da1dn

    desc[2] = a0 * a1
    ddescdn[2] = a0 * da1dn + da0dn * a1
    ddescdz[2] = da0dz * a1
    ddescdx2[2] = da0dx2 * a1

    desc[3] = a0**2
    ddescdn[3] = 2*a0 * da0dn
    ddescdz[3] = 2*a0 * da0dz
    ddescdx2[3] = 2*a0 * da0dx2

    desc[4] = a1**2
    ddescdn[4] = 2*a1 * da1dn

    return desc, ddescdn, ddescdz, ddescdx2

def get_compact_chi_desc(zeta, chi):
    zeros = np.zeros_like(zeta)
    zterm = 1 - zeta**8
    dzterm = -8*zeta**7
    desc = np.array([1-chi**2, chi**2 * zterm, (chi+chi**2)/2*zterm,
                     chi-chi**3, chi**2-chi**4])
    ddescdz = np.array([zeros, chi**2 * dzterm, (chi+chi**2)/2*dzterm,
                        zeros, zeros])
    ddescdchi = np.array([-2*chi, 2*chi*zterm, (1+2*chi)/2*zterm,
                          1-3*chi**2, 2*chi-4*chi**3])
    return desc, ddescdz, ddescdchi

def get_rmn15_desc(n, zeta, x2, chi, F):
    a0, da0dn, da0dz, da0dx2 = get_combined_amix_desc(n, zeta, x2)
    a1, da1dz, da1dchi = get_compact_chi_desc(zeta, chi)
    desc = np.zeros((25, n.shape[0]))
    ddescdn = np.zeros((25, n.shape[0]))
    ddescdz = np.zeros((25, n.shape[0]))
    ddescdx2 = np.zeros((25, n.shape[0]))
    ddescdchi = np.zeros((25, n.shape[0]))
    ddescdf = np.zeros((25, n.shape[0]))
    for i in range(5):
        for j in range(5):
            k = 5*i+j
            desc[k] = a0[i] * a1[j] * (F - 1)
            ddescdn[k] = da0dn[i] * a1[j] * (F - 1)
            ddescdz[k] = da0dz[i] * a1[j] + a0[i] * da1dz[j] * (F - 1)
            ddescdx2[k] = da0dx2[i] * a1[j] * (F - 1)
            ddescdchi[k] = a0[i] * da1dchi[j] * (F - 1)
            ddescdf[k] = a0[i] * a1[j]
    return desc, ddescdn, ddescdz, ddescdx2, ddescdchi

def get_rmn15_desc2(n, zeta, x2, chi, version):
    v = get_mn15_rho_desc(n)[0]
    u = 1 / (1 + 0.004 * x2)
    w = chi
    m1 = v * u
    m2 = v * (u**2 - u)
    m3 = (v**2 - v) * u
    if version == 'a': # For F-1
        wterms = np.array([w/2+0.5, w**2-1, w**3-w])
        wterms2 = wterms[1:]
    elif version == 'b': # For 1
        wterms = np.array([(w-w**2)/2, w**3-w])
        wterms2 = wterms.copy()
    else: # For (f-1)^2 * exp(-(f-1)^2)
        wterms = np.array([w, w**2-1, w**3-w])
        wterms2 = wterms[1:]
    wterms *= 1 - ((1+zeta)**(4./3) + (1-zeta)**(4./3) - 2) / (2**(4./3) - 2)
    wterms = np.append(wterms, wterms2, axis=0)
    return np.concatenate([m1 * wterms, m2 * wterms,
                           m3 * wterms], axis=0)

def get_chidesc_small(chi):
    return np.array([chi**2-chi, chi**3-chi, chi**4-chi**2]),\
           np.array([2*chi-1, 3*chi**2-1, 4*chi**3-2*chi])

def get_amix_schmidt(n, zeta, x2, chi):
    """
    Mixing parameter defined by Schmidt and Kummel 2014
    """
    zeta = np.minimum(zeta, 1-1e-8)
    zeta = np.maximum(zeta, -1+1e-8)
    phi = 0.5 * ((1+zeta)**(2./3) + (1-zeta)**(2./3))
    num = 1 - chi**4 * zeta**12
    t2 = (np.pi / 3)**(1./3) / (16 * phi**2) * x2 * n**(1./3)
    den = 1 + 0.5 * t2
    f = num / den
    dfdchi = -zeta**12 / den * 4 * chi**3
    dfdz = -12 * zeta**11 * chi**4 / den
    dfdt2 = -0.5 * num / den**2
    dt2dz = -2 * t2 / phi * (1./3) * ((1+zeta)**(-1./3) - (1-zeta)**(-1./3))
    dfdz += dfdt2 * dt2dz
    dfdx2 = dfdt2 * (np.pi / 3)**(1./3) / (16 * phi**2) * n**(1./3)
    # TODO used to be -1/3 below, but I think this was a typo
    dfdn = dfdt2 * (1./3) * t2 / n
    return f, dfdn, dfdz, dfdx2, dfdchi

def get_amix_schmidt2(n, zeta, x2, chi, order=1, mixer='c', zorder=1):
    """
    Mixing parameter defined by Schmidt and Kummel 2014
    """
    if mixer == 'c':
        chip = chi**2
        dchip = 2*chi
    elif mixer == 'l':
        chip = chi**4+chi-chi**5
        dchip = 4*chi**3+1-5*chi**4
    elif mixer == 'r':
        chip = chi**4-chi+chi**5
        dchip = 4*chi**3-1+5*chi**4
    else:
        chip = 1
        dchip = 0
    phi = ((1+zeta)**(4./3) + (1-zeta)**(4./3) - 2) / (2**(4./3) - 2)
    dphi = 4./3 * ((1+zeta)**(1./3) - (1-zeta)**(1./3)) / (2**(4./3) - 2)
    if zorder > 1:
        phi = phi**zorder
        dphi = zorder * phi**(zorder-1)
    elif zorder < 1:
        phi = 1
        dphi = 0
    num = 1 - chip * phi
    t2 = (np.pi / 3)**(1./3) / 16 * x2 * n**(1./3)
    den = 1 + 0.5 * t2
    f = num / den**order
    dfdchi = -phi / den * dchip
    dfdz = -dphi * chip / den
    dfdt2 = -0.5 * order * num / den**(order+1)
    dfdx2 = dfdt2 * (np.pi / 3)**(1./3) / 16 * n**(1./3)
    # TODO used to be -1/3 below, but I think this was a typo
    dfdn = dfdt2 * (1./3) * t2 / n
    return f, dfdn, dfdz, dfdx2, dfdchi

def get_amix_psts(rhou, rhod, g2, Do):
    """
    Mixing parameter for PSTS.
    Note: Returns 1-a_{mix}, where a_{mix} is the PSTS mixing parameter
    """
    rhot = rhou + rhod

    elim, vxclim = get_baseline_inf_z(rhou, rhod, g2, Do)

    exlda = 2**(1.0 / 3) * LDA_FACTOR * rhou**(4.0/3)
    exlda += 2**(1.0 / 3) * LDA_FACTOR * rhod**(4.0/3)
    dinvldau = -2**(1.0 / 3) * (4.0/3) * LDA_FACTOR * rhou**(1.0/3) / exlda**2 * (rhou + rhod)
    dinvldau += 1 / exlda
    dinvldad = -2**(1.0 / 3) * (4.0/3) * LDA_FACTOR * rhod**(1.0/3) / exlda**2 * (rhou + rhod)
    dinvldad += 1 / exlda
    exlda /= (rhot)
    u = elim / exlda
    amix = 1 - 1/(1 + A*np.log(1 + B*u))
    damix = (A*B)/((1 + B*u)*(1 + A*np.log(1 + B*u))**2)
    vxclim[0][:,0] = vxclim[0][:,0] / exlda + elim * dinvldau
    vxclim[0][:,1] = vxclim[0][:,1] / exlda + elim * dinvldad
    vxclim[1] /= exlda[:,np.newaxis]
    vxclim[2] /= exlda[:,np.newaxis]
    for i in range(3):
        vxclim[i] *= damix[:,np.newaxis]

    return amix, vxclim
