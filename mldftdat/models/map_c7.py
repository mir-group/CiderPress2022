# Autocode from mathematica for VSXC-type contribs
import numpy as np
from pyscf.dft.libxc import eval_xc

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

def fill_vxc_ss_(vxc, spin, dn, dg2, dt, df=None):
    vxc[0][:,spin] += dn
    vxc[1][:,2*spin] += dg2
    vxc[2][:,spin] += dt
    if df is not None:
        vxc[3][:,spin] += df
def fill_vxc_os_(vxc, dn, dg2, dt, df=None,
                 dftdnu=None, dftdnd=None,
                 dftdfu=None, dftdfd=None):
    vxc[0][:,0] += dn
    vxc[0][:,1] += dn
    vxc[1][:,0] += dg2
    vxc[1][:,1] += 2 * dg2
    vxc[1][:,2] += dg2
    vxc[2][:,0] += dt
    vxc[2][:,1] += dt
    if df is not None:
        vxc[0][:,0] += df * dftdnu
        vxc[0][:,1] += df * dftdnd
        vxc[3][:,0] += df * dftdfu
        vxc[3][:,1] += df * dftdfd
def fill_vxc_base_ss_(vxc, vterm, multerm, spin):
    if vterm[0] is not None:
        vxc[0][:,spin] += vterm[0] * multerm
    if vterm[1] is not None:
        vxc[1][:,2*spin] += vterm[1] * multerm
    if vterm[2] is not None:
        vxc[2][:,spin] += vterm[2] * multerm
def fill_vxc_base_os_(vxc, vterm, multerm):
    multerm = multerm.reshape(-1, 1)
    if vterm[0] is not None:
        vxc[0] += vterm[0] * multerm
    if vterm[1] is not None:
        vxc[1] += vterm[1] * multerm
    if vterm[2] is not None:
        vxc[2] += vterm[2] * multerm

class VSXCContribs():

    def __init__(self, cx, c0, c1, dx, d0, d1,
                 fterm_scale=1.0):
        self.cx = cx
        self.c0 = c0
        self.c1 = c1
        self.dx = dx
        self.d0 = d0
        self.d1 = d1
        self.cf = fterm_scale

    def get_rs(self, n):
        rs = (4 * np.pi * n / 3)**(-1.0/3)
        return rs, -rs / (3 * n)

    def get_zeta(self, nu, nd):
        zeta = (-nd + nu)/(nd + nu + 1e-10)
        dzetau = (2*nd)/((nd + nu)**2 + 1e-10)
        dzetad = (-2*nu)/((nd + nu)**2 + 1e-10)
        return zeta, dzetau, dzetad

    def pw92term(self, rs, code):
        # 0 - ec0
        # 1 - ec1
        # 2 - alphac
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

    def pw92(self, rs, zeta):
        ec0, dec0 = self.pw92term(rs, 0)
        ec1, dec1 = self.pw92term(rs, 1)
        alphac, dalphac = self.pw92term(rs, 2)

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

    def get_phi0(self, zeta):
        G = (1 - zeta**12)*(1 - 2.363*(-1 + ((1 - zeta)**1.3333333333333333 + (1 + zeta)**1.3333333333333333)/2.))
        dG = -1.1815*(1 - zeta**12)*((-4*(1 - zeta)**0.3333333333333333)/3. + (4*(1 + zeta)**0.3333333333333333)/3.) - 12*zeta**11*(1 - 2.363*(-1 + ((1 - zeta)**1.3333333333333333 + (1 + zeta)**1.3333333333333333)/2.))
        return G, dG

    def get_phi1(self, zeta):
        phi = ((1 - zeta)**0.6666666666666666 + (1 + zeta)**0.6666666666666666)/2.
        dphi = (-2/(3.*(1 - zeta)**0.3333333333333333) + 2/(3.*(1 + zeta)**0.3333333333333333))/2.
        return phi, dphi

    def baseline0inf(self, zeta, s2):
        G, dG = self.get_phi0(zeta)
        elim = b1c*G*np.log(1 + (1 - np.e)/(np.e*(1 + 4*chiinf*s2)**0.25))
        dedG = b1c*np.log(1 + (1 - np.e)/(np.e*(1 + 4*chiinf*s2)**0.25))
        deds2 = -((b1c*chiinf*(1 - np.e)*G)/(np.e*(1 + 4*chiinf*s2)**1.25*(1 + (1 - np.e)/(np.e*(1 + 4*chiinf*s2)**0.25))))
        return elim, dedG * dG, deds2

    def baseline1inf(self, zeta, s2, ss=False):
        if ss:
            phi, dphi = 2**(-1.0/3), 0
        else:
            phi, dphi = self.get_phi1(zeta)
        elim = gamma*phi**3*np.log(1.0000000001 - (1 + (4*chi*s2)/phi**2)**(-0.25))
        dedphi = (-2*chi*gamma*s2)/((1 + (4*chi*s2)/phi**2)**1.25*(1.0000000001 - (1 + (4*chi*s2)/phi**2)**(-0.25))) + 3*gamma*phi**2*np.log(1.0000000001 - (1 + (4*chi*s2)/phi**2)**(-0.25))
        deds2 = (chi*gamma*phi)/((1 + (4*chi*s2)/phi**2)**1.25*(1.0000000001 - (1 + (4*chi*s2)/phi**2)**(-0.25)))
        return elim, dedphi*dphi, deds2

    def baseline_inf(self, nu, nd, g2, D):
        # WARNING: derivs not implemented for this version
        #D = [D]
        N = nu.shape[0]
        s2, ds2n, ds2g2 = self.get_s2(nu+nd, g2 + 1e-30)
        zeta, dzetau, dzetad = self.get_zeta(nu, nd)
        e0lim, de0dzeta, de0ds2 = self.baseline0inf(zeta, s2)
        e1lim, de1dzeta, de1ds2 = self.baseline1inf(zeta, s2)
        elim = e1lim * D[0] + e0lim * (1 - D[0])
        dedzeta = de1dzeta * D[0] + de0dzeta * (1 - D[0])
        deds2 = de1ds2 * D[0] + de0ds2 * (1 - D[0])
        tmp = e1lim - e0lim
        vxc = [np.zeros((N,2)), np.zeros((N,3)),
               np.zeros((N,2))]
        #fill_vxc_os_(vxc, tmp * D[1],
        #             tmp * D[2],
        #             tmp * D[3])
        vxc[0][:,0] += dedzeta * dzetau + deds2 * ds2n
        vxc[0][:,1] += dedzeta * dzetad + deds2 * ds2n
        vxc[1][:,0] += deds2 * ds2g2
        vxc[1][:,1] += 2 * deds2 * ds2g2
        vxc[1][:,2] += deds2 * ds2g2
        return elim, vxc
    
    def baseline0(self, rs, zeta, s2):
        lda = -(b1c/(1 + b2c*np.sqrt(rs) + b3c*rs))
        dlda = (b1c*(b3c + b2c/(2.*np.sqrt(rs))))/(1 + b2c*np.sqrt(rs) + b3c*rs)**2
        G, dG = self.get_phi0(zeta)
        EC = G*(lda + b1c*np.log(1 + (-1 + np.exp(-lda/b1c))*(1 - (1 + 4*chiinf*s2)**(-0.25))))
        dECdlda = G*(1 - (1 - (1 + 4*chiinf*s2)**(-0.25))/(np.exp(lda/b1c)*(1 + (-1 + np.exp(-lda/b1c))*(1 - (1 + 4*chiinf*s2)**(-0.25)))))
        dECds2 = (b1c*chiinf*(-1 + np.exp(-lda/b1c))*G)/((1 + 4*chiinf*s2)**1.25*(1 + (-1 + np.exp(-lda/b1c))*(1 - (1 + 4*chiinf*s2)**(-0.25))))
        dECdGZ = lda + b1c*np.log(1 + (-1 + np.exp(-lda/b1c))*(1 - (1 + 4*chiinf*s2)**(-0.25)))
        return EC, dECdlda * dlda, dECdGZ * dG, dECds2

    def baseline1(self, lda, rs, zeta, s2, ss=False): 
        Pi = np.pi
        Log = np.log
        Exp = np.exp
        if ss:
            phi, dphi = 2**(-1.0/3), 0
        else:
            phi, dphi = self.get_phi1(zeta)
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
        dedrs += dedt2 * dt2drs + dedbeta * dbetadrs
        deds2 = dedt2 * dt2ds2
        dedphi += dedt2 * dt2dphi + dedw1 * dw1dphi
        
        return e, dedlda, dedrs, dedphi * dphi, deds2

    def ss_baseline(self, n, g2):
        N = n.shape[0]
        rs, drs = self.get_rs(n)
        s2, ds2n, ds2g2 = self.get_s2(n, g2)
        #lda, dlda = eval_xc(',LDA_C_PW_MOD', (n, 0*n), spin=1)[:2]
        lda, dlda = self.pw92term(rs, 1)
        #dlda = (dlda[0][:,0] - lda) / n
        e, dedlda, dedrs, _, deds2 = self.baseline1(lda, rs, 1, s2, ss=True)
        vxc = [None, None, None, None]
        #vxc[0] = dedrs * drs + deds2 * ds2n + dedlda * dlda
        vxc[0] = (dedrs + dedlda * dlda) * drs + deds2 * ds2n
        vxc[1] = deds2 * ds2g2 * n
        vxc[0] = vxc[0] * n + e
        return e, vxc

    def os_baseline(self, nu, nd, g2, type = 0):
        N = nu.shape[0]
        rs, drs = self.get_rs(nu+nd)
        zeta, dzetau, dzetad = self.get_zeta(nu, nd)
        s2, ds2n, ds2g2 = self.get_s2(nu+nd, g2)
        if type == 0:
            e, dedrs, dedzeta, deds2 = self.baseline0(rs, zeta, s2)
        else:
            lda, dldadrs, dldadzeta = self.pw92(rs, zeta)
            e, dedlda, dedrs, dedzeta, deds2 = self.baseline1(lda, rs, zeta, s2)
            dedrs += dedlda * dldadrs
            dedzeta += dedzeta * dldadzeta
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

    def get_separate_xef_terms(self, f, return_deriv=False):
        f = f - 1
        fterm0 = np.exp(-self.cf * f**2)
        res = np.array([fterm0 * f**i for i in range(5)])
        res[0,:] -= 1
        return res

    def get_separate_sl_terms(self, x2, chi, gamma):
        desc = np.zeros((14,x2.shape[0]))
        dx2 = np.zeros((14,x2.shape[0]))
        dchi = np.zeros((14,x2.shape[0]))
        u = gamma * x2 / (1 + gamma * x2)
        du = gamma / (1 + gamma * x2)**2
        a0, a1, a2, da0, da1, da2 = self.get_chi_desc(chi)
        ind = 0
        for i in range(4):
            diff = u**(i+1) - u**(i+2)
            ddiff = (i+1) * u**(i) - (i+2) * u**(i+1)
            desc[ind] = diff
            dx2[ind] = (i+1) * ddiff
            ind += 1
            desc[ind] = diff * a0
            dx2[ind] = ddiff * a0
            dchi[ind] = diff * da0 
            ind += 1
            desc[ind] = diff * a1
            dx2[ind] = ddiff * a1
            dchi[ind] = diff * da1
            ind += 1
            #desc[ind] = u**(i+1) * a2
            #dx2[ind] = (i+1) * u**i * a2
            #dchi[ind] = u**(i+1) * da2
            #ind += 1
        desc[ind] = a0
        dchi[ind] = da0
        ind += 1
        desc[ind] = a1
        dchi[ind] = da1
        #ind += 1
        #desc[ind] = a2
        #dchi[ind] = da2
        return desc, dx2, dchi

    def sl_terms(self, x2, chi, gamma, c):
        y, dydx2, dydchi = self.get_separate_sl_terms(x2, chi, gamma, c)
        return c.dot(y), c.dot(dydx2), c.dot(dydchi)

    def xef_terms(self, f, c):
        y = 0
        d = 0
        f = f - 1
        fterm0 = np.exp(-self.cf * f**2)
        for i in range(4):
            y += c[i] * f**(i+1)
            d += c[i] * ((i+1) * f**i - 2 * self.cf * f**(i+2))
        return y * fterm0, d * fterm0

    def get_x2(self, n, g2):
        return g2/n**2.6666666666666665,\
               (-8*g2)/(3.*n**3.6666666666666665),\
               n**(-2.6666666666666665)

    def get_s2(self, n, g2):
        a, b, c = self.get_x2(n, g2)
        return a / sprefac**2 + 1e-10, b / sprefac**2, c / sprefac**2

    def get_alpha(self, n, zeta, g2, t):
        d = 0.5 * ((1+zeta)**(5./3) + (1-zeta)**(5./3))
        dd = (5./6) * ((1+zeta)**(2./3) + (1-zeta)**(2./3))
        alpha = (t - g2/(8*n)) / (CFC*n**(5./3)*d)
        return alpha,\
               -0.6 * alpha / n + g2/(CFC*n**(11./3)),\
               -alpha / d * dd,\
               -1.0 / (8 * CFC * n**(8./3)),\
               1.0 / (CFC * n**(5./3))

    def get_chi(self, alpha):
        chi = (1 - alpha) / (1 + alpha)
        return chi, -2 / (1 + alpha)**2

    def get_chi_full_deriv(self, n, zeta, g2, t):
        alpha = self.get_alpha(n, zeta, g2, t)
        chi = self.get_chi(alpha[0])
        return chi[0],\
               chi[1] * alpha[1],\
               chi[1] * alpha[2],\
               chi[1] * alpha[3],\
               chi[1] * alpha[4]

    def get_chi_desc(self, chi):
        return chi, chi**2, chi**3, 1, 2*chi, 3*chi**2

    def get_amix(self, n, zeta, x2, chi):
        zeta = np.minimum(zeta, 1-1e-8)
        zeta = np.maximum(zeta, -1+1e-8)
        phi = 0.5 * ((1+zeta)**(2./3) + (1-zeta)**(2./3))
        chip = 0.5 * (1 + np.cos(np.pi*chi**4))
        num = 1 - chip * zeta**12
        t2 = (np.pi / 3)**(1./3) / (16 * phi**2) * x2 * n**(1./3)
        den = 1 + 0.5 * t2
        f = num / den
        dfdchi = -zeta**12 / den * (-2 * np.pi * chi**3) * np.sin(np.pi * chi**4)
        dfdz = -12 * zeta**11 * chip / den
        dfdt2 = -0.5 * num / den**2
        dt2dz = -2 * t2 / phi * (1./3) * ((1+zeta)**(-1./3) + (1-zeta)**(-1./3))
        dfdz += dfdt2 * dt2dz
        dfdx2 = dfdt2 * (np.pi / 3)**(1./3) / (16 * phi**2) * n**(1./3)
        dfdn = dfdt2 * (-1./3) * t2 / n
        return f, dfdn, dfdz, dfdx2, dfdchi

    def xefc(self, nu, nd, g2u, g2o, g2d, tu, td, fu, fd,
             include_baseline=True, include_aug_sl=True,
             include_aug_nl=True):

        ldaxu = 2**(1.0/3) * LDA_FACTOR * nu**(4.0/3)
        dldaxu = 2**(1.0/3) * 4.0 / 3 * LDA_FACTOR * nu**(1.0/3)
        ldaxd = 2**(1.0/3) * LDA_FACTOR * nd**(4.0/3)
        dldaxd = 2**(1.0/3) * 4.0 / 3 * LDA_FACTOR * nd**(1.0/3)
        ldaxt = ldaxu + ldaxd
        ft = (fu * ldaxu + fd * ldaxd) / ldaxt
        dftdfu = ldaxu / ldaxt
        dftdfd = ldaxd / ldaxt
        # double check these derivatives
        dftdnu = ldaxd * (fu - fd) / ldaxt**2 * dldaxu
        dftdnd = ldaxu * (fd - fu) / ldaxt**2 * dldaxd

        g2 = g2u + g2d + 2 * g2o
        nt = nu + nd
        x2 = self.get_x2(nt/2**(1./3), g2)
        x2u = self.get_x2(nu, g2u)
        x2d = self.get_x2(nd, g2d)
        zeta = self.get_zeta(nu, nd)
        chi = self.get_chi_full_deriv(nt, zeta, g2, tu+td)
        chiu = self.get_chi_full_deriv(nu, 1, g2u, tu)
        chid = self.get_chi_full_deriv(nd, 1, g2d, td)
        c0, v0 = self.os_baseline(nu, nd, g2, type=0)
        c1, v1 = self.os_baseline(nu, nd, g2, type=1)
        amix, vmixn, vmixz, vmixx2, vmixchi = self.get_amix(nt, zeta[0], x2[0], chi[0])
        ldaxm = (ldaxu * amix, ldaxd * amix)
        N = cx.shape[0]
        vxc = [np.zeros((N,2)),
               np.zeros((N,3)),
               np.zeros((N,2)),
               np.zeros((N,2))]
        # deriv wrt n, zeta, x2, chi, F
        vtmp = [np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)]
        vtmpu = [np.zeros(N), np.zeros(N), np.zeros(N)]
        vtmpd = [np.zeros(N), np.zeros(N), np.zeros(N)]
        tot = 0

        sl0, dsl0dx2, dsl0dchi = self.sl_terms(x2[0], chi[0], gammaos, self.d0)
        sl1, dsl1dx2, dsl1dchi = self.sl_terms(x2[0], chi[0], gammaos, self.d1)
        slu, dsludx2, dsludchi = self.sl_terms(x2u[0], chiu[0], gammax, self.dx)
        sld, dslddx2, dslddchi = self.sl_terms(x2d[0], chid[0], gammax, self.d)

        y0, deriv0 = self.xef_terms(ft, self.c0)
        y1, deriv1 = self.xef_terms(ft, self.c1)
        yu, derivu = self.xef_terms(fu, self.cx)
        yd, derivd = self.xef_terms(fd, self.cx)

        tot += sl0 * c0
        tot += sl1 * c1 * (1-chi[0]**4)
        tot += ldaxm[0] * slu
        tot += ldaxm[1] * sld
        tot += y0 * c0
        tot += y1 * c1 * (1-chi[0]**4)
        tot += ldaxm[0] * yu
        tot += ldaxm[1] * yd
        # enhancment terms on c0
        vtmp[2] += c0 * dsl0dx2
        vtmp[3] += c0 * dsl0dchi
        vtmp[4] += c0 * deriv0
        # enhancment terms on c1
        tmp = c1 * (1-chi[0]**4)
        vtmp[2] += tmp * dsl1dx2
        vtmp[3] += tmp * dsl1dchi
        vtmp[4] += tmp * deriv1
        vtmp[3] += -4 * chi[0]**3 * c1 * (sl1 + y1)
        # amix derivs and exchange-like rho derivs
        tmp = ldaxu * (slu + yu) + ldaxd * (sld + yd)
        vtmp[0] += tmp * vmixn
        vtmp[1] += tmp * vmixz
        vtmp[2] += tmp * vmixx2
        vtmp[3] += tmp * vmixchi
        # exchange-like enhancment derivs
        tmp = ldaxu * amix
        vtmpu[0] += tmp * dsludx2
        vtmpu[1] += tmp * dsludchi
        vtmpu[2] += tmp * derivu
        tmp = ldaxd * amix
        vtmpd[0] += tmp * dslddx2
        vtmpd[1] += tmp * dslddchi
        vtmpd[2] += tmp * derivd
        # baseline derivs
        fill_vxc_base_os_(vxc, v0, sl0 + y0)
        fill_vxc_base_os_(vxc, v1, sl1 + y1)
        vxc[0][:,0] += dldaxu * amix * (slu + yu)
        vxc[0][:,1] += dldaxd * amix * (sld + yd)

        # put everything into vxc
        tmp = vtmp[0] + vtmp[2] * x2[1] + vtmp[3] * chi[1]
        tmp2 = vtmp[1] + vtmp[3] * chi[2] # deriv wrt zeta
        vxc[0][:,0] += tmp + tmp2 * zeta[1]
        vxc[0][:,1] += tmp + tmp2 * zeta[2]
        tmp = vtmp[2] * x2[2] + vtmp[3] * chi[3]
        vxc[1][:,0] += tmp
        vxc[1][:,1] += 2 * tmp
        vxc[1][:,2] += tmp
        tmp = vtmp[3] * chi[4]
        vxc[2][:,0] += tmp
        vxc[2][:,1] += tmp
        vxc[0][:,0] += vtmp[4] * dftdnu
        vxc[0][:,1] += vtmp[4] * dftdnd
        vxc[3][:,0] += vtmp[4] * dftdfu
        vxc[3][:,1] += vtmp[4] * dftdfd

        vxc[0][:,0] += vtmpu[0] * x2u[1] + vtmpu[1] * chiu[1]
        vxc[1][:,0] += vtmpu[0] * x2u[2] + vtmpu[1] * chiu[3]
        vxc[2][:,0] += vtmpu[1] * chiu[4]
        vxc[3][:,0] += vtmpu[2]

        vxc[0][:,1] += vtmpd[0] * x2d[1] + vtmpd[1] * chid[1]
        vxc[1][:,2] += vtmpd[0] * x2d[2] + vtmpd[1] * chid[3]
        vxc[2][:,1] += vtmpd[1] * chid[4]
        vxc[3][:,1] += vtmpd[2]

        rhou, rhod = nu, nd
        vxc[0][rhou<1e-7,0] = 0
        vxc[1][rhou<1e-7,0] = 0
        vxc[2][rhou<1e-7,0] = 0
        vxc[3][rhou<1e-7,0] = 0
        vxc[0][rhod<1e-7,1] = 0
        vxc[1][rhod<1e-7,2] = 0
        vxc[2][rhod<1e-7,1] = 0
        vxc[3][rhod<1e-7,1] = 0
        vxc[1][np.minimum(rhou,rhod)<1e-7,1] = 0
        
        return tot, vxc

    def xefc2(self, nu, nd, g2u, g2o, g2d, tu, td, exu, exd,
              include_baseline=True, include_aug_sl=True,
              include_aug_nl=True):

        ldaxu = 2**(1.0/3) * LDA_FACTOR * nu**(4.0/3)
        dldaxu = 2**(1.0/3) * 4.0 / 3 * LDA_FACTOR * nu**(1.0/3)
        ldaxd = 2**(1.0/3) * LDA_FACTOR * nd**(4.0/3)
        dldaxd = 2**(1.0/3) * 4.0 / 3 * LDA_FACTOR * nd**(1.0/3)
        ldaxt = ldaxu + ldaxd
        ft = (exu + exd) / ldaxt
        fu = exu / ldaxu
        fd = exd / ldaxd
        dftdxu = 1 / ldaxt
        dftdxd = 1 / ldaxt
        # double check these derivatives
        dftdnu = -ft / ldaxt * dldaxu
        dftdnd = -ft / ldaxt * dldaxd

        dfudnu = -fu / ldaxu * dldaxu
        dfddnd = -fd / ldaxd * dldaxd
        dfudxu = 1 / ldaxu
        dfddxd = 1 / ldaxd

        g2 = g2u + g2d + 2 * g2o
        nt = nu + nd
        x2 = self.get_x2(nt/2**(1./3), g2)
        x2u = self.get_x2(nu, g2u)
        x2d = self.get_x2(nd, g2d)
        zeta = self.get_zeta(nu, nd)
        chi = self.get_chi_full_deriv(nt, zeta, g2, tu+td)
        chiu = self.get_chi_full_deriv(nu, 1, g2u, tu)
        chid = self.get_chi_full_deriv(nd, 1, g2d, td)
        c0, v0 = self.os_baseline(nu, nd, g2, type=0)
        c1, v1 = self.os_baseline(nu, nd, g2, type=1)
        amix, vmixn, vmixz, vmixx2, vmixchi = self.get_amix(nt, zeta[0], x2[0], chi[0])
        ldaxm = (ldaxu * amix, ldaxd * amix)
        N = cx.shape[0]
        vxc = [np.zeros((N,2)),
               np.zeros((N,3)),
               np.zeros((N,2)),
               np.zeros((N,2))]
        # deriv wrt n, zeta, x2, chi, F
        vtmp = [np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)]
        vtmpu = [np.zeros(N), np.zeros(N), np.zeros(N)]
        vtmpd = [np.zeros(N), np.zeros(N), np.zeros(N)]
        tot = 0

        sl0, dsl0dx2, dsl0dchi = self.sl_terms(x2[0], chi[0], gammaos, self.d0)
        sl1, dsl1dx2, dsl1dchi = self.sl_terms(x2[0], chi[0], gammaos, self.d1)
        slu, dsludx2, dsludchi = self.sl_terms(x2u[0], chiu[0], gammax, self.dx)
        sld, dslddx2, dslddchi = self.sl_terms(x2d[0], chid[0], gammax, self.d)

        y0, deriv0 = self.xef_terms(ft, self.c0)
        y1, deriv1 = self.xef_terms(ft, self.c1)
        yu, derivu = self.xef_terms(fu, self.cx)
        yd, derivd = self.xef_terms(fd, self.cx)

        tot += sl0 * c0
        tot += sl1 * c1 * (1-chi[0]**4)
        tot += ldaxm[0] * slu
        tot += ldaxm[1] * sld
        tot += y0 * c0
        tot += y1 * c1 * (1-chi[0]**4)
        tot += ldaxm[0] * yu
        tot += ldaxm[1] * yd
        # enhancment terms on c0
        vtmp[2] += c0 * dsl0dx2
        vtmp[3] += c0 * dsl0dchi
        vtmp[4] += c0 * deriv0
        # enhancment terms on c1
        tmp = c1 * (1-chi[0]**4)
        vtmp[2] += tmp * dsl1dx2
        vtmp[3] += tmp * dsl1dchi
        vtmp[4] += tmp * deriv1
        vtmp[3] += -4 * chi[0]**3 * c1 * (sl1 + y1)
        # amix derivs and exchange-like rho derivs
        tmp = ldaxu * (slu + yu) + ldaxd * (sld + yd)
        vtmp[0] += tmp * vmixn
        vtmp[1] += tmp * vmixz
        vtmp[2] += tmp * vmixx2
        vtmp[3] += tmp * vmixchi
        # exchange-like enhancment derivs
        tmp = ldaxu * amix
        vtmpu[0] += tmp * dsludx2
        vtmpu[1] += tmp * dsludchi
        vtmpu[2] += tmp * derivu
        tmp = ldaxd * amix
        vtmpd[0] += tmp * dslddx2
        vtmpd[1] += tmp * dslddchi
        vtmpd[2] += tmp * derivd
        # baseline derivs
        fill_vxc_base_os_(vxc, v0, sl0 + y0)
        fill_vxc_base_os_(vxc, v1, sl1 + y1)
        vxc[0][:,0] += dldaxu * amix * (slu + yu)
        vxc[0][:,1] += dldaxd * amix * (sld + yd)

        # put everything into vxc
        tmp = vtmp[0] + vtmp[2] * x2[1] + vtmp[3] * chi[1]
        tmp2 = vtmp[1] + vtmp[3] * chi[2] # deriv wrt zeta
        vxc[0][:,0] += tmp + tmp2 * zeta[1]
        vxc[0][:,1] += tmp + tmp2 * zeta[2]
        tmp = vtmp[2] * x2[2] + vtmp[3] * chi[3]
        vxc[1][:,0] += tmp
        vxc[1][:,1] += 2 * tmp
        vxc[1][:,2] += tmp
        tmp = vtmp[3] * chi[4]
        vxc[2][:,0] += tmp
        vxc[2][:,1] += tmp
        vxc[0][:,0] += vtmp[4] * dftdnu
        vxc[0][:,1] += vtmp[4] * dftdnd
        vxc[3][:,0] += vtmp[4] * dftdxu
        vxc[3][:,1] += vtmp[4] * dftdxd

        vxc[0][:,0] += vtmpu[0] * x2u[1] + vtmpu[1] * chiu[1] + vtmpu[2] * dfudnu
        vxc[1][:,0] += vtmpu[0] * x2u[2] + vtmpu[1] * chiu[3]
        vxc[2][:,0] += vtmpu[1] * chiu[4]
        vxc[3][:,0] += vtmpu[2] * dfudxu

        vxc[0][:,1] += vtmpd[0] * x2d[1] + vtmpd[1] * chid[1] + vtmpd[2] * dfddnd
        vxc[1][:,2] += vtmpd[0] * x2d[2] + vtmpd[1] * chid[3]
        vxc[2][:,1] += vtmpd[1] * chid[4]
        vxc[3][:,1] += vtmpd[2] * dfddxd

        rhou, rhod = nu, nd
        vxc[0][rhou<1e-7,0] = 0
        vxc[1][rhou<1e-7,0] = 0
        vxc[2][rhou<1e-7,0] = 0
        vxc[3][rhou<1e-7,0] = 0
        vxc[0][rhod<1e-7,1] = 0
        vxc[1][rhod<1e-7,2] = 0
        vxc[2][rhod<1e-7,1] = 0
        vxc[3][rhod<1e-7,1] = 0
        vxc[1][np.minimum(rhou,rhod)<1e-7,1] = 0
        
        return tot, vxc
