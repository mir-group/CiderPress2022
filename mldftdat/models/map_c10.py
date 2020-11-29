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

    def __init__(self, d, c, dx, cx,
                 fterm_scale=1.0):
        self.cx = cx
        self.dx = dx
        self.c = c
        self.d = d
        self.cf = fterm_scale

    def get_rs(self, n):
        rs = (4 * np.pi * n / 3)**(-1.0/3)
        return rs, -rs / (3 * n)

    def get_zeta(self, nu, nd):
        zeta = (-nd + nu)/(nd + nu)
        dzetau = (2*nd)/((nd + nu)**2)
        dzetad = (-2*nu)/((nd + nu)**2)
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
        zeta = np.minimum(zeta, 0.9999)
        zeta = np.maximum(zeta, -0.9999)
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

    def baseline_inf(self, zeta, x2, chi):
        a = 17. / 3
        D = a * (1 - chi) / (a - chi)
        dDdchi = (a - a**2) / (a - chi)**2
        s2 = x2 / sprefac**2
        e0lim, de0dzeta, de0ds2 = self.baseline0inf(zeta, s2)
        e1lim, de1dzeta, de1ds2 = self.baseline1inf(zeta, s2)
        elim = e1lim * D + e0lim * (1 - D)
        dedzeta = de1dzeta * D + de0dzeta * (1 - D)
        deds2 = de1ds2 * D + de0ds2 * (1 - D)
        dedchi = (e1lim - e0lim) * dDdchi
        return elim, dedzeta, deds2 / sprefac**2, dedchi
    
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
        if return_deriv:
            dres = np.array([i * f**(i-1) - 2 * self.cf * f**(i+1) for i in range(5)])
            dres *= fterm0
            return res, dres
        return res

    def get_separate_sl_terms(self, x2, chi, gamma):
        desc = np.zeros((15, x2.shape[0]))
        dx2 = np.zeros((15, x2.shape[0]))
        dchi = np.zeros((15, x2.shape[0]))
        u = gamma * x2 / (1 + gamma * x2)
        du = gamma / (1 + gamma * x2)**2
        a0, a1, a2, a3, da0, da1, da2, da3 = self.get_chi_desc(chi)
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

    def get_separate_xefa_terms(self, F, chi):
        x, dx = self.get_separate_xef_terms(F, return_deriv=True)
        x = x[1:4]
        dx = dx[1:4]
        desc = np.zeros((13, F.shape[0]))
        df = np.zeros((13, F.shape[0]))
        dchi = np.zeros((13, F.shape[0]))
        a0, a1, a2, a3, da0, da1, da2, da3 = self.get_chi_desc(chi)
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

    def sl_terms(self, x2, chi, gamma, c):
        y, dydx2, dydchi = self.get_separate_sl_terms(x2, chi, gamma)
        return np.dot(c, y), np.dot(c, dydx2), np.dot(c, dydchi)

    def nl_terms(self, F, chi, c):
        y, dydx2, dydchi = self.get_separate_xefa_terms(F, chi)
        return np.dot(c, y), np.dot(c, dydx2), np.dot(c, dydchi)

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
        return g2/(n**2.6666666666666665+1e-16),\
               (-8*g2)/(3.*n**3.6666666666666665+1e-16),\
               1/(n**(2.6666666666666665)+1e-16)

    def get_s2(self, n, g2):
        a, b, c = self.get_x2(n, g2)
        return a / sprefac**2 + 1e-10, b / sprefac**2, c / sprefac**2

    def get_alpha(self, n, zeta, g2, t):
        d = 0.5 * ((1+zeta)**(5./3) + (1-zeta)**(5./3))
        dd = (5./6) * ((1+zeta)**(2./3) - (1-zeta)**(2./3))
        alpha = (t - g2/(8*n)) / (CFC*n**(5./3)*d)
        alphap = np.append(alpha[n>1e-8], [0])
        return alpha,\
               ((-5 * t + g2 / n) / n) / (3 * CFC * d * n**(5./3)),\
               -alpha / d * dd,\
               -1.0 / (8 * CFC * n**(8./3) * d),\
               1.0 / (CFC * n**(5./3) * d)
#-(5.0/3) * alpha / n + (g2 / (8*n**2)) / (CFC * n**(5./3) * d),\
#(-5 * t + g2 / n) / (3 * CFC * d * n**(8./3)),\

    def get_chi(self, alpha):
        chi = (1 - alpha) / (1 + alpha)
        return chi, -2 / (1 + alpha)**2

    def get_chi_full_deriv(self, n, zeta, g2, t):
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

    def get_chi_desc(self, chi):
        return chi, chi**2, chi**3, chi**4, np.ones_like(chi), 2*chi, 3*chi**2, 4*chi**3

    def get_chi_desc2(self, chi):
        y = np.zeros((6, chi.shape[0]))
        dy = np.zeros((6, chi.shape[0]))
        for i in range(6):
            y[i] = chi**(i+3) - chi**(i+1)
            dy[i] = (i+3) * chi**(i+2) - (i+1) * chi**(i)
        y = np.append([chi, chi**2, chi**2-chi], y, axis=0)
        dy = np.append([np.ones_like(chi), 2*chi, 2*chi - 1], y, axis=0)
        return y, dy

    def get_chi_desc3(self, chi):
        y = np.zeros((6, chi.shape[0]))
        dy = np.zeros((6, chi.shape[0]))
        for i in range(6):
            y[i] = chi**(i+3) - chi**(i+1)
            dy[i] = (i+3) * chi**(i+2) - (i+1) * chi**(i)
        y = np.append([chi, chi**2], y, axis=0)
        dy = np.append([np.ones_like(chi), chi**2], y, axis=0)
        return y, dy

    def get_amix(self, n, zeta, x2, chi):
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

    def get_amix2(self, n, zeta, x2, chi):
        elim, dedzeta, dedx2, dedchi = self.baseline_inf(zeta, x2, chi)

        phi = ((1+zeta)**(4./3) + (1-zeta)**(4./3)) / 2
        dphi = (2./3) * ((1+zeta)**(1./3) - (1-zeta)**(1./3))
        exlda = LDA_FACTOR * n**(1.0/3) * phi
        dinvldan = -1 / (3 * n * exlda)
        dinvldaz = -dphi / (phi * exlda)

        u = elim / exlda
        amix = 1 - 1/(1 + A*np.log(1 + B*u))
        damix = (A*B)/((1 + B*u)*(1 + A*np.log(1 + B*u))**2)

        dadn = damix * elim * dinvldan
        dadz = damix * (elim * dinvldaz + dedzeta / exlda)
        dadx2 = damix * dedx2 / exlda
        dadchi = damix * dedchi / exlda

        return amix, dadn, dadz, dadx2, dadchi

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
        x2 = self.get_x2(nt, g2)
        x2u = self.get_x2(nu, g2u)
        x2d = self.get_x2(nd, g2d)
        zeta = self.get_zeta(nu, nd)
        
        chi = self.get_chi_full_deriv(nt, zeta[0], g2, tu+td)
        chiu = self.get_chi_full_deriv(nu, 1, g2u, tu)
        chid = self.get_chi_full_deriv(nd, 1, g2d, td)
        chiu[0][np.isnan(chiu[0])] = 0
        chid[0][np.isnan(chid[0])] = 0
        
        c0, v0 = self.os_baseline(nu, nd, g2, type=0)
        c1, v1 = self.os_baseline(nu, nd, g2, type=1)
        c0 *= nt
        c1 *= nt
        amix, vmixn, vmixz, vmixx2, vmixchi = self.get_amix(nt, zeta[0], x2[0], chi[0])
        ldaxm = (ldaxu * amix, ldaxd * amix)
        N = nt.shape[0]
        vxc = [np.zeros((N,2)),
               np.zeros((N,3)),
               np.zeros((N,2)),
               np.zeros((N,2))]
        # deriv wrt n, zeta, x2, chi, F
        vtmp = [np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)]
        vtmpu = [np.zeros(N), np.zeros(N), np.zeros(N)]
        vtmpd = [np.zeros(N), np.zeros(N), np.zeros(N)]
        tot = 0

        gammax = 0.004
        achi, dachi = self.get_chi_desc2(chi[0])
        A = 17.0 / 3
        slc = A * (1 - chi[0]) / (A - chi[0]) - np.dot(self.d, achi[2:])
        dslc = (A - A**2) / (A - chi[0])**2 - np.dot(self.d, dachi[2:])
        slc0 = np.dot(self.c, achi[[0,1,3,4,5,6,7,8]])
        dslc0 = np.dot(self.c, dachi[[0,1,3,4,5,6,7,8]])
        slu, dsludx2, dsludchi = self.sl_terms(x2u[0], chiu[0], gammax, self.dx)
        sld, dslddx2, dslddchi = self.sl_terms(x2d[0], chid[0], gammax, self.dx)
        nlu, dnludf, dnludchi = self.nl_terms(fu, chiu[0], self.cx)
        nld, dnlddf, dnlddchi = self.nl_terms(fd, chid[0], self.cx)

        tot += c1 * slc + c0 * (1 - slc)
        tot += c0 * slc0
        tot += ldaxm[0] * slu
        tot += ldaxm[1] * sld
        tot += ldaxm[0] * nlu
        tot += ldaxm[1] * nld
        # enhancment terms on c1 and c0
        vtmp[3] += (c1 - c0) * dslc
        vtmp[3] += c0 * dslc0
        
        # amix derivs and exchange-like rho derivs
        tmp = ldaxu * (slu + nlu) + ldaxd * (sld + nld)
        cond = nt>1e-4
        vtmp[0][cond] += (tmp * vmixn)[cond]
        vtmp[1][cond] += (tmp * vmixz)[cond]
        vtmp[2][cond] += (tmp * vmixx2)[cond]
        vtmp[3][cond] += (tmp * vmixchi)[cond]
        # exchange-like enhancment derivs
        tmp = ldaxu * amix
        vtmpu[0] += tmp * dsludx2
        vtmpu[1] += tmp * (dsludchi + dnludchi)
        vtmpu[2] += tmp * dnludf
        tmp = ldaxd * amix
        vtmpd[0] += tmp * dslddx2
        vtmpd[1] += tmp * (dslddchi + dnlddchi)
        vtmpd[2] += tmp * dnlddf
        # baseline derivs
        fill_vxc_base_os_(vxc, v0, 1-slc + slc0)
        fill_vxc_base_os_(vxc, v1, slc)
        vxc[0][:,0] += dldaxu * amix * (slu + nlu)
        vxc[0][:,1] += dldaxd * amix * (sld + nld)

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

        thr = 1e-6
        rhou, rhod = nu, nd
        tot[(rhou+rhod)<thr] = 0
        vxc[0][rhou<thr,0] = 0
        vxc[1][rhou<thr,0] = 0
        vxc[2][rhou<thr,0] = 0
        vxc[3][rhou<thr,0] = 0
        vxc[0][rhod<thr,1] = 0
        vxc[1][rhod<thr,2] = 0
        vxc[2][rhod<thr,1] = 0
        vxc[3][rhod<thr,1] = 0
        vxc[1][np.sqrt(rhou*rhod)<thr,1] = 0
        
        return tot, vxc

    def xefc2(self, nu, nd, g2u, g2o, g2d, tu, td, exu, exd,
              include_baseline=True, include_aug_sl=True,
              include_aug_nl=True):

        ldaxu = 2**(1.0/3) * LDA_FACTOR * nu**(4.0/3) + 1e-16
        dldaxu = 2**(1.0/3) * 4.0 / 3 * LDA_FACTOR * nu**(1.0/3)
        ldaxd = 2**(1.0/3) * LDA_FACTOR * nd**(4.0/3) + 1e-16
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
        x2 = self.get_x2(nt, g2)
        x2u = self.get_x2(nu, g2u)
        x2d = self.get_x2(nd, g2d)
        zeta = self.get_zeta(nu, nd)
        
        chi = self.get_chi_full_deriv(nt, zeta[0], g2, tu+td)
        chiu = self.get_chi_full_deriv(nu, 1, g2u, tu)
        chid = self.get_chi_full_deriv(nd, 1, g2d, td)
        chiu[0][np.isnan(chiu[0])] = 0
        chid[0][np.isnan(chid[0])] = 0
        
        c0, v0 = self.os_baseline(nu, nd, g2, type=0)
        c1, v1 = self.os_baseline(nu, nd, g2, type=1)
        c0 *= nt
        c1 *= nt
        amix, vmixn, vmixz, vmixx2, vmixchi = self.get_amix(nt, zeta[0], x2[0], chi[0])
        ldaxm = (ldaxu * amix, ldaxd * amix)
        N = nt.shape[0]
        vxc = [np.zeros((N,2)),
               np.zeros((N,3)),
               np.zeros((N,2)),
               np.zeros((N,2))]
        # deriv wrt n, zeta, x2, chi, F
        vtmp = [np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)]
        vtmpu = [np.zeros(N), np.zeros(N), np.zeros(N)]
        vtmpd = [np.zeros(N), np.zeros(N), np.zeros(N)]
        tot = 0

        gammax = 0.004
        achi, dachi = self.get_chi_desc2(chi[0])
        A = 17.0 / 3
        slc = A * (1 - chi[0]) / (A - chi[0]) - np.dot(self.d, achi[2:])
        dslc = (A - A**2) / (A - chi[0])**2 - np.dot(self.d, dachi[2:])
        slc0 = np.dot(self.c, achi[[0,1,3,4,5,6,7,8]])
        dslc0 = np.dot(self.c, dachi[[0,1,3,4,5,6,7,8]])
        slu, dsludx2, dsludchi = self.sl_terms(x2u[0], chiu[0], gammax, self.dx)
        sld, dslddx2, dslddchi = self.sl_terms(x2d[0], chid[0], gammax, self.dx)
        nlu, dnludf, dnludchi = self.nl_terms(fu, chiu[0], self.cx)
        nld, dnlddf, dnlddchi = self.nl_terms(fd, chid[0], self.cx)

        tot += c1 * slc + c0 * (1 - slc)
        tot += c0 * slc0
        tot += ldaxm[0] * slu
        tot += ldaxm[1] * sld
        tot += ldaxm[0] * nlu
        tot += ldaxm[1] * nld
        # enhancment terms on c1 and c0
        vtmp[3] += (c1 - c0) * dslc
        vtmp[3] += c0 * dslc0
        
        # amix derivs and exchange-like rho derivs
        tmp = ldaxu * (slu + nlu) + ldaxd * (sld + nld)
        cond = nt>1e-4
        vtmp[0][cond] += (tmp * vmixn)[cond]
        vtmp[1][cond] += (tmp * vmixz)[cond]
        vtmp[2][cond] += (tmp * vmixx2)[cond]
        vtmp[3][cond] += (tmp * vmixchi)[cond]
        # exchange-like enhancment derivs
        tmp = ldaxu * amix
        vtmpu[0] += tmp * dsludx2
        vtmpu[1] += tmp * (dsludchi + dnludchi)
        vtmpu[2] += tmp * dnludf
        tmp = ldaxd * amix
        vtmpd[0] += tmp * dslddx2
        vtmpd[1] += tmp * (dslddchi + dnlddchi)
        vtmpd[2] += tmp * dnlddf
        # baseline derivs
        fill_vxc_base_os_(vxc, v0, 1 - slc + slc0)
        fill_vxc_base_os_(vxc, v1, slc)
        vxc[0][:,0] += dldaxu * amix * (slu + nlu)
        vxc[0][:,1] += dldaxd * amix * (sld + nld)

        #vtmp[3] *= 0
        #vtmpu[1] *= 0
        #vtmpd[1] *= 0
        #vtmp[1] *= 0
        #vtmp[0] *= 0
        #vtmp[2] *= 0
        #vtmp[4] *= 0
        #vtmpu[2] *= 0
        #vtmpu[0] *= 0
        #vtmpd[0] *= 0
        #vtmpd[2] *= 0

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

        thr = 1e-6
        rhou, rhod = nu, nd
        tot[(rhou+rhod)<thr] = 0
        vxc[0][rhou<thr,0] = 0
        vxc[1][rhou<thr,0] = 0
        vxc[2][rhou<thr,0] = 0
        vxc[3][rhou<thr,0] = 0
        vxc[0][rhod<thr,1] = 0
        vxc[1][rhod<thr,2] = 0
        vxc[2][rhod<thr,1] = 0
        vxc[3][rhod<thr,1] = 0
        vxc[1][np.sqrt(rhou*rhod)<thr,1] = 0
        #print("NANS", np.isnan(vxc[0]).any(), np.isnan(vxc[1]).any(), np.isnan(vxc[2]).any(), np.isnan(vxc[3]).any())

        return tot, vxc
