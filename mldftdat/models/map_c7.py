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

    def __init__(self, css, cos, cx, cm, ca,
                 dss, dos, dx, dm, da,
                 bx=None, bss=None, bos=None,
                 fterm_scale=1.0):
        self.cx = cx
        self.css = css
        self.cos = cos
        self.cm = cm
        self.ca = ca
        self.dx = dx
        self.dss = dss
        self.dos = dos
        self.dm = dm
        self.da = da
        if bx is None:
            self.bx = [0] * 4
        else:
            self.bx = bx
        if bss is None:
            self.bss = [0] * 4
        else:
            self.bss = bss
        if bos is None:
            self.bos = [0] * 4
        else:
            self.bos = bos
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

    def get_separate_xef_terms(self, f):
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
        chi = 1 / (1 + alpha)
        return chi, -chi**2
        chi = 1 / (1 + alpha**2)
        return chi, -2 * alpha * chi**2

    def get_chi_desc(self, chi):
        chip = 2 * chi - 1
        return chip, chip**2, chip**3, 2, 4*chip, 6*chip**2
        #return np.cos(np.pi * chi), 0.5 * (1 + np.cos(2*np.pi*chi)),\
        #       -np.pi * np.sin(np.pi*chi), -np.pi * np.sin(2*np.pi*chi)

    def get_D(self, n, g2, t):
        y = 1 - g2/(8.*n*t)
        dydn = g2/(8.*n**2*t)
        dydg2 = -1/(8.*n*t)
        dydt = g2/(8.*n*t**2)
        dy = 0.5 * np.pi * np.sin(np.pi * y)
        dy = 0
        y = 0.5 * (1 - np.cos(np.pi * y))
        return y, dy * dydn, dy * dydg2, dy * dydt

    def get_amix(self, n, zeta, x2, chi):
        zeta = np.minimum(zeta, 1-1e-8)
        zeta = np.maximum(zeta, -1+1e-8)
        phi = 0.5 * ((1+zeta)**(2./3) + (1-zeta)**(2./3))
        chi = 0.5 * (1 + np.cos(np.pi*(2*chi-1)**4))
        num = 1 - chi * zeta**12
        t2 = (np.pi / 3)**(1./3) / (16 * phi**2) * x2 * n**(1./3)
        den = 1 + 0.5 * t2
        f = num / den
        dfdchi = -zeta**12 / den
        dfdz = -2 * zeta * chi / den
        dfdt2 = -0.5 * num / den**2
        dt2dz = -2 * t2 / phi * (1./3) * ((1+zeta)**(-1./3) + (1-zeta)**(-1./3))
        dfdz += dfdt2 * dt2dz
        dfdx2 = dfdt2 * (np.pi / 3)**(1./3) / (16 * phi**2) * n**(1./3)
        dfdn = dfdt2 * (-1./3) * t2 / n
        return f, dfdn, dfdz, dfdx2, dfdchi

    """
    def get_amix(self, rhou, rhod, g2, chi):
        rhot = rhou + rhod

        Do = 0.5 * (1 - np.cos(2 * np.pi * chi))

        elim, vxclim = self.baseline_inf(rhou, rhod, g2, Do)

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
    """
    """
    def get_D(self, n, g2, t):
        y = 1 - g2/(8.*n*t)
        dydn = g2/(8.*n**2*t)
        dydg2 = -1/(8.*n*t)
        dydt = g2/(8.*n*t**2)
        dy = 0.5 * np.pi * np.sin(np.pi * y)
        dy = 0
        y = 0.5 * (1 - np.cos(np.pi * y))
        return y, dy * dydn, dy * dydg2, dy * dydt

    def get_amix(self, rhou, rhod, g2, Do):
        rhot = rhou + rhod

        elim, vxclim = self.baseline_inf(rhou, rhod, g2, Do)

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
    """

    def single_corr(self, x2, z, alpha, d):
        gamma = self.gammafunc(x2, z, alpha)
        corrfunc = self.corrfunc(x2, z, gamma[0], d)
        return corrfunc[0], corrfunc[1] + corrfunc[3] * gamma[1],\
                            corrfunc[2] + corrfunc[3] * gamma[2]

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
        alpha = self.get_alpha(nt, g2, tu+td)
        alphau = self.get_alpha(nu, g2u, tu)
        alphad = self.get_alpha(nd, g2d, td)
        chi = self.get_chi(alpha[0])
        chiu = self.get_chi(alphau[0])
        chid = self.get_chi(alphad[0])
        cx, vx = self.os_baseline(nu, nd, g2, type=0)
        amix, vmixn, vmixz, vmixx2, vmixchi = self.get_amix(nt, zeta[0], x2[0], chi[0])
        ldaxm = (ldaxu * amix, ldaxd * amix)
        N = cx.shape[0]
        vxc = [np.zeros((N,2)),
               np.zeros((N,3)),
               np.zeros((N,2)),
               np.zeros((N,2))]
        tot = 0

        sl, dsldx2, dsldchi = self.sl_terms(x2[0], chi[0], gammaos, self.dx)
        slu, dsludx2, dsludchi = self.sl_terms(x2u[0], chiu[0], gammaos, self.da)
        sld, dslddx2, dslddchi = self.sl_terms(x2d[0], chid[0], gammaos, self.da)

        yx, derivx = self.xef_terms(ft, self.cx)
        yau, derivau = self.xef_terms(fu, self.ca)
        yad, derivad = self.xef_terms(fd, self.ca)

        tot += sl * cx
        tot += ldaxm[0] * slu
        tot += ldaxm[1] * sld
        tot += yx * cx
        tot += ldaxm[0] * yau
        tot += ldaxm[1] * yad
        fill_vxc_os2_(vxc, cx, dsldx2, dsldchi, derivx,
                      dftdfu, dftdfd, dftdnu, dftdnd)
        fill_vxc_ss2_(vxc, 0, cx, dsludx2, dsludchi, derivau)
        fill_vxc_ss2_(vxc, 1, cx, dslddx2, dslddchi, derivad)
        fill_vxc_base_os_(vxc, vx, sl + yx)
        vxc[0][:,0] += dldaxu * amix * (slu + yau)
        vxc[0][:,1] += dldaxd * amix * (sld + yad)
        fill_vxc_os2_(vxc, ldaxu * (slu + yau) + ldaxd * (slud + yad),
                      vmixx2, vmixchi)
        # need to take n-derivs for amix

        if include_aug_sl:
            x2u = self.get_x2(nu, g2u)
            x2d = self.get_x2(nd, g2d)
            x2 = self.get_x2((nu+nd)/2**(1.0/3), g2)
            zu = self.get_z(nu, tu)
            zd = self.get_z(nd, td)
            z = self.get_z((nu+nd)/2**(2.0/3), tu+td)
            z[1][:] /= 2**(2.0/3)
            x2[1][:] /= 2**(1.0/3)
            cfo = self.single_corr(x2[0], z[0], alphaos, self.dos)
            cfx = self.single_corr(x2[0], z[0], alphaos, self.dx)
            cfau = self.single_corr(x2u[0], zu[0], alphax, self.da)
            cfad = self.single_corr(x2d[0], zd[0], alphax, self.da)

            tot += cfo[0] * co * Do[0]
            tot += cfx[0] * cx
            tot += ldaxm[0] * cfau[0]
            tot += ldaxm[1] * cfad[0]

            tmp = co * cfo[0]
            fill_vxc_base_os_(vxc, vo, cfo[0] * Do[0])
            fill_vxc_base_os_(vxc, vx, cfx[0])
            fill_vxc_os_(vxc, tmp * Do[1],
                         tmp * Do[2],
                         tmp * Do[3])

            vxc[0][:,0] += dldaxu * amix * cfau[0]
            vxc[0][:,1] += dldaxd * amix * cfad[0]
            tmp = ldaxu * cfau[0] + ldaxd * cfad[0]
            fill_vxc_base_os_(vxc, vxcmix, tmp)
            
            for c, cf in zip([co * Do[0], cx],
                             [cfo, cfx]):
                fill_vxc_os_(vxc, c * (cf[1] * x2[1] + cf[2] * z[1]),
                             c * (cf[1] * x2[2]),
                             c * (cf[2] * z[2]))
            fill_vxc_ss_(vxc, 0, ldaxm[0] * (cfau[1] * x2u[1] + cfau[2] * zu[1]),
                         ldaxm[0] * cfau[1] * x2u[2],
                         ldaxm[0] * cfau[2] * zu[2])
            fill_vxc_ss_(vxc, 1, ldaxm[1] * (cfad[1] * x2d[1] + cfad[2] * zd[1]),
                         ldaxm[1] * cfad[1] * x2d[2],
                         ldaxm[1] * cfad[2] * zd[2])
            
        if include_aug_nl:
            

            tot += yo * co * Do[0]
            tot += yx * cx
            tot += yau * ldaxm[0]
            tot += yad * ldaxm[1]

            tmp = yo * co
            fill_vxc_base_os_(vxc, vo, yo * Do[0])
            fill_vxc_base_os_(vxc, vx, yx)
            fill_vxc_os_(vxc, tmp * Do[1],
                         tmp * Do[2],
                         tmp * Do[3])

            vxc[0][:,0] += dldaxu * amix * yau
            vxc[0][:,1] += dldaxd * amix * yad
            tmp = ldaxu * yau + ldaxd * yad
            fill_vxc_base_os_(vxc, vxcmix, tmp)

            tmp = co * Do[0] * derivo + cx * derivx
            vxc[0][:,0] += tmp * dftdnu
            vxc[0][:,1] += tmp * dftdnd
            vxc[3][:,0] += tmp * dftdfu + ldaxm[0] * derivau
            vxc[3][:,1] += tmp * dftdfd + ldaxm[1] * derivad

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

        nt = nu + nd
        ldaxu = 2**(1.0/3) * LDA_FACTOR * nu**(4.0/3)
        dldaxu = 2**(1.0/3) * 4.0 / 3 * LDA_FACTOR * nu**(1.0/3)
        ldaxd = 2**(1.0/3) * LDA_FACTOR * nd**(4.0/3)
        dldaxd = 2**(1.0/3) * 4.0 / 3 * LDA_FACTOR * nd**(1.0/3)
        ldaxt = ldaxu + ldaxd
        fu = exu / ldaxu
        fd = exd / ldaxd
        ft = (exu + exd) / ldaxt
        dfudxu = 1 / ldaxu
        dfudnu = -4 * fu / (3 * nu)
        dfddxd = 1 / ldaxd
        dfddnd = -4 * fd / (3 * nd)
        dftdxu = 1 / ldaxt
        dftdxd = 1 / ldaxt
        # double check these
        dftdnu = -ft / ldaxt * dldaxu
        dftdnd = -ft / ldaxt * dldaxd

        g2 = g2u + g2d + 2 * g2o
        co, vo = self.os_baseline(nu, nd, g2, type=1)
        cx, vx = self.os_baseline(nu, nd, g2, type=0)
        co *= nt
        cx *= nt
        Do = self.get_D(nu+nd, g2, tu+td)
        amix, vxcmix = self.get_amix(nu, nd, g2, Do)
        ldaxm = (ldaxu * amix, ldaxd * amix)
        N = co.shape[0]
        vxc = [np.zeros((N,2)),
               np.zeros((N,3)),
               np.zeros((N,2)),
               np.zeros((N,2))]
        if include_baseline:
            tot = co * Do[0] + cx * (1 - Do[0])
            tmp = (co - cx)
            fill_vxc_os_(vxc, tmp * Do[1],
                         tmp * Do[2],
                         tmp * Do[3])
            fill_vxc_base_os_(vxc, vo, Do[0])
            fill_vxc_base_os_(vxc, vx, (1-Do[0]))
        else:
            tot = 0

        if include_aug_sl:
            x2u = self.get_x2(nu, g2u)
            x2d = self.get_x2(nd, g2d)
            x2 = self.get_x2((nu+nd)/2**(1.0/3), g2)
            zu = self.get_z(nu, tu)
            zd = self.get_z(nd, td)
            z = self.get_z((nu+nd)/2**(2.0/3), tu+td)
            z[1][:] /= 2**(2.0/3)
            x2[1][:] /= 2**(1.0/3)
            cfo = self.single_corr(x2[0], z[0], alphaos, self.dos)
            cfx = self.single_corr(x2[0], z[0], alphaos, self.dx)
            cfau = self.single_corr(x2u[0], zu[0], alphax, self.da)
            cfad = self.single_corr(x2d[0], zd[0], alphax, self.da)

            tot += cfo[0] * co * Do[0]
            tot += cfx[0] * cx
            tot += ldaxm[0] * cfau[0]
            tot += ldaxm[1] * cfad[0]

            tmp = co * cfo[0]
            fill_vxc_base_os_(vxc, vo, cfo[0] * Do[0])
            fill_vxc_base_os_(vxc, vx, cfx[0])
            fill_vxc_os_(vxc, tmp * Do[1],
                         tmp * Do[2],
                         tmp * Do[3])

            vxc[0][:,0] += dldaxu * amix * cfau[0]
            vxc[0][:,1] += dldaxd * amix * cfad[0]
            tmp = ldaxu * cfau[0] + ldaxd * cfad[0]
            fill_vxc_base_os_(vxc, vxcmix, tmp)
            
            for c, cf in zip([co * Do[0], cx],
                             [cfo, cfx]):
                fill_vxc_os_(vxc, c * (cf[1] * x2[1] + cf[2] * z[1]),
                             c * (cf[1] * x2[2]),
                             c * (cf[2] * z[2]))
            fill_vxc_ss_(vxc, 0, ldaxm[0] * (cfau[1] * x2u[1] + cfau[2] * zu[1]),
                         ldaxm[0] * cfau[1] * x2u[2],
                         ldaxm[0] * cfau[2] * zu[2])
            fill_vxc_ss_(vxc, 1, ldaxm[1] * (cfad[1] * x2d[1] + cfad[2] * zd[1]),
                         ldaxm[1] * cfad[1] * x2d[2],
                         ldaxm[1] * cfad[2] * zd[2])
            
        if include_aug_nl:
            yo, derivo = self.xef_terms(ft, self.cos)
            yx, derivx = self.xef_terms(ft, self.cx)
            yau, derivau = self.xef_terms(fu, self.ca)
            yad, derivad = self.xef_terms(fd, self.ca)

            tot += yo * co * Do[0]
            tot += yx * cx
            tot += yau * ldaxm[0]
            tot += yad * ldaxm[1]

            tmp = yo * co
            fill_vxc_base_os_(vxc, vo, yo * Do[0])
            fill_vxc_base_os_(vxc, vx, yx)
            fill_vxc_os_(vxc, tmp * Do[1],
                         tmp * Do[2],
                         tmp * Do[3])

            vxc[0][:,0] += dldaxu * amix * yau
            vxc[0][:,1] += dldaxd * amix * yad
            tmp = ldaxu * yau + ldaxd * yad
            fill_vxc_base_os_(vxc, vxcmix, tmp)

            tmp = co * Do[0] * derivo + cx * derivx
            vxc[0][:,0] += tmp * dftdnu + ldaxm[0] * derivau * dfudnu
            vxc[0][:,1] += tmp * dftdnd + ldaxm[1] * derivad * dfddnd
            vxc[3][:,0] += tmp * dftdxu + ldaxm[0] * derivau * dfudxu
            vxc[3][:,1] += tmp * dftdxd + ldaxm[1] * derivad * dfddxd

        thr = 1e-6
        rhou, rhod = nu, nd
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
