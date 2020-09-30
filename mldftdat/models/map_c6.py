# Autocode from mathematica for VSXC-type contribs
import numpy as np
from pyscf.dft.libxc import eval_xc

LDA_FACTOR = - 3.0 / 4.0 * (3.0 / np.pi)**(1.0/3)
alphax = 0.001867
alphass, alphaos = 0.00515088, 0.00304966
CF = 0.3 * (6 * np.pi**2)**(2.0/3)

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
                 bx=None, bss=None, bos=None):
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
        #print(len(self.dss), len(self.dos))

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
        fill_vxc_os_(vxc, tmp * D[1],
                     tmp * D[2],
                     tmp * D[3])
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
            print(zeta.shape)
            lda, dldadrs, dldadzeta = self.pw92(rs, zeta)
            e, dedlda, dedrs, dedzeta, deds2 = self.baseline1(lda, rs, zeta, s2)
            dedrs += dedlda * dldadrs
            print(dedzeta.shape)
            print(dldadzeta.shape)
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

    def gammafunc(self, x2, z, alpha):
        y = 1 + alpha * (x2 + z)
        dydx2 = alpha
        dydz = alpha
        return y, dydx2, dydz

    def get_separate_corrfunc_terms(self, x2, z, gamma):
        return np.array([-1 + 1/gamma,
                         x2 / gamma**2,
                         z / gamma**2,
                         x2**2 / gamma**3,
                         x2*z / gamma**3,
                         z**2 / gamma**3])

    def corrfunc(self, x2, z, gamma, d):
        #print(d)
        d0, d1, d2, d3, d4, d5 = d
        # NOTE: 0 in HEG limit
        y = d0*(-1 + 1/gamma) + (d1*x2 + d2*z)/gamma**2 + (d3*x2**2 + d4*x2*z + d5*z**2)/gamma**3
        dydx2 = (d1*gamma + 2*d3*x2 + d4*z)/gamma**3
        dydz = (d2*gamma + d4*x2 + 2*d5*z)/gamma**3
        dydgamma = -((d0*gamma**2 + 3*d3*x2**2 + 2*gamma*(d1*x2 + d2*z) + 3*z*(d4*x2 + d5*z))/gamma**4)
        return y, dydx2, dydz, dydgamma

    def get_separate_xef_terms(self, f):
        fterm0 = np.exp(-2 * f**2)
        return np.array([fterm0 * f**i for i in range(5)])

    def grad_terms(self, x2, gamma, c):
        y = 0
        dy = 0
        u = gamma * x2 / (1 + gamma * x2)
        du = gamma / (1 + gamma * x2)**2
        for i in range(4):
            y += c[i] * u**(i+1)
            dy += c[i] * (i+1) * u**i
        return y, dy * du

    def xef_terms(self, f, c):
        y = 0
        d = 0
        fterm0 = np.exp(-2 * f**2)
        for i in range(4):
            y += c[i] * f**(i+1)
            d += c[i] * ((i+1) * f**i - 4 * f**(i+2))
        return y * fterm0, d * fterm0

    def get_x2(self, n, g2):
        return g2/n**2.6666666666666665,\
               (-8*g2)/(3.*n**3.6666666666666665),\
               n**(-2.6666666666666665)

    def get_s2(self, n, g2):
        a, b, c = self.get_x2(n, g2)
        return a / sprefac**2, b / sprefac**2, c / sprefac**2

    def get_z(self, n, t):
        return -2*CF + (2*t)/n**1.6666666666666667,\
               (-10*t)/(3.*n**2.6666666666666665),\
               2/n**1.6666666666666667

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

    def xefc(self, nu, nd, g2u, g2o, g2d, tu, td, fu, fd):

        g2 = g2u + g2d + 2 * g2o
        nt = nu + nd
        co, vo = self.os_baseline(nu, nd, g2, type=1)
        cx, vx = self.os_baseline(nu, nd, g2, type=0)
        co *= nt
        cx *= nt
        Do = self.get_D(nu+nd, g2, tu+td)
        tot = co + (cx - co) * (1 - Do[0])
        N = cu.shape[0]
        vxc = [np.zeros((N,2)),
               np.zeros((N,3)),
               np.zeros((N,2)),
               np.zeros((N,2))]
        tmp = (co - cx)
        fill_vxc_os_(vxc, tmp * Do[1],
                     tmp * do[2],
                     tmp * Do[3])
        fill_vxc_base_os_(vxc, vo, Do[0])
        fill_vxc_base_os_(vxc, vx, (1-Do[0]))

        """
        cfo = self.single_corr(x2[0], z[0], alphaos, self.dos)
        cfx = self.single_corr(x2[0], z[0], alphaos, self.dx)
        cfm = self.single_corr(x2[0], z[0], alphaos, self.dm)

        for c, cf in zip([co, cx, (cx-co) * (1-Do[0])], [cfo, cfx, cfm]):
            fill_vxc_os_(vxc, c * (cf[1] * x2[1] + cf[2] * z[1]),
                         c * (cf[1] * x2[2]),
                         c * (xf[2] * z[2]))
        """

        return tot, vxc