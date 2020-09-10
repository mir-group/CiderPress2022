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
b2c = 0.00889
b3c = 0.125541
gamma = 0.031091

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
        zeta = (-nd + nu)/(nd + nu)
        dzetau = (2*nd)/(nd + nu)**2
        dzetad = (-2*nu)/(nd + nu)**2
        return zeta, dzetau, dzetad

    def get_phi0(self, zeta):
        ###G = (1 - zeta**12)*(1 - 2.3621*(-1 + ((1 - zeta)**1.3333333333333333 + (1 + zeta)**1.3333333333333333)/2.))
        ###dG = -1.18105*(1 - zeta**12)*((-4*(1 - zeta)**0.3333333333333333)/3. + (4*(1 + zeta)**0.3333333333333333)/3.) - 12*zeta**11*(1 - 2.3621*(-1 + ((1 - zeta)**1.3333333333333333 + (1 + zeta)**1.3333333333333333)/2.))
        #G = (1 - zeta**12)*(1 - 2.3631*(-1 + ((1 - zeta)**1.3333333333333333 + (1 + zeta)**1.3333333333333333)/2.))
        #dG = -1.18155*(1 - zeta**12)*((-4*(1 - zeta)**0.3333333333333333)/3. + (4*(1 + zeta)**0.3333333333333333)/3.) - 12*zeta**11*(1 - 2.3631*(-1 + ((1 - zeta)**1.3333333333333333 + (1 + zeta)**1.3333333333333333)/2.))
        G = (1 - zeta**12)*(1 - 2.363*(-1 + ((1 - zeta)**1.3333333333333333 + (1 + zeta)**1.3333333333333333)/2.))
        dG = -1.1815*(1 - zeta**12)*((-4*(1 - zeta)**0.3333333333333333)/3. + (4*(1 + zeta)**0.3333333333333333)/3.) - 12*zeta**11*(1 - 2.363*(-1 + ((1 - zeta)**1.3333333333333333 + (1 + zeta)**1.3333333333333333)/2.))
        return G, dG

    def get_phi1(self, zeta):
        phi = ((1 - zeta)**0.6666666666666666 + (1 + zeta)**0.6666666666666666)/2.
        dphi = (-2/(3.*(1 - zeta + 1e-30)**0.3333333333333333) + 2/(3.*(1 + zeta + 1e-30)**0.3333333333333333))/2.
        return phi, dphi

    def baseline0inf(self, zeta, s2):
        G, dG = self.get_phi0(zeta)
        elim = b1c*G*np.log(1 + (1 - np.e)/(np.e*(1 + 4*chiinf*s2)**0.25))
        dedG = b1c*np.log(1 + (1 - np.e)/(np.e*(1 + 4*chiinf*s2)**0.25))
        deds2 = -((b1c*chiinf*(1 - np.e)*G)/(np.e*(1 + 4*chiinf*s2)**1.25*(1 + (1 - np.e)/(np.e*(1 + 4*chiinf*s2)**0.25))))
        return elim, dedG * dG, deds2

    def baseline1inf(self, zeta, s2):
        phi, dphi = self.get_phi1(zeta)
        elim = gamma*phi**3*np.log(1. - (1 + (4*chi*s2)/phi**2)**(-0.25))
        dedphi = (-2*chi*gamma*s2)/((1 + (4*chi*s2)/phi**2)**1.25*(1. - (1 + (4*chi*s2)/phi**2)**(-0.25))) + 3*gamma*phi**2*np.log(1. - (1 + (4*chi*s2)/phi**2)**(-0.25))
        deds2 = (chi*gamma*phi)/((1 + (4*chi*s2)/phi**2)**1.25*(1. - (1 + (4*chi*s2)/phi**2)**(-0.25)))
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

    def baseline1(self, lda, rs, zeta, s2):
        Pi = np.pi
        Log = np.Log
        Exp = np.exp
        phi, dphi = self.get_phi1(zeta)
        beta = (0.066725*(1 + 0.1*rs))/(1 + 0.1778*rs)
        dbetadrs = (-0.011863705000000002*(1 + 0.1*rs))/(1 + 0.1778*rs)**2 + 0.006672500000000001/(1 + 0.1778*rs)
        w1 = -1 + E**(-lda/(gamma*phi**3))
        dw1dlda = -(1/(E**(lda/(gamma*phi**3))*gamma*phi**3))
        dw1dphi = (3*lda)/(E**(lda/(gamma*phi**3))*gamma*phi**4)
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
        dedphi += dedt2 * dt2dphi + dedw1 * dw1dbeta

        #phi, dphi = self.get_phi1(zeta)
        #EC = lda + gamma*phi**3*np.log(1 + (-1 + np.exp(-lda/(gamma*phi**3)))*(1 - (1 + (0.40229927701638585*(1 + 0.1*rs)*s2)/((-1 + np.exp(-lda/(gamma*phi**3)))*gamma*phi**2*(1 + 0.1778*rs)*rs))**(-0.25)))
        #dECdlda = 1 + (gamma*phi**3*((0.10057481925409646*(1 + 0.1*rs)*s2)/(np.exp(lda/(gamma*phi**3))*(-1 + np.exp(-lda/(gamma*phi**3)))*gamma**2*phi**5*(1 + 0.1778*rs)*rs*(1 + (0.40229927701638585*(1 + 0.1*rs)*s2)/((-1 + np.exp(-lda/(gamma*phi**3)))*gamma*phi**2*(1 + 0.1778*rs)*rs))**1.25) - (1 - (1 + (0.40229927701638585*(1 + 0.1*rs)*s2)/((-1 + np.exp(-lda/(gamma*phi**3)))*gamma*phi**2*(1 + 0.1778*rs)*rs))**(-0.25))/(np.exp(lda/(gamma*phi**3))*gamma*phi**3)))/(1 + (-1 + np.exp(-lda/(gamma*phi**3)))*(1 - (1 + (0.40229927701638585*(1 + 0.1*rs)*s2)/((-1 + np.exp(-lda/(gamma*phi**3)))*gamma*phi**2*(1 + 0.1778*rs)*rs))**(-0.25)))
        #dECdrs = ((-1 + np.exp(-lda/(gamma*phi**3)))*gamma*phi**3*((-0.40229927701638585*(1 + 0.1*rs)*s2)/((-1 + np.exp(-lda/(gamma*phi**3)))*gamma*phi**2*(1 + 0.1778*rs)*rs**2) - (0.0715288114535134*(1 + 0.1*rs)*s2)/((-1 + np.exp(-lda/(gamma*phi**3)))*gamma*phi**2*(1 + 0.1778*rs)**2*rs) + (0.040229927701638586*s2)/((-1 + np.exp(-lda/(gamma*phi**3)))*gamma*phi**2*(1 + 0.1778*rs)*rs)))/(4.*(1 + (0.40229927701638585*(1 + 0.1*rs)*s2)/((-1 + np.exp(-lda/(gamma*phi**3)))*gamma*phi**2*(1 + 0.1778*rs)*rs))**1.25*(1 + (-1 + np.exp(-lda/(gamma*phi**3)))*(1 - (1 + (0.40229927701638585*(1 + 0.1*rs)*s2)/((-1 + np.exp(-lda/(gamma*phi**3)))*gamma*phi**2*(1 + 0.1778*rs)*rs))**(-0.25))))
        #dECds2 = (0.10057481925409646*phi*(1 + 0.1*rs))/((1 + 0.1778*rs)*rs*(1 + (0.40229927701638585*(1 + 0.1*rs)*s2)/((-1 + np.exp(-lda/(gamma*phi**3)))*gamma*phi**2*(1 + 0.1778*rs)*rs))**1.25*(1 + (-1 + np.exp(-lda/(gamma*phi**3)))*(1 - (1 + (0.40229927701638585*(1 + 0.1*rs)*s2)/((-1 + np.exp(-lda/(gamma*phi**3)))*gamma*phi**2*(1 + 0.1778*rs)*rs))**(-0.25))))
        #dECdGZ = (gamma*phi**3*(((-1 + np.exp(-lda/(gamma*phi**3)))*((-1.2068978310491576*lda*(1 + 0.1*rs)*s2)/(np.exp(lda/(gamma*phi**3))*(-1 + np.exp(-lda/(gamma*phi**3)))**2*gamma**2*phi**6*(1 + 0.1778*rs)*rs) - (0.8045985540327717*(1 + 0.1*rs)*s2)/((-1 + np.exp(-lda/(gamma*phi**3)))*gamma*phi**3*(1 + 0.1778*rs)*rs)))/(4.*(1 + (0.40229927701638585*(1 + 0.1*rs)*s2)/((-1 + np.exp(-lda/(gamma*phi**3)))*gamma*phi**2*(1 + 0.1778*rs)*rs))**1.25) + (3*lda*(1 - (1 + (0.40229927701638585*(1 + 0.1*rs)*s2)/((-1 + np.exp(-lda/(gamma*phi**3)))*gamma*phi**2*(1 + 0.1778*rs)*rs))**(-0.25)))/(np.exp(lda/(gamma*phi**3))*gamma*phi**4)))/(1 + (-1 + np.exp(-lda/(gamma*phi**3)))*(1 - (1 + (0.40229927701638585*(1 + 0.1*rs)*s2)/((-1 + np.exp(-lda/(gamma*phi**3)))*gamma*phi**2*(1 + 0.1778*rs)*rs))**(-0.25))) + 3*gamma*phi**2*np.log(1 + (-1 + np.exp(-lda/(gamma*phi**3)))*(1 - (1 + (0.40229927701638585*(1 + 0.1*rs)*s2)/((-1 + np.exp(-lda/(gamma*phi**3)))*gamma*phi**2*(1 + 0.1778*rs)*rs))**(-0.25)))
        return e, dedlda, dedrs, dedphi, deds2

    def ss_baseline(self, n, g2):
        N = n.shape[0]
        rs, drs = self.get_rs(n)
        s2, ds2n, ds2g2 = self.get_s2(n, g2)
        lda, dlda = eval_xc(',LDA_C_PW_MOD', (n, 0*n), spin=1)[:2]
        dlda = (dlda[0][:,0] - lda) / n
        e, dedlda, dedrs, _, deds2 = self.baseline1(lda, rs, 1, s2)
        vxc = [None, None, None, None]
        #vxc[0] = dedrs * drs + deds2 * ds2n + dedlda * dlda[0][:,0]
        vxc[0] = dedrs * drs + deds2 * ds2n + dedlda * dlda
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
            lda, dlda = eval_xc(',LDA_C_PW_MOD', (nu, nd), spin=1)[:2]
            dldau = (dlda[0][:,0] - lda) / (nu + nd)
            dldad = (dlda[0][:,1] - lda) / (nu + nd)
            e, dedlda, dedrs, dedzeta, deds2 = self.baseline1(lda, rs, zeta, s2)
        vxc = [np.zeros((N,2)), np.zeros((N,3)), None, None]
        vxc[0][:,0] = dedrs * drs + dedzeta * dzetau + deds2 * ds2n
        vxc[0][:,1] = dedrs * drs + dedzeta * dzetad + deds2 * ds2n
        if type == 1:
            #vxc[0][:,0] += dedlda * dlda[0][:,0]
            vxc[0][:,0] += dedlda * dldau
            #vxc[0][:,1] += dedlda * dlda[0][:,1]
            vxc[0][:,1] += dedlda * dldad
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

    def corrfunc(self, x2, z, gamma, d):
        #print(d)
        d0, d1, d2, d3, d4, d5 = d
        # NOTE: 0 in HEG limit
        y = d0*(-1 + 1/gamma) + (d1*x2 + d2*z)/gamma**2 + (d3*x2**2 + d4*x2*z + d5*z**2)/gamma**3
        dydx2 = (d1*gamma + 2*d3*x2 + d4*z)/gamma**3
        dydz = (d2*gamma + d4*x2 + 2*d5*z)/gamma**3
        dydgamma = -((d0*gamma**2 + 3*d3*x2**2 + 2*gamma*(d1*x2 + d2*z) + 3*z*(d4*x2 + d5*z))/gamma**4)
        return y, dydx2, dydz, dydgamma

    def xef_terms(self, f, c):
        y = 0
        d = 0
        fterm = (1 - f**6) / (1 + f**6)
        dterm = (-12*f**5)/(1 + f**6)**2
        for i in range(4):
            y += c[i] * fterm**(i+1)
            d += c[i] * (i+1) * fterm**i
        return y, d * dterm

    def grad_terms(self, x2, gamma, c):
        y = 0
        dy = 0
        u = gamma * x2 / (1 + gamma * x2)
        du = gamma / (1 + gamma * x2)**2
        for i in range(4):
            y += c[i] * u**(i+1)
            dy += c[i] * (i+1) * u**i
        return y, dy * du

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
        y = 0.5 * (1 - np.cos(np.pi * y))
        return y, dy * dydn, dy * dydg2, dy * dydt

    def single_corr(self, x2, z, alpha, d):
        gamma = self.gammafunc(x2, z, alpha)
        corrfunc = self.corrfunc(x2, z, gamma[0], d)
        return corrfunc[0], corrfunc[1] + corrfunc[3] * gamma[1],\
                            corrfunc[2] + corrfunc[3] * gamma[2]

    def get_amix(self, rhou, rhod, g2u, g2o, g2d, Do):
        rhot = rhou + rhod

        g2 = (g2u + 2 * g2o + g2d)
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
        """
        Return tot
        Return yu, yd, yo, yx
        Return derivs wrt nu, nd, g2u, g2o, d2g, tu, td, fu, fd
            that arise from y*
        """
        g2 = g2u + g2d + 2 * g2o
        nt = nu + nd
        cu, vu = self.ss_baseline(nu, g2u)
        cd, vd = self.ss_baseline(nd, g2d)
        co, vo = self.os_baseline(nu, nd, g2, type=1)
        cx, vx = self.os_baseline(nu, nd, g2, type=0)
        co *= nt
        cx *= nt
        cu *= nu
        cd *= nd
        co -= cu + cd
        vo[0][:,0] -= vu[0]
        vo[0][:,1] -= vd[0]
        vo[1][:,0] -= vu[1]
        vo[1][:,2] -= vd[1]

        ldaxu = 2**(1.0/3) * LDA_FACTOR * nu**(4.0/3)
        dldaxu = 2**(1.0/3) * 4.0 / 3 * LDA_FACTOR * nu**(1.0/3)
        ldaxd = 2**(1.0/3) * LDA_FACTOR * nd**(4.0/3)
        dldaxd = 2**(1.0/3) * 4.0 / 3 * LDA_FACTOR * nd**(1.0/3)
        ldaxt = LDA_FACTOR * (nu+nd)**(4.0/3)
        ft = (fu * ldaxu + fd * ldaxd) / ldaxt
        dftdfu = (2**0.3333333333333333*nu**1.3333333333333333)/(nd + nu)**1.3333333333333333
        dftdfd = (2**0.3333333333333333*nd**1.3333333333333333)/(nd + nu)**1.3333333333333333
        dftdnu = (4*2**0.3333333333333333*(-(fd*nd**1.3333333333333333) + fu*nd*nu**0.3333333333333333))/(3.*(nd + nu)**2.3333333333333335)
        dftdnd = (4*2**0.3333333333333333*(fd*nd**0.3333333333333333*nu - fu*nu**1.3333333333333333))/(3.*(nd + nu)**2.3333333333333335)

        Du = self.get_D(nu, g2u, tu)
        Dd = self.get_D(nd, g2d, td)
        Do = self.get_D(nu+nd, g2, tu+td)

        amix, vxcmix = self.get_amix(nu, nd, g2u, g2o, g2d, Do)

        yu, derivu = self.xef_terms(fu, self.css)
        yd, derivd = self.xef_terms(fd, self.css)
        yo, derivo = self.xef_terms(ft, self.cos)
        ym, derivm = self.xef_terms(ft, self.cm)
        yx, derivx = self.xef_terms(ft, self.cx)
        yau, derivau = self.xef_terms(fu, self.ca)
        yad, derivad = self.xef_terms(fd, self.ca)

        ym += 1
        yo += 1
        yu += 1
        yd += 1
        tot = 0
        
        tot = cu * Du[0] * yu + cd * Dd[0] * yd \
              + co * yo + cx * yx \
              + (cx - co) * (1 - Do[0]) * ym \
              + ldaxu * amix * yau \
              + ldaxd * amix * yad
        
        N = cu.shape[0]
        vxc = [np.zeros((N,2)),
               np.zeros((N,3)),
               np.zeros((N,2)),
               np.zeros((N,2))]

        fill_vxc_ss_(vxc, 0, cu * Du[1] * yu,
                     cu * Du[2] * yu,
                     cu * Du[3] * yu,
                     cu * Du[0] * derivu + ldaxu * amix * derivau)
        fill_vxc_ss_(vxc, 1, cd * Dd[1] * yd,
                     cd * Dd[2] * yd,
                     cd * Dd[3] * yd,
                     cd * Dd[0] * derivd + ldaxd * amix * derivad)
        tmp = (co - cx) * ym
        fill_vxc_os_(vxc, tmp * Do[1],
                     tmp * Do[2],
                     tmp * Do[3],
                     (cx - co) * (1 - Do[0]) * derivm\
                        + co * derivo + cx * derivx,
                     dftdnu, dftdnd, dftdfu, dftdfd)

        # x2, dx2dn, dx2dg2
        x2u = self.get_x2(nu, g2u)
        x2d = self.get_x2(nd, g2d)
        # z, dzdn, dzdt
        zu = self.get_z(nu, tu)
        zd = self.get_z(nd, td)

        x2 = (x2u, x2d)
        z = (zu, zd)
        css = (cu, cd)
        ldaxm = (ldaxu * amix, ldaxd * amix)

        cfu = self.single_corr(x2u[0], zu[0], alphass, self.dss)
        cfd = self.single_corr(x2d[0], zd[0], alphass, self.dss)
        cfo = self.single_corr(x2u[0]+x2d[0], zu[0]+zd[0], alphaos, self.dos)
        cfx = self.single_corr(x2u[0]+x2d[0], zu[0]+zd[0], alphaos, self.dx)
        cfm = self.single_corr(x2u[0]+x2d[0], zu[0]+zd[0], alphaos, self.dm)
        cfau = self.single_corr(x2u[0], zu[0], alphax, self.da)
        cfad = self.single_corr(x2d[0], zd[0], alphax, self.da)

        tot += cu * Du[0] * cfu[0] + cd * Dd[0] * cfd[0] \
               + co * cfo[0] + cx * cfx[0] \
               + (cx - co) * (1 - Do[0]) * cfm[0] \
               + ldaxu * amix * cfau[0] \
               + ldaxd * amix * cfad[0]
        
        #tot += cx * cfx[0] + ldaxu * amix * cfau[0] + ldaxd * amix * cfad[0]
        cfss = (cfu, cfd)
        cfssa = (cfau, cfad)
        Dss = (Du, Dd)
        cm = (cx - co) * (1 - Do[0])

        for i in range(2):
            fill_vxc_ss_(vxc, i,
                         ldaxm[i] * (cfssa[i][1] * x2[i][1] \
                            + cfssa[i][2] * z[i][1]),
                         ldaxm[i] * cfssa[i][1] * x2[i][2],
                         ldaxm[i] * cfssa[i][2] * z[i][2])
            tmp = css[i] * Dss[i][0]
            fill_vxc_ss_(vxc, i,
                         tmp * (cfss[i][1] * x2[i][1] \
                            + cfss[i][2] * z[i][1]), # deriv wrt nu
                         tmp * cfss[i][1] * x2[i][2], # deriv wrt sigma_u
                         tmp * cfss[i][2] * z[i][2]) # deriv wrt tau_u
            for c, cf in [(co, cfo), (cx, cfx), (cm, cfm)]:
                fill_vxc_ss_(vxc, i,
                             c * (cf[1] * x2[i][1] + cf[2] * z[i][1]),
                             c * cf[1] * x2[i][2],
                             c * cf[2] * z[i][2])
            fill_vxc_ss_(vxc, i,
                         css[i] * cfss[i][0] * Dss[i][1],
                         css[i] * cfss[i][0] * Dss[i][2],
                         css[i] * cfss[i][0] * Dss[i][3])

        tmp = (co - cx) * cfm[0]
        fill_vxc_os_(vxc, tmp * Do[1],
                     tmp * Do[2],
                     tmp * Do[3])
       
        vxc[0][:,0] += dldaxu * amix * (yau + cfau[0])
        vxc[0][:,1] += dldaxd * amix * (yad + cfad[0])

        #fill_vxc_base_os_(vxc, vxcmix, ldaxu * (yau + cfau[0])
        #                             + ldaxd * (yad + cfad[0]))

        fill_vxc_base_ss_(vxc, vu, Du[0] * (yu + cfu[0]), 0)
        fill_vxc_base_ss_(vxc, vd, Dd[0] * (yd + cfd[0]), 1)
        fill_vxc_base_os_(vxc, vo, yo + cfo[0] - (1 - Do[0]) * (ym + cfm[0]))
        fill_vxc_base_os_(vxc, vx, yx + cfx[0] + (1 - Do[0]) * (ym + cfm[0]))

        rhou, rhod = nu, nd
        vxc[0][rhou<1e-8,0] = 0
        vxc[1][rhou<1e-8,0] = 0
        vxc[2][rhou<1e-8,0] = 0
        vxc[3][rhou<1e-8,0] = 0
        vxc[0][rhod<1e-8,1] = 0
        vxc[1][rhod<1e-8,2] = 0
        vxc[2][rhod<1e-8,1] = 0
        vxc[3][rhod<1e-8,1] = 0
        vxc[1][np.minimum(rhou,rhod)<1e-8,1] = 0

        print('check', nu[np.argmax(np.abs(np.sum(vxc[0] * nd[:,np.newaxis], axis=-1)))])
        print('check', g2u[np.argmax(np.abs(np.sum(vxc[0] * nd[:,np.newaxis], axis=-1)))])
        print('check', tu[np.argmax(np.abs(np.sum(vxc[0] * nd[:,np.newaxis], axis=-1)))])
        print('check', nd[np.argmax(np.abs(np.sum(vxc[0] * nd[:,np.newaxis], axis=-1)))])
        print('check', g2d[np.argmax(np.abs(np.sum(vxc[0] * nd[:,np.newaxis], axis=-1)))])
        print('check', td[np.argmax(np.abs(np.sum(vxc[0] * nd[:,np.newaxis], axis=-1)))])
        #print(g2u[np.argmax(vxc[1])])
        #print(tu[np.argmax(vxc[2])])
        #print(fu[np.argmax(vxc[3])])

        return tot, vxc
