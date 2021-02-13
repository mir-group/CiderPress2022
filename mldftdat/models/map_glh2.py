# Autocode from mathematica for VSXC-type contribs
import numpy as np
from pyscf.dft.libxc import eval_xc
from mldftdat.xcutil.cdesc import *

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

    def __init__(self, c, fterm_scale=2.0):
        self.c = c
        self.cf = fterm_scale

    def xefc(self, nu, nd, g2u, g2o, g2d, tu, td, exu, exd, hfx=True):

        ldaxu = 2**(1.0/3) * LDA_FACTOR * nu**(4.0/3) + 1e-20
        dldaxu = 2**(1.0/3) * 4.0 / 3 * LDA_FACTOR * nu**(1.0/3)
        ldaxd = 2**(1.0/3) * LDA_FACTOR * nd**(4.0/3) + 1e-20
        dldaxd = 2**(1.0/3) * 4.0 / 3 * LDA_FACTOR * nd**(1.0/3)
        ldaxt = ldaxu + ldaxd
        if hfx:
            ft = (exu + exd) / ldaxt
            fu = exu / ldaxu
            fd = exd / ldaxd
            dftdxu = 1 / ldaxt
            dftdxd = 1 / ldaxt
            dftdnu = -ft / ldaxt * dldaxu
            dftdnd = -ft / ldaxt * dldaxd
            dfudnu = -fu / ldaxu * dldaxu
            dfddnd = -fd / ldaxd * dldaxd
            dfudxu = 1 / ldaxu
            dfddxd = 1 / ldaxd
        else:
            fu = exu
            fd = exd
            ft = (ldaxu * fu + ldaxd * fd) / ldaxt
            dftdfu = ldaxu / ldaxt
            dftdfd = ldaxd / ldaxt
            dftdnu = ldaxd * (fu - fd) / ldaxt**2 * dldaxu
            dftdnd = ldaxu * (fd - fu) / ldaxt**2 * dldaxd
            dfudnu = -fu / ldaxu * dldaxu
            dfddnd = -fd / ldaxd * dldaxd

        g2 = g2u + g2d + 2 * g2o
        nt = nu + nd
        x2 = get_x2(nt, g2)
        x2u = get_x2(nu, g2u)
        x2d = get_x2(nd, g2d)
        zeta = get_zeta(nu, nd)
        
        chi = get_chi_full_deriv(nt, zeta[0], g2, tu+td)
        chiu = get_chi_full_deriv(nu, 1, g2u, tu)
        chid = get_chi_full_deriv(nd, -1, g2d, td)
        chiu[0][np.isnan(chiu[0])] = 0
        chid[0][np.isnan(chid[0])] = 0
        
        c0, dc0dn, dc0dz, dc0dx2 = get_os_baseline2(nt, zeta[0], x2[0], type=0)
        c1, dc1dn, dc1dz, dc1dx2 = get_os_baseline2(nt, zeta[0], x2[0], type=1)
        c0 *= nt
        c1 *= nt
        amix, vmixn, vmixz, vmixx2, vmixchi = \
            get_amix_schmidt2(nt, zeta[0], x2[0], chi[0])
        a2mix, v2mixn, v2mixz, v2mixx2, v2mixchi = \
            get_amix_schmidt2(nt, zeta[0], x2[0], chi[0],
                              mixer='n')
        
        ldaxm = (ldaxu * amix, ldaxd * amix)
        N = nt.shape[0]
        vxc = [np.zeros((N,2)),
               np.zeros((N,3)),
               np.zeros((N,2)),
               np.zeros((N,2))]
        # deriv wrt n, zeta, x2, chi, F
        vtmp = [np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)]
        # deriv wrt x2u, chiu, Fu
        vtmpu = [np.zeros(N), np.zeros(N), np.zeros(N)]
        # deriv wrt x2d, chid, Fd
        vtmpd = [np.zeros(N), np.zeros(N), np.zeros(N)]
        tot = 0

        gamma = 0.004 * 2**(2./3)
        gammax = 0.004
        gammass = gammax

        gt = gamma * x2[0] / (1 + gamma * x2[0])
        gtu = gammass * x2u[0] / (1 + gammass * x2u[0])
        gtd = gammass * x2d[0] / (1 + gammass * x2d[0])
        dgt = gamma / (1 + gamma * x2[0])**2
        dgtu = gammass / (1 + gammass * x2u[0])**2
        dgtd = gammass / (1 + gammass * x2d[0])**2

        Fexp = np.exp(-2*(ft-1)**2) * (ft-1)
        Fexpu = np.exp(-2*(fu-1)**2) * (fu-1)
        Fexpd = np.exp(-2*(fd-1)**2) * (fd-1)
        dFexp = np.exp(-2*(ft-1)**2) * (1 - 4 * (ft-1)**2)
        dFexpu = np.exp(-2*(fu-1)**2) * (1 - 4 * (fu-1)**2)
        dFexpd = np.exp(-2*(fd-1)**2) * (1 - 4 * (fd-1)**2)

        achi, dachi = get_chidesc_small(chi[0])
        A = 17.0 / 3
        slc = A * (1 - chi[0]) / (A - chi[0]) - np.dot(self.c[:3], achi)
        dslc = (A - A**2) / (A - chi[0])**2 - np.dot(self.c[:3], dachi)
        slu, dsludx2, dsludchi = [self.c[5:13].dot(term) for term in \
                                  get_sl_small(x2u[0], chiu[0], gammax)]
        sld, dslddx2, dslddchi = [self.c[5:13].dot(term) for term in \
                                  get_sl_small(x2d[0], chid[0], gammax)]
        nlu, dnludf, dnludchi = [self.c[13:20].dot(term) for term in \
                                 get_xefa_small(fu, chiu[0])]
        nld, dnlddf, dnlddchi = [self.c[13:20].dot(term) for term in \
                                 get_xefa_small(fd, chid[0])]

        tot += c1 * slc + c0 * (1 - slc)
        tot += self.c[3] * c1 * amix * (ft-1)
        tot += self.c[4] * c0 * amix * (ft-1)
        tot += ldaxm[0] * slu
        tot += ldaxm[1] * sld
        tot += ldaxm[0] * nlu
        tot += ldaxm[1] * nld
        tot += self.c[20] * c1 * (1-chi[0]**2) * Fexp
        tot += self.c[21] * c0 * Fexp
        ldaf = ldaxu * Fexpu + ldaxd * Fexpd
        tot += self.c[22] * amix**2 * ldaf
        tot += self.c[23] * a2mix * ldaf
        tot += self.c[24] * a2mix**2 * ldaf
        tot += self.c[25] * c1 * amix * gt
        tot += self.c[26] * c0 * amix * gt
        ldag = ldaxu * gtu + ldaxd * gtd
        tot += self.c[27] * amix**2 * ldag
        tot += self.c[28] * a2mix * ldag
        tot += self.c[29] * a2mix**2 * ldag
        # enhancment terms on c1 and c0
        vtmp[3] += (c1 - c0) * dslc
        vtmp[4] += (self.c[3] * c1 * amix + self.c[4] * c0 * amix)
        vtmp[4] += self.c[20] * c1 * (1-chi[0]**2) * dFexp \
                  + self.c[21] * c0 * dFexp
        vtmp[3] -= 2 * chi[0] * self.c[20] * c1 * Fexp
        vtmp[2] += (self.c[25] * c1 + self.c[26] * c0) * amix * dgt

        # amix derivs and exchange-like rho derivs
        tmp = ldaxu * (slu + nlu) + ldaxd * (sld + nld) \
            + self.c[3] * c1 * (ft-1) + self.c[4] * c0 * (ft-1)
        tmp += 2 * amix * (self.c[22] * ldaf + self.c[27] * ldag) \
               + self.c[25] * c1 * gt + self.c[26] * c0 * gt
        
        cond = nt > 1e-4
        vtmp[0][cond] += (tmp * vmixn)[cond]
        vtmp[1][cond] += (tmp * vmixz)[cond]
        vtmp[2][cond] += (tmp * vmixx2)[cond]
        vtmp[3][cond] += (tmp * vmixchi)[cond]

        tmp = (self.c[23] * ldaf + self.c[28] * ldag)\
            + 2 * a2mix * (self.c[24] * ldaf + self.c[29] * ldag)

        cond = nt > 1e-4
        vtmp[0][cond] += (tmp * v2mixn)[cond]
        vtmp[1][cond] += (tmp * v2mixz)[cond]
        vtmp[2][cond] += (tmp * v2mixx2)[cond]
        #vtmp[3][cond] += (tmp * v2mixchi)[cond]

        # exchange-like enhancment derivs
        tmp = ldaxm[0]
        vtmpu[0] += tmp * dsludx2
        vtmpu[1] += tmp * (dsludchi + dnludchi)
        vtmpu[2] += tmp * dnludf
        tmp = ldaxm[1]
        vtmpd[0] += tmp * dslddx2
        vtmpd[1] += tmp * (dslddchi + dnlddchi)
        vtmpd[2] += tmp * dnlddf
        
        tmp = self.c[22] * amix**2 + self.c[23] * a2mix + self.c[24] * a2mix**2
        vtmpu[2] += tmp * ldaxu * dFexpu
        vtmpd[2] += tmp * ldaxd * dFexpd
        tmp = self.c[27] * amix**2 + self.c[28] * a2mix + self.c[29] * a2mix**2
        vtmpu[0] += tmp * ldaxu * dgtu
        vtmpd[0] += tmp * ldaxd * dgtd

        # baseline derivs
        tmp = 1 - slc + self.c[4] * amix * (ft-1)
        tmp += self.c[21] * Fexp + self.c[26] * amix * gt
        vtmp[0] += tmp * dc0dn
        vtmp[1] += tmp * dc0dz
        vtmp[2] += tmp * dc0dx2
        tmp = slc + self.c[3] * amix * (ft-1)
        tmp += self.c[20] * (1-chi[0]**2) * Fexp + self.c[25] * amix * gt
        vtmp[0] += tmp * dc1dn
        vtmp[1] += tmp * dc1dz
        vtmp[2] += tmp * dc1dx2

        tmp = amix * (slu + nlu)
        tmp += amix**2 * (self.c[22] * Fexpu + self.c[27] * gtu)
        tmp += a2mix * (self.c[23] * Fexpu + self.c[28] * gtu)
        tmp += a2mix**2 * (self.c[24] * Fexpu + self.c[29] * gtu)
        vxc[0][:,0] += dldaxu * tmp
        tmp = amix * (sld + nld)
        tmp += amix**2 * (self.c[22] * Fexpd + self.c[27] * gtd)
        tmp += a2mix * (self.c[23] * Fexpd + self.c[28] * gtd)
        tmp += a2mix**2 * (self.c[24] * Fexpd + self.c[29] * gtd)
        vxc[0][:,1] += dldaxd * tmp

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
        if hfx:
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
        else:
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
        #print("NANS", np.isnan(vxc[0]).any(), np.isnan(vxc[1]).any(), np.isnan(vxc[2]).any(), np.isnan(vxc[3]).any())

        return tot, vxc

    def xefc1(self, nu, nd, g2u, g2o, g2d, tu, td, fu, fd):
        return self.xefc(nu, nd, g2u, g2o, g2d, tu, td, fu, fd, hfx=False)

    def xefc2(self, nu, nd, g2u, g2o, g2d, tu, td, exu, exd):
        return self.xefc(nu, nd, g2u, g2o, g2d, tu, td, exu, exd, hfx=True)
