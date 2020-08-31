# Autocode from mathematica for VSXC-type contribs
import numpy as np

LDA_FACTOR = - 3.0 / 4.0 * (3.0 / np.pi)**(1.0/3)
alphax = 0.001867
alphass, alphaos = 0.00515088, 0.00304966
CF = 0.3 * (6 * np.pi**2)**(2.0/3)

class VSXCContribs():

    def __init__(self, cx, css, cos, dx, dss, dos,
                 bx=None, bss=None, bos=None):
        self.cx = cx
        self.css = css
        self.cos = cos
        self.dx = dx
        self.dss = dss
        self.dos = dos
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

    def gammafunc(self, x2, z, alpha):
        y = 1 + alpha * (x2 + z)
        dydx2 = alpha
        dydz = alpha
        return y, dydx2, dydz

    def corrfunc(self, x2, z, gamma, d):
        #print(d)
        d0, d1, d2, d3, d4, d5 = d
        y = 1 + d0*(-1 + 1/gamma) + (d1*x2 + d2*z)/gamma**2 + (d3*x2**2 + d4*x2*z + d5*z**2)/gamma**3
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

    def get_z(self, n, t):
        return -2*CF + (2*t)/n**1.6666666666666667,\
               (-10*t)/(3.*n**2.6666666666666665),\
               2/n**1.6666666666666667

    def getD(self, n, g2, t):
        y = 1 - g2/(8.*n*t)
        dydn = g2/(8.*n**2*t)
        dydg2 = -1/(8.*n*t)
        dydt = g2/(8.*n*t**2)
        return y, dydn, dydg2, dydt

    def single_corr(self, x2, z, alpha, d):
        gamma = self.gammafunc(x2, z, alpha)
        corrfunc = self.corrfunc(x2, z, gamma[0], d)
        return corrfunc[0], corrfunc[1] + corrfunc[3] * gamma[1],\
                            corrfunc[2] + corrfunc[3] * gamma[2]

    def corr_fock(self, cu, cd, co, cx,
                  nu, nd, g2u, g2o, g2d, tu, td, fu, fd):
        """
        Return tot
        Return yu, yd, yo, yx
        Return derivs wrt nu, nd, g2u, g2o, d2g, tu, td, fu, fd
            that arise from y*
        """

        ldaxu = 2**(1.0/3) * nu**(4.0/3)
        ldaxd = 2**(1.0/3) * nd**(4.0/3)
        ldaxt = (nu+nd)**(4.0/3)
        ft = (fu * ldaxu + fd * ldaxd) / ldaxt
        dftdfu = (2**0.3333333333333333*nu**1.3333333333333333)/(nd + nu)**1.3333333333333333
        dftdfd = (2**0.3333333333333333*nd**1.3333333333333333)/(nd + nu)**1.3333333333333333
        dftdnu = (4*2**0.3333333333333333*(-(fd*nd**1.3333333333333333) + fu*nd*nu**0.3333333333333333))/(3.*(nd + nu)**2.3333333333333335)
        dftdnd = (4*2**0.3333333333333333*(fd*nd**0.3333333333333333*nu - fu*nu**1.3333333333333333))/(3.*(nd + nu)**2.3333333333333335)

        Du = self.getD(nu, g2u, tu)
        Dd = self.getD(nd, g2d, td)
        Do = self.get_D(nu+nd, g2u+2*g2o+g2d, tu+td)

        yu, derivu = self.xef_terms(fu, self.css)
        yd, derivd = self.xef_terms(fd, self.css)
        yo, derivo = self.xef_terms(ft, self.cos)
        yx, derivx = self.xef_terms(ft, self.cx)

        tot = cu * Du[0] * yu + cd * Dd[0] * yd \
              + co * Do[0] * yo + cx * (1 - Do[0]) * yx
        oterms = [co * yo * Do[i] for i in range(4)]
        xterms = [-cx * yx * Do[i] for i in range(4)]
        xterms[0] += cx * yx
        uterms = [cu * yu * Du[i] for i in range(4)]
        dterms = [cd * yd * Dd[i] for i in range(4)]
        dg2o = 2 * (co * yo - cx * yx) * Do[2]

        dostdf = Do[0] * co * derivo + (1 - Do[0]) * cx * derivx

        uderivs = [oterms[i] + xterms[i] + uterms[i] for i in range(1,4)]
        uderivs[0] += (1 - Do[0]) * cx * derivx * dftdnu
        uderivs[0] += Do[0] * co * derivo * dftdnu
        uderivs.append(dostdf * dftdfu)

        dderivs = [oterms[i] + xterms[i] + dterms[i] for i in range(1,4)]
        dderivs[0] += (1 - Do[0]) * cx * derivx * dftdnd
        dderivs[0] += Do[0] * co * derivo * dftdnd
        dderivs.append(dostdf * dftdfd)

        """
        dnu = cu * yu * Du[1] + co * (yo - yx) * Do[1]
        dnu += co * Do[0] * (derivo + derivx) * dftdnu
        dnd = cd * yd * Dd[1] + co * (yo - yx) * Do[1]
        dnd += co * Do[0] * (derivo + derivx) * dftdnd
        dg2u = cu * yu * Du[2] + co * (yo - yx) * Do[2]
        dg2o = 2 * co * (yo + yx) * Do[2]
        dg2d = cd * yd * Dd[2] + co * (yo - yx) * Do[2]
        dtu = cu * yu * Du[3] + co * (yo - yx) * Do[3]
        dtd = cu * yu * Dd[3] + co * (yo - yx) * Do[3]
        dfu = cu * Du[0] * derivu + co * (Do[0] * (derivo - derivx) + derivx) * dftdfu
        dfu = cd * Dd[0] * derivd + co * (Do[0] * (derivo - derivx) + derivx) * dftdfd
        """

        return tot, (yu * Du[0], yd * Dd[0], yo * Do[0], yx * (1 - Do[0])),\
               uderivs, dderivs, dg2o

    def corr_mn(self, cu, cd, co, cx, nu, nd, g2u, g2d, g2o, tu, td):

        # x2, dx2dn, dx2dg2
        x2u = self.get_x2(nu, g2u)
        x2d = self.get_x2(nd, g2d)
        # z, dzdn, dzdt
        zu = self.get_z(nu, tu)
        zd = self.get_z(nd, td)

        Du = self.getD(nu, g2u, tu)
        Dd = self.getD(nd, g2d, td)
        Do = self.getD(nu+nd, g2u+2*g2o+g2d, tu+td)

        cfu = self.single_corr(x2u[0], zu[0], alphass, self.dss)
        cfd = self.single_corr(x2d[0], zd[0], alphass, self.dss)
        cfo = self.single_corr(x2u[0]+x2d[0], zu[0]+zd[0], alphaos, self.dos)
        cfx = self.single_corr(x2u[0]+x2d[0], zu[0]+zd[0], alphaos, self.dx)
        yu, yd, yo, yx = cfu[0], cfd[0], cfo[0], cfx[0]

        print(np.mean(zu[0]*nu), np.mean(zd[0]*nd), np.mean(x2u[0]*nu),
              np.mean(x2d[0]*nd), np.mean(Du[0]*nu), np.mean(Dd[0]*nd))

        tot = cu * Du[0] * yu + cd * Dd[0] * yd \
              + co * Do[0] * yo + cx * (1 - Do[0]) * yx
        oterms = [co * yo * Do[i] for i in range(4)]
        xterms = [-cx * yx * Do[i] for i in range(4)]
        xterms[0] += cx * yx
        uterms = [cu * yu * Du[i] for i in range(4)]
        dterms = [cd * yd * Dd[i] for i in range(4)]
        dg2o = 2 * (co * yo - cx * yx) * Do[2]

        uderivs = [oterms[i] + xterms[i] + uterms[i] for i in range(1,4)]
        dderivs = [oterms[i] + xterms[i] + dterms[i] for i in range(1,4)]

        cfu = (cfu[0], # corr enhancement factor
               cfu[1] * x2u[1] + cfu[2] * zu[1], # deriv wrt nu
               cfu[1] * x2u[2], # deriv wrt sigma_u
               cfu[2] * zu[2]) # deriv wrt tau_u
        cfd = (cfd[0],
               cfd[1] * x2d[1] + cfd[2] * zd[1],
               cfd[1] * x2d[2],
               cfd[2] * zd[2])

        cfo0 = (cfo[0],
                cfo[1] * x2u[1] + cfo[2] * zu[1],
                cfo[1] * x2u[2],
                cfo[2] * zu[2])
        cfo1 = (cfo[0],
                cfo[1] * x2d[1] + cfo[2] * zd[1],
                cfo[1] * x2d[2],
                cfo[2] * zd[2])

        cfx0 = (cfx[0],
                cfx[1] * x2u[1] + cfx[2] * zu[1],
                cfx[1] * x2u[2],
                cfx[2] * zu[2])
        cfx1 = (cfx[0],
                cfx[1] * x2d[1] + cfx[2] * zd[1],
                cfx[1] * x2d[2],
                cfx[2] * zd[2])

        for i in range(3):
            uderivs[i] += cfu[i+1] * cu * Du[0]
            uderivs[i] += cfo0[i+1] * co * Do[0]
            uderivs[i] += cfx0[i+1] * cx * (1 - Do[0])
            dderivs[i] += cfd[i+1] * cd * Dd[0]
            dderivs[i] += cfo1[i+1] * co * Do[0]
            dderivs[i] += cfx1[i+1] * cx * (1 - Do[0])

        tot = cu * Du[0] * cfu[0] + cd * Dd[0] * cfd[0] \
              + co * Do[0] * cfo[0] + cx * (1 - Do[0]) * cfx[0]

        return tot, (yu * Du[0], yd * Dd[0], yo * Do[0], yx * (1 - Do[0])),\
               uderivs, dderivs, dg2o
        
    def ex_mn(self, n, g2, t):

        # x2, dx2dn, dx2dg2
        x2 = self.get_x2(n, g2)
        # z, dzdn, dzdt
        z = self.get_z(n, t)

        cf = self.single_corr(x2[0], z[0], alphax, self.dx)

        cf = (cf[0] - 1,
              cf[1] * x2[1] + cf[2] * z[1],
              cf[1] * x2[2],
              cf[2] * z[2])

        ecf = [2**(1.0/3) * LDA_FACTOR * n**(4.0/3) * c for c in cf]
        ecf[1] += 2**(1.0/3) * (4.0/3) * LDA_FACTOR * n**(1.0/3) * cf[0]

        return ecf

    def ex_fock(self, n, f):

        y, deriv = self.xef_terms(f, self.cx)
        return 2**(1.0/3) * LDA_FACTOR * n**(4.0/3) * y,\
               2**(1.0/3) * (4.0/3) * LDA_FACTOR * n**(1.0/3) * y,\
               2**(1.0/3) * LDA_FACTOR * n**(4.0/3) * deriv
