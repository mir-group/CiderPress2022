# Autocode from mathematica for VSXC-type contribs

alpha_x = 0.001867
alpha_ss, alpha_os = 0.00515088, 0.00304966
CF = 0.3 * (6 * np.pi**2)**(2.0/3)

class VSXCContribs():

    def __init__(self, cx, css, cos, dx, dss, dos):
        self.cx = cx
        self.css = css
        self.cos = cos
        self.dx = dx
        self.dss = dss
        self.dos = dos

    def gammafunc(x2, z, alpha):
        y = 1 + alpha * (x2 + z)
        dydx2 = alpha
        dydz = alpha
        return y, dydx2, dydz

    def corrfunc(self, x2, z, gamma, d):
        y = (-(d0*(-1 + gamma)*gamma**2) + gamma**3 + d1*gamma*x2 + d3*x2**2 + d2*gamma*z + d4*x2*z + d5*z**2)/gamma**3
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

    def get_x2(n, g2):
        return g2/n**2.6666666666666665, (-8*g2)/(3.*n**3.6666666666666665), n**(-2.6666666666666665)

    def get_z(n, t):
        return -2*CF + (2*t)/n**1.6666666666666667,\
               (-10*t)/(3.*n**2.6666666666666665),\
               2/n**1.6666666666666667

    def getD(n, g2, t):
        y = 1 - g2/(8.*n*t)
        dydn = g2/(8.*n**2*t)
        dydg2 = -1/(8.*n*t)
        dydt = g2/(8.*n*t**2)
        return y, dydn, dydg2, dydt

    def single_corr(self, x2, z, alpha, d):
        gamma = self.gammafunc(x2, z, alpha)
        corrfunc = self.corrfunc(x2, z, gamma, d)
        return corrfunc[0], corrfunc[1] + corrfunc[3] * gamma[1],\
                            corrfunc[2] + corrfunc[3] * gamma[2]

    def corr_fock(self, cu, cd, co, vuu, vdd, vou, vod,
                  nu, nd, g2u, g2d, tu, td, fu, fd):

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

        yu, derivu = self.xef_terms(fu, self.css)
        yd, derivd = self.xef_terms(fd, self.css)
        yo, derivo = self.xef_terms(ft, self.cos)

        cyu = cu * yu
        cyd = cd * yd

        uterms = [cyu * Du[i] for i in range(4)]
        dterms = [cyd * Dd[i] for i in range(4)]
        uterms += [cu * Du[0] * derivu]
        dterms += [cd * Dd[0] * derivd]
        uterms[1] += vuu * yu * Du[0]
        dterms[1] += vdd * yd * Dd[0]
        uterms[1] += vou * yo + co * derivo * dftdnu
        oterms[1] += vod * yo + co * derivo * dftdnd
        uterms[-1] += co * derivo * dftdfu
        dterms[-1] += co * derivo * dftdfd

        return uterms[0] + dterms[0] + co * yo, uterms[1:], dterms[1:]

    def ex_fock(self, n, f):

        y, deriv = self.xef_terms(f, self.cx)
        return 2**(1.0/3) * LDA_FACTOR * n**(4.0/3) * y,\
               2**(1.0/3) * (4.0/3) * LDA_FACTOR * n**(1.0/3) * y,\
               2**(1.0/3) * LDA_FACTOR * n**(4.0/3) * deriv

    def corr_mn(self, cu, cd, co, vuu, vdd, vou, vod, nu, nd, g2u, g2d, tu, td):

        # x2, dx2dn, dx2dg2
        x2u = self.get_x2(nu, g2u)
        x2d = self.get_x2(nd, g2d)
        # z, dzdn, dzdt
        zu = self.get_z(nu, tu)
        zd = self.get_z(nd, td)

        cfuu = self.single_corr(x2u[0], zu[0], alphass, self.dss)
        cfdd = self.single_corr(x2d[0], zd[0], alphass, self.dss)
        cfud = self.single_corr(x2u[0]+x2d[0], zu[0]+zd[0], alphaos, self.dos)

        cfuu = (cfuu[0],
                cfuu[1] * x2u[1] + cfuu[2] * zu[1],
                cfuu[1] * x2u[2],
                cfuu[2] * zu[2])
        cfdd = (cfdd[0],
                cfdd[1] * x2d[1] + cfdd[2] * zd[1],
                cfdd[1] * x2d[2],
                cfdd[2] * zd[2])

        cfud0 = (cfud[0],
                 cfud[1] * x2u[1] + cfud[2] * zu[1],
                 cfud[1] * x2u[2],
                 cfud[2] * zu[2])
        cfud1 = (cfud[0],
                 cfud[1] * x2d[1] + cfud[2] * zd[1],
                 cfud[1] * x2d[2],
                 cfud[2] * zd[2])

        tot = cu * cfuu[0] + cd * cfdd[0] + co * cfud[0]

        cfuu_tmp = cfuu[0]
        cfuu = (Du[0] * tmp for tmp in cfuu)
        cfuu[1] += cfuu_tmp * Du[1]
        cfuu[2] += cfuu_tmp * Du[2]
        dfuu[3] += cfuu_tmp * Du[3]

        cfdd_tmp = cfdd[0]
        cfdd = (Dd[0] * tmp for tmp in cfdd)
        cfdd[1] += cfdd_tmp * Dd[1]
        cfdd[2] += cfdd_tmp * Dd[2]
        cfdd[3] += cfdd_tmp * Dd[3]

        uderiv = [cu * cfuu[i] + co * cfuu0[i] for i in range(1,4)]
        dderiv = [cd * cfdd[i] + co * cfud1[i] for i in range(1,4)]
        uderiv[0] += cfuu[0] * vuu + cfud[0] * vou
        dderiv[0] += cfdd[0] * vdd + cfud[0] * vod

        return tot, uderiv, dderiv
        
    def ex_mn(self, n, g2, t):

        # x2, dx2dn, dx2dg2
        x2 = self.get_x2(n, g2)
        # z, dzdn, dzdt
        z = self.get_z(n, t)

        cf = self.single_corr(x2[0], z[0], alphax, self.dx)

        cf = (cf[0],
              cf[1] * x2[1] + cf[2] * z[1],
              cf[1] * x2[2],
              cf[2] * z[2])

        cf = [2**(1.0/3) * LDA_FACTOR * n**(4.0/3) * c for c in cf]
        cf[1] += 2**(1.0/3) * (4.0/3) * LDA_FACTOR * n**(1.0/3)

        return cf





