# Autocode from mathematica for VSXC-type contribs
import numpy as np

LDA_FACTOR = - 3.0 / 4.0 * (3.0 / np.pi)**(1.0/3)
alphax = 0.001867
alphass, alphaos = 0.00515088, 0.00304966
CF = 0.3 * (6 * np.pi**2)**(2.0/3)

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

    def get_z(self, n, t):
        return -2*CF + (2*t)/n**1.6666666666666667,\
               (-10*t)/(3.*n**2.6666666666666665),\
               2/n**1.6666666666666667

    def get_D(self, n, g2, t):
        y = 1 - g2/(8.*n*t)
        dydn = g2/(8.*n**2*t)
        dydg2 = -1/(8.*n*t)
        dydt = g2/(8.*n*t**2)
        y = 0.5 * (1 - np.cos(np.pi * y))
        dy = 0.5 * np.pi * np.sin(np.pi * y)
        return y, dy * dydn, dy * dydg2, dy * dydt

    def single_corr(self, x2, z, alpha, d):
        gamma = self.gammafunc(x2, z, alpha)
        corrfunc = self.corrfunc(x2, z, gamma[0], d)
        return corrfunc[0], corrfunc[1] + corrfunc[3] * gamma[1],\
                            corrfunc[2] + corrfunc[3] * gamma[2]

    def get_amix(self, rhou, rhod, g2u, g2o, g2d, Do):
        rhot = rhou + rhod

        A = 2.74
        B = 132
        sprefac = 2 * (3 * np.pi**2)**(1.0/3)
        s2 = (g2u + 2 * g2o + g2d) / (sprefac**2 * rhot**(8.0/3) + 1e-20)

        zeta = (rhou - rhod) / (rhot)
        phi = ((1-zeta)**(2.0/3) + (1+zeta)**(2.0/3))/2
        phi43 = ((1-zeta)**(4.0/3) + (1+zeta)**(4.0/3))/2
        phi43 = (1 - 2.3631 * (phi43 - 1)) * (1-zeta**12)
        chi_inf = 0.128026
        chi = 0.72161
        b1c = 0.0285764
        gamma_eps = 0.031091

        part1 = b1c * np.log(1 + (1-np.e)/np.e / (1 + 4 * chi_inf * s2)**(0.25))
        part1 *= phi43
        part2 = gamma_eps * phi**3 * np.log((1 - 1 / (1 + 4 * chi * s2)**(0.25)) + 1e-30)
        epslim = part1 * (1-Do) + part2 * Do
        exlda = 2**(1.0 / 3) * LDA_FACTOR * rhou**(4.0/3)
        exlda += 2**(1.0 / 3) * LDA_FACTOR * rhod**(4.0/3)
        exlda /= (rhot)
        amix = 1 - 1 / (1 + A * np.log(1 + B * (epslim / exlda)))

    def xefc(self, cu, cd, co, cx, vu, vd, vo, vx,
             nu, nd, g2u, g2o, g2d, tu, td, fu, fd):
        """
        Return tot
        Return yu, yd, yo, yx
        Return derivs wrt nu, nd, g2u, g2o, d2g, tu, td, fu, fd
            that arise from y*
        """

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
        Do = self.get_D(nu+nd, g2u+2*g2o+g2d, tu+td)

        amix = self.get_amix(nu, nd, g2u, g2o, g2d, Do)

        yu, derivu = self.xef_terms(fu, self.css)
        yd, derivd = self.xef_terms(fd, self.css)
        yo, derivo = self.xef_terms(ft, self.cos)
        ym, derivm = self.xef_terms(ft, self.cm)
        yx, derivx = self.xef_terms(ft, self.cx)
        yau, derivau = self.xef_terms(fu, self.ca)
        yad, derivad = self.xef_terms(fd, self.ca)

        ym[0] += 1
        yo[0] += 1
        yu[0] += 1
        yd[0] += 1

        tot = cu * Du[0] * yu + cd * Dd[0] * yd \
              + co * yo + cx * yx \
              + (cx - co) * (1 - Do[0]) * ym \
              + ldaxu * (1 - amix) * yau \
              + ldaxd * (1 - amix) * yad

        N = cu.shape[0]
        vxc = [np.zeros((N,2)),
               np.zeros((N,3)),
               np.zeros((N,2)),
               np.zeros((N,2))]

        def fill_vxc_ss_(vxc, spin, dn, dg2, dt, df):
            vxc[0][:,spin] += dn
            vxc[1][:,2*spin] += dg2
            vxc[2][:,spin] += dt
            vxc[3][:,spin] += df
        def fill_vxc_os_(vxc, dn, dg2, dt, df):
            vxc[0][:,0] += dn + df * dftdnu
            vxc[0][:,1] += dn + df * dftdnd
            vxc[1][:,0] += dg2
            vxc[1][:,1] += 2 * dg2
            vxc[1][:,2] += dg2
            vxc[2][:,0] += dt
            vxc[2][:,1] += dt
            vxc[3][:,0] += df * dftdfu
            vxc[3][:,1] += df * dftdfd
        def fill_vxc_base_ss_(vxc, vterm, multerm, spin):
            if vterm[0] is not None:
                vxc[0][:,spin] += vterm[0][:,spin] * multerm
            if vterm[1] is not None:
                vxc[1][:,2*spin] += vterm[1][:,2*spin] * multerm
            if vterm[3] is not None:
                vxc[2][:,spin] += vterm[3][:,spin] * multerm
        def fill_vxc_base_os_(vxc, vterm, multerm):
            multerm = multerm.reshape(-1, 1)
            if vterm[0] is not None:
                vxc[0] += vterm[0] * multerm
            if vterm[1] is not None:
                vxc[1] += vterm[1] * multerm
            if vterm[3] is not None:
                vxc[2] += vterm[3] * multerm

        fill_vxc_ss_(vxc, 0, cu * Du[1] * yu,
                     cu * Du[2] * yu,
                     cu * Du[3] * yu,
                     cu * Du[0] * derivu + ldaxu * (1 - amix) * derivau)
        fill_vxc_ss_(vxc, 1, cd * Dd[1] * yd,
                     cd * Dd[2] * yd,
                     cd * Dd[3] * yd,
                     cd * Dd[0] * derivd + ldaxd * (1 - amix) * derivad)
        tmp = (co - cx) * ym
        fill_vxc_os_(vxc, tmp * Do[1],
                     tmp * Do[2],
                     tmp * Do[3],
                     (cx - co) * (1 - Do[0]) * derivm\
                        + co * derivo + cx * derivx)

        # x2, dx2dn, dx2dg2
        x2u = self.get_x2(nu, g2u)
        x2d = self.get_x2(nd, g2d)
        # z, dzdn, dzdt
        zu = self.get_z(nu, tu)
        zd = self.get_z(nd, td)

        x2 = (x2u, x2d)
        z = (zu, zd)
        css = (cu, cd)
        ldaxm = (ldaxu * (1 - amix), ldaxd * (1 - amix))

        cfu = self.single_corr(x2u[0], zu[0], alphass, self.dss)
        cfd = self.single_corr(x2d[0], zd[0], alphass, self.dss)
        cfo = self.single_corr(x2u[0]+x2d[0], zu[0]+zd[0], alphaos, self.dos)
        cfx = self.single_corr(x2u[0]+x2d[0], zu[0]+zd[0], alphaos, self.dx)
        cfm = self.single_corr(x2u[0]+x2d[0], zu[0]+zd[0], alphaos, self.dm)
        cfau = self.single_corr(x2u[0], zu[0], alphass, self.da)
        cfad = self.single_corr(x2d[0], zd[0], alphass, self.da)

        cfss = (cfu, cfd)
        cfssa = (cfau, cfad)
        Dss = (Du, Dd)
        cm = (cx - co) * (1 - Do[0])

        for i in range(2):
            fill_vxc_ss_(vxc, i,
                         ldaxm[i] * (cfssa[i][1] * x2[i][1] \
                            + cfssa[i][2] * z[i][1]),
                         ldaxm[i] * cfssa[i][1] * x2[i][2],
                         ldaxm[i] * cfssa[i][2] * z[i][2], 0)
            tmp = css[i] * Dss[i]
            fill_vxc_ss_(vxc, i,
                         tmp * (cfss[i][1] * x2[i][1] \
                            + cfss[i][2] * z[i][1]), # deriv wrt nu
                         tmp * cfss[i][1] * x2[i][2], # deriv wrt sigma_u
                         tmp * cfss[i][2] * z[i][2], 0) # deriv wrt tau_u
            for c, cf in [(co, cfo), (cx, cfx), (cm, cfm)]:
                fill_vxc_ss_(vxc, i,
                             c * (cf[1] * x2[i][1] + cf[2] * z[i][1]),
                             c * cf[1] * x2[i][2],
                             c * cf[2] * z[i][2], 0)
            fill_vxc_ss_(vxc, i,
                         css[i] * cfss[i][0] * Dss[i][1],
                         css[i] * cfss[i][0] * Dss[i][2],
                         css[i] * cfss[i][0] * Dss[i][3],
                         0)

        tmp = (co - cx) * cfm[0]
        fill_vxc_os_(vxc, tmp * Do[1],
                     tmp * Do[2],
                     tmp * Do[3],
                     0)
        
        fill_vxc_base_ss_(vxc, vu, Du[0] * (yu + cfu[0]), 0)
        fill_vxc_base_ss_(vxc, dldaxu, yau + cfau[0], 0)
        fill_vxc_base_ss_(vxc, vd, Dd[0] * (yd + cfd[0]), 1)
        fill_vxc_base_ss_(vxc, dldaxd, yad + cfad[0], 0)
        fill_vxc_base_os_(vxc, vo, yo + cfo[0] - (1 - Do[0]) * (ym + cfm[0]))
        fill_vxc_base_os_(vxc, vx, yx + cfx[0] + (1 - Do[0]) * (ym + cfm[0]))

        return tot, vxc
