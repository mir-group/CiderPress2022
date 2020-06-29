import jax.numpy as np 
from jax import grad, jit, vmap
import matplotlib.pyplot as plt 
from mldftdat.density import tail_fx, tail_fx_direct, tail_fx_deriv, tail_fx_deriv_direct

sprefac = 2 * (3 * np.pi * np.pi)**(1.0/3)
s0 = 1 / (0.5 * sprefac / np.pi**(1.0/3) * 2**(1.0/3))
hprefac = 1.0 / 3 * (4 * np.pi**2 / 3)**(1.0 / 3)

def hex(s):
    sp = 0.5 * sprefac * s / np.pi**(1.0/3) * 2**(1.0/3)
    #return hprefac * 2.0 / 3 * sp / np.log(1.5 * sp)
    return hprefac * 2.0 * sp / (3 * np.log(sp)) * (1 - (1 + 1.5 * np.log(sp)) / sp**3 )
hex = jit(hex)

def hex2(s):
    sp = 0.5 * sprefac * s / np.pi**(1.0/3) * 2**(1.0/3)
    return hprefac * 2.0 / 3 * sp / np.arcsinh(0.5 * sp)
    #return hprefac * 2.0 * sp / (3 * np.arcsinh(sp-s0)) * (1 - (1 + 1.5 * np.log(sp)) / sp**3 )

def hex3(s):
    sp = 0.5 * sprefac * s / np.pi**(1.0/3) * 2**(1.0/3)
    mu = 0.21951
    l = 0.5 * sprefac / np.pi**(1.0/3) * 2**(1.0/3)
    b = (mu - l**2 * hprefac / 18) / l**2 / (4 * hprefac / 3 - 1)
    return (hprefac * 4 / 3 - 1) / (1 + sp**2)

def hex4(s):
    sp = 0.5 * sprefac * s / np.pi**(1.0/3) * 2**(1.0/3)
    hex0 = hex(s)
    mix = 1 / (1 + np.exp(20 * (sp - 1.05)))
    return hprefac * (1 + 0.004 * sp**2) * mix + hex0 * (1 - mix)

def hex5(s):
    sp = 0.5 * sprefac * s / np.pi**(1.0/3) * 2**(1.0/3)
    expansion = 2 + sp**2 / 12 - 17 * s**4 / 2880 + 367 * s**6 / 483840\
                - 27859 * s**8 / 232243200 + 1295803 * s**10 / 61312204800
    return hprefac * 2.0 /3 * expansion

def hexc(s):
    mu = 0.21951
    l = 0.5 * sprefac / np.pi**(1.0/3) * 2**(1.0/3)
    a = 1 - hprefac * 4 / 3
    b = mu - l**2 * hprefac / 18
    return (a + b * s**2) / (1 + (l*s/2)**4)

def nheg(s):
    mu = 0.21951
    return 1 + mu * s**2

s = np.linspace(s0 + 1e-3, 5, 10000)
y = hex(s)
s2 = np.linspace(0.0001, 5, 10000)
y2 = hex2(s2)
#y3 = hex2(s2) - hex3(s2)
y3 = hex2(s2) + hexc(s2)
y4 = hex4(s2)
y5 = hex5(s2) + hexc(s2)

def hexsum(s):
    return np.sum(hex(s))

dhex = grad(hexsum, 0)
dhexsum = lambda s: np.sum(dhex(s))
ddhex = grad(dhexsum, 0)

dy = dhex(s)
ddy = ddhex(s)

x0 = 0.1
y_target = hex(x0+0.3) - hprefac
dy_target = dhex(x0+0.3)
ddy_target = ddhex(x0+0.3)

"""
C = y_target / 2
aq, bq, cq = 2, -2 * dy_target / C, (dy_target / C)**2 - ddy_target / C
A = (-bq + np.sqrt(bq**2 - 4*aq*cq)) / (2 * aq)
B = dy_target / C - A
print(A, B, C)

yfit = hprefac + C * (np.exp(A * (s2-x0)) + np.exp(B * (s2-x0)))
print(yfit)
"""


import numpy
y6 = numpy.array(y5)
y6[s2 > 0.025] = y3[s2 > 0.025]

aarr = numpy.array([[x0**2, x0**4, x0**6],\
                [2*x0, 4*x0**3, 6*x0**5],\
                [2, 12*x0**2, 30*x0**4]])
barr = numpy.array([y_target, dy_target, ddy_target])
soln = numpy.linalg.solve(aarr, barr)
yfit = soln[0] * s2**2 + soln[1] * s2**4 + soln[2] * s2**6 + hprefac

aarr = numpy.array([[x0**4, x0**6, x0**8],\
                [4*x0**3, 6*x0**5, 8*x0**7],\
                [12*x0**2, 30*x0**4, 56*x0**6]])
barr = numpy.array([y_target, dy_target, ddy_target])
soln = numpy.linalg.solve(aarr, barr)
yfit = soln[0] * (s2-0.3)**4 + soln[1] * (s2-0.3)**6 + soln[2] * (s2-0.3)**8 + hprefac

print(np.min(yfit[s2 < s0] - hprefac))

print(soln)

print(hprefac)
print(hprefac * sprefac * 0.5 / np.pi**(1.0/3) * 2**(1.0/3))

plt.plot(s, y, label='exact')
#plt.plot(s, dy, label='dyds')
#plt.plot(s2[s2 < 0.6], yfit[s2 < 0.6], label='yfit')
#plt.plot(s, ddy, label='d2yds2')
#plt.plot(s2, y2, label='approx1')
#plt.plot(s2, y3, label='approx2')
plt.plot(s2, y5, label='approx5')
plt.plot(s2, tail_fx_direct(numpy.array(s2)), label='func')
plt.plot(s2, tail_fx_deriv_direct(numpy.array(s2)), label='deriv')
plt.plot(s2[s2<1], nheg(s2[s2<1]), label='NHEG')
#plt.plot(s2, y4, label='approx3')
plt.scatter(s0, hex(s0+1e-5))
plt.ylim(0,4)
plt.legend()
plt.show()

print(np.trapz(tail_fx_deriv_direct(numpy.array(s2[s2<4])), s2[s2<4]))
print(tail_fx_direct(numpy.array([4])))


x = np.linspace(0, 1, 500)
y = (x**2 + 3 * x**3) / (1 + x**3)**2
y = 0.5 - np.cos(2 * np.pi * x) / 2
#plt.plot(x,y)
plt.plot(x,1-(1-y)**2)
plt.plot(x,y**4)
y2 = x * (1-x) / np.max(x * (1-x))
y2 = 1 - (1-y2)**3
plt.plot(x, y2)
plt.show()

import numpy as np

def partition_chi(x):
    y2 = x * (1-x) / np.max(x * (1-x))
    y2 = 1 - (1-y2)**3
    y = 0.5 - np.cos(2 * np.pi * x) / 2
    p1 = y**4
    p2 = 1-(1-y)**2 - y**4
    p3 = y2 - (1-(1-y)**2)
    p4 = 1 - y2
    p5 = p4.copy()
    p6 = p3.copy()
    p7 = p2.copy()
    p2[x > 0.5] = 0
    p3[x > 0.5] = 0
    p4[x > 0.5] = 0
    p5[x < 0.5] = 0
    p6[x < 0.5] = 0
    p7[x < 0.5] = 0
    return p1, p2, p3, p4, p5, p6, p7

ps = partition_chi(numpy.array(x))
for i in range(7):
    plt.plot(x, ps[i])
plt.show()
