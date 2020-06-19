import numpy as np 
import matplotlib.pyplot as plt 

zs, wts = np.polynomial.legendre.leggauss(5)
NUMPHI = 6
phis = np.linspace(0, 2*np.pi, num=NUMPHI, endpoint=False)
dphi = 2 * np.pi / NUMPHI
dphi_lst = dphi * np.ones(phis.shape)

rs = np.linspace(0.001, 6, 1000)
rsw0 = np.append([0], rs)
drs = rsw0[1:] - rsw0[:-1]
rwts = rs**2 * drs

points = np.vstack(np.meshgrid(rs, zs, phis, indexing='ij')).reshape(3,-1)
weights3d = np.vstack(np.meshgrid(rwts, wts, dphi_lst, indexing='ij')).reshape(3,-1)
rhos = weights3d[1]
weights = np.cumprod(weights3d, axis=0)[-1,:]
xs = points[0] * np.sqrt(1-points[1]**2) * np.cos(points[2])
ys = points[0] * np.sqrt(1-points[1]**2) * np.sin(points[2])
zs = points[0] * points[1]
coords = np.array([xs, ys, zs]).T

rs = np.linalg.norm(coords, axis=1)
func = (1 / (4*np.pi)) * np.exp(-rs**2)

plt.scatter(rs, func)
plt.show()
print(np.dot(func, weights))
print(np.sqrt(np.pi)/4)

"""
print(points, weights3d)
print(coords, wts)
print(rs)
print(xs**2+ys**2)
print(np.sqrt(xs**2+ys**2+zs**2))
"""