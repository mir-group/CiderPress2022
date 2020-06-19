import numpy as np 
from scipy.linalg import eigh
from numpy.linalg import norm

a = np.random.rand(3,3)
a = (a + a.T) / 2
b = [[1, 0.2, 0.1], [0.2, 1, 0], [0.1, 0, 2]]
w, v = eigh(a, b)
print(w, v)
binv = np.linalg.inv(b)
binva = np.dot(binv, a)
print(np.dot(v.transpose(), np.dot(binv, v)))
print(np.dot(v.transpose(), np.dot(b, v)))
print(np.dot(v.transpose(), np.dot(a, v)), w)
vec = np.dot(binva, v[:,0])
print(vec / norm(vec), v[:,0] / norm(v[:,0]))
print(w[0] * np.dot(b, v[:,0]), np.dot(a, v[:,0]))
