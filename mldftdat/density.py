import numpy as np
from mldftdat.pyscf_utils import *

LDA_FACTOR = - 3.0 / 4.0 * (3.0 / np.pi)**(1.0/3)

def ldax(n):
    return LDA_FACTOR * n**(4.0/3)

def ldaxp(n):
    return 0.5 * ldax(2 * n)

def lsda(nu, nd):
    return ldaxp(nu) + ldaxp(nd)

# check this all makes sense with scaling

def get_x_nonlocal_descriptors_nsp(rho_data, tau_data, coords, weights):
    # calc ws_radii for single spin (1/n is factor of 2 larger)
    ws_radii = get_ws_radii(rho_data[0]) * 2**(1.0/3)
    nonlocal_data = get_nonlocal_data(rho_data, tau_data, ws_radii, coords, weights)
    if np.isnan(nonlocal_data).any():
        raise ValueError('Part of nonlocal_data is nan %d' % np.count_nonzero(np.isnan(nonlocal_data)))
    # note: ws_radii calculated in the regularization call does not have the
    # factor of 2^(1/3), but it only comes in linearly so it should be fine
    res = get_regularized_nonlocal_data(nonlocal_data, rho_data)
    if np.isnan(res).any():
        raise ValueError('Part of regularized result is nan %d' % np.count_nonzero(np.isnan(res)))
    return res

def get_exchange_descriptors(rho_data, tau_data, coords,
                             weights, restricted = True):
    if restricted:
        lc = get_dft_input(rho_data)[:3]
        nlc = get_x_nonlocal_descriptors_nsp(rho_data, tau_data,
                                                coords, weights)
        return np.append(lc, nlc, axis=0)
    else:
        lcu = get_dft_input(rho_data[0] * 2)[:3]
        nlcu = get_x_nonlocal_descriptors_nsp(rho_data[0] * 2,
                                                tau_data[0] * 2,
                                                coords, weights)
        lcd = get_dft_input(rho_data[1] * 2)[:3]
        nlcd = get_x_nonlocal_descriptors_nsp(rho_data[1] * 2,
                                                tau_data[1] * 2,
                                                coords, weights)
    return np.append(lcu, nlcu, axis=0),\
           np.append(lcd, nlcd, axis=0)

"""
The following two routines are from
J. Tao, J. Chem. Phys. 115, 3519 (2001) (doi: 10.1063/1.1388047)
"""

A = 0.704 # maybe replace with sqrt(6/5)?
B = 2 * np.pi / 9 * np.sqrt(6.0/5)
FXP0 = 27 / 50 * 10 / 81
FXI = 1.0 / 3 * (4*np.pi**2 / 3)**(1.0/3)
#MU = 10/81
MU = 0.21
C1 = 1.0 / 3 * (4*np.pi**2 / 3)**(1.0/3)
C2 = 1 - C1
C3 = 0.19697 * np.sqrt(0.704)
C4 = (C3**2 - 0.09834 * MU) / C3**3

def edmgga(rho_data):
    print(rho_data.shape)
    gradn = np.linalg.norm(rho_data[1:4], axis=0)
    tau0 = 3 / 10 * 3 * np.pi**2 * rho_data[0]**(5/3) + 1e-6
    tauw = 1 / 8 * gradn**2 / (rho_data[0] + 1e-6)
    QB = tau0 - rho_data[5] + tauw + 0.25 * rho_data[4]
    QB /= tau0
    x = A * QB + np.sqrt(1 + (A*QB)**2)
    FX = C1 + (C2 * x) / (1 + C3 * np.sqrt(x) * np.arcsinh(C4 * (x-1)))
    return FX

def edmgga_loc(rho_data):
    print(rho_data.shape)
    gradn = np.linalg.norm(rho_data[1:4], axis=0)
    tau0 = 3 / 10 * 3 * np.pi**2 * rho_data[0]**(5/3) + 1e-6
    tauw = 1 / 8 * gradn**2 / (rho_data[0] + 1e-6)
    QB = tau0 - rho_data[5] + 0.125 * rho_data[4]
    QB /= tau0
    x = A * QB + np.sqrt(1 + (A*QB)**2)
    FX = C1 + (C2 * x) / (1 + C3 * np.sqrt(x) * np.arcsinh(C4 * (x-1)))
    return FX
