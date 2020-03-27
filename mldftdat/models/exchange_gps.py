from mldftdat.gp import DFTGPR
from mldftdat.density import edmgga
from mldftdat.data import *
import numpy as np
from pyscf.dft.libxc import eval_xc

def xed_to_y_edmgga(xed, rho_data):
    y = xed / (ldax(rho_data[0]) - 1e-7)
    return y - edmgga(rho_data)

def y_to_xed_edmgga(y, rho_data):
    fx = np.array(y) + edmgga(rho_data)
    return fx * ldax(rho_data[0])

def xed_to_y_pbe(xed, rho_data):
    pbex = eval_xc('PBE,', rho_data)[0] * rho_data[0]
    return (xed - pbex) / (ldax(rho_data[0]) - 1e-7)

def y_to_xed_pbe(y, rho_data):
    yp = y * ldax(rho_data[0])
    pbex = eval_xc('PBE,', rho_data)[0] * rho_data[0]
    return yp + pbex

"""
def get_edmgga_descriptors(X, rho_data, num=1):
    gradn = np.linalg.norm(rho_data[1:4], axis=0)
    tau0 = 3 / 10 * 3 * np.pi**2 * rho_data[0]**(5/3) + 1e-6
    tauw = 1 / 8 * gradn**2 / (rho_data[0] + 1e-6)
    QB = tau0 - rho_data[5] + tauw + 0.25 * rho_data[4]
    QB /= tau0
    x = A * QB + np.sqrt(1 + (A*QB)**2)
    x = np.arcsinh(x - 1)
    X = get_gp_x_descriptors(X, num = num)
    np.append()
"""

class PBEGPR(DFTGPR):

    def __init__(self, num_desc, init_kernel = None, use_algpr = False):
        super(PBEGPR, self).__init__(num_desc, descriptor_getter = None,
                       xed_y_converter = (xed_to_y_pbe, y_to_xed_pbe),
                       init_kernel = init_kernel, use_algpr = use_algpr)


"""
class EDMGPR(DFTGPR):

    def __init__(num_desc, init_kernel = None, use_algpr = False):
        super.__init__()
"""


