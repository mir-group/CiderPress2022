import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
import numpy as np 

def plot_data_atom(mol, coords, values, value_name, rmax, units,
                   ax=None):
    mol.build()
    rs = np.linalg.norm(coords, axis=1)
    if ax is None:
        plt.scatter(rs, values, label=value_name)
        plt.xlim(0, rmax)
        plt.xlabel('$r$ (Bohr radii)')
        plt.ylabel(units)
        plt.legend()
        plt.title(mol._atom[0][0])
    else:
        ax.scatter(rs, values, label=value_name)
        ax.set_xlim(0, rmax)
        ax.set_xlabel('$r$ (Bohr radii)')
        ax.set_ylabel(units)
        ax.legend()
        #plt.title(mol._atom[0][0])

def plot_data_diatomic(mol, coords, values, value_name, units, bounds,
                        ax = None):
    mol.build()
    diff = np.array(mol._atom[1][1]) - np.array(mol._atom[0][1])
    direction = diff / np.linalg.norm(diff)
    zs = np.dot(coords, direction)
    print(zs.shape, values.shape)
    if ax is None:
        plt.scatter(zs, values, label=value_name)
        plt.xlabel('$z$ (Bohr radii)')
        plt.ylabel(units)
        plt.xlim(bounds[0], bounds[1])
        plt.legend()
    else:
        ax.scatter(zs, values, label=value_name)
        ax.set_xlabel('$z$ (Bohr radii)')
        ax.set_ylabel(units)
        ax.set_xlim(bounds[0], bounds[1])
        ax.legend()
    if mol._atom[0][0] == mol._atom[1][0]:
        title = '{}$_2$'.format(mol._atom[0][0])
    else:
        title = mol._atom[0][0] + mol._atom[1][0]
    #plt.title(title)

def plot_surface_diatomic(mol, zs, rs, values, value_name, units,
                            bounds, scales = None):
    condition = np.logical_and(rs < bounds[2],
                np.logical_and(zs > bounds[0], zs < bounds[1]))
    rs = rs[condition]
    zs = zs[condition]
    values = values[condition]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    print(zs.shape, rs.shape, values.shape)
    ax.scatter(zs, rs, values)
    print(scales)
    ax.set_title('Surface plot')

def quick_plot(rho, v_true, v_pred, name = None):
    """
    Plot true and predicted values against charge density
    """
    plt.scatter(rho, v_true, label='true')
    plt.scatter(rho, v_pred, label='predicted')
    plt.xlabel('density')
    plt.legend()
    if name is None:
        plt.show()
    else:
        plt.title(name)
        plt.savefig(name)
        plt.cla()
