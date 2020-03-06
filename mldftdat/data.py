import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d

def get_unique_coord_indexes_spherical(coords):
    rs = np.linalg.norm(coords, axis=1)
    unique_rs = np.array([])
    indexes = []
    for i, r in enumerate(rs):
        if (np.abs(unique_rs - r) > 1e-7).all():
            unique_rs = np.append(unique_rs, [r], axis=0)
            indexes.append(i)
    return indexes

def plot_data_atom(mol, coords, values, value_name, rmax, units):
    mol.build()
    rs = np.linalg.norm(coords, axis=1)
    plt.scatter(rs, values, label=value_name)
    plt.xlim(0, rmax)
    plt.xlabel('$r$ (Bohr radii)')
    plt.ylabel(units)
    plt.legend()
    plt.title(mol._atom[0][0])

def get_zr_diatomic(mol, coords):
    mol.build()
    diff = np.array(mol._atom[1][1]) - np.array(mol._atom[0][1])
    direction = diff / np.linalg.norm(diff)
    zs = np.dot(coords, direction)
    zvecs = np.outer(zs, direction)
    print(zvecs.shape)
    rs = np.linalg.norm(coords - zvecs, axis=1)
    return zs, rs

def plot_data_diatomic(mol, coords, values, value_name, units, bounds):
    mol.build()
    diff = np.array(mol._atom[1][1]) - np.array(mol._atom[0][1])
    direction = diff / np.linalg.norm(diff)
    zs = np.dot(coords, direction)
    print(zs.shape, values.shape)
    plt.scatter(zs, values, label=value_name)
    plt.xlabel('$z$ (Bohr radii)')
    plt.ylabel(units)
    plt.xlim(bounds[0], bounds[1])
    plt.legend()
    if mol._atom[0][0] == mol._atom[1][0]:
        title = '{}$_2$'.format(mol._atom[0][0])
    else:
        title = mol._atom[0][0] + mol._atom[1][0]
    plt.title(title)

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
