"""
Helper functions for analyzing data.
FUNCTIONS USED FOR PROJECT ANALYSIS:
analyze_rad3 was used for the energy convergence plots.
get_ar2_plot was used for the dissociation curve plots.
"""

import yaml
import numpy as np
import matplotlib.pyplot as plt

def load_data(functional, radi_method, rad, ang, prune=True):
    prune = 'nwchem' if prune else 'noprune'
    with open('gridbench_{}_{}_{}_{}_{}.yaml'.format(
        functional, prune, radi_method, rad, ang), 'r') as f:
        d = yaml.load(f, Loader=yaml.Loader)
    return d

def load_data_norm(functional, radi_method, rad, ang, prune=True):
    prune = 'nwchem' if prune else 'noprune'
    with open('gridbench_{}_{}_{}_{}_{}_norm.yaml'.format(
        functional, prune, radi_method, rad, ang), 'r') as f:
        d = yaml.load(f, Loader=yaml.Loader)
    return d

def get_ar2_plot(d):
    """
    Takes a dictionary d as an arugment, which is the output
    of a GridBenchmark task (see pyscf_tasks.py).
    Returns a numpy array of interatomic distances in Angstrom
    and energies in Hartree atomic units.
    """
    dists = []
    ens = []
    for k in d.keys():
        if isinstance(k, int):
            dists.append(k/100.)
            ens.append(d[k]['e_tot'])
    return np.array(dists), np.array(ens)

def water_ae(d):
    return -d['H2O']['e_tot'] + d['O']['e_tot'] + 2 * d['H']['e_tot']

def no_ae(d):
    return -d['NO']['e_tot'] + d['O']['e_tot'] + d['N']['e_tot']

def sf6_ae(d):
    return -d['SF6']['e_tot'] + d['S']['e_tot'] + 6 * d['F']['e_tot']

def analyze_level(functional, ang, prune=True):
    for radi_method in [0,1,2,3,4]:
        ds = {}
        lvls = [(35, 86), (50, 194), (60, 302), (75, 302), (99, 590)]
        for i in range(5):
            rad, ang = lvls[i]
            ds[i] = load_data(functional, radi_method, rad, ang, prune)
        mses = []
        for i in range(4):
            mse = 0
            count = 0
            #for k in ['H', 'O', 'N', 'H2O', 'S', 'F', 'SF6', 'Ar', 'NO']:
            for k in ['H2O', 'SF6', 'Ar', 'NO']:
                mse += (ds[i][k]['e_tot'] - ds[4][k]['e_tot'])**2
                count += 1
            mse = np.sqrt(mse / count)
            mses.append(mse)
        print(radi_method, mses)
        mses = []
        for i in range(4):
            mse = 0
            count = 4
            mse += (water_ae(ds[i]) - water_ae(ds[4]))**2
            mse += (no_ae(ds[i]) - no_ae(ds[4]))**2
            mse += (sf6_ae(ds[i]) - sf6_ae(ds[4]))**2
            mse += (ds[i]['Ar']['e_tot'] - ds[4]['Ar']['e_tot'])**2
            mse = np.sqrt(mse / count)
            mses.append(mse)
        print(radi_method, mses)

def analyze_level(functional, radi_method, prune=True, num=5):
    ds = {}
    lvls = [(35, 86), (50, 194), (75, 302), (99,590), (250,974)][:num]
    maxr = lvls[-1][0]
    ds = {}
    for rad, ang in lvls:
        ds[rad] = load_data(functional, radi_method, rad, ang, prune)
    mses = []
    for rad, ang in lvls[:-1]:
        mse = 0
        count = 0
        atoms = ['H', 'O', 'N', 'S', 'F', 'Ar']
        aw = 1 / np.array([1, 2, 2, 3, 2, 3])
        for i, k in enumerate(atoms):
            mse += (ds[rad][k]['e_tot'] - ds[maxr][k]['e_tot'])**2 * aw[i]
        mols = ['H2O', 'SF6', 'NO']
        mw = 1 / np.array([2, 6, 1])
        for i, k in enumerate(mols):
            mse += (ds[rad][k]['e_tot'] - ds[maxr][k]['e_tot'])**2 * mw[i]
            count += 1
        #mse += (water_ae(ds[rad]) - water_ae(ds[maxr]))**2 / 2
        #mse += (no_ae(ds[rad]) - no_ae(ds[maxr]))**2
        #mse += (sf6_ae(ds[rad]) - sf6_ae(ds[maxr]))**2 / 6
        tw = np.sum(aw) + np.sum(mw)
        mse = np.sqrt(mse / tw)
        mses.append(mse)
    return mses

def analyze_rad(functional, ang, prune=True):
    for radi_method in [0,1,2,3,4]:
        ds = {}
        for rad in [35,50,60,75,99]:
            ds[rad] = load_data(functional, radi_method, rad, ang, prune)
        mses = []
        for rad in [35,50,60,75]:
            mse = 0
            count = 0
            #for k in ['H', 'O', 'N', 'H2O', 'S', 'F', 'SF6', 'Ar', 'NO']:
            for k in ['H2O', 'SF6', 'Ar']:#, 'NO']:
                mse += (ds[rad][k]['e_tot'] - ds[99][k]['e_tot'])**2
                count += 1
            mse = np.sqrt(mse / count)
            mses.append(mse)
        print(radi_method, mses)
        mses = []
        for rad in [35,50,60,75]:
            mse = 0
            count = 4
            mse += (water_ae(ds[rad]) - water_ae(ds[99]))**2
            mse += (no_ae(ds[rad]) - no_ae(ds[99]))**2
            mse += (sf6_ae(ds[rad]) - sf6_ae(ds[99]))**2
            mse += (ds[rad]['Ar']['e_tot'] - ds[99]['Ar']['e_tot'])**2
            mse = np.sqrt(mse / count)
            mses.append(mse)
        print(radi_method, mses)

radi_methods = ['GC2', 'TGC2', 'DE2', 'CC', 'GL', 'GJ']
def analyze_rad2(functional, radi_method, ang, prune=True):
    ds = {}
    for rad in [35,50,60,75,99]:
        ds[rad] = load_data(functional, radi_method, rad, ang, prune)
    mses = []
    for rad in [35,50,60,75]:
        mse = 0
        count = 0
        atoms = ['H', 'O', 'N', 'S', 'F', 'Ar']
        aw = 1 / np.array([1, 2, 2, 3, 2, 3])
        for i, k in enumerate(atoms):
            mse += (ds[rad][k]['e_tot'] - ds[99][k]['e_tot'])**2 * aw[i]
        mols = ['H2O', 'SF6', 'NO']
        mw = 1 / np.array([2, 6, 1])
        for i, k in enumerate(mols):
            mse += (ds[rad][k]['e_tot'] - ds[99][k]['e_tot'])**2 * mw[i]
            count += 1
        #mse += (water_ae(ds[rad]) - water_ae(ds[99]))**2 / 2
        #mse += (no_ae(ds[rad]) - no_ae(ds[99]))**2
        #mse += (sf6_ae(ds[rad]) - sf6_ae(ds[99]))**2 / 6
        tw = np.sum(aw) + np.sum(mw)
        mse = np.sqrt(mse / tw)
        mses.append(mse)
    return mses

def analyze_rad3(functional, radi_method, ang, prune=True):
    """
    Return the RMSE scores of a quadrature scheme for a given functional
    Args:
        functional (str): XC functional name
        radi_method (int): Radial scheme code (see radi_methods list above)
        ang (int): Number of angular grid points
        prune (bool, True): Whether grid pruning was used.
    """
    ds = {}
    refd = load_data(functional, 0, 250, 974, False)
    for rad in [35,50,60,75,99]:
        ds[rad] = load_data(functional, radi_method, rad, ang, prune)
    mses = []
    for rad in [35,50,60,75,99]:
        mse = 0
        count = 0
        atoms = ['H', 'O', 'N', 'S', 'F', 'Ar']
        aw = 1 / np.array([1, 2, 2, 3, 2, 3])
        for i, k in enumerate(atoms):
            mse += (ds[rad][k]['e_tot'] - refd[k]['e_tot'])**2 * aw[i]
        mols = ['H2O', 'SF6', 'NO']
        mw = 1 / np.array([2, 6, 1])
        for i, k in enumerate(mols):
            mse += (ds[rad][k]['e_tot'] - refd[k]['e_tot'])**2 * mw[i]
            count += 1
        tw = np.sum(aw) + np.sum(mw)
        mse = np.sqrt(mse / tw)
        mses.append(mse)
    return mses

def analyze_radn(functional, radi_method, ang, prune=True):
    """
    Return the RMSE scores of a quadrature scheme for a given functional
    Args:
        functional (str): XC functional name
        radi_method (int): Radial scheme code (see radi_methods list above)
        ang (int): Number of angular grid points
        prune (bool, True): Whether grid pruning was used.
    """
    ds = {}
    refd = load_data(functional, 0, 250, 974, False)
    for rad in [35,50,60,75,99]:
        ds[rad] = load_data_norm(functional, radi_method, rad, ang, prune)
    mses = []
    for rad in [35,50,60,75,99]:
        mse = 0
        count = 0
        atoms = ['H', 'O', 'N', 'S', 'F', 'Ar']
        aw = 1 / np.array([1, 2, 2, 3, 2, 3])
        for i, k in enumerate(atoms):
            mse += (ds[rad][k]['e_tot'] - refd[k]['e_tot'])**2 * aw[i]
        mols = ['H2O', 'SF6', 'NO']
        mw = 1 / np.array([2, 6, 1])
        for i, k in enumerate(mols):
            mse += (ds[rad][k]['e_tot'] - refd[k]['e_tot'])**2 * mw[i]
            count += 1
        tw = np.sum(aw) + np.sum(mw)
        mse = np.sqrt(mse / tw)
        mses.append(mse)
    return mses

def analyze_ang(functional, radi_method, rad, prune=True):
    ds = {}
    for ang in [86, 194, 302, 590]:
        ds[ang] = load_data(functional, radi_method, rad, ang, prune)
    mses = []
    for ang in [86, 194, 302]:
        mse = 0
        count = 0
        #for k in ['H', 'O', 'N', 'H2O', 'S', 'F', 'SF6', 'Ar', 'NO']:
        for k in ['H2O', 'SF6', 'Ar', 'NO']:
            mse += (ds[ang][k]['e_tot'] - ds[590][k]['e_tot'])**2
            count += 1
        mse = np.sqrt(mse / count)
        mses.append(mse)
    print(radi_method, mses)
    mses = []
    for ang in [86, 194, 302]:
        mse = 0
        count = 4
        mse += (water_ae(ds[ang]) - water_ae(ds[590]))**2
        mse += (no_ae(ds[ang]) - no_ae(ds[590]))**2
        mse += (sf6_ae(ds[ang]) - sf6_ae(ds[590]))**2
        mse += (ds[ang]['Ar']['e_tot'] - ds[590]['Ar']['e_tot'])**2
        mse = np.sqrt(mse / count)
        mses.append(mse)
    print(radi_method, mses)
