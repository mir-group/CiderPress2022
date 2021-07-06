import numpy as np 
from mldftdat.workflow_utils import get_save_dir, SAVE_ROOT
from mldftdat.density import get_exchange_descriptors2, LDA_FACTOR,\
                             get_ldax, get_ldax_dens
import os, json, yaml
from sklearn.metrics import r2_score
from pyscf.dft.libxc import eval_xc
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from mldftdat.lowmem_analyzers import RHFAnalyzer, UHFAnalyzer
from mldftdat.pyscf_utils import transform_basis_1e, run_scf
from pyscf.dft.numint import eval_ao, eval_rho
from pyscf.scf.stability import uhf_internal
from pyscf import gto
from collections import Counter
from ase.data import chemical_symbols, atomic_numbers, ground_state_magnetic_moments
from ase import Atoms
from ase.geometry.analysis import Analysis as ASEAnalysis



###############################################################################
# Dataset loading and cleaning utilities for compiling datasets and training. #
###############################################################################


def get_unique_coord_indexes_spherical(coords):
    rs = np.linalg.norm(coords, axis=1)
    unique_rs = np.array([])
    indexes = []
    for i, r in enumerate(rs):
        if (np.abs(unique_rs - r) > 1e-7).all():
            unique_rs = np.append(unique_rs, [r], axis=0)
            indexes.append(i)
    return indexes

def density_similarity_atom(rho1, rho2, grid, mol, exponent=1, inner_r=0.2):
    class PGrid():
        def __init__(self, coords, weights):
            self.coords = coords
            self.weights = weights
    rs = np.linalg.norm(get_unique_coord_indexes_spherical(grid.coords), axis=1)
    rs.sort()
    weights = 0 * rs
    vals1 = np.zeros(rho1.shape)
    vals2 = np.zeros(rho2.shape)
    all_rs = np.linalg.norm(grid.coords, axis=1)
    for i, r in enumerate(all_rs):
        j = np.argmin(np.abs(r - rs))
        weights[j] += grid.weights[i]
        vals1[...,j] += rho1[...,i] * grid.weights[i]
        vals2[...,j] += rho2[...,i] * grid.weights[i]
    vals1 /= weights
    vals2 /= weights
    weights[rs < inner_r] = 0
    diff = np.abs(vals1 - vals2)**exponent
    return np.dot(diff, weights)**(1.0/exponent)

def density_similarity(rho1, rho2, grid, mol, exponent=1, inner_r=0.2):
    weights = grid.weights.copy()
    for atom in mol._atom:
        coord = np.array(atom[1])
        rel_coords = grid.coords - coord
        rel_r = np.linalg.norm(rel_coords, axis = 1)
        weights[rel_r < inner_r] = 0
    diff = np.abs(rho1 - rho2)**exponent
    return np.dot(diff, weights)**(1.0/exponent)

def rho_data_from_calc(calc, grid, is_ccsd=False):
    ao = eval_ao(calc.mol, grid.coords, deriv=2)
    dm = calc.make_rdm1()
    if is_ccsd:
        if len(dm.shape) == 3:
            trans_mo_coeff = np.transpose(calc.mo_coeff, axes=(0,2,1))
        else:
            trans_mo_coeff = calc.mo_coeff.T
        dm = transform_basis_1e(dm, trans_mo_coeff)
    print ('NORMALIZATION', np.trace(dm.dot(calc.mol.get_ovlp())))
    rho = eval_rho(calc.mol, ao, dm, xctype='MGGA')
    return rho

def get_zr_diatomic(mol, coords):
    mol.build()
    diff = np.array(mol._atom[1][1]) - np.array(mol._atom[0][1])
    direction = diff / np.linalg.norm(diff)
    zs = np.dot(coords, direction)
    zvecs = np.outer(zs, direction)
    print(zvecs.shape)
    rs = np.linalg.norm(coords - zvecs, axis=1)
    return zs, rs

def load_descriptors(dirname, count=None, val_dirname=None, load_wt=False,
                     binary=True):
    if binary:
        X = np.load(os.path.join(dirname, 'desc.npy')).transpose()
    else:
        X = np.loadtxt(os.path.join(dirname, 'desc.npz')).transpose()
    if count is not None:
        X = X[:count]
    else:
        count = X.shape[0]
    if val_dirname is None:
        val_dirname = dirname
    if binary:
        y = np.load(os.path.join(val_dirname, 'val.npy'))[:count]
        rho_data = np.load(os.path.join(dirname, 'rho.npy'))[:,:count]
        if load_wt:
            wt = np.load(os.path.join(dirname, 'wt.npy'))[:count]
            return X, y, rho_data, wt
    else:
        y = np.loadtxt(os.path.join(val_dirname, 'val.npz'))[:count]
        rho_data = np.loadtxt(os.path.join(dirname, 'rho.npz'))[:,:count]
        if load_wt:
            wt = np.loadtxt(os.path.join(dirname, 'wt.npz'))[:count]
            return X, y, rho_data, wt
    return X, y, rho_data

def filter_descriptors(X, y, rho_data, tol=1e-3, wt=None):
    if rho_data.ndim == 3:
        condition = np.sum(rho_data[:,0,:], axis=0) > tol
    else:
        condition = rho_data[0] > tol
    X = X[...,condition,:]
    y = y[...,condition]
    rho = rho_data[...,0,condition]
    rho_data = rho_data[...,:,condition]
    if wt is not None:
        wt = wt[condition]
        return X, y, rho, rho_data, wt
    return X, y, rho, rho_data

def get_descriptors(dirname, num=1, count=None, tol=1e-3):
    """
    Get exchange energy descriptors from the dataset directory.
    Returns a number of descriptors per point equal
    to num.

    Order info:
        0,   1, 2,     3,     4,      5,       6,       7
        rho, s, alpha, |dvh|, intdvh, intdrho, intdtau, intrho
        need to regularize 4, 6, 7
    """
    X, y, rho_data = load_descriptors(dirname, count)
    rho = rho_data[0]

    X = get_gp_x_descriptors(X, num=num)
    y = get_y_from_xed(y, rho)

    return filter_descriptors(X, y, rho_data, tol)

def true_metric(y_true, y_pred, rho):
    """
    Find relative and absolute mse, as well as r2
    score, for the exchange energy per particle (epsilon_x)
    from the true and predicted enhancement factor
    y_true and y_pred.
    """
    res_true = get_x(y_true, rho)
    res_pred = get_x(y_pred, rho)
    return np.sqrt(np.mean(((res_true - res_pred) / (1))**2)),\
           np.sqrt(np.mean(((res_true - res_pred) / (res_true + 1e-7))**2)),\
           score(res_true, res_pred)

def score(y_true, y_pred):
    """
    r2 score
    """
    #y_mean = np.mean(y_true)
    #return 1 - ((y_pred-y_true)**2).sum() / ((y_pred-y_mean)**2).sum()
    return r2_score(y_true, y_pred)



###################################################################
# Used in the make_error_table scripts to test exchange energies. #
###################################################################


def predict_exchange(analyzer, model=None, restricted=True, return_desc=False):
    """
    model:  If None, return exact exchange results
            If str, evaluate the exchange energy of that functional.
            Otherwise, assume sklearn model and run predict function.
    """
    from mldftdat.dft.xc_models import MLFunctional
    if not restricted:
        raise NotImplementedError('unrestricted case not available for this function yet')
    rho_data = analyzer.rho_data
    tau_data = analyzer.tau_data
    coords = analyzer.grid.coords
    weights = analyzer.grid.weights
    rho = rho_data[0,:]
    if model is None:
        neps = analyzer.get_fx_energy_density()
        eps = neps / (rho + 1e-7)
    elif model == 'EDM':
        fx = edmgga(rho_data)
        neps = fx * get_ldax_dens(rho)
        eps = fx * get_ldax(rho)
    elif type(model) == str:
        eps = eval_xc(model + ',', rho_data)[0]
        neps = eps * rho
    elif hasattr(model, 'coeff_sets'):
        xdesc = get_exchange_descriptors2(analyzer, restricted=restricted,
                                          version=model.desc_version,
                                          a0=model.a0,
                                          fac_mul=model.fac_mul,
                                          amin=model.amin)
        neps = model.predict(xdesc.transpose(), vec_eval=False)
        eps = neps / rho
        if return_desc:
            X = model.get_descriptors(xdesc.transpose(), rho_data, num = model.num)
    elif isinstance(model, MLFunctional):
        N = analyzer.grid.weights.shape[0]
        desc  = np.zeros((N, len(model.desc_list)))
        ddesc = np.zeros((N, len(model.desc_list)))
        xdesc = get_exchange_descriptors2(analyzer, restricted=restricted,
                                          version=model.desc_version)
        for i, d in enumerate(model.desc_list):
            desc[:,i], ddesc[:,i] = d.transform_descriptor(xdesc, deriv = 1)
        xef = model.get_F(desc)
        eps = LDA_FACTOR * xef * analyzer.rho_data[0]**(1.0/3)
        neps = LDA_FACTOR * xef * analyzer.rho_data[0]**(4.0/3)
    else:
        from pyscf import lib
        xdesc = get_exchange_descriptors2(analyzer, restricted=restricted,
                                          version=model.desc_version,
                                          a0=model.a0,
                                          fac_mul=model.fac_mul,
                                          amin=model.amin)
        gridsize = xdesc.shape[1]
        neps, std = np.zeros(gridsize), np.zeros(gridsize)
        blksize = 20000
        for p0, p1 in lib.prange(0, gridsize, blksize):
            #neps[p0:p1], std[p0:p1] = model.predict(xdesc.T[p0:p1], return_std=True)
            neps[p0:p1] = model.predict(xdesc.T[p0:p1], return_std=False)
        #print('integrated uncertainty', np.dot(np.abs(std), np.abs(weights)))
        eps = neps / rho
        if return_desc:
            X = model.get_descriptors(xdesc.transpose())
    xef = neps / (get_ldax_dens(rho) - 1e-7)
    #fx_total = np.dot(neps, weights)
    fx_total = np.dot(neps[rho>1e-6], weights[rho>1e-6])
    if return_desc:
        return xef, eps, neps, fx_total, X
    else:
        return xef, eps, neps, fx_total

def predict_total_exchange_unrestricted(analyzer, model=None):
    if isinstance(analyzer, RHFAnalyzer):
        return predict_exchange(analyzer, model)[3]
    from mldftdat.dft.xc_models import MLFunctional
    rho_data = analyzer.rho_data
    tau_data = analyzer.tau_data
    coords = analyzer.grid.coords
    weights = analyzer.grid.weights
    rho = rho_data[:,0,:]
    if model is None:
        neps = analyzer.get_fx_energy_density()
    elif model == 'EDM':
        fxu = edmgga(2 * rho_data[0])
        fxd = edmgga(2 * rho_data[1])
        neps = 0.5 * fxu * get_ldax_dens(2 * rho[0]) \
             + 0.5 * fxd * get_ldax_dens(2 * rho[1])
    elif type(model) == str:
        eps = eval_xc(model + ',', rho_data, spin=analyzer.mol.spin)[0]
        neps = eps * (rho[0] + rho[1])
    else:
        xdescu, xdescd = get_exchange_descriptors2(analyzer, restricted=False,
                                                   version=model.desc_version,
                                                   a0=model.a0,
                                                   fac_mul=model.fac_mul,
                                                   amin=model.amin)
        neps = 0.5 * model.predict(xdescu.transpose())
        neps[rho[0]<=1e-6] = 0
        nepsd = 0.5 * model.predict(xdescd.transpose())
        nepsd[rho[1]<=1e-6] = 0
        neps = neps + nepsd
    fx_total = np.dot(neps, weights)
    return fx_total

def predict_correlation(analyzer, model=None, num=1,
                        restricted=True, version='a'):
    """
    model:  If None, return ccsd correlation energy (density)
            If str, evaluate the correlation energy of that functional.
            Otherwise, assume sklearn model and run predict function.
    """
    from mldftdat.dft.xc_models import MLFunctional
    from mldftdat.models.correlation_gps import CorrGPR
    rho_data = analyzer.rho_data
    tau_data = analyzer.tau_data
    coords = analyzer.grid.coords
    weights = analyzer.grid.weights
    if restricted:
        rho = rho_data[0,:]
    else:
        rho = rho_data[0,0,:] + rho_data[1,0,:]

    if model is None:
        neps = analyzer.get_corr_energy_density()
    elif type(model) == str:
        eps = eval_xc(',' + model, rho_data, spin = 0 if restricted else 1)[0]
        neps = eps * rho
    elif isinstance(model, CorrGPR):
        from pyscf import lib
        xdesc = get_exchange_descriptors2(analyzer, restricted = restricted, version = version)
        if restricted:
            gridsize = xdesc.shape[1]
        else:
            gridsize = xdesc[0].shape[1]
        neps = np.zeros(gridsize)
        blksize = 20000
        for p0, p1 in lib.prange(0, gridsize, blksize):
            if restricted:
                neps[p0:p1] = model.predict(xdesc.T[p0:p1],
                                             rho_data[:,p0:p1] / 2)
            else:
                neps[p0:p1] = model.predict(np.stack((xdesc[0].T[p0:p1],\
                                            xdesc[1].T[p0:p1])),
                                            rho_data[:,:,p0:p1])
    else:
        xdesc = get_exchange_descriptors2(analyzer, restricted = restricted, version = version)
        if restricted:
            neps = model.predict(xdesc.T, rho_data / 2)
        else:
            neps = model.predict(np.stack((xdesc[0].T, xdesc[1].T)), rho_data)
    eps = neps / (rho + 1e-20)
    fx_total = np.dot(neps, weights)
    return eps, neps, fx_total



################################################################
# Tools for reading and data crunching from ACCDB or CIDER DB. #
################################################################


def get_accdb_formula_entry(entry_names, fname):
    with open(fname, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            lines[i] = line.split(',')
        formulas = {}
        for line in lines:
            counts = line[1:-1:2]
            structs = line[2:-1:2]
            energy = float(line[-1])
            counts = [int(c) for c in counts]
            formulas[line[0]] = {'structs': structs, 'counts': counts, 'energy': energy}
    if type(entry_names) == str:
        return formulas[entry_names]
    res = {}
    for name in entry_names:
        res[name] = formulas[name]
    return res

def get_run_total_energy(dirname):
    with open(os.path.join(dirname, 'run_info.json'), 'r') as f:
        data = json.load(f)
    return data['e_tot']

def get_nbond(atom):
    symbols = [a[0] for a in atom]
    positions = [a[1] for a in atom]
    atoms = Atoms(symbols=symbols, positions=positions)
    ana = ASEAnalysis(atoms)
    bonds = ana.unique_bonds[0]
    nb = 0
    for b in bonds:
        nb += len(b)
    return nb

def get_run_energy_and_nbond(dirname):
    with open(os.path.join(dirname, 'run_info.json'), 'r') as f:
        data = json.load(f)
    nb = get_nbond(data['mol']['atom'])
    return data['e_tot'], nb

default_return_data = ['e_tot', 'nelectron']
def get_run_data(dirname, return_data=default_return_data):
    with open(os.path.join(dirname, 'run_info.json'), 'r') as f:
        data = json.load(f)
    return {name : data.get(name) for name in return_data}

def read_accdb_structure(struct_id):
    ACCDB_DIR = os.environ.get('ACCDB')
    fname = '{}.xyz'.format(os.path.join(ACCDB_DIR, 'Geometries', struct_id))
    with open(fname, 'r') as f:
        print(fname)
        lines = f.readlines()
        natom = int(lines[0])
        charge_and_spin = lines[1].split()
        charge = int(charge_and_spin[0].strip().strip(','))
        spin = int(charge_and_spin[1].strip().strip(',')) - 1
        symbols = []
        coords = []
        for i in range(natom):
            line = lines[2+i]
            symbol, x, y, z = line.split()
            if symbol.isdigit():
                symbol = int(symbol)
            else:
                symbol = symbol[0] + symbol[1:].lower()
            symbols.append(symbol)
            coords.append([x,y,z])
        struct = Atoms(symbols, positions = coords)
    return struct, os.path.join('ACCDB', struct_id), spin, charge

from ase import Atoms

def get_accdb_data(formula, FUNCTIONAL, BASIS, per_bond=False):
    pred_energy = 0
    if per_bond:
        nbond = 0
        #nbond = None
    for sname, count in zip(formula['structs'], formula['counts']):
        struct, mol_id, spin, charge = read_accdb_structure(sname)
        #if spin == 0:
        #    CALC_TYPE = 'RKS'
        #else:
        #    CALC_TYPE = 'UKS'
        CALC_TYPE = 'UKS'
        dname = get_save_dir(os.environ['MLDFTDB'], CALC_TYPE, BASIS, mol_id, FUNCTIONAL)
        if not os.path.exists(dname):
            CALC_TYPE = 'RKS'
            dname = get_save_dir(os.environ['MLDFTDB'], CALC_TYPE, BASIS, mol_id, FUNCTIONAL)
        if per_bond:
            en, nb = get_run_energy_and_nbond(dname)
            pred_energy += count * en
            print('NB', nb)
            nbond += count * nb
            #if nbond is None:
            #    nbond = nb
        else:
            pred_energy += count * get_run_total_energy(dname)
    
    if per_bond:
        return pred_energy, formula['energy'], abs(nbond)
    else:
        return pred_energy, formula['energy']

def get_accdb_mol_ids(formula):
    pred_energy = 0
    mol_ids = []
    for sname, count in zip(formula['structs'], formula['counts']):
        struct, mol_id, spin, charge = read_accdb_structure(sname)
        mol_ids.append(mol_id)        
    return mol_ids

def get_accdb_data_point(data_point_names, FUNCTIONAL, BASIS):
    single = False
    if not isinstance(data_point_names, list):
        data_point_names = [data_point_names]
        single = True
    result = {}
    for data_point_name in data_point_names:
        db_name, ref_name = data_point_name.split('_', 1)
        dataset_eval_name = os.path.join(db_name, 'DatasetEval.csv')
        formula = get_accdb_formula_entry(ref_name, dataset_eval_name)
        pred_energy, energy = get_accdb_data(formula, FUNCTIONAL, BASIS)
        result[data_point_name] = {
            'pred' : pred_energy,
            'true' : energy
        }
    if single:
        return result[data_point_names[0]]
    return result

def get_accdb_formulas(dataset_eval_name):
    with open(dataset_eval_name, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            lines[i] = line.split(',')
        formulas = {}
        for line in lines:
            counts = line[1:-1:2]
            structs = line[2:-1:2]
            energy = float(line[-1])
            counts = [int(c) for c in counts]
            formulas[line[0]] = {'structs': structs, 'counts': counts, 'energy': energy}
    return formulas

def get_accdb_performance(dataset_eval_name, FUNCTIONAL, BASIS, data_names,
                          per_bond=False, comp_functional=None):
    formulas = get_accdb_formulas(dataset_eval_name)
    result = {}
    errs = []
    nbonds = 0
    for data_point_name, formula in list(formulas.items()):
        if data_point_name not in data_names:
            #print(data_point_name)
            continue
        pred_energy, energy, nbond = get_accdb_data(formula, FUNCTIONAL, BASIS,
                                                    per_bond=True)
        nbonds += nbond
        result[data_point_name] = {
            'pred' : pred_energy,
            'true' : energy
        }
        if comp_functional is not None:
            pred_ref, _, _ = get_accdb_data(formula, comp_functional, BASIS,
                                            per_bond=True)
            energy = pred_ref
            result[data_point_name]['true'] = pred_ref
        print(pred_energy-energy, pred_energy, energy)
        errs.append(pred_energy-energy)
    errs = np.array(errs)
    print(errs.shape)
    me = np.mean(errs)
    mae = np.mean(np.abs(errs))
    rmse = np.sqrt(np.mean(errs**2))
    std = np.std(errs)
    if per_bond:
        return nbonds, np.sum(errs) / nbonds, np.sum(np.abs(errs)) / nbonds
    else:
        return me, mae, rmse, std, result

def get_accdb_mol_set(dataset_eval_name, data_names):
    formulas = get_accdb_formulas(dataset_eval_name)    
    result = {}
    errs = []
    all_mols = set([])
    for data_point_name, formula in list(formulas.items()):
        if data_point_name not in data_names:
            continue
        mol_ids = get_accdb_mol_ids(formula)
        for mol_id in mol_ids:
            all_mols.add(mol_id)
    return all_mols

def load_run_info(mol_id, calc_type, functional, basis):
    d = get_save_dir(SAVE_ROOT, calc_type, basis, mol_id, functional)
    with open(os.path.append(d, 'run_info.json'), 'r') as f:
        data = json.load(f)
    return data

def get_mol_atoms(mol, functional, basis):
    atoms = [atomic_numbers[a[0]] for a in mol._atom]
    formula = Counter(atoms)
    elems = []
    for Z in list(formula.keys()):
        symb = chemical_symbols[Z]
        spin = int(ground_state_magnetic_moments[Z])
        count = formula[Z]
        atom_id = 'atoms/{}-{}-{}'.format(Z, symb, spin)
        elems.append((atom_id, count, 'UKS' if spin else 'RKS'))
    return elems

def get_augg2_ae(data_file, functional, basis, per_bond=False):
    with open(data_file, 'r') as f:
        data = yaml.load(f, Loader=yaml.Loader)
    calc_type = data['calc_type']
    mol_ids = data['mols']
    aes = {}
    for mol_id in mol_ids:
        dirname = get_save_dir(SAVE_ROOT, calc_type, basis, mol_id, functional)
        d = get_run_data(dirname, return_data=['mol', 'e_tot'])
        mol = gto.mole.unpack(d['mol'])
        mol.verbose = int(mol.verbose)
        mol.charge = int(mol.charge)
        mol.spin = int(mol.spin)
        mol.symmetry = (mol.symmetry == 'True') or (mol.symmetry == True)
        mol.build()
        elems = get_mol_atoms(mol, functional, basis)
        ae = d['e_tot']
        if per_bond:
            nb = max(get_nbond(d['mol']['atom']), 1)
        else:
            nb = 1
        for atom_id, count, atom_calc_type in elems:
            dirname = get_save_dir(SAVE_ROOT, atom_calc_type, basis, atom_id, functional)
            en = get_run_data(dirname, return_data=['e_tot'])['e_tot']
            ae -= en * count
        aes[mol_id] = ae / nb
    return aes
    
