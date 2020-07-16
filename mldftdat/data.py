import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
from mldftdat.workflow_utils import get_save_dir
from mldftdat.density import get_exchange_descriptors, get_exchange_descriptors2, edmgga
import os
from sklearn.metrics import r2_score
from pyscf.dft.libxc import eval_xc
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from mldftdat.analyzers import RHFAnalyzer, UHFAnalyzer
from mldftdat.pyscf_utils import transform_basis_1e
#from mldftdat.models.nn import Predictor
from pyscf.dft.numint import eval_ao, eval_rho

LDA_FACTOR = - 3.0 / 4.0 * (3.0 / np.pi)**(1.0/3)

def get_unique_coord_indexes_spherical(coords):
    rs = np.linalg.norm(coords, axis=1)
    unique_rs = np.array([])
    indexes = []
    for i, r in enumerate(rs):
        if (np.abs(unique_rs - r) > 1e-7).all():
            unique_rs = np.append(unique_rs, [r], axis=0)
            indexes.append(i)
    return indexes

def density_similarity_atom(rho1, rho2, grid, mol, exponent = 1, inner_r = 0.2):
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

def density_similarity(rho1, rho2, grid, mol, exponent = 1, inner_r = 0.2):
    weights = grid.weights.copy()
    for atom in mol._atom:
        coord = np.array(atom[1])
        rel_coords = grid.coords - coord
        rel_r = np.linalg.norm(rel_coords, axis = 1)
        weights[rel_r < inner_r] = 0
    diff = np.abs(rho1 - rho2)**exponent
    return np.dot(diff, weights)**(1.0/exponent)

def rho_data_from_calc(calc, grid, is_ccsd = False):
    ao = eval_ao(calc.mol, grid.coords, deriv=2)
    dm = calc.make_rdm1()
    if is_ccsd:
        if len(dm.shape) == 3:
            trans_mo_coeff = np.transpose(calc.mo_coeff, axes=(0,2,1))
        else:
            trans_mo_coeff = calc.mo_coeff.T
        dm = transform_basis_1e(dm, trans_mo_coeff)
    rho = eval_rho(calc.mol, ao, dm, xctype='MGGA')
    return rho

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

def get_zr_diatomic(mol, coords):
    mol.build()
    diff = np.array(mol._atom[1][1]) - np.array(mol._atom[0][1])
    direction = diff / np.linalg.norm(diff)
    zs = np.dot(coords, direction)
    zvecs = np.outer(zs, direction)
    print(zvecs.shape)
    rs = np.linalg.norm(coords - zvecs, axis=1)
    return zs, rs

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

def compile_dataset(DATASET_NAME, MOL_IDS, SAVE_ROOT, CALC_TYPE, FUNCTIONAL, BASIS,
                    Analyzer, spherical_atom = False, locx = False,
                    append_all_rho_data = False):

    import time
    all_descriptor_data = None
    all_rho_data = None
    all_values = []

    for MOL_ID in MOL_IDS:
        print('Working on {}'.format(MOL_ID))
        data_dir = get_save_dir(SAVE_ROOT, CALC_TYPE, BASIS, MOL_ID, FUNCTIONAL)
        start = time.monotonic()
        analyzer = Analyzer.load(data_dir + '/data.hdf5')
        end = time.monotonic()
        print('analyzer load time', end - start)
        if spherical_atom:
            start = time.monotonic()
            indexes = get_unique_coord_indexes_spherical(analyzer.grid.coords)
            end = time.monotonic()
            print('index scanning time', end - start)
        start = time.monotonic()
        descriptor_data = get_exchange_descriptors(analyzer.rho_data,
                                                   analyzer.tau_data,
                                                   analyzer.grid.coords,
                                                   analyzer.grid.weights,
                                                   restricted = True)
        if append_all_rho_data:
            from mldftdat import pyscf_utils
            ao_data, rho_data = pyscf_utils.get_mgga_data(analyzer.mol,
                                                        analyzer.grid,
                                                        analyzer.rdm1)
            ddrho = pyscf_utils.get_rho_second_deriv(analyzer.mol,
                                                    analyzer.grid,
                                                    analyzer.rdm1,
                                                    ao_data)
            descriptor_data = np.append(descriptor_data, analyzer.rho_data, axis=0)
            descriptor_data = np.append(descriptor_data, analyzer.tau_data, axis=0)
            descriptor_data = np.append(descriptor_data, ddrho, axis=0)
        end = time.monotonic()
        print('get descriptor time', end - start)
        if locx:
            print('Getting loc fx')
            #values = analyzer.get_loc_fx_energy_density()
            values = analyzer.get_smooth_fx_energy_density()
        else:
            values = analyzer.get_fx_energy_density()
        descriptor_data = descriptor_data
        rho_data = analyzer.rho_data
        if spherical_atom:
            values = values[indexes]
            descriptor_data = descriptor_data[:,indexes]
            rho_data = rho_data[:,indexes]

        if all_descriptor_data is None:
            all_descriptor_data = descriptor_data
        else:
            all_descriptor_data = np.append(all_descriptor_data, descriptor_data,
                                            axis = 1)
        if all_rho_data is None:
            all_rho_data = rho_data
        else:
            all_rho_data = np.append(all_rho_data, rho_data, axis=1)
        all_values = np.append(all_values, values)

    save_dir = os.path.join(SAVE_ROOT, 'DATASETS', DATASET_NAME)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    rho_file = os.path.join(save_dir, 'rho.npz')
    desc_file = os.path.join(save_dir, 'desc.npz')
    val_file = os.path.join(save_dir, 'val.npz')
    np.savetxt(rho_file, all_rho_data)
    np.savetxt(desc_file, all_descriptor_data)
    np.savetxt(val_file, all_values)
    #gp = DFTGP(descriptor_data, values, 1e-3)

def compile_dataset2(DATASET_NAME, MOL_IDS, SAVE_ROOT, CALC_TYPE, FUNCTIONAL, BASIS,
                    Analyzer, spherical_atom = False, locx = False, lam = 0.5,
                    version = 'a'):

    import time
    from pyscf import scf
    all_descriptor_data = None
    all_rho_data = None
    all_values = []
    all_weights = []
    cutoffs = []

    for MOL_ID in MOL_IDS:
        print('Working on {}'.format(MOL_ID))
        data_dir = get_save_dir(SAVE_ROOT, CALC_TYPE, BASIS, MOL_ID, FUNCTIONAL)
        start = time.monotonic()
        analyzer = Analyzer.load(data_dir + '/data.hdf5')
        analyzer.get_ao_rho_data()
        if type(analyzer.calc) == scf.hf.RHF:
            restricted = True
        else:
            restricted = False
        end = time.monotonic()
        print('analyzer load time', end - start)
        if spherical_atom:
            start = time.monotonic()
            indexes = get_unique_coord_indexes_spherical(analyzer.grid.coords)
            end = time.monotonic()
            print('index scanning time', end - start)
        start = time.monotonic()
        if restricted:
            descriptor_data = get_exchange_descriptors2(analyzer, restricted = True, version=version)
        else:
            descriptor_data_u, descriptor_data_d = \
                              get_exchange_descriptors2(analyzer, restricted = False, version=version)
            descriptor_data = np.append(descriptor_data_u, descriptor_data_d,
                                        axis = 1)
        """
        if append_all_rho_data:
            from mldftdat import pyscf_utils
            ao_data, rho_data = pyscf_utils.get_mgga_data(analyzer.mol,
                                                        analyzer.grid,
                                                        analyzer.rdm1)
            ddrho = pyscf_utils.get_rho_second_deriv(analyzer.mol,
                                                    analyzer.grid,
                                                    analyzer.rdm1,
                                                    ao_data)
            if restricted:
                descriptor_data = np.append(descriptor_data, analyzer.rho_data, axis=0)
                descriptor_data = np.append(descriptor_data, analyzer.tau_data, axis=0)
                descriptor_data = np.append(descriptor_data, ddrho, axis=0)
            else:
                tmp1 = 2 * np.append(analyzer.rho_data[0], analyzer.rho_data[1], axis=1)
                tmp2 = 2 * np.append(analyzer.tau_data[0], analyzer.tau_data[1], axis=1)
                tmp3 = 2 * np.append(ddrho[0], ddrho[1], axis=1)
                descriptor_data = np.append(descriptor_data, tmp1, axis=0)
                descriptor_data = np.append(descriptor_data, tmp2, axis=0)
                descriptor_data = np.append(descriptor_data, tmp3, axis=0)
        """
        end = time.monotonic()
        print('get descriptor time', end - start)
        if locx:
            print('Getting loc fx')
            values = analyzer.get_loc_fx_energy_density(lam = lam, overwrite=True)
            if not restricted:
                values = 2 * np.append(analyzer.loc_fx_energy_density_u,
                                       analyzer.loc_fx_energy_density_d)
        else:
            values = analyzer.get_fx_energy_density()
            if not restricted:
                values = 2 * np.append(analyzer.fx_energy_density_u,
                                       analyzer.fx_energy_density_d)
        rho_data = analyzer.rho_data
        if not restricted:
            rho_data = 2 * np.append(rho_data[0], rho_data[1], axis=1)
        if spherical_atom:
            values = values[indexes]
            descriptor_data = descriptor_data[:,indexes]
            rho_data = rho_data[:,indexes]

        if all_descriptor_data is None:
            all_descriptor_data = descriptor_data
        else:
            all_descriptor_data = np.append(all_descriptor_data, descriptor_data,
                                            axis = 1)
        if all_rho_data is None:
            all_rho_data = rho_data
        else:
            all_rho_data = np.append(all_rho_data, rho_data, axis=1)
        all_values = np.append(all_values, values)
        all_weights = np.append(all_weights, analyzer.grid.weights)
        if not restricted:
            # two copies for unrestricted case
            all_weights = np.append(all_weights, analyzer.grid.weights)
        cutoffs.append(all_values.shape[0])

    save_dir = os.path.join(SAVE_ROOT, 'DATASETS', DATASET_NAME)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    rho_file = os.path.join(save_dir, 'rho.npz')
    desc_file = os.path.join(save_dir, 'desc.npz')
    val_file = os.path.join(save_dir, 'val.npz')
    wt_file = os.path.join(save_dir, 'wt.npz')
    cut_file = os.path.join(save_dir, 'cut.npz')
    np.savetxt(rho_file, all_rho_data)
    np.savetxt(desc_file, all_descriptor_data)
    np.savetxt(val_file, all_values)
    np.savetxt(wt_file, all_weights)
    np.savetxt(cut_file, np.array(cutoffs))

def ldax(n):
    return LDA_FACTOR * n**(4.0/3)

def ldax_dens(n):
    return LDA_FACTOR * n**(1.0/3)

def get_gp_x_descriptors(X, num=1, selection=None):
    X = X[:,(0,1,2,3,4,5,7,6)]
    if selection is not None:
        num = 7
    #print(np.max(X, axis=0))
    #print(np.min(X, axis=0))
    #rho, X = X[:,0], X[:,1:1+num]
    rho, X = X[:,0], X[:,1:]
    X[:,0] = np.log(1+X[:,0])
    #X[:,1] = np.log(0.5 * (1 + X[:,1]))
    X[:,1] = 1 / (1 + X[:,1]**2) - 0.5
    X[:,3] = np.arcsinh(X[:,3])
    X[:,4] = np.arcsinh(X[:,4])
    X[:,6] = np.arcsinh(X[:,6])
    X[:,5] = np.log(X[:,5] / 6)
    #if num > 5:
    #    X[:,5] = np.arcsinh(X[:,5])
    #if num > 6:
    #    X[:,6] = np.log(X[:,6] / 6)
    if selection is None:
        X = X[:,(0,1,2,5,4,3,6)]
        return X[:,:num]
    else:
        return X[:,selection]

def load_descriptors(dirname, count=None, val_dirname = None, load_wt = False):
    X = np.loadtxt(os.path.join(dirname, 'desc.npz')).transpose()
    if count is not None:
        X = X[:count]
    else:
        count = X.shape[0]
    if val_dirname is None:
        val_dirname = dirname
    y = np.loadtxt(os.path.join(val_dirname, 'val.npz'))[:count]
    rho_data = np.loadtxt(os.path.join(dirname, 'rho.npz'))[:,:count]
    if load_wt:
        wt = np.loadtxt(os.path.join(dirname, 'wt.npz'))[:count]
        return X, y, rho_data, wt
    return X, y, rho_data

def filter_descriptors(X, y, rho_data, tol=1e-3, wt = None):
    condition = rho_data[0] > tol
    X = X[condition,:]
    y = y[condition]
    rho = rho_data[0,condition]
    rho_data = rho_data[:,condition]
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

def get_xed_from_y(y, rho):
    """
    Get the exchange energy density (n * epsilon_x)
    from the exchange enhancement factor y
    and density rho.
    """
    return rho * get_x(y, rho)

def get_x(y, rho):
    #return np.exp(y) * ldax_dens(rho)
    return (y + 1) * ldax_dens(rho)

def get_y_from_xed(xed, rho):
    #return np.log(xed / (ldax(rho) - 1e-7) + 1e-7)
    return xed / (ldax(rho) - 1e-12) - 1

def true_metric(y_true, y_pred, rho):
    """
    Find relative and absolute mse, as well as r2
    score, for the exchange energy density (n * epsilon_x)
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

def predict_exchange(analyzer, model=None, num=1,
                     restricted = True, return_desc = False, version = 'a'):
    """
    model:  If None, return exact exchange results
            If str, evaluate the exchange energy of that functional.
            Otherwise, assume sklearn model and run predict function.
    """
    from mldftdat.models.nn import Predictor
    from mldftdat.dft.xc_models import MLFunctional
    from mldftdat.models.integral_gps import AddEDMGPR
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
        neps = fx * ldax(rho)
        eps = fx * ldax_dens(rho)
    elif type(model) == str:
        eps = eval_xc(model + ',', rho_data)[0]
        neps = eps * rho
    elif type(model) == GPR:
        xdesc = get_exchange_descriptors(rho_data, tau_data, coords,
                                         weights, restricted = restricted)
        rho = xdesc[0,:]
        X = get_gp_x_descriptors(xdesc.transpose(), num=num)
        y = get_xed_from_y(analyzer.get_fx_energy_density(), rho)
        y_pred, std = model.predict(X, return_std = True)
        eps = get_x(y_pred, rho)
        neps = rho * eps
    elif isinstance(model, Predictor):
        xdesc = get_exchange_descriptors2(analyzer, restricted = restricted, version = version)
        neps = model.predict(xdesc.transpose(), rho_data)
        eps = neps / rho
        if return_desc:
            X = model.get_descriptors(xdesc.transpose(), rho_data, num = model.num)
    elif isinstance(model, MLFunctional):
        N = analyzer.grid.weights.shape[0]
        desc  = np.zeros((N, len(model.desc_list)))
        ddesc = np.zeros((N, len(model.desc_list)))
        xdesc = get_exchange_descriptors2(analyzer, restricted = restricted, version = version)
        for i, d in enumerate(model.desc_list):
            desc[:,i], ddesc[:,i] = d.transform_descriptor(xdesc, deriv = 1)
        xef = model.get_F(desc)
        eps = LDA_FACTOR * xef * analyzer.rho_data[0]**(1.0/3)
        neps = LDA_FACTOR * xef * analyzer.rho_data[0]**(4.0/3)
    elif isinstance(model, AddEDMGPR):
        from pyscf import lib
        xdesc = get_exchange_descriptors2(analyzer, restricted = restricted, version = version)
        gridsize = xdesc.shape[1]
        neps, std = np.zeros(gridsize), np.zeros(gridsize)
        blksize = 10000
        for p0, p1 in lib.prange(0, ngrid, blksize):
            neps[p0:p1], std[p0:p1] = model.predict(xdesc.T[p0:p1],
                                                    rho_data[:,p0:p1], return_std = True)
        if return_desc:
            X = model.get_descriptors(xdesc.transpose(), rho_data, num = model.num)
    else:# type(model) == integral_gps.NoisyEDMGPR:
        xdesc = get_exchange_descriptors2(analyzer, restricted = restricted, version = version)
        neps, std = model.predict(xdesc.transpose(), rho_data, return_std = True)
        print('integrated uncertainty', np.sqrt(np.dot(std**2, weights)))
        eps = neps / rho
        if return_desc:
            X = model.get_descriptors(xdesc.transpose(), rho_data, num = model.num)
    """else:
        xdesc = get_exchange_descriptors(rho_data, tau_data, coords,
                                         weights, restricted = restricted)
        #neps = model.predict(xdesc.transpose(), rho)
        neps, std = model.predict(xdesc.transpose(), rho_data, return_std = True)
        print('integrated uncertainty', np.sqrt(np.dot(std**2, weights)))
        eps = neps / rho
        if return_desc:
            X = model.get_descriptors(xdesc.transpose(), rho_data, num = model.num)
    """
    xef = neps / (ldax(rho) - 1e-7)
    fx_total = np.dot(neps, weights)
    if return_desc:
        return xef, eps, neps, fx_total, X
    else:
        return xef, eps, neps, fx_total

def predict_total_exchange_unrestricted(analyzer, model=None, num=1, version = 'a'):
    if isinstance(analyzer, RHFAnalyzer):
        return predict_exchange(analyzer, model, num, version = version)[3]
    from mldftdat.models.nn import Predictor
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
        neps = 0.5 * fxu * ldax(2 * rho[0]) + 0.5 * fxd * ldax(2 * rho[1])
    elif type(model) == str:
        eps = eval_xc(model + ',', rho_data, spin=analyzer.mol.spin)[0]
        #epsu = eval_xc(model + ',', 2 * rho_data[0])[0]
        #epsd = eval_xc(model + ',', 2 * rho_data[1])[0]
        #print(eps.shape, rho_data.shape, analyzer.mol.spin)
        neps = eps * (rho[0] + rho[1])
    elif isinstance(model, MLFunctional):
        N = analyzer.grid.weights.shape[0]
        neps = 0
        xdescu, xdescd = get_exchange_descriptors2(analyzer, restricted = False, version = version)
        for xdesc, rho_data in [(xdescu, analyzer.rho_data[0]), (xdescd, analyzer.rho_data[1])]:
            desc  = np.zeros((N, len(model.desc_list)))
            ddesc = np.zeros((N, len(model.desc_list)))
            for i, d in enumerate(model.desc_list):
                desc[:,i], ddesc[:,i] = d.transform_descriptor(xdesc, deriv = 1)
            xef = model.get_F(desc)
            neps += LDA_FACTOR * xef * rho_data[0]**(4.0/3) * 2**(1.0/3)
    else:
        xdescu, xdescd = get_exchange_descriptors2(analyzer, restricted = False, version = version)
        neps = 0.5 * model.predict(xdescu.transpose(), 2 * rho_data[0])
        neps += 0.5 * model.predict(xdescd.transpose(), 2 * rho_data[1])
    """
    else:
        xdescu, xdescd = get_exchange_descriptors(rho_data, tau_data, coords,
                                         weights, restricted = False)
        neps = 0.5 * model.predict(xdescu.transpose(), 2 * rho_data[0])
        neps += 0.5 * model.predict(xdescd.transpose(), 2 * rho_data[1])
    """
    fx_total = np.dot(neps, weights)
    return fx_total

def error_table(dirs, Analyzer, mlmodel, num = 1):
    models = ['LDA', 'PBE', 'SCAN', 'EDM', mlmodel]
    errlst = [[] for _ in models]
    fxlst_pred = [[] for _ in models]
    fxlst_true = []
    count = 0
    NMODEL = len(models)
    ise = np.zeros(NMODEL)
    tse = np.zeros(NMODEL)
    rise = np.zeros(NMODEL)
    rtse = np.zeros(NMODEL)
    for d in dirs:
        print(d.split('/')[-1])
        analyzer = Analyzer.load(os.path.join(d, 'data.hdf5'))
        weights = analyzer.grid.weights
        rho = analyzer.rho_data[0,:]
        condition = rho > 3e-3
        xef_true, eps_true, neps_true, fx_total_true = predict_exchange(analyzer)
        print(np.std(xef_true[condition]), np.std(eps_true[condition]))
        fxlst_true.append(fx_total_true)
        count += eps_true.shape[0]
        for i, model in enumerate(models):
            xef_pred, eps_pred, neps_pred, fx_total_pred = \
                predict_exchange(analyzer, model = model, num = num)
            print(fx_total_pred - fx_total_true, np.std(xef_pred[condition]))

            ise[i] += np.dot((eps_pred[condition] - eps_true[condition])**2, weights[condition])
            tse[i] += ((eps_pred[condition] - eps_true[condition])**2).sum()
            rise[i] += np.dot((xef_pred[condition] - xef_true[condition])**2, weights[condition])
            rtse[i] += ((xef_pred[condition] - xef_true[condition])**2).sum()

            fxlst_pred[i].append(fx_total_pred)
            errlst[i].append(fx_total_pred - fx_total_true)
        print(errlst[-1][-1])
        print()
    fxlst_true = np.array(fxlst_true)
    fxlst_pred = np.array(fxlst_pred)
    errlst = np.array(errlst)

    print(count, len(dirs))

    fx_total_rmse = np.sqrt(np.mean(errlst**2, axis=1))
    rmise = np.sqrt(ise / len(dirs))
    rmse = np.sqrt(tse / count)
    rrmise = np.sqrt(rise / len(dirs))
    rrmse = np.sqrt(rtse / count)

    columns = ['RMSE EX', 'RMISE', 'RMSE', 'Rel. RMISE', 'Rel. RMSE']
    rows = models[:NMODEL-1] + ['ML']
    errtbl = np.array([fx_total_rmse, rmise, rmse, rrmise, rrmse]).transpose()

    return (fxlst_true, fxlst_pred, errlst),\
           (columns, rows, errtbl)

def error_table_unrestricted(dirs, Analyzer, mlmodel, num = 1):
    models = ['LDA', 'PBE', 'SCAN', 'EDM', mlmodel]
    errlst = [[] for _ in models]
    fxlst_pred = [[] for _ in models]
    fxlst_true = []
    count = 0
    NMODEL = len(models)
    ise = np.zeros(NMODEL)
    tse = np.zeros(NMODEL)
    rise = np.zeros(NMODEL)
    rtse = np.zeros(NMODEL)
    for d in dirs:
        print(d.split('/')[-1])
        analyzer = Analyzer.load(os.path.join(d, 'data.hdf5'))
        analyzer.get_ao_rho_data()
        weights = analyzer.grid.weights
        rho = analyzer.rho_data[0,:]
        condition = rho > 3e-3
        fx_total_true = predict_total_exchange_unrestricted(analyzer)
        fxlst_true.append(fx_total_true)
        count += 1
        for i, model in enumerate(models):
            fx_total_pred = predict_total_exchange_unrestricted(analyzer, model = model, num = num)
            print(fx_total_pred - fx_total_true)

            #ise[i] += np.dot((eps_pred[condition] - eps_true[condition])**2, weights[condition])
            #tse[i] += ((eps_pred[condition] - eps_true[condition])**2).sum()
            #rise[i] += np.dot((xef_pred[condition] - xef_true[condition])**2, weights[condition])
            #rtse[i] += ((xef_pred[condition] - xef_true[condition])**2).sum()

            fxlst_pred[i].append(fx_total_pred)
            errlst[i].append(fx_total_pred - fx_total_true)
        print(errlst[-1][-1])
        print()
    fxlst_true = np.array(fxlst_true)
    fxlst_pred = np.array(fxlst_pred)
    errlst = np.array(errlst)

    print(count, len(dirs))

    fx_total_rmse = np.sqrt(np.mean(errlst**2, axis=1))
    rmise = np.sqrt(ise / len(dirs))
    rmse = np.sqrt(tse / count)
    rrmise = np.sqrt(rise / len(dirs))
    rrmse = np.sqrt(rtse / count)

    columns = ['RMSE EX', 'RMISE', 'RMSE', 'Rel. RMISE', 'Rel. RMSE']
    rows = models[:NMODEL-1] + ['ML']
    errtbl = np.array([fx_total_rmse, rmise, rmse, rrmise, rrmse]).transpose()

    return (fxlst_true, fxlst_pred, errlst),\
           (columns, rows, errtbl)

def error_table2(dirs, Analyzer, mlmodel, num = 1):
    models = ['LDA', 'PBE', 'SCAN', 'EDM', mlmodel]
    errlst = [[] for _ in models]
    fxlst_pred = [[] for _ in models]
    fxlst_true = []
    count = 0
    NMODEL = len(models)
    ise = np.zeros(NMODEL)
    tse = np.zeros(NMODEL)
    rise = np.zeros(NMODEL)
    rtse = np.zeros(NMODEL)
    data_dict = {}
    data_dict['true'] = {'rho_data': None, 'eps_data': []}
    for model in models:
        if type(model) != str:
            data_dict['ML'] = {'eps_data': [], 'desc_data': None}
        else:
            data_dict[model] = {'eps_data': []}
    for d in dirs:
        print(d.split('/')[-1])
        analyzer = Analyzer.load(os.path.join(d, 'data.hdf5'))
        weights = analyzer.grid.weights
        rho = analyzer.rho_data[0,:]
        condition = rho > 3e-3
        xef_true, eps_true, neps_true, fx_total_true = predict_exchange(analyzer)
        print(np.std(xef_true[condition]), np.std(eps_true[condition]))
        fxlst_true.append(fx_total_true)
        count += eps_true.shape[0]
        data_dict['true']['eps_data'] = np.append(data_dict['true']['eps_data'], eps_true)
        if data_dict['true'].get('rho_data') is None:
            data_dict['true']['rho_data'] = analyzer.rho_data
        else:
            data_dict['true']['rho_data'] = np.append(data_dict['true']['rho_data'],
                                                analyzer.rho_data, axis=1)
        for i, model in enumerate(models):
            if type(model) == str:
                xef_pred, eps_pred, neps_pred, fx_total_pred = \
                    predict_exchange(analyzer, model = model, num = num)
                desc_data = None
            else:
                xef_pred, eps_pred, neps_pred, fx_total_pred, desc_data = \
                    predict_exchange(analyzer, model = model, num = num,
                        return_desc = True)
            print(fx_total_pred - fx_total_true, np.std(xef_pred[condition]))

            ise[i] += np.dot((eps_pred[condition] - eps_true[condition])**2, weights[condition])
            tse[i] += ((eps_pred[condition] - eps_true[condition])**2).sum()
            rise[i] += np.dot((xef_pred[condition] - xef_true[condition])**2, weights[condition])
            rtse[i] += ((xef_pred[condition] - xef_true[condition])**2).sum()

            fxlst_pred[i].append(fx_total_pred)
            errlst[i].append(fx_total_pred - fx_total_true)

            if desc_data is None:
                data_dict[model]['eps_data'] = np.append(
                                data_dict[model]['eps_data'],
                                eps_pred)
            else:
                data_dict['ML']['eps_data'] = np.append(
                                data_dict['ML']['eps_data'],
                                eps_pred)
                if data_dict['ML'].get('desc_data') is None:
                    data_dict['ML']['desc_data'] = desc_data
                else:
                    data_dict['ML']['desc_data'] = np.append(
                                data_dict['ML']['desc_data'],
                                desc_data, axis=0)

        print(errlst[-1][-1])
        print()
    fxlst_true = np.array(fxlst_true)
    fxlst_pred = np.array(fxlst_pred)
    errlst = np.array(errlst)

    print(count, len(dirs))

    fx_total_rmse = np.sqrt(np.mean(errlst**2, axis=1))
    rmise = np.sqrt(ise / len(dirs))
    rmse = np.sqrt(tse / count)
    rrmise = np.sqrt(rise / len(dirs))
    rrmse = np.sqrt(rtse / count)

    columns = ['RMSE EX', 'RMISE', 'RMSE', 'Rel. RMISE', 'Rel. RMSE']
    rows = models[:NMODEL-1] + ['ML']
    errtbl = np.array([fx_total_rmse, rmise, rmse, rrmise, rrmse]).transpose()

    return (fxlst_true, fxlst_pred, errlst),\
           (columns, rows, errtbl),\
           data_dict

def error_table3(dirs, Analyzer, mlmodel, dbpath, num = 1, version='a'):
    from collections import Counter
    from ase.data import chemical_symbols, atomic_numbers, ground_state_magnetic_moments
    models = ['MGGA_X_GVT4', 'PBE', 'SCAN', 'MGGA_X_TM', mlmodel]
    errlst = [[] for _ in models]
    ae_errlst = [[] for _ in models]
    fxlst_pred = [[] for _ in models]
    ae_fxlst_pred = [[] for _ in models]
    fxlst_true = []
    ae_fxlst_true = []
    count = 0
    NMODEL = len(models)
    ise = np.zeros(NMODEL)
    tse = np.zeros(NMODEL)
    rise = np.zeros(NMODEL)
    rtse = np.zeros(NMODEL)
    for d in dirs:
        print(d.split('/')[-1])
        analyzer = Analyzer.load(os.path.join(d, 'data.hdf5'))
        atoms = [atomic_numbers[a[0]] for a in analyzer.mol._atom]
        formula = Counter(atoms)
        element_analyzers = {}
        for Z in list(formula.keys()):
            symbol = chemical_symbols[Z]
            spin = int(ground_state_magnetic_moments[Z])
            letter = 'R' if spin == 0 else 'U'
            path = '{}/{}KS/PBE/aug-cc-pvtz/atoms/{}-{}-{}/data.hdf5'.format(
                        dbpath, letter, Z, symbol, spin)
            if letter == 'R':
                element_analyzers[Z] = RHFAnalyzer.load(path)
            else:
                element_analyzers[Z] = UHFAnalyzer.load(path)
        weights = analyzer.grid.weights
        rho = analyzer.rho_data[0,:]
        condition = rho > 3e-5
        fx_total_ref_true = 0
        for Z in list(formula.keys()):
            fx_total_ref_true += formula[Z] \
                                 * predict_total_exchange_unrestricted(
                                        element_analyzers[Z], version=version)
        xef_true, eps_true, neps_true, fx_total_true = predict_exchange(analyzer, version = version)
        print(np.std(xef_true[condition]), np.std(eps_true[condition]))
        fxlst_true.append(fx_total_true)
        ae_fxlst_true.append(fx_total_true - fx_total_ref_true)
        count += eps_true.shape[0]
        for i, model in enumerate(models):
            fx_total_ref = 0
            for Z in list(formula.keys()):
                fx_total_ref += formula[Z] \
                                * predict_total_exchange_unrestricted(
                                    element_analyzers[Z],
                                    model = model, num = num, version = version)
            xef_pred, eps_pred, neps_pred, fx_total_pred = \
                predict_exchange(analyzer, model = model, num = num, version = version)
            print(fx_total_pred - fx_total_true, fx_total_pred - fx_total_true \
                                                 - (fx_total_ref - fx_total_ref_true))

            ise[i] += np.dot((eps_pred[condition] - eps_true[condition])**2, weights[condition])
            tse[i] += ((eps_pred[condition] - eps_true[condition])**2).sum()
            rise[i] += np.dot((xef_pred[condition] - xef_true[condition])**2, weights[condition])
            rtse[i] += ((xef_pred[condition] - xef_true[condition])**2).sum()

            fxlst_pred[i].append(fx_total_pred)
            ae_fxlst_pred[i].append(fx_total_pred - fx_total_ref)
            errlst[i].append(fx_total_pred - fx_total_true)
            ae_errlst[i].append(fx_total_pred - fx_total_true \
                                - (fx_total_ref - fx_total_ref_true))
        print(errlst[-1][-1], ae_errlst[-1][-1])
        print()
    fxlst_true = np.array(fxlst_true)
    fxlst_pred = np.array(fxlst_pred)
    errlst = np.array(errlst)
    ae_errlst = np.array(ae_errlst)

    print(count, len(dirs))

    fx_total_rmse = np.sqrt(np.mean(errlst**2, axis=1))
    ae_fx_total_rmse = np.sqrt(np.mean(ae_errlst**2, axis=1))
    rmise = np.sqrt(ise / len(dirs))
    rmse = np.sqrt(tse / count)
    rrmise = np.sqrt(rise / len(dirs))
    rrmse = np.sqrt(rtse / count)

    columns = ['RMSE AEX', 'RMSE EX', 'RMISE', 'RMSE', 'Rel. RMISE', 'Rel. RMSE']
    rows = models[:NMODEL-1] + ['ML']
    errtbl = np.array([ae_fx_total_rmse, fx_total_rmse, rmise, rmse, rrmise, rrmse]).transpose()

    return (fxlst_true, fxlst_pred, errlst, ae_errlst),\
           (columns, rows, errtbl)

def calculate_atomization_energy(DBPATH, CALC_TYPE, BASIS, MOL_ID,
                                 FUNCTIONAL = None, mol = None,
                                 use_db = True,
                                 save_atom_analyzer = False,
                                 save_mol_analyzer = False,
                                 full_analysis = False):
    from mldftdat import lowmem_analyzers
    from collections import Counter
    from ase.data import chemical_symbols, atomic_numbers, ground_state_magnetic_moments
    from mldftdat.pyscf_utils import run_scf, run_cc
    from pyscf import gto
    from mldftdat.dft.xc_models import MLFunctional

    if type(FUNCTIONAL) == str:
        CALC_NAME = os.path.join(CALC_TYPE, FUNCTIONAL)
    else:
        CALC_NAME = CALC_TYPE

    if CALC_TYPE in ['CCSD', 'CCSD_T']:
        Analyzer = lowmem_analyzers.CCSDAnalyzer
    elif CALC_TYPE in ['UCCSD', 'UCCSD_T']:
        Analyzer = lowmem_analyzers.UCCSDAnalyzer
    elif CALC_TYPE in ['RKS', 'RHF']:
        Analyzer = lowmem_analyzers.RHFAnalyzer
    elif CALC_TYPE in ['UKS', 'UHF']:
        Analyzer = lowmem_analyzers.UHFAnalyzer

    print(type(FUNCTIONAL))
    print(isinstance(FUNCTIONAL, MLFunctional))

    def run_calc(mol, path, calc_type, Analyzer, save):
        if os.path.isfile(path) and use_db:
            analyzer = Analyzer.load(path)
            if '_T' in calc_type:
                if analyzer.e_tri is None and mol.nelectron > 2:
                    analyzer.calc_pert_triples()
                return analyzer.calc.e_tot + analyzer.e_tri, analyzer.calc
            else:
                return analyzer.calc.e_tot, analyzer.calc

        else:
            if calc_type == 'CCSD' or (calc_type == 'CCSD_T' and mol.nelectron < 3):
                mf = run_scf(mol, 'RHF')
                mycc = run_cc(mf)
                e_tot = mycc.e_tot
                calc = mycc
            elif (calc_type == 'UCCSD') or (calc_type == 'UCCSD_T' and mol.nelectron < 3):
                mf = run_scf(mol, 'UHF')
                mycc = run_cc(mf)
                e_tot = mycc.e_tot
                calc = mycc
            elif calc_type == 'CCSD_T':
                mf = run_scf(mol, 'RHF')
                mycc = run_cc(mf)
                e_tri = mycc.ccsd_t()
                e_tot = mycc.e_tot + e_tri
                calc = mycc
            elif calc_type == 'UCCSD_T':
                mf = run_scf(mol, 'UHF')
                mycc = run_cc(mf)
                e_tri = mycc.ccsd_t()
                e_tot = mycc.e_tot + e_tri
                calc = mycc
            elif FUNCTIONAL is None:
                mf = run_scf(mol, calc_type)
                e_tot = mf.e_tot
                calc = mf
            elif type(FUNCTIONAL) == str:
                mf = run_scf(mol, calc_type, functional = FUNCTIONAL)
                e_tot = mf.e_tot
                calc = mf
            elif isinstance(FUNCTIONAL, MLFunctional):
                if 'RKS' in path:
                    from mldftdat.dft.numint4 import setup_rks_calc
                    mf = run_scf(mol, 'RKS', functional = 'PBE')
                    dm0 = mf.make_rdm1()
                    #dm0 = None
                    #mf = setup_rks_calc(mol, FUNCTIONAL, mlc = True, vv10_coeff = (6.0, 0.01))
                    mf = setup_rks_calc(mol, FUNCTIONAL, mlc = True)
                    mf.xc = None
                    #mf.xc = ',MGGA_C_SCAN'
                else:
                    from mldftdat.dft.numint4 import setup_uks_calc
                    mf = run_scf(mol, 'UKS', functional = 'PBE')
                    dm0 = mf.make_rdm1()
                    #dm0 = None
                    #mf = setup_uks_calc(mol, FUNCTIONAL, mlc = True, vv10_coeff = (6.0, 0.01))
                    mf = setup_uks_calc(mol, FUNCTIONAL, mlc = True)
                    mf.xc = None
                    #mf.xc = ',MGGA_C_SCAN'
                mf.kernel(dm0 = dm0)
                e_tot = mf.e_tot
                calc = mf

            if save:
                analyzer = Analyzer(mf)
                if full_analysis:
                    analyzer.perform_full_analysis()
                analyzer.dump(path)
            return e_tot, calc

    mol_path = os.path.join(DBPATH, CALC_NAME, BASIS, MOL_ID, 'data.hdf5')
    if mol is None:
        analyzer = Analyzer.load(mol_path)
        mol = analyzer.mol
    mol.basis = BASIS
    mol.build() 
    mol_energy, mol_calc = run_calc(mol, mol_path, CALC_TYPE, Analyzer, save_mol_analyzer)

    atoms = [atomic_numbers[a[0]] for a in mol._atom]
    formula = Counter(atoms)
    element_analyzers = {}
    atomic_energies = {}
    atomic_calcs = {}

    atomization_energy = mol_energy
    for Z in list(formula.keys()):
        symbol = chemical_symbols[Z]
        spin = int(ground_state_magnetic_moments[Z])
        atm = gto.Mole()
        atm.atom = symbol
        atm.spin = spin
        atm.basis = BASIS
        atm.build()
        if CALC_TYPE in ['CCSD', 'UCCSD']:
            ATOM_CALC_TYPE = 'CCSD' if spin == 0 else 'UCCSD'
            AtomAnalyzer = lowmem_analyzers.CCSDAnalyzer if spin == 0\
                           else lowmem_analyzers.UCCSDAnalyzer
        elif CALC_TYPE in ['CCSD_T', 'UCCSD_T']:
            ATOM_CALC_TYPE = 'CCSD_T' if spin == 0 else 'UCCSD_T'
            AtomAnalyzer = lowmem_analyzers.CCSDAnalyzer if spin == 0\
                           else lowmem_analyzers.UCCSDAnalyzer
        elif CALC_TYPE in ['RKS', 'UKS']:
            ATOM_CALC_TYPE = 'RKS' if spin == 0 else 'UKS'
            AtomAnalyzer = lowmem_analyzers.RHFAnalyzer if spin == 0\
                           else lowmem_analyzers.UHFAnalyzer
        else:
            ATOM_CALC_TYPE = 'RHF' if spin == 0 else 'UHF'
            AtomAnalyzer = lowmem_analyzers.RHFAnalyzer if spin == 0\
                           else lowmem_analyzers.UHFAnalyzer
        if type(FUNCTIONAL) == str:
            ATOM_CALC_NAME = os.path.join(ATOM_CALC_TYPE, FUNCTIONAL)
        else:
            ATOM_CALC_NAME = ATOM_CALC_TYPE
        path = os.path.join(
                            DBPATH, ATOM_CALC_NAME, BASIS,
                            'atoms/{}-{}-{}/data.hdf5'.format(
                                Z, symbol, spin)
                           )
        print(path)
        atomic_energies[Z], atomic_calcs[Z] = run_calc(atm, path, ATOM_CALC_TYPE,
                                                       AtomAnalyzer, save_atom_analyzer)
        atomization_energy -= formula[Z] * atomic_energies[Z]

    return mol, atomization_energy, mol_energy, atomic_energies, mol_calc, atomic_calcs
    
