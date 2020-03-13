import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
import os
from sklearn.metrics import r2_score
from mldftdat.density import get_exchange_descriptors
from pyscf.dft.libxc import eval_xc

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

def compile_dataset(DATASET_NAME, MOL_IDS, CALC_TYPE, FUNCTIONAL, BASIS,
                    spherical_atom = False):

    all_descriptor_data = None
    all_rho_data = None
    all_values = []

    for MOL_ID in MOL_IDS:
        print('Working on {}'.format(MOL_ID))
        data_dir = get_save_dir(SAVE_ROOT, CALC_TYPE, BASIS, MOL_ID, FUNCTIONAL)
        start = time.monotonic()
        analyzer = RHFAnalyzer.load(data_dir + '/data.hdf5')
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
        end = time.monotonic()
        print('get descriptor time', end - start)
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

def ldax(n):
    return LDA_FACTOR * n**(4.0/3)

def ldax_dens(n):
    return LDA_FACTOR * n**(1.0/3)

def get_gp_x_descriptors(X, y, num=1):
    X = X[:,(0,1,2,3,4,5,7,6)]
    #print(np.max(X, axis=0))
    #print(np.min(X, axis=0))
    rho, X = X[:,0], X[:,1:1+num]
    X[:,0] = np.log(1+X[:,0])
    if num > 1:
        X[:,1] = np.log(0.5 * (1 + X[:,1]))
    if num > 3:
        #X[:,3] = np.log(1-X[:,3])
        X[:,3] = np.arcsinh(X[:,3])
        #fac = np.max(np.abs(X[:,3])) / 3
        #X[:,3] = np.arctan(X[:,3] / fac)
        #X[:,3] = np.arctan(X[:,3])
    if num > 4:
        X[:,4] = np.arcsinh(X[:,4])
    if num > 6:
        X[:,6] = np.arcsinh(X[:,6])
    if num > 5:
        X[:,5] = np.log(X[:,5] / 6)
    #if num > 5:
    #    X[:,5] = np.arcsinh(X[:,5])
    #if num > 6:
    #    X[:,6] = np.log(X[:,6] / 6)
    y = np.log(y / (ldax(rho) - 1e-7) + 1e-7)
    return X, y

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
    X = np.loadtxt(os.path.join(dirname, 'desc.npz')).transpose()
    if count is not None:
        X = X[:count]
    else:
        count = X.shape[0]
    y = np.loadtxt(os.path.join(dirname, 'val.npz'))[:count]
    X, y = get_gp_x_descriptors(X, y, num=num)

    rho_data = np.loadtxt(os.path.join(dirname, 'rho.npz'))[:,:count]
    rho = rho_data[0,:]

    condition = rho > tol

    X = X[condition]
    y = y[condition]

    rho = rho[:count][condition]
    rho_data = rho_data[:,condition]

    return X, y, rho, rho_data

def get_x(y, rho):
    """
    Get the exchange energy density (n * epsilon_x)
    from the exchange enhancement factor y
    and density rho.
    """
    return np.exp(y) * ldax_dens(rho)

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
                     restricted = True):
    """
    model:  If None, return exact exchange results
            If str, evaluate the exchange energy of that functional.
            Otherwise, assume sklearn model and run predict function.
    """
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
    elif type(model) == str:
        eps = eval_xc(model + ',', rho_data)[0]
        neps = eps * rho
    else:
        xdesc = get_exchange_descriptors(rho_data, tau_data, coords,
                                         weights, restricted = restricted)
        X, y = get_gp_x_descriptors(xdesc.transpose(),
                                    analyzer.get_fx_energy_density(), num = num)
        y_pred, std = model.predict(X, return_std = True)
        eps = get_x(y_pred, rho)
        neps = rho * eps
    xef = neps / (ldax(rho) + 1e-7)
    fx_total = np.dot(neps, weights)
    return xef, eps, neps, fx_total

def error_table(dirs, Analyzer, mlmodel, num = 1):
    models = ['LDA', 'PBE', 'SCAN', mlmodel]
    errlst = [[] for _ in models]
    fxlst_pred = [[] for _ in models]
    fxlst_true = []
    count = 0
    ise = np.zeros(4)
    tse = np.zeros(4)
    rise = np.zeros(4)
    rtse = np.zeros(4)
    for d in dirs:
        analyzer = Analyzer.load(os.path.join(d, 'data.hdf5'))
        weights = analyzer.grid.weights
        xef_true, eps_true, neps_true, fx_total_true = predict_exchange(analyzer)
        fxlst_true.append(fx_total_true)
        count += eps_true.shape[0]
        for i, model in enumerate(models):
            xef_pred, eps_pred, neps_pred, fx_total_pred = \
                predict_exchange(analyzer, model = model, num = num)

            ise[i] += np.dot((eps_pred - eps_true)**2, weights)
            tse[i] += ((eps_pred - eps_true)**2).sum()
            rise[i] += np.dot((xef_pred - xef_true)**2, weights)
            rtse[i] += ((xef_pred - xef_true)**2).sum()

            fxlst_pred[i].append(fx_total_pred)
            errlst[i].append(fx_total_pred - fx_total_true)

    fxlst_true = np.array(fxlst_true)
    fxlst_pred = np.array(fxlst_pred)
    errlst = np.array(errlst)

    fx_total_rmse = np.sqrt(np.mean(errlst**2, axis=1))
    rmise = np.sqrt(ise / count)
    rmse = np.sqrt(tse / count)
    rrmise = np.sqrt(rise / len(dirs))
    rrmse = np.sqrt(rtse / len(dirs))

    columns = ['RMSE EX', 'RMISE', 'RMSE', 'Rel. RMISE', 'Rel. RMSE']
    rows = models[:3] + ['ML']
    errtbl = np.array([fx_total_rmse, rmise, rmse, rrmise, rrmse]).transpose()

    return (fxlst_true, fxlst_pred, errlst),\
           (columns, rows, errtbl)
