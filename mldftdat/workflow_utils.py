import time, psutil, os
from pyscf import gto, lib

def safe_mem_cap_mb():
    return int(psutil.virtual_memory().available // 16e6)

def time_func(func, *args):
    start_time = time.monotonic()
    res = func(*args)
    finish_time = time.monotonic()
    return res, finish_time - start_time

def get_functional_db_name(functional):
    functional = functional.replace(',', '_')
    functional = functional.replace(' ', '_')
    functional = functional.upper()
    return functional

def get_save_dir(root, calc_type, basis, mol_id, functional=None):
    if functional is not None:
        calc_type = calc_type + '/' + get_functional_db_name(functional)
    return os.path.join(root, calc_type, basis, mol_id)

def mol_from_dict(mol_dict):
    for item in ['charge', 'spin', 'symmetry', 'verbose']:
        if type(mol_dict[item]).__module__ == np.__name__:
            mol_dict[item] = mol_dict[item].item()
    mol = gto.mole.unpack(mol_dict)
    mol.build()
    return mol

def load_calc(fname):
    analyzer_dict = lib.chkfile.load(fname, 'analyzer')
    mol = mol_from_dict(analyzer_dict['mol'])
    calc_type = analyzer_dict['calc_type']
    calc = CALC_TYPES[calc_type](mol)
    calc.__dict__.update(analyzer_dict['calc'])
    return calc, calc_type

def load_analyzer_data(dirname, fname = 'data.hdf5'):
    data_file = os.path.join(dirname, fname)
    return lib.chkfile.load(data_file, 'analyzer/data')

def get_dft_input(rho_data):
    r_s = (3.0 / (4 * np.pi * rho_data[0]))**(1.0/3)
    rho = rho_data[0]
    sprefac = 2 * (3 * np.pi * np.pi)**(1.0/3)
    n43 = rho[0]**(4.0/3)
    mag_grad = np.linalg.norm(rho_data[1:4,:], axis=0)
    s = mag_grad / (sprefac * n43)
    tau_w = mag_grad**2 / (8 * rho)
    tau_unif = (3.0/10) * (3*np.pi**2)**(2.0/3) * rho**(5.0/3)
    tau = rho_data[5]
    alpha = (tau - tau_w) / tau_unif
    return rho, s, alpha, tau_w, tau_unif

def get_nonlocal_data(rho_data, tau_data, ws_radii, coords, weights):
    vals = []

    for i in range(weights.shape[0]):
        ws_radius = ws_radii[i]
        vecs = coords - coords[i]
        rs = np.linalg.norm(vecs, axis=1)
        exp_weights = np.exp(- rs / ws_radius)
        drho = rho_data[1:4,:]
        dvh = np.dot(rho_data[1:4,:] / rs, weights)
        # r dot nabla rho
        rddrho = np.dot(vecs, drho)
        # r dot nabla v_ha
        rddvh = np.dot(vecs, dvh)
        rddvh_int = np.dot(weights, rddvh)
        rddrho_int = np.dot(weights, rddrho)
        dtau = tau_data[1:4,:]
        rddtau = np.dot(vecs, dtau)
        rddtau_int = np.dot(weights, rddtau)

        vals.append([np.linalg.norm(dvh, axis=0), rddvh_int, rddrho_int, rddtau_int])

    return np.array(vals).tranpose()

def regularize_nonlocal_data(nonlocal_data, rho, s, alpha):
    sprefac = 2 * (3 * np.pi * np.pi)**(1.0/3)
    n43 = rho**(4.0/3)
    nonlocal_data[0,:] /= tau_unif
    nonlocal_data[1,:] /= (sprefac * n43)
    nonlocal_data[3,:] /= tau_unif


