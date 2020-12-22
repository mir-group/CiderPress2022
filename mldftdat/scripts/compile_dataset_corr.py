import time
from pyscf import scf
import os, time
import nympy as np
from mldftdat.analyzers import RHFAnalyzer, UHFAnalyzer
from mldftdat.workflow_utils import get_save_dir
from mldftdat.density import get_exchange_descriptors, get_exchange_descriptors2,\
                             edmgga, LDA_FACTOR
import logging

def compile_dataset_corr(DATASET_NAME, MOL_IDS, SAVE_ROOT, CALC_TYPE, BASIS,
                    Analyzer, spherical_atom = False, locx = False, lam = 0.5,
                    version = 'a'):

    import time
    from pyscf import scf
    all_descriptor_data_u = None
    all_descriptor_data_d = None
    all_rho_data_u = None
    all_rho_data_d = None
    all_values = []
    all_weights = []
    cutoffs = []

    for MOL_ID in MOL_IDS:
        logging.info('Working on {}'.format(MOL_ID))
        data_dir = get_save_dir(SAVE_ROOT, CALC_TYPE, BASIS, MOL_ID)
        start = time.monotonic()
        analyzer = Analyzer.load(data_dir + '/data.hdf5')
        analyzer.get_ao_rho_data()
        if type(analyzer.calc) == scf.hf.RHF or CALC_TYPE == 'CCSD':
            restricted = True
        else:
            restricted = False
        end = time.monotonic()
        logging.info('analyzer load time', end - start)
        if spherical_atom:
            start = time.monotonic()
            indexes = get_unique_coord_indexes_spherical(analyzer.grid.coords)
            end = time.monotonic()
            logging.info('index scanning time', end - start)
        start = time.monotonic()
        if restricted:
            descriptor_data = get_exchange_descriptors2(analyzer,
                restricted = True, version=version)
            descriptor_data_u = descriptor_data
            descriptor_data_d = descriptor_data
        else:
            descriptor_data_u, descriptor_data_d = \
                              get_exchange_descriptors2(analyzer,
                                    restricted = False, version=version)
        end = time.monotonic()
        logging.info('get descriptor time', end - start)
        
        values = analyzer.get_corr_energy_density()

        rho_data = analyzer.rho_data
        if not restricted:
            rho_data_u, rho_data_d = 2 * rho_data[0], 2 * rho_data[1]
        else:
            rho_data_u, rho_data_d = rho_data, rho_data
        if spherical_atom:
            values = values[indexes]
            descriptor_data_u = descriptor_data_u[:,indexes]
            descriptor_data_d = descriptor_data_d[:,indexes]
            rho_data_u = rho_data_u[:,indexes]
            rho_data_d = rho_data_d[:,indexes]

        if all_descriptor_data_u is None:
            all_descriptor_data_u = descriptor_data_u
            all_descriptor_data_d = descriptor_data_d
        else:
            all_descriptor_data_u = np.append(all_descriptor_data_u, descriptor_data_u,
                                              axis = 1)
            all_descriptor_data_d = np.append(all_descriptor_data_d, descriptor_data_d,
                                              axis = 1)
        if all_rho_data_u is None:
            all_rho_data_u = rho_data_u
            all_rho_data_d = rho_data_d
        else:
            all_rho_data_u = np.append(all_rho_data_u, rho_data_u, axis=1)
            all_rho_data_d= np.append(all_rho_data_d, rho_data_d, axis=1)
        all_values = np.append(all_values, values)
        all_weights = np.append(all_weights, analyzer.grid.weights)
        if not restricted:
            # two copies for unrestricted case
            all_weights = np.append(all_weights, analyzer.grid.weights)
        cutoffs.append(all_values.shape[0])

    save_dir = os.path.join(SAVE_ROOT, 'DATASETS', DATASET_NAME)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    rho_file = os.path.join(save_dir, 'rho.npy')
    desc_file = os.path.join(save_dir, 'desc.npy')
    val_file = os.path.join(save_dir, 'val.npy')
    wt_file = os.path.join(save_dir, 'wt.npy')
    cut_file = os.path.join(save_dir, 'cut.npy')
    np.save(rho_file, np.stack([all_rho_data_u, all_rho_data_d], axis=0))
    np.save(desc_file, np.stack([all_descriptor_data_u, all_descriptor_data_d], axis=0))
    np.save(val_file, all_values)
    np.save(wt_file, all_weights)
    np.save(cut_file, np.array(cutoffs))
