import time
from pyscf import scf
import os, time
import nympy as np
from mldftdat.analyzers import RHFAnalyzer, UHFAnalyzer
from mldftdat.workflow_utils import get_save_dir, SAVE_ROOT
from mldftdat.density import get_exchange_descriptors, get_exchange_descriptors2,\
                             edmgga, LDA_FACTOR
import logging

# TODO NOT FINISHED

def compile_dataset(DATASET_NAME, MOL_IDS, SAVE_ROOT, CALC_TYPE, FUNCTIONAL, BASIS,
                    Analyzer, spherical_atom=False, locx=False,
                    append_all_rho_data=False):

    import time
    all_descriptor_data = None
    all_rho_data = None
    all_values = []

    for MOL_ID in MOL_IDS:
        logging.info('Working on {}'.format(MOL_ID))
        data_dir = get_save_dir(SAVE_ROOT, CALC_TYPE, BASIS, MOL_ID, FUNCTIONAL)
        start = time.monotonic()
        analyzer = Analyzer.load(data_dir + '/data.hdf5')
        end = time.monotonic()
        logging.info('analyzer load time', end - start)
        if spherical_atom:
            start = time.monotonic()
            indexes = get_unique_coord_indexes_spherical(analyzer.grid.coords)
            end = time.monotonic()
            logging.info('index scanning time', end - start)
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
        logging.info('get descriptor time', end - start)
        if locx:
            logging.info('Getting loc fx')
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
    rho_file = os.path.join(save_dir, 'rho.npy')
    desc_file = os.path.join(save_dir, 'desc.npy')
    val_file = os.path.join(save_dir, 'val.npy')
    np.save(rho_file, all_rho_data)
    np.save(desc_file, all_descriptor_data)
    np.save(val_file, all_values)