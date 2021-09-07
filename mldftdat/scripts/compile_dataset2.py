import time
from pyscf import scf
import os, time
import numpy as np
from mldftdat.lowmem_analyzers import RHFAnalyzer, UHFAnalyzer
from mldftdat.workflow_utils import get_save_dir, SAVE_ROOT, load_mol_ids
from mldftdat.density import get_exchange_descriptors2, LDA_FACTOR, GG_AMIN
from mldftdat.data import get_unique_coord_indexes_spherical
import logging
import yaml

from argparse import ArgumentParser

"""
Script to compile a dataset from the CIDER DB for training a CIDER functional.
"""

def compile_dataset2(DATASET_NAME, MOL_IDS, SAVE_ROOT, CALC_TYPE, FUNCTIONAL, BASIS,
                    spherical_atom=False, locx=False, lam=0.5,
                    version='a', **gg_kwargs):

    all_descriptor_data = None
    all_rho_data = None
    all_values = []
    all_weights = []
    cutoffs = []

    if locx:
        raise ValueError('locx setting not supported in this version! (but might be later)')
        Analyzer = loc_analyzers.UHFAnalyzer if 'U' in CALC_TYPE \
                   else loc_analyzers.RHFAnalyzer
    else:
        Analyzer = UHFAnalyzer if 'U' in CALC_TYPE else RHFAnalyzer

    for MOL_ID in MOL_IDS:
        logging.info('Computing descriptors for {}'.format(MOL_ID))
        data_dir = get_save_dir(SAVE_ROOT, CALC_TYPE, BASIS, MOL_ID, FUNCTIONAL)
        start = time.monotonic()
        analyzer = Analyzer.load(data_dir + '/data.hdf5')
        analyzer.get_ao_rho_data()
        if type(analyzer.calc) == scf.hf.RHF:
            restricted = True
        else:
            restricted = False
        end = time.monotonic()
        logging.info('Analyzer load time {}'.format(end - start))
        if spherical_atom:
            start = time.monotonic()
            indexes = get_unique_coord_indexes_spherical(analyzer.grid.coords)
            end = time.monotonic()
            logging.info('Index scanning time {}'.format(end - start))
        start = time.monotonic()
        if restricted:
            descriptor_data = get_exchange_descriptors2(
                analyzer, restricted=True, version=version,
                **gg_kwargs
            )
        else:
            descriptor_data_u, descriptor_data_d = \
                              get_exchange_descriptors2(
                                analyzer, restricted=False, version=version,
                                **gg_kwargs
                              )
            descriptor_data = np.append(descriptor_data_u, descriptor_data_d,
                                        axis = 1)
        end = time.monotonic()
        logging.info('Get descriptor time {}'.format(end - start))
        if locx:
            logging.info('Getting loc fx with lambda={}'.format(lam))
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
            weights = analyzer.grid.weights[indexes]
        else:
            weights = analyzer.grid.weights

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
        all_weights = np.append(all_weights, weights)
        if not restricted:
            # two copies for unrestricted case
            all_weights = np.append(all_weights, weights)
        cutoffs.append(all_values.shape[0])

    DATASET_NAME = os.path.basename(DATASET_NAME)
    save_dir = os.path.join(SAVE_ROOT, 'DATASETS',
                            FUNCTIONAL, BASIS, version, DATASET_NAME)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    rho_file = os.path.join(save_dir, 'rho.npy')
    desc_file = os.path.join(save_dir, 'desc.npy')
    val_file = os.path.join(save_dir, 'val.npy')
    wt_file = os.path.join(save_dir, 'wt.npy')
    cut_file = os.path.join(save_dir, 'cut.npy')
    np.save(rho_file, all_rho_data)
    np.save(desc_file, all_descriptor_data)
    np.save(val_file, all_values)
    np.save(wt_file, all_weights)
    np.save(cut_file, np.array(cutoffs))
    settings = {
        'DATASET_NAME': DATASET_NAME,
        'MOL_IDS': MOL_IDS,
        'SAVE_ROOT': SAVE_ROOT,
        'CALC_TYPE': CALC_TYPE,
        'FUNCTIONAL': FUNCTIONAL,
        'BASIS': BASIS,
        'spherical_atom': spherical_atom,
        'locx': locx,
        'lam': lam,
        'version': version
    }
    settings.update(gg_kwargs)
    with open(os.path.join(save_dir, 'settings.yaml'), 'w') as f:
        yaml.dump(settings, f)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    m_desc = 'Compile datset of exchange descriptors'

    parser = ArgumentParser(description=m_desc)
    parser.add_argument('mol_id_file', type=str,
                        help='yaml file from which to read mol_ids to parse')
    parser.add_argument('basis', metavar='basis', type=str,
                        help='basis set code')
    parser.add_argument('--functional', metavar='functional', type=str, default=None,
                        help='exchange-correlation functional, HF for Hartree-Fock')
    parser.add_argument('--spherical-atom', action='store_true',
                        default=False, help='whether dataset contains spherical atoms')
    parser.add_argument('--locx', action='store_true',
                        default=False, help='whether to use transformed exchange hole')
    parser.add_argument('--lam', default=0.5, type=float,
                        help='lambda factor for exchange hole, only used if locx=True')
    parser.add_argument('--version', default='c', type=str,
                        help='version of descriptor set. Default c')
    parser.add_argument('--gg-a0', default=8.0, type=float)
    parser.add_argument('--gg-facmul', default=1.0, type=float)
    parser.add_argument('--gg-amin', default=GG_AMIN, type=float)
    parser.add_argument('--suffix', default=None, type=str,
                        help='customize data directories with this suffix')
    args = parser.parse_args()

    version = args.version.lower()
    assert version in ['a', 'b', 'c']

    calc_type, mol_ids = load_mol_ids(args.mol_id_file)
    assert ('HF' in calc_type) or (args.functional is not None),\
           'Must specify functional if not using HF reference.'
    if args.mol_id_file.endswith('.yaml'):
        mol_id_code = args.mol_id_file[:-5]
    else:
        mol_id_code = args.mol_id_file

    dataname = 'XTR{}_{}'.format(version.upper(), mol_id_code.upper())
    if args.spherical_atom:
        pass#dataname = 'SPH_' + dataname
    if args.locx:
        dataname = 'LOCX_' + dataname
    if args.suffix is not None:
        dataname = dataname + '_' + args.suffix

    # TODO remove this if locx supported in the future
    args.locx = False

    if version == 'c':
        compile_dataset2(
            dataname, mol_ids, SAVE_ROOT, calc_type, args.functional, args.basis,
            spherical_atom=args.spherical_atom, locx=args.locx, lam=args.lam,
            version=version, a0=args.gg_a0, fac_mul=args.gg_facmul,
            amin=args.gg_amin
        )
    else:
        compile_dataset2(
            dataname, mol_ids, SAVE_ROOT, calc_type, args.functional, args.basis, 
            spherical_atom=args.spherical_atom, locx=args.locx, lam=args.lam,
            version=version, a0=args.gg_a0, fac_mul=args.gg_facmul,
            amin=args.gg_amin

        )
