from mldftdat.models.glh_correlation import *
from mldftdat.workflow_utils import SAVE_ROOT
from argparse import ArgumentParser
import yaml
import logging

import numpy as np

def get_data_dir(args):
    return os.path.join(SAVE_ROOT, 'DATASETS/GLH', args.functional,
                        args.basis, args.datadir)

def store_mols(args):
    save_file = os.path.join(get_data_dir(args), 'mols.yaml')
    with open(args.mol_file, 'r') as f:
        d = yaml.load(f, Loader=yaml.Loader)
    MOL_IDS = d['MOL_IDS']
    IS_RESTRICTED_LIST = d['IS_RESTRICTED_LIST']
    VAL_SET = d.get('VAL_SET')
    mol_id_full = d['mol_id_full']
    os.makedirs(os.path.dirname(save_file))
    store_mols_in_order(save_file, SAVE_ROOT, MOL_IDS, IS_RESTRICTED_LIST,
                        VAL_SET=VAL_SET, mol_id_full=mol_id_full,
                        functional=args.functional, basis=args.basis)

def store_etot(args):
    mol_file = os.path.join(get_data_dir(args), 'mols.yaml')
    save_file = os.path.join(get_data_dir(args), 'etot.npy')
    store_total_energies_dataset(save_file, mol_file,
                                 functional=args.functional,
                                 basis=args.basis)

def store_vv10(args):
    mol_file = os.path.join(get_data_dir(args), 'mols.yaml')
    save_file = os.path.join(get_data_dir(args), 'vv10.npy')
    coeff_sets = args.vv10_coeff
    if len(coeff_sets) % 2 != 0:
        raise ValueError('VV10 coeffs must be in pairs')
    nsets = len(coeff_sets) // 2
    for i in range(nsets):
        coeffs = (coeff_sets[2*i], coeff_sets[2*i+1])
        store_vv10_contribs_dataset(save_file, mol_file, NLC_COEFS=coeffs,
                                    functional=args.functional,
                                    basis=args.basis)

def store_desc(args):
    mol_file = os.path.join(get_data_dir(args), 'mols.yaml')
    save_file = os.path.join(get_data_dir(args), args.desc_name)
    if args.mlfunc is not None:
        mlfunc = load(args.mlfunc)
    else:
        mlfunc = None
    if args.desc_getter is None:
        desc_getter = default_desc_getter
    else:
        if args.desc_module is None:
            raise ValueError('Must specify desc_module if desc_getter is specified.')
        mpath, mname = os.path.dirname(args.desc_module),\
                       os.path.basename(args.desc_module)
        sys.path.append(mpath)
        desc_getter = __import__(mname).globals()[args.desc_getter]
        print(desc_getter)
    store_corr_contribs_dataset(save_file, mol_file, mlfunc,
                                desc_getter=desc_getter,
                                functional=args.functional,
                                basis=args.basis)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    m_desc = 'Compile and train Generalized Local Hybrid models'
    parser = ArgumentParser(description=m_desc)
    subparsers = parser.add_subparsers()
    parser.add_argument('datadir', type=str,
                        help='Directory in which to save mols')
    parser.add_argument('--basis', default=DEFAULT_BASIS, type=str,
                        help='Basis set to use')
    parser.add_argument('--functional', default=DEFAULT_FUNCTIONAL,
                        type=str)

    mol_parser = subparsers.add_parser('store_mols')
    mol_parser.add_argument('mol_file', type=str,
                            help='File in which settings are stored')
    mol_parser.set_defaults(func=store_mols)

    desc_parser = subparsers.add_parser('store_desc')
    desc_parser.add_argument('--desc-name', default='desc',
                             help='Name for descriptor file.')
    desc_parser.add_argument('--desc-getter', default=None,
                             help='Name of function used to evaluate descriptors')
    desc_parser.add_argument('--desc-module', default=None,
                             help='Name of module in which desc_getter is stored')
    desc_parser.add_argument('--mlfunc', default=None,
                             help='ML exchange functional. If None, use exact exchange')
    desc_parser.set_defaults(func=store_desc)

    etot_parser = subparsers.add_parser('store_etot')
    etot_parser.set_defaults(func=store_etot)

    vv10_parser = subparsers.add_parser('store_vv10')
    vv10_parser.add_argument('vv10_coeff', type=float, nargs='+',
                             help='Pairs of VV10 coeffs, e.g. 5.9 0.0093 6.0 0.01')
    vv10_parser.set_defaults(func=store_vv10)

    args = parser.parse_args()

    args.func(args)

