from argparse import ArgumentParser
import os
import numpy as np
from joblib import dump
from mldftdat.workflow_utils import SAVE_ROOT
from mldftdat.models.gp import *
from mldftdat.data import load_descriptors, filter_descriptors
import yaml

def parse_settings(args):
    fname = args.datasets_list[0]
    if args.suffix is not None:
        fname = fname + '_' + args.suffix
    fname = os.path.join(SAVE_ROOT, 'DATASETS', args.functional,
                         args.basis, args.version, fname)
    print(fname)
    with open(os.path.join(fname, 'settings.yaml'), 'r') as f:
        d = yaml.load(f, Loader=yaml.Loader)
    args.gg_a0 = d.get('a0')
    args.gg_amin = d.get('amin')
    args.gg_facmul = d.get('fac_mul')

def parse_dataset(args, i, val=False):
    if val:
        fname = args.validation_set[2*i]
        n = int(args.validation_set[2*i+1])
    else:
        fname = args.datasets_list[2*i]
        n = int(args.datasets_list[2*i+1])
    if args.suffix is not None:
        fname = fname + '_' + args.suffix
    fname = os.path.join(SAVE_ROOT, 'DATASETS', args.functional,
                         args.basis, args.version, fname)
    print(fname)
    X, y, rho_data = load_descriptors(fname)
    if val:
        # offset in case repeat datasets are used
        X, y, rho_data = X[n//2+1:,:], y[n//2+1:], rho_data[:,n//2+1:]
    X, y, rho, rho_data = filter_descriptors(X, y, rho_data,
                                             tol=args.density_cutoff)
    print(X.shape, n)
    if args.randomize:
        inds = np.arange(X.shape[0])
        np.random.shuffle(inds)
        X = X[inds,:]
        y = y[inds]
        rho = rho[inds]
        rho_data = rho_data[:,inds]
    return X[::n,:], y[::n], rho[::n], rho_data[:,::n]

def parse_list(lststr, T=int):
    return [T(substr) for substr in lststr.split(',')]

def main():
    parser = ArgumentParser(description='Trains a GP exchange model')

    parser.add_argument('save_file', type=str)
    parser.add_argument('feature_file', type=str,
                        help='serialized FeatureList object in yaml format')
    parser.add_argument('datasets_list', nargs='+',
        help='pairs of dataset names and inverse sampling densities')
    parser.add_argument('basis', metavar='basis', type=str,
                        help='basis set code')
    parser.add_argument('--functional', metavar='functional', type=str, default=None,
                        help='exchange-correlation functional, HF for Hartree-Fock')
    parser.add_argument('-r', '--randomize', action='store_true')
    parser.add_argument('-c', '--density-cutoff', type=float, default=1e-4)
    #parser.add_argument('-m', '--model-class', type=str, default=None)
    #parser.add_argument('-k', '--kernel', help='kernel initialization strategy', type=str, default=None)
    parser.add_argument('-s', '--seed', help='random seed', default=0, type=int)
    parser.add_argument('-vs', '--validation-set', nargs='+')
    parser.add_argument('-d', '--delete-k', action='store_true',
                        help='Delete L (LL^T=K the kernel matrix) to save disk space. Need to refit when reloading to calculate covariance.')
    parser.add_argument('--heg', action='store_true', help='HEG exact constraint')
    parser.add_argument('--tail', action='store_true', help='atomic tail exact constraint')
    parser.add_argument('-o', '--desc-order', default=None,
                        help='comma-separated list of descriptor order with no spaces. must start with 0,1.')
    parser.add_argument('-l', '--length-scale', default=None,
                        help='comma-separated list initial length-scale guesses')
    parser.add_argument('--length-scale-mul', type=float, default=1.0,
                        help='Used for automatic length-scale initial guess')
    parser.add_argument('-a', '--agpr', action='store_true',
                        help='Whether to use Additive RBF. If False, use RBF')
    parser.add_argument('-as', '--agpr-scale', default=None)
    parser.add_argument('-ao', '--agpr-order', default=2, type=int)
    parser.add_argument('-an', '--agpr-nsingle', default=1, type=int)
    parser.add_argument('-x', '--xed-y-code', default='CHACHIYO', type=str)
    parser.add_argument('-on', '--optimize-noise', action='store_true',
                        help='Whether to optimzie exponent of density noise.')
    parser.add_argument('-v', '--version', default='c', type=str,
                        help='version of descriptor set. Default c')
    parser.add_argument('--suffix', default=None, type=str,
                        help='customize data directories with this suffix')
    args = parser.parse_args()

    parse_settings(args)

    np.random.seed(args.seed)

    feature_list = FeatureList.load(args.feature_file)

    if args.length_scale is not None:
        args.length_scale = parse_list(args.length_scale, T=float)
    if args.agpr_scale is not None:
        args.agpr_scale = parse_list(args.agpr_scale, T=float)
    if args.desc_order is not None:
        args.desc_order = parse_list(args.desc_order)

    assert len(args.datasets_list) % 2 == 0, 'Need pairs of entries for datasets list.'
    assert len(args.datasets_list) != 0, 'Need training data'
    nd = len(args.datasets_list) // 2

    if args.validation_set is None:
        nv = 0
    else:
        assert len(args.validation_set) % 2 == 0, 'Need pairs of entries for datasets list.'
        nv = len(args.validation_set) // 2

    X, y, rho, rho_data = parse_dataset(args, 0)
    for i in range(1, nd):
        Xn, yn, rhon, rho_datan, = parse_dataset(args, i)
        X = np.append(X, Xn, axis=0)
        y = np.append(y, yn, axis=0)
        rho = np.append(rho, rhon, axis=0)
        rho_data = np.append(rho_data, rho_datan, axis=1)
    if nv != 0:
        Xv, yv, rhov, rho_datav = parse_dataset(args, 0, val=True)
    for i in range(1, nv):
        Xn, yn, rhon, rho_datan, = parse_dataset(args, i, val=True)
        Xv = np.append(Xv, Xn, axis=0)
        yv = np.append(yv, yn, axis=0)
        rhov = np.append(rhov, rhon, axis=0)
        rho_datav = np.append(rho_datav, rho_datan, axis=1)

    gpcls = DFTGPR
    gpr = gpcls.from_settings(X, feature_list, args)
    gpr.fit(X, y, add_heg=args.heg, add_tail=args.tail)
    #if args.heg:
    #    gpr.add_heg_limit()

    print('FINAL KERNEL', gpr.gp.kernel_)
    if nv != 0:
        pred = gpr.xed_to_y(gpr.predict(Xv), Xv)
        abserr = np.abs(pred - gpr.xed_to_y(yv, Xv))
        print('MAE VAL SET', np.mean(abserr))

    # Always attach the arguments to the object to keep track of settings.
    gpr.args = args
    if args.delete_k:
        gpr.L_ = None
    dump(gpr, args.save_file)

if __name__ == '__main__':
    main()
