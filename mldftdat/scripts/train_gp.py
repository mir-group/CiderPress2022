from argparse import ArgumentParser
import numpy as np
from joblib import dump

def parse_dataset(i, val=False):
    if val:
        fname = args.validation_set[2*i]
        n = args.validation_set[2*i+1]
    else:
        fname = args.datasets_list[2*i]
        n = int(args.datasets_list[2*i+1])
    X, y, rho_data = load_descriptors(fname)
    if val:
        # offset in case repeat datasets are used
        X, y, rho_data = X[n//2+1:], y[n//2+1:], rho_data[:,n//2+1:]
    X, y, rho, rho_data = filter_descriptors(X, y, rho_data,
                                             tol=args.density_cutoff)
    if args.randomize:
        np.random.shuffle(X)
        np.random.shuffle(y)
        np.random.shuffle(rho)
        np.random.shuffle(rho_data.T)
    return X, y, rho, rho_data

def parse_list(lststr, T=int):
    return [T(substr) for substr in lststr.split(',')]

parser = ArgumentParser(description='Trains a GP exchange model')

parser.add_argument('save_file', nargs=1, type=str)
parser.add_argument('feature_file', nargs=1, type=str,
                    help='serialized FeatureList object in yaml format')
parser.add_argument('datasets_list', nargs='+',
    help='pairs of dataset names and inverse sampling densities')
parser.add_argument('-r', '--randomize', action='store_true')
parser.add_argument('-c', '--density-cutoff', type=float)
parser.add_argument('-m', '--model-class', type=str, default=None)
parser.add_argument('-k', '--kernel', help='kernel initialization strategy', type=str, default=None)
parser.add_argument('-s', '--seed', help='random seed', default=0, type=int)
parser.add_argument('-v', '--validation-set', nargs='+')
parser.add_argument('-d', '--delete-k', action='store_true',
                    help='Delete L (LL^T=K the kernel matrix) to save disk space. Need to refit when reloading to calculate covariance.')
parser.add_argument('--heg', action='store_true', help='HEG exact constraint')
parser.add_argument('-o', '--desc-order', default=None,
                    help='comma-separated list of descriptor order with no spaces. must start with 0,1.')
parser.add_argument('-l', '--length-scale', default=None,
                    help='comma-separated list initial length-scale guesses')
parser.add_argument('--length-scale-mul', type=float, default=1.0,
                    help='Used for automatic length-scale initial guess')
parser.add_argument('-as', '--agpr-scale', default=None)
parser.add_argument('-ao', '--agpr-order', default=2, type=int)
parser.add_argument('-an', '--agpr-nsingle', default=1, type=int)
parser.add_argument('-x', '--xed-y-code', default='CHACHIYO', type=str)
args = parser.parse_args()

feature_list = FeatureList.load(args.feature_file)

if args.length_scale is not None:
    args.length_scale = parse_list(args.length_scale)
if args.agpr_scale is not None:
    args.agpr_scale = parse_list(args.agpr_scale, T=float)

numpy.random_seed(args.seed)

assert len(args.datasets_list) % 2 == 0, 'Need pairs of entries for datasets list.'
assert len(args.datasets_list) != 0, 'Need training data'
nd = len(args.datasets_list) // 2

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

gpcls = load_gp_class(args.model_class)
gpr = gpcls.from_settings(X, feature_list, args)
gpr.fit(X, y, rho_data)
if args.heg:
    gpr.add_heg_limit()

pred = gpr.xed_to_y(gpr.predict(Xv, rho_datav), rho_datav)
abserr = np.abs(pred - gpr.xed_to_y(yv, rho_datav))
print('FINAL KERNEL', gpr.gp.kernel_)
print('MAE VAL SET', np.mean(abserr))

# Always attach the arguments to the object to keep track of settings.
gpr.args = args
if args.delete_k:
    gpr.L_ = None
dump(gpr, args.save_file)
