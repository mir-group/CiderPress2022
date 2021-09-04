from mldftdat.data import predict_exchange, predict_correlation,\
                          predict_total_exchange_unrestricted
from mldftdat.lowmem_analyzers import RHFAnalyzer, UHFAnalyzer
from mldftdat.workflow_utils import get_save_dir, SAVE_ROOT, load_mol_ids
import numpy as np 
from collections import Counter
from ase.data import chemical_symbols, atomic_numbers,\
                     ground_state_magnetic_moments
from argparse import ArgumentParser
import pandas as pd
from joblib import dump, load
import yaml
import os
import sys

def load_models(model_file):
    with open(model_file, 'r') as f:
        d = yaml.load(f, Loader=yaml.Loader)
        names = []
        models = []
        for name in d:
            names.append(name)
            if d[name] is None:
                models.append(name)
            elif os.path.isfile(d[name]):
                models.append(load(d[name]))
                print(models[-1].desc_version, models[-1].amin, models[-1].a0, models[-1].fac_mul)
            else:
                models.append(d[name])
        return names, models

def error_table(dirs, Analyzer, models, rows):
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
                predict_exchange(analyzer, model=model)
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
    errtbl = np.array([fx_total_rmse, rmise, rmse, rrmise, rrmse]).transpose()

    return (fxlst_true, fxlst_pred, errlst),\
           (columns, rows, errtbl)

def error_table_unrestricted(dirs, Analyzer, models, rows):
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
            fx_total_pred = predict_total_exchange_unrestricted(
                                analyzer, model=model)
            print(fx_total_pred - fx_total_true)
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
    errtbl = np.array([fx_total_rmse, rmise, rmse, rrmise, rrmse]).transpose()

    return (fxlst_true, fxlst_pred, errlst),\
           (columns, rows, errtbl)

def error_table2(dirs, Analyzer, models, rows):
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
    errtbl = np.array([fx_total_rmse, rmise, rmse, rrmise, rrmse]).transpose()

    return (fxlst_true, fxlst_pred, errlst),\
           (columns, rows, errtbl),\
           data_dict

def error_table3(dirs, Analyzer, models, rows, basis, functional):
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
            path = '{}/{}KS/{}/{}/atoms/{}-{}-{}/data.hdf5'.format(
                        SAVE_ROOT, letter, functional, basis, Z, symbol, spin)
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
                                        element_analyzers[Z])
        xef_true, eps_true, neps_true, fx_total_true = \
            predict_exchange(analyzer)
        fxlst_true.append(fx_total_true)
        ae_fxlst_true.append(fx_total_true - fx_total_ref_true)
        count += eps_true.shape[0]
        for i, model in enumerate(models):
            fx_total_ref = 0
            for Z in list(formula.keys()):
                fx_total_ref += formula[Z] \
                                * predict_total_exchange_unrestricted(
                                    element_analyzers[Z],
                                    model=model)
            xef_pred, eps_pred, neps_pred, fx_total_pred = \
                predict_exchange(analyzer, model=model)
            print(fx_total_pred, fx_total_true,
                fx_total_ref, fx_total_ref_true)
            print(fx_total_pred - fx_total_true,
                  fx_total_pred - fx_total_true \
                  - (fx_total_ref - fx_total_ref_true))

            ise[i] += np.dot((eps_pred[condition] - eps_true[condition])**2,
                             weights[condition])
            tse[i] += ((eps_pred[condition] - eps_true[condition])**2).sum()
            rise[i] += np.dot((xef_pred[condition] - xef_true[condition])**2,
                              weights[condition])
            rtse[i] += ((xef_pred[condition] - xef_true[condition])**2).sum()

            fxlst_pred[i].append(fx_total_pred)
            ae_fxlst_pred[i].append(fx_total_pred - fx_total_ref)
            errlst[i].append(fx_total_pred - fx_total_true)
            ae_errlst[i].append(fx_total_pred - fx_total_true \
                                - (fx_total_ref - fx_total_ref_true))
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
    errtbl = np.array([ae_fx_total_rmse, fx_total_rmse, rmise, rmse, rrmise, rrmse]).transpose()

    return (fxlst_true, fxlst_pred, errlst, ae_errlst),\
           (columns, rows, errtbl)

def error_table3u(dirs, Analyzer, models, rows, basis, functional):
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
            path = '{}/{}KS/{}/{}/atoms/{}-{}-{}/data.hdf5'.format(
                        SAVE_ROOT, letter, functional, basis, Z, symbol, spin)
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
                                        element_analyzers[Z])
        fx_total_true = \
            predict_total_exchange_unrestricted(analyzer)
        fxlst_true.append(fx_total_true)
        ae_fxlst_true.append(fx_total_true - fx_total_ref_true)
        count += 1
        for i, model in enumerate(models):
            fx_total_ref = 0
            for Z in list(formula.keys()):
                fx_total_ref += formula[Z] \
                                * predict_total_exchange_unrestricted(
                                    element_analyzers[Z],
                                    model=model)
            fx_total_pred = \
                predict_total_exchange_unrestricted(analyzer, model=model)
            print(fx_total_pred, fx_total_true,
                fx_total_ref, fx_total_ref_true)
            print(fx_total_pred - fx_total_true,
                  fx_total_pred - fx_total_true \
                  - (fx_total_ref - fx_total_ref_true))

            fxlst_pred[i].append(fx_total_pred)
            ae_fxlst_pred[i].append(fx_total_pred - fx_total_ref)
            errlst[i].append(fx_total_pred - fx_total_true)
            ae_errlst[i].append(fx_total_pred - fx_total_true \
                                - (fx_total_ref - fx_total_ref_true))
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
    errtbl = np.array([ae_fx_total_rmse, fx_total_rmse, rmise, rmse, rrmise, rrmse]).transpose()

    return (fxlst_true, fxlst_pred, errlst, ae_errlst),\
           (columns, rows, errtbl)

def error_table_corr(dirs, Analyzer, models, rows):
    from collections import Counter
    from ase.data import chemical_symbols, atomic_numbers, ground_state_magnetic_moments
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
        analyzer.get_ao_rho_data()
        atoms = [atomic_numbers[a[0]] for a in analyzer.mol._atom]
        formula = Counter(atoms)
        element_analyzers = {}
        for Z in list(formula.keys()):
            symbol = chemical_symbols[Z]
            spin = int(ground_state_magnetic_moments[Z])
            letter = '' if spin == 0 else 'U'
            path = '{}/{}CCSD/aug-cc-pvtz/atoms/{}-{}-{}/data.hdf5'.format(
                        SAVE_ROOT, letter, Z, symbol, spin)
            if letter == '':
                element_analyzers[Z] = CCSDAnalyzer.load(path)
            else:
                element_analyzers[Z] = UCCSDAnalyzer.load(path)
            element_analyzers[Z].get_ao_rho_data()
        weights = analyzer.grid.weights
        rho = analyzer.rho_data[0,:]
        condition = rho > 3e-5
        fx_total_ref_true = 0
        for Z in list(formula.keys()):
            restricted = True if type(element_analyzers[Z]) == CCSDAnalyzer else False
            _, _, fx_total_ref_tmp = predict_correlation(
                                        element_analyzers[Z],
                                        restricted=restricted)
            fx_total_ref_true += formula[Z] * fx_total_ref_tmp
        eps_true, neps_true, fx_total_true = \
            predict_correlation(analyzer)
        fxlst_true.append(fx_total_true)
        ae_fxlst_true.append(fx_total_true - fx_total_ref_true)
        count += eps_true.shape[0]
        for i, model in enumerate(models):
            fx_total_ref = 0
            for Z in list(formula.keys()):
                restricted = True if type(element_analyzers[Z]) == CCSDAnalyzer else False
                _, _, fx_total_tmp = predict_correlation(
                                        element_analyzers[Z],
                                        model=model, restricted=restricted)
                fx_total_ref += formula[Z] * fx_total_tmp
            eps_pred, neps_pred, fx_total_pred = \
                predict_correlation(analyzer, model=model)
            print(fx_total_pred - fx_total_true,
                  fx_total_pred - fx_total_true \
                  - (fx_total_ref - fx_total_ref_true))

            ise[i] += np.dot((eps_pred[condition] - eps_true[condition])**2,
                             weights[condition])
            tse[i] += ((eps_pred[condition] - eps_true[condition])**2).sum()

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
    errtbl = np.array([ae_fx_total_rmse, fx_total_rmse, rmise, rmse, rrmise, rrmse]).transpose()

    return (fxlst_true, fxlst_pred, errlst, ae_errlst),\
           (columns, rows, errtbl)


if __name__ == '__main__':
    m_desc = 'Compute, print, and return errors of different methods for prediction of exchange and correlation energies.'

    parser = ArgumentParser(description=m_desc)
    parser.add_argument('version', type=str,
                        help=('1, 2, 3, u, or c.\n'
                        '1: Total exchange error for spin-restricted systems\n'
                        '2: Same as above but also returns data for ML descriptors\n'
                        '3: Total and atomization exchange error for spin-restricted systems\n'
                        'u: Total exchange error for spin-unrestricted systems.'))
                        #'u: Total exchange error for spin-unrestricted systems\n'
                        #'c: Total correlation exchange error.'))
    parser.add_argument('model_file', type=str,
                        help='yaml file containing list of models and how to load them.')
    parser.add_argument('mol_file', type=str,
                        help='yaml file containing list of directories and calc type')
    parser.add_argument('basis', metavar='basis', type=str,
                        help='basis set code')
    parser.add_argument('--functional', metavar='functional', type=str, default=None,
                        help='exchange-correlation functional, HF for Hartree-Fock')
    parser.add_argument('--save-file', type=str, default=None,
                        help='If not None, save error table to this file.')
    args = parser.parse_args()

    calc_type, mol_ids = load_mol_ids(args.mol_file)
    #if args.version.lower() == 'c' and ('CCSD' not in calc_type):
    #    raise ValueError('Wrong calc_type')
    #elif 'CCSD' in calc_type:
    #    raise ValueError('Wrong calc_type')
    if 'CCSD' in calc_type:
        raise ValueError('CCSD analyzers not currently supported.')
    elif 'U' in calc_type and (args.version not in ['u', 'c']):
        raise ValueError('Wrong calc_type')

    if calc_type in ['RKS', 'RHF']:
        Analyzer = RHFAnalyzer
    elif calc_type in ['UKS', 'UHF']:
        Analyzer = UHFAnalyzer
    #elif calc_type == 'CCSD':
    #    Analyzer = CCSDAnalyzer
    #elif calc_type == 'UCCSD':
    #    Analyzer = UCCSDAnalyzer
    else:
        raise ValueError('Incorrect or unsupported calc_type {}'.format(calc_type))

    dirs = []
    for mol_id in mol_ids:
        dirs.append(get_save_dir(SAVE_ROOT, calc_type, args.basis,
                                 mol_id, args.functional))

    rows, models = load_models(args.model_file)

    if args.version == '1':
        res1, res2 = error_table(dirs, Analyzer, models, rows)
        fxlst_true, fxlst_pred, errlst = res1
        columns, rows, errtbl = res2
        print(res1)
        print(res2)
        df = pd.DataFrame(errtbl, index=rows, columns=columns)
        print(df.to_latex())
    elif args.version == '2':
        res1, res2, res3 = error_table2(dirs, Analyzer, models, rows)
        fxlst_true, fxlst_pred, errlst = res1
        columns, rows, errtbl = res2
        print(res1)
        for sublst in res1[2]:
            print(np.mean(sublst), np.std(sublst))
        print(res2)
        for key in res3.keys():
            print(key)
            for key2 in res3[key].keys():
                print(key2, res3[key][key2].shape)
        df = pd.DataFrame(errtbl, index=rows, columns=columns)
        from pyscf import lib
        lib.chkfile.dump('errtbl_out_%d.hdf5' % NUM, 'data', res3)
        print(df.to_latex())
    elif args.version == '3':
        res1, res2 = error_table3(dirs, Analyzer, models, rows, args.basis, args.functional)
        fxlst_true, fxlst_pred, errlst, ae_errlst = res1
        columns, rows, errtbl = res2
        print(res1)
        for sublst in res1[2]:
            print(np.mean(sublst), np.std(sublst))
        df = pd.DataFrame(errtbl, index=rows, columns=columns)
        print(df.to_latex())
    elif args.version == 'u':
        res1, res2 = error_table3u(dirs, Analyzer, models, rows, args.basis, args.functional)
        fxlst_true, fxlst_pred, errlst, ae_errlst = res1
        columns, rows, errtbl = res2
        print(res1)
        for sublst in res1[2]:
            print(np.mean(sublst), np.std(sublst))
        df = pd.DataFrame(errtbl, index=rows, columns=columns)
        print(df.to_latex())
    elif args.version == 'c':
        res1, res2 = error_table_corr(dirs, Analyzer, models, rows)
        fxlst_true, fxlst_pred, errlst, ae_errlst = res1
        columns, rows, errtbl = res2
        print(res1)
        for sublst in res1[2]:
            print(np.mean(sublst), np.std(sublst))
        df = pd.DataFrame(errtbl, index=rows, columns=columns)
        print(df.to_latex())
    
    if args.save_file is not None:
        df.to_csv(args.save_file)

