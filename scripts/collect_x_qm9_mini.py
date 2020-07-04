#from mldftdat.gp import DFTGP
from mldftdat.data import get_unique_coord_indexes_spherical
from mldftdat.density import get_exchange_descriptors
from mldftdat.lowmem_analyzers import RHFAnalyzer
from mldftdat.workflow_utils import get_save_dir
from setup_fireworks import SAVE_ROOT
from sklearn.model_selection import train_test_split
import numpy as np 
import os

CALC_TYPE = 'RKS'
FUNCTIONAL = 'LDA_VWN'
MOL_IDS = next(os.walk(get_save_dir(SAVE_ROOT, CALC_TYPE, 'aug-cc-pvtz', 'qm9', FUNCTIONAL)))[1]
MOL_IDS = ['qm9/{}'.format(s) for s in MOL_IDS]
MOL_IDS = ['qm9/3-H2O']
print(MOL_IDS)
#exit()
BASIS = 'aug-cc-pvtz'

all_descriptor_data = None
all_values = []

for MOL_ID in MOL_IDS:
    print('Working on {}'.format(MOL_ID))
    data_dir = get_save_dir(SAVE_ROOT, CALC_TYPE, BASIS, MOL_ID, FUNCTIONAL)
    print('load analyzer')
    analyzer = RHFAnalyzer.load(data_dir + '/data.hdf5')
    print('get descriptors')
    descriptor_data = get_exchange_descriptors(analyzer.rho_data,
                                               analyzer.tau_data,
                                               analyzer.grid.coords,
                                               analyzer.grid.weights,
                                               restricted = True)
    values = analyzer.get_fx_energy_density()
    print(np.min(values))
    descriptor_data = descriptor_data
    print(np.max(descriptor_data[0]))
    print(values.shape, descriptor_data.shape)
    if all_descriptor_data is None:
        all_descriptor_data = descriptor_data
    else:
        all_descriptor_data = np.append(all_descriptor_data, descriptor_data,
                                        axis = 1)
    all_values = np.append(all_values, values)

print(all_descriptor_data.shape, all_values.shape)
save_dir = os.path.join(SAVE_ROOT, 'DATASETS')
#desc_file = os.path.join(save_dir, 'qm9_x_rho_data.npz')
#fx_file = os.path.join(save_dir, 'qm9_x_fx.npz')
#np.savetxt(desc_file, all_descriptor_data)
#np.savetxt(fx_file, all_values)
#gp = DFTGP(descriptor_data, values, 1e-3)
