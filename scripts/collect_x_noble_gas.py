from mldftdat.gp import DFTGP
from mldftdat.data import get_unique_coord_indexes_spherical
from mldftdat.density import get_exchange_descriptors
from mldftdat.lowmem_analyzers import RKSAnalyzer
from mldftdat.workflow_utils import get_save_dir
from setup_fireworks import SAVE_ROOT
from sklearn.model_selection import train_test_split
import numpy as np 
import os

CALC_TYPE = 'RKS'
FUNCTIONAL = 'LDA_VWN'
MOL_IDS = ['2-He', '10-Ne', '18-Ar', '36-Kr']
BASIS = 'cc-pcvtz'

all_descriptor_data = [[]]
all_values = []

for MOL_ID in MOL_IDS:
    data_dir = get_save_dir(SAVE_ROOT, CALC_TYPE, BASIS, MOL_ID, FUNCTIONAL)
    analyzer = RKSAnalyzer.load(data_dir)
    indexes = get_unique_coord_indexes_spherical(analyzer.grid.coords)
    descriptor_data = get_exchange_descriptors(analyzer.rho_data,
                                               analyzer.tau_data,
                                               analyzer.grid.coords,
                                               analyzer.grid.weights,
                                               restricted = True)
    values = analyzer.get_fx_energy_density()
    all_descriptor_data = np.append(all_descriptor_data, descriptor_data,
                                    axis = 1)
    all_values = np.append(all_values, values, axis = 0)

print(all_descriptor_data.shape)
save_dir = os.path.join(SAVE_ROOT, 'DATASETS')
desc_file = os.path.join(save_dir, 'noblex_rho_data.npz')
fx_file = os.path.join(save_dir, 'noblex_fx.npz')
np.savetxt(desc_file, all_descriptor_data)
np.savetxt(fx_file, all_values)
#gp = DFTGP(descriptor_data, values, 1e-3)
