from mldftdat.analyzers import CCSDAnalyzer
from mldftdat.pyscf_utils import get_lowmem_ee_energy
import numpy as np 

analyzer = CCSDAnalyzer.load('test_files/CCSD_He.hdf5')
ee_ref = analyzer.ee_energy_density

ee_tst = get_lowmem_ee_energy(analyzer.calc, analyzer.mo_vele_mat, analyzer.mo_vals,
                                dm1 = analyzer.mo_rdm1)
#ee_tst += analyzer.ha_energy_density

print(ee_tst.shape, ee_ref.shape)
print(np.linalg.norm(ee_tst-ee_ref))
print(ee_tst[:20], ee_ref[:20])
tot_tst = np.dot(ee_tst, analyzer.grid.weights)
tot_ref = np.dot(ee_ref, analyzer.grid.weights)
print(tot_tst, tot_ref)

print(np.linalg.eig(analyzer.mo_rdm1))
