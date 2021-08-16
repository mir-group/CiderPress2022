from pyscf import gto, dft
from mldftdat.lowmem_analyzers import RHFAnalyzer
import os

mols = {
    'ATOMS/He': 'He',
    'ATOMS/Ne': 'Ne',
    'MOLS/H2': 'H 0 0 0; H 0 0 0.74',
    'MOLS/HF': 'H 0 0 0; F 0 0 0.93'
}

for mol_id, geom in mols.items():
    mol = gto.M(atom=geom, basis='def2-tzvp')
    ks = dft.RKS(mol)
    ks.xc = 'PBE'
    ks.kernel()
    ana = RHFAnalyzer(ks)
    ana.perform_full_analysis()
    os.makedirs('test_files/RKS/PBE/def2-tzvp/{}'.format(mol_id), exist_ok=True)
    ana.dump('test_files/RKS/PBE/def2-tzvp/{}/data.hdf5'.format(mol_id))
