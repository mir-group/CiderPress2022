import os
from mldftdat.lowmem_analyzers import CCSDAnalyzer, UCCSDAnalyzer

for root, dirs, files in os.walk(os.environ['MLDFTDB']):
    if ('CCSD' in root) and 'data.hdf5' in files:
        print(root)
        fname = os.path.join(root, 'data.hdf5')
        old_size = os.path.getsize(fname)
        if 'UCCSD' in root:
            analyzer = UCCSDAnalyzer.load(fname, max_mem = 1000)
        else:
            analyzer = CCSDAnalyzer.load(fname, max_mem = 1000)
        analyzer.get_corr_energy_density()
        print(analyzer.ecorr_dens.shape)
        analyzer.dump(fname)
        #assert new_size > old_size
