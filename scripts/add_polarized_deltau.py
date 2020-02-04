import os

for root, dirs, files in os.walk(os.environ['MLDFTDB']):
    if ('UHF' in root or 'UKS' in root or 'UCCSD' in root) and 'data.hdf5' in files:
        fname = os.path.join(root, 'data.hdf5')
        old_size = os.path.getsize(fname)
        analyzer = UHFAnalyzer.load(fname, max_mem = 10)
        print(analyzer.tau_data.shape)
        analyzer.tau_data = None
        analyzer.get_ao_rho_data()
        print(analyzer.tau_data.shape)
        analyzer.dump(fname)
        new_size = os.path.getsize(fname)
        assert new_size > old_size
        print(root)
        break
