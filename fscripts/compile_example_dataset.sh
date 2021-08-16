export MLDFTDB=test_files
python mldftdat/scripts/compile_dataset2.py test_files/mol_id_lists/atoms.yaml \
    def2-tzvp --functional PBE --version a \
    --gg-a0 1.0 --gg-facmul 0.03125 --gg-amin 0.0625 \
    --suffix TEST --spherical-atom
python mldftdat/scripts/compile_dataset2.py test_files/mol_id_lists/mols.yaml \
    def2-tzvp --functional PBE --version a \
    --gg-a0 1.0 --gg-facmul 0.03125 --gg-amin 0.0625 \
    --suffix TEST
