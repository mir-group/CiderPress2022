export MLDFTDB=test_files
python mldftdat/scripts/train_gp.py test_files/gpr_example.joblib \
    test_files/feat_example.yaml \
    ATOMS 1 MOLS 100 def2-tzvp --functional PBE --suffix TEST \
    -r -d -c 1e-6 -s 42 --heg -x CHACHIYO -on -v c \
    -o 0,1,2,3,4
export MLDFTDB=test_files
python mldftdat/scripts/train_gp.py test_files/agpr_example.joblib \
    test_files/feat_example.yaml \
    ATOMS 1 MOLS 100 def2-tzvp --functional PBE --suffix TEST \
    -r -d -c 1e-6 -s 42 --heg -x CHACHIYO -on -v c \
    -o 0,1,2,3,4 --agpr -as 1e-5,1e-5,1e-2 \
    -vs MOLS 100
