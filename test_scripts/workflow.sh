python test_scripts/compute_example_data.py
sh test_scripts/compile_example_dataset.sh
python test_scripts/make_feat_list.py
sh test_scripts/train_example_gp.sh
sh test_scripts/gp_to_spline_example.sh