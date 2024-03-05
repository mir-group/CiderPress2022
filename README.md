# CiderPress

**NOTICE**: This repository only contains the code for our 2022 paper in JCTC (https://pubs.acs.org/doi/10.1021/acs.jctc.1c00904). If you are looking for the code to run the CIDER calculations in our 2023 preprint (https://arxiv.org/abs/2303.00682), please see the new codebase at https://github.com/mir-group/CiderPressLite.

Tools for training and evaluating CIDER functionals for use in Density Functional Theory calculations.

**NOTE**: Due to issues with LFS bandwidth, the CIDER_X_AHW.yaml file containing the functional from the JCTC2022 paper has been removed from the repository and is available here instead: https://drive.google.com/file/d/113fMxP3ElYEAFKF-qso7wlNnrKRTLl_Y/view?usp=sharing.

To use the repository, first clone it and then download CIDER_X_AHW.yaml and place it in the functionals/ directory. We will work on a more permanent solution for storing the data in the near future. Please post an issue if you have difficulty accessing anything.

Also note that we will soon release a new version of CIDER featuring more accurate, more transferable, and much faster functionals (https://arxiv.org/abs/2303.00682), but are still in the peer review process.

## What is the CIDER model?

Machine Learning (ML) has recently gained attention as a means to fit more accurate Exchange-Correlation (XC) functionals for use in Density Functional Theory (DFT). We have recently developed the CIDER model, a set of features that efficiently describe the electronic environment around a reference point in real space and can be used as input for ML-XC functionals. **CIDER** stands for **C**ompressed scale-**I**nvariant **DE**nsity **R**epresentation, which refers to the fact that the descriptors are invariant under squishing or expanding of the density while maintaining its shape. This property makes it efficient for learning the XC functional, especially the exchange energy.

## How can I run a CIDER calculation?

See the `test_scripts/run_example_calculation.py` for examples of running calculations with CIDER functionals. The B3LYP-CIDER functional (Example 3) is stored in this repo using git-lfs, while the others can be generated with the example training scripts (see below). WARNING: Functional derivatives for the un-mapped GP functional are experimental, and may be buggy. Tests show the analytical derivatives compared to finite difference have significant errors. The MAPPED (Spline) Functionals have well-tested and accurate functional derivatives.

## How can I train a CIDER functional?

See the scripts contained in test_scripts/ for examples of running the different scripts used to train CIDER functionals. Calling `sh test_scripts/workflow.sh` runs a complete workflow to generate a GP functional (directly calls a Gaussian Process to evaluate a functional) and a Spline Functional (maps the GP to a spline for computational efficiency). See `mldftdat.scripts` for the command options for different training-related scripts.

## External Licenses

Parts of this code (namely `mldftdat.dft.numint` and `mldftdat.models.kernels`) copy a decent portion of code from PySCF (https://github.com/pyscf/pyscf) and scikit-learn (https://github.com/scikit-learn/scikit-learn), respectively. As such, these two modules contain the appropriate copyright notices, and the licenses for these codes are stored in the `EXTERNAL_LICENSES/` directory.

## Questions and Comments

Find a bug? Areas of code unclearly documented? Other questions? Feel free to contact
Kyle Bystrom at kylebystrom@gmail.com AND/OR create an issue on the Github page at https://github.com/mir-group/CiderPress.

## Citing

If you use the CiderPress code or our functional design approach in your research, please cite the following work:
```
@article{Bystrom2022,
author = {Bystrom, Kyle and Kozinsky, Boris},
doi = {10.1021/acs.jctc.1c00904},
issn = {1549-9618},
journal = {J. Chem. Theory Comput.},
month = {apr},
number = {4},
pages = {2180--2192},
title = {{CIDER: An Expressive, Nonlocal Feature Set for Machine Learning Density Functionals with Exact Constraints}},
url = {https://pubs.acs.org/doi/10.1021/acs.jctc.1c00904},
volume = {18},
year = {2022}
}
```
