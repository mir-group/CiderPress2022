# CiderPress

Tools for training and evaluating CIDER functionals for use in Density Functional Theory calculations.

## What is the CIDER model?

Machine Learning (ML) has recently gained attention as a means to fit more accurate Exchange-Correlation (XC) functionals for use in Density Functional Theory (DFT). We have recently developed the CIDER model, a set of features that efficiently describe the electronic environment around a reference point in real space and can be used as input for ML-XC functionals. **CIDER** stands for **C**ompressed scale-**I**nvariant **DE**nsity **R**epresentation, which refers to the fact that the descriptors are invariant under squishing or expanding of the density while maintaining its shape. This property makes it efficient for learning the XC functional, especially the exchange energy.

## How can I run a CIDER calculation?

See the `test_scripts/run_example_calculation.py` for examples of running calculations with CIDER functionals. The B3LYP-CIDER functional (Example 3) is stored in this repo using git-lfs, while the others can be generated with the example training scripts (see below). WARNING: Functional derivatives for the un-mapped GP functional are experimental, and may be buggy. Tests show the analytical derivatives compared to finite difference have significant errors. The MAPPED (Spline) Functionals have well-tested and accurate functional derivatives.

## How can I train a CIDER functional?

See the scripts contained in test_scripts/ for examples of running the different scripts used to train CIDER functionals. Calling `sh test_scripts/workflow.sh` runs a complete workflow to generate a GP functional (directly calls a Gaussian Process to evaluate a functional) and a Spline Functional (maps the GP to a spline for computational efficiency). See `mldftdat.scripts` for the command options for different training-related scripts.

## Questions and Comments

Find a bug? Areas of code unclearly documented? Other questions? Feel free to contact
Kyle Bystrom at kylebystrom@gmail.com AND/OR create an issue on the Github page at https://github.com/mir-group/CiderPress.
