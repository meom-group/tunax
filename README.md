[![Documentation Status](https://readthedocs.org/projects/tunax/badge/?version=latest)](https://tunax.readthedocs.io/en/latest/?badge=latest)

# Description
This package provides a framework for calibrating the parameters of vertical physics schemes of ocean circulation models using variational optimization. The parameters are calibrated through the minimization of an 'objective function' which compares model predictions with 'Large Eddy Simulations' (LES). *Tunax* is written in JAX in order to use automatic differentiation for computing the gradient of the objective function with respect to model parameters. 

# Package organisation
The closures are implemented in the folder `tunax/closures/` and they are wrapped by a 'single column model' (SCM) implemented in `tunax/model.py`. The calibration part is in `tunax/fitter.py`. The folder `obs/` contains examples of databases that can be used for the calibration. The folder `notebooks/` contains example of usage of this framework.

# Installation
*Tunax* is packaged with *poetry*. To install it and create a virtual environnement with all the dependencies, one have to execute at the root of this repository
```shell
poetry install
```
*Tunax* is installed by default with JAX on CPU.
