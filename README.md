![Python](https://img.shields.io/badge/dynamic/yaml?url=https://raw.githubusercontent.com/meom-group/tunax/master/.github/workflows/python-package.yml&label=Python&query=$.jobs.build.strategy.matrix["python-version"])
[![Documentation Status](https://readthedocs.org/projects/tunax/badge/?version=latest)](https://tunax.readthedocs.io/en/latest/?badge=latest)
[![Tests](https://github.com/meom-group/tunax/actions/workflows/run_tests.yaml/badge.svg)](https://github.com/meom-group/tunax/actions/workflows/run_tests.yaml)
[![License: CC-BY-NC](https://img.shields.io/badge/License-CC--BY--NC-saddlebrown.svg)](./LICENSE)

# Description
This package provides a framework for calibrating the parameters of vertical physics schemes of ocean circulation models using variational optimization. The parameters are calibrated through the minimization of an 'objective function' which compares model predictions with 'Large Eddy Simulations' (LES). *Tunax* is written in JAX in order to use automatic differentiation for computing the gradient of the objective function with respect to model parameters.

# Package organisation
The closures are implemented in the folder `tunax/closures/` and they are wrapped by a 'single column model' (SCM) implemented in `tunax/model.py`. The calibration part is in `tunax/fitter.py`. The folder `notebooks/` contains example of usage of this framework. The folder `docs/` is for the documentation configuration which is available [here] (https://tunax.readthedocs.io/en/latest/).

# Installation
## Stable version
*Tunax* is pip installable
```shell
pip install tunax
```
*Tunax* is installed by default with JAX on CPU.
