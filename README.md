![Python](https://img.shields.io/badge/dynamic/yaml?url=https://raw.githubusercontent.com/meom-group/tunax/master/.github/workflows/run_tests.yaml&label=Python&query=$.jobs.test.strategy.matrix["python-version"]&color=seagreen)
[![PyPi](https://img.shields.io/badge/dynamic/xml?url=https://pypi.org/rss/project/tunax/releases.xml&label=PyPi&query=/rss/channel/item[1]/title)](https://pypi.org/project/tunax/)
[![Documentation Status](https://readthedocs.org/projects/tunax/badge/?version=latest)](https://tunax.readthedocs.io/en/latest/?badge=latest)
[![Tests](https://github.com/meom-group/tunax/actions/workflows/run_tests.yaml/badge.svg)](https://github.com/meom-group/tunax/actions/workflows/run_tests.yaml)
[![License: CC-BY-NC](https://img.shields.io/badge/License-CC--BY--NC-chocolate.svg)](LICENSE)

# Description
This package provides a framework for calibrating the parameters of vertical physics schemes of ocean circulation models using variational optimization. The parameters are calibrated through the minimization of an 'objective function' which compares model predictions with 'Large Eddy Simulations' (LES). *Tunax* is written in JAX in order to use automatic differentiation for computing the gradient of the objective function with respect to model parameters.

# Repository organisation
## Sources
The source code of tunax is in the folder `tunax/`. The two main modules are `tunax/model.py` which contains the implementation of the forward Single Column Model and `tunax/fitter.py` which contains the implementation of the calibration part. The various physical closures equations that *Tunax* can calibrate are implemented in the folder `tunax/closures/`. The other modules implement the object used by the model and the callibrator and they are described in the [documentation](https://tunax.readthedocs.io/en/latest/).

## Notebooks
The folder `notebooks/` contains some notebooks of demonstration of *tunax*.

# Installation
*Tunax* is installed by default with JAX on CPU.
## Stable version
*Tunax* is pip installable the following command will install the lastest stable release :
```shell
pip install tunax
```

## Development release
The most recent additions to *Tunax* since the last release are implemented in the `main` branch of this repository. To benefit from them one can clone or fork this repository and install *Tunax* with the package manager *poetry* :
```shell
poetry install --no-dev
```

## Notebook use
To use the notebooks with the specific packages that they use, one can clone or fork the branch `main` or another of this repository and install *Tunax* with the package manager *poetry* :
```shell
poetry install
```
