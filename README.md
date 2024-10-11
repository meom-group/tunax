# Tunax

## Description
This package is a framework for doing differential calibration of physical closures of the vertical components of global ocean models. To obtain the differentiability, *Tunax* is written in JAX. The parameters of these closures are calibrated using the databases of "obersations" that are typically outputs of LES ('Large Eddy Simulations').

## Package organisation
The closures are implemented in the folder `tunax/closures/` and they are wrapped by a single vertical column model implemented in `tunax/model.py`. The calibration part is in `tunax/fitter.py`. The folder `obs/` contains examples of databases that can be used for the calibration. The folder `notebooks/` contains example of usage of this framework.

## Installation
*Tunax* is packaged with *poetry*. To install it and create a virtual environnement with all the dependencies, one have to execute at the root of this repository
```shell
poetry install
```
*Tunax* is installed by default with JAX on CPU.