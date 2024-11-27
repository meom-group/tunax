# `k-epsilon_Kato-Phillips.ipynb`
In this notebook, we will demonstrate how get started with _Tunax_ to run a forward model and do a perfect model calibration. Our approach will use the $k-\varepsilon$ closure and will be based on the idealized Kato-Phillips [1] case. This case is characterized by the absence of heat flux and the presence of uniform zonal wind forcing. In a _perfect-model_ framework, the “observations” used for calibration are outputs of a forward model run, generated using a specific set of $k-\varepsilon$ parameters. The goal is for _Tunax_ to successfully retrieve these original parameters through the calibration process.

# `k-epsilon_Wagner_LES.ipynb` (work in progress)
This notebook show the use of *Tunax* to calibrated 4 parameters of `k-\varepsilon` on a database of LES simulations done by Gregory Wagner. It is still a "logbook" : it's a work in progress.