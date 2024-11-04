"""
Abstraction for the calibration kernel.

This module contain the main class :class:`Fitter` which is the code of Tunax
for the calibration of the closure parts. The classes
:class:`FittableParameter` and :class:`FittableParametersSet` are used to make
a link between the optimization part and the closures structures. These class
can be obtained by the prefix :code:`tunax.fitter.` or directly by
:code:`tunax.`.

"""

from __future__ import annotations
from typing import Callable, Optional, List, Dict

import optax
import equinox as eqx
import numpy as np
import jax.numpy as jnp
from jaxtyping import Float, Array
from jax import grad

from tunax.space import Trajectory
from tunax.closure import ClosureParametersAbstract, Closure
from tunax.closures_registry import CLOSURES_REGISTRY
from tunax.model import SingleColumnModel
from tunax.database import Database


class FittableParameter(eqx.Module):
    """
    Calibration configuration for one parameter.

    An instance of this class must be created for every parameter of the
    closure that will be calibrated and for every not default values during
    calibration.

    Parameters
    ----------
    do_fit : bool
        cf. attribute.
    val : float, default=0.
        cf. attribute.

    Attributes
    ----------
    do_fit : bool
        The parameter will be calibrated.
    val : float, default=0.
        If :attr:`do_fit` : the initial value for calibration (at the first
        step of the calibration) ; if :attr:`do_fit` is false : the constant
        value to take for this parameter if it's not the default one in the
        closure.

    """

    do_fit: bool
    val: float = 0.


class FittableParametersSet(eqx.Module):
    """
    Complete closure calibration parameters.

    This class is the set of all the configurations on the closure parameters
    for the calibration. It makes the link between the array on which the
    optimizer works and the closure parameters class.

    Parameters
    ----------
    coef_fit_dict : Dict[str, FittableParameter]
        cf. attribute.
    closure_name : str
        Name of the chosen closure, must be a key of
        :attr:`~closures_registry.CLOSURES_REGISTRY`, see its documentation
        for the available closures.

    Attributes
    ----------
    coef_fit_dico : Dict[str, FittableParameter]
        The set of all the configurations of all the parameters that will be
        calibrated and the one constants but not with the default value of the
        closure.
    closure : Closure
        The abstraction that represent the used closure.
    
    """

    coef_fit_dict: Dict[str, FittableParameter]
    closure: Closure

    def __init__(
            self,
            coef_fit_dict: Dict[str, FittableParameter],
            closure_name: str
        ) -> FittableParametersSet:
        self.coef_fit_dict = coef_fit_dict
        self.closure = CLOSURES_REGISTRY[closure_name]

    @property
    def n_calib(self) -> int:
        """
        Number of variables that are calibrated.

        Returns
        -------
        nc : int
            Number of variables that are calibrated.
        """
        nc = 0
        for coef_fit in self.coef_fit_dict.values():
            if coef_fit.do_fit:
                nc += 1
        return nc

    def fit_to_closure(
            self,
            x: Float[Array, 'nc']
        ) -> ClosureParametersAbstract:
        """
        Transforms an fitted array in a set of closure parameters.

        This method copy the fixed non-default values of :attr:`coef_fit_dict`
        and copy the calibrated values from :code:`x`. Which is simply the
        parameters values in the order that is indicated by
        :attr:`coef_fit_dict`.

        Parameters
        ----------
        x : Float[~jax.Array, 'nc']
            The array on which the optimize works to find the best values. It
            is the array of the parameters that are calibrated.

        Returns
        -------
        clo_params : ClosureParametersAbstract
            The instance of the closure parameters class (child class of
            :class:`~closure.ClosureParametersAbstract`) with the modifications
            of the calibration step.
        """
        clo_coef_dico = {}
        i_x = 0
        for coef_name, coef_fit in self.coef_fit_dict.items():
            if coef_fit.do_fit:
                clo_coef_dico[coef_name] = x[i_x]
                i_x += 1
            else:
                clo_coef_dico[coef_name] = coef_fit.val
        return self.closure.parameters_class(**clo_coef_dico)

    def gen_init_val(self) -> Float[Array, 'nc']:
        """
        Produce the fitted array for the first calibration step.

        This method simply copy the initial values of the calibrated
        coefficients in an array :code:`x` which will be used as the first
        calibration step for the optimizer.

        Returns
        -------
        x : Float[~jax.Array, 'nc']
            The initial vector for the optimizer at the first step of
            calibration.
        """
        x = []
        for coef_fit in self.coef_fit_dict.values():
            if coef_fit.do_fit:
                x.append(coef_fit.val)
        return jnp.array(x)


class Fitter(eqx.Module):
    r"""
    Reprensentation of a complete calibration configuration.

    A fitter is a link between the calibrated parameters configuration
    :attr:`coef_fit_params`, a :attr:`database` of observations to fit the
    model on, a loss function :attr:`loss` and some optimizer parameters. An
    instance can be call (with no parameters) to run the calibration. The
    :code:`__init__` method build a set of models corresponding to the given
    time-step :code:`dt` , the given closure :code:`closure_name` and the
    different initial states and physical cases extracted from the
    :attr:`database`.

    Parameters
    ----------
    coef_fit_params : FittableParametersSet
        cf. attribute.
    database : Database
        cf. attribute.
    dt : float
        The time-step used for defininf the forward model that will be
        calibrated :math:`[\text s]`.
    loss : Callable[[List[Trajectory], Database], float]
        cf. attribute.
    nloop : int, default=100
        cf. attribute.
    nit_loss : int, default = -1
        cf. attribute.
    learning_rate : float, default=0.001
        cf. attribute.
    verbatim : bool, default = True
        cf. attribute.
    output_path : Optional[str], default = '.'
        cf. attribute.

    Attributes
    ----------
    coef_fit_params: FittableParametersSet
        Parametrization of the closure parameters that must be calibrated and
        the one which are fixed with non-default values.
    database: Database
        Database of *observation* used for calibration, the optimizer will
        make the model fit to them.
    model_list : List[SingleColumnModel]
        List of model instances that represent the physical case and the
        initial condition for every *observation* in the database. At each
        calibration step, they will be call to compute the loss function.
    loss: Callable[[List[Trajectory], Database], float]
        Abstraction that the user create to describe its own loss function
        which will be minimized by the fitter. The function should represent
        how much the model with its closure fits to the :attr:`database`

        Parameters
        ----------
        trajectories : List[Trajectory]
            List of trajectories computed by the model and corresponding to
            each observation of the :attr:`database` case in the same order.
        database: Database
            cf.parameter
        
        Returns
        -------
        loss : float, positive
            The quantity that will be minimized by the fitter. The user must
            compute a quantity that compares the :code:`trajectories` done by
            the forward model (with the current values for the parameters of
            the closure at the current calibration step), and the trajectories
            from the :attr:`database`.

    nloop : int
        Maximum number of calibration loops.
    nit_loss : int, default = -1
        Number of iterations of every computation of the loss function in the
        output for diagnostic. Set to -1 to never compute it. Useless if
        :attr:`output_path` is :code:`None`.
    learning_rate : float, default=0.001
        Learning rate of the optimizer algorithm : how much it is fast at each
        step.
    verbatim : bool, default = True
        Print in the terminal the evolution of the calibration.
    output_path : Optional[str], default = '.'
        If :code:`None`, don't write the evolution on numpy files ; else must
        finishes by :code:`.npz` : in this file the compressed numpy output
        will be written.

    Note
    ----
    - During the calibration the gradient of the loss function is computed at
      every iterations, but not the loss function itself. The set of
      :attr:`nit_loss` increase the cost of the iteration if the value of the
      cost funtion is computed too often.
    
    - The output is written at every step so its readable by another python
      kernel during the calibration. To access to this data, one have to read
      the :code:`.npz` file with the :func:`numpy.load` function (with
      :code:`allow_pickle` set one :code:`True`), and then access to the
      evolutions of the calibrated vector (the closures paramters that are
      calibrated) with :code:`['x']`, to their gradients with :code:`['grads']`
      these are 2 dimensional arrays, the first dimension being the calibration
      iterations and the second one the parameters list. If :attr:`nit_loss` is
      not equal to -1, one can access to the evolution of the loss function
      with :code:`['loss_it']` which records the indexes of the iterations
      where the loss function is computed and :code:`['loss_values']` the
      corresponding values of the loss function.

    """

    coef_fit_params: FittableParametersSet
    database: Database
    model_list: List[SingleColumnModel]
    loss: Callable[[List[Trajectory], Database], float]
    nloop: int = 100
    nit_loss: int = -1
    learning_rate: float = 0.001
    verbatim: bool = True
    output_path: Optional[str] = eqx.field(default='.')

    def __init__(
            self,
            coef_fit_params: FittableParametersSet,
            database: Database,
            dt: float,
            loss: Callable[[List[Trajectory], Database], float],
            nloop: int = 100,
            nit_loss: int = -1,
            learning_rate: float = 0.001,
            verbatim: bool = True,
            output_path: Optional[str] = './'
        ) -> Fitter:
        # same attributes
        self.coef_fit_params = coef_fit_params
        self.database = database
        self.loss = loss
        self.nloop = nloop
        self.nit_loss = nit_loss
        self.learning_rate = learning_rate
        self.verbatim = verbatim
        self.output_path = output_path
        # building models list
        model_list = []
        for obs in self.database.observations:
            traj = obs.trajectory
            init_state = traj.extract_state(0)
            time = traj.time
            # extract time configuration from the trajectories of the database
            out_dt = float(time[1] - time[0])
            time_frame = float((time[-1] + out_dt)/3600.)
            closure_name = coef_fit_params.closure.name
            model = SingleColumnModel(
                time_frame, dt, out_dt, init_state, obs.case, closure_name
            )
            model_list.append(model)
        self.model_list = model_list
        # write the initialized values
        if output_path is not None:
            nc = coef_fit_params.n_calib
            empty_arr_2d = np.array([[] for _ in range(nc)])
            empty_arr_1d = np.array([])
            np.savez(
                output_path, x=empty_arr_2d, grads=empty_arr_2d,
                loss_it=empty_arr_1d, loss_values=empty_arr_1d
            )

    def loss_wrapped(self, x: Float[Array, 'nc']):
        """
        Wrapping of :attr:`loss` that takes only an array in argument.

        This method runs every model for each observations with the set of
        closure parameters corresponding to x, then it computes and returns the
        the value of the loss function.

        Parameters
        ----------
        x : Float[~jax.Array, 'nc']
            An array that represent the values of the parameters of the closure
            that are in calibration.

        Returns
        -------
        loss : float, positive
            Value of the loss function for the :code:`x` values of the closure
            parameters.
        """
        closure_parameters = self.coef_fit_params.fit_to_closure(x)
        scm_set = []
        for model in self.model_list:
            traj = model.compute_trajectory_with(closure_parameters)
            scm_set.append(traj)
        return self.loss(scm_set, self.database)


    def __call__(self):
        """
        Execute the callibration.

        First the optimizer is selected and parametrized with optax and the
        gradient function of the loss is computed. Then in the calibration
        loop, the gradient is evaluated on the currents values of the closure
        parameters, the eventual output are computed and the optax optimizer
        is updated. The optmizer used is :func:`optax.adam`.

        Returns
        -------
        closure_params : ClosureParametersAbstract
            The instance of the closure parameters changed with the final
            value of the calibrated parameters.
        """
        optimizer = optax.adam(self.learning_rate)
        x = self.coef_fit_params.gen_init_val()
        opt_state = optimizer.init(x)
        grad_loss = grad(self.loss_wrapped)
        for i in range(self.nloop):
            # compute the gradient
            grads = grad_loss(x)
            # print evolution
            if self.verbatim:
                print(f"""
                    loop {i}
                    x {x}
                    grads {grads}
                """)
            # write evolution
            if self.output_path is not None:
                data = np.load(self.output_path, allow_pickle=True)
                x_ev = data['x']
                grads_ev = data['grads']
                loss_it = data['loss_it']
                loss_values = data['loss_values']
                x_ev = np.hstack([x_ev, x.reshape(-1, 1)])
                grads_ev = np.hstack([grads_ev, grads.reshape(-1, 1)])
                if self.nit_loss != -1 and i%self.nit_loss == 0:
                    loss_it = np.append(loss_it, i)
                    loss_values = np.append(loss_values, self.loss_wrapped(x))
                np.savez(
                    self.output_path, x=x_ev, grads=grads_ev, loss_it=loss_it,
                    loss_values=loss_values
                )
            # update the optimizer
            updates, opt_state = optimizer.update(grads, opt_state)
            x = optax.apply_updates(x, updates)
        return self.coef_fit_params.fit_to_closure(x)
