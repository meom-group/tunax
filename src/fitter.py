"""
Optimizer
"""
from __future__ import annotations
import optax
import equinox as eqx
import jax.numpy as jnp
from jax import grad, jit
from typing import Dict , Callable, List

from database import Obs, ObsSet
from model import SingleColumnModel, Trajectory
from closure import Closure
from case import Case
from closures_registry import CLOSURES_REGISTRY


class FittableParameter(eqx.Module):
    do_fit: bool
    min_bound: float = 0.
    max_bound: float = 0.
    init_val: float = 0.
    fixed_val: float = 0.

    def __init__(self, do_fit, min_bound=0., max_bound=0., fixed_val=0., init_val=0.):
        self.do_fit = do_fit
        if do_fit:
            self.min_bound = min_bound
            self.max_bound = max_bound
            self.init_val = init_val
        else:
            self.fixed_val = fixed_val


class FittableParametersSet(eqx.Module):
    coef_fit_dico: Dict[str, FittableParameter]
    closure: Closure

    def __init__(self, coef_fit_dico: Dict[str, FittableParameter], closure_name: str):
        self.coef_fit_dico = coef_fit_dico
        self.closure = CLOSURES_REGISTRY[closure_name]

    def fit_to_closure(self, x):
        """
        Conversion of the array of the values to fit to the dictionnary of the closure params
        """
        clo_coef_dico = {}
        i_x = 0
        for coef_name, coef_fit in self.coef_fit_dico.items():
            if coef_fit.do_fit:
                clo_coef_dico[coef_name] = x[i_x]
                i_x += 1
            else:
                clo_coef_dico[coef_name] = coef_fit.fixed_val
        return self.closure.parameters_class(**clo_coef_dico)
    
    def gen_init_val(self):
        x = []
        for coef_fit in self.coef_fit_dico.values():
            if coef_fit.do_fit:
                x.append(coef_fit.init_val)
        return jnp.array(x)


class Fitter(eqx.Module):
    coef_fit_params: FittableParametersSet
    obs_set: ObsSet
    loss: Callable[[List[Trajectory], ObsSet], float]
    nloop: int
    learning_rate: float
    verbatim: bool
    model_list: List[SingleColumnModel]

    def __init__(
            self,
            coef_fit_params: FittableParametersSet,
            obs_set: ObsSet,
            loss: Callable[[List[Trajectory], ObsSet], float],
            nloop: int,
            learning_rate: float,
            verbatim: bool,
            dt: float,
            closure_name: str
        ) -> Fitter:
        """
        Built the list of models from initial conditions and physical cases
        of every observations
        """
        # same attributes
        self.coef_fit_params = coef_fit_params
        self.obs_set = obs_set
        self.loss = loss
        self.nloop = nloop
        self.learning_rate = learning_rate
        self.verbatim = verbatim
        # building models list
        model_list = []
        for obs in self.obs_set.observations:
            traj = obs.trajectory
            init_state = traj.extract_state(0)
            time = traj.time
            out_dt = float(time[1] - time[0])
            time_frame = float((time[-1] + out_dt)/3600.)
            model = SingleColumnModel(
                time_frame, dt, out_dt, traj.grid, init_state, obs.case, closure_name
            )
            model_list.append(model)
        self.model_list = model_list

    def loss_wrapped(self, x):
        """
        Run every model (for each observations) for the set of closure
        parameters corresponding to x, and return the loss function
        """
        closure_parameters = self.coef_fit_params.fit_to_closure(x)
        scm_set = []
        for model in self.model_list:
            traj = model.compute_trajectory_with(closure_parameters)
            scm_set.append(traj)
        return self.loss(scm_set, self.obs_set)
    

    def __call__(self):
        x_history = []
        grads_history = []
        optimizer = optax.adam(self.learning_rate)
        x = self.coef_fit_params.gen_init_val()
        opt_state = optimizer.init(x)
        grad_loss = grad(self.loss_wrapped)
        for i in range(self.nloop):
            grads = grad_loss(x)
            x_history.append(x)
            grads_history.append(grads)
            updates, opt_state = optimizer.update(grads, opt_state)
            x = optax.apply_updates(x, updates)
            if self.verbatim:
                print(f"""
                    loop {i}
                    x {x}
                    grads {grads}
                """)
        return x, x_history, grads_history
    

    
    # def run_cut(self, nt_cut, traj):
    #     """temporary test"""
    #     initial_learning_rate = self.learning_rate
    #     # scheduler = optax.exponential_decay(
    #     #     init_value=initial_learning_rate,
    #     #     transition_steps=5,
    #     #     decay_rate= 0.8
    #     # )
    #     x_history = []
    #     grads_history = []
    #     optimizer = optax.adam(learning_rate=self.learning_rate)
    #     x = self.coef_fit_params.gen_init_val()
    #     opt_state = optimizer.init(x)

    #     def loss_cut(x):
    #         s = 0
    #         closure_parameters = self.coef_fit_params.fit_to_closure(x)
    #         cut_model = eqx.tree_at(lambda t: t.nt, self.model, nt_cut)
    #         for i_cut in range(self.model.nt//nt_cut):
    #             i_t_cut = i_cut*nt_cut
    #             state0_cut = eqx.tree_at(lambda tree: tree.t, self.model.init_state, traj.t[i_t_cut, :])
    #             state0_cut = eqx.tree_at(lambda tree: tree.s, state0_cut, traj.s[i_t_cut, :])
    #             state0_cut = eqx.tree_at(lambda tree: tree.u, state0_cut, traj.u[i_t_cut, :])
    #             state0_cut = eqx.tree_at(lambda tree: tree.v, state0_cut, traj.v[i_t_cut, :])
    #             cut_model = eqx.tree_at(lambda t: t.init_state, cut_model, state0_cut)

    #             traj_cut = cut_model.compute_trajectory_with(closure_parameters)

    #             s += jnp.sum((traj_cut.t[-1, :] - traj.t[i_t_cut+nt_cut-1, :])**2)
    #         return s

    #     grad_loss = grad(loss_cut)

    #     for i in range(self.nloop):
    #         grads = grad_loss(x)
    #         x_history.append(x)
    #         grads_history.append(grads)
    #         updates, opt_state = optimizer.update(grads, opt_state)
    #         x = optax.apply_updates(x, updates)
    #         if self.verbatim:
    #             print(f"""
    #                 loop {i}
    #                 x {x}
    #                 grads {grads}
    #             """)
    #     return x, x_history, grads_history     
    