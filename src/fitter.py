"""
Optimizer
"""
import optax
import equinox as eqx
import jax.numpy as jnp
from model import SingleColumnModel, Trajectory
from jax import grad
from database import ObsSet
from typing import Dict , Callable
from closure import Closure
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
    nloop: int
    model: SingleColumnModel
    obs_set: ObsSet
    learning_rate: float
    verbatim: bool
    loss: Callable[[], Trajectory]

    def loss_wrapped(self, x):
        closure_parameters = self.coef_fit_params.fit_to_closure(x)
        def model_wrapped():
            return self.model.compute_trajectory_with(closure_parameters)
        return self.loss(model_wrapped, self.obs_set)

     
    def __call__(self):
        initial_learning_rate = self.learning_rate
        scheduler = optax.exponential_decay(
            init_value=initial_learning_rate,
            transition_steps=5,
            decay_rate= 0.8
        )
        x_history = []
        grads_history = []
        optimizer = optax.adam(scheduler)
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
    

    # def plot_res(self, xf):
    #     zr = self.model.grid.zr
    #     n_out = self.model.n_out
    #     x0 = self.coef_fit_params.gen_init_val()
    #     vertical_physic_0 = self.coef_fit_params.fit_to_closure(x0)
    #     h0 = self.model.run(vertical_physic_0)
    #     for i in range(n_out):
    #         if i == 0:
    #             plt.plot(h0.t[-1, :], zr, 'k--', label='u0')
    #         else:
    #             plt.plot(h0.u[-1, :], zr, 'k--')
    #     vertical_physic_f = self.coef_fit_params.fit_to_closure(xf)
    #     hf = self.model.run(vertical_physic_f)
    #     for i in range(n_out):
    #         if i == 0:
    #             plt.plot(hf.t[-1, :], zr, 'r:', label='uf')
    #         else:
    #             plt.plot(hf.t[-1, :], zr, 'r:')
    #     for i in range(n_out):
    #         if i == 0:
    #             plt.plot(self.obs.t[-1, :], zr, 'g', label='obj')
    #         else:
    #             plt.plot(self.obs.t[-1, :], zr, 'g')
    #     plt.legend()