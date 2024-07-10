"""
Optimizer
"""
import optax
import equinox as eqx
import jax.numpy as jnp
from case import Case
from grid import Grid
from model import Model, State, VerticalPhysics
from jax import grad
import matplotlib.pyplot as plt
from typing import Dict

class Fittable(eqx.Module):
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

class CoefFitParams(eqx.Module):
    coef_fit_dico: Dict[str, Fittable]

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
        return VerticalPhysics(**clo_coef_dico)
    
    def gen_init_val(self):
        x = []
        for coef_fit in self.coef_fit_dico.values():
            if coef_fit.do_fit:
                x.append(coef_fit.init_val)
        return jnp.array(x)


class Fitter(eqx.Module):
    coef_fit_params: CoefFitParams
    nloop: int
    nt: int
    dt: float
    grid: Grid
    state0: State
    case: Case
    state_obs: State
    learning_rate: float
    verbatim: bool
    

    def loss(self, x):
        nt = self.nt
        dt = self.dt
        g = self.grid
        s0 = self.state0
        case = self.case
        vertical_physic = self.coef_fit_params.fit_to_closure(x)
        exp = Model(nt, dt, g, s0, case, vertical_physic)
        sf = exp.run()
        return sf.cost(self.state_obs)

     
    def __call__(self):
        optimizer = optax.adam(self.learning_rate)
        x = self.coef_fit_params.gen_init_val()
        opt_state = optimizer.init(x)
        for i in range(self.nloop):
            grads = grad(self.loss)(x)
            updates, opt_state = optimizer.update(grads, opt_state)
            x = optax.apply_updates(x, updates)
            if self.verbatim:
                print(f"""
                    loop {i}
                    x {x}
                    grads {grads}
                """)
        return x
    

    def plot_res(self, xf):
        x0 = self.coef_fit_params.gen_init_val()
        vertical_physic_0 = self.coef_fit_params.fit_to_closure(x0)
        exp0 = Model(self.nt, self.dt, self.grid, self.state0, self.case, vertical_physic_0)
        plt.plot(exp0.run().u, '--', label='u0')
        vertical_physic_f = self.coef_fit_params.fit_to_closure(xf)
        expf = Model(self.nt, self.dt, self.grid, self.state0, self.case, vertical_physic_f)
        plt.plot(expf.run().u, ':', label='uf')
        plt.plot(self.state_obs.u, label='obj')
        plt.legend()