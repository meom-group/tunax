"""
Optimizer
"""
import optax
import equinox as eqx
import jax.numpy as jnp
from model import Experiment
from model import State
from jax import grad
import matplotlib.pyplot as plt


class Fitter(eqx.Module):
    nloop: int
    exp0: Experiment
    s_obs: State
    learning_rate: float
    verbatim: bool

    def loss(self, clo_par):
        nt = self.exp0.nt
        g = self.exp0.grid
        s0 = self.exp0.state0
        case = self.exp0.case
        exp = Experiment(nt, g, s0, case, clo_par)
        sf = exp.run()
        return sf.cost(self.s_obs)

     
    def fit_loop(self):
        optimizer = optimizer = optax.adam(self.learning_rate)
        opt_state = optimizer.init(self.exp0.clo_par) 
        clo_par = self.exp0.clo_par
        for i in range(self.nloop):
            grads = grad(self.loss)(clo_par)
            updates, opt_state = optimizer.update(grads, opt_state)
            clo_par = optax.apply_updates(clo_par, updates)
            if self.verbatim:
                print(f"""
                    loop {i}
                    clo_par {clo_par}
                    grads {grads}
                """)
        return clo_par
    
    def plot_res(self, clo_parf):
        plt.plot(self.exp0.run().u, '--', label='u0')
        ef = Experiment(self.exp0.nt, self.exp0.grid, self.exp0.state0, self.exp0.case, clo_parf)
        plt.plot(ef.run().u, ':', label='uf')
        plt.plot(self.s_obs.u, label='obj')
        plt.legend()