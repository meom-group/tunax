"""
Physical model
"""

import equinox as eqx
import jax.numpy as jnp
from grid import Grid
from case import Case
from typing import Dict


class State(eqx.Module):
    """
    Define the state at one time step on one grid.
    """
    grid: Grid
    u: jnp.ndarray

    def cost(self, obs):
        return jnp.sum((self.u-obs.u)**2)

class VerticalPhysics(eqx.Module):
    alpha: float
    beta: float

    def __call__(self, state: State):
        return self.alpha*self.beta

class Model(eqx.Module):
    """
    Define an experiment
    """
    nt: int
    dt: float
    grid: Grid
    state0: State
    case: Case
    vertical_physic: VerticalPhysics

    def set_initial_state(self, state: State):
        self.state0 = state

    def step(self, state: State):
        diff = self.vertical_physic(state)
        u_np1_c = state.u[1:-1] + diff*self.dt*self.case.flx/(state.grid.hz[1:-1]**2) * (state.u[2:]-2*state.u[1:-1]+state.u[:-2])
        u_np1_l = state.u[0] + diff*self.dt*self.case.flx/(state.grid.hz[0]**2) * (state.u[1]-2*state.u[0])
        u_np1_d = state.u[-1] + diff*self.dt*self.case.flx/(state.grid.hz[-1]**2) * (-2*state.u[-1]+state.u[-2])
        u_np1 = jnp.concatenate([jnp.array([u_np1_l]), u_np1_c, jnp.array([u_np1_d])])
        return State(state.grid, u_np1)
    
    def run(self):
        state = self.state0
        for _ in range(self.nt):
            state = self.step(state)
        return state
