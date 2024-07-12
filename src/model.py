"""
Physical model
"""

import warnings
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from grid import Grid
from case import Case
from typing import List
from xarray import Dataset


class State(eqx.Module):
    """
    Define the state at one time step on one grid.
    """
    grid: Grid
    u: jnp.ndarray

    def cost(self, obs):
        return jnp.sum((self.u-obs.u)**2)

class History(eqx.Module):
    """
    Define the history of a simulation.
    """
    grid: Grid
    time: jnp.ndarray
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
    n_out: int
    grid: Grid
    state0: State
    case: Case
    vertical_physic: VerticalPhysics

    def __init__(self, nt: int, dt: float, out_dt: float, grid: Grid,
                 state0: State, case: Case, vertical_physics: VerticalPhysics):
        n_out = out_dt/dt
        if not n_out.is_integer():
            raise ValueError('out_dt should be a multiple of dt.')
        if not nt % n_out == 0:
            warnings.warn('The number of steps nt is not proportial to the\
                           number of steps by out out_dt/dt : the last step\
                           will be the last multiple of out_dt/dt before nt.')
        self.nt = nt
        self.dt = dt
        self.n_out = int(n_out)
        self.grid = grid
        self.state0 = state0
        self.case = case
        self.vertical_physic = vertical_physics

    def set_initial_state(self, state: State):
        self.state0 = state

    def step(self, state: State):
        diff = self.vertical_physic(state)
        u_np1_c = state.u[1:-1] + diff*self.dt*self.case.flx/(state.grid.hz[1:-1]**2) * (state.u[2:]-2*state.u[1:-1]+state.u[:-2])
        u_np1_l = state.u[0] + diff*self.dt*self.case.flx/(state.grid.hz[0]**2) * (state.u[1]-2*state.u[0])
        u_np1_d = state.u[-1] + diff*self.dt*self.case.flx/(state.grid.hz[-1]**2) * (-2*state.u[-1]+state.u[-2])
        u_np1 = jnp.concatenate([jnp.array([u_np1_l]), u_np1_c, jnp.array([u_np1_d])])
        return State(state.grid, u_np1)
    
    def gen_history(self, time: jnp.ndarray, states_list: List[State]):
        u_list = [s.u for s in states_list]
        return History(self.grid, time, jnp.vstack(u_list))
    
    def __call__(self):
        states_list = []
        state = self.state0
        for i_t in range(self.nt):
            if i_t % self.n_out == 0:
                states_list.append(state)
            state = self.step(state)
        time = jnp.arange(0, self.nt*self.dt, self.n_out*self.dt)
        return self.gen_history(time, states_list)
