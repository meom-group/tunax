"""
Single column model
"""

import warnings
import equinox as eqx
import jax.numpy as jnp
import xarray as xr

from grid import Grid
from case import Case
from state import State
from closure import ClosureParametersAbstract, ClosureStateAbstract, Closure
from closures_registry import CLOSURES_REGISTRY
from scm_oce import lmd_swfrac, advance_tra_ed, advance_dyn_cor_ed


class Trajectory(eqx.Module):
    """
    Define the history of a simulation.
    """
    grid: Grid
    time: jnp.ndarray
    u: jnp.ndarray
    v: jnp.ndarray
    t: jnp.ndarray
    s: jnp.ndarray
    
    def to_ds(self):
        variables = {'u': (('time', 'zr'), self.u),
                     'v': (('time', 'zr'), self.v),
                     't': (('time', 'zr'), self.t),
                     's': (('time', 'zr'), self.s)}
        coords = {'time': self.time,
                  'zr': self.grid.zr,
                  'zw': self.grid.zw}
        return xr.Dataset(variables, coords)

    

class SingleColumnModel(eqx.Module):
    """
    Define an experiment
    """
    nt: int
    dt: float
    n_out: int
    grid: Grid
    init_state: State
    case: Case
    closure: Closure

    def __init__(self, nt: int, dt: float, out_dt: float, grid: Grid,
                 init_state: State, case: Case, closure_name: str):
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
        self.init_state = init_state
        self.case = case
        self.closure = CLOSURES_REGISTRY[closure_name]

    def step(self, state: State, closure_state: ClosureStateAbstract, closure_parameters: ClosureParametersAbstract, swr_frac: jnp.ndarray):
        state, closure_state = self.closure.step_fun(state, closure_state, closure_parameters, self.case)

        # ecrire ca mieux par rapport au state
        rflx_sfc = self.case.rflx_sfc_max
        stflx = jnp.array([self.case.tflx_sfc, self.case.sflx_sfc])
        btflx = jnp.array([self.case.tflx_btm, self.case.sflx_btm])


        t_new, s_new = advance_tra_ed(
            state.t, state.s, stflx, rflx_sfc,swr_frac, btflx, self.grid.hz,
            state.akt, self.grid.zw, state.eps, self.case.alpha,
            self.dt)

        u_new, v_new = advance_dyn_cor_ed(
            state.u, state.v, self.case.ustr_sfc, self.case.vstr_sfc,
            self.case.ustr_btm, self.case.vstr_btm, self.grid.hz,
            state.akv, self.case.fcor, self.dt)

        state = eqx.tree_at(lambda tree: tree.t, state, t_new)
        state = eqx.tree_at(lambda t: t.s, state, s_new)
        state = eqx.tree_at(lambda t: t.u, state, u_new)
        state = eqx.tree_at(lambda t: t.v, state, v_new)

        return state, closure_state
    
    def compute_trajectory_with(self, closure_parameters: ClosureParametersAbstract) -> Trajectory:
        # initialize the model
        states_list = []
        state = self.init_state
        closure_state = self.closure.state_class(self.grid)
        swr_frac = lmd_swfrac(self.grid.hz)

        # loop the model
        for i_t in range(self.nt):
            if i_t % self.n_out == 0:
                states_list.append(state)
            state, closure_state = self.step(state, closure_state, closure_parameters, swr_frac)
        time = jnp.arange(0, self.nt*self.dt, self.n_out*self.dt)

        # generate trajectory
        u_list = [s.u for s in states_list]
        v_list = [s.v for s in states_list]
        t_list = [s.t for s in states_list]
        s_list = [state.s for state in states_list]
        trajectory = Trajectory(
            self.grid, time, jnp.vstack(u_list), jnp.vstack(v_list),
            jnp.vstack(t_list), jnp.vstack(s_list))
        return trajectory
