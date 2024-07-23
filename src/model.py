"""
Physical model
"""

import warnings
import equinox as eqx
import jax.numpy as jnp
import xarray as xr
from jax import vmap, jit
from grid import Grid
from case import Case
from typing import Dict
from scm_oce import lmd_swfrac, advance_tra_ed, advance_dyn_cor_ed, rho_eos_lin
from closures.k_eps import compute_shear, compute_tke_eps_bdy, advance_turb_eps, advance_turb_tke, compute_ev_ed_filt

def piecewise_linear_ramp(z: float, z0: float, f0: float)-> float:
    """
    Apply to `z` a function linear by part and continuous : the part for
    `z`<`z0` is constantly equal to 0 and the part for `z`>`z0` is a linear
    function which values 0 in `z0` and `f0` in 0.

    Parameters
    ----------
    z : float
        the value where to apply the function
    z0 : float
        the point of connexion of the two linear parts of the function
    f0 : float
        the value of the function in 0
    
    Returns
    -------
    fz : float
        the value of the function in `z`
    """
    return f0*(z/-z0+1) * (z>z0)

def piecewise_linear_flat(z: float, z0: float, f0: float, sl: float) -> float:
    """
    Apply to `z` a function linear by part and continuous : the part for
    `z`<`z0` is linear of slope `sl` and the part for `z`>`z0` is constant
    equals to `f0`.

    Parameters
    ----------
    z : float
        the value where to apply the function
    z0 : float
        the point of connexion of the two linear parts of the function
    f0 : float
        the value of the function in 0 and in the right part of the funcion
    sl : float
        the slope of the left part of the function
    
    Returns
    -------
    fz : float
        the value of the function in `z`
    """
    return f0 + sl*(z-z0) * (z<z0)


class State(eqx.Module):
    """
    Define the state at one time step on one grid.

    Parameters
    ----------
    grid : Grid
        spatial grid
    u : jnp.ndarray, float(nz)
        zonal velocity at the next step[m.s-1]
    v : jnp.ndarray, float(nz)
        meridional velocity at the next step [m.s-1]
    t : jnp.ndarray, float(nz)
        temperature at the next step [C]
    s : jnp.ndarray, float(nz)
        current salinity [psu]

    Attributes
    ----------
    grid : Grid
        spatial grid
    u : jnp.ndarray, float(nz)
        zonal velocity at the next step[m.s-1]
    v : jnp.ndarray, float(nz)
        meridional velocity at the next step [m.s-1]
    t : jnp.ndarray, float(nz)
        temperature at the next step [C]
    s : jnp.ndarray, float(nz)
        current salinity [psu]

    """
    grid: Grid
    t: jnp.ndarray
    s: jnp.ndarray
    u: jnp.ndarray
    v: jnp.ndarray
    akt: jnp.ndarray
    akv: jnp.ndarray
    eps: jnp.ndarray


    def __init__(self, grid: Grid, t=None, s=None, u=None, v=None, akt=None,
                 akv=None, eps=None):
        if t == None:
            t = jnp.zeros(grid.nz)
        if s == None:
            s = jnp.zeros(grid.nz)
        if u == None:
            u = jnp.zeros(grid.nz)
        if v == None:
            v = jnp.zeros(grid.nz)
        if akt == None:
            akt = jnp.zeros(grid.nz+1)
        if akv == None:
            akv = jnp.zeros(grid.nz+1)
        if eps == None:
            eps = jnp.zeros(grid.nz+1)
        self.grid = grid
        self.t = t
        self.s = s
        self.u = u
        self.v = v
        self.akt = akt
        self.akv = akv
        self.eps = eps

    def cost(self, obs):
        if not isinstance(obs, State):
            raise ValueError("Obs should be a State object")
        u_l2 = jnp.sum((self.u-obs.u)**2)
        v_l2 = jnp.sum((self.v-obs.v)**2)
        t_l2 = jnp.sum((self.t-obs.t)**2)
        s_l2 = jnp.sum((self.s-obs.s)**2)
        return u_l2 + v_l2 + t_l2 + s_l2

    def init_t(self, hmxl: float=20., t_sfc: float=21., strat_t: float=5.1e-2):
        """
        Return a State object where t is linear by part and continous : linear
        under -`hmxl` with a slope of `strat_t`, and constant equals to `t_sfc`
        above -`hmxl`.
        """
        if hmxl < 0:
            raise ValueError('hmxl should be positive')
        maped_fun = vmap(piecewise_linear_flat, in_axes=(0, None, None, None))
        t_new = maped_fun(self.grid.zr, -hmxl, t_sfc, strat_t)
        return eqx.tree_at(lambda tree: tree.t, self, t_new)

    def init_s(self, hmxl: float=20., s_sfc: float=35., strat_s: float=5.1e-2):
        """
        Return a State object where s is linear by part and continous : linear
        under -`hmxl` with a slope of `strat_s`, and constant equals to `s_sfc`
        above -`hmxl`.
        """
        if hmxl < 0:
            raise ValueError('hmxl should be positive')
        maped_fun = vmap(piecewise_linear_flat, in_axes=(0, None, None, None))
        s_new = maped_fun(self.grid.zr, -hmxl, s_sfc, strat_s)
        return eqx.tree_at(lambda t: t.s, self, s_new)

    def init_u(self, hmxl: float=20., u_sfc: float=0.):
        """
        Return a State object where u is continuous and linear by part :
        constant equals to 0 under -`hmxl`, and linear above -`hmxl` with the
        value `u_sfc` at 0.
        """
        if hmxl < 0:
            raise ValueError('hmxl should be positive')
        maped_fun = vmap(piecewise_linear_ramp, in_axes=(0, None, None))
        u_new = maped_fun(self.grid.zr, -hmxl, u_sfc)
        return eqx.tree_at(lambda t: t.u, self, u_new)

    def init_v(self, hmxl: float=20., v_sfc: float=0.):
        """
        Return a State object where u is continuous and linear by part :
        constant equals to 0 under -`hmxl`, and linear above -`hmxl` with the
        value `v_sfc` at 0.
        """
        if hmxl < 0:
            raise ValueError('hmxl should be positive')
        maped_fun = vmap(piecewise_linear_ramp, in_axes=(0, None, None))
        v_new = maped_fun(self.grid.zr, -hmxl, v_sfc)
        return eqx.tree_at(lambda t: t.v, self, v_new)

    def init_akt(self, akt_min: float=1e-5):
        """
        Return a State object where akt is constant equal to `akt_min`.
        """
        akt_new = jnp.full(self.grid.nz+1, akt_min)
        return eqx.tree_at(lambda t: t.akt_new, self, akt_new)

    def init_akt(self, akv_min: float=1e-4):
        """
        Return a State object where akv is constant equal to `akv_min`.
        """
        akv_new = jnp.full(self.grid.nz+1, akv_min)
        return eqx.tree_at(lambda t: t.akv_new, self, akv_new)

    def init_eps(self, eps_min: float=1e-12):
        """
        Return a State object where eps is constant equal to `eps_min`.
        """
        eps_new = jnp.full(self.grid.nz+1, eps_min)
        return eqx.tree_at(lambda t: t.eps_new, self, eps_new)
    
    def init_all(self):
        state = self.init_t()
        state = state.init_s()
        state = state.init_u()
        state = state.init_v()
        state = state.init_akt()
        state = state.init_akv()
        state = state.init_eps()
        return state


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


class ParametersValuesSet(eqx.Module):
    closure: str
    parameters_values_closure: eqx.Module

    def __init__(self, closure: str, parameters_value_dict: Dict[str, float]):
        self.closure = closure
        match closure:
            case 'k-epsilon':
                self.parameters_values_set = ParametersValuesKeps(**parameters_value_dict)
            case _:
                raise ValueError(f'The closure {closure} is not supported')


class ClosureBase(eqx.Module):
    

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
    closure: str

    def __init__(self, nt: int, dt: float, out_dt: float, grid: Grid,
                 init_state: State, case: Case, closure: str):
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
        self.closure = closure

    def step(self, state: State, keps_state: KepsState, swr_frac: jnp.ndarray, par_val_set: ParametersValuesSet):
        keps_state = keps(par_val_set, state, keps_state, self.case)

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

        return State(state.grid, u_new, v_new, t_new, s_new), keps_state
    
    def run(self, par_val_set: ParametersValuesSet):
        states_list = []
        state = self.init_state
        keps_state = KepsState(self.grid)
        swr_frac = lmd_swfrac(self.grid.hz)
        for i_t in range(self.nt):
            if i_t % self.n_out == 0:
                states_list.append(state)
            state, keps_state = self.step(state, keps_state, swr_frac, par_val_set)
        time = jnp.arange(0, self.nt*self.dt, self.n_out*self.dt)
        # generate history
        u_list = [s.u for s in states_list]
        v_list = [s.v for s in states_list]
        t_list = [s.t for s in states_list]
        s_list = [state.s for state in states_list]
        history = Trajectory(
            self.grid, time, jnp.vstack(u_list), jnp.vstack(v_list),
            jnp.vstack(t_list), jnp.vstack(s_list))
        return history
