"""
State of a column
"""

import equinox as eqx
import jax.numpy as jnp
from jax import vmap
from grid import Grid

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
        return eqx.tree_at(lambda t: t.akt, self, akt_new)

    def init_akv(self, akv_min: float=1e-4):
        """
        Return a State object where akv is constant equal to `akv_min`.
        """
        akv_new = jnp.full(self.grid.nz+1, akv_min)
        return eqx.tree_at(lambda t: t.akv, self, akv_new)

    def init_eps(self, eps_min: float=1e-12):
        """
        Return a State object where eps is constant equal to `eps_min`.
        """
        eps_new = jnp.full(self.grid.nz+1, eps_min)
        return eqx.tree_at(lambda t: t.eps, self, eps_new)
    
    def init_all(self):
        state = self.init_t()
        state = state.init_s()
        state = state.init_u()
        state = state.init_v()
        state = state.init_akt()
        state = state.init_akv()
        state = state.init_eps()
        return state