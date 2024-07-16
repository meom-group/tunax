"""
Physical model
"""

import warnings
import equinox as eqx
import jax.numpy as jnp
from jax import lax
from jax import vmap, jit
from grid import Grid
from case import Case
from typing import List

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
    u: jnp.ndarray
    v: jnp.ndarray
    t: jnp.ndarray
    s: jnp.ndarray

    def __init__(self, grid: Grid, u=None, v=None, t=None, s=None):
        if u == None:
            u = jnp.zeros(grid.nz)
        if v == None:
            v = jnp.zeros(grid.nz)
        if t == None:
            t = jnp.zeros(grid.nz)
        if s == None:
            s = jnp.zeros(grid.nz)
        self.grid = grid
        self.u = u
        self.v = v
        self.t = t
        self.s = s

    def cost(self, obs):
        if not isinstance(obs, State):
            raise ValueError("Obs should be a State object")
        u_l2 = jnp.sum((self.u-obs.u)**2)
        v_l2 = jnp.sum((self.v-obs.v)**2)
        t_l2 = jnp.sum((self.t-obs.t)**2)
        s_l2 = jnp.sum((self.s-obs.s)**2)
        return u_l2 + v_l2 + t_l2 + s_l2

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


class History(eqx.Module):
    """
    Define the history of a simulation.
    """
    grid: Grid
    time: jnp.ndarray
    u: jnp.ndarray
    v: jnp.ndarray
    t: jnp.ndarray
    s: jnp.ndarray

    def cost(self, obs):
        u_cost = jnp.sum((self.u-obs.u)**2)
        v_cost = jnp.sum((self.v-obs.v)**2)
        t_cost = jnp.sum((self.t-obs.t)**2)
        s_cost = jnp.sum((self.s-obs.s)**2)
        return t_cost

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

    def step(self, state: State):
        diff = self.vertical_physic(state)

        ## Temperature
        # get attributes
        hz = self.grid.hz
        zw = self.grid.zw
        nz = self.grid.nz
        dt = dt
        grav = self.case.grav
        cp = self.case.cp
        alpha = self.case.alpha
        rflx_sfc = self.case.rflx_sfc_max ### CHANGE THE SOLAR FLUX



        # 1 - Compute fluxes associated with solar penetration and surface boundary
        # condition
        # 1.1 - temperature
        # surface heat flux (including latent and solar components)
        fc = fc.at[N].set(self.case.tflx_sfc + rflx_sfc)
        # penetration of solar heat flux
        fc = fc.at[1:N].set(rflx_sfc * swr_frac[1:N])
        # apply flux divergence
        t_new = hz*state.t + dt*(fc[1:] - fc[:-1])
        cffp = eps[1:] / (cp - alpha * grav * zw[1:])
        cffm = eps[:-1] / (cp - alpha * grav * zw[:-1])
        t_new = t_new + dt * 0.5 * hz * (cffp + cffm)

        # 2 - Implicit integration for vertical diffusion
        # 1.1 - temperature
        # right hand side for the tridiagonal problem
        t_new = t_new.at[0].add(-dt * btflx[0])
        # solve tridiagonal problem
        t_new = tridiag_solve(hz, Akt, temp, dt)
        
        u_new = state.u
        v_new = state.v
        t_new = state.t
        v_new = state.v
        return State(state.grid, u_new, v_new, t_new, v_new)
    
    def gen_history(self, time: jnp.ndarray, states_list: List[State]):
        u_list = [s.u for s in states_list]
        v_list = [s.v for s in states_list]
        t_list = [s.t for s in states_list]
        s_list = [state.s for state in states_list]
        return History(self.grid, time, jnp.vstack(u_list), jnp.vstack(v_list), 
                       jnp.vstack(t_list), jnp.vstack(s_list))
    
    def __call__(self):
        states_list = []
        state = self.state0
        for i_t in range(self.nt):
            if i_t % self.n_out == 0:
                states_list.append(state)
            state = self.step(state)
        time = jnp.arange(0, self.nt*self.dt, self.n_out*self.dt)
        return self.gen_history(time, states_list)


def step_t():



@jit
def tridiag_solve(Hz, Ak, f, dt):
    """
    Solve the tridiagonal problem associated with the implicit in time
    treatment of vertical diffusion/viscosity.

    Parameters
    ----------
    Hz : float(N)
        layer thickness [m]
    Ak : float(N+1)
        eddy diffusivity/viscosity [m2/s]
    f : float(N) [modified]
        (in: right-hand side) (out:solution of tridiagonal problem)
    dt : float
        time-step [s]
    
    Returns
    -------
    f : float(N) [modified]
        (in: right-hand side) (out:solution of tridiagonal problem)
    """
    # local variables
    N, = Hz.shape
    a = jnp.zeros(N)
    b = jnp.zeros(N)
    c = jnp.zeros(N)
    q = jnp.zeros(N)
     
    # fill the coefficients for the tridiagonal matrix
    difA = -2.0 * dt * Ak[1:N-1] / (Hz[:N-2] + Hz[1:N-1])
    difC = -2.0 * dt * Ak[2:N] / (Hz[2:N] + Hz[1:N-1])
    a = a.at[1:N-1].set(difA)
    c = c.at[1:N-1].set(difC)
    b = b.at[1:N-1].set(Hz[1:N-1] - difA - difC)

    # bottom boundary condition
    a = a.at[0].set(0.0)
    difC = -2.0 * dt * Ak[1] / (Hz[1] + Hz[0])
    c = c.at[0].set(difC)
    b = b.at[0].set(Hz[0] - difC)

    # surface boundary condition
    difA = -2.0 * dt * Ak[N-1] / (Hz[N-2] + Hz[N-1])
    a = a.at[N-1].set(difA)
    c = c.at[N-1].set(0.0)
    b = b.at[N-1].set(Hz[N-1] - difA)

    # forward sweep
    cff = 1.0 / b[0]
    q = q.at[0].set(-c[0] * cff)
    f = f.at[0].multiply(cff)
    
    def body_fun1(k, x):
        f = x[0, :]
        q = x[1, :]
        cff = 1.0 / (b[k] + a[k] * q[k-1])
        q = q.at[k].set(-cff * c[k])
        f = f.at[k].set(cff * (f[k] - a[k] * f[k-1]))
        return jnp.stack([f, q])
    f_q = jnp.stack([f, q])
    f_q = lax.fori_loop(1, N, body_fun1, f_q)
    f = f_q[0, :]
    q = f_q[1, :]

    # backward substitution
    body_fun2 = lambda k, x: x.at[N-2-k].add(q[N-2-k] * x[N-1-k])
    f = lax.fori_loop(0, N-1, body_fun2, f)
    
    return f