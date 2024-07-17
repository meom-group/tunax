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
from scm_oce import lmd_swfrac, tridiag_solve, rho_eos_lin
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
    
    def init_all(self):
        state = self.init_u()
        state = state.init_v()
        state = state.init_t()
        state = state.init_s()
        return state


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

class KepsState(eqx.Module):
    grid: Grid
    akv: jnp.ndarray
    akt: jnp.ndarray
    eps: jnp.ndarray
    tke: jnp.ndarray
    cmu: jnp.ndarray
    cmu_prim: jnp.ndarray

    def __init__(self, grid: Grid, akv_min: float=1e-4, akt_min: float=1e-5,
                 eps_min: float=1e-12, tke_min: float=1e-6, cmu_min: float=0.1,
                 cmu_prim_min: float=0.1):
        self.grid = grid
        self.akv = jnp.full(grid.nz+1, akv_min)
        self.akt = jnp.full(grid.nz+1, akt_min)
        self.eps = jnp.full(grid.nz+1, eps_min)
        self.tke = jnp.full(grid.nz+1, tke_min)
        self.cmu = jnp.full(grid.nz+1, cmu_min)
        self.cmu_prim = jnp.full(grid.nz, cmu_prim_min)


class KepsParams(eqx.Module):
    eos_params: jnp.ndarray
    pnm: jnp.ndarray
    betaCoef: jnp.array

    def __init__(self):
        self.eos_params = jnp.array([1024., 2e-4, 2e-4, 2., 35.]) # [rho0,alpha,beta,Tref,Sref]
        self.pnm = jnp.array([3., -1., 1.5])
        self.betaCoef = jnp.array([1.44, 1.92, -0.4, 1.])

    def __call__(self, state: State, keps_state: KepsState, case: Case):
        # attributes
        nz = state.grid.nz
        tke = keps_state.tke
        akv = keps_state.akv
        eps = keps_state.eps
        cmu = keps_state.cmu
        cmu_prim = keps_state.cmu_prim
        u = state.u
        v = state.v
        zr = state.grid.zr
        hz = state.grid.hz

        rho, bvf = rho_eos_lin(state.t, state.s, zr, self.eos_params)

        shear2 = compute_shear(u, v, u, v, zr)

        tke1 = 0.5*(tke[nz]+tke[nz-1]) 
        tke2 = 0.5*(tke[0]+tke[1])
        tkemin = 1e-6 # CHANGER AVEC tke_min
        akvmin = 1e-4
        aktmin = 1e-5
        epsmin = 1e-12 # CHANGER AVEC eps_min
        z0b = 1e-14 # CHANGER AVEC eps_min
        OneOverSig_psi = 1/1.3
        OneOverSig_k = 1.
        dt = 10. # CHANGER

        tke_sfc, tke_bot, ftke_sfc, ftke_bot, eps_sfc, eps_bot, feps_sfc, feps_bot = compute_tke_eps_bdy(
            case.ustr_sfc, case.vstr_sfc, case.ustr_btm, case.vstr_btm, z0b, tke1, hz[-1], tke2, hz[0], OneOverSig_psi, self.pnm, tkemin, epsmin)
        
        # CA C'est bien nul il faut changer
        bdy_tke_sfc = jnp.zeros(2)
        bdy_tke_bot = jnp.zeros(2)
        bdy_eps_sfc = jnp.zeros(2)
        bdy_eps_bot = jnp.zeros(2)
        if bdy_tke_sfc[0] < 0.5: bdy_tke_sfc = bdy_tke_sfc.at[1].set(tke_sfc)
        else: bdy_tke_sfc = bdy_tke_sfc.at[1].set(ftke_sfc)
        if bdy_tke_bot[0] < 0.5: bdy_tke_bot = bdy_tke_bot.at[1].set(tke_bot)
        else: bdy_tke_bot = bdy_tke_bot.at[1].set(ftke_bot)
        if bdy_eps_sfc[0] < 0.5: bdy_eps_sfc = bdy_eps_sfc.at[1].set(eps_sfc)
        else: bdy_eps_sfc = bdy_eps_sfc.at[1].set(feps_sfc)
        if bdy_eps_bot[0] < 0.5: bdy_eps_bot = bdy_eps_bot.at[1].set(eps_bot)
        else: bdy_eps_bot = bdy_eps_bot.at[1].set(feps_bot)


        tke_new, wtke = advance_turb_tke(tke, bvf, shear2, OneOverSig_k*akv, akv, keps_state.akt, eps, hz, dt, tkemin, bdy_tke_sfc, bdy_tke_bot)
        eps_new = advance_turb_eps(eps, bvf, shear2, OneOverSig_psi*akv, cmu, cmu_prim, tke, tke_new, hz, dt, self.betaCoef, epsmin, bdy_eps_sfc, bdy_eps_bot)
        akv_new, akt_new, cmu_new, cmu_prim_new, eps_new = compute_ev_ed_filt(tke_new, eps_new, bvf, shear2 , self.pnm, akvmin, aktmin, epsmin)
        
        keps_state = eqx.tree_at(lambda t: t.akv, keps_state, akv_new)
        keps_state = eqx.tree_at(lambda t: t.akt, keps_state, akt_new)
        keps_state = eqx.tree_at(lambda t: t.eps, keps_state, eps_new)
        keps_state = eqx.tree_at(lambda t: t.tke, keps_state, tke_new)
        keps_state = eqx.tree_at(lambda t: t.cmu, keps_state, cmu_new)
        keps_state = eqx.tree_at(lambda t: t.cmu_prim, keps_state, cmu_prim_new)

        return keps_state


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
    keps_params: KepsParams

    def __init__(self, nt: int, dt: float, out_dt: float, grid: Grid,
                 state0: State, case: Case, keps_params: KepsParams):
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
        self.keps_params = keps_params

    def step_t(self, state: State, keps_state: KepsState, swr_frac: jnp.ndarray):
        # get attributes
        hz = self.grid.hz
        zw = self.grid.zw
        nz = self.grid.nz
        dt = self.dt
        grav = self.case.grav
        cp = self.case.cp
        alpha = self.case.alpha
        rflx_sfc = self.case.rflx_sfc_max ### CHANGE THE SOLAR FLUX

        # 1 - Compute fluxes associated with solar penetration and surface
        # boundary condition
        # surface heat flux (including latent and solar components)
        fc_top = self.case.tflx_sfc + rflx_sfc
        # penetration of solar heat flux
        fc_mid = rflx_sfc * swr_frac[1:nz]
        fc = jnp.concatenate([jnp.array([0.]), fc_mid, jnp.array([fc_top])])
        # apply flux divergence
        t_new = hz*state.t + dt*(fc[1:] - fc[:-1])
        cffp = keps_state.eps[1:] / (cp - alpha * grav * zw[1:])
        cffm = keps_state.eps[:-1] / (cp - alpha * grav * zw[:-1])
        t_new = t_new + dt*0.5*hz*(cffp + cffm)

        # 2 - Implicit integration for vertical diffusion
        # right hand side for the tridiagonal problem
        t_new = t_new.at[0].add(-dt * self.case.tflx_btm)
        # solve tridiagonal problem
        return tridiag_solve(hz, keps_state.akt, t_new, dt) 

    def step(self, state: State, keps_state: KepsState, swr_frac: jnp.ndarray):
        keps_state = self.keps_params(state, keps_state, self.case)

        t_new = self.step_t(state, keps_state, swr_frac)

        u_new = state.u
        v_new = state.v
        v_new = state.v
        return State(state.grid, u_new, v_new, t_new, v_new), keps_state
    
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
        keps_state = KepsState(self.grid)
        swr_frac = lmd_swfrac(self.grid.hz)
        for i_t in range(self.nt):
            if i_t % self.n_out == 0:
                states_list.append(state)
            state, keps_state = self.step(state, keps_state, swr_frac)
        time = jnp.arange(0, self.nt*self.dt, self.n_out*self.dt)
        return self.gen_history(time, states_list)
