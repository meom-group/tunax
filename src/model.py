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
from typing import List
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


class KepsParams(eqx.Module):
    a1: float
    a2: float
    a3: float
    nn: float
    sf_d0: float
    sf_d1: float
    sf_d2: float
    sf_d3: float
    sf_d4: float
    sf_d5: float
    sf_n0: float
    sf_n1: float
    sf_n2: float
    sf_nb0: float
    sf_nb1: float
    sf_nb2: float
    lim_am0: float
    lim_am1: float
    lim_am2: float
    lim_am3: float
    lim_am4: float
    lim_am5: float
    lim_am6: float


    def __init__(self,
        c1: float=5.0,
        c2: float=0.8,
        c3: float=1.968,
        c4: float=1.136,
        c5: float=0.,
        c6: float=0.4,
        cb1: float=5.95,
        cb2: float=0.6,
        cb3: float=1.,
        cb4: float=0.,
        cb5: float=0.33333,
        cbb: float=0.72
    ):

        a1 = 0.66666666667 - 0.5*c2
        a2 = 1.0 - 0.5*c3
        a3 = 1.0 - 0.5*c4
        a5 = 0.5 - 0.5*c6
        nn = 0.5*c1
        nb = cb1
        ab1 = 1.0 - cb2
        ab2 = 1.0 - cb3
        ab3 = 2.0*(1.0 - cb4)
        ab5 = 2.0*cbb*(1.0 - cb5)
        sf_d0 = 36.0*nn*nn*nn*nb*nb
        sf_d1 = 84.0*a5*ab3*nn*nn*nb + 36.0*ab5*nn*nn*nn*nb
        sf_d2 = 9.0*(ab2*ab2 - ab1*ab1)*nn*nn*nn - 12.0*(a2*a2 - 3.0*a3*a3)*nn*nb*nb
        sf_d3 = 12.0*a5*ab3*(a2*ab1 - 3.0*a3*ab2)*nn + 12.0*a5*ab3*(a3*a3 - a2*a2)*nb + 12.0*ab5*(3.0*a3*a3 - a2*a2)*nn*nb
        sf_d4 = 48.0*a5*a5*ab3*ab3*nn + 36.0*a5*ab3*ab5*nn*nn
        sf_d5 = 3.0*(a2*a2 - 3.0*a3*a3)*(ab1*ab1 - ab2*ab2)*nn
        sf_n0 = 36.0*a1*nn*nn*nb*nb
        sf_n1 = -12.0*a5*ab3*(ab1 + ab2)*nn*nn + 8.0*a5*ab3*(6.0*a1 - a2 - 3.0*a3)*nn*nb + 36.0*a1*ab5*nn*nn*nb
        sf_n2 = 9.0*a1*(ab2*ab2 - ab1*ab1)*nn*nn
        sf_nb0 = 12.0*ab3*nn*nn*nn*nb
        sf_nb1 = 12.0*a5*ab3*ab3*nn*nn
        sf_nb2 = 9.0*a1*ab3*(ab1 - ab2)*nn*nn + (6.0*a1*(a2 - 3.0*a3) - 4.0*(a2*a2 - 3.0*a3*a3))*ab3*nn*nb
        lim_am0 = sf_d0*sf_n0
        lim_am1 = sf_d0*sf_n1 + sf_d1*sf_n0
        lim_am2 = sf_d1*sf_n1 + sf_d4*sf_n0
        lim_am3 = sf_d4*sf_n1
        lim_am4 = sf_d2*sf_n0
        lim_am5 = sf_d2*sf_n1 + sf_d3*sf_n0
        lim_am6 = sf_d3*sf_n1

        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.nn = nn
        self.sf_d0 = sf_d0
        self.sf_d1 = sf_d1
        self.sf_d2 = sf_d2
        self.sf_d3 = sf_d3
        self.sf_d4 = sf_d4
        self.sf_d5 = sf_d5
        self.sf_n0 = sf_n0
        self.sf_n1 = sf_n1
        self.sf_n2 = sf_n2
        self.sf_nb0 = sf_nb0
        self.sf_nb1 = sf_nb1
        self.sf_nb2 = sf_nb2
        self.lim_am0 = lim_am0
        self.lim_am1 = lim_am1
        self.lim_am2 = lim_am2
        self.lim_am3 = lim_am3
        self.lim_am4 = lim_am4
        self.lim_am5 = lim_am5
        self.lim_am6 = lim_am6

def keps(keps_params: KepsParams, state: State, keps_state: KepsState, case: Case):
    # changer ca
    eos_params = jnp.array([1024., 2e-4, 2e-4, 2., 35.])
    pnm = jnp.array([3., -1., 1.5])
    betaCoef = jnp.array([1.44, 1.92, -0.4, 1.])

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

    rho, bvf = rho_eos_lin(state.t, state.s, zr, eos_params)

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
        case.ustr_sfc, case.vstr_sfc, case.ustr_btm, case.vstr_btm, z0b, tke1, hz[-1], tke2, hz[0], OneOverSig_psi, pnm, tkemin, epsmin, keps_params)
    
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
    eps_new = advance_turb_eps(eps, bvf, shear2, OneOverSig_psi*akv, cmu, cmu_prim, tke, tke_new, hz, dt, betaCoef, epsmin, bdy_eps_sfc, bdy_eps_bot)
    akv_new, akt_new, cmu_new, cmu_prim_new, eps_new = compute_ev_ed_filt(tke_new, eps_new, bvf, shear2 , pnm, akvmin, aktmin, epsmin, keps_params)
    
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

    def __init__(self, nt: int, dt: float, out_dt: float, grid: Grid,
                 state0: State, case: Case):
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

    def step(self, state: State, keps_state: KepsState, swr_frac: jnp.ndarray, keps_params: KepsParams):
        keps_state = keps(keps_params, state, keps_state, self.case)

        rflx_sfc = self.case.rflx_sfc_max # AJOUTER MODULATION
        stflx = jnp.array([self.case.tflx_sfc, self.case.sflx_sfc])
        btflx = jnp.array([self.case.tflx_btm, self.case.sflx_btm])


        t_new, s_new = advance_tra_ed(state.t, state.s, stflx, rflx_sfc, swr_frac, btflx, self.grid.hz, keps_state.akt, self.grid.zw, keps_state.eps, self.case.alpha, self.dt)

        u_new, v_new = advance_dyn_cor_ed(state.u, state.v, self.case.ustr_sfc, self.case.vstr_sfc, self.case.ustr_btm, self.case.vstr_btm, self.grid.hz, keps_state.akv, self.case.fcor, self.dt)

        return State(state.grid, u_new, v_new, t_new, s_new), keps_state
    
    def gen_history(self, time: jnp.ndarray, states_list: List[State]):
        u_list = [s.u for s in states_list]
        v_list = [s.v for s in states_list]
        t_list = [s.t for s in states_list]
        s_list = [state.s for state in states_list]
        return Trajectory(self.grid, time, jnp.vstack(u_list), jnp.vstack(v_list), 
                       jnp.vstack(t_list), jnp.vstack(s_list))
    
    def run(self, keps_params: KepsParams):
        states_list = []
        state = self.state0
        keps_state = KepsState(self.grid)
        swr_frac = lmd_swfrac(self.grid.hz)
        for i_t in range(self.nt):
            if i_t % self.n_out == 0:
                states_list.append(state)
            state, keps_state = self.step(state, keps_state, swr_frac, keps_params)
        time = jnp.arange(0, self.nt*self.dt, self.n_out*self.dt)
        return self.gen_history(time, states_list)
