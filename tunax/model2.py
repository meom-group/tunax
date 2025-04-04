"""
Single column forward model core.

This module contains the main class :class:`SingleColumnModel` which is the
core of Tunax for implementing computing the evolution of a water column of the
ocean. It also contains functions that are used in this computation. The model
was traduced from Frotran to JAX with the work of Florian Lemarié and Manolis
Perrot [1]_, the translation was done in part using the work of Anthony Zhou,
Linnia Hawkins and Pierre Gentine [2]_. this class and these functions can be
obtained by the prefix :code:`tunax.model.` or directly by :code:`tunax.`.

References
----------
.. [1] M. Perrot and F. Lemarié. Energetically consistent Eddy-Diffusivity
    Mass-Flux convective schemes. Part I: Theory and Models (2024). url :
    `hal.science/hal-04439113 <https://hal.science/hal-04439113>`_.
.. [2] A. Zhou, L. Hawkins and P. Gentine. Proof-of-concept: Using ChatGPT to
    Translate and Modernize an Earth System Model from Fortran to Python/JAX
    (2024). url : `arxiv.org/abs/2405.00018
    <https://arxiv.org/abs/2405.00018>`_.

"""

from __future__ import annotations
from typing import Tuple, List, TypeAlias
from functools import partial

import equinox as eqx
import jax.numpy as jnp
from jax import lax, jit, vmap
from jaxtyping import Float, Array

from tunax.case import Case
from tunax.space import Grid, State, Trajectory
from tunax.functions import tridiag_solve2, add_boundaries
from tunax.closure import (
    ClosureParametersAbstract, ClosureStateAbstract, Closure
)
from tunax.closures_registry import CLOSURES_REGISTRY

StatesTime: TypeAlias = Tuple[State, ClosureStateAbstract, float]


class SingleColumnModel2(eqx.Module):
    nt: int
    dt: float
    p_out: int
    init_state: State
    case: Case
    closure: Closure
    output_path: str = ''
    diags: List[str]

    def __init__(
            self,
            nt: int,
            dt: float,
            p_out: int,
            init_state: State,
            case: Case,
            closure_name: str,
            output_path: str = '',
            diags: List[str] = []
        ) -> SingleColumnModel2:
        self.nt = nt
        self.dt = dt
        self.p_out = p_out
        self.init_state = init_state
        self.case = case
        self.closure = CLOSURES_REGISTRY[closure_name]
        self.output_path = output_path
        self.diags = diags

    def step(
        self,
        state: State,
        closure_state: ClosureStateAbstract,
        time: float,
        closure_parameters: ClosureParametersAbstract
    ) -> StatesTime:
        dt = self.dt
        case = self.case
        # advance closure state (compute eddy-diffusivity and viscosity)
        closure_state = self.closure.step_fun(
            state, closure_state, dt, closure_parameters, case
        )
        # advance tracers
        state = advance_tra_ed(
            state, closure_state.akt, dt, case, time
        )
        # advance velocities
        state = advance_dyn_cor_ed(
            state, closure_state.akv, dt, case
        )
        time += self.dt
        return state, closure_state, time
    
    # jit_step = partial(jit, )

    def run_partial(
            self,
            state0: State,
            closure_state0: ClosureStateAbstract,
            time0: float,
            n_steps: int,
            closure_parameters: ClosureParametersAbstract
        ) ->  StatesTime:
        def scan_fn(
                carry: StatesTime,
                _: type[None]
            ) -> Tuple[StatesTime, type[None]]:
            state, closure_state, time = carry
            state, closure_state, time = self.step(state, closure_state, time, closure_parameters)
            return (state, closure_state, time), None
        (state, closure_state, time), _ = lax.scan(
            scan_fn, (state0, closure_state0, time0), jnp.arange(n_steps)
        )
        return state, closure_state, time

    def run(self, closure_parameters: ClosureParametersAbstract):# -> Trajectory:
        init_closure_state = self.closure.state_class(self.init_state.grid, closure_parameters)

        def scan_fn(carry: StatesTime, _: type[None]) -> Tuple[StatesTime, StatesTime]:
            state, closure_state, time = carry
            state, closure_state, time = self.run_partial(
                state, closure_state, time, self.p_out, closure_parameters
            )
            return (state, closure_state, time), (state, closure_state, time)

        n_steps_out = self.nt//self.p_out
        (_, _, _), (states, closure_states, times) = lax.scan(
            scan_fn, (self.init_state, init_closure_state, 0.), jnp.arange(n_steps_out)
        )

        tra_dict = {}
        for tracer in self.case.eos_tracers:
            tra_dict[tracer] = getattr(states, tracer)
        if self.case.do_pt:
            tra_dict['pt'] = states.pt

        return Trajectory(self.init_state.grid, times, states.u, states.v, **tra_dict)


def lmd_swfrac(hz: Float[Array, 'nz']) -> Float[Array, 'nz+1']:
    r"""
    Compute solar forcing.

    Compute fraction of solar shortwave flux penetrating to specified depth due
    to exponential decay in Jerlov water type. This function is called once
    before running the model.

    Parameters
    ----------
    hz : Float[~jax.Array, 'nz']
        Thickness of cells from deepest to shallowest :math:`[\text m]`.

    Returns
    -------
    swr_frac : Float[~jax.Array, 'nz+1']
        Fraction of solar penetration throught the water column
        :math:`[\text{dimensionless}]`.
    """
    nz, = hz.shape
    mu1 = 0.35
    mu2 = 23.0
    r1 = 0.58
    attn1 = -1.0 / mu1
    attn2 = -1.0 / mu2

    xi1 = attn1 * hz
    xi2 = attn2 * hz

    def lax_step(sdwk, k):
        sdwk1, sdwk2 = sdwk
        sdwk1 = lax.cond(xi1[nz-k] > -20, lambda x: x*jnp.exp(xi1[nz-k]),
                             lambda x: 0.*x, sdwk1)
        sdwk2 = lax.cond(xi2[nz-k] > -20, lambda x: x*jnp.exp(xi2[nz-k]),
                             lambda x: 0.*x, sdwk2)
        return (sdwk1, sdwk2), sdwk1+sdwk2

    _, swr_frac = lax.scan(lax_step, (r1, 1.0 - r1), jnp.arange(1, nz+1))
    return jnp.concat((swr_frac[::-1], jnp.array([1.])))

def tracer_flux(
        tracer: str, # t, s, b, pt
        case: Case,
        grid: Grid,
        time: float
    ) -> Float[Array, 'nz+1']:
    """
    compute flux difference from forcings for tracers
    """
    forcing = getattr(case, f'{tracer}_forcing')
    forcing_type = getattr(case, f'{tracer}_forcing_type')
    match forcing_type:
        case 'borders': # les valeurs du forcing en tra.m.s-1
            df = add_boundaries(
                -forcing[0], jnp.zeros(grid.zr.shape[0]-2), forcing[1]
            )
        case 'constant':# les valeurs du forcing en tra.s-1
            vec_fun = vmap(forcing)
            df = grid.hz*vec_fun(grid.zr)
        case 'variable':# les valeurs du forcing en tra.s-1
            vec_fun = vmap(lambda z: forcing(z, time))
            df = grid.hz*vec_fun(grid.zr)
    return df

def advance_tra_ed(
        state: State,
        akt: Float[Array, 'nz+1'],
        dt: float,
        case: Case,
        time: float,
    ) -> Tuple[Float[Array, 'nz'], Float[Array, 'nz']]:
    r"""
    Integrate vertical diffusion term for tracers.

    First the flux divergences are computed taking in account the forcings.
    Then the diffusion equation of the tracers system is solved, and the
    tracers at next time-step are returned. The solved equation is for
    :math:`C` a tracer :

    :math:`\partial _z ( K_m \partial _z C) + \partial _t C + F = 0`

    where :math:`F` is the representation of the forcings.
    CHANGE HERE

    Parameters
    ----------
    state : State
        Curent state of the water column.
    akt : Float[~jax.Array, 'nz+1']
        Current eddy-diffusivity :math:`K_m` on the interfaces of the cells
        :math:`\left[\text m ^2 \cdot \text s ^{-1}\right]`.
    dt : float
        Time-step of the integration step :math:`[\text s]`.
    case : Case
        Physical case and forcings of the experiment.
CHANGER HERE
    Returns
    -------
    t : Float[Array, 'nz']
        Temperature on the center of the cells at next step
        :math:`[° \text C]`.
    s : Float[Array, 'nz']
        Salinity on the center of the cells at next step :math:`[\text{psu}]`.
    """
    hz = state.grid.hz
    tracers = [tra for tra in case.eos_tracers]
    if case.do_pt:
        tracers.append('pt')
    for tracer in tracers:
        tra = getattr(state, tracer)
        df = tracer_flux(tracer, case, state.grid, time)
        df = hz*tra + dt*df
        tra = diffusion_solver(akt, hz, df, dt)
        state = eqx.tree_at(lambda t: getattr(t, tracer), state, tra)

    return state

def advance_dyn_cor_ed(
        state: State,
        akv: Float[Array, 'nz+1'],
        dt: float,
        case: Case
    ) -> Tuple[Float[Array, 'nz'], Float[Array, 'nz']]:
    r"""
    Integrate vertical diffusion and Coriolis terms for momentum.

    First the Coriolis term is computed, then the momentum forcings are applied
    and finally, the diffusion equation is solved. The momentum at next time-
    step is returned. The equation which is solved is :

    :math:`\partial_z (K_v \partial_z U) + F_{\text{cor}}(U) + F = 0`

    where :math:`F_{\text{cor}}` represent the Coriolis effect, and :math:`F`
    represent the effect of the forcings.
    CHANGE HERE

    Parameters
    ----------
    u : Float[~jax.Array, 'nz']
        Current zonal velocity on the center of the cells
        :math:`\left[\text m \cdot \text s^{-1}\right]`.
    v : Float[~jax.Array, 'nz']
        Current meridional velocity on the center of the cells
        :math:`\left[\text m \cdot \text s^{-1}\right]`.
    akv : Float[~jax.Array, 'nz+1']
        Current eddy-viscosity :math:`K_v` on the interfaces of the cells
        :math:`\left[\text m ^2 \cdot \text s ^{-1}\right]`.
    hz : Float[~jax.Array, 'nz']
        Thickness of cells from deepest to shallowest :math:`[\text m]`.
    dt : float
        Time-step of the integration step :math:`[\text s]`.
    case : Case
        Physical case and forcings of the experiment.

    Returns
    -------
    u : Float[Array, 'nz']
        Zonal velocity on the center of the cells at the next time step
        :math:`\left[\text m \cdot \text s^{-1}\right]`.
    v : Float[Array, 'nz']
        Meridional velocity on the center of the cells at the next time step
        :math:`\left[\text m \cdot \text s^{-1}\right]`.
    """
    gamma_cor = 0.55
    fcor = case.fcor
    u = state.u
    v = state.v
    hz = state.grid.hz

    # 1 - Compute Coriolis term
    cff = (dt * fcor) ** 2
    cff1 = 1 / (1 + gamma_cor * gamma_cor * cff)
    fu = cff1 * hz * ((1-gamma_cor*(1-gamma_cor)*cff)*u + dt*fcor*v)
    fv = cff1 * hz * ((1-gamma_cor*(1-gamma_cor)*cff)*v - dt*fcor*u)

    # 2 - Apply surface and bottom forcing
    fu = fu.at[-1].add(dt * case.ustr_sfc)
    fv = fv.at[-1].add(dt * case.vstr_sfc)
    fu = fu.at[0].add(-dt * case.ustr_btm)
    fv = fv.at[0].add(-dt * case.vstr_btm)

    # 3 - Implicit integration for vertical viscosity
    u = diffusion_solver(akv, hz, fu, dt)
    v = diffusion_solver(akv, hz, fv, dt)

    # 4 - Update the state
    state = eqx.tree_at(lambda t: t.u, state, u)
    state = eqx.tree_at(lambda t: t.v, state, v)

    return state

def diffusion_solver(
        ak: Float[Array, 'nz+1'],
        hz: Float[Array, 'nz'],
        f: Float[Array, 'nz'],
        dt: float
    ) -> Float[Array, 'nz']:
    r"""
    Solve a diffusion problem with finite volumes.

    The diffusion problems can be written

    :math:`\partial _z (K \partial _z X) + \dfrac f {\Delta t \Delta x} = 0`

    where we are searching for :math:`X` and where :math:`f` represents the
    temporal derivative and forcings. This function transforms this problem in
    a tridiagonal system and then solve it.

    Parameters
    ----------
    ak : Float[~jax.Array, 'nz+1']
        Diffusion at the cell interfaces :math:`K` in
        :math:`\left[\text m ^2 \cdot \text s ^{-1}\right]`.
    hz : Float[~jax.Array, 'nz']
        Thickness of cells from deepest to shallowest
        :math:`\left[\text m\right]`.
    f : Float[~jax.Array, 'nz']
        Right-hand flux of the equation :math:`f` in
        :math:`[[X] \cdot \text m ]`.
    dt : float
        Time-step of discretisation :math:`[\text s]`.
    
    Returns
    -------
    x : Float[~jax.Array, 'nz']
        Solution of the diffusion problem
        :math:`X` in :math:`\left[[X]\right]`.
    """
    # fill the coefficients for the tridiagonal matrix
    a_in = -2.0 * dt * ak[1:-2] / (hz[:-2] + hz[1:-1])
    c_in = -2.0 * dt * ak[2:-1] / (hz[2:] + hz[1:-1])
    b_in = hz[1:-1] - a_in - c_in

    # bottom boundary condition
    c_btm = -2.0 * dt * ak[1] / (hz[1] + hz[0])
    b_btm = hz[0] - c_btm

    # surface boundary condition
    a_sfc = -2.0 * dt * ak[-2] / (hz[-2] + hz[-1])
    b_sfc = hz[-1] - a_sfc

    # concatenations
    a = add_boundaries(0., a_in, a_sfc)
    b = add_boundaries(b_btm, b_in, b_sfc)
    c = add_boundaries(c_btm, c_in, 0.)

    x = tridiag_solve2(a, b, c, f)

    return x
