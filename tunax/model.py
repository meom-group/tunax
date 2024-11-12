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
import warnings
from typing import Tuple, List
from functools import partial

import equinox as eqx
import jax.numpy as jnp
from jax import lax, jit
from jaxtyping import Float, Array

from tunax.case import Case
from tunax.space import State, Trajectory
from tunax.functions import (
    tridiag_solve, add_boundaries, _format_to_single_line
)
from tunax.closure import (
    ClosureParametersAbstract, ClosureStateAbstract, Closure
)
from tunax.closures_registry import CLOSURES_REGISTRY


class SingleColumnModel(eqx.Module):
    r"""
    Single column forward model core.

    This forward model of tunax is the combination of 4 things : the physical
    case :attr:`case`, an initial state of the water column
    :attr:`init_state`, the time information with :attr:`time_frame`,
    :attr:`dt` and :attr:`out_dt` and the abstraction of the chosen closure
    for eddy-diffusivity :attr:`closure`. Adding a set of parameters for the
    closure one can run the model with the method
    :meth:`compute_trajectory_with`.

    Parameters
    ----------
    time_frame : float
        Total time of the simulation :math:`[\text h]`.
    dt : float
        Time-step of integration for every iteration :math:`[\text s]`.
    out_dt : float
        Time-step for the output writing :math:`[\text s]`.
    init_state : State
        cf. attribute.
    case : Case
        cf. attribute.
    closure_name : str
        Name of the chosen closure, must be a key of
        :attr:`~closures_registry.CLOSURES_REGISTRY`, see its documentation
        for the available closures.

    Attributes
    ----------
    nt : int
        Number of integration interations.
    dt : float
        Time-step of integration for every iteration :math:`[\text s]`.
    n_out : int
        Number of of time-steps between every output.
    init_state : State
        Initial physical state of the water column.
    case : Case
        Physical case and forcings of the experiment.
    closure : Closure
        Abstraction representing the chosen closure.
    
    Warnings
    --------
    - If :code:`time_frame` is not proportional to the time-step :attr:`dt`.
    - If :code:`time_frame` is not proportional to the out time-step
      :code:`out_dt`.
    
    Raises
    ------
    ValueError
        If :code:`out_dt` is not proportional to the time step :attr:`dt`.
    ValueError
        If :code:`closure_name` is not registerd in
        :attr:`~closures_registry.CLOSURES_REGISTRY`.

    Note
    ----
    To make this forward model compatible with the fitter part of Tunax, the
    parameters of the closure are only given during of the call of the run
    with :meth:`compute_trajectory_with`.
        
    """

    nt: int
    dt: float
    n_out: int
    init_state: State
    case: Case
    closure: Closure

    def __init__(
            self,
            time_frame: float,
            dt: float,
            out_dt: float,
            init_state: State,
            case: Case,
            closure_name: str
        ) -> SingleColumnModel:
        # time parameters transformation
        n_out = out_dt/dt
        nt = time_frame*3600/dt

        # warnings and errors on time parameters coherence
        if not n_out.is_integer():
            raise ValueError('`out_dt` should be a multiple of `dt`.')
        if not nt % n_out == 0:
            warnings.warn(_format_to_single_line("""
                The `time_frame`is not proportional to the out time-step
                `out_dt`, the last step will be computed a few before the
                `time_frame`.
            """))
        if not nt.is_integer():
            warnings.warn(_format_to_single_line("""
                The `time_frame`is not proportional to the time-step `dt`, the
                last step will be computed a few before the time_frame.
            """))
        if not closure_name in CLOSURES_REGISTRY:
            raise ValueError(_format_to_single_line("""
                `closure_name` not registerd in CLOSURES_REGISTRY.
            """))

        # write attributes
        self.nt = int(nt)
        self.dt = dt
        self.n_out = int(n_out)
        self.init_state = init_state
        self.case = case
        self.closure = CLOSURES_REGISTRY[closure_name]

    def compute_trajectory_with(
            self,
            closure_parameters: ClosureParametersAbstract
        ) -> Trajectory:
        """
        Run the model with a specific set of closure parameters.

        This method is the main one for runing the model. It calls :attr:`nt`
        times the function :func:`step` and regulary writes the output to build
        the :class:`~space.Trajectory` output.

        Parameters
        ----------
        closure_parameters : ClosureParametersAbstract
            A set of parameters of the used closure.

        Returns
        -------
        trajectory : Trajectory
            Timeseries of the evolution of the variables of the model every
            :code:`out_dt`.
        """
        # initialize the model
        states_list: List[State] = []
        state = self.init_state
        closure_state = self.closure.state_class(
            self.init_state.grid, closure_parameters
        )
        swr_frac = lmd_swfrac(self.init_state.grid.hz)

        # loop the model
        for i_t in range(self.nt):
            if i_t % self.n_out == 0:
                states_list.append(state)
            state, closure_state = step(
                self.dt, self.case, self.closure, state, closure_state,
                closure_parameters, swr_frac
            )
        time = jnp.arange(0, self.nt*self.dt, self.n_out*self.dt)

        # generate trajectory
        u_list = [s.u for s in states_list]
        v_list = [s.v for s in states_list]
        t_list = [s.t for s in states_list]
        s_list = [state.s for state in states_list]
        trajectory = Trajectory(
            self.init_state.grid, time, jnp.vstack(t_list), jnp.vstack(s_list),
            jnp.vstack(u_list), jnp.vstack(v_list)
        )
        return trajectory


@partial(jit, static_argnames=('dt', 'case', 'closure'))
def step(
        dt: float,
        case: Case,
        closure: Closure,
        state: State,
        closure_state: ClosureStateAbstract,
        closure_parameters: ClosureParametersAbstract,
        swr_frac: Float[Array, 'nz+1']
    ) -> Tuple[State, ClosureStateAbstract]:
    r"""
    Run one time-step of the model.
    

    This functions first call the closure to compute the eddy-diffusivity and
    viscosity, and then integrate the equations of tracers and momentum. It
    modifies the :code:`state` with these new values and then returns the new
    :code:`state` and :code:`closure_state`.

    Parameters
    ----------
    dt : float
        Time-step of integration for every iteration :math:`[\text s]`.
    case : Case
        Physical case and forcings of the experiment.
    closure : Closure
        Abstraction representing the chosen closure.
    state : State
        Curent state of the water column.
    closure_state : ClosureStateAbstract
        Curent state of the water column for the closure variables.
    closure_parameters : ClosureParametersAbstract
        A set of parameters of the used with the :code:`closure`.
    swr_frac : Float[~jax.Array, 'nz+1']
        Fraction of solar penetration throught the water column
        :math:`[\text{dimensionless}]`.

    Returns
    -------
    state : State
        State of the water column at next time-step.
    closure_state : ClosureStateAbstract
        State of the water column at next time-step for the closure variables.

    Note
    ----
    This function is jitted with JAX, it should make it faster, but the
    :func:`~jax.jit` decorator can be removed.
    """
    grid = state.grid

    # advance closure state (compute eddy-diffusivity and viscosity)
    closure_state = closure.step_fun(
        state, closure_state, dt, closure_parameters, case
    )

    # advance tracers
    t_new, s_new = advance_tra_ed(
        state.t, state.s, closure_state.akt, swr_frac, grid.hz, dt, case
    )

    # advance velocities
    u_new, v_new = advance_dyn_cor_ed(
        state.u, state.v, grid.hz, closure_state.akv, dt, case
    )

    # write the new state
    state = eqx.tree_at(lambda tree: tree.t, state, t_new)
    state = eqx.tree_at(lambda t: t.s, state, s_new)
    state = eqx.tree_at(lambda t: t.u, state, u_new)
    state = eqx.tree_at(lambda t: t.v, state, v_new)

    return state, closure_state


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


def advance_tra_ed(
        t: Float[Array, 'nz'],
        s: Float[Array, 'nz'],
        akt: Float[Array, 'nz+1'],
        swr_frac: Float[Array, 'nz+1'],
        hz: Float[Array, 'nz'],
        dt: float,
        case: Case
    ) -> Tuple[Float[Array, 'nz'], Float[Array, 'nz']]:
    r"""
    Integrate vertical diffusion term for tracers.

    First the flux divergences are computed taking in account the forcings.
    Then the diffusion equation of the tracers system is solved, and the
    tracers at next time-step are returned. The solved equation is for
    :math:`C` a tracer :

    :math:`\partial _z ( K_m \partial _z C) + \partial _t C + F = 0`

    where :math:`F` is the representation of the forcings.

    Parameters
    ----------
    t : Float[~jax.Array, 'nz']
        Current temperature on the center of the cells :math:`[° \text C]`.
    s : Float[~jax.Array, 'nz']
        Current salinity on the center of the cells :math:`[\text{psu}]`.
    akt : Float[~jax.Array, 'nz+1']
        Current eddy-diffusivity :math:`K_m` on the interfaces of the cells
        :math:`\left[\text m ^2 \cdot \text s ^{-1}\right]`.
    swr_frac : Float[~jax.Array, 'nz+1']
        Fraction of solar penetration throught the water column
        :math:`[\text{dimensionless}]`.
    hz : Float[~jax.Array, 'nz']
        Thickness of cells from deepest to shallowest :math:`[\text m]`.
    dt : float
        Time-step of the integration step :math:`[\text s]`.
    case : Case
        Physical case and forcings of the experiment.

    Returns
    -------
    t : Float[Array, 'nz']
        Temperature on the center of the cells at next step
        :math:`[° \text C]`.
    s : Float[Array, 'nz']
        Salinity on the center of the cells at next step :math:`[\text{psu}]`.
    """
    # 1 - Fluxes
    # Temperature
    fc_t = case.rflx_sfc_max * swr_frac
    fc_t = fc_t.at[-1].add(case.tflx_sfc)
    fc_t = fc_t.at[0].set(0.)
    # apply flux divergence
    dft = hz*t + dt*(fc_t[1:] - fc_t[:-1])
    # Salinity
    fc_s = jnp.zeros(t.shape[0]+1)
    fc_s.at[-1].set(case.sflx_sfc)
    # apply flux divergence
    dfs = hz*s + dt*(fc_s[1:] - fc_s[:-1])

    # 2 - Implicit integration for vertical diffusion
    # Temperature
    dft = dft.at[0].add(-dt * case.tflx_btm)
    t = diffusion_solver(akt, hz, dft, dt)
    # Salinity
    dfs = dfs.at[0].add(-dt * case.sflx_btm)
    s = diffusion_solver(akt, hz, dfs, dt)

    return t, s


def advance_dyn_cor_ed(
        u: Float[Array, 'nz'],
        v: Float[Array, 'nz'],
        hz: Float[Array, 'nz'],
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

    return u, v


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

    x = tridiag_solve(a, b, c, f)

    return x
