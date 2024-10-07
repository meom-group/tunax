"""
Classes and functions for the single column model.

The Single Column Model (SCM) computes the evolution of a water column of the
ocean. Model traduced from [1] in Fortran using the work of [2].

Classes
-------
SingleColumnModel
    physical model of a water column of the ocean

Functions
---------
lmd_swfrac
    compute fraction of solar shortwave flux penetrating to specified depth due
    to exponential decay in Jerlov water type
advance_tra_ed
    integrate vertical diffusion term for tracers
advance_dyn_cor_ed
    integrate vertical viscosity and Coriolis terms for dynamics
tridiag_solve
    solve the tridiagonal problem associated with the implicit in time
    treatment of vertical diffusion and viscosity

References
----------
.. [1] M. Perrot and F. Lemarié. Energetically consistent Eddy-Diffusivity
    Mass-Flux convective schemes. Part I: Theory and Models (2024).
    https://hal.science/hal-04439113
.. [2] A. Zhou, L. Hwkins and P. Gentine. Proof-of-concept: Using ChatGPT to
    Translate and Modernize an Earth System Model from Fortran to Python/JAX
    (2024). https://arxiv.org/abs/2405.00018

"""


from __future__ import annotations
import warnings
from typing import Tuple, List
from functools import partial

import equinox as eqx
import jax.numpy as jnp
from jax import lax, jit

from case import Case
from state import Grid, State, Trajectory
from closures_registry import CLOSURES_REGISTRY
from functions import tridiag_solve, add_boundaries, format_to_single_line
from closure import ClosureParametersAbstract, ClosureStateAbstract, Closure


class SingleColumnModel(eqx.Module):
    """
    Physical model of a water column of the ocean.

    Attributes
    ----------
    nt : int
        number of time-steps
    dt : float
        time-step for every iteration [s]
    n_out : int
        number of states output
    grid : Grid
        spatial grid of a water column
    init_state : State
        initial physical state
    case : Case
        physical case and forcings
    closure : Closure
        object representing the chosed closure

    Parameters
    ----------
    time_frame : float
        total time of the simulation [h]
    dt : float
        time-step for every iteration [s]
    out_dt : float
        time-step for the output [s]
    grid : Grid
        spatial grid of a water column
    init_state : State
        initial physical state
    case : Case
        physical case and forcings
    closure_name : str
        name of the turbulent closure that will be used

    Methods
    -------
    step (jitted version)
        run one time-step of the model
    compute_trajectory_with
        run the model with a specific set of closure parameters
        
    """

    nt: int
    dt: float
    n_out: int
    grid: Grid
    init_state: State
    case: Case
    closure: Closure

    def __init__(
            self,
            time_frame: float,
            dt: float,
            out_dt: float,
            grid: Grid,
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
            warnings.warn(format_to_single_line("""
                The `time_frame`is not proportional to the out time-step
                `out_dt`, the last step will be computed a few before the
                `time_frame`.
            """))
        if not nt.is_integer():
            warnings.warn(format_to_single_line("""
                The `time_frame`is not proportional to the time-step `dt`, the
                last step will be computed a few before the time_frame.
            """))
        if not closure_name in CLOSURES_REGISTRY:
            raise ValueError(format_to_single_line("""
                `closure_name` not registerd in CLOSURES_REGISTRY.
            """))

        # write attributes
        self.nt = int(nt)
        self.dt = dt
        self.n_out = int(n_out)
        self.grid = grid
        self.init_state = init_state
        self.case = case
        self.closure = CLOSURES_REGISTRY[closure_name]

    def compute_trajectory_with(
            self,
            closure_parameters: ClosureParametersAbstract
        ) -> Trajectory:
        """
        Run the model with a specific set of closure parameters.

        Parameters
        ----------
        closure_parameters : ClosureParametersAbstract
            a set of parameters of the used closure

        Returns
        -------
        trajectory : Trajectory
            timeseries of the different states of the run of the model
        """
        # initialize the model
        states_list: List[State] = []
        state = self.init_state
        closure_state = self.closure.state_class(self.grid, closure_parameters)
        swr_frac = lmd_swfrac(self.grid.hz)

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
            self.grid, time, jnp.vstack(t_list), jnp.vstack(s_list),
            jnp.vstack(u_list), jnp.vstack(v_list))
        return trajectory


@partial(jit, static_argnames=('dt', 'case', 'closure'))
def step(
        dt: float,
        case: Case,
        closure: Closure,
        state: State,
        closure_state: ClosureStateAbstract,
        closure_parameters: ClosureParametersAbstract,
        swr_frac: jnp.ndarray
    ) -> Tuple[State, ClosureStateAbstract]:
    """
    Run one time-step of the model.

    Parameters
    ----------
    dt : float
        time-step for every iteration [s]
    case : Case
        physical case and forcings
    closure : Closure
        object representing the chosed closure
    closure_state : ClosureStateAbstract
        curent state of the closure
    closure_parameters : ClosureParametersAbstract
        a set of parameters of the used closure
    swr_frac : jnp.ndarray, float(nz+1)
        fraction of solar penetration

    Returns
    -------
    state : State
        state of the system at the next step
    closure_state : ClosureStateAbstract
        state of the closure at the next step
    """
    grid = state.grid

    # advance closure state (compute eddy-diffusivity and viscosity)
    closure_state = closure.step_fun(
        state, closure_state, dt, closure_parameters, case)

    # advance tracers
    t_new, s_new = advance_tra_ed(
        state.t, state.s, closure_state.akt, closure_state.eps, swr_frac,
        grid.zw, grid.hz, dt, case)

    # advance velocities
    u_new, v_new = advance_dyn_cor_ed(
        state.u, state.v, grid.hz, closure_state.akv, dt, case)

    # write the new state
    state = eqx.tree_at(lambda tree: tree.t, state, t_new)
    state = eqx.tree_at(lambda t: t.s, state, s_new)
    state = eqx.tree_at(lambda t: t.u, state, u_new)
    state = eqx.tree_at(lambda t: t.v, state, v_new)

    return state, closure_state


def lmd_swfrac(hz: jnp.ndarray) -> jnp.ndarray:
    """
    Compute fraction of solar shortwave flux penetrating to specified depth due
    to exponential decay in Jerlov water type.

    Parameters
    ----------
    hz : jnp.ndarray, float(nz)
        thickness of cells from deepest to shallowest [m]

    Returns
    -------
    swr_frac : jnp.ndarray, float(nz+1)
        fraction of solar penetration
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
    return jnp.concat((swr_frac[::-1], jnp.array([1])))


def advance_tra_ed(
        t: jnp.ndarray,
        s: jnp.ndarray,
        akt: jnp.ndarray,
        eps: jnp.ndarray,
        swr_frac: jnp.ndarray,
        zw: jnp.ndarray,
        hz: jnp.ndarray,
        dt: float,
        case: Case
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    r"""
    Integrate vertical diffusion term for tracers.

    Parameters
    ----------
    t : jnp.ndarray, float(nz)
        current temperature [C]
    s : jnp.ndarray, float(nz)
        current salinity [psu]
    akt : jnp.ndarray, float(nz+1)
        eddy-diffusivity [m2.s-1]
    eps : jnp.ndarray, float(nz+1)
        TKE dissipation [m2.s-3]
    swr_frac : jnp.ndarray, float(nz+1)
        fraction of solar penetration
    zw : jnp.ndarray, float(nz+1)
        depths of cell interfaces from deepest to shallowest [m]
    hz : jnp.ndarray, float(nz)
        thickness of cells from deepest to shallowest [m]
    dt : float
        time-step [s]
    case : Case
        physical case

    Returns
    -------
    t : jnp.ndarray, float(nz)
        temperature at the next step [C]
    s : jnp.ndarray, float(nz)
        salinity at the next step [psu]

    Notes
    -----
    for the vectorized version, ntra should be equal to 2
    \[ \overline{\phi}^{n+1,*} = \overline{\phi}^n + \Delta t \partial_z \left(
    K_m \partial_z  \overline{\phi}^{n+1,*} \right) \]
    """
    nz, = t.shape
    fc = jnp.zeros(nz+1)
    cp = case.cp
    alpha = case.alpha
    grav = case.grav

    # 1 - Compute fluxes associated with solar penetration and surface boundary
    # condition
    # Temperature
    # surface heat flux (including latent and solar components)
    # penetration of solar heat flux
    fc_t = case.rflx_sfc_max * swr_frac
    # latent component heat flux
    fc_t = fc_t.at[-1].add(case.tflx_sfc)
    fc_t = fc_t.at[0].set(0.)
    # apply flux divergence
    t = hz*t + dt*(fc_t[1:] - fc_t[:-1])
    cffp = eps[1:] / (cp - alpha * grav * zw[1:])
    cffm = eps[:-1] / (cp - alpha * grav * zw[:-1])
    t = t + dt * 0.5 * hz * (cffp + cffm)
    # Salinity
    fc_s = jnp.zeros(nz+1)
    fc_s.at[-1].set(case.sflx_sfc)
    # apply flux divergence
    s = hz*s + dt*(fc[1:] - fc[:-1])

    # 2 - Implicit integration for vertical diffusion
    # Temperature
    t = t.at[0].add(-dt * case.tflx_btm)
    t = diffusion_solver(akt, hz, t, dt)
    # Ssalinity
    s = s.at[0].add(-dt * case.sflx_btm)
    s = diffusion_solver(akt, hz, s, dt)

    return t, s


def advance_dyn_cor_ed(
        u: jnp.ndarray,
        v: jnp.ndarray,
        hz: jnp.ndarray,
        akv: jnp.ndarray,
        dt: float,
        case: Case
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    r"""
    Integrate vertical viscosity and Coriolis terms for dynamics.

    Parameters
    ----------
    u : jnp.ndarray, float(nz)
        current zonal velocity [m.s-1]
    v : jnp.ndarray, float(nz)
        current meridional velocity [m.s-1]
    akv : jnp.ndarray, float(nz+1)
        eddy-viscosity [m2.s-1]
    hz : jnp.ndarray, float(nz)
        thickness of cells from deepest to shallowest [m]
    dt : float
        time-step [s]
    case : Case
        physical case

    Returns
    -------
    u : jnp.ndarray, float(nz)
        zonal velocity at the next step [m.s-1]
    v : jnp.ndarray, float(nz)
        meridional velocity at the next step [m.s-1]

    Notes
    -----
    1 - Compute Coriolis term
    if n is even
    \begin{align*}
    u^{n+1,\star} &= u^n + \Delta t f v^n \\
    v^{n+1,\star} &= v^n - \Delta t f u^{n+1,\star}
    \end{align*}
    if n is odd
    \begin{align*}
    v^{n+1,\star} &= v^n - \Delta t f u^n \\
    u^{n+1,\star} &= u^n + \Delta t f v^{n+1,\star}
    \end{align*}

    2 - Apply surface and bottom forcing

    3 - Implicit integration for vertical viscosity
    \begin{align*}
    \mathbf{u}^{n+1,\star \star} &= \mathbf{u}^{n+1,\star} + \Delta t
    \partial_z \left(  K_m \partial_z  \mathbf{u}^{n+1,\star \star} \right)  \\
    \end{align*}
    """
    gamma_cor = 0.55
    fcor = case.fcor

    # 1 - Compute Coriolis term
    cff = (dt * fcor) ** 2
    cff1 = 1 / (1 + gamma_cor * gamma_cor * cff)
    u = cff1 * hz * ((1-gamma_cor*(1-gamma_cor)*cff)*u + dt*fcor*v)
    v = cff1 * hz * ((1-gamma_cor*(1-gamma_cor)*cff)*v - dt*fcor*u)

    # 2 - Apply surface and bottom forcing
    u = u.at[-1].add(dt * case.ustr_sfc)
    v = v.at[-1].add(dt * case.vstr_sfc)
    u = u.at[0].add(-dt * case.ustr_btm)
    v = v.at[0].add(-dt * case.vstr_btm)

    # 3 - Implicit integration for vertical viscosity
    u = diffusion_solver(akv, hz, u, dt)
    v = diffusion_solver(akv, hz, v, dt)

    return u, v


def diffusion_solver(
        ak: jnp.ndarray,
        hz: jnp.ndarray,
        f: jnp.ndarray,
        dt: float
    ) -> jnp.ndarray:
    """
    Solve the tridiagonal problem associated with the implicit in time
    treatment of vertical diffusion and viscosity.

    Parameters
    ----------
    ak : jnp.ndarray, float(nz+1)
        eddy-diffusivity or eddy-viscosity [m2.s-1]
    hz : jnp.ndarray, float(nz)
        thickness of cells from deepest to shallowest [m]
    f : jnp.ndarray, float(nz)
        right-hand side
    dt : float
        time-step [s]
    
    Returns
    -------
    f : jnp.ndarray, float(nz)
        solution of tridiagonal problem
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

    f = tridiag_solve(a, b, c, f)

    return f


# @jit
# def compute_evd(bvf, Akv, Akt, AkEvd):
#     """
#     Compute enhanced vertical diffusion/viscosity where the density profile is
#     unstable.

#     Parameters
#     ----------
#     bvf : float(N+1)
#         Brunt Vaisala frequency [s-2]
#     Akv : float(N+1) [modified]
#         eddy-viscosity [m2/s]
#     Akt : float(N+1) [modified]
#         eddy-diffusivity [m2/s]
#     AkEvd : float
#         value of enhanced diffusion [m2/s]

#     Returns
#     -------
#     Akv : float(N+1) [modified]
#         eddy-viscosity [m2/s]
#     Akt : float(N+1) [modified]
#         eddy-diffusivity [m2/s]
#     """
#     Akv = jnp.where(bvf<=-1e-12, AkEvd, Akv)
#     Akt = jnp.where(bvf<=-1e-12, AkEvd, Akt)

#     return Akv, Akt




# @jit
# def compute_mxl(bvf, rhoc, zr, zref, rhoRef):
#     """
#     Compute mixed layer depth.

#     Parameters
#     ----------
#     bvf : float(N+1)
#         Brunt Vaisala frequancy [s-2]
#     rhoc : float
#         thermal expension coefficient [kg m-3]
#     zr : float(N)
#         depth at cell center [m]
#     zref : float
#         no description
#     rhoRef : float
#         no description

#     Returns
#     -------
#     hmxl : float
#         mixed layer depth [m]
#     """
#     N = bvf.shape[0]-1
#     # find the ref depth index
#     kstart = N - 1
#     cond_fun1 = lambda val: val[1][val[0]] > zref
#     body_fun1 = lambda val: (val[0]-1, val[1])
#     kstart, _ = lax.while_loop(cond_fun1, body_fun1, (N-1, zr))

#     bvf_c = rhoc * (grav / rhoRef)
#     # initialize at the near bottom value
#     hmxl = zr[kstart]


#     cond_fun2 = lambda val: ((val[0]>0) & (val[1] < bvf_c))
#     def body_fun2(val):
#         (k, cff_k, cff_km1, bvf, zr) = val
#         cff_new = cff_k + jnp.maximum(bvf[k], 0.0) * (zr[k]-zr[k-1])
#         cff_km1 = cff_k
#         cff_k = cff_new
#         k -= 1
#         return (k, cff_k, cff_km1, bvf, zr)

#     (k, cff_k, cff_km1, _, _) = lax.while_loop(cond_fun2, body_fun2,
#                                                (kstart, 0., 0., bvf, zr))

#     hmxl_new = ((cff_k-bvf_c)*zr[k+1] + (bvf_c-cff_km1)*zr[k]) / \
#         (cff_k - cff_km1)
#     hmxl = lax.select(cff_k >= bvf_c, hmxl_new, hmxl)

#     return hmxl


# @jit
# def rho_eos(temp, salt, zr, zw, rhoRef):
#     """Compute density anomaly and Brunt Vaisala frequency via nonlinear
#     Equation Of State (EOS).

#     Parameters
#     ----------
#     temp : float(N)
#         temperature [C]
#     salt : float(N)
#         salinity [psu]
#     zr : float(N)
#         depth at cell centers [m]
#     zw : float(N+1)
#         depth at cell interfaces [m]
#     rhoRef : float
#         no description

#     Returns
#     -------
#     bvf : float(N+1)
#         Brunt Vaisala frequancy [s-2]
#     rho : float(N)
#         density anomaly [kg/m3]
#     """
#     # returned variables
#     N, = temp.shape
#     bvf = jnp.zeros(N+1)
#     rho = jnp.zeros(N)

#     # local variables
#     rho1 = jnp.zeros(N)
#     K_up = jnp.zeros(N)
#     K_dw = jnp.zeros(N)

#     # constants
#     r00, r01, r02, r03, r04, r05 = 999.842594, 6.793952E-2, -9.095290E-3, \
#         1.001685E-4, -1.120083E-6, 6.536332E-9
#     r10, r11, r12, r13, r14, r20 = 0.824493, -4.08990E-3, 7.64380E-5, \
#         -8.24670E-7, 5.38750E-9, 4.8314E-4
#     rS0, rS1, rS2 = -5.72466E-3, 1.02270E-4, -1.65460E-6
#     k00, k01, k02, k03, k04 = 19092.56, 209.8925, -3.041638, -1.852732e-3, \
#         -1.361629e-5
#     k10, k11, k12, k13 = 104.4077, -6.500517, 0.1553190, 2.326469e-4
#     ks0, ks1, ks2 = -5.587545, 0.7390729, -1.909078e-2
#     b00, b01, b02, b03, b10, b11, b12, bs1 = 0.4721788, 0.01028859, \
#         -2.512549e-4, -5.939910e-7, -0.01571896, -2.598241e-4, 7.267926e-6, \
#         2.042967e-3
#     e00, e01, e02, e10, e11, e12 = 1.045941e-5, -5.782165e-10, 1.296821e-7, \
#         -2.595994e-7, -1.248266e-9, -3.508914e-9

#     dr00 = r00 - rhoRef

#     # density anomaly
#     sqrtTs = jnp.sqrt(salt)
#     rho1 = dr00 + temp*(r01 + temp*(r02 + temp*(r03 + temp*(r04 + temp*r05))))\
#         + salt*(r10 + temp*(r11 + temp*(r12 + temp*(r13 + temp*r14))) +
#         sqrtTs*(rS0 + temp*(rS1 + temp*rS2)) + salt*r20)


#     k0 = temp*(k01 + temp*(k02 + temp*(k03 + temp*k04))) + \
#         salt*(k10 + temp*(k11 + temp*(k12 + temp*k13)) +
#         sqrtTs*(ks0 + temp*(ks1 + temp*ks2)))

#     k1 = b00 + temp*(b01 + temp*(b02 + temp*b03)) + \
#         salt*(b10 + temp*(b11 + temp*b12) + sqrtTs*bs1)
#     k2 = e00 + temp*(e01 + temp*e02) + salt*(e10 + temp*(e11 + temp*e12))

#     dpth = -zr
#     cff = k00 - 0.1*dpth
#     cff1 = k0 + dpth*(k1 + k2*dpth)
#     rho = (rho1*cff*(k00 + cff1) - 0.1*dpth*rhoRef*cff1) / (cff*(cff + cff1))

#     K_up = k0 - zw[1:]*(k1 - k2*zw[1:])
#     K_dw = k0 - zw[:-1]*(k1 - k2*zw[:-1])

#     cff = grav / rhoRef
#     cff1 = -0.1*zw[1:-1]
#     bvf = -cff*((rho1[1:]-rho1[:-1]) * (k00+K_dw[1:]) * (k00+K_up[:-1]) -
# cff1*(rhoRef*(K_dw[1:] - K_up[:-1]) + k00*(rho1[1:] - rho1[:-1]) +
# rho1[1:]*K_dw[1:] - rho1[:-1]*K_up[:-1])) / ((k00 + K_dw[1:] - cff1)*(k00 +
# K_up[:-1] - cff1)*(zr[1:] - zr[:-1]))

#     bvf = jnp.concat((jnp.array([0.]), bvf, jnp.array([0])))

#     return rho, bvf
