"""
k-epsilon closure parameters, states and functions.

This module contains the implementation of the k-epsilon closure described as
a GLS module as in [1]. The parameters of the closure are available in
the `KepsParameters` class and one can play with them. The function `keps_step`
compute one time-step of the closure, which means that it computes the eddy-
diffusivity and viscosity.

Classes
-------
KepsParameters
    concrete class from ClosureParametersAbstract for k-epsilon closure
KepsState
    concrete class from ClosureStateAbstract for k-epsilon closure

Functions
---------
keps_step (jitted version)
    main function, run one time-step of the k-epsilon closure
compute_rho_eos
    compute density anomaly and Brunt Vaisala frequency via linear EOS
compute_shear
    compute shear production term for TKE equation
compute_tke_eps_bc
    compute top and bottom boundary conditions for TKE and GLS equation
advance_turb
    integrate TKE or epsilon quantities
compute_diag
    computes the diagnostic variables of k-epsilon closure

References
----------
.. [1] L. Umlauf and H. Burchard. A generic length-scale equation for
    geophysical turbulence models (2003). Journal of Marine Research 61
    pp. 235-265. doi : 10.1357/002224003322005087
    
"""


import sys
from functools import partial
from typing import Tuple

import equinox as eqx
import jax.numpy as jnp
from jax import jit, lax

sys.path.append('..')
from case import Case
from state import Grid, State
from functions import tridiag_solve, add_boundaries
from closure import ClosureParametersAbstract, ClosureStateAbstract


class KepsParameters(ClosureParametersAbstract):
    """
    Set of the physical constants and coefficients used in k-epsilon closure.

    Attributes
    ----------
    c1 : float, default=5.
        k-epsilon parameter for the dissipation of the corelation tensor
        pressure/velocity (Umlauf and Burchard notations) [dimensionless]
    c2 : float, default=0.8
        k-epsilon parameter for the dissipation of the corelation tensor
        pressure/velocity (Umlauf and Burchard notations) [dimensionless]
    c3 : float, default=1.968
        k-epsilon parameter for the dissipation of the corelation tensor
        pressure/velocity (Umlauf and Burchard notations) [dimensionless]
    c4 : float, default=1.136
        k-epsilon parameter for the dissipation of the corelation tensor
        pressure/velocity (Umlauf and Burchard notations) [dimensionless]
    c5 : float, default=0.
        k-epsilon parameter for the dissipation of the corelation tensor
        pressure/velocity (Umlauf and Burchard notations) [dimensionless]
    c6 : float, default=0.4
        k-epsilon parameter for the dissipation of the corelation tensor
        pressure/velocity (Umlauf and Burchard notations) [dimensionless]
    cb1 : float, default=5.95
        k-epsilon parameter for the dissipation of the corelation tensor
        buoyancy/velocity (Umlauf and Burchard notations) [dimensionless]
    cb2 : float, default=.6
        k-epsilon parameter for the dissipation of the corelation tensor
        buoyancy/velocity (Umlauf and Burchard notations) [dimensionless]
    cb3 : float, default=1.
        k-epsilon parameter for the dissipation of the corelation tensor
        buoyancy/velocity (Umlauf and Burchard notations) [dimensionless]
    cb4 : float, default=0.
        k-epsilon parameter for the dissipation of the corelation tensor
        buoyancy/velocity (Umlauf and Burchard notations) [dimensionless]
    cb5 : float, default=0.3333
        k-epsilon parameter for the dissipation of the corelation tensor
        buoyancy/velocity (Umlauf and Burchard notations) [dimensionless]
    cbb : float, default=.72
        k-epsilon parameter for the dissipation of the corelation tensor
        buoyancy/velocity (Umlauf and Burchard notations) [dimensionless]
    c_mu0 : float, default=0.5477
        k-epsilon parameter which links the mixing length to the dissipation
        (Umlauf and Burchard notations) [dimensionless]
    sig_k : float, default=1.
        k-epsilon parameter Schmit number for the dissipation of TKE (Umlauf
        and Burchard notations) [dimensionless]
    sig_eps : float, default=1.3
        k-epsilon parameter Schmit number for the dissipation of epsilon
        (Umlauf and Burchard notations) [dimensionless]
    c_eps1 : float, default=1.44
        k-epsilon parameter correction of the epsilon equation (Umlauf and
        Burchard notations) [dimensionless]
    c_eps2 : float, default=1.92
        k-epsilon parameter correction of the epsilon equation (Umlauf and
        Burchard notations) [dimensionless]
    c_eps3m : float, default=-0.4
        k-epsilon parameter correction of the epsilon equation (Umlauf and
        Burchard notations) [dimensionless]
    c_eps3p : float, default=1.
        k-epsilon parameter correction of the epsilon equation (Umlauf and
        Burchard notations) [dimensionless]
    chk_grav : float, default=1400.
        charnock coefficient times gravity [dimensionless]
    galp: float, default=0.53
        parameter for Galperin mixing length limitation [dimensionless]
    z0s_min : float, default=1e-2
        minimal surface roughness length [m]
    z0b_min : float, default=1e-4
        minimal bottom roughness length [m]
    z0b : float, default=1e-14
        bottom roughness length [m]
    akt_min : float, default=1e-5
        minimal and initialization value of eddy-diffusivity [m2.s-1]
    akt_min : float, default=1e-4
        minimal and initialization value of eddy-viscosity [m2.s-1]
    tke_min : float, default=1e-6
        minimal and initialization value of turbulent kinetic energy (TKE)
        [m3.s-2]
    eps_min : float, default=1e-12
        minimal and initialization value of TKE dissipation [m2.s-3]
    c_mu_min : float, default=0.1
        minimal and initialization value of c_mu in GLS formalisim
        [dimensionless]
    c_mu_prim_min : float, default=0.1
        minimal and initialization value of c_mu' in GLS formalisim
        [dimensionless]
    dir_sfc: bool, default=False
        apply a Dirichlet boundary condition at the surface for TKE, else
        apply a Neumann boundary condition
    dir_btm: bool, default=True
        apply a Dirichlet boundary condition at the bottom for TKE, else
        apply a Neumann boundary condition
    gls_p : float, default=3
        GLS coefficient p to define k-epsilon [dimensionless]
    gls_m : float, default=1.5
        GLS coefficient m to define k-epsilon [dimensionless]
    gls_n : float, default=-1
        GLS coefficient n to define k-epsilon [dimensionless]
    sf_d0 : float
        limitation coefficient for k-epsilon computed from the parameters (not
        a parameter for init)
    sf_d1 : float
        limitation coefficient for k-epsilon computed from the parameters (not
        a parameter for init)
    sf_d2 : float
        limitation coefficient for k-epsilon computed from the parameters (not
        a parameter for init)
    sf_d3 : float
        limitation coefficient for k-epsilon computed from the parameters (not
        a parameter for init)
    sf_d4 : float
        limitation coefficient for k-epsilon computed from the parameters (not
        a parameter for init)
    sf_d5 : float
        limitation coefficient for k-epsilon computed from the parameters (not
        a parameter for init)
    sf_n0 : float
        limitation coefficient for k-epsilon computed from the parameters (not
        a parameter for init)
    sf_n1 : float
        limitation coefficient for k-epsilon computed from the parameters (not
        a parameter for init)
    sf_n2 : float
        limitation coefficient for k-epsilon computed from the parameters (not
        a parameter for init)
    sf_nb0 : float
        limitation coefficient for k-epsilon computed from the parameters (not
        a parameter for init)
    sf_nb1 : float
        limitation coefficient for k-epsilon computed from the parameters (not
        a parameter for init)
    sf_nb2 : float
        limitation coefficient for k-epsilon computed from the parameters (not
        a parameter for init)
    lim_am0 : float
        limitation coefficient for k-epsilon computed from the parameters (not
        a parameter for init)
    lim_am1 : float
        limitation coefficient for k-epsilon computed from the parameters (not
        a parameter for init)
    lim_am2 : float
        limitation coefficient for k-epsilon computed from the parameters (not
        a parameter for init)
    lim_am3 : float
        limitation coefficient for k-epsilon computed from the parameters (not
        a parameter for init)
    lim_am4 : float
        limitation coefficient for k-epsilon computed from the parameters (not
        a parameter for init)
    lim_am5 : float
        limitation coefficient for k-epsilon computed from the parameters (not
        a parameter for init)
    lim_am6 : float
        limitation coefficient for k-epsilon computed from the parameters (not
        a parameter for init)
    
    """

    # k-epsilon coefficients (Umlauf and Burchard notations)
    c1: float = 5.
    c2: float = .8
    c3: float = 1.968
    c4: float = 1.136
    c5: float = 0.
    c6: float = .4
    cb1: float = 5.95
    cb2: float = .6
    cb3: float = 1.
    cb4: float = 0.
    cb5: float = 0.3333
    cbb: float = 0.72
    c_mu0: float = 0.5477
    sig_k: float = 1.
    sig_eps: float = 1.3
    c_eps1: float = 1.44
    c_eps2: float = 1.92
    c_eps3m: float = -.4
    c_eps3p: float = 1.
    # physical constants
    chk_grav: float = 1400.
    galp: float = .53
    z0s_min: float = 1e-2
    z0b_min: float = 1e-2
    z0b: float = 1e-14
    akt_min: float = 1e-5
    akv_min: float = 1e-4
    tke_min: float = 1e-6
    eps_min: float = 1e-12
    c_mu_min: float = .1
    c_mu_prim_min: float = .1
    # physical case
    dir_sfc: bool = False
    dir_btm: bool = True
    # GLS coefficient for k-epsilon
    gls_p: float = 3
    gls_m: float = 1.5
    gls_n: float = -1
    # limitation coefficients computed from k-epsilon coefficients
    sf_d0: float = eqx.field(init=False)
    sf_d1: float = eqx.field(init=False)
    sf_d2: float = eqx.field(init=False)
    sf_d3: float = eqx.field(init=False)
    sf_d4: float = eqx.field(init=False)
    sf_d5: float = eqx.field(init=False)
    sf_n0: float = eqx.field(init=False)
    sf_n1: float = eqx.field(init=False)
    sf_n2: float = eqx.field(init=False)
    sf_nb0: float = eqx.field(init=False)
    sf_nb1: float = eqx.field(init=False)
    sf_nb2: float = eqx.field(init=False)
    lim_am0: float = eqx.field(init=False)
    lim_am1: float = eqx.field(init=False)
    lim_am2: float = eqx.field(init=False)
    lim_am3: float = eqx.field(init=False)
    lim_am4: float = eqx.field(init=False)
    lim_am5: float = eqx.field(init=False)
    lim_am6: float = eqx.field(init=False)

    def __post_init__(self):
        a1 = .66666666667 - .5*self.c2
        a2 = 1 - .5*self.c3
        a3 = 1 - .5*self.c4
        a5 = .5 - .5*self.c6
        nn = .5*self.c1
        nb = self.cb1
        ab1 = 1 - self.cb2
        ab2 = 1 - self.cb3
        ab3 = 2*(1 - self.cb4)
        ab5 = 2*self.cbb*(1-self.cb5)
        sf_d0 = 36*nn*nn*nn*nb*nb
        sf_d1 = 84*a5*ab3*nn*nn*nb + 36*ab5*nn*nn*nn*nb
        sf_d2 = 9*(ab2*ab2 - ab1*ab1)*nn*nn*nn - 12*(a2*a2 - 3*a3*a3)*nn*nb*nb
        sf_d3 = 12*a5*ab3*(a2*ab1 - 3*a3*ab2)*nn + \
            12*a5*ab3*(a3*a3 - a2*a2)*nb + 12*ab5*(3*a3*a3 - a2*a2)*nn*nb
        sf_d4 = 48*a5*a5*ab3*ab3*nn + 36*a5*ab3*ab5*nn*nn
        sf_d5 = 3*(a2*a2 - 3*a3*a3)*(ab1*ab1 - ab2*ab2)*nn
        sf_n0 = 36*a1*nn*nn*nb*nb
        sf_n1 = -12*a5*ab3*(ab1 + ab2)*nn*nn + \
            8*a5*ab3*(6*a1 - a2 - 3*a3)*nn*nb + 36*a1*ab5*nn*nn*nb
        sf_n2 = 9*a1*(ab2*ab2 - ab1*ab1)*nn*nn
        sf_nb0 = 12*ab3*nn*nn*nn*nb
        sf_nb1 = 12*a5*ab3*ab3*nn*nn
        sf_nb2 = 9*a1*ab3*(ab1 - ab2)*nn*nn + \
            (6*a1*(a2 - 3*a3) - 4*(a2*a2 - 3*a3*a3))*ab3*nn*nb
        lim_am0 = sf_d0*sf_n0
        lim_am1 = sf_d0*sf_n1 + sf_d1*sf_n0
        lim_am2 = sf_d1*sf_n1 + sf_d4*sf_n0
        lim_am3 = sf_d4*sf_n1
        lim_am4 = sf_d2*sf_n0
        lim_am5 = sf_d2*sf_n1 + sf_d3*sf_n0
        lim_am6 = sf_d3*sf_n1
        object.__setattr__(self, 'sf_d0', sf_d0)
        object.__setattr__(self, 'sf_d1', sf_d1)
        object.__setattr__(self, 'sf_d2', sf_d2)
        object.__setattr__(self, 'sf_d3', sf_d3)
        object.__setattr__(self, 'sf_d4', sf_d4)
        object.__setattr__(self, 'sf_d5', sf_d5)
        object.__setattr__(self, 'sf_n0', sf_n0)
        object.__setattr__(self, 'sf_n1', sf_n1)
        object.__setattr__(self, 'sf_n2', sf_n2)
        object.__setattr__(self, 'sf_nb0', sf_nb0)
        object.__setattr__(self, 'sf_nb1', sf_nb1)
        object.__setattr__(self, 'sf_nb2', sf_nb2)
        object.__setattr__(self, 'lim_am0', lim_am0)
        object.__setattr__(self, 'lim_am1', lim_am1)
        object.__setattr__(self, 'lim_am2', lim_am2)
        object.__setattr__(self, 'lim_am3', lim_am3)
        object.__setattr__(self, 'lim_am4', lim_am4)
        object.__setattr__(self, 'lim_am5', lim_am5)
        object.__setattr__(self, 'lim_am6', lim_am6)


class KepsState(ClosureStateAbstract):
    """
    Define the state of the k-epsilon closure at one time-step on one grid.

    Parameters
    ----------
    grid : Grid
        spatial grid
    keps_params : KepsParameters
        define the initialization values of the variables

    Attributes
    ----------
    grid : Grid
        spatial grid
    akt : jnp.ndarray, float(nz+1)
        eddy-diffusivity [m2.s-1]
    akv : jnp.ndarray, float(nz+1)
        eddy-viscosity [m2.s-1]
    tke : jnp.ndarray, float(nz+1)
        turbulent kinetic energy (TKE) [m2.s-2]
    eps : jnp.ndarray, float(nz+1)
        TKE dissipation [m2.s-3]
    c_mu : jnp.ndarray, float(nz+1)
        c_mu in GLS formalisim [dimensionless]
    c_mu_prim : jnp.ndarray, float(nz+1)
        c_mu' in GLS formalisim [dimensionless]

    """

    grid: Grid
    akt: jnp.ndarray
    akv: jnp.ndarray
    tke: jnp.ndarray
    eps: jnp.ndarray
    c_mu: jnp.ndarray
    c_mu_prim: jnp.ndarray

    def __init__(self, grid: Grid, keps_params: KepsParameters):
        self.grid = grid
        nz = grid.nz
        self.akt = jnp.full(nz+1, keps_params.akt_min)
        self.akv = jnp.full(nz+1, keps_params.akv_min)
        self.tke = jnp.full(nz+1, keps_params.tke_min)
        self.eps = jnp.full(nz+1, keps_params.eps_min)
        self.c_mu = jnp.full(nz+1, keps_params.c_mu_min)
        self.c_mu_prim = jnp.full(nz+1, keps_params.c_mu_prim_min)


@partial(jit, static_argnames=('case',))
def keps_step(
        state: State,
        keps_state: KepsState,
        dt: float,
        keps_params: KepsParameters,
        case: Case
    ) -> KepsState:
    """
    Run one time-step of the k-epsilon closure.

    Parameters
    ----------
    state : State
        curent state of the system
    keps_state : KepsState
        curent state of k-epsilon
    dt : float
        time-step for every iteration [s]
    keps_params: KepsParameters
        k-epsilon parameters
    case : Case
        physical case
    
    Returns
    -------
    keps_state : KepsState
        state of k-epsilon at the next iteration
    """
    akt = keps_state.akt
    akv = keps_state.akv
    tke = keps_state.tke
    eps = keps_state.eps
    c_mu = keps_state.c_mu
    c_mu_prim = keps_state.c_mu_prim
    u = state.u
    v = state.v
    zr = state.grid.zr
    hz = state.grid.hz

    # prognostic computations
    _, bvf = compute_rho_eos(state.t, state.s, zr, case)
    shear2 = compute_shear(u, v, u, v, zr)
    tke_sfc_bc, tke_btm_bc, eps_sfc_bc, eps_btm_bc = compute_tke_eps_bc(
        tke, hz, keps_params, case
    )

    # integrations
    tke_new = advance_turb(
        akt, akv, tke, tke, eps, c_mu, c_mu_prim, bvf, shear2, hz, dt,
        tke_sfc_bc, tke_btm_bc, eps_sfc_bc, eps_btm_bc, keps_params, True
    )
    eps_new = advance_turb(
        akt, akv, tke, tke_new, eps, c_mu, c_mu_prim, bvf, shear2, hz, dt,
        tke_sfc_bc, tke_btm_bc, eps_sfc_bc, eps_btm_bc, keps_params, False
    )

    # diagnostic variables
    akt_new, akv_new, eps_new, c_mu_new, c_mu_prim_new = compute_diag(
        tke_new, eps_new, bvf, shear2, keps_params
    )

    keps_state = eqx.tree_at(lambda t: t.akv, keps_state, akv_new)
    keps_state = eqx.tree_at(lambda t: t.akt, keps_state, akt_new)
    keps_state = eqx.tree_at(lambda t: t.tke, keps_state, tke_new)
    keps_state = eqx.tree_at(lambda t: t.eps, keps_state, eps_new)
    keps_state = eqx.tree_at(lambda t: t.c_mu, keps_state, c_mu_new)
    keps_state = eqx.tree_at(lambda t: t.c_mu_prim, keps_state, c_mu_prim_new)

    return keps_state


def compute_rho_eos(
        t: jnp.ndarray,
        s: jnp.ndarray,
        zr: jnp.ndarray,
        case: Case
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    r"""
    Compute density anomaly and Brunt Vaisala frequency via linear Equation Of
    State (EOS).

    Parameters
    ----------
    t : jnp.ndarray float(nz)
        temperature [C]
    s : jnp.ndarray float(nz)
        salinity [psu]
    zr : jnp.ndarray, float(nz)
        depths of cell centers from deepest to shallowest [m]
    case : Case
        physical case and forcings

    Returns
    -------
    bvf : float(nz+1)
        Brunt Vaisala frequency [s-2]
    rho : float(nz)
        density anomaly [kg.m-3]

    Notes
    -----
    rho
    \(   \rho_{k} = \rho_0 \left( 1 - \alpha (\theta - 2) + \beta (S - 35)
    \right)  \)
    bvf
    \(   (nz^2)_{k+1/2} = - \frac{g}{\rho_0}  \frac{ \rho_{k+1}-\rho_{k} }
    {\Delta z_{k+1/2}} \)
    """
    rho0 = case.rho0
    rho = rho0 * (1. - case.alpha*(t-case.t_rho_ref) + \
                  case.beta*(s-case.s_rho_ref))
    cff = 1./(zr[1:]-zr[:-1])
    bvf_in = - cff*case.grav/rho0 * (rho[1:]-rho[:-1])
    bvf = add_boundaries(0., bvf_in, bvf_in[-1])

    return rho, bvf


def compute_shear(
        u_n: jnp.ndarray,
        v_n: jnp.ndarray,
        u_np1: jnp.ndarray,
        v_np1: jnp.ndarray,
        zr: jnp.ndarray
    ) -> jnp.ndarray:
    r"""
    Compute shear production term for TKE equation.
    
    Parameters
    ----------
    u_n : jnp.ndarray, float(nz)
        current zonal velocity [m.s-1]
    v_n : jnp.ndarray, float(nz)
        current meridional velocity [m.s-1]
    u_np1 : jnp.ndarray, float(nz)
        zonal velocity at the next step [m.s-1]
    v_np1 : jnp.ndarray, float(nz)
        meridional velocity at the next step [m.s-1]
    zr : jnp.ndarray, float(nz)
        depths of cell centers from deepest to shallowest [m]
        

    Returns
    -------
    shear2 : jnp.ndarray, float(nz+1)
        shear production term [m2.s-3]

    Notes
    -----
    Shear production term using discretization from Burchard (2002)
    \( {\rm Sh}_{k+1/2} = \frac{ 1 }{ \Delta z_{k+1/2}^2 } ( u_{k+1}^n -
    u_{k}^n ) ( u_{k+1}^{n+1/2} - u_{k}^{n+1/2} )  \)
    """
    cff = 1.0 / (zr[1:] - zr[:-1])**2
    du = 0.5*cff * (u_np1[1:]-u_np1[:-1]) * \
        (u_n[1:]+u_np1[1:]-u_n[:-1]-u_np1[:-1])
    dv = 0.5*cff * (v_np1[1:]-v_np1[:-1]) * \
        (v_n[1:]+v_np1[1:]-v_n[:-1]-v_np1[:-1])
    shear2_in = du + dv
    return add_boundaries(0., shear2_in, 0.)


def compute_tke_eps_bc(
        tke: jnp.ndarray,
        hz: jnp.ndarray,
        keps_params: KepsParameters,
        case: Case
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute top and bottom boundary conditions for TKE and GLS equation.

    Parameters
    ----------
    tke : jnp.ndarray, float(nz+1)
        turbulent kinetic energy (TKE) [m3.s-2]
    hz : jnp.ndarray, float(nz)
        thickness of cells from deepest to shallowest [m]
    keps_params: KepsParameters
        k-epsilon parameters
    case : Case
        physical case

    Returns
    -------
    tke_sfc_bc : float
        TKE value for surface boundary condition (Dirichlet [m2.s-2] or Neumann
        [m3.s-3])
    tke_btm_bc : float
        TKE value for bottom boundary condition (Dirichlet [m2.s-2] or Neumann
        [m3.s-3])
    eps_sfc_bc : float
        epsilon value for surface boundary condition (Dirichlet [m2.s-3] or
        Neumann [m3.s-4])
    eps_btm_bc : float
        epsilon value for surface boundary condition (Dirichlet [m2.s-3] or
        Neumann [m3.s-4])
    """
    # constants
    rp, rm, rn = keps_params.gls_p, keps_params.gls_m, keps_params.gls_n
    c_mu0 = keps_params.c_mu0
    cm0inv2 = 1./c_mu0**2
    vkarmn = case.vkarmn
    chk = keps_params.chk_grav/case.grav
    sig_eps = keps_params.sig_eps

    # velocity scales
    ustar2_sfc = jnp.sqrt(case.ustr_sfc**2 + case.vstr_sfc**2)
    ustar2_bot = jnp.sqrt(case.ustr_btm**2 + case.vstr_btm**2)

    # TKE Dirichlet boundary condition
    tke_sfc_dir = jnp.maximum(keps_params.tke_min, cm0inv2*ustar2_sfc)
    tke_btm_dir = jnp.maximum(keps_params.tke_min, cm0inv2*ustar2_bot)

    # TKE Neumann boundary condition
    tke_sfc_neu = 0.0
    tke_btm_neu = 0.0

    # epsilon surface conditions
    z0_s = jnp.maximum(keps_params.z0s_min, chk*ustar2_sfc)
    lgthsc = vkarmn*(0.5*hz[-1] + z0_s)
    tke_sfc = 0.5*(tke[-1]+tke[-2])
    eps_sfc_dir = jnp.maximum(keps_params.eps_min, \
        c_mu0**rp * lgthsc**rn * tke_sfc**rm)
    eps_sfc_neu = -rn*vkarmn/sig_eps * c_mu0**(rp+1) * \
        tke_sfc**(rm+.5) * lgthsc**rn

    # epsilon bottom conditions
    z0b = jnp.maximum(keps_params.z0b, keps_params.z0b_min)
    lgthsc = vkarmn *(0.5*hz[0] + z0b)
    tke_btm = 0.5*(tke[0]+tke[1])
    eps_btm_dir = jnp.maximum(keps_params.eps_min, \
        c_mu0**rp * lgthsc**rn * tke_btm**rm)
    eps_btm_neu = -rn*vkarmn/sig_eps * c_mu0**(rp+1) * \
        tke_btm**(rm+.5) * lgthsc**rn

    tke_sfc_bc = jnp.where(keps_params.dir_sfc, tke_sfc_dir, tke_sfc_neu)
    tke_btm_bc = jnp.where(keps_params.dir_btm, tke_btm_dir, tke_btm_neu)
    eps_sfc_bc = jnp.where(keps_params.dir_sfc, eps_sfc_dir, eps_sfc_neu)
    eps_btm_bc = jnp.where(keps_params.dir_btm, eps_btm_dir, eps_btm_neu)

    return tke_sfc_bc, tke_btm_bc, eps_sfc_bc, eps_btm_bc


def advance_turb(
        akt: jnp.ndarray,
        akv: jnp.ndarray,
        tke: jnp.ndarray,
        tke_np1: jnp.ndarray,
        eps: jnp.ndarray,
        c_mu: jnp.ndarray,
        c_mu_prim: jnp.ndarray,
        bvf: jnp.ndarray,
        shear2: jnp.ndarray,
        hz: jnp.ndarray,
        dt: float,
        tke_sfc_bc: float,
        tke_btm_bc: float,
        eps_sfc_bc: float,
        eps_btm_bc: float,
        keps_params: KepsParameters,
        do_tke: bool
    ) -> jnp.ndarray:
    """
    Integrate TKE or epsilon quantities.

    Parameters
    ----------
    akt : jnp.ndarray, float(nz+1)
        current eddy-diffusivity [m2.s-1]
    akv : jnp.ndarray, float(nz+1)
        current eddy-viscosity [m2.s-1]
    tke : jnp.ndarray, float(nz+1)
        current turbulent kinetic energy (TKE) [m2.s-2]
    tke : jnp.ndarray, float(nz+1)
        turbulent kinetic energy (TKE) at next step (usefull only for epsilon
        integration) [m2.s-2]
    eps : jnp.ndarray, float(nz+1)
        current TKE dissipation [m2.s-3]
    c_mu : jnp.ndarray, float(nz+1)
        current c_mu in GLS formalisim [dimensionless]
    c_mu_prim : jnp.ndarray, float(nz+1)
        current c_mu' in GLS formalisim [dimensionless]
    bvf : float(nz+1)
        current Brunt Vaisala frequency [s-2]
    shear2 : jnp.ndarray, float(nz+1)
        current shear production term [m2.s-3]
    hz : jnp.ndarray, float(nz)
        thickness of cells from deepest to shallowest [m]
    dt : float
        time-step [s]
    tke_sfc_bc : float
        TKE value for surface boundary condition (Dirichlet [m2.s-2] or Neumann
        [m3.s-3])
    tke_btm_bc : float
        TKE value for bottom boundary condition (Dirichlet [m2.s-2] or Neumann
        [m3.s-3])
    eps_sfc_bc : float
        epsilon value for surface boundary condition (Dirichlet [m2.s-3] or
        Neumann [m3.s-4])
    eps_btm_bc : float
        epsilon value for surface boundary condition (Dirichlet [m2.s-3] or
        Neumann [m3.s-4])
    keps_params: KepsParameters
        k-epsilon parameters
    do_tke : bool
        integrate TKE if True and epsilon if False

    Returns
    -------
    vec : jnp.ndarray, float(nz+1)
        TKE or epsilon at next step (depending on `do_tke`)
    """
    # fill the matrix off-diagonal terms for the tridiagonal problem
    cff = -0.5*dt
    ak_vec = jnp.where(do_tke, akv/keps_params.sig_k, akv/keps_params.sig_eps)
    a_in = cff*(ak_vec[1:-1]+ak_vec[:-2]) / hz[:-1]
    c_in = cff*(ak_vec[1:-1]+ak_vec[2:]) / hz[1:]

    # shear and buoyancy production
    s_prod_tke = akv[1:-1]*shear2[1:-1]
    b_prod_tke = -akt[1:-1]*bvf[1:-1]
    s_prod_eps = keps_params.c_eps1*c_mu[1:-1]*tke[1:-1]*shear2[1:-1]
    b_prod_eps = -c_mu_prim[1:-1] * tke[1:-1] * \
        (keps_params.c_eps3m*jnp.maximum(bvf[1:-1], 0) + \
         keps_params.c_eps3p*jnp.minimum(bvf[1:-1], 0))
    s_prod = jnp.where(do_tke, s_prod_tke, s_prod_eps)
    b_prod = jnp.where(do_tke, b_prod_tke, b_prod_eps)

    # diagonal and f term
    cff = 0.5*(hz[:-1] + hz[1:])
    f_tke_in = lax.select(b_prod+s_prod > 0,
                            cff*(tke[1:-1]+dt*(b_prod+s_prod)),
                            cff*(tke[1:-1]+dt*s_prod))
    f_eps_in = lax.select(b_prod+s_prod > 0,
                            cff*(eps[1:-1]+dt*(b_prod+s_prod)),
                            cff*(eps[1:-1]+dt*s_prod))
    f_in = jnp.where(do_tke, f_tke_in, f_eps_in)
    b_tke_in = lax.select((b_prod + s_prod) > 0,
        cff*(1. + dt*eps[1:-1]/tke[1:-1]) - a_in - c_in,
        cff*(1. + dt*(eps[1:-1] - b_prod)/tke[1:-1]) - a_in - c_in)
    b_eps_in = lax.select((b_prod + s_prod) > 0,
        cff*(1. + dt*keps_params.c_eps2*eps[1:-1]/tke_np1[1:-1]) - a_in - c_in,
        cff*(1. + dt*keps_params.c_eps2*eps[1:-1]/tke_np1[1:-1] - \
             dt*b_prod/eps[1:-1]) - a_in - c_in)
    b_in = jnp.where(do_tke, b_tke_in, b_eps_in)

    # surface boundary condition
    dir_sfc = keps_params.dir_sfc
    a_sfc = jnp.where(dir_sfc, 0., -0.5*(ak_vec[-1] + ak_vec[-2]))
    b_sfc = jnp.where(dir_sfc, 1., 0.5*(ak_vec[-1] + ak_vec[-2]))
    sfc_bc = jnp.where(do_tke, tke_sfc_bc, eps_sfc_bc)
    f_sfc = jnp.where(dir_sfc, sfc_bc, hz[-1]*sfc_bc)

    # bottom boundary condition
    dir_btm = keps_params.dir_btm
    b_btm = jnp.where(dir_sfc, 1., -0.5*(ak_vec[0] + ak_vec[1]))
    c_btm = jnp.where(dir_sfc, 0., 0.5*(ak_vec[0] + ak_vec[1]))
    btm_bc = jnp.where(do_tke, tke_btm_bc, eps_btm_bc)
    f_btm = jnp.where(dir_btm, btm_bc, hz[0]*btm_bc)

    # vectors rassembly
    a = add_boundaries(0., a_in, a_sfc)
    b = add_boundaries(b_btm, b_in, b_sfc)
    c = add_boundaries(c_btm, c_in, 0.)
    f = add_boundaries(f_btm, f_in, f_sfc)

    # solve tridiagonal problem
    f = tridiag_solve(a, b, c, f)

    vec_min = jnp.where(do_tke, keps_params.tke_min, keps_params.eps_min)
    vec = jnp.maximum(f, vec_min)

    return vec

def compute_diag(
        tke: jnp.ndarray,
        eps: jnp.ndarray,
        bvf: jnp.ndarray,
        shear2: jnp.ndarray,
        keps_params: KepsParameters
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, \
               jnp.ndarray]:
    """
    Computes the diagnostic variables of k-epsilon closure.

    Parameters
    ----------
    tke : jnp.ndarray, float(nz+1)
        turbulent kinetic energy (TKE) at next step [m2.s-2]
    eps : jnp.ndarray, float(nz+1)
        TKE dissipation at next step [m2.s-3]
    bvf : float(-1+1)
        current Brunt Vaisala frequency [s-2]
    shear2 : jnp.ndarray, float(nz+1)
        current shear production term [m2.s-3]
    keps_params: KepsParameters
        k-epsilon parameters

    Returns
    -------
    akt : jnp.ndarray, float(nz+1)
        eddy-diffusivity at next step [m2.s-1]
    akv : jnp.ndarray, float(nz+1)
        eddy-viscosity at next step [m2.s-1]
    eps : jnp.ndarray, float(nz+1)
        TKE dissipation at next step [m2.s-3]
    c_mu : jnp.ndarray, float(nz+1)
        c_mu in GLS formalisim at next step [dimensionless]
    c_mu_prim : jnp.ndarray, float(nz)
        c_mu' in GLS formalisim at next step [dimensionless]
    """
    # parameters
    akv_min = keps_params.akv_min
    akt_min = keps_params.akt_min
    eps_min = keps_params.eps_min
    sf_d0 = keps_params.sf_d0
    sf_d1 = keps_params.sf_d1
    sf_d2 = keps_params.sf_d2
    sf_d3 = keps_params.sf_d3
    sf_d4 = keps_params.sf_d4
    sf_d5 = keps_params.sf_d5
    sf_n0 = keps_params.sf_n0
    sf_n1 = keps_params.sf_n1
    sf_n2 = keps_params.sf_n2
    sf_nb0 = keps_params.sf_nb0
    sf_nb1 = keps_params.sf_nb1
    sf_nb2 = keps_params.sf_nb2
    lim_am0 = keps_params.lim_am0
    lim_am1 = keps_params.lim_am1
    lim_am2 = keps_params.lim_am2
    lim_am3 = keps_params.lim_am3
    lim_am4 = keps_params.lim_am4
    lim_am5 = keps_params.lim_am5
    lim_am6 = keps_params.lim_am6
    c_mu0 = keps_params.c_mu0
    rp, rm, rn = keps_params.gls_p, keps_params.gls_m, keps_params.gls_n
    e1 = 3 + rp/rn
    e2 = 1.5 + rm/rn
    e3 = -1/rn
    filter_cof = .5

    # minimum value of alpha_n to ensure that alpha_m is positive
    alpha_n_min = 0.5*(- (sf_d1 + sf_nb0) + jnp.sqrt((sf_d1 + sf_nb0)**2 - \
        4.0*sf_d0*(sf_d4 + sf_nb1))) / (sf_d4 + sf_nb1)

    # Galperin limitation : l <= l_li
    l_lim = keps_params.galp*jnp.sqrt(2.0*tke[1:-1] / \
        jnp.maximum(1e-14, bvf[1:-1]))

    # limitation on psi (use MAX because rn is negative)
    cff = c_mu0**rp * l_lim**rn * tke[1:-1]**rm
    eps = eps.at[1:-1].set(jnp.maximum(eps[1:-1], cff))
    epsilon = c_mu0**e1 * tke[1:-1]**e2 * eps[1:-1]**e3
    epsilon = jnp.maximum(epsilon, eps_min)

    # compute alpha_n and alpha_m
    cff = (tke[1:-1] / epsilon)**2
    am = add_boundaries(0., cff*shear2[1:-1], 0.)
    an = add_boundaries(0., cff*bvf[1:-1], 0.)
    alpha_n = an[1:-1] + filter_cof*(0.5*an[2:] - an[1:-1] + 0.5*an[:-2])
    alpha_m = am[1:-1] + filter_cof*(0.5*am[2:] - am[1:-1] + 0.5*am[:-2])

    # limitation of alpha_n and alpha_m
    alpha_n = jnp.minimum(jnp.maximum(0.73*alpha_n_min, alpha_n), 1e10)
    alpha_m_max = (lim_am0 + lim_am1*alpha_n + lim_am2*alpha_n**2 + \
        lim_am3*alpha_n**3) / (lim_am4 + lim_am5*alpha_n + \
        lim_am6*alpha_n**2)
    alpha_m = jnp.minimum(alpha_m, alpha_m_max)

    # compute stability functions
    denom = sf_d0 + sf_d1*alpha_n + sf_d2*alpha_m + sf_d3*alpha_n*alpha_m \
        + sf_d4*alpha_n**2 + sf_d5*alpha_m**2
    cff = 1./denom
    c_mu_in = cff*(sf_n0 + sf_n1*alpha_n + sf_n2*alpha_m)
    c_mu = add_boundaries(keps_params.c_mu_min, c_mu_in, keps_params.c_mu_min)
    c_mu_prim_in = cff*(sf_nb0 + sf_nb1*alpha_n + sf_nb2*alpha_m)
    c_mu_prim = add_boundaries(
        keps_params.c_mu_prim_min, c_mu_prim_in, keps_params.c_mu_prim_min
    )
    epsilon = c_mu0**e1 * tke[1:-1]**e2 * eps[1:-1]**e3
    epsilon = jnp.maximum(epsilon, eps_min)

    # finalize the computation of akv and akt
    cff = tke[1:-1]**2 / epsilon
    akt_in = jnp.maximum(cff*c_mu_prim[1:-1], akt_min)
    akv_in = jnp.maximum(cff*c_mu[1:-1], akv_min)

    akt_btm = jnp.maximum(1.5*akt_in[0] - 0.5*akt_in[1], akt_min)
    akt_sfc = jnp.maximum(1.5*akt_in[-1] - 0.5*akt_in[-2], akt_min)
    akv_btm = jnp.maximum(1.5*akv_in[0] - 0.5*akv_in[1], akv_min)
    akv_sfc = jnp.maximum(1.5*akv_in[-1] - 0.5*akv_in[-2], akv_min)

    akt = add_boundaries(akt_btm, akt_in, akt_sfc)
    akv = add_boundaries(akv_btm, akv_in, akv_sfc)

    return akt, akv, eps, c_mu, c_mu_prim
