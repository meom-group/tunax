r"""
:math:`k-\varepsilon` closure parameters, states and computation functions.

This module contains the implementation of the :math:`k-\varepsilon` model
described as a GLS case as in [1]_ as a :class:`~closure.Closure` instance. The
model was traduced from Frotran to JAX with the work of Florian Lemarié and
Manolis Perrot [2]_, the translation was done in part using the work of Anthony
Zhou, Linnia Hawkins and Pierre Gentine [3]_. The parameters of the closure are
available in the :class:`KepsParameters` class, the closure state in
:class:`KepsState` class. The function :attr:`keps_step` compute one time-step
of the closure, which means that it computes the eddy-diffusivity and
viscosity. The module contains other functions that are used by this main one.
These classes and the function step can be obtained by the prefix
:code:`tunax.closures.k_epsilon` or directly by :code:`tunax.closures`.

References
----------
.. [1] L. Umlauf and H. Burchard. A generic length-scale equation for
    geophysical turbulence models (2003). Journal of Marine Research 61
    pp. 235-265. doi : `10.1357/002224003322005087 <https://www.semanticscholar
    .org/paper/A-generic-length-scale-equation-for-geophysical-Umlauf-Burchard/
    24fd6403615fc7a6c5d9b6156e4f1e8d4d280af2>`_.
.. [2] M. Perrot and F. Lemarié. Energetically consistent Eddy-Diffusivity
    Mass-Flux convective schemes. Part I: Theory and Models (2024). url :
    `hal.science/hal-04439113 <https://hal.science/hal-04439113>`_.
.. [3] A. Zhou, L. Hawkins and P. Gentine. Proof-of-concept: Using ChatGPT to
    Translate and Modernize an Earth System Model from Fortran to Python/JAX
    (2024). url : `arxiv.org/abs/2405.00018
    <https://arxiv.org/abs/2405.00018>`_.
    
"""


from __future__ import annotations
from functools import partial
from typing import Tuple

import equinox as eqx
import jax.numpy as jnp
from jax import lax, jit
from jaxtyping import Float, Array

from tunax.case import Case
from tunax.space import Grid, State
from tunax.functions import tridiag_solve, add_boundaries
from tunax.closure import ClosureParametersAbstract, ClosureStateAbstract


class KepsParameters(ClosureParametersAbstract):
    r"""
    Parameters and constants for :math:`k-\varepsilon`.

    The first 17 attributes are the parameters of :math:`k-\varepsilon` that
    may be calibrated. This class also contains some physical constants used
    in the closure computing. The last 19 attributes are the one for the
    stability function that are computed from the parameters of
    :math:`k-\varepsilon`.

    Parameters
    ----------
    c1 : float, default=5.
        cf. attribute.
    c2 : float, default=0.8
        cf. attribute.
    c3 : float, default=1.968
        cf. attribute.
    c4 : float, default=1.136
        cf. attribute.
    c5 : float, default=0.
        cf. attribute.
    c6 : float, default=0.4
        cf. attribute.
    cb1 : float, default=5.95
        cf. attribute.
    cb2 : float, default=.6
        cf. attribute.
    cb3 : float, default=1.
        cf. attribute.
    cb4 : float, default=0.
        cf. attribute.
    cb5 : float, default=0.3333
        cf. attribute.
    cbb : float, default=.72
        cf. attribute.
    c_mu0 : float, default=0.5477
        cf. attribute.
    sig_k : float, default=1.
        cf. attribute.
    sig_eps : float, default=1.3
        cf. attribute.
    c_eps1 : float, default=1.44
        cf. attribute.
    c_eps2 : float, default=1.92
        cf. attribute.
    c_eps3m : float, default=-0.4
        cf. attribute.
    c_eps3p : float, default=1.
        cf. attribute.
    chk_grav : float, default=1400.
        cf. attribute.
    galp: float, default=0.53
        cf. attribute.
    z0s_min : float, default=1e-2
        cf. attribute.
    z0b_min : float, default=1e-4
        cf. attribute.
    z0b : float, default=1e-14
        cf. attribute.
    akt_min : float, default=1e-5
        cf. attribute.
    akt_min : float, default=1e-4
        cf. attribute.
    tke_min : float, default=1e-6
        cf. attribute.
    eps_min : float, default=1e-12
        cf. attribute.
    c_mu_min : float, default=0.1
        cf. attribute.
    c_mu_prim_min : float, default=0.1
        cf. attribute.
    dir_sfc: bool, default=False
        cf. attribute.
    dir_btm: bool, default=True
        cf. attribute.
    gls_p : float, default=3
        cf. attribute.
    gls_m : float, default=1.5
        cf. attribute.
    gls_n : float, default=-1
        cf. attribute.

    Attributes
    ----------
    c1 : float, default=5.
        :math:`k-\varepsilon` parameter :math:`c_1` for the dissipation of the
        corelation tensor pressure/velocity (Umlauf and Burchard notations)
        :math:`[\text{dimensionless}]`.
    c2 : float, default=0.8
        :math:`k-\varepsilon` parameter :math:`c_2` for the dissipation of the
        corelation tensor pressure/velocity (Umlauf and Burchard notations)
        :math:`[\text{dimensionless}]`.
    c3 : float, default=1.968
        :math:`k-\varepsilon` parameter :math:`c_3` for the dissipation of the
        corelation tensor pressure/velocity (Umlauf and Burchard notations)
        :math:`[\text{dimensionless}]`.
    c4 : float, default=1.136
        :math:`k-\varepsilon` parameter :math:`c_4` for the dissipation of the
        corelation tensor pressure/velocity (Umlauf and Burchard notations)
        :math:`[\text{dimensionless}]`.
    c5 : float, default=0.
        :math:`k-\varepsilon` parameter :math:`c_5` for the dissipation of the
        corelation tensor pressure/velocity (Umlauf and Burchard notations)
        :math:`[\text{dimensionless}]`.
    c6 : float, default=0.4
        :math:`k-\varepsilon` parameter :math:`c_6` for the dissipation of the
        corelation tensor pressure/velocity (Umlauf and Burchard notations)
        :math:`[\text{dimensionless}]`.
    cb1 : float, default=5.95
        :math:`k-\varepsilon` parameter :math:`c_{b1}` for the dissipation of
        the corelation tensor buoyancy/velocity (Umlauf and Burchard notations)
        :math:`[\text{dimensionless}]`.
    cb2 : float, default=.6
        :math:`k-\varepsilon` parameter :math:`c_{b2}` for the dissipation of
        the corelation tensor buoyancy/velocity (Umlauf and Burchard notations)
        :math:`[\text{dimensionless}]`.
    cb3 : float, default=1.
        :math:`k-\varepsilon` parameter :math:`c_{b3}` for the dissipation of
        the corelation tensor buoyancy/velocity (Umlauf and Burchard notations)
        :math:`[\text{dimensionless}]`.
    cb4 : float, default=0.
        :math:`k-\varepsilon` parameter :math:`c_{b4}` for the dissipation of
        the corelation tensor buoyancy/velocity (Umlauf and Burchard notations)
        :math:`[\text{dimensionless}]`.
    cb5 : float, default=0.3333
        :math:`k-\varepsilon` parameter :math:`c_{b5}` for the dissipation of
        the corelation tensor buoyancy/velocity (Umlauf and Burchard notations)
        :math:`[\text{dimensionless}]`.
    cbb : float, default=.72
        :math:`k-\varepsilon` parameter :math:`c_{b}` for the dissipation of
        the corelation tensor buoyancy/velocity (Umlauf and Burchard notations)
        :math:`[\text{dimensionless}]`.
    c_mu0 : float, default=0.5477
        :math:`k-\varepsilon` parameter :math:`c_\mu^0` which links the mixing
        length to the dissipation (Umlauf and Burchard notations)
        :math:`[\text{dimensionless}]`.
    sig_k : float, default=1.
        :math:`k-\varepsilon` parameter :math:`\sigma_k` Schmit number for the
        dissipation of TKE (Umlauf and Burchard notations)
        :math:`[\text{dimensionless}]`.
    sig_eps : float, default=1.3
        :math:`k-\varepsilon` parameter :math:`\sigma_\varepsilon` Schmit
        number for the dissipation of :math:`\varepsilon` (Umlauf and Burchard
        notations) :math:`[\text{dimensionless}]`.
    c_eps1 : float, default=1.44
        :math:`k-\varepsilon` parameter :math:`c_{\varepsilon 1}` correction of
        the :math:`\varepsilon` equation (Umlauf and Burchard notations)
        :math:`[\text{dimensionless}]`.
    c_eps2 : float, default=1.92
        :math:`k-\varepsilon` parameter :math:`c_{\varepsilon 2}` correction of
        the :math:`\varepsilon` equation (Umlauf and Burchard notations)
        :math:`[\text{dimensionless}]`.
    c_eps3m : float, default=-0.4
        :math:`k-\varepsilon` parameter :math:`c_{\varepsilon 3}^-` correction
        of the :math:`\varepsilon` equation (Umlauf and Burchard notations)
        :math:`[\text{dimensionless}]`.
    c_eps3p : float, default=1.
        :math:`k-\varepsilon` parameter :math:`c_{\varepsilon 3}^+` correction
        of the :math:`\varepsilon` equation (Umlauf and Burchard notations)
        :math:`[\text{dimensionless}]`.
    chk_grav : float, default=1400.
        Charnock coefficient times gravity :math:`[\text{dimensionless}]`.
    galp: float, default=0.53
        Parameter for Galperin mixing length limitation
        :math:`[\text{dimensionless}]`.
    z0s_min : float, default=1e-2
        Minimal surface roughness length :math:`[\text m]`.
    z0b_min : float, default=1e-4
        Minimal bottom roughness length :math:`[\text m]`.
    z0b : float, default=1e-14
        Bottom roughness length :math:`[\text m]`.
    akt_min : float, default=1e-5
        Minimal and initialization value of eddy-diffusivity
        :math:`\left[\text m^2 \cdot \text s^{-1} \right]`.
    akv_min : float, default=1e-4
        Minimal and initialization value of eddy-viscosity
        :math:`\left[\text m^2 \cdot \text s^{-1} \right]`.
    tke_min : float, default=1e-6
        Minimal and initialization value of turbulent kinetic energy (TKE)
        :math:`\left[\text m^3 \cdot \text s^{-2} \right]`.
    eps_min : float, default=1e-12
        Minimal and initialization value of TKE dissipation
        :math:`\left[\text m^2 \cdot \text s^{-3} \right]`.
    c_mu_min : float, default=0.1
        Minimal and initialization value of :math:`c_\mu` in GLS formalisim
        :math:`[\text{dimensionless}]`.
    c_mu_prim_min : float, default=0.1
        Minimal and initialization value of `c_\mu'` in GLS formalisim
        :math:`[\text{dimensionless}]`.
    dir_sfc: bool, default=False
        Apply a Dirichlet boundary condition at the surface for TKE, else
        apply a Neumann boundary condition.
    dir_btm: bool, default=True
        Apply a Dirichlet boundary condition at the bottom for TKE, else
        apply a Neumann boundary condition.
    gls_p : float, default=3
        GLS coefficient :math:`p` to define :math:`k-\varepsilon`
        :math:`[\text{dimensionless}]`.
    gls_m : float, default=1.5
        GLS coefficient :math:`m` to define :math:`k-\varepsilon`
        :math:`[\text{dimensionless}]`.
    gls_n : float, default=-1
        GLS coefficient :math:`n` to define :math:`k-\varepsilon`
        :math:`[\text{dimensionless}]`.
    sf_d0 : float (not a parameter, computed from the above attributes)
        Limitation coefficient for :math:`k-\varepsilon` computed from the
        parameters :math:`[\text{dimensionless}]`.
    sf_d1 : float (not a parameter, computed from the above attributes)
        Limitation coefficient for :math:`k-\varepsilon` computed from the
        parameters :math:`[\text{dimensionless}]`.
    sf_d2 : float (not a parameter, computed from the above attributes)
        Limitation coefficient for :math:`k-\varepsilon` computed from the
        parameters :math:`[\text{dimensionless}]`.
    sf_d3 : float (not a parameter, computed from the above attributes)
        Limitation coefficient for :math:`k-\varepsilon` computed from the
        parameters :math:`[\text{dimensionless}]`.
    sf_d4 : float (not a parameter, computed from the above attributes)
        Limitation coefficient for :math:`k-\varepsilon` computed from the
        parameters :math:`[\text{dimensionless}]`.
    sf_d5 : float (not a parameter, computed from the above attributes)
        Limitation coefficient for :math:`k-\varepsilon` computed from the
        parameters :math:`[\text{dimensionless}]`.
    sf_n0 : float (not a parameter, computed from the above attributes)
        Limitation coefficient for :math:`k-\varepsilon` computed from the
        parameters :math:`[\text{dimensionless}]`.
    sf_n1 : float (not a parameter, computed from the above attributes)
        Limitation coefficient for :math:`k-\varepsilon` computed from the
        parameters :math:`[\text{dimensionless}]`.
    sf_n2 : float (not a parameter, computed from the above attributes)
        Limitation coefficient for :math:`k-\varepsilon` computed from the
        parameters :math:`[\text{dimensionless}]`.
    sf_nb0 : float (not a parameter, computed from the above attributes)
        Limitation coefficient for :math:`k-\varepsilon` computed from the
        parameters :math:`[\text{dimensionless}]`.
    sf_nb1 : float (not a parameter, computed from the above attributes)
        Limitation coefficient for :math:`k-\varepsilon` computed from the
        parameters :math:`[\text{dimensionless}]`.
    sf_nb2 : float (not a parameter, computed from the above attributes)
        Limitation coefficient for :math:`k-\varepsilon` computed from the
        parameters :math:`[\text{dimensionless}]`.
    lim_am0 : float (not a parameter, computed from the above attributes)
        Limitation coefficient for :math:`k-\varepsilon` computed from the
        parameters :math:`[\text{dimensionless}]`.
    lim_am1 : float (not a parameter, computed from the above attributes)
        Limitation coefficient for :math:`k-\varepsilon` computed from the
        parameters :math:`[\text{dimensionless}]`.
    lim_am2 : float (not a parameter, computed from the above attributes)
        Limitation coefficient for :math:`k-\varepsilon` computed from the
        parameters :math:`[\text{dimensionless}]`.
    lim_am3 : float (not a parameter, computed from the above attributes)
        Limitation coefficient for :math:`k-\varepsilon` computed from the
        parameters :math:`[\text{dimensionless}]`.
    lim_am4 : float (not a parameter, computed from the above attributes)
        Limitation coefficient for :math:`k-\varepsilon` computed from the
        parameters :math:`[\text{dimensionless}]`.
    lim_am5 : float (not a parameter, computed from the above attributes)
        Limitation coefficient for :math:`k-\varepsilon` computed from the
        parameters :math:`[\text{dimensionless}]`.
    lim_am6 : float (not a parameter, computed from the above attributes)
        Limitation coefficient for :math:`k-\varepsilon` computed from the
        parameters :math:`[\text{dimensionless}]`.
    
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
        # stability function coefficients
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
    r"""
    Define the state of the water column for the :math:`k-\varepsilon` model.

    The first initilisation is done from the minimal values of the different
    variables given in an instance of :class:'KepsParameters`.

    Parameters
    ----------
    grid : Grid
        cf. attribute.
    keps_params : KepsParameters
        Used to define the initialization values of the variables.

    Attributes
    ----------
    grid : Grid
        Geometry of the water column, should be the same than for the
        :class:`~space.State` instance used in the model.
    akt : Float[~jax.Array, 'nz+1']
        Eddy-diffusivity on the interfaces of the cells
        :math:`\left[\text m ^2 \cdot \text s ^{-1}\right]`.
    akv : Float[~jax.Array, 'nz+1']
        Eddy-viscosity on the interfaces of the cells
        :math:`\left[\text m ^2 \cdot \text s ^{-1}\right]`.
    tke :  Float[~jax.Array, 'nz+1']
        Turbulent kinetic energy (TKE) denoted :math:`k` on the interfaces of
        the cells :math:`\left[\text m ^2 \cdot \text s ^{-2}\right]`.
    eps :  Float[~jax.Array, 'nz+1']
        TKE dissipation denoted :math:`\varepsilon` on the interfaces of the
        cells :math:`\left[\text m ^2 \cdot \text s ^{-3}\right]`.
    c_mu :  Float[~jax.Array, 'nz+1']
        :math:`c_\mu` in GLS formalisim on the interfaces of the cells
        :math:`[\text{dimensionless}]`.
    c_mu_prim :  Float[~jax.Array, 'nz+1']
        :math:`c_\mu'` in GLS formalisim on the interfaces of the cells
        :math:`[\text{dimensionless}]`.

    """

    grid: Grid
    akt: Float[Array, 'nz+1']
    akv: Float[Array, 'nz+1']
    tke: Float[Array, 'nz+1']
    eps: Float[Array, 'nz+1']
    c_mu: Float[Array, 'nz+1']
    c_mu_prim: Float[Array, 'nz+1']

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
    r"""
    Run one time-step of the :math:`k-\varepsilon` closure.

    The purpose of this function is to get the eddy-diffusivity and
    eddy-viscosity at the next time-step. It works in 3 steps

    1. The Brunt–Väisälä frequency and the shear is computed from the
       :code:`state` and the boundary conditions are computed.
    2. The equations on :math:`k` and :math:`\varepsilon` are solved and their
       values are computed for the next time step.
    3. The eddy-diffusivity and viscosity are computed as diagnostic variables
       and the :math:`keps_state` is updated.

    Parameters
    ----------
    state : State
        Current state of the water column.
    keps_state : KepsState
        Current state of the water column for the variables used by
        :math:`k-\varepsilon`.
    dt : float
        Time-step of the forward model :math:`[\text s]`.
    keps_params: KepsParameters
        Values of the parameters used by :math:`k-\varepsilon` (time-
        independant).
    case : Case
        Physical parameters and forcings of the model run.
    
    Returns
    -------
    keps_state : KepsState
            State of the water column for the variables used by
            :math:`k-\varepsilon` at the next time-step.

    Note
    ----
    This function is jitted with JAX, it should make it faster, but the
    :func:`~jax.jit` decorator can be removed.
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
        t: Float[Array, 'nz'],
        s: Float[Array, 'nz'],
        zr: Float[Array, 'nz'],
        case: Case
    ) -> Tuple[Float[Array, 'nz+1'], Float[Array, 'nz']]:
    r"""
    Compute density anomaly and Brunt–Väisälä frequency.
    
    Prognostic computation via linear Equation Of State (EOS) :

    :math:`\rho = \rho_0(1-\alpha (T-T_0) + \beta (S-S_0))`

    :math:`N^2 = - \dfrac g {\rho_0} \partial_z \rho`

    Parameters
    ----------
    t : Float[~jax.Array, 'nz']
        Temperature on the center of the cells :math:`[° \text C]`.
    s : Float[~jax.Array, 'nz']
        Salinity on the center of the cells :math:`[\text{psu}]`.
    zr : Float[~jax.Array, 'nz']
        Depths of cell centers from deepest to shallowest :math:`[\text m]`
    case : Case
        Physical parameters and forcings of the model run.

    Returns
    -------
    bvf : Float[Array, 'nz+1']
        Brunt–Väisälä frequency squared :math:`N^2` on cell interfaces
        :math:`\left[\text s^{-2}\right]`.
    rho : Float[Array, 'nz']
        Density anomaly :math:`\rho` on cell interfaces 
        :math:`\left[\text {kg} \cdot \text m^{-3}\right]`
    """
    rho0 = case.rho0
    rho = rho0 * (1. - case.alpha*(t-case.t_rho_ref) + \
                  case.beta*(s-case.s_rho_ref))
    cff = 1./(zr[1:]-zr[:-1])
    bvf_in = - cff*case.grav/rho0 * (rho[1:]-rho[:-1])
    bvf = add_boundaries(0., bvf_in, bvf_in[-1])

    return rho, bvf


def compute_shear(
        u_n: Float[Array, 'nz'],
        v_n: Float[Array, 'nz'],
        u_np1: Float[Array, 'nz'],
        v_np1: Float[Array, 'nz'],
        zr: Float[Array, 'nz']
    ) -> Float[Array, 'nz+1']:
    r"""
    Compute shear production term for TKE equation.

    The prognostic equations are

    :math:`S_h^2 = \partial_Z U^n \cdot \partial_z U^{n+1/2}`

    where :math:`U^{n+1/2}` is the mean between :math:`U^n` and
    :math:`U^{n+1}`.
    
    Parameters
    ----------
    u_n : Float[~jax.Array, 'nz']
        Current zonal velocity on the center of the cells
        :math:`\left[\text m \cdot \text s^{-1}\right]`.
    v_n : Float[~jax.Array, 'nz']
        Current meridional velocity on the center of the cells
        :math:`\left[\text m \cdot \text s^{-1}\right]`.
    u_np1 : Float[~jax.Array, 'nz']
        Zonal velocity on the center of the cells at the next time step
        :math:`\left[\text m \cdot \text s^{-1}\right]`.
    v_np1 : Float[~jax.Array, 'nz']
        Meridional velocity on the center of the cells at the next time step
        :math:`\left[\text m \cdot \text s^{-1}\right]`.
    zr : Float[~jax.Array, 'nz']
        Depths of cell centers from deepest to shallowest :math:`[\text m]`
        

    Returns
    -------
    shear2 : Float[~jax.Array, 'nz+1']
        Shear production squared :math:`S_h^2` on cell interfaces
        :math:`\left[\text m ^2 \cdot \text s ^{-3}\right]`.
    """
    cff = 1.0 / (zr[1:] - zr[:-1])**2
    du = 0.5*cff * (u_np1[1:]-u_np1[:-1]) * \
        (u_n[1:]+u_np1[1:]-u_n[:-1]-u_np1[:-1])
    dv = 0.5*cff * (v_np1[1:]-v_np1[:-1]) * \
        (v_n[1:]+v_np1[1:]-v_n[:-1]-v_np1[:-1])
    shear2_in = du + dv
    return add_boundaries(0., shear2_in, 0.)


def compute_tke_eps_bc(
        tke: Float[Array, 'nz+1'],
        hz: Float[Array, 'nz'],
        keps_params: KepsParameters,
        case: Case
    ) -> Tuple[float, float, float, float]:
    r"""
    Compute top and bottom boundary conditions for TKE and :math:`\varepsilon`.

    Parameters
    ----------
    tke : Float[~jax.Array, 'nz+1']
        Turbulent kinetic energy (TKE) denoted :math:`k` on the interfaces of
        the cells :math:`\left[\text m ^2 \cdot \text s ^{-2}\right]`.
    hz : Float[~jax.Array, 'nz']
        Thickness of cells from deepest to shallowest :math:`[\text m]`.
    keps_params : KepsParameters
        Values of the parameters used by :math:`k-\varepsilon`.
    case : Case
        Physical parameters and forcings of the model run.

    Returns
    -------
    tke_sfc_bc : float
        TKE value for surface boundary condition (Dirichlet
        :math:`\left[\text m ^2 \cdot \text s ^{-2}\right]` or Neumann
        :math:`\left[\text m ^3 \cdot \text s ^{-3}\right]`).
    tke_btm_bc : float
        TKE value for bottom boundary condition (Dirichlet
        :math:`\left[\text m ^2 \cdot \text s ^{-2}\right]` or Neumann
        :math:`\left[\text m ^3 \cdot \text s ^{-3}\right]`).
    eps_sfc_bc : float
        :math:`\varepsilon` value for surface boundary condition (Dirichlet
        :math:`\left[\text m ^2 \cdot \text s ^{-3}\right]` or Neumann
        :math:`\left[\text m ^3 \cdot \text s ^{-4}\right]`).
    eps_btm_bc : float
        :math:`\varepsilon` value for surface boundary condition (Dirichlet
        :math:`\left[\text m ^2 \cdot \text s ^{-3}\right]` or Neumann
        :math:`\left[\text m ^3 \cdot \text s ^{-4}\right]`).

    Note
    ----
    The kind of boundary conditions between Neumann and Dirichlet are register
    in the parameters :code:`keps_params`.
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
        akt: Float[Array, 'nz+1'],
        akv: Float[Array, 'nz+1'],
        tke: Float[Array, 'nz+1'],
        tke_np1: Float[Array, 'nz+1'],
        eps: Float[Array, 'nz+1'],
        c_mu: Float[Array, 'nz+1'],
        c_mu_prim: Float[Array, 'nz+1'],
        bvf: Float[Array, 'nz+1'],
        shear2: Float[Array, 'nz+1'],
        hz: Float[Array, 'nz'],
        dt: float,
        tke_sfc_bc: float,
        tke_btm_bc: float,
        eps_sfc_bc: float,
        eps_btm_bc: float,
        keps_params: KepsParameters,
        do_tke: bool
    ) -> Float[Array, 'nz+1']:
    r"""
    Integrate TKE or :math:`\varepsilon` quantities.

    First the shear and buoyancy production are computed, then they are used in
    the building of the tridiagonal problem, the boundary conditions are then
    added and finally the tridiagonal problem is solved.

    Parameters
    ----------
    akt : Float[~jax.Array, 'nz+1']
        Current eddy-diffusivity on the interfaces of the cells
        :math:`\left[\text m ^2 \cdot \text s ^{-1}\right]`.
    akv : Float[~jax.Array, 'nz+1']
        Current eddy-viscosity on the interfaces of the cells
        :math:`\left[\text m ^2 \cdot \text s ^{-1}\right]`.
    tke : Float[~jax.Array, 'nz+1']
        Current turbulent kinetic energy (TKE) denoted :math:`k` on the
        interfaces of the cells
        :math:`\left[\text m ^2 \cdot \text s ^{-2}\right]`.
    tke_np1 : Float[~jax.Array, 'nz+1']
        Turbulent kinetic energy (TKE) denoted :math:`k` on the interfaces of
        the cells at next step (usefull only for :math:`\varepsilon`
        integration) :math:`\left[\text m ^2 \cdot \text s ^{-2}\right]`.
    eps : Float[~jax.Array, 'nz+1']
        Current TKE dissipation denoted :math:`\varepsilon` on the interfaces
        of the cells :math:`\left[\text m ^2 \cdot \text s ^{-3}\right]`.
    c_mu : Float[~jax.Array, 'nz+1']
        Current :math:`c_\mu` in GLS formalisim on the interfaces of the cells
        :math:`[\text{dimensionless}]`.
    c_mu_prim : Float[~jax.Array, 'nz+1']
        Current :math:`c_\mu'` in GLS formalisim on the interfaces of the cells
        :math:`[\text{dimensionless}]`.
    bvf : float(nz+1)
        Current Brunt–Väisälä frequency squared :math:`N^2` on cell interfaces
        :math:`\left[\text s^{-2}\right]`.
    shear2 : Float[~jax.Array, 'nz+1']
        Current shear production squared :math:`S_h^2` on cell interfaces
        :math:`\left[\text m ^2 \cdot \text s ^{-3}\right]`.
    hz : Float[~jax.Array, 'nz']
        Thickness of cells from deepest to shallowest :math:`[\text m]`.
    dt : float
        Time-step of the forward model :math:`[\text s]`.
    tke_sfc_bc : float
        TKE value for surface boundary condition (Dirichlet
        :math:`\left[\text m ^2 \cdot \text s ^{-2}\right]` or Neumann
        :math:`\left[\text m ^3 \cdot \text s ^{-3}\right]`).
    tke_btm_bc : float
        TKE value for bottom boundary condition (Dirichlet
        :math:`\left[\text m ^2 \cdot \text s ^{-2}\right]` or Neumann
        :math:`\left[\text m ^3 \cdot \text s ^{-3}\right]`).
    eps_sfc_bc : float
        :math:`\varepsilon` value for surface boundary condition (Dirichlet
        :math:`\left[\text m ^2 \cdot \text s ^{-3}\right]` or Neumann
        :math:`\left[\text m ^3 \cdot \text s ^{-4}\right]`).
    eps_btm_bc : float
        :math:`\varepsilon` value for surface boundary condition (Dirichlet
        :math:`\left[\text m ^2 \cdot \text s ^{-3}\right]` or Neumann
        :math:`\left[\text m ^3 \cdot \text s ^{-4}\right]`).
    keps_params : KepsParameters
        Values of the parameters used by :math:`k-\varepsilon`.
    do_tke : bool
        If :code:`True` solve the equation for TKE, else for
        :math:`\varepsilon`.

    Returns
    -------
    vec : Float[~jax.Array, 'nz+1']
        TKE or :math:`\varepsilon` at next step (depending on :code:`do_tke`).
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
        tke: Float[Array, 'nz+1'],
        eps: Float[Array, 'nz+1'],
        bvf: Float[Array, 'nz+1'],
        shear2: Float[Array, 'nz+1'],
        keps_params: KepsParameters
    ) -> Tuple[Float[Array, 'nz+1'], Float[Array, 'nz+1'],
               Float[Array, 'nz+1'], Float[Array, 'nz+1'],
               Float[Array, 'nz+1']]:
    r"""
    Computes the diagnostic variables of :math:`k-\varepsilon` closure.

    This function first apply the Galperin limitation, then it computes
    :math:`c_\mu'` and :math:`c_\mu` with the stability function, and finally
    it computes the eddy-diffusivity and viscosity with these variables.

    Parameters
    ----------
    tke : Float[~jax.Array, 'nz+1']
        Turbulent kinetic energy (TKE) denoted :math:`k` on the interfaces of
        the cells at next step
        :math:`\left[\text m ^2 \cdot \text s ^{-2}\right]`.
    eps : Float[~jax.Array, 'nz+1']
        TKE dissipation denoted :math:`\varepsilon` on the interfaces of the
        cells at next step :math:`\left[\text m ^2 \cdot \text s ^{-3}\right]`.
    bvf : Float[~jax.Array, 'nz+1']
        Current Brunt–Väisälä frequency squared :math:`N^2` on cell interfaces
        :math:`\left[\text s^{-2}\right]`.
    shear2 : Float[~jax.Array, 'nz+1']
        Current shear production squared :math:`S_h^2` on cell interfaces
        :math:`\left[\text m ^2 \cdot \text s ^{-3}\right]`.
    keps_params : KepsParameters
        Values of the parameters used by :math:`k-\varepsilon`.

    Returns
    -------
    akt : Float[Array, 'nz+1']
        Eddy-diffusivity on the interfaces of the cells at next step
        :math:`\left[\text m ^2 \cdot \text s ^{-1}\right]`.
    akv : Float[Array, 'nz+1']
        Eddy-viscosity on the interfaces of the cells at next step
        :math:`\left[\text m ^2 \cdot \text s ^{-1}\right]`.
    eps : Float[Array, 'nz+1']
        TKE dissipation denoted :math:`\varepsilon` on the interfaces
        of the cells at next step
        :math:`\left[\text m ^2 \cdot \text s ^{-3}\right]`.
    c_mu : Float[Array, 'nz+1']
        :math:`c_\mu` in GLS formalisim on the interfaces of the cells at next
        step :math:`[\text{dimensionless}]`.
    c_mu_prim : Float[Array, 'nz+1']
        :math:`c_\mu'` in GLS formalisim on the interfaces of the cells at next
        step :math:`[\text{dimensionless}]`.
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

    # minimum value of alpha_n to ensure that alpha_m is positive
    alpha_n_min = 0.5*(- (sf_d1 + sf_nb0) + jnp.sqrt((sf_d1 + sf_nb0)**2 - \
        4.0*sf_d0*(sf_d4 + sf_nb1))) / (sf_d4 + sf_nb1)

    # Galperin limitation : l <= l_li
    l_lim = keps_params.galp*jnp.sqrt(2.0*tke[1:-1] / \
        jnp.maximum(1e-14, bvf[1:-1]))

    # limitation (use MAX because rn is negative)
    cff = c_mu0**rp * l_lim**rn * tke[1:-1]**rm
    eps = eps.at[1:-1].set(jnp.maximum(eps[1:-1], cff))
    epsilon = c_mu0**e1 * tke[1:-1]**e2 * eps[1:-1]**e3
    epsilon = jnp.maximum(epsilon, eps_min)

    # compute alpha_n and alpha_m
    cff = (tke[1:-1] / epsilon)**2
    alpha_m = cff*shear2[1:-1]
    alpha_n = cff*bvf[1:-1]

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
