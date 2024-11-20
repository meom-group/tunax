"""
Geometry and variables of the model.

This module contains the objects that are used in Tunax to describe the
geometry of the water column in :class:`Grid`, the variables of the water
column at one time-step in :class:`State` and the time-series of the model
computation in :class:`Trajectories`. These classes can be obtained by the
prefix :code:`tunax.space.` or directly by :code:`tunax.`.

"""

from __future__ import annotations
from typing import Optional, Tuple, List

import equinox as eqx
import xarray as xr
import jax.numpy as jnp
from jax import vmap
from jaxtyping import Float, Array

from tunax.case import Case
from tunax.functions import add_boundaries


TRACERS_NAMES = ['t', 's', 'b', 'pt']


def _piecewise_linear_ramp(z: float, z0: float, f0: float)-> float:
    r"""
    Mathemacial function used for the state initialisation.

    Apply to z a function linear by part and continuous :
    f(z) = 0 if z < zm and f(z) = f0 (1-z/zm) else.

    Parameters
    ----------
    z : float
        The value where to apply the function.
    zm : float
        The point of connexion of the two linear parts of the function.
    f0 : float
        The value of the function in 0.

    Returns
    -------
    fz : float
        The value of the function in z.
    """
    return f0*(z/-z0+1) * (z>z0)

def _piecewise_linear_flat(z: float, zm: float, f0: float, sl: float) -> float:
    r"""
    Mathemacial function used for the state initialisation.

    Apply to z a function linear by part and continuous :
    f(z) = f0 + s_l (z-zm) if z < z_m and f(z) = f0 else.

    Parameters
    ----------
    z : float
        The value where to apply the function.
    zm : float
        The point of connexion of the two linear parts of the function.
    f0 : float
        The value of the function in 0 and in the right part of the
        funcion.
    sl : float
        The slope of the left part of the function.

    Returns
    -------
    fz : float
        The value of the function in z.
    """
    return f0 + sl*(z-zm) * (z<zm)


class Grid(eqx.Module):
    r"""
    One dimensional spatial geometry of a water column.

    This mesh is made up of a number of :attr:`nz` of cells (:attr:`zr`) of
    potentially varying thickness (:attr:`hz`), separated by interface points
    (:attr:`zw`) and extending from the ocean surface at a depth of :math:`0`
    to the ocean floor at a depth of :attr:`hbot`.

    Parameters
    ----------
    zr : Float[~jax.Array, 'nz']
        cf. attribute.
    zw : Float[~jax.Array, 'nz+1']
        cf. attribute.

    Attributes
    ----------
    nz : int
        Number of cells.
    hbot : float
        Depth of the water column :math:`[\text m]`.
    zr : Float[~jax.Array, 'nz']
        Depths of cell centers from deepest to shallowest :math:`[\text m]`
    zw : Float[~jax.Array, 'nz+1']
        Depths of cell interfaces from deepest to shallowest :math:`[\text m]`.
    hz : Float[~jax.Array, 'nz']
        Thickness of cells from deepest to shallowest :math:`[\text m]`.

    Note
    ----
    The constructor :code:`__init__` takes only :attr:`zr` and :attr:`zw` as
    as arguments and construct the other attributes from them. The centers of
    the cells :attr:`zr` are not necessarly the middle between the interfaces
    :attr:`zw`.

    """

    nz: int
    hbot: float
    zr: Float[Array, 'nz']
    zw: Float[Array, 'nz+1']
    hz: Float[Array, 'nz']

    def __init__(self, zr: Float[Array, 'nz'], zw: Float[Array, 'nz+1']):
        self.nz = zr.shape[0]
        self.hbot = zw[0]
        self.zw = zw
        self.zr = zr
        self.hz = zw[1:] - zw[:-1]

    def find_index(self, h: float) -> int:
        r"""
        Find the index of a depth.

        Find the index :code:`i` so that the depth :code:`h` is in cell
        :code:`i`, which means :math:`z^w_i \leqslant -h \leqslant z^w_{i+1}`
        if :math:`h \leqslant 0` and :math:`i=-1` if :math:`h>0`.

        Parameters
        ----------
        h : float, positive
            The depth to search the index :math:`[\text m]`.

        Returns
        -------
        i : int
            The index corresponding to the depth :code:`h`.
        """
        return int(jnp.searchsorted(self.zw, -h, side='right')) - 1

    @classmethod
    def linear(cls, nz: int, hbot: int) -> Grid:
        r"""
        Creates a grid with equal thickness cells.

        The grid instance will have :attr:`nz` cells of equal thickness for a
        depth of :attr:`hbot`.

        Parameters
        ----------
        nz : int
            Number of cells.
        hbot : float, positive
            Depth of the water column :math:`[\text m]`.
        """
        zw = jnp.linspace(-hbot, 0, nz+1)
        zr = 0.5*(zw[:-1]+zw[1:])
        return cls(zr, zw)

    @classmethod
    def analytic(
            cls,
            nz: int,
            hbot: float,
            hc: float,
            theta: float=6.5
        ) -> Grid:
        r"""
        Creates a grid of type analytic.

        The grid instance will have a depth of :attr:`hbot` and :attr:`nz`
        cells of thickness almost equals above :code:`hc` and wider under, the
        strecht parameter being defined by :code:`theta`.

        Parameters
        ----------
        nz : int
            Number of cells.
        hbot : float, positive
            Depth of the water column :math:`[\text m]`.
        hc : float, positive
            Reference depth :math:`[\text m]`.
        theta : float, default=6.5
            Stretching parameter toward the surface
            :math:`[\text{dimensionless}]`.
        """
        sc_w = jnp.linspace(-1, 0, nz+1)
        sc_r = (sc_w[:-1] + sc_w[1:])/2
        cs_r = (1-jnp.cosh(theta*sc_r))/(jnp.cosh(theta)-1)
        cs_w = (1-jnp.cosh(theta*sc_w))/(jnp.cosh(theta)-1)
        zw = (hc*sc_w + hbot*cs_w) * hbot/(hbot+hc)
        zr = (hc*sc_r + hbot*cs_r) * hbot/(hbot+hc)
        return cls(zr, zw)

    @classmethod
    def orca75(cls, hbot: float) -> Grid:
        r"""
        Creates the ORCA 75 grid from NEMO.

        The whole grid is created then levels between the depth :attr:`hbot`
        and :math:`0` are extracted.

        Parameters
        ----------
        hbot : float, positive
            Depth of the water column :math:`[\text m]`.
        """
        nz_orca = 75
        zsur = -3958.95137127683
        za2 = 100.7609285
        za0 = 103.9530096
        za1 = 2.415951269
        zkth = 15.3510137
        zkth2 = 48.02989372
        zacr = 7.
        zacr2 = 13.
        sc_r = jnp.arange(nz_orca-0.5, 0.5, -1)
        sc_w = jnp.arange(nz_orca, 0, -1)
        zw_orca = -(zsur + za0*sc_w + za1*zacr*jnp.log(jnp.cosh((sc_w-zkth)/\
            zacr)) + za2*zacr2*jnp.log(jnp.cosh((sc_w-zkth2)/zacr2)))
        zr_orca = -(zsur + za0*sc_r + za1*zacr*jnp.log(jnp.cosh((sc_r-zkth)/\
            zacr)) + za2*zacr2*jnp.log(jnp.cosh((sc_r-zkth2)/zacr2)))
        ibot = jnp.argmin(zw_orca <= -hbot)
        if ibot == 0:
            ibot = 1
        zw_orca = zw_orca.at[-1].set(0.)
        zw = zw_orca[ibot-1:]
        zr = zr_orca[ibot-1:]
        return cls(zr, zw)

    @classmethod
    def load(cls, ds: xr.Dataset) -> Grid:
        """
        Creates the grid defined by a dataset :code:`ds` of an observation.

        The dataset must be formated to have the variables corresponding to
        :attr:`zr` and :attr:`zw`.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset from which to extract the grid.
        """
        zw = jnp.array(ds['zw'], dtype=jnp.float32)
        zr = jnp.array(ds['zr'], dtype=jnp.float32)
        return cls(zr, zw)


class State(eqx.Module):
    r"""
    Water column state at one time-step.

    This state is defined on a :attr:`grid` describing the geometry, and is
    composed of the variables of the water column : the values of the momentum
    and the tracers on this :attr:`grid`. The call of the constructor build a
    state on the :attr:`grid` with all the variables set to :math:`0`.

    Parameters
    ----------
    grid : Grid
        cf. attribute.

    Attributes
    ----------
    grid : Grid
        Geometry of the water column.
    u : Float[~jax.Array, 'nz']
        Zonal velocity on the center of the cells :math:`\left[\text m \cdot
        \text s^{-1}\right]`.
    v : Float[~jax.Array, 'nz']
        Meridional velocity on the center of the cells :math:`\left[\text m
        \cdot \text s^{-1}\right]`.
    t : Float[~jax.Array, 'nz']
        Temperature on the center of the cells :math:`[° \text C]`.
    s : Float[~jax.Array, 'nz']
        Salinity on the center of the cells :math:`[\text{psu}]`.

    """

    grid: Grid
    u: Float[Array, 'nz']
    v: Float[Array, 'nz']
    t: Optional[Float[Array, 'nz']] = None
    s: Optional[Float[Array, 'nz']] = None
    b: Optional[Float[Array, 'nz']] = None
    pt: Optional[Float[Array, 'nz']] = None

    @classmethod
    def zeros(cls, grid: Grid, tracers: List[str]) -> State:
        """
        Initialize an instance with all variables equals to zero from a grid.

        Parameters
        ----------
        grid : Grid
            Geometry of the water column.

        Returns
        -------
        state : State
            An instance defined on the grid with all variables set to 0.        
        """
        zero_array = jnp.zeros(grid.nz)
        tracers_dict = {}
        for tracer_name in tracers:
            tracers_dict[tracer_name] = zero_array
        return State(grid, u=zero_array, v=zero_array, **tracers_dict)

    def init_u(self, hmxl: float=20., u_sfc: float=0.) -> State:
        r"""
        Initialize zonal velocity with a classical wind stratification.

        Return a State object where :attr:`u` is continuous and linear by part
        :math:`u(z) = \begin{cases}
        0 & \text{if } z < h_{\text{mxl}}\\
        u_{\text{sfc}} \left( 1 - \dfrac z {h_{\text{mxl}}}\right) &
        \text{else} \end{cases}`

        Parameters
        ----------
        hmxl : float, default=20.
            Mixed layer depth :math:`[\text m]`.
        u_sfc : float, default=0.
            Surface zonal velocity :math:`\left[\text m \cdot \text
            s^{-1}\right]`.

        Returns
        -------
        state : State
            The :code:`self` object with the the new value of zonal velocity.
        """
        maped_fun = vmap(_piecewise_linear_ramp, in_axes=(0, None, None))
        u_new = maped_fun(self.grid.zr, -hmxl, u_sfc)
        return eqx.tree_at(lambda t: t.u, self, u_new)

    def init_v(self, hmxl: float=20., v_sfc: float=0.) -> State:
        r"""
        Initialize meridional velocity with a classical wind stratification.

        Return a State object where :attr:`v` is continuous and linear by part
        :math:`v(z) = \begin{cases}
        0 & \text{if } z < h_{\text{mxl}}\\
        v_{\text{sfc}} \left( 1 - \dfrac z {h_{\text{mxl}}}\right) &
        \text{else} \end{cases}`

        Parameters
        ----------
        hmxl : float, default=20.
            Mixed layer depth :math:`[\text m]`.
        u_sfc : float, default=0.
            Surface meridional velocity :math:`\left[\text m \cdot \text
            s^{-1}\right]`.

        Returns
        -------
        state : State
            The :code:`self` object with the the new value of meridional
            velocity.
        """
        maped_fun = vmap(_piecewise_linear_ramp, in_axes=(0, None, None))
        v_new = maped_fun(self.grid.zr, -hmxl, v_sfc)
        return eqx.tree_at(lambda t: t.v, self, v_new)

    def init_t(
            self,
            hmxl: float=20.,
            t_sfc: float=21.,
            strat_t: float=5.1e-2
        ) -> State:
        r"""
        Initialize temperature with a classical tracer stratification.

        Return a State object where :attr:`t` is linear by part and continous
        :math:`T(z) = \begin{cases}
        t_{\text{sfc}} + S_T(z-h_{\text{mxl}}) & \text{if } z <
        h_{\text{mxl}}\\
        t_{\text{sfc}} & \text{else}
        \end{cases}`

        Parameters
        ----------
        hmxl : float, default=20.
            Mixed layer depth :math:`[\text m]`.
        t_sfc : float, default=21.
            Surface temperature :math:`[° \text C]`.
        strat_t : float, default=5.1e-2
            Thermal stratification above the mixed layer noted by :math:`S_T`
            :math:`[\text K \cdot \text m ^{-1}]`.

        Returns
        -------
        state : State
            The :code:`self` object with the the new value of temperature.
        """
        maped_fun = vmap(_piecewise_linear_flat, in_axes=(0, None, None, None))
        t_new = maped_fun(self.grid.zr, -hmxl, t_sfc, strat_t)
        return eqx.tree_at(lambda tree: tree.t, self, t_new)

    def init_s(
            self,
            hmxl: float=20.,
            s_sfc: float=35.,
            strat_s: float=1.3e-2
        ) -> State:
        r"""
        Initialize salinity with a classical tracer stratification.

        Return a State object where :attr:`s` is linear by part and continous
        :math:`S(z) = \begin{cases}
        s_{\text{sfc}} + S_S(z-h_{\text{mxl}}) & \text{if } z <
        h_{\text{mxl}}\\
        s_{\text{sfc}} & \text{else}
        \end{cases}`

        Parameters
        ----------
        hmxl : float, default=20.
            Mixed layer depth :math:`[\text m]`.
        s_sfc : float, default=21.
            Surface salinity :math:`[\text{psu}]`.
        strat_t : float, default=5.1e-2
            Salinity stratification above the mixed layer noted by :math:`S_T`
            :math:`[\text{psu} \cdot \text m ^{-1}]`.

        Returns
        -------
        state : State
            The :code:`self` object with the the new value of temperature.
        """
        maped_fun = vmap(_piecewise_linear_flat, in_axes=(0, None, None, None))
        s_new = maped_fun(self.grid.zr, -hmxl, s_sfc, strat_s)
        return eqx.tree_at(lambda t: t.s, self, s_new)

    def compute_eos(
            self,
            case: Case
        ) -> Tuple[Float[Array, 'nz+1'], Float[Array, 'nz']]:
        r"""
        Compute density anomaly and Brunt–Väisälä frequency.
        
        Prognostic computation via linear Equation Of State (EOS) :

        :math:`\rho = \rho_0(1-\alpha (T-T_0) + \beta (S-S_0))`

        :math:`N^2 = - \dfrac g {\rho_0} \partial_z \rho`

        Parameters
        ----------
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
        match case.eos_tracers:
            case 't':
                rho = rho0 * (1. - case.alpha*(self.t-case.t_rho_ref))
            case 's':
                rho = rho0 * (1. + case.beta*(self.s-case.s_rho_ref))
            case 'ts':
                rho = rho0 * (1. - case.alpha*(self.t-case.t_rho_ref) + \
                    case.beta*(self.s-case.s_rho_ref))
            case 'b':
                rho = rho0*(self.b+1)
        cff = 1./(self.grid.zr[1:]-self.grid.zr[:-1])
        bvf_in = - cff*case.grav/rho0 * (rho[1:]-rho[:-1])
        bvf = add_boundaries(0., bvf_in, bvf_in[-1])
        return rho, bvf

    def compute_shear(
            self,
            u_np1: Float[Array, 'nz'],
            v_np1: Float[Array, 'nz']
        ) -> Float[Array, 'nz+1']:
        r"""
        Compute shear production term for TKE equation.

        The prognostic equations are

        :math:`S_h^2 = \partial_Z U^n \cdot \partial_z U^{n+1/2}`

        where :math:`U^{n+1/2}` is the mean between :math:`U^n` and
        :math:`U^{n+1}`.
        
        Parameters
        ----------
        u_np1 : Float[~jax.Array, 'nz']
            Zonal velocity on the center of the cells at the next time step
            :math:`\left[\text m \cdot \text s^{-1}\right]`.
        v_np1 : Float[~jax.Array, 'nz']
            Meridional velocity on the center of the cells at the next time step
            :math:`\left[\text m \cdot \text s^{-1}\right]`.            

        Returns
        -------
        shear2 : Float[~jax.Array, 'nz+1']
            Shear production squared :math:`S_h^2` on cell interfaces
            :math:`\left[\text m ^2 \cdot \text s ^{-3}\right]`.
        """
        u_n = self.u
        v_n = self.v
        cff = 1.0 / (self.grid.zr[1:] - self.grid.zr[:-1])**2
        du = 0.5*cff * (u_np1[1:]-u_np1[:-1]) * \
            (u_n[1:]+u_np1[1:]-u_n[:-1]-u_np1[:-1])
        dv = 0.5*cff * (v_np1[1:]-v_np1[:-1]) * \
            (v_n[1:]+v_np1[1:]-v_n[:-1]-v_np1[:-1])
        shear2_in = du + dv
        return add_boundaries(0., shear2_in, 0.)


class Trajectory(eqx.Module):
    r"""
    Define the history of a simulation or an observation.

    Contains the timeseries of the momentum and the tracers throught the space
    of the :attr:`grid` and the :attr:`time`.

    Parameters
    ----------
    grid : Grid
        cf. attribute.
    time : Float[~jax.Array, 'nt']
        cf. attribute.
    u : Float[~jax.Array, 'nt nz']
        cf. attribute.
    v : Float[~jax.Array, 'nt nz']
        cf. attribute.
    t : Float[~jax.Array, 'nt nz']
        cf. attribute.
    s : Float[~jax.Array, 'nt nz']
        cf. attribute.

    Attributes
    ----------
    grid : Grid
        Geometry of the water column.
    time : Float[~jax.Array, 'nt']
        Time at each steps of observation from the begining of the simulation
        :math:`[\text s]`.
    u : Float[~jax.Array, 'nt nz']
        Time-serie of zonal velocity :math:`\left[\text m \cdot \text
        s^{-1}\right]`.
    v : Float[~jax.Array, 'nt nz']
        Time-serie of meridional velocity :math:`\left[\text m \cdot \text
        s^{-1}\right]`.
    t : Float[~jax.Array, 'nt nz']
        Time-serie of temperature :math:`[\text C°]`.
    s : Float[~jax.Array, 'nt nz']
        Time-serie of salinity :math:`[\text{psu}]`.

    """

    grid: Grid
    time: Float[Array, 'nt']
    u: Float[Array, 'nt nz']
    v: Float[Array, 'nt nz']
    t: Optional[Float[Array, 'nt nz']] = None
    s: Optional[Float[Array, 'nt nz']] = None
    b: Optional[Float[Array, 'nt nz']] = None
    pt: Optional[Float[Array, 'nt nz']] = None

    def to_ds(self) -> xr.Dataset:
        """
        Exports the trajectory in an xarray.Dataset.

        The dimensions of the dataset are :attr:`time`, :code:`grid.zr` and
        :code:`grid.zw`, the variables are :attr:`u`, :attr:`v`, :attr:`t` and
        :attr:`s`, all defined on the dimensions (:attr:`time`, :code:`zr`).

        Returns
        -------
        ds : xarray.Dataset
            Dataset of the trajectory.
        """
        variables = {'u': (('time', 'zr'), self.u),
                     'v': (('time', 'zr'), self.v)}
        for tracer_name in TRACERS_NAMES:
            tracer = getattr(self, tracer_name)
            if tracer is not None:
                variables[tracer_name] = (('time', 'zr'), tracer)
        coords = {'time': self.time,
                  'zr': self.grid.zr,
                  'zw': self.grid.zw}
        return xr.Dataset(variables, coords)

    def extract_state(self, i_time: int) -> State:
        """
        Extracts the water column state at one time index.

        Parameters
        ----------
        i_time : int
            The time index of the moment to extract the state.

        Returns
        -------
        state : State
            The state of the trajectory at the time of index :code:`i_time`.
        """
        tracers_dict = {}
        for tracer_name in TRACERS_NAMES:
            tracer = getattr(self, tracer_name)
            if tracer is not None:
                tracers_dict[tracer_name] = tracer[i_time, :]
        return State(
            self.grid, self.u[i_time, :], self.v[i_time, :], **tracers_dict)
