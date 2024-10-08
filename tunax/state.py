"""
Geometry and variables of the model.

This module contains the objects that are used in Tunax to describe the
geometry of the water column in `Grid`, the variables of the water column at
one time-step in `State` and the time-series of the model computation in
`Trajectories`.

Classes
-------
Grid
    spatial geometry of a water column
State
    define the state at one time-step on one grid
Trajectories
    define the history of a simulation

Functions
---------
piecewise_linear_ramp
    mathemacial function used for the state initialisation
piecewise_linear_flat
    mathemacial function used for the state initialisation

"""


from __future__ import annotations

import equinox as eqx
import xarray as xr
import jax.numpy as jnp
from jax import vmap


class Grid(eqx.Module):
    """
    Spatial geometry of a water column.

    Parameters
    ----------
    zr : jnp.ndarray, float(nz)
        depths of cell centers from deepest to shallowest [m]
    zw : jnp.ndarray, float(nz+1)
        depths of cell interfaces from deepest to shallowest [m]

    Attributes
    ----------
    nz : int
        number of cells
    h : float
        depth of the column
    zr : jnp.ndarray, float(nz)
        depths of cell centers from deepest to shallowest [m]
    zw : jnp.ndarray, float(nz+1)
        depths of cell interfaces from deepest to shallowest [m]
    hz : jnp.ndarray, float(nz)
        thickness of cells from deepest to shallowest [m]

    Class methods
    -------------
    linear
        creates a grid with equal thickness cells
    analytic
        creates a grid of type analytic where the steps are almost constants
        above hc and wider under
    orca75
        reates the ORCA 75 levels grid and extracts the levels between -h and 0
    load
        creates the the grid defined by a dataset of an observation
        
    """

    nz: int
    h: float
    zr: jnp.ndarray
    zw: jnp.ndarray
    hz: jnp.ndarray

    def __init__(self, zr:jnp.ndarray, zw: jnp.ndarray):
        self.nz = zr.shape[0]
        self.h = zw[0]
        self.zw = zw
        self.zr = zr
        self.hz = zw[1:] - zw[:-1]

    def find_index(self, h: float) -> int:
        """
        Find the index i so that hmxl is in cell i, which means that
        zw[i] <= -hxml < zw[i+1], and -1 if -hmxl < zw[0].

        Parameters
        ----------
        h : float, positive
            the depth to search the index
        
        Returns
        -------
        i : int
            the index corresponding to the depth h
        """
        return int(jnp.searchsorted(self.zw, -h, side='right')) - 1

    @classmethod
    def linear(cls, nz: int, h: int) -> Grid:
        """
        Creates a grid with equal thickness cells.

        Parameters
        ----------
        nz : int
            number of cells
        h : float
            maximum depth (positive) [m]
        
        Returns
        -------
        grid : Grid

        """
        zw = jnp.linspace(-h, 0, nz+1)
        zr = 0.5*(zw[:-1]+zw[1:])
        return cls(zr, zw)

    @classmethod
    def analytic(cls, nz: int, h: float, hc: float, theta: float=6.5) -> Grid:
        """
        Creates a grid of type analytic where the steps are almost constants
        above hc and wider under.

        Parameters
        ----------
        nz : int
            number of cells
        h : float
            maximum depth (positive) [m]
        hc : float
            reference depth (positive) [m]
        theta : float, default=6.5
            stretching parameter toward the surface

        Returns
        -------
        grid : Grid

        """
        sc_w = jnp.linspace(-1, 0, nz+1)
        sc_r = (sc_w[:-1] + sc_w[1:])/2
        cs_r = (1-jnp.cosh(theta*sc_r))/(jnp.cosh(theta)-1)
        cs_w = (1-jnp.cosh(theta*sc_w))/(jnp.cosh(theta)-1)
        zw = (hc*sc_w + h*cs_w) * h/(h+hc)
        zr = (hc*sc_r + h*cs_r) * h/(h+hc)
        return cls(zr, zw)

    @classmethod
    def orca75(cls, h: float) -> Grid:
        """
        Creates the ORCA 75 levels grid and extracts the levels between -h and
        0.

        Parameters
        ----------
        h : float
            maximum depth (positive) [m]

        Returns
        -------
        grid : Grid

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
        ibot = jnp.argmin(zw_orca <= -h)
        if ibot == 0:
            ibot = 1
        zw_orca = zw_orca.at[-1].set(0.)
        zw = zw_orca[ibot-1:]
        zr = zr_orca[ibot-1:]
        return cls(zr, zw)

    @classmethod
    def load(cls, ds: xr.Dataset) -> Grid:
        """
        Creates the the grid defined by a dataset of an observation.

        Parameters
        ----------
        ds : xr.Dataset
            observation dataset

        Returns
        -------
        grid : Grid

        """
        zw = jnp.array(ds['zw'], dtype=jnp.float32)
        zr = jnp.array(ds['zr'], dtype=jnp.float32)
        return cls(zr, zw)


def piecewise_linear_ramp(z: float, z0: float, f0: float)-> float:
    """
    Mathemacial function used for the state initialisation

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
    Define the state at one time-step on one grid.

    Parameters
    ----------
    grid : Grid
        spatial grid
    u : jnp.ndarray, float(nz)
        zonal velocity [m.s-1]
    v : jnp.ndarray, float(nz)
        meridional velocity [m.s-1]
    t : jnp.ndarray, float(nz)
        temperature [C]
    s : jnp.ndarray, float(nz)
        salinity [psu]

    Attributes
    ----------
    grid : Grid
        spatial grid
    u : jnp.ndarray, float(nz)
        zonal velocity [m.s-1]
    v : jnp.ndarray, float(nz)
        meridional velocity [m.s-1]
    t : jnp.ndarray, float(nz)
        temperature [C]
    s : jnp.ndarray, float(nz)
        salinity [psu]

    Methods
    -------
    init_u
        initialize zonal velocity with a classical wind stratification
    init_v
        initialize meridional velocity with a classical wind stratification
    init_t
        initialize temperature with a classical stratification
    init_s
        initialize salinity with a classical stratification

    """

    grid: Grid
    t: jnp.ndarray
    s: jnp.ndarray
    u: jnp.ndarray
    v: jnp.ndarray

    def init_u(self, hmxl: float=20., u_sfc: float=0.) -> State:
        """
        Initialize zonal velocity with a classical wind stratification.

        Return a State object where u is continuous and linear by part :
        constant equals to 0 under -`hmxl`, and linear above -`hmxl` with the
        value `u_sfc` at 0.

        Parameters
        ----------
        hmxl : float, default=20.
            mixed layer depth [m]
        u_sfc : float, default=0.
            surface zonal velocity [m.s-1]
        
        Returns
        -------
        state : State
            the `self` object with the the new initialization of zonal velocity
        """
        maped_fun = vmap(piecewise_linear_ramp, in_axes=(0, None, None))
        u_new = maped_fun(self.grid.zr, -hmxl, u_sfc)
        return eqx.tree_at(lambda t: t.u, self, u_new)

    def init_v(self, hmxl: float=20., v_sfc: float=0.) -> State:
        """
        Initialize meridional velocity with a classical wind stratification.

        Return a State object where u is continuous and linear by part :
        constant equals to 0 under -`hmxl`, and linear above -`hmxl` with the
        value `v_sfc` at 0.

        Parameters
        ----------
        hmxl : float, default=20.
            mixed layer depth [m]
        v_sfc : float, default=0.
            surface meridional velocity [m.s-1]
        
        Returns
        -------
        state : State
            the `self` object with the the new initialization of meridional
            velocity
        """
        maped_fun = vmap(piecewise_linear_ramp, in_axes=(0, None, None))
        v_new = maped_fun(self.grid.zr, -hmxl, v_sfc)
        return eqx.tree_at(lambda t: t.v, self, v_new)

    def init_t(
            self,
            hmxl: float=20.,
            t_sfc: float=21.,
            strat_t: float=5.1e-2
        ) -> State:
        """
        Initialize temperature with a classical stratification.

        Return a State object where t is linear by part and continous : linear
        under -`hmxl` with a slope of `strat_t`, and constant equals to `t_sfc`
        above -`hmxl`.

        Parameters
        ----------
        hmxl : float, default=20.
            mixed layer depth [m]
        t_sfc : float, default=21.
            surface temperature [C°]
        strat_t : float, default=5.1e-2
            thermal stratification above the mixed layer [K.m-1]

        Returns
        -------
        state : State
            the `self` object with the the new initialization of temperature
        """
        maped_fun = vmap(piecewise_linear_flat, in_axes=(0, None, None, None))
        t_new = maped_fun(self.grid.zr, -hmxl, t_sfc, strat_t)
        return eqx.tree_at(lambda tree: tree.t, self, t_new)

    def init_s(
            self,
            hmxl: float=20.,
            s_sfc: float=35.,
            strat_s: float=1.3e-2
        ) -> State:
        """
        Initialize salinity with a classical stratification.

        Return a State object where s is linear by part and continous : linear
        under -`hmxl` with a slope of `strat_s`, and constant equals to `s_sfc`
        above -`hmxl`.

        Parameters
        ----------
        hmxl : float, default=20.
            mixed layer depth [m]
        s_sfc : float, default=21.
            surface salinity [psu]
        strat_t : float, default=1.3e-2
            salinity stratification above the mixed layer [psu.m-1]

        Returns
        -------
        state : State
            the `self` object with the the new initialization of salinity
        """
        maped_fun = vmap(piecewise_linear_flat, in_axes=(0, None, None, None))
        s_new = maped_fun(self.grid.zr, -hmxl, s_sfc, strat_s)
        return eqx.tree_at(lambda t: t.s, self, s_new)


class Trajectory(eqx.Module):
    """
    Define the history of a simulation with the time-series of the variables.

    Attributes
    ----------
    grid : Grid
        spatial grid
    time : jnp.ndarray, float(nt)
        time at each steps from the begining of the simulation [s]
    u : jnp.ndarray, float(nt, nz)
        time-serie of zonal velocity [m.s-1]
    v : jnp.ndarray, float(nt, nz)
        time-serie of meridional velocity [m.s-1]
    t : jnp.ndarray, floatnt, nz)
        time-serie of temperature [C]
    s : jnp.ndarray, float(nt, nz)
        time-serie of salinity [psu]

    Methods
    -------
    to_ds
        exports the trajectory in an xr.Dataset
    extract_state
        extracts the water column state at one time index

    """

    grid: Grid
    time: jnp.ndarray
    t: jnp.ndarray
    s: jnp.ndarray
    u: jnp.ndarray
    v: jnp.ndarray

    def to_ds(self) -> xr.Dataset:
        """
        Exports the trajectory in an xr.Dataset.

        The dimensions of the dataset are `time`, `grid.zr` and `grid.zw`, the
        variables are `u`, `v`, `t` and `s`, all defined on the dimensions
        (`time`, `zr`).

        Returns
        -------
        ds : xr.Dataset
            Dataset of the trajectory
        """
        variables = {'u': (('time', 'zr'), self.u),
                     'v': (('time', 'zr'), self.v),
                     't': (('time', 'zr'), self.t),
                     's': (('time', 'zr'), self.s)}
        coords = {'time': self.time,
                  'zr': self.grid.zr,
                  'zw': self.grid.zw}
        return xr.Dataset(variables, coords)

    def extract_state(self, i_time: int) -> State:
        """
        Etracts the water column state at one time index.

        Parameters
        ----------
        i_time : int
            the time index of the moment to extract the state

        Returns
        -------
        state : State
            the state of the trajectory at the time of index `i_time`
        """
        return State(self.grid, self.t[i_time, :], self.s[i_time, :],
                     self.u[i_time, :], self.v[i_time, :])
