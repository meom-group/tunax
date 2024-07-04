"""
Module for managing the space grids.

Classes
-------
Grid
    abstraction describing a one dimentional spatial grid

"""

import xarray as xr
import equinox as eqx
import jax.numpy as jnp
from typing import Type, TypeVar


class Grid(eqx.Module):
    """
    Abstraction describing a one dimentional spatial grid

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
    ORCA75
        reates the ORCA 75 levels grid and extracts the levels between -h and 0
    load
        creates the the grid defined by a dataset of an observation
        
    """

    nz: int
    zr: jnp.ndarray
    zw: jnp.ndarray
    hz: jnp.ndarray
    
    def __init__(self, zr:jnp.ndarray, zw: jnp.ndarray):
        """
        Creates a grid object from the centers and interfaces of the cells.

        Attributes
        ----------
        nz : int
            number of cells
        zr : jnp.ndarray, float(nz)
            depths of cell centers from deepest to shallowest [m]
        zw : jnp.ndarray, float(nz+1)
            depths of cell interfaces from deepest to shallowest [m]
        hz : jnp.ndarray, float(nz)
            thickness of cells from deepest to shallowest [m]
        
        Returns
        -------
        grid : Grid

        """
        self.nz = zw.shape[0]
        self.zw = zw
        self.zr = zr
        self.hz = zw[1:] - zw[:-1]

    @classmethod
    def linear(cls, nz: int, h: int):
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
    def analytic(cls, nz: int, h: float, hc: float, theta: float=6.5):
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
    def ORCA75(cls, h: float):
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
    def load(cls, ds: xr.Dataset):
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
