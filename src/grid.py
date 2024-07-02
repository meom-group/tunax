"""
Module servant à définir les grilles verticales.
"""

import xarray as xr
import equinox as eqx
import jax.numpy as jnp


class Grid(eqx.Module):
    """
    Attributes
    ----------
    nz : int
        number of cells
    zr : jnp.ndarray float(nz)
        depths of cell centers from deepest to shallowest [m]
    zw : jnp.ndarray float(nz+1)
        depths of cell interfaces from deepest to shallowest [m]
    hz : jnp.ndarray float(nz)
        thickness of cells from deepest to shallowest [m]
    """
    nz: int
    zw: jnp.ndarray
    zr: jnp.ndarray
    hz: jnp.ndarray
    
    def __init__(self, zw: jnp.ndarray, zr:jnp.ndarray):
        self.nz = zw.shape[0]
        self.zw = zw
        self.zr = zr
        self.hz = zr[1:] - zr[:-1]

    @classmethod
    def linear(cls, nz: int, h: int):
        """
        Creates a grid with equal thickness cells.

        Arguments
        ---------
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
        return cls(zw, zr)

    @classmethod
    def analytic(cls, nz: int, h: float, hc: float, theta: float=6.5):
        """
        Creates a grid of type analytic where the steps are almost constants
        above hc and wider under.

        Arguments
        ---------
        nz : int
            number of cells
        h : float
            maximum depth (positive) [m]
        hc : float
            reference depth (positive) [m]
        theta : float
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
        return cls(zw, zr)

    @classmethod
    def ORCA75(cls, h: float):
        """
        Creates the ORCA 75 levels grid and extract the levels between 0 and
        -h.

        Arguments
        ---------
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
        zw_orca = -(zsur + za0*sc_w + za1*zacr*jnp.log(jnp.cosh((sc_w-zkth)/zacr)) + za2*zacr2*jnp.log(jnp.cosh((sc_w-zkth2)/zacr2)))
        zr_orca = -(zsur + za0*sc_r + za1*zacr*jnp.log(jnp.cosh((sc_r-zkth)/zacr)) + za2*zacr2*jnp.log(jnp.cosh((sc_r-zkth2)/zacr2)))
        ibot = jnp.argmin(zw_orca <= -h)
        if ibot == 0:
            ibot = 1
        zw_orca = zw_orca.at[-1].set(0.)
        zw = zw_orca[ibot-1:]
        zr = zr_orca[ibot-1:]
        return cls(zw, zr)

    @classmethod
    def load(cls, ds: xr.Dataset):
        zw = jnp.array(ds['zw'], dtype=jnp.float32)
        zr = jnp.array(ds['zr'], dtype=jnp.float32)
        return cls(zw, zr)