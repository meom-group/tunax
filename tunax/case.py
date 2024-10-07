"""
Physical parameters and forcings of an experience of the model

Constants
---------
OMEGA
    angular velocity of earth [rad.s-1]
RAD_DEG
    ration between radian and degress [rad.degrees-1]

Classes
-------
Case
    Physical parameters and forcings of an experience of the model

"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp


OMEGA = 7.292116e-05
RAD_DEG = jnp.pi/180


class Case(eqx.Module):
    """
    Physical parameters and forcings of an experience of the model

    Attributes
    ----------
    rho0 : float, default=1024.
        default density [kg.m-3]
    grav : float, default=9.81
        gravity acceleration [m.s-2]
    cp : float, default=3985.
        specific heat capacity of saltwater [J.kg-1.K-1]
    alpha : float, default=2e-4
        thermal expansion coefficient [K-1]
    beta : float, default=8e-4
        salinity expansion coefficient [psu-1]
    t_rho_ref : float, default=0.
        reference temperature for the density computation [C]
    s_rho_ref : float, default=35.
        reference salinity for the density computation [psu]
    vkarmn : float, default=0.384
        Von Kármán constant [dimensionless]
    fcor : float, default=0.
        Coriolis frequency [rad.s-1]
    ustr_sfc : float, default=0.
        zonal wind stress [m2.s-2]
    ustr_btm : float, default=0.
        zonal current stress at the bottom [m2.s-2]
    vstr_sfc : float, default=0.
        meridional wind stress [m2.s-2]
    vstr_btm : float, default=0.
        meridional current stress at the bottom [m2.s-2]
    tflx_sfc : float, default=0.
        non-penetrative heat flux at the surface [K.m.s-1]
    tflx_btm : float, default=0.
        non-penetrative heat flux at the bottom [K.m.s-1]
    sflx_sfc : float, default=0.
        fresh water flux at the surface [psu.m.s-1]
    sflx_btm : float, default=0.
        fresh water flux at the bottom [psu.m.s-1]
    rflx_sfc_max : float, default=0.
        maximum solar radiation flux at the surface [K.m.s-1]
    
    Methods
    -------
    set_lat
        set the Coriolis frequency from the latitude
    set_u_cur
        set the zonal current stress with the zonal current
    set_v_wind
        set the meridional wind stress with the meridional wind
    set_v_cur
        set the meridional current stress with the meridional current
    set_tpw_sfc
        set the heat flux at surface from the heat power
    set_tpw_btm
        set the heat flux at bottom from the heat power
    set_rpw_sfc_max
        set the maximum solar radiation from the solar power

    Notes
    -----
    To modify the physical constants, it's better to recreate a new instance.

    """

    # physical constants
    rho0: float = 1024.
    grav: float = 9.81
    cp: float = 3985.
    alpha: float = 2e-4
    beta: float = 8e-4
    t_rho_ref: float = 0.
    s_rho_ref: float = 35.
    vkarmn: float = 0.384
    # forcings
    fcor: float = 0.
    ustr_sfc: float = 0.
    ustr_btm: float = 0.
    vstr_sfc: float = 0.
    vstr_btm: float = 0.
    tflx_sfc: float = 0.
    tflx_btm: float = 0.
    sflx_sfc: float = 0.
    sflx_btm: float = 0.
    rflx_sfc_max: float = 0.

    def set_lat(self, lat: float) -> Case:
        """
        Set the Coriolis frequency from the latitude.

        Parameters
        ----------
        lat : float
            latitude of the water column [degrees]
        
        Returns
        -------
        case : Case
            case with the modified Coriolis frequency`fcor`
        """
        fcor = 2.*OMEGA*jnp.sin(RAD_DEG*lat)
        case = eqx.tree_at(lambda t: t.fcor, self, fcor)
        return case
    
    def set_u_wind(self, u_wind: float) -> Case:
        """
        Set the zonal wind stress with the zonal wind.

        Parameters
        ----------
        u_wind : float
            zonal wind speed [m.s-1]
        
        Returns
        -------
        case : Case
            case with the modified zonal wind stress `ustr_sfc`
        """
        case = eqx.tree_at(lambda t: t.ustr_sfc, self, u_wind**2)
        return case
    
    def set_u_cur(self, u_cur: float) -> Case:
        """
        Set the zonal current stress with the zonal current.

        Parameters
        ----------
        u_cur : float
            zonal current speed [m.s-1]
        
        Returns
        -------
        case : Case
            case with the modified zonal current stress `ustr_btm`
        """
        case = eqx.tree_at(lambda t: t.ustr_btm, self, u_cur**2)
        return case
    
    def set_v_wind(self, v_wind: float) -> Case:
        """
        Set the meridional wind stress with the meridional wind.

        Parameters
        ----------
        v_wind : float
            meridional wind speed [m.s-1]
        
        Returns
        -------
        case : Case
            case with the modified meridional wind stress `vstr_sfc`
        """
        case = eqx.tree_at(lambda t: t.vstr_sfc, self, v_wind**2)
        return case
    
    def set_v_cur(self, v_cur: float) -> Case:
        """
        Set the meridional current stress with the meridional current.

        Parameters
        ----------
        v_cur : float
            meridional current speed [m.s-1]
        
        Returns
        -------
        case : Case
            case with the modified meridional current stress `vstr_btm`
        """
        case = eqx.tree_at(lambda t: t.vstr_btm, self, v_cur**2)
        return case
    
    def set_tpw_sfc(self, tpw_sfc: float) -> Case:
        """
        Set the heat flux at surface from the heat power.

        Parameters
        ----------
        tpw_sfc : float
            non-penetrative heat power at the surface [W.m-2]
        
        Returns
        -------
        case : Case
            case with the modified heat power
        """
        tflx_sfc = tpw_sfc/(self.rho0*self.cp)
        case = eqx.tree_at(lambda t: t.tflx_sfc, self, tflx_sfc)
        return case
    
    def set_tpw_btm(self, tpw_btm: float) -> Case:
        """
        Set the heat flux at bottom from the heat power.

        Parameters
        ----------
        tpw_btm : float
            non-penetrative heat power at the bottom [W.m-2]
        
        Returns
        -------
        case : Case
            case with the modified heat power
        """
        tflx_btm = tpw_btm/(self.rho0*self.cp)
        case = eqx.tree_at(lambda t: t.tflx_btm, self, tflx_btm)
        return case
    
    def set_rpw_sfc_max(self, rpw_sfc_max: float) -> Case:
        """
        Set the maximum solar radiation from the solar power

        Parameters
        ----------
        rpw_sfc_max : float
            maximum solar radiation power at the surface (penetrative) [W.m-2]
        
        Returns
        -------
        case : Case
            case with the modified heat power
        """
        rflx_sfc_max = rpw_sfc_max/(self.rho0*self.cp)
        case = eqx.tree_at(lambda t: t.rflx_sfc_max, self, rflx_sfc_max)
        return case
