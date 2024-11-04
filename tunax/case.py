"""
Physical parameters and forcings.

This module comes down to Case class. This class can be obtained by the prefix
:code:`tunax.case.` or directly by :code:`tunax.`.

"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax import device_get


_OMEGA = 7.292116e-05
"""float: Rotation rate of the Earth [rad.s-1]."""
_RAD_DEG = jnp.pi/180.
"""float: Measure of one degree in radiant [rad.°-1]."""


class Case(eqx.Module):
    r"""
    Physical parameters and forcings.

    This class contains all the physical constants, and the constant forcings
    that definine an experience for the model.

    Parameters
    ----------
    rho0 : float, default=1024.
        cf. attribute.
    grav : float, default=9.81
        cf. attribute.
    cp : float, default=3985.
        cf. attribute.
    alpha : float, default=2e-4
        cf. attribute.
    beta : float, default=8e-4
        cf. attribute.
    t_rho_ref : float, default=0.
        cf. attribute.
    s_rho_ref : float, default=35.
        cf. attribute.
    vkarmn : float, default=0.384
        cf. attribute.
    fcor : float, default=0.
        cf. attribute.
    ustr_sfc : float, default=0.
        cf. attribute.
    ustr_btm : float, default=0.
        cf. attribute.
    vstr_sfc : float, default=0.
        cf. attribute.
    vstr_btm : float, default=0.
        cf. attribute.
    tflx_sfc : float, default=0.
        cf. attribute.
    tflx_btm : float, default=0.
        cf. attribute.
    sflx_sfc : float, default=0.
        cf. attribute.
    sflx_btm : float, default=0.
        cf. attribute.
    rflx_sfc_max : float, default=0.
        cf. attribute.

    Attributes
    ----------
    rho0 : float, default=1024.
        Default density of saltwater :math:`[\text{kg} \cdot \text{m}^{-3}]`.
    grav : float, default=9.81
        Gravity acceleration :math:`[\text{m} \cdot \text{s}^{-2}]`.
    cp : float, default=3985.
        Specific heat capacity of saltwater
        :math:`[\text{J} \cdot \text{kg}^{-1} \cdot \text{K}^{-1}]`.
    alpha : float, default=2e-4
        Thermal expansion coefficient :math:`[\text{K}^{-1}]`.
    beta : float, default=8e-4
        Salinity expansion coefficient :math:`[\text{psu}^{-1}]`.
    t_rho_ref : float, default=0.
        Reference temperature for the density computation :math:`[° \text C]`.
    s_rho_ref : float, default=35.
        Reference salinity for the density computation :math:`[\text{psu}]`.
    vkarmn : float, default=0.384
        Von Kármán constant :math:`[\text{dimensionless}]`.
    fcor : float, default=0.
        Coriolis frequency at the water column
        :math:`[\text{rad} \cdot \text{s}^{-1}]`.
    ustr_sfc : float, default=0.
        Zonal wind stress :math:`[\text{m}^{2} \cdot \text{s}^{-2}]`.
    ustr_btm : float, default=0.
        Zonal current stress at the bottom
        :math:`[\text{m}^{2} \cdot \text{s}^{-2}]`.
    vstr_sfc : float, default=0.
        Meridional wind stress :math:`[\text{m}^{2} \cdot \text{s}^{-2}]`.
    vstr_btm : float, default=0.
        Meridional current stress at the bottom
        :math:`[\text{m}^{2} \cdot \text{s}^{-2}]`.
    tflx_sfc : float, default=0.
        Non-penetrative heat flux at the surface
        :math:`[\text{K} \cdot \text{m} \cdot \text{s}^{-1}]`.
    tflx_btm : float, default=0.
        Non-penetrative heat flux at the bottom
        :math:`[\text{K} \cdot \text{m} \cdot \text{s}^{-1}]`.
    sflx_sfc : float, default=0.
        Fresh water flux at the surface
        :math:`[\text{psu} \cdot \text{m} \cdot \text{s}^{-1}]`.
    sflx_btm : float, default=0.
        Fresh water flux at the bottom
        :math:`[\text{psu} \cdot \text{m} \cdot \text{s}^{-1}]`.
    rflx_sfc_max : float, default=0.
        Maximum solar radiation flux at the surface
        :math:`[\text{K} \cdot \text{m} \cdot \text{s}^{-1}]`.

    """

    # physcal constants
    rho0: float = 1024.
    grav: float = 9.81
    cp: float = 3985.
    alpha: float = 2e-4
    beta: float = 8e-4
    t_rho_ref: float = 0.
    s_rho_ref: float = 35.
    vkarmn: float = 0.384
    #forcings
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
            Latitude of the water column :math:`[°]`.
        
        Returns
        -------
        case : Case
            The :code:`self` object with the the new value of :attr:`fcor`.
        """
        fcor = float(device_get(2.*_OMEGA*jnp.sin(_RAD_DEG*lat)))
        print(type(fcor))
        print(fcor)
        case = eqx.tree_at(lambda t: t.fcor, self, fcor)
        return case

    def set_u_wind(self, u_wind: float) -> Case:
        r"""
        Set the zonal wind stress with the zonal wind speed.

        Parameters
        ----------
        u_wind : float
            Zonal wind speed :math:`[\text m \cdot \text s ^{-1}]`.
        
        Returns
        -------
        case : Case
            The :code:`self` object with the new value of :attr:`ustr_sfc`.
        """
        case = eqx.tree_at(lambda t: t.ustr_sfc, self, u_wind**2)
        return case

    def set_u_cur(self, u_cur: float) -> Case:
        r"""
        Set the zonal current stress with the zonal current.

        Parameters
        ----------
        u_cur : float
            Zonal current speed :math:`[\text m \cdot \text s ^{-1}]`.
        
        Returns
        -------
        case : Case
            The :code:`self` object with the new value of :attr:`ustr_btm`.
        """
        case = eqx.tree_at(lambda t: t.ustr_btm, self, u_cur**2)
        return case

    def set_v_wind(self, v_wind: float) -> Case:
        r"""
        Set the meridional wind stress with the meridional wind.

        Parameters
        ----------
        v_wind : float
            Meridional wind speed :math:`[\text m \cdot \text s ^{-1}]`.
        
        Returns
        -------
        case : Case
            The :code:`self` object with the new value of :attr:`vstr_sfc`.
        """
        case = eqx.tree_at(lambda t: t.vstr_sfc, self, v_wind**2)
        return case

    def set_v_cur(self, v_cur: float) -> Case:
        r"""
        Set the meridional current stress with the meridional current.

        Parameters
        ----------
        v_cur : float
            Meridional current speed :math:`[\text m \cdot \text s ^{-1}]`.
        
        Returns
        -------
        case : Case
            The :code:`self` object with the new value of :attr:`vstr_btm`.
        """
        case = eqx.tree_at(lambda t: t.vstr_btm, self, v_cur**2)
        return case

    def set_tpw_sfc(self, tpw_sfc: float) -> Case:
        r"""
        Set the heat flux at surface from the heat power.

        Parameters
        ----------
        tpw_sfc : float
            Non-penetrative heat power at the surface
            :math:`[\text W \cdot \text m ^{-2}]`.
        
        Returns
        -------
        case : Case
            The :code:`self` object with the new value of :attr:`tflx_sfc`.
        """
        tflx_sfc = tpw_sfc/(self.rho0*self.cp)
        case = eqx.tree_at(lambda t: t.tflx_sfc, self, tflx_sfc)
        return case

    def set_tpw_btm(self, tpw_btm: float) -> Case:
        r"""
        Set the heat flux at bottom from the heat power.

        Parameters
        ----------
        tpw_btm : float
            Non-penetrative heat power at the bottom
            :math:`[\text W \cdot \text m ^{-2}]`.
        
        Returns
        -------
        case : Case
            The :code:`self` object with the new value of :attr:`tflx_btm`.
        """
        tflx_btm = tpw_btm/(self.rho0*self.cp)
        case = eqx.tree_at(lambda t: t.tflx_btm, self, tflx_btm)
        return case

    def set_rpw_sfc_max(self, rpw_sfc_max: float) -> Case:
        r"""
        Set the maximum solar radiation from the solar power

        Parameters
        ----------
        rpw_sfc_max : float
            Maximum solar radiation power at the surface (penetrative)
            :math:`[\text W \cdot \text m ^{-2}]`.
        
        Returns
        -------
        case : Case
            The :code:`self` object with the new value of :attr:`rflx_sfc_max`.
        """
        rflx_sfc_max = rpw_sfc_max/(self.rho0*self.cp)
        case = eqx.tree_at(lambda t: t.rflx_sfc_max, self, rflx_sfc_max)
        return case
