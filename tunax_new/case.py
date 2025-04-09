"""
Physical parameters and forcings.

This module comes down to Case class. This class can be obtained by the prefix
:code:`tunax.case.` or directly by :code:`tunax.`.

"""

from __future__ import annotations
from typing import Union, Tuple, Callable, Optional

import equinox as eqx
import jax.numpy as jnp
from jax import device_get


_OMEGA = 7.292116e-05
"""float: Rotation rate of the Earth [rad.s-1]."""
_RAD_DEG = jnp.pi/180.
"""float: Measure of one degree in radiant [rad.°-1]."""

ForcingType = Union[
    Tuple[float, float],
    Callable[[float], float],
    Callable[[float, float], float]
]


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
    eos_tracers: CHANGER HERE
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
    eos_tracers: str = eqx.field(default='t', static=True)
    alpha: float = 2e-4
    beta: float = 8e-4
    t_rho_ref: float = 0.
    s_rho_ref: float = 35.
    do_pt: bool = eqx.field(default=False, static=True)
    vkarmn: float = 0.384
    # dynamic forcings
    fcor: float = 0.
    ustr_sfc: float = 0.
    ustr_btm: float = 0.
    vstr_sfc: float = 0.
    vstr_btm: float = 0.
    # tracers forcings
    t_forcing: Optional[ForcingType] = eqx.field(default=None, static=True)
    s_forcing: Optional[ForcingType] = eqx.field(default=None, static=True)
    b_forcing: Optional[ForcingType] = eqx.field(default=None, static=True)
    pt_forcing: Optional[ForcingType] = eqx.field(default=None, static=True)

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
        case = eqx.tree_at(lambda t: t.fcor, self, fcor)
        return case

    def set_dynamic_forcing_speed(
            self,
            direction: str, # u or v,
            boundary: str, # sfc or btm
            cur_speed: float):
        """
        CHANGE HERE
        """
        arg_name = f'{direction}str_{boundary}'
        case = eqx.tree_at(lambda t: getattr(t, arg_name), self, cur_speed**2)
        return case

    def set_tracers_forcing_power(
            self,
            tracer: str, # t, s, b or pt
            boundary: str, # sfc or btm
            power: float):
        """
        CHANGE HERE
        """
        arg_name = f'{tracer}flx_{boundary}'
        flux = power/(self.rho0*self.cp)
        case = eqx.tree_at(lambda t: getattr(t, arg_name), self, flux)
        return case
