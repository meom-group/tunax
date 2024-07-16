"""
"""

import equinox as eqx
import jax.numpy as jnp

NB_S_H = 3600                           # number of seconds in one hour
NB_S_D = 86400                          # number of seconds in a day
OMEGA = 7.292116e-05                    # angular velocity of earth [rad.s-1]
RAD_DEG = jnp.pi/180                    # ration between radian and degress [rad.degrees-1]

class Case(eqx.Module):
    """
    Define the physical parameters of an experiment
    """
    # physical constants
    rho0: float                             # default density [kg.m-3]
    grav: float                             # gravity acceleration [m.s-2]
    cp: float                               # specific heat capacity of saltwater [J.kg-1.C-1]
    alpha: float                            # thermal expansion coefficient [C-1]
    beta: float                             # salinity expansion coefficient [psu-1]
    t_ref: float                            # reference temperature in linear EOS [C]
    s_ref: float                            # reference salinity in linear EOS [psu]

    # forcings
    fcor: float                             # Coriolis frequency [rad.s-1]
    ustr_sfc: float                         # zonal wind stress [m2.s-2]
    ustr_btm: float                         # zonal current stress at the bottom [m2.s-2]
    vstr_sfc: float                         # meridional wind stress [m2.s-2]
    vstr_btm: float                         # meridional current stress at the bottom [m2.s-2]
    tflx_sfc: float                         # non-penetrative heat flux at the surface [K.m.s-1]
    tflx_btm: float                         # non-penetrative heat flux at the bottom [K.m.s-1]
    sflx_sfc: float                         # fresh water flux at the surface [psu.m.s-1]
    sflx_btm: float                         # fresh water flux at the bottom [psu.m.s-1]
    rflx_sfc_max: float                     # maximum solar radiation flux at the surface [K.m.s-1]
    do_diurnal_cycle: bool                  # apply a diurnal cycle for the solar radiation flux


    def __init__(self,
                 # physics constants
                 grav: float = 9.81,                    # gravity acceleration [m.s-2]
                 rho0: float = 1024.,                   # default density [kg.m-3]
                 cp: float = 3985.,                     # specific heat capacity of saltwater [J.kg-1.C-1]
                 t_coef: float = .2048,                 # thermal expansion coefficient [kg.m-3.C-1]
                 s_coef: float = .2048,                 # salinity expansion coefficient [kg.m-3.psu-1]
                 t_ref: float = 2.,                     # reference temperature in linear EOS [C]
                 s_ref: float = 35.,                    # reference salinity in linear EOS [psu]

                 # physics parameters
                 lat: float = 45.,                      # latitude of the column [degrees]
                 u_wind: float = 0.,                    # zonal wind speed [m.s-1]
                 u_cur: float = 0.,                     # zonal current speed at the bottom [m.s-1]
                 v_wind: float = 0.,                    # meridional wind speed [m.s-1]
                 v_cur: float = 0.,                     # meridional current speed at the bottom [m.s-1]
                 tflx_sfc_W: float = -500.,             # non-penetrative heat flux at the surface [W.m-2]
                 tflx_btm_W: float = 0.,                # non-penetrative heat flux at the bottom [W.m-2]
                 sflx_sfc: float = 0.,                  # fresh water flux at the surface [psu.m.s-1]
                 sflx_btm: float = 0.,                  # fresh water flux at the bottom [psu.m.s-1]
                 rflx_sfc_max_W: float = 0.,            # maximum solar radiation flux at the surface (penetrative heat flux) [W.m-2]
                 do_diurnal_cycle: bool = False,        # apply a diurnal cycle for the solar radiation flux
            ):
        # physical constants
        self.rho0 = rho0
        self.grav = grav
        self.cp = cp
        self.alpha = t_coef/rho0
        self.beta = s_coef/rho0
        self.t_ref = t_ref
        self.s_ref = s_ref

        # physical parameters
        self.fcor = 2.*OMEGA*jnp.sin(RAD_DEG*lat)
        self.ustr_sfc = u_wind**2
        self.ustr_btm = u_cur**2
        self.vstr_sfc = v_wind**2
        self.vstr_btm = v_cur**2
        cff = 1./(rho0*cp)
        self.tflx_sfc = tflx_sfc_W*cff
        self.tflx_btm = tflx_btm_W*cff
        self.sflx_sfc = sflx_sfc
        self.sflx_btm = sflx_btm
        self.rflx_sfc_max = rflx_sfc_max_W*cff
        self.do_diurnal_cycle = do_diurnal_cycle
