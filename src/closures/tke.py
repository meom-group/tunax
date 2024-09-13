"""
tke closure
"""

import sys
import equinox as eqx
import jax.numpy as jnp
from jax import jit, lax
from functools import partial

sys.path.append('..')
from state import Grid
from closure import ClosureParametersAbstract, ClosureStateAbstract
from state import State
from case import Case



### CONSTANTS TO REMOVE
grav = 9.81  # Gravity of Earth
vkarmn = 0.41  # Von Karman constant
cp = 3985.0  # Specific heat capacity of saltwater [J/kg K]
rsmall = 1.e-20  # Small constant

# NEMO constants
ceps_nemo = 0.5*jnp.sqrt(2.)  # Constant c_epsilon in NEMO
cm_nemo = 0.1  # Constant c_m in NEMO
ce_nemo = 0.1  # Constant c_e in NEMO
Ric_nemo = 2. / (2. + ceps_nemo / cm_nemo)  # Critical Richardson number

# MNH constants
ceps_mnh = 0.845  # Constant c_epsilon in MesoNH
cm_mnh = 0.126  # Constant c_m in MesoNH
ct_mnh = 0.143  # Constant c_s in MesoNH
ce_mnh = 0.34  # Constant c_e in MesoNH
Ric_mnh = 0.143  # Critical Richardson number

# RS81 constants
ceps_r81 = 0.7  # Constant c_epsilon in Redelsperger & Sommeria 1981
cm_r81 = 0.0667  # Constant c_m in Redelsperger & Sommeria 1981
ct_r81 = 0.167  # Constant c_s in Redelsperger & Sommeria 1981
ce_r81 = 0.4  # Constant c_e in Redelsperger & Sommeria 1981
Ric_r81 = 0.139  # Critical Richardson number



### CLOSURE STRUCTURE
class TkeParameters(ClosureParametersAbstract):
    mxl_min0: float = 0.04  # Minimum surface value for mixing lengths [m]
    pdlrmin: float = 0.1  # Minimum value for the inverse Prandtl number
    bshear: float = 1.e-20 # Minimum shear


class TkeState(ClosureStateAbstract):
    grid: Grid
    tke: jnp.ndarray
    lupw: jnp.ndarray
    ldwn: jnp.ndarray
    wtke: jnp.ndarray

    def __init__(self, grid: Grid, tke_min: float=1e-6, cmu_min: float=0.1,
                 cmu_prim_min: float=0.1):
        self.grid = grid
        ### PAS FINI LA
        


def tke_step(state: State,  tke_state: TkeState, tke_params: TkeParameters, case: Case):
    # changer ca
    eos_params = jnp.array([1024., 2e-4, 2e-4, 2., 35.])
    ED_tke_const = 1
    ED_tke_sfc_dirichlet = False
    dt = 30.
    akvmin = 1e-4
    aktmin = 1e-5
    mxlmin = 1.


    # attributes
    nz = state.grid.nz
    zr = state.grid.zr
    u = state.u
    v = state.akt
    akv = state.akv
    akt = state.akt
    hz = state.grid.hz

    # Compute Brunt-Vaisala frequency bvf for TKE production/destruction term
    rho, bvf = rho_eos_lin(state.t, state.s, zr, eos_params)

    # Compute boundary conditions for TKE
    tkemin = 1e-6 # CHANGER AVEC tke_min
    tke_sfc,tke_bot, flux_sfc = compute_tke_bdy(
        case.ustr_sfc,  case.vstr_sfc, ED_tke_const, 0., 0., tkemin)
    
    # Compute TKE production by shear
    shear = compute_shear(u, v, u, v, akv, zr)
    
    # Advance TKE
    tke_new, Prdtl, eps_new, residual, wtke_new = advance_tke(
        tke_state.tke, tke_state.lupw, tke_state.ldwn, akv, akt, hz,
        zr, bvf, shear, tke_state.wtke, dt, tke_sfc, tke_bot, flux_sfc,
        ED_tke_sfc_dirichlet, ED_tke_const,
        tkemin, tke_params)
    
    # Finalize eddy-viscosity/diffusivity computation
    lupw_new, ldwn_new = compute_mxl(
        tke_new, bvf, hz, case.ustr_sfc, case.vstr_sfc, mxlmin, tke_params)
    
    akv_new, akt_new = compute_ed(
        tke_new, lupw_new, ldwn_new, Prdtl, True,
        ED_tke_const, akvmin, aktmin)
    
    # Apply EVD if necessary (add this part)

    
    state = eqx.tree_at(lambda t: t.akv, state, akv_new)
    state = eqx.tree_at(lambda t: t.akt, state, akt_new)
    state = eqx.tree_at(lambda t: t.eps, state, eps_new)
    keps_state = eqx.tree_at(lambda t: t.tke, keps_state, tke_new)
    keps_state = eqx.tree_at(lambda t: t.lupw, keps_state, lupw_new)
    keps_state = eqx.tree_at(lambda t: t.ldwn, keps_state, ldwn_new)
    keps_state = eqx.tree_at(lambda t: t.wtke, keps_state, wtke_new)

    return state, tke_state



### INNER STRUCTURE
@jit
def rho_eos_lin(temp, salt, zr, eos_params):
    r"""
    Compute density anomaly and Brunt Vaisala frequency via linear Equation Of
    State (EOS)

    Parameters
    ----------
    temp : float(N)
        temperature [C]
    salt : float(N)
        salinity [psu]
    zr : float(N)
        depth at cell centers [m]
    eos_params : float([nb eos params])
        no description

    Returns
    -------
    bvf : float(N+1)
        Brunt Vaisala frequency [s-2]
    rho : float(N)
        density anomaly [kg/m3]

    Notes
    -----
    rho
    \(   \rho_{k} = \rho_0 \left( 1 - \alpha (\theta - 2) + \beta (S - 35)
    \right)  \)
    bvf
    \(   (N^2)_{k+1/2} = - \frac{g}{\rho_0}  \frac{ \rho_{k+1}-\rho_{k} }
    {\Delta z_{k+1/2}} \)
    """
    # returned variables
    N, = temp.shape
    bvf = jnp.zeros(N+1)
    rho = jnp.zeros(N)

    rhoRef = eos_params[0]
    alpha = eos_params[1]
    beta = eos_params[2]
    t0 = eos_params[3]
    s0 = eos_params[4]
    
    rho = rhoRef * (1.0 - alpha * (temp - t0) + beta * (salt - s0))
    
    cff = 1.0 / (zr[1:] - zr[:-1])
    bvf = bvf.at[1:N].set(-cff * (grav / rhoRef) * (rho[1:] - rho[:-1]))
    bvf = bvf.at[0].set(0.0)
    bvf = bvf.at[N].set(bvf[N-1])
    
    return rho, bvf


@partial(jit, static_argnums=2,)
def compute_tke_bdy(taux, tauy, tke_const, bc_ap, wp0, tkemin):
    r"""
    Compute top and bottom boundary conditions for TKE equation.

    Parameters
    ----------
    taux : float
        zonal surface stress      [m2/s2]
    tauy : float
        meridional surface stress [m2/s2]
    tke_const : int
        choice of TKE constants
    bc_ap : float
        choice of TKE constants
    wp0 : float
        surface value for plume vertical velocity [m/s]
    tkemin : float
        no description

    Returns
    -------
    tke_sfc : float
        surface value for Dirichlet condition [m2/s2]
    tke_bot : float
        bottom value for Dirichlet condition [m2/s2]
    flux_sfc : float
        surface TKE ED flux for Neumann condition [m3/s3]

    Notes
    -----
    free convection case
    \( {\rm tke\_sfc\_dirichlet = True}  \)

    tke_sfc
    \( k_{\rm sfc} = 1.5 \times 10^{-3}\;{\rm m}^2\;{\rm s}^{-2} \)

    flux_sfc
    energetically consistent boundary condition \( F_{\rm sfc}^k = \left. 
    K_e \partial_z k \right|_{\rm sfc} \)
    """
    # velocity scale
    ustar2 = jnp.sqrt(taux**2 + tauy**2)

    # surface boundary condition
    def false_fun(ustar):
        if tke_const == 0:
            cff = 1. / jnp.sqrt(cm_nemo * ceps_nemo)
        elif tke_const == 1:
            cff = 1. / jnp.sqrt(cm_mnh * ceps_mnh)
        else:
            cff = 1. / jnp.sqrt(cm_r81 * ceps_r81)
        return cff * ustar2
    tke_sfc = lax.cond(ustar2==0.0, lambda _: 0.0001, false_fun, ustar2)

    # surface TKE ED flux
    flux_sfc = 0.5 * bc_ap * wp0**3

    # bottom boundary condition
    tke_bot = tkemin

    return tke_sfc, tke_bot, flux_sfc


@jit
def compute_shear(u_n, v_n, u_np1, v_np1, Akv, zr):
    r"""
    Compute shear production term for TKE equation.

    Parameters
    ----------
    u_n : float(N)
        velocity components at time n    [m/s]
    v_n : float(N)
        velocity components at time n    [m/s]
    u_np1 : float(N)
        velocity components at time n+1  [m/s]
    v_np1 : float(N)
        velocity components at time n+1  [m/s]
    Akv : float(N+1)
        eddy-viscosity [m2/s]
    zr : float(N)
        depth at cell centers [m]

    Returns
    -------
    shear2 : float(N+1)
        shear production term [m2/s3]

    Notes
    -----
    Shear production term using discretization from Burchard (2002)
    \( {\rm Sh}_{k+1/2} = \frac{ (K_m)_{k+1/2} }{ \Delta z_{k+1/2}^2 }
    ( u_{k+1}^n - u_{k}^n ) ( u_{k+1}^{n+1/2} - u_{k}^{n+1/2} )  \)
    """
    # returned variables
    N, = u_n.shape

    cff = Akv[1:N] / (zr[1:] - zr[:N-1])**2
    du = 0.5 *cff*(u_np1[1:] - u_np1[:N-1])*(u_n[1:] + u_np1[1:] -
                                            u_n[:N-1] - u_np1[:N-1])
    dv = 0.5 *cff*(v_np1[1:] - v_np1[:N-1])*(v_n[1:] + v_np1[1:] -
                                               v_n[:N-1] - v_np1[:N-1])
    
    shear2_in = du + dv
    return jnp.concat([jnp.array([0]), shear2_in, jnp.array([0])])


@partial(jit, static_argnums=(14, 15))
def advance_tke(tke_n, lup, ldwn, Akv, Akt, Hz, zr, bvf, shear2, wtke, dt,
                tke_sfc, tke_bot, flux_sfc, dirichlet_bdy_sfc, tke_const,
                tkemin, tke_params):
    r"""
    TKE time stepping, advance tke from time step n to n+1.

    Parameters
    ----------
    tke_n : float(N+1)
        TKE at time n    [m2/s2]
    lup : float(N+1)
        upward mixing length [m]
    ldwn : float(N+1)
        downward mixing length [m]
    Akv : float(N+1)
        eddy-viscosity [m2/s]
    Akt : float(N+1)
        eddy-diffusion [m2/s]
    Hz : float(N)
        layer thickness [m]
    zr : float(N)
        depth at cell centers [m]
    bvf : float(N+1)
        Brunt Vaisala frequency [s-2]
    shear2 : float(N+1)
        shear tke production term [m2/s3]
    wtke : float(N) [modified]
        Diagnostics : w'e term  [m3/s3]
    dt : float
        time-step [s]
    tke_sfc : float
        surface boundary condition for TKE [m2/s2]
    tke_bot : float
        bottom boundary condition for TKE [m2/s2]
    flux_sfc : float
        surface TKE ED flux for Neumann condition [m3/s3]
    dirichlet_bdy_sfc : bool
        Nature of the TKE surface boundary condition (T:dirichlet,F:Neumann)
    tke_const : int
        choice of TKE constants
    tkemin : float
        no description

    Returns
    -------
    tke_np1 : float(N+1)
        TKE at time n+1    [m2/s2]
    pdlr : float(N+1)
        inverse of turbulent Prandtl number
    eps : float(N+1)
        TKE dissipation term [m2/s3]
    residual : float
        diagnostics : TKE spuriously added to guarantee that tke >= tke_min
        [m3/s3]
    wtke : float(N) [modified]
        Diagnostics : w'e term  [m3/s3]

    Notes
    -----
    dissipative mixing lengths
    \( (l_\epsilon)_{k+1/2} = \sqrt{ l_{\rm up} l_{\rm dwn} }   \)

    inverse Prandtl number function of Richardson number
    \( {\rm Ri}_{k+1/2} = (K_m)_{k+1/2} (N^2)_{k+1/2} / {\rm Sh}_{k+1/2} \)
    \( ({\rm Pr}_t)^{-1}_{k+1/2} = \max\left( {\rm Pr}_{\min}^{-1} ,
    \frac{{\rm Ri}_c}{ \max( {\rm Ri}_c, {\rm Ri}_{k+1/2}  ) } \right)     \)

    construct the right hand side
    \(  {\rm rhs}_{k+1/2} = {\rm Sh}_{k+1/2} - (K_s N^2)_{k+1/2} +
    {\rm Sh}_{k+1/2}^{\rm p} + (-a^{\rm p} w^{\rm p} B^{\rm p})_{k+1/2} +
    {\rm TOM}_{k+1/2}   \)

    right hand side for tridiagonal problem
    \( f_{k+1/2} = k^n_{k+1/2} + \Delta t {\rm rhs}_{k+1/2} + \frac{1}{2}
    \Delta t c_\epsilon \frac{ k^n_{k+1/2} \sqrt{k^n_{k+1/2}} }
    {(l_\epsilon)_{k+1/2}}   \)

    boundary conditions if dirichlet_bdy_sfc
    \( {\rm dirichlet\_bdy\_sfc = True}\qquad  \rightarrow \qquad
    k_{N+1/2}^{n+1} = k_{\rm sfc}  \)
    else
    \( {\rm dirichlet\_bdy\_sfc = False}\qquad  \rightarrow \qquad
    k_{N+1/2}^{n+1} - k_{N+1/2}^{n} = 2 \frac{\Delta z_{N}
    F_{\rm sfc}^k}{(K_e)_{N+1/2}+ (K_e)_{N-1/2}}  \)
    """
    # returned variables
    N = tke_n.shape[0]-1
    tke_np1 = jnp.zeros(N+1)
    pdlr = jnp.zeros(N+1)
    eps = jnp.zeros(N+1)
    residual = 0.0

    # local variables
    ff = jnp.zeros(N+1)

    # boundaries
    if dirichlet_bdy_sfc:
        tke_np1 = tke_np1.at[N].set(tke_sfc)
    else:
        tke_np1 = tke_np1.at[N].set(tkemin)
    tke_np1 = tke_np1.at[0].set(tke_bot)
    tke_np1 = tke_np1.at[1:N].set(tkemin)

    # initialize constants
    if tke_const == 0:
        ceps = ceps_nemo
        Ric = Ric_nemo
        isch = ce_nemo / cm_nemo
    elif tke_const == 1:
        ceps = ceps_mnh
        Ric = Ric_mnh
        isch = ce_mnh / cm_mnh
    else:
        ceps = ceps_r81
        Ric = Ric_r81
        isch = ce_r81 / cm_r81

    # dissipative mixing lengths
    mxld = jnp.zeros(N+1)
    mxld = mxld.at[1:N].set(jnp.sqrt(lup[1:N] * ldwn[1:N]))

    # inverse Prandtl number function of Richardson number
    sh2 = shear2[1:N] # shear2 is already multiplied by Akv
    buoy = bvf[1:N]
    Ri = jnp.maximum(buoy, 0.0) * Akv[1:N] / (sh2 + tke_params.bshear)
    pdlr = pdlr.at[1:N].set(jnp.maximum(tke_params.pdlrmin, Ric / jnp.maximum(Ric, Ri)))

    # constants for TKE dissipation term
    cff1 = 0.5
    cff2 = 1.5
    cff3 = cff1 / cff2

    # construct the right hand side
    rhs = shear2[1:N] - Akt[1:N] * bvf[1:N]
    # dissipation divided by tke
    eps = eps.at[1:N].set(cff2*ceps*jnp.sqrt(tke_n[1:N]) / mxld[1:N])
    # increase rhs if too small to guarantee that tke > tke_min
    rhsmin = (tkemin-tke_n[1:N])/dt + eps[1:N]*tkemin - cff3*eps[1:N]*tke_n[1:N]
    # right hand side for tridiagonal problem
    ff = ff.at[1:N].set(tke_n[1:N] + dt*jnp.maximum(rhs, rhsmin) + dt*cff3*eps[1:N]*tke_n[1:N])
    residual_vals = lax.select(rhs < rhsmin, (zr[1:N] - zr[:N-1])*(rhsmin - rhs), jnp.zeros(N-1))
    residual = jnp.sum(residual_vals)

    # boundary conditions
    ff = ff.at[0].set(tke_bot)
    if dirichlet_bdy_sfc:
        ff = ff.at[N].set(tke_sfc)
    else:
        ff = ff.at[N].set(2 * Hz[N-2] * flux_sfc / (isch * (Akv[N] + Akv[N-1])))

    # solve the tridiagonal problem
    ff = tridiag_solve_tke(Hz, isch*Akv, zr, eps, ff, dt, dirichlet_bdy_sfc)

    tke_np1 = jnp.maximum(ff, tkemin)

    # diagnostics
    # store the TKE dissipation term for diagnostics
    eps = eps.at[1:N].set(ceps*(cff2*tke_np1[1:N] - cff1*tke_n[1:N])*(jnp.sqrt(tke_n[1:N]) / mxld[1:N]))
    eps = eps.at[0].set(0.0)
    eps = eps.at[N].set(0.0)

    # store the ED contribution to w'e turbulent flux
    wtke = wtke + -0.5*isch*(Akv[1:] + Akv[:N])*(tke_np1[1:] - tke_np1[:N]) / Hz

    return tke_np1, pdlr, eps, residual, wtke


@jit
def compute_mxl(tke, bvf, Hz, taux, tauy, mxlmin, tke_params):
    r"""
    Compute mixing length scale.

    Parameters
    ----------
    tke : float(N+1)
        turbulent kinetic energy [m2/s2]
    bvf : float(N+1)
        Brunt Vaisala frequency [s-2]
    Hz : float(N)
        layer thickness [m]
    taux : float
        surface stress [m2/s2]
    tauy : float
        surface stress [m2/s2]
    mxlmin : float
        no description

    Returns
    -------
    lup : float(N+1)
        upward mixing length [m]
    ldwn : float(N+1)
        downward mixing length [m]

    Notes
    -----
    ustar2
    \( u_{\star}^2 = \sqrt{\tau_x^2 + \tau_y^2} \)

    buoyancy length scale
    \(  (l_{\rm up})_{k+1/2}=(l_{\rm dwn})_{k+1/2}=(l_{D80})_{k+1/2} =
    \sqrt{\frac{2 k_{k+1/2}^{n+1}}{ \jnp.maximum( (N^2)_{k+1/2}, (N^2)_{\min} ) }}\)

    limit
    \( (l_{\rm dwn})_{k+1/2} \) such that \( \partial_z (l_{\rm dwn})_{k}
    \le 1 \) the bottom boundary condition is
    \( (l_{\rm dwn})_{1/2} = l_{\min} \)

    limit
    \( (l_{\rm up})_{k-1/2} \) such that \( - \partial_z (l_{\rm up})_{k} \le
    1 \) the surface boundary condition is\( (l_{\rm up})_{N+1/2} =
    \frac{\kappa}{g} (2 \times 10^{-5}) u_{\star}^2 \)
    """
    # returned variables
    N = tke.shape[0]-1
    lup = jnp.full(N+1, mxlmin)
    ldwn = jnp.full(N+1, mxlmin)

    # local variables
    ld80 = jnp.zeros(N+1)
    
    ustar2 = jnp.sqrt(taux**2 + tauy**2)
    
    # initialize lup and ldwn arrays
    lup = jnp.full(N+1, mxlmin)
    ldwn = jnp.full(N+1, mxlmin)
    
    # compute ld80 array
    rn2 = jnp.maximum(bvf, rsmall) # interior value : l=sqrt(2*e/n^2)
    # buoyancy length scale
    ld80 = jnp.maximum(jnp.sqrt(2 * tke / rn2), mxlmin)
    
    # physical limits for the mixing lengths
    ldwn = ldwn.at[0].set(0.0)
    body_fun1 = lambda k, x: x.at[k].set(jnp.minimum(x[k-1] + Hz[k-1], ld80[k]))
    ldwn = lax.fori_loop(1, N+1, body_fun1, ldwn)
    
    # surface mixing length = F(stress)=vkarmn*2.e5*taum/(rho0*g)
    raug = vkarmn * 2e5 / grav
    lup = lup.at[N].set(jnp.maximum(tke_params.mxl_min0, raug * ustar2)) # surface boundary condition
    
    body_fun = lambda k, x: x.at[N-1-k].set(jnp.minimum(x[N-k] + Hz[N-1-k], ld80[N-1-k]))
    lup = lax.fori_loop(0, N, body_fun, lup)
    
    # ensures that avm(jpk) = 0.
    lup = lup.at[N].set(0.0)
    
    return lup, ldwn


@partial(jit, static_argnums=(4, 5))
def compute_ed(tke, lup, ldwn, pdlr, extrap_sfc, tke_const, Akvmin, Aktmin):
    r"""
    Compute the vertical eddy viscosity and diffusivity.

    Parameters
    ----------
    tke : float(N+1)
        turbulent kinetic energy [m2/s2]
    lup : float(N+1)
        upward mixing length [m]
    ldwn : float(N+1)
        downward mixing length [m]
    pdlr : float(N+1)
        inverse turbulent Prandtl number
    extrap_sfc : bool
        (T) extrapolate eddy coefficients to the surface
    tke_const : int
        choice of TKE constants
    Akvmin : float
        no description
    Aktmin : float
        no description

    Returns
    -------
    Akv : float(N+1)
        eddy-viscosity [m2/s]
    Akt : float(N+1)
        eddy-diffusivity [m2/s]

    Notes
    -----
    compute "master" mixing length
    \( (l_m)_{k+1/2} = \min( (l_{\rm up})_{k+1/2}, (l_{\rm dwn})_{k+1/2} ) \)

    compute eddy-viscosity
     \( (K_m)_{k+1/2} = C_m l_m \sqrt{k_{k+1/2}^{n+1}}  \)

    compute eddy-diffusivity
    \( (K_s)_{k+1/2} = ({\rm Pr}_t)^{-1}_{k+1/2}   (K_m)_{k+1/2} \)

    warning Akv
    \( {\rm extrap\_sfc = True} \qquad \rightarrow \qquad (K_m)_{N+1/2} =
    \frac{3}{2} (K_m)_{N-1/2} - \frac{1}{2} (K_m)_{N-3/2} \)
    """
    # returned variables
    N = tke.shape[0]-1
    Akv = jnp.zeros(N+1)
    Akt = jnp.zeros(N+1)

    # initialize constants
    if tke_const == 0:
        cm = cm_nemo
    elif tke_const == 1:
        cm = cm_mnh
    else:
        cm = cm_r81

    # Compute eddy-viscosity and eddy-diffusivity
    mxlm = jnp.minimum(lup, ldwn) # compute "master" mixing length
    av = cm * mxlm * jnp.sqrt(tke) # compute eddy-viscosity
    Akv = jnp.maximum(av, Akvmin)
    Akt = jnp.maximum(pdlr * av, Aktmin) # compute eddy-diffusivity
    Akv = Akv.at[N].set(0.0)
    Akt = Akt.at[N].set(0.0)

    # warning : extrapolation ignores the variations of Hz with depth
    if extrap_sfc:
        Akv = Akv.at[N].set(1.5 * Akv[N-1] - 0.5 * Akv[N-2])
        Akt = Akt.at[N].set(1.5 * Akt[N-1] - 0.5 * Akt[N-2])

    return Akv, Akt


@partial(jit, static_argnums=6,)
def tridiag_solve_tke(Hz, Ak, zr, eps, f, dt, dirichlet_bdy_sfc):
    """
    Solve the tridiagonal problem associated with the implicit in time
    treatment of TKE equation.

    Parameters
    ----------
    Hz : float(N)
        layer thickness [m]
    Ak : float(N+1)
        eddy-diffusivity for TKE [m2/s]
    zr : float(N)
        depth at cell centers [m]
    eps : float(N+1)
        TKE dissipation term divided by TKE [s-1]
    f : float(N+1) [modified]
        (in) rhs for tridiagonal problem
        (out) solution of the tridiagonal problem
    dt : float
        time step [s]
    dirichlet_bdy_sfc : bool
        nature of the TKE boundary condition

    Returns
    -------
    f : float(N+1) [modified]
        (in) rhs for tridiagonal problem
        (out) solution of the tridiagonal problem
    """
    # local variables
    N, = Hz.shape
    a = jnp.zeros(N+1)
    b = jnp.zeros(N+1)
    c = jnp.zeros(N+1)
    q = jnp.zeros(N+1)
    
    difA = -0.5 * dt * (Ak[:N-1] + Ak[1:N]) / (Hz[:N-1] * (zr[1:] - zr[:N-1]))
    difC = -0.5 * dt * (Ak[2:] + Ak[1:N]) / (Hz[1:] * (zr[1:] - zr[:N-1]))
    difB = 1.0 - difA - difC + dt*eps[1:N]

    a = a.at[1:N].set(difA)
    b = b.at[1:N].set(difB)
    c = c.at[1:N].set(difC)

    # bottom boundary condition    
    b = b.at[0].set(1.0)
    if dirichlet_bdy_sfc:
        b = b.at[N].set(1.0)
    else:
        a = a.at[N].set(-1.0)
        b = b.at[N].set(1.0)

    # forward sweep
    cff = 1/b[0]
    q = q.at[0].set(-c[0] * cff)
    f = f.at[0].multiply(cff)

    def body_fun(k, x):
        f = x[0, :]
        q = x[1, :]
        cff = 1.0 / (b[k] + a[k] * q[k-1])
        q = q.at[k].set(-cff * c[k])
        f = f.at[k].set(cff * (f[k] - a[k] * f[k-1]))
        return jnp.stack([f, q])
    fq = jnp.stack([f, q])
    fq = lax.fori_loop(1, N+1, body_fun, fq)
    f = fq[0, :]
    q = fq[1, :]
    
    # backward substitution
    body_fun = lambda k, x: x.at[N-1-k].add(q[N-1-k] * x[N-k])
    f = lax.fori_loop(0, N, body_fun, f)
    
    return f
