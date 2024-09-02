import jax.numpy as jnp
from jax import jit, lax
import equinox as eqx
import sys
sys.path.append('..')
from grid import Grid
from model import State
from case import Case
from closure import ClosureParametersAbstract, ClosureStateAbstract


grav = 9.81  # Gravity of Earth
vkarmn = 0.41  # Von Karman constant
galp = 0.53  # parameter for Galperin mixing length limitation
chk = 1400. / grav  # charnock coefficient
Zosmin = 1.e-2  # min surface roughness length
Zobmin = 1.e-4  # min bottom roughness length
z0b = 1.e-14    # bottom roughness length


class KepsParameters(ClosureParametersAbstract):
    a1: float
    a2: float
    a3: float
    nn: float
    sf_d0: float
    sf_d1: float
    sf_d2: float
    sf_d3: float
    sf_d4: float
    sf_d5: float
    sf_n0: float
    sf_n1: float
    sf_n2: float
    sf_nb0: float
    sf_nb1: float
    sf_nb2: float
    lim_am0: float
    lim_am1: float
    lim_am2: float
    lim_am3: float
    lim_am4: float
    lim_am5: float
    lim_am6: float


    def __init__(self,
        c1: float=5.0,
        c2: float=0.8,
        c3: float=1.968,
        c4: float=1.136,
        c5: float=0.,
        c6: float=0.4,
        cb1: float=5.95,
        cb2: float=0.6,
        cb3: float=1.,
        cb4: float=0.,
        cb5: float=0.33333,
        cbb: float=0.72
    ):

        a1 = 0.66666666667 - 0.5*c2
        a2 = 1.0 - 0.5*c3
        a3 = 1.0 - 0.5*c4
        a5 = 0.5 - 0.5*c6
        nn = 0.5*c1
        nb = cb1
        ab1 = 1.0 - cb2
        ab2 = 1.0 - cb3
        ab3 = 2.0*(1.0 - cb4)
        ab5 = 2.0*cbb*(1.0 - cb5)
        sf_d0 = 36.0*nn*nn*nn*nb*nb
        sf_d1 = 84.0*a5*ab3*nn*nn*nb + 36.0*ab5*nn*nn*nn*nb
        sf_d2 = 9.0*(ab2*ab2 - ab1*ab1)*nn*nn*nn - 12.0*(a2*a2 - 3.0*a3*a3)*nn*nb*nb
        sf_d3 = 12.0*a5*ab3*(a2*ab1 - 3.0*a3*ab2)*nn + 12.0*a5*ab3*(a3*a3 - a2*a2)*nb + 12.0*ab5*(3.0*a3*a3 - a2*a2)*nn*nb
        sf_d4 = 48.0*a5*a5*ab3*ab3*nn + 36.0*a5*ab3*ab5*nn*nn
        sf_d5 = 3.0*(a2*a2 - 3.0*a3*a3)*(ab1*ab1 - ab2*ab2)*nn
        sf_n0 = 36.0*a1*nn*nn*nb*nb
        sf_n1 = -12.0*a5*ab3*(ab1 + ab2)*nn*nn + 8.0*a5*ab3*(6.0*a1 - a2 - 3.0*a3)*nn*nb + 36.0*a1*ab5*nn*nn*nb
        sf_n2 = 9.0*a1*(ab2*ab2 - ab1*ab1)*nn*nn
        sf_nb0 = 12.0*ab3*nn*nn*nn*nb
        sf_nb1 = 12.0*a5*ab3*ab3*nn*nn
        sf_nb2 = 9.0*a1*ab3*(ab1 - ab2)*nn*nn + (6.0*a1*(a2 - 3.0*a3) - 4.0*(a2*a2 - 3.0*a3*a3))*ab3*nn*nb
        lim_am0 = sf_d0*sf_n0
        lim_am1 = sf_d0*sf_n1 + sf_d1*sf_n0
        lim_am2 = sf_d1*sf_n1 + sf_d4*sf_n0
        lim_am3 = sf_d4*sf_n1
        lim_am4 = sf_d2*sf_n0
        lim_am5 = sf_d2*sf_n1 + sf_d3*sf_n0
        lim_am6 = sf_d3*sf_n1

        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.nn = nn
        self.sf_d0 = sf_d0
        self.sf_d1 = sf_d1
        self.sf_d2 = sf_d2
        self.sf_d3 = sf_d3
        self.sf_d4 = sf_d4
        self.sf_d5 = sf_d5
        self.sf_n0 = sf_n0
        self.sf_n1 = sf_n1
        self.sf_n2 = sf_n2
        self.sf_nb0 = sf_nb0
        self.sf_nb1 = sf_nb1
        self.sf_nb2 = sf_nb2
        self.lim_am0 = lim_am0
        self.lim_am1 = lim_am1
        self.lim_am2 = lim_am2
        self.lim_am3 = lim_am3
        self.lim_am4 = lim_am4
        self.lim_am5 = lim_am5
        self.lim_am6 = lim_am6

class KepsState(ClosureStateAbstract):
    grid: Grid
    tke: jnp.ndarray
    cmu: jnp.ndarray
    cmu_prim: jnp.ndarray

    def __init__(self, grid: Grid, tke_min: float=1e-6, cmu_min: float=0.1,
                 cmu_prim_min: float=0.1):
        self.grid = grid
        self.tke = jnp.full(grid.nz+1, tke_min)
        self.cmu = jnp.full(grid.nz+1, cmu_min)
        self.cmu_prim = jnp.full(grid.nz, cmu_prim_min)


def keps_step(state: State, keps_state: KepsState, keps_params: KepsParameters, case: Case):
    # changer ca
    eos_params = jnp.array([1024., 2e-4, 2e-4, 2., 35.])
    pnm = jnp.array([3., -1., 1.5])
    betaCoef = jnp.array([1.44, 1.92, -0.4, 1.])

    # attributes
    nz = state.grid.nz
    tke = keps_state.tke
    akv = keps_state.akv
    eps = keps_state.eps
    cmu = keps_state.cmu
    cmu_prim = keps_state.cmu_prim
    u = state.u
    v = state.v
    zr = state.grid.zr
    hz = state.grid.hz

    rho, bvf = rho_eos_lin(state.t, state.s, zr, eos_params)

    shear2 = compute_shear(u, v, u, v, zr)

    tke1 = 0.5*(tke[nz]+tke[nz-1]) 
    tke2 = 0.5*(tke[0]+tke[1])
    tkemin = 1e-6 # CHANGER AVEC tke_min
    akvmin = 1e-4
    aktmin = 1e-5
    epsmin = 1e-12 # CHANGER AVEC eps_min
    z0b = 1e-14 # CHANGER AVEC eps_min
    OneOverSig_psi = 1/1.3
    OneOverSig_k = 1.
    dt = 10. # CHANGER

    tke_sfc, tke_bot, ftke_sfc, ftke_bot, eps_sfc, eps_bot, feps_sfc, feps_bot = compute_tke_eps_bdy(
        case.ustr_sfc, case.vstr_sfc, case.ustr_btm, case.vstr_btm, z0b, tke1, hz[-1], tke2, hz[0], OneOverSig_psi, pnm, tkemin, epsmin, keps_params)
    
    # CA C'est bien nul il faut changer
    bdy_tke_sfc = jnp.zeros(2)
    bdy_tke_bot = jnp.zeros(2)
    bdy_eps_sfc = jnp.zeros(2)
    bdy_eps_bot = jnp.zeros(2)
    if bdy_tke_sfc[0] < 0.5: bdy_tke_sfc = bdy_tke_sfc.at[1].set(tke_sfc)
    else: bdy_tke_sfc = bdy_tke_sfc.at[1].set(ftke_sfc)
    if bdy_tke_bot[0] < 0.5: bdy_tke_bot = bdy_tke_bot.at[1].set(tke_bot)
    else: bdy_tke_bot = bdy_tke_bot.at[1].set(ftke_bot)
    if bdy_eps_sfc[0] < 0.5: bdy_eps_sfc = bdy_eps_sfc.at[1].set(eps_sfc)
    else: bdy_eps_sfc = bdy_eps_sfc.at[1].set(feps_sfc)
    if bdy_eps_bot[0] < 0.5: bdy_eps_bot = bdy_eps_bot.at[1].set(eps_bot)
    else: bdy_eps_bot = bdy_eps_bot.at[1].set(feps_bot)


    tke_new, wtke = advance_turb_tke(tke, bvf, shear2, OneOverSig_k*akv, akv, keps_state.akt, eps, hz, dt, tkemin, bdy_tke_sfc, bdy_tke_bot)
    eps_new = advance_turb_eps(eps, bvf, shear2, OneOverSig_psi*akv, cmu, cmu_prim, tke, tke_new, hz, dt, betaCoef, epsmin, bdy_eps_sfc, bdy_eps_bot)
    akv_new, akt_new, cmu_new, cmu_prim_new, eps_new = compute_ev_ed_filt(tke_new, eps_new, bvf, shear2 , pnm, akvmin, aktmin, epsmin, keps_params)
    
    keps_state = eqx.tree_at(lambda t: t.akv, keps_state, akv_new)
    keps_state = eqx.tree_at(lambda t: t.akt, keps_state, akt_new)
    keps_state = eqx.tree_at(lambda t: t.eps, keps_state, eps_new)
    keps_state = eqx.tree_at(lambda t: t.tke, keps_state, tke_new)
    keps_state = eqx.tree_at(lambda t: t.cmu, keps_state, cmu_new)
    keps_state = eqx.tree_at(lambda t: t.cmu_prim, keps_state, cmu_prim_new)




@jit
def compute_tke_eps_bdy(ustr_sfc, vstr_sfc, ustr_bot, vstr_bot, z0b, tke_n_sfc,
                        Hz_sfc, tke_n_bot, Hz_bot, OneOverSig, pnm, tke_min,
                        eps_min, keps_params):
    """
    Compute top and bottom boundary conditions for TKE and GLS equation.

    Parameters
    ----------
    ustr_sfc : float
        zonal surface stress      [m2/s2]
    vstr_sfc : float
        meridional surface stress [m2/s2]
    ustr_bot : float
        zonal bottom stress      [m2/s2]
    vstr_bot : float
        meridional bottom stress [m2/s2]
    z0b : float
        bottom roughness length [m]
    tke_n_sfc : float
        [m2/s2]
    Hz_sfc : float
        [m]
    tke_n_bot : float
        [m2/s2]
    Hz_bot : float
        [m]
    OneOverSig : float
        [-]
    pnm : float(3)
        no description
    tke_min : float
        [m2/s2]
    eps_min : float
        [m2/s3]

    Returns
    -------
    tke_sfc : float
        surface value for Dirichlet condition [m2/s2]
    tke_bot : float
        bottom value for Dirichlet condition [m2/s2]
    ftke_sfc : float
        surface TKE flux for Neumann condition [m3/s3]
    ftke_bot : float
        bottom TKE flux for Neumann condition [m3/s3]
    eps_sfc : float
        surface EPS value for Dirichlet condition [m2/s3]
    eps_bot : float
        bottom EPS value for Dirichlet condition [m2/s3]
    feps_sfc : float
        surface EPS flux for Neumann condition [m3/s4]
    feps_bot : float
        bottom EPS flux for Neumann condition [m3/s4]
    """
    a1 = keps_params.a1
    a2 = keps_params.a2
    a3 = keps_params.a3
    nn = keps_params.nn

    # Calculate cm0
    cm0 = ((a2**2 - 3.0*a3**2 + 3.0*a1*nn) / (3.0*nn**2))**0.25
    rp, rn, rm = pnm[0], pnm[1], pnm[2]
    cm0inv2 = 1.0 / cm0**2

    # Velocity scales
    ustar2_sfc = jnp.sqrt(ustr_sfc**2 + vstr_sfc**2)
    ustar2_bot = jnp.sqrt(ustr_bot**2 + vstr_bot**2)

    # TKE Dirichlet boundary condition
    tke_sfc = jnp.maximum(tke_min, cm0inv2*ustar2_sfc)
    tke_bot = jnp.maximum(tke_min, cm0inv2*ustar2_bot)

    # TKE Neumann boundary condition
    ftke_sfc = 0.0
    ftke_bot = 0.0

    # Surface conditions
    z0_s = jnp.maximum(Zosmin, chk*ustar2_sfc) # Charnock
    lgthsc = vkarmn*(0.5*Hz_sfc + z0_s)
    eps_sfc = jnp.maximum(eps_min, (cm0**rp)*(lgthsc**rn)*(tke_n_sfc**rm))
    feps_sfc = -rn*cm0**(rp + 1.0)*vkarmn*OneOverSig*(tke_n_sfc**(rm + 0.5))*\
        (lgthsc**rn)

    # Bottom conditions
    z0_b = jnp.maximum(z0b, Zobmin)
    lgthsc = vkarmn *(0.5 *Hz_bot + z0_b)
    eps_bot = jnp.maximum(eps_min, (cm0**rp) *(lgthsc**rn) *(tke_n_bot**rm))
    feps_bot = -rn*cm0**(rp + 1.0)*vkarmn*OneOverSig*(tke_n_bot**(rm + 0.5))*\
        (lgthsc**rn)

    return tke_sfc, tke_bot, ftke_sfc, ftke_bot, eps_sfc, eps_bot, feps_sfc, \
        feps_bot


@jit
def compute_shear(u_n, v_n, u_np1, v_np1, zr):
    r"""
    Compute shear production term for TKE equation.
    
    Parameters
    ----------
    u_n : float(N)
        velocity components at time n [m/s]
    v_n : float(N)
        velocity components at time n [m/s]
    u_np1 : float(N)
        velocity components at time n+1 [m/s]
    v_np1 : float(N)
        velocity components at time n+1 [m/s]
    zr : float(N)
        depth at cell centers [m]
        

    Returns
    -------
    shear2 : float(N+1)
        shear production term [m2/s3]

    Notes
    -----
    Shear production term using discretization from Burchard (2002)
    \( {\rm Sh}_{k+1/2} = \frac{ 1 }{ \Delta z_{k+1/2}^2 } ( u_{k+1}^n -
    u_{k}^n ) ( u_{k+1}^{n+1/2} - u_{k}^{n+1/2} )  \)
    """
    # returned variables
    N, = zr.shape
    shear2 = jnp.zeros(N+1)

    # for k in range(N-1):
    cff = 1.0 / (zr[1:] - zr[:N-1])**2
    du = 0.5*cff*(u_np1[1:]-u_np1[:N-1])*(u_n[1:]+u_np1[1:]-u_n[:N-1]-
                                    u_np1[:N-1])
    dv = 0.5*cff*(v_np1[1:]-v_np1[:N-1])*(v_n[1:]+v_np1[1:]-v_n[:N-1]-
                                        v_np1[:N-1])
    
    shear2_in = du + dv
    return jnp.concat([jnp.array([0]), shear2_in, jnp.array([0])])


@jit
def compute_ev_ed_filt(tke, gls, bvf, shear2, pnm, nuwm, nuws, eps_min, keps_params):
    """
    No description.

    Parameters
    ----------
    tke : float(N+1)
        no description
    gls : float(N+1) [modified]
        no description
    bvf : float(N+1)
        no description
    shear2 : float(N+1)
        no description
    pnm : float(3)
        no description
    nuwm : float
        no description
    nuws : float
        no description
    eps_min : float
        no description

    Returns
    -------
    Akv : float(N+1)
        eddy-viscosity [m2/s]
    Akt : float(N+1)
        eddy-diffusivity [m2/s]
    c_mu : float(N+1)
        no description
    c_mu_prim : float(N+1)
        no description
    gls : float(N+1) [modified]
        no description
    """
    a1 = keps_params.a1
    a2 = keps_params.a2
    a3 = keps_params.a3
    nn = keps_params.nn
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

    # returned variables
    N = tke.shape[0]-1
    Akv = jnp.zeros(N+1)
    Akt = jnp.zeros(N+1)
    c_mu = jnp.zeros(N+1)
    c_mu_prim = jnp.zeros(N+1)

    # local variables
    aN = jnp.zeros(N+1)
    aM = jnp.zeros(N+1)
    filter_cof = 0.5

    cm0 = ((a2**2 - 3.0*a3**2 + 3.0*a1*nn) / (3.0*nn**2))**0.25
    rp, rn, rm = pnm[0], pnm[1], pnm[2]
    e1 = 3.0 + rp / rn
    e2 = 1.5 + rm / rn
    e3 = -1.0 / rn

    # Minimum value of alpha_n to ensure that alpha_m is positive
    alpha_n_min = 0.5*(- (sf_d1 + sf_nb0) + jnp.sqrt((sf_d1 + sf_nb0)**2 - \
        4.0*sf_d0*(sf_d4 + sf_nb1))) / (sf_d4 + sf_nb1)

    # Compute Akv & Lscale
    # Galperin limitation : l <= l_li
    L_lim = galp*jnp.sqrt(2.0*tke[1:N]) / jnp.sqrt(jnp.maximum(1e-14, bvf[1:N]))
    # Limitation on psi (use MAX because rn is negative)
    cff = (cm0**rp)*(L_lim**rn)*(tke[1:N]**rm)
    gls = gls.at[1:N].set(jnp.maximum(gls[1:N], cff))
    epsilon = (cm0**e1)*(tke[1:N]**e2)*(gls[1:N]**e3)
    epsilon = jnp.maximum(epsilon, eps_min)
    # Compute alpha_n and alpha_m
    cff = (tke[1:N] / epsilon)**2
    aM = aM.at[1:N].set(cff*shear2[1:N])
    aN = aN.at[1:N].set(cff*bvf[1:N])

    alpha_n = aN[1:N] + filter_cof*(0.5*aN[2:] - aN[1:N] + 0.5*aN[:N-1])
    alpha_m = aM[1:N] + filter_cof*(0.5*aM[2:] - aM[1:N] + 0.5*aM[:N-1])
    # Limitation of alpha_n and alpha_m
    alpha_n = jnp.minimum(jnp.maximum(0.73*alpha_n_min, alpha_n), 1.0e10)
    alpha_m_max = (lim_am0 + lim_am1*alpha_n + lim_am2*alpha_n**2 + \
        lim_am3*alpha_n**3) / (lim_am4 + lim_am5*alpha_n + \
        lim_am6*alpha_n**2)
    alpha_m = jnp.minimum(alpha_m, alpha_m_max)
    # Compute stability functions
    Denom = sf_d0 + sf_d1*alpha_n + sf_d2*alpha_m + sf_d3*alpha_n*alpha_m \
        + sf_d4*alpha_n**2 + sf_d5*alpha_m**2
    cff = 1.0 / Denom
    c_mu = c_mu.at[1:N].set(cff*(sf_n0 + sf_n1*alpha_n + sf_n2*alpha_m))
    c_mu_prim = c_mu_prim.at[1:N].set(cff*(sf_nb0 + sf_nb1*alpha_n + sf_nb2*alpha_m))

    # for k in range(1, N):
    epsilon = (cm0**e1)*(tke[1:N]**e2)*(gls[1:N]**e3)
    epsilon = jnp.maximum(epsilon, eps_min)
    # Finalize the computation of Akv and Akt
    cff = tke[1:N]**2 / epsilon
    Akv = Akv.at[1:N].set(jnp.maximum(cff*c_mu[1:N], nuwm))
    Akt = Akt.at[1:N].set(jnp.maximum(cff*c_mu_prim[1:N], nuws))

    Akv = Akv.at[N].set(jnp.maximum(1.5*Akv[N-1] - 0.5*Akv[N-2], nuwm))
    Akv = Akv.at[0].set(jnp.maximum(1.5*Akv[1] - 0.5*Akv[2], nuwm))
    Akt = Akt.at[N].set(jnp.maximum(1.5*Akt[N-1] - 0.5*Akt[N-2], nuws))
    Akt = Akt.at[0].set(jnp.maximum(1.5*Akt[1] - 0.5*Akt[2], nuws))

    return Akv, Akt, c_mu, c_mu_prim, gls


@jit
def advance_turb_tke(tke_n, bvf, shear2, Ak_tke, Akv, Akt, diss, Hz, dt,
                     tke_min, bdy_sfc, bdy_bot):
    """
    No Description.

    Parameters
    ----------
    tke_n : float(N+1)
        no description
    bvf : float(N+1)
        no description
    shear2 : float(N+1)
        no description
    Ak_tke : float(N+1)
        no description
    Akv : float(N+1)
        no description
    Akt : float(N+1)
        no description
    diss : float(N+1)
        no description
    Hz : float(N)
        no description
    dt : float
        no description
    tke_min : float
        no description
    bdy_sfc : float(2)
        bdy_sfc(1) = 0. -> Dirichlet, bdy_sfc(1) = 1. -> Neumann
    bdy_bot : float(2)
        bdy_bot(1) = 0. -> Dirichlet, bdy_top(1) = 1. -> Neumann

    Returns
    -------
    tke_np1 : float(N+1)
        no description
    wtke : float(N)
        no description
    """
    # returned variables
    N = tke_n.shape[0]-1
    tke_np1 = jnp.zeros(N+1)
    wtke = jnp.zeros(N)

    # local variables
    aa = jnp.zeros(N+1)
    bb = jnp.zeros(N+1)
    cc = jnp.zeros(N+1)
    rhs = jnp.zeros(N+1)
    q = jnp.zeros(N+1)

    # fill in the tridiagonal matrix
    # Off-diagonal terms for the tridiagonal problem
    cff = -0.5*dt
    aa = aa.at[1:N].set(cff*(Ak_tke[1:N] + Ak_tke[:N-1]) / Hz[:N-1])
    cc = cc.at[1:N].set(cff*(Ak_tke[1:N] + Ak_tke[2:]) / Hz[1:])

    # Shear and buoyancy production
    Sprod = Akv[1:N]*shear2[1:N]
    Bprod = -Akt[1:N]*bvf[1:N]
    # Diagonal and rhs term
    cff = 0.5*(Hz[:N-1] + Hz[1:])
    invG = 1.0 / tke_n[1:N]
    rhs_in = lax.select((Bprod + Sprod) > 0, cff*(tke_n[1:N] + dt*(Bprod + Sprod)), cff*(tke_n[1:N] + dt*Sprod))
    rhs = rhs.at[1:N].set(rhs_in)
    bb_in = lax.select((Bprod + Sprod) > 0, cff*(1.0 + dt*diss[1:N]*invG) - aa[1:N] - cc[1:N], cff*(1.0 + dt*(diss[1:N] - Bprod)*invG) - aa[1:N] - cc[1:N])
    bb = bb.at[1:N].set(bb_in)

    # surface boundary condition
    cond_sfc = bdy_sfc[0] < 0.5
    aa_sfc = cond_sfc*0. + (1-cond_sfc)*-0.5*(Ak_tke[N] + Ak_tke[N-1])
    bb_sfc = cond_sfc*1. + (1-cond_sfc)*0.5*(Ak_tke[N] + Ak_tke[N-1])
    rhs_sfc = cond_sfc*bdy_sfc[1] + (1-cond_sfc)*Hz[N-1]*bdy_sfc[1]
    aa = aa.at[N].set(aa_sfc)
    bb = bb.at[N].set(bb_sfc)
    rhs = rhs.at[N].set(rhs_sfc)

    # bottom boundary condition
    cond_bot = bdy_bot[0] < 0.5
    bb_bot = cond_bot*1. + (1-cond_bot)*-0.5*(Ak_tke[0] + Ak_tke[1])
    cc_bot = cond_bot*0. + (1-cond_bot)*0.5*(Ak_tke[0] + Ak_tke[1])
    rhs_bot = cond_bot*bdy_bot[1] + (1-cond_bot)*Hz[0]*bdy_bot[1]
    bb = bb.at[0].set(bb_bot)
    cc = cc.at[0].set(cc_bot)
    rhs = rhs.at[0].set(rhs_bot)

    # Solve tridiagonal problem
    cff = 1.0 / bb[0]
    q = q.at[0].set(-cc[0]*cff)
    rhs = rhs.at[0].multiply(cff)
    def body_fun1(k, x):
        q = x[0, :]
        rhs = x[1, :]
        cff = 1.0 / (bb[k] + aa[k]*q[k-1])
        q = q.at[k].set(-cff*cc[k])
        rhs = rhs.at[k].set(cff*(rhs[k] - aa[k]*rhs[k-1]))
        return jnp.stack([q, rhs])
    q_rhs = jnp.stack([q, rhs])
    q_rhs = lax.fori_loop(1, N+1, body_fun1, q_rhs)
    q = q_rhs[0, :]
    rhs = q_rhs[1, :]

    body_fun2 = lambda k, x: x.at[N-k].set(x[N-k] + q[N-k]*x[N-k+1])
    rhs = lax.fori_loop(1, N+1, body_fun2, rhs)
    tke_np1 = jnp.maximum(rhs, tke_min)

    wtke = -0.5*(Ak_tke[1:] + Ak_tke[:N])*(tke_np1[1:] - tke_np1[:N])/Hz

    return tke_np1, wtke


@jit
def advance_turb_eps(eps_n, bvf, shear2, Ak_eps, c_mu, c_mu_prim, tke_n,
                     tke_np1, Hz, dt, beta, eps_min, bdy_sfc, bdy_bot):
    """
    No description.

    Parameters
    ----------
    eps_n : float(N+1)
        no description
    bvf : float(N+1)
        no description
    shear2 : float(N+1)
        no description
    Ak_eps : float(N+1)
        no description
    c_mu : float(N+1)
        no description
    c_mu_prim : float(N+1)
        no description
    tke_n : float(N+1)
        no description
    tke_np1 : float(N+1)
        no description
    Hz : float(N)
        no description
    dt : float
        no description
    beta : float(4)
        no description
    eps_min : float
        no description
    bdy_sfc : float(2)
        bdy_sfc(1) = 0. -> Dirichlet, bdy_sfc(1) = 1. -> Neumann
    bdy_bot : float(2)
        bdy_bot(1) = 0. -> Dirichlet, bdy_top(1) = 1. -> Neumann

    Returns
    -------
    eps_np1 : float(N+1)
        no description
    """
    # returned variables
    N = eps_n.shape[0]-1
    eps_np1 = jnp.zeros(N+1)

    # local variables
    aa = jnp.zeros(N+1)
    bb = jnp.zeros(N+1)
    cc = jnp.zeros(N+1)
    rhs = jnp.zeros(N+1)
    q = jnp.zeros(N+1)

    beta1, beta2, beta3m, beta3p = beta
    # fill in the tridiagonal matrix
    # Off-diagonal terms for the tridiagonal problem
    cff = -0.5*dt
    aa = aa.at[1:N].set(cff*(Ak_eps[1:N] + Ak_eps[:N-1]) / Hz[:N-1])
    cc = cc.at[1:N].set(cff*(Ak_eps[1:N] + Ak_eps[2:]) / Hz[1:])


    # Shear and buoyancy production
    Sprod = beta1*c_mu[1:N]*tke_n[1:N]*shear2[1:N]
    Bprod = -c_mu_prim[1:N]*tke_n[1:N]*(beta3m*jnp.maximum(bvf[1:N], 0) + \
                beta3p*jnp.minimum(bvf[1:N], 0))
    cff = 0.5*(Hz[:N-1] + Hz[1:])
    invG = 1.0 / eps_n[1:N]
    rhs_in = lax.select((Bprod + Sprod) > 0, cff*(eps_n[1:N] + dt*(Bprod + Sprod)), cff*(eps_n[1:N] + dt*Sprod))
    rhs = rhs.at[1:N].set(rhs_in)
    bb_in = lax.select((Bprod + Sprod) > 0, cff*(1.0 + dt*beta2*eps_n[1:N] / tke_np1[1:N]) - aa[1:N] - cc[1:N], cff*(1.0 + dt*beta2*eps_n[1:N] / tke_np1[1:N] - dt*invG*Bprod) - aa[1:N] - cc[1:N])
    bb = bb.at[1:N].set(bb_in)

    # surface boundary condition
    cond_sfc = bdy_sfc[0] < 0.5
    aa_sfc = cond_sfc*0. + (1-cond_sfc)*-0.5*(Ak_eps[N] + Ak_eps[N-1])
    bb_sfc = cond_sfc*1. + (1-cond_sfc)*0.5*(Ak_eps[N] + Ak_eps[N-1])
    rhs_sfc = cond_sfc*bdy_sfc[1] + (1-cond_sfc)*Hz[N-1]*bdy_sfc[1]
    aa = aa.at[N].set(aa_sfc)
    bb = bb.at[N].set(bb_sfc)
    rhs = rhs.at[N].set(rhs_sfc)

    # bottom boundary condition
    cond_bot = bdy_bot[0] < 0.5
    bb_bot = cond_bot*1. + (1-cond_bot)*-0.5*(Ak_eps[0] + Ak_eps[1])
    cc_bot = cond_bot*0. + (1-cond_bot)*0.5*(Ak_eps[0] + Ak_eps[1])
    rhs_bot = cond_bot*bdy_bot[1] + (1-cond_bot)*Hz[0]*bdy_bot[1]
    bb = bb.at[0].set(bb_bot)
    cc = cc.at[0].set(cc_bot)
    rhs = rhs.at[0].set(rhs_bot)

    # Solve tridiagonal problem
    cff = 1.0 / bb[0]
    q = q.at[0].set(-cc[0]*cff)
    rhs = rhs.at[0].multiply(cff)
    def body_fun1(k, x):
        q = x[0, :]
        rhs = x[1, :]
        cff = 1.0 / (bb[k] + aa[k]*q[k-1])
        q = q.at[k].set(-cff*cc[k])
        rhs = rhs.at[k].set(cff*(rhs[k] - aa[k]*rhs[k-1]))
        return jnp.stack([q, rhs])
    q_rhs = jnp.stack([q, rhs])
    q_rhs = lax.fori_loop(1, N+1, body_fun1, q_rhs)
    q = q_rhs[0, :]
    rhs = q_rhs[1, :]

    body_fun2 = lambda k, x: x.at[N-k].set(x[N-k] + q[N-k]*x[N-k+1])
    rhs = lax.fori_loop(1, N+1, body_fun2, rhs)

    eps_np1 = jnp.maximum(rhs, eps_min)

    return eps_np1