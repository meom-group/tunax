"""
Usefull functions for the single column model
"""

import jax.numpy as jnp
from jax import jit, lax

grav = 9.81  # Gravity of Earth
cp = 3985.0  # Specific heat capacity of saltwater [J/kg K]

@jit
def advance_tra_ed(t_n, s_n, stflx, srflx, swr_frac, btflx, Hz, Akt, zw, eps, alpha,
                   dt):
    r"""
    Integrate vertical diffusion term for tracers.

    Parameters
    ----------
    t_n : float(N, ntra)
        temperature at time step n
    s_n : float(N, ntra)
        salinity at time step n
    stflx : float(ntra)
        surface tracer fluxes
    srflx : float
        surface radiative flux [W/m2]
    swr_frac : float(N+1)
        fraction of solar penetration
    btflx : float(ntra)
        surface tracer fluxes
    Hz : float(N)
        layer thickness [m]
    Akt : float(N+1)
        eddy-diffusivity [m2/s]
    zw : float(N+1)
        depth at cell interfaces [m]
    eps : float(N+1)
        TKE dissipation [m2/s3]
    alpha : float
        thermal expension coefficient [C-1]
    dt : float
        time-step [s]

    Returns
    -------
    t_np1 : float(N, ntra)
        temperature at time step n+1
    s_np1 : float(N, ntra)
        salinity at time step n+1

    Notes
    -----
    for the vectorized version, ntra should be equal to 2
    \[ \overline{\phi}^{n+1,*} = \overline{\phi}^n + \Delta t \partial_z \left(
    K_m \partial_z  \overline{\phi}^{n+1,*} \right) \]
    """
    # returned variables
    N, = t_n.shape
    temp = jnp.zeros(N)
    sal = jnp.zeros(N)

    # local variables
    fc = jnp.zeros(N+1)

    # 1 - Compute fluxes associated with solar penetration and surface boundary
    # condition
    # 1.1 - temperature
    # surface heat flux (including latent and solar components)
    fc = fc.at[N].set(stflx[0] + srflx)
    # penetration of solar heat flux
    fc = fc.at[1:N].set(srflx * swr_frac[1:N])
    # apply flux divergence
    temp = Hz*t_n + dt*(fc[1:] - fc[:-1])
    cffp = eps[1:] / (cp - alpha * grav * zw[1:])
    cffm = eps[:-1] / (cp - alpha * grav * zw[:-1])
    temp = temp + dt * 0.5 * Hz * (cffp + cffm)
    # 1.2 - salinity
    fc = fc.at[N].set(stflx[1]) # Salinity (fresh water flux)
    fc = fc.at[1:N].set(0.0)
    # apply flux divergence
    sal = Hz*s_n + dt*(fc[1:] - fc[:-1])

    # 2 - Implicit integration for vertical diffusion
    # 1.1 - temperature
    # right hand side for the tridiagonal problem
    temp = temp.at[0].add(-dt * btflx[0])
    # solve tridiagonal problem
    temp = tridiag_solve(Hz, Akt, temp, dt)
    # 1.2 - salinity
    # right hand side for the tridiagonal problem
    sal = sal.at[0].add(-dt * btflx[1])
    # solve tridiagonal problem
    sal = tridiag_solve(Hz, Akt, sal, dt)

    return temp, sal


@jit
def advance_dyn_cor_ed(u_n, v_n, ustr_sfc, vstr_sfc, ustr_bot, vstr_bot, Hz,
                       Akv, fcor, dt):
    r"""
    Integrate vertical viscosity and Coriolis terms for dynamics.

    Parameters
    ----------
    u_n : float(N)
        u-velocity component at time n [m/s]
    v_n : float(N)
        v-velocity component at time n [m/s]
    ustr_sfc : float
        zonal surface stress      [m2/s2]
    vstr_sfc : float
        meridional surface stress [m2/s2]
    ustr_bot : float
        zonal surface stress      [m2/s2]
    vstr_bot : float
        meridional surface stress [m2/s2]
    Hz : float(N)
        layer thickness [m]
    Akv : float(N+1)
        eddy-viscosity [m2/s]
    fcor : float
        Coriolis frequaency [s-1]
    dt : float
        time-step [s]

    Returns
    -------
    u_np1 : float(N)
        u-velocity component at time n+1 [m/s]
    v_np1 : float(N)
        v-velocity component at time n+1 [m/s]

    Notes
    -----
    1 - Compute Coriolis term
    if n is even
    \begin{align*}
    u^{n+1,\star} &= u^n + \Delta t f v^n \\
    v^{n+1,\star} &= v^n - \Delta t f u^{n+1,\star}
    \end{align*}
    if n is odd
    \begin{align*}
    v^{n+1,\star} &= v^n - \Delta t f u^n \\
    u^{n+1,\star} &= u^n + \Delta t f v^{n+1,\star}
    \end{align*}

    2 - Apply surface and bottom forcing

    3 - Implicit integration for vertical viscosity
    \begin{align*}
    \mathbf{u}^{n+1,\star \star} &= \mathbf{u}^{n+1,\star} + \Delta t
    \partial_z \left(  K_m \partial_z  \mathbf{u}^{n+1,\star \star} \right)  \\
    \end{align*}
    """
    # returned variables
    N, = u_n.shape
    u_np1 = jnp.zeros(N)
    v_np1 = jnp.zeros(N)

    # local variables
    gamma_Cor = 0.55

    # 1 - Compute Coriolis term
    cff = (dt * fcor) ** 2
    cff1 = 1 / (1 + gamma_Cor * gamma_Cor * cff)
    u_np1 = cff1 * Hz * ((1-gamma_Cor*(1-gamma_Cor)*cff)*u_n + dt*fcor*v_n)
    v_np1 = cff1 * Hz * ((1-gamma_Cor*(1-gamma_Cor)*cff)*v_n - dt*fcor*u_n)

    # 2 - Apply surface and bottom forcing
    u_np1 = u_np1.at[-1].add(dt * ustr_sfc) # sustr is in m2/s2 here
    v_np1 = v_np1.at[-1].add(dt * vstr_sfc)
    u_np1 = u_np1.at[0].add(-dt * ustr_bot) # sustr is in m2/s2 here
    v_np1 = v_np1.at[0].add(-dt * vstr_bot)

    # 3 - Implicit integration for vertical viscosity
    # u-component
    u_np1 = tridiag_solve(Hz, Akv, u_np1, dt) # invert tridiagonal matrix
    # v-component
    v_np1 = tridiag_solve(Hz, Akv, v_np1, dt) # invert tridiagonal matrix

    return u_np1, v_np1


@jit
def compute_evd(bvf, Akv, Akt, AkEvd):
    """
    Compute enhanced vertical diffusion/viscosity where the density profile is
    unstable.

    Parameters
    ----------
    bvf : float(N+1)
        Brunt Vaisala frequency [s-2]
    Akv : float(N+1) [modified]
        eddy-viscosity [m2/s]
    Akt : float(N+1) [modified]
        eddy-diffusivity [m2/s]
    AkEvd : float
        value of enhanced diffusion [m2/s]
    
    Returns
    -------
    Akv : float(N+1) [modified]
        eddy-viscosity [m2/s]
    Akt : float(N+1) [modified]
        eddy-diffusivity [m2/s]
    """
    Akv = jnp.where(bvf<=-1e-12, AkEvd, Akv)
    Akt = jnp.where(bvf<=-1e-12, AkEvd, Akt)
    
    return Akv, Akt


@jit
def lmd_swfrac(Hz):
    """
    Compute fraction of solar shortwave flux penetrating to specified depth due
    to exponential decay in Jerlov water type.

    Parameters
    ----------
    Hz : float(N)
        layer thickness [m]

    Returns
    -------
    swr_frac : float(N+1)
        fraction of solar penetration
    """
    # returned variables
    N, = Hz.shape
    mu1 = 0.35
    mu2 = 23.0
    r1 = 0.58
    attn1 = -1.0 / mu1
    attn2 = -1.0 / mu2

    xi1 = attn1 * Hz
    xi2 = attn2 * Hz

    def step(sdwk, k):
        sdwk1, sdwk2 = sdwk
        sdwk1 = lax.cond(xi1[N-k] > -20, lambda x: x*jnp.exp(xi1[N-k]),
                             lambda x: 0.*x, sdwk1)
        sdwk2 = lax.cond(xi2[N-k] > -20, lambda x: x*jnp.exp(xi2[N-k]),
                             lambda x: 0.*x, sdwk2)
        return (sdwk1, sdwk2), sdwk1+sdwk2
    
    _, swr_frac = lax.scan(step, (r1, 1.0 - r1), jnp.arange(1, N+1))
    return jnp.concat((swr_frac[::-1], jnp.array([1])))


@jit
def compute_mxl(bvf, rhoc, zr, zref, rhoRef):
    """
    Compute mixed layer depth.

    Parameters
    ----------
    bvf : float(N+1)
        Brunt Vaisala frequancy [s-2]
    rhoc : float
        thermal expension coefficient [kg m-3]
    zr : float(N)
        depth at cell center [m]
    zref : float
        no description
    rhoRef : float
        no description

    Returns
    -------
    hmxl : float
        mixed layer depth [m]
    """
    N = bvf.shape[0]-1
    # find the ref depth index
    kstart = N - 1
    cond_fun1 = lambda val: val[1][val[0]] > zref
    body_fun1 = lambda val: (val[0]-1, val[1])
    kstart, _ = lax.while_loop(cond_fun1, body_fun1, (N-1, zr))

    bvf_c = rhoc * (grav / rhoRef)
    # initialize at the near bottom value
    hmxl = zr[kstart]

   
    cond_fun2 = lambda val: ((val[0]>0) & (val[1] < bvf_c))
    def body_fun2(val):
        (k, cff_k, cff_km1, bvf, zr) = val
        cff_new = cff_k + jnp.maximum(bvf[k], 0.0) * (zr[k]-zr[k-1])
        cff_km1 = cff_k
        cff_k = cff_new
        k -= 1
        return (k, cff_k, cff_km1, bvf, zr)

    (k, cff_k, cff_km1, _, _) = lax.while_loop(cond_fun2, body_fun2,
                                               (kstart, 0., 0., bvf, zr))

    hmxl_new = ((cff_k-bvf_c)*zr[k+1] + (bvf_c-cff_km1)*zr[k]) / \
        (cff_k - cff_km1)
    hmxl = lax.select(cff_k >= bvf_c, hmxl_new, hmxl)
        
    return hmxl


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


@jit
def rho_eos(temp, salt, zr, zw, rhoRef):
    """Compute density anomaly and Brunt Vaisala frequency via nonlinear
    Equation Of State (EOS).

    Parameters
    ----------
    temp : float(N)
        temperature [C]
    salt : float(N)
        salinity [psu]
    zr : float(N)
        depth at cell centers [m]
    zw : float(N+1)
        depth at cell interfaces [m]
    rhoRef : float
        no description

    Returns
    -------
    bvf : float(N+1)
        Brunt Vaisala frequancy [s-2]
    rho : float(N)
        density anomaly [kg/m3]
    """
    # returned variables
    N, = temp.shape
    bvf = jnp.zeros(N+1)
    rho = jnp.zeros(N)

    # local variables
    rho1 = jnp.zeros(N)
    K_up = jnp.zeros(N)
    K_dw = jnp.zeros(N)

    # constants
    r00, r01, r02, r03, r04, r05 = 999.842594, 6.793952E-2, -9.095290E-3, \
        1.001685E-4, -1.120083E-6, 6.536332E-9
    r10, r11, r12, r13, r14, r20 = 0.824493, -4.08990E-3, 7.64380E-5, \
        -8.24670E-7, 5.38750E-9, 4.8314E-4
    rS0, rS1, rS2 = -5.72466E-3, 1.02270E-4, -1.65460E-6
    k00, k01, k02, k03, k04 = 19092.56, 209.8925, -3.041638, -1.852732e-3, \
        -1.361629e-5
    k10, k11, k12, k13 = 104.4077, -6.500517, 0.1553190, 2.326469e-4
    ks0, ks1, ks2 = -5.587545, 0.7390729, -1.909078e-2
    b00, b01, b02, b03, b10, b11, b12, bs1 = 0.4721788, 0.01028859, \
        -2.512549e-4, -5.939910e-7, -0.01571896, -2.598241e-4, 7.267926e-6, \
        2.042967e-3
    e00, e01, e02, e10, e11, e12 = 1.045941e-5, -5.782165e-10, 1.296821e-7, \
        -2.595994e-7, -1.248266e-9, -3.508914e-9

    dr00 = r00 - rhoRef

    # density anomaly
    sqrtTs = jnp.sqrt(salt)
    rho1 = dr00 + temp*(r01 + temp*(r02 + temp*(r03 + temp*(r04 + temp*r05))))\
        + salt*(r10 + temp*(r11 + temp*(r12 + temp*(r13 + temp*r14))) +
        sqrtTs*(rS0 + temp*(rS1 + temp*rS2)) + salt*r20)


    k0 = temp*(k01 + temp*(k02 + temp*(k03 + temp*k04))) + \
        salt*(k10 + temp*(k11 + temp*(k12 + temp*k13)) +
        sqrtTs*(ks0 + temp*(ks1 + temp*ks2)))

    k1 = b00 + temp*(b01 + temp*(b02 + temp*b03)) + \
        salt*(b10 + temp*(b11 + temp*b12) + sqrtTs*bs1)
    k2 = e00 + temp*(e01 + temp*e02) + salt*(e10 + temp*(e11 + temp*e12))
    
    dpth = -zr
    cff = k00 - 0.1*dpth
    cff1 = k0 + dpth*(k1 + k2*dpth)
    rho = (rho1*cff*(k00 + cff1) - 0.1*dpth*rhoRef*cff1) / (cff*(cff + cff1))

    K_up = k0 - zw[1:]*(k1 - k2*zw[1:])
    K_dw = k0 - zw[:-1]*(k1 - k2*zw[:-1])

    cff = grav / rhoRef
    cff1 = -0.1*zw[1:-1]
    bvf = -cff*((rho1[1:]-rho1[:-1]) * (k00+K_dw[1:]) * (k00+K_up[:-1]) - cff1*(rhoRef*(K_dw[1:] - K_up[:-1]) + k00*(rho1[1:] - rho1[:-1]) + rho1[1:]*K_dw[1:] - rho1[:-1]*K_up[:-1])) / ((k00 + K_dw[1:] - cff1)*(k00 + K_up[:-1] - cff1)*(zr[1:] - zr[:-1]))

    bvf = jnp.concat((jnp.array([0.]), bvf, jnp.array([0])))

    return rho, bvf


@jit
def tridiag_solve(Hz, Ak, f, dt):
    """
    Solve the tridiagonal problem associated with the implicit in time
    treatment of vertical diffusion/viscosity.

    Parameters
    ----------
    Hz : float(N)
        layer thickness [m]
    Ak : float(N+1)
        eddy diffusivity/viscosity [m2/s]
    f : float(N) [modified]
        (in: right-hand side) (out:solution of tridiagonal problem)
    dt : float
        time-step [s]
    
    Returns
    -------
    f : float(N) [modified]
        (in: right-hand side) (out:solution of tridiagonal problem)
    """
    # local variables
    N, = Hz.shape
    a = jnp.zeros(N)
    b = jnp.zeros(N)
    c = jnp.zeros(N)
    q = jnp.zeros(N)
     
    # fill the coefficients for the tridiagonal matrix
    difA = -2.0 * dt * Ak[1:N-1] / (Hz[:N-2] + Hz[1:N-1])
    difC = -2.0 * dt * Ak[2:N] / (Hz[2:N] + Hz[1:N-1])
    a = a.at[1:N-1].set(difA)
    c = c.at[1:N-1].set(difC)
    b = b.at[1:N-1].set(Hz[1:N-1] - difA - difC)

    # bottom boundary condition
    a = a.at[0].set(0.0)
    difC = -2.0 * dt * Ak[1] / (Hz[1] + Hz[0])
    c = c.at[0].set(difC)
    b = b.at[0].set(Hz[0] - difC)

    # surface boundary condition
    difA = -2.0 * dt * Ak[N-1] / (Hz[N-2] + Hz[N-1])
    a = a.at[N-1].set(difA)
    c = c.at[N-1].set(0.0)
    b = b.at[N-1].set(Hz[N-1] - difA)

    # forward sweep
    cff = 1.0 / b[0]
    q = q.at[0].set(-c[0] * cff)
    f = f.at[0].multiply(cff)
    
    def body_fun1(k, x):
        f = x[0, :]
        q = x[1, :]
        cff = 1.0 / (b[k] + a[k] * q[k-1])
        q = q.at[k].set(-cff * c[k])
        f = f.at[k].set(cff * (f[k] - a[k] * f[k-1]))
        return jnp.stack([f, q])
    f_q = jnp.stack([f, q])
    f_q = lax.fori_loop(1, N, body_fun1, f_q)
    f = f_q[0, :]
    q = f_q[1, :]

    # backward substitution
    body_fun2 = lambda k, x: x.at[N-2-k].add(q[N-2-k] * x[N-1-k])
    f = lax.fori_loop(0, N-1, body_fun2, f)
    
    return f
