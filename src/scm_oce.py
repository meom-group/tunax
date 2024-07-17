import jax.numpy as jnp
from jax import lax, jit

from grid import Grid

grav = 9.81

@jit
def lmd_swfrac(hz: jnp.ndarray):
    """
    Compute fraction of solar shortwave flux penetrating to specified depth due
    to exponential decay in Jerlov water type.

    Parameters
    ----------
    hz : float(nz)
        layer thickness [m]

    Returns
    -------
    swr_frac : float(nz+1)
        fraction of solar penetration
    """
    # returned variables
    nz, = hz.shape
    mu1 = 0.35
    mu2 = 23.0
    r1 = 0.58
    attn1 = -1.0 / mu1
    attn2 = -1.0 / mu2

    xi1 = attn1 * hz
    xi2 = attn2 * hz

    def step(sdwk, k):
        sdwk1, sdwk2 = sdwk
        sdwk1 = lax.cond(xi1[nz-k] > -20, lambda x: x*jnp.exp(xi1[nz-k]),
                             lambda x: 0.*x, sdwk1)
        sdwk2 = lax.cond(xi2[nz-k] > -20, lambda x: x*jnp.exp(xi2[nz-k]),
                             lambda x: 0.*x, sdwk2)
        return (sdwk1, sdwk2), sdwk1+sdwk2
    
    _, swr_frac = lax.scan(step, (r1, 1.0 - r1), jnp.arange(1, nz+1))
    return jnp.concat((swr_frac[::-1], jnp.array([1])))

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
def tridiag_solve(hz, akt, f: jnp.ndarray, dt):
    """
    Solve the tridiagonal problem associated with the implicit in time
    treatment of vertical diffusion/viscosity.

    Parameters
    ----------
    hz : float(nz)
        layer thickness [m]
    akt : float(nz+1)
        eddy diffusivity/viscosity [m2/s]
    f : float(nz) [modified]
        (in: right-hand side) (out:solution of tridiagonal problem)
    dt : float
        time-step [s]
    
    Returns
    -------
    f : float(nz) [modified]
        (in: right-hand side) (out:solution of tridiagonal problem)
    """
    # local variables
    nz, = hz.shape
    a = jnp.zeros(nz)
    b = jnp.zeros(nz)
    c = jnp.zeros(nz)
    q = jnp.zeros(nz)
     
    # fill the coefficients for the tridiagonal matrix
    difA = -2.0 * dt * akt[1:nz-1] / (hz[:nz-2] + hz[1:nz-1])
    difC = -2.0 * dt * akt[2:nz] / (hz[2:nz] + hz[1:nz-1])
    a = a.at[1:nz-1].set(difA)
    c = c.at[1:nz-1].set(difC)
    b = b.at[1:nz-1].set(hz[1:nz-1] - difA - difC)

    # bottom boundary condition
    a = a.at[0].set(0.0)
    difC = -2.0 * dt * akt[1] / (hz[1] + hz[0])
    c = c.at[0].set(difC)
    b = b.at[0].set(hz[0] - difC)

    # surface boundary condition
    difA = -2.0 * dt * akt[nz-1] / (hz[nz-2] + hz[nz-1])
    a = a.at[nz-1].set(difA)
    c = c.at[nz-1].set(0.0)
    b = b.at[nz-1].set(hz[nz-1] - difA)

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
    f_q = lax.fori_loop(1, nz, body_fun1, f_q)
    f = f_q[0, :]
    q = f_q[1, :]

    # backward substitution
    body_fun2 = lambda k, x: x.at[nz-2-k].add(q[nz-2-k] * x[nz-1-k])
    f = lax.fori_loop(0, nz-1, body_fun2, f)
    
    return f