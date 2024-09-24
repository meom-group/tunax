import jax.numpy as jnp
from jax import jit, lax


@jit
def tridiag_solve(
        a: jnp.ndarray,
        b: jnp.ndarray,
        c: jnp.ndarray,
        f: jnp.ndarray
    ) -> jnp.ndarray:
    """
    Solve a trigiagonal problem MQ=F by recurrence.

    Parameters
    ----------
    a : jnp.ndarray, float(n)
        left diagonal of the matrix M
    b : jnp.ndarray, float(n)
        middle diagonal of the matrix M
    c : jnp.ndarray, float(n)
        right diagonal of the matrix M
    f : jnp.ndarray, float(n)
        right hand of the equation
    
    Returns
    -------
    f : jnp.ndarray, float(n)
        solution of tridiagonal problem
    """
    n, = a.shape
    # forward sweep
    cff = 1.0 / b[0]
    f = f.at[0].multiply(cff)
    q = jnp.zeros(n)
    q = q.at[0].set(-c[0] * cff)
    
    def body_fun1(k, x):
        f = x[0, :]
        q = x[1, :]
        cff = 1.0 / (b[k] + a[k] * q[k-1])
        q = q.at[k].set(-cff * c[k])
        f = f.at[k].set(cff * (f[k] - a[k] * f[k-1]))
        return jnp.stack([f, q])
    f_q = jnp.stack([f, q])
    f_q = lax.fori_loop(1, n, body_fun1, f_q)
    f = f_q[0, :]
    q = f_q[1, :]

    # backward substitution
    body_fun2 = lambda k, x: x.at[n-1-k].add(q[n-1-k] * x[n-k])
    f = lax.fori_loop(1, n, body_fun2, f)
    
    return f


@jit
def add_boundaries(
        vec_btm: float,
        vec_in: jnp.ndarray,
        vec_sfc: float
    ) -> jnp.ndarray:
    """
    Concatenate the three parts of a vector : surface, bottom and inside

    Parameters
    ----------
    vec_btm : float
        bottom value of the vector
    vec_in : jnp.ndarray, float(n)
        middle values of the vector
    vec_sfc : float
        surface value of the vector

    Returns
    -------
    vec : jnp.ndarray, float(n+2)
        concatenated vector
    """
    return jnp.concat([jnp.array([vec_btm]), vec_in, jnp.array([vec_sfc])])


def format_to_single_line(text: str) -> str:
    """
    Transforms a multiple line text in a line string by removing indentations.

    Parameters
    ----------
    text : str
        text on multiple lines to transform

    Returns
    -------
    line : str
        text on a single line removed from indentations
    """
    lines = text.splitlines()
    stripped_lines = [line.strip() for line in lines]
    single_line = " ".join(stripped_lines)
    return " ".join(single_line.split())
