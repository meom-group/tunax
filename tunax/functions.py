"""
Usefull calculation functions.

Thefunctions in this module are supposed to be used in various other modules.
They can be called by the prefix :code:`tunax.functions.` or directly by
:code:`tunax.`.

"""

import jax.numpy as jnp
from jax import lax, jit
from jaxtyping import Float, Array


@jit
def tridiag_solve(
        a: Float[Array, 'n'],
        b: Float[Array, 'n'],
        c: Float[Array, 'n'],
        f: Float[Array, 'n']
    ) -> Float[Array, 'n']:
    r"""
    Solve a trigiagonal problem.

    The tridiagonal problem can be written :math:`\mathbb MX = F` where
    :math:`\mathbb M = \begin{pmatrix} b_1 & c_1 & & \\
    a_2 & \ddots & \ddots & \\
    & \ddots & \ddots & c_{n-1} \\
    & & a_n & b_n
    \end{pmatrix}`
    and :math:`F = \begin{pmatrix} f_1 \\ \vdots \\ f_n \end{pmatrix}`.
    The problem is solved by recurrence using :mod:`jax.lax` 


    Parameters
    ----------
    a : Float[~jax.Array, 'n']
        Left diagonal of :math:`\mathbb M`, the first element is not used.
    b : Float[~jax.Array, 'n']
        Middle diagonal of :math:`\mathbb M`.
    c : Float[~jax.Array, 'n']
        Right diagonal of :math:`\mathbb M`, the last element is not used.
    f : Float[~jax.Array, 'n']
        Right hand of the equation :math:`F`.
    
    Returns
    -------
    x : Float[~jax.Array, 'nz']
        Solution :math:`X` of tridiagonal problem.
    """
    n, = a.shape
    # forward sweep
    cff = 1.0 / b[0]
    f = f.at[0].multiply(cff)
    q = jnp.zeros(n)
    q = q.at[0].set(-c[0] * cff)

    def body_fun1(k: int, x: Float[Array, 'n']):
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
    def body_fun2(k: int, x: Float[Array, 'n']):
        return x.at[n-1-k].add(q[n-1-k] * x[n-k])
    x = lax.fori_loop(1, n, body_fun2, f)

    return x


def add_boundaries(
        vec_btm: float,
        vec_in: Float[Array, 'n-2'],
        vec_sfc: float
    ) -> Float[Array, 'n']:
    """
    Concatenate the three parts of a vector : surface, bottom and inside.

    This functions is made to avoid loops and make JAX more efficient by
    writing the calculations with vectorization when possible.

    Parameters
    ----------
    vec_btm : float
        Bottom value of the vector.
    vec_in : Float[~jax.Array, 'n-2']
        Middle values of the vector.
    vec_sfc : float
        Surface value of the vector.

    Returns
    -------

    vec : Float[~jax.Array, 'n']
        Concatenated vector.
    """
    return jnp.concat([jnp.array([vec_btm]), vec_in, jnp.array([vec_sfc])])


def _format_to_single_line(text: str) -> str:
    """
    Transforms a multiple line text in a line string by removing indentations.

    In the code the error and warning messages are written on multiple lines
    with indentations to respect the PEP8 maximum line length of 79 characters
    and the consistency of indentations. This function is used to show
    correctly these messages on one line and without the indentations.

    Parameters
    ----------
    text : str
        Text on multiple lines to transform.

    Returns
    -------
    line : str
        Text on a single line removed from indentations.
    """
    lines = text.splitlines()
    stripped_lines = [line.strip() for line in lines]
    single_line = " ".join(stripped_lines)
    return " ".join(single_line.split())
