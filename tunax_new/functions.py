"""
Usefull calculation functions.

The functions in this module are supposed to be used in various other modules.
They can be called by the prefix :code:`tunax.functions.` or directly by
:code:`tunax.`.

"""

from typing import Tuple

import jax.numpy as jnp
from jax import lax
from jaxtyping import Float, Array


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
    def forward_scan_scal(carry: Tuple[float, float], x: jnp.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        f_im1, q_im1 = carry
        a, b, c, f = x
        cff = 1./(b+a*q_im1)
        f_i = cff*(f-a*f_im1)
        q_i = -cff*c
        carry = f_i, q_i
        return carry, carry
    init = f[0]/b[0], -c[0]/b[0]
    xs = jnp.stack([a, b, c, f])[:, 1:].T
    _, (f, q) = lax.scan(forward_scan_scal, init, xs)
    f = jnp.concat([jnp.array([init[0]]), f])
    q = jnp.concat([jnp.array([init[1]]), q])

    def reverse_scan_scal(carry: float, x: jnp.ndarray) -> Tuple[float, float]:
        q_rev, f_rev = x
        carry = f_rev + q_rev*carry
        return carry, carry
    init = f[-1]
    xs = jnp.stack([q[::-1], f[::-1]])[:, 1:].T
    _, x = lax.scan(reverse_scan_scal, init, xs)
    x = jnp.concat([jnp.array([init]), x])

    return x[::-1]


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
