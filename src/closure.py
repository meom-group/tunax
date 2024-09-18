"""
Abstract classes to define closures
"""

import equinox as eqx
import jax.numpy as jnp
from abc import ABC
from typing import Callable, Type, TypeVar, Generic

from case import Case
from state import Grid, State


class ClosureParametersAbstract(eqx.Module, ABC):
    pass

CloParType = TypeVar('CloParType', bound=ClosureParametersAbstract)

class ClosureStateAbstract(eqx.Module, ABC):
    """
    Abstract class basis for the closure states.

    Parameters
    ----------
    grid : Grid
        spatial grid
    akt : jnp.ndarray, float(nz+1)
        eddy-diffusivity [m2.s-1]
    akv : jnp.ndarray, float(nz+1)
        eddy-viscosity [m2.s-1]
    eps : jnp.ndarray, float(nz+1)
        TKE dissipation [m2.s-3]

    Notes
    -----
    For a closure, its state class defines the state of the water column for
    the usefull variables of the closure. The variables `akt`, `akv` and `eps`
    are mandatory because they are used in the tracer and velocity integration.

    """
    
    grid: Grid
    akt: jnp.ndarray
    akv: jnp.ndarray
    eps: jnp.ndarray

CloStateType = TypeVar('CloStateType', bound=ClosureStateAbstract)

class Closure(eqx.Module, Generic[CloStateType, CloParType]):
    parameters_class: Type[ClosureParametersAbstract]
    state_class: Type[ClosureStateAbstract]
    step_fun: Callable[[State, CloStateType, CloParType, Case], CloStateType]