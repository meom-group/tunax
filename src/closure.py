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
    grid: Grid
    akt: jnp.ndarray
    akv: jnp.ndarray
    eps: jnp.ndarray
    # ces trois variables sont utilisées pour avancer vitesse et traceurs donc nécessaires

CloStateType = TypeVar('CloStateType', bound=ClosureStateAbstract)

class Closure(eqx.Module, Generic[CloStateType, CloParType]):
    parameters_class: Type[ClosureParametersAbstract]
    state_class: Type[ClosureStateAbstract]
    step_fun: Callable[[State, CloStateType, CloParType, Case], CloStateType]