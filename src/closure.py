"""
Abstract classes to define closures
"""

import equinox as eqx
from abc import ABC, abstractmethod
from typing import Callable, Tuple, Type, TypeVar, Generic

from grid import Grid
from case import Case
from state import State


class ClosureParametersAbstract(eqx.Module, ABC):
    pass

CloParType = TypeVar('CloParType', bound=ClosureParametersAbstract)

class ClosureStateAbstract(eqx.Module, ABC):
    grid: Grid

CloStateType = TypeVar('CloStateType', bound=ClosureStateAbstract)

class Closure(eqx.Module, Generic[CloStateType, CloParType]):
    parameters_class: Type[ClosureParametersAbstract]
    state_class: Type[ClosureStateAbstract]
    step_fun: Callable[[State, CloStateType, CloParType, Case], Tuple[State, CloStateType]]