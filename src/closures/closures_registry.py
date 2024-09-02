"""
Registry of all the closures and abstract classes for defining them
"""

import equinox as eqx
from abc import ABC, abstractmethod
from typing import Type, Callable, Tuple
from model import State


class ClosureParametersAbstract(eqx.Module, ABC):
    pass

class ClosureStateAbstract(eqx.Module, ABC):

    @classmethod
    @abstractmethod
    def gen_init_state(cls):
        pass

class Closure(eqx.Module):
    parameters_class: Type[ClosureParametersAbstract]
    state_class: Type[ClosureStateAbstract]
    step_fun: Callable[[State, ClosureStateAbstract, ClosureParametersAbstract], Tuple[State, ClosureStateAbstract]]



# maybe a better way to do the registry, class and methods to add one ?
# how to deal with the default closures ?

from k_epsilon import KepsParameters, KepsState, keps_step


closure_registry = {
    'k-epsilon': Closure(KepsParameters, KepsState, keps_step)
}