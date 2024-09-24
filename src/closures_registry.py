"""
Registry of closures
"""

from typing import Dict

from closure import Closure

from closures.k_epsilon import KepsParameters, KepsState, keps_step
from closures.tke import TkeParameters, TkeState, tke_step

CLOSURES_REGISTRY: Dict[str, Closure] = {
    'k-epsilon': Closure(KepsParameters, KepsState, keps_step),
    'tke': Closure(TkeParameters, TkeState, tke_step)
}