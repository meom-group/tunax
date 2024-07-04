"""
"""

import equinox as eqx


class Case(eqx.Module):
    """
    Define the physical parameters of an experiment
    """
    flx: float
    dt: float