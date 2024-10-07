"""
This module reads and transforms the observation data in a python object that
the module of calibration can understand.
"""

from __future__ import annotations
import yaml
import equinox as eqx
import xarray as xr
import jax.numpy as jnp
from typing import List, Dict

from state import Trajectory, Grid
from case import Case

DIM_NAME_LIST = ['t', 'z']
VAR_NAME_LIST = ['u', 'v', 'temp', 'salt']

class Obs(eqx.Module):
    trajectory: Trajectory
    case: Case

    @classmethod
    def from_files(
            cls,
            nc_path: str,
            yaml_path: str,
            var_names: Dict[str, str]
        ) -> Obs:
        ds = xr.load_dataset(nc_path)
        # dimensions
        zr = jnp.array(ds[var_names['zr']].values)
        zw = jnp.array(ds[var_names['zw']].values)
        grid = Grid(zr, zw)
        time = jnp.array(ds[var_names['time']].values)
        nt, = time.shape
        nz = grid.nz
        # variables
        t_name = var_names['t']
        if t_name == '':
            t = jnp.full((nt, nz), 21.)
        else:
            t = jnp.array(ds[var_names['t']].values)
        s_name = var_names['s']
        if s_name == '':
            s = jnp.full((nt, nz), 35.)
        else:
            s = jnp.array(ds[var_names['s']].values)
        u_name = var_names['u']
        if u_name == '':
            u = jnp.full((nt, nz), 0.)
        else:
            u = jnp.array(ds[var_names['u']].values)
        v_name = var_names['v']
        if v_name == '':
            v = jnp.full((nt, nz), 0.)
        else:
            v = jnp.array(ds[var_names['v']].values)
        # writing trajectory
        trajectory = Trajectory(grid, time, t, s, u, v)

        with open(yaml_path, 'r') as f:
            metadatas = yaml.safe_load(f)
        
        case = Case()
        case_attributes = list(vars(case).keys())
        for att in case_attributes:
            if att in var_names.keys():
                case = eqx.tree_at(
                    lambda t: getattr(t, att), case,
                    metadatas[var_names[att]])

        return cls(trajectory, case)
        



class ObsSet(eqx.Module):
    """
    Represent a set of many observations with eventually different time and
    space scaling.

    Attributes
    ----------
    observations : List[Obs]
        represent the set of files that will be used for the scaling
    metadatas : Any
        not described yet
    """
    observations: List[Obs]