"""
This module reads and transforms the observation data in a python object that
the module of calibration can understand.
"""

import equinox as eqx
import xarray as xr
import json
from typing import List, Any
import numpy as np


DIM_NAME_LIST = ['t', 'z']
VAR_NAME_LIST = ['u', 'v', 'temp', 'salt']

class Obs(eqx.Module):
    """
    Represent one observation (LES or measures). It corresponds to one file.
    
    Attributes
    ----------
     : xarray.Dataset
        represent the normalized variables of the file (normalized in the meaning
        of the variable names formality)
    parameters : 
    metadatas : Any
        not described yet
    """
    variables: xr.Dataset
    parameters: Any
    metadatas: Any

def nc_to_obs(nc_filename: str, des_filename: str) -> Obs:
    """
    Create a Obs object from a netCDF file. It reads the file, then it
    normalizes the vairable names.

    Parameters
    ----------
    nc_filename : str
        path and filename of the netCDF file, from the current directory and
        with the ".nc" extension
    des_filename : str
        path and filename of the description of the netCDF file. It is as
        .json file which indicates where to find every information in the nc file.

    Returns
    -------
    observation : Obs
        Obs object created from the netCDF file
    """
    ds = xr.open_dataset(nc_filename)

    variables = {}
    coords = {}

    description = json.load(open(des_filename, 'r'))
    for dim_name in DIM_NAME_LIST:
        params = description[dim_name]
        type = params['type']
        if type == 'create':
            raise ValueError("Can't create a dimension")
        else:
            arr = get_nc_var(ds, dim_name, params)
            coords[dim_name] = arr

    for var_name in VAR_NAME_LIST:
        params = description[var_name]
        arr = get_nc_var(ds, var_name, params, (coords['t'].size, coords['z'].size))
        variables[var_name] = (('t', 'z'), arr)
    
    norm_variables = xr.Dataset(variables, coords=coords)

    return Obs(variables=norm_variables, parameters=Any, metadatas=None)

def get_nc_var(ds: xr.Dataset, var_name: str, params: Any, shape = None) -> np.ndarray:
    match params['type']:
        case 'copy':
            nc_var_name = params['variable_name']
            return ds[nc_var_name].values
        case 'transormation':
            nc_var_name = params['variable_name']
            mul_factor = params['mul_factor']
            add_factor = params['add_factor']
            return ds[nc_var_name].values*mul_factor+add_factor
        case 'create':
            return np.full(shape, params['const_value'])
        case _:
            raise ValueError('Entry "type" should be copy, transformation or create')



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
    metadatas: Any