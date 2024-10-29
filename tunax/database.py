"""
Abstraction for calibration databases.


This module the objects that are used in Tunax to describe a :class:`Database`
of observations used for a calibration. By *obersvations* (:class:`Obs`), we
refer to a set of time-series representing a physical experiment, a measurment
or a simulation like a Large Eddy Simulation (LES) for example. These classes
can be obtained by the prefix :code:`tunax.database.` or directly by
:code:`tunax.`.

"""

from __future__ import annotations
from typing import List, Dict

import yaml
import xarray as xr
import equinox as eqx
import jax.numpy as jnp

from tunax.space import Grid, Trajectory
from tunax.case import Case


class Obs(eqx.Module):
    """
    Abstraction to represent an *obersation*.

    The *observations* represent every elements of a database, each one
    represent a simulation or a measurement with their own time-series of
    variables and the physical case which is linked to them

    Parameters
    ----------
    trajectory : Trajectory
        cf. attribute.
    case : Case
        cf. attribute.

    Attributes
    ----------
    trajectory : Trajectory
        The time-series of the variables that represent this obervation.
    case : Case
        The physical case that represent this observation.

    Raises
    ------
    ValueError
        If the :attr:`~space.Trajectory.time` of :attr:`trajectory` is not
        build with constant time-steps.

    """

    trajectory: Trajectory
    case: Case

    def __init__(self, trajectory: Trajectory, case: Case):
        time = trajectory.time
        steps = time[1:] - time[:-1]
        if not jnp.all(steps == steps[0]):
            raise ValueError('Tunax only handle constant output time-steps')
        self.trajectory = trajectory
        self.case = case

    @classmethod
    def from_files(
            cls,
            nc_path: str,
            yaml_path: str,
            var_names: Dict[str, str]
        ) -> Obs:
        """
        Create an instance from a *netcdf* and a *yaml* files.

        This class method build a trajectory from the :code:`.nc` file
        :code:`nc_path`, it build the physical parameters from the
        configuration file :code:`yaml_path`. :code:`var_names` is used to do
        the link between Tunax name convention and the one from the used
        database.

        Parameters
        ----------
        nc_path : str
            Path of the *netcdf* file that contains the time-series of the
            observation trajectory. The file should contains at least the
            three dimensions :attr:`~space.Grid.zr` :attr:`~space.Grid.zw` and
            :attr:`~space.Trajectory.time`. The time-series can be created with
            default values if they are not present in the file. Otherwise, they
            must have the good dimensions described in
            :class:`~space.Trajectory`.
        yaml_path : str
            Path of the *yaml* file that contains the parameters and forcing
            that describe the observation. The parameters should be float
            numbers and directly accessible from the root of the file with
            a key. Only the parameters that are described in
            :class:`~case.Case` will be takend in account.
        var_names : Dict[str, str]
            Link between the convention names in Tunax and the ones in the
            database. The keys are the Tunax names and the values are the names
            in the database. It works for variables of the
            :class:`~space.Trajectory` and fornthe parameters of
            :class:`~case.Case`. It must at least contains entries for
            :attr:`~space.Grid.zr` :attr:`~space.Grid.zw` and
            :attr:`~space.Trajectory.time`
        
        Returns
        -------
        obs : Obs
            An object that represent these files as an observation.
        """
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

        with open(yaml_path, 'r', encoding='utf-8') as f:
            metadatas = yaml.safe_load(f)

        case = Case()
        case_attributes = list(vars(case).keys())
        for att in case_attributes:
            if att in var_names.keys():
                case = eqx.tree_at(
                    lambda t: getattr(t, att), case,
                    metadatas[var_names[att]])

        return cls(trajectory, case)


class Database(eqx.Module):
    """
    Represent a set of several observations that form a database.

    Parameters
    ----------
    observations : List[Obs]
        cf. attribute.

    Attributes
    ----------
    observations : List[Obs]
        A list of several observations with potentially various forcings,
        geometry and time configuration.


    """

    observations: List[Obs]
