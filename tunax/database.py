"""
Abstraction for calibration databases.


This module the objects that are used in Tunax to describe a :class:`Database` of observations used
for a calibration. By *obersvations* (:class:`Obs`), we refer to a set of time-series representing a
physical experiment, a measurment or a simulation like a Large Eddy Simulation (LES) for example.
These classes can be obtained by the prefix :code:`tunax.database.` or directly by :code:`tunax.`.

"""

from __future__ import annotations
import warnings
from typing import Union, Optional, Tuple, List, Dict, Any

import yaml
import xarray as xr
import equinox as eqx
import numpy as np
import jax.numpy as jnp
from h5py import File as H5pyFile
from jaxtyping import Array, Float

from tunax.space import Grid, Trajectory, TRACERS_NAMES
from tunax.case import Case
from tunax.functions import _format_to_single_line
from tunax.model import SingleColumnModel


type DimsType = Tuple[Optional[int]]


def get_var_jl(
        jl_file: H5pyFile,
        var_names: Dict[str, str],
        var: str,
        n: int,
        dims: Union[DimsType, Dict[str, DimsType]] = (None),
        suffix: str = ''
    ) -> Float[Array, 'n']:
    """
    TO CHECK
    n : int
        le numéro des indices auxquels récupérer l'array 1D et un None pour la dim de cet array u
        alors un dictionnaire où on récupère le tuple à partir du nom de la variable
    suffix : str
        pour rajouter les temps
    """
    jl_var = jl_file[f'{var_names[var]}{suffix}']
    # sélection éventuelle du bon tuple de dims (cas où on donne un
    # dictionnaire de dims pour chaque varialbe)
    if isinstance(dims, dict):
        dims = dims[var]
    if len(jl_var.shape) != len(dims):
        raise ValueError(_format_to_single_line("""
            `dims` must the length of the number of dimension of the corresponding `var` array in
            the `jl_file`.
        """))
    dims_slice = tuple(slice(None) if x is None else x for x in dims)
    jl_var_1d = jl_var[dims_slice]
    double_shift = jl_var_1d.shape[0] - n
    shift = double_shift//2
    if double_shift%2 == 1:
        warnings.warn(_format_to_single_line("""
            The length array from the `jl_file` of the variable `var` minus `n` is an odd number :
            the removed boundaries are taken 1 point thinner on the bottom side than on the surface
            side.
        """))
        return jl_var_1d[shift:-shift-1]
    else:
        return jl_var_1d[shift:-shift]


class Data(eqx.Module):
    """
    CHANGE HERE
    Abstraction to represent an *obersation*.

    The *observations* represent every elements of a database, each one represent a simulation or a
    measurement with their own time-series of variables and the physical case which is linked to
    them.

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
        If the :attr:`~space.Trajectory.time` of :attr:`trajectory` is not build with constant
        time-steps.

    """

    trajectory: Trajectory
    case: Case
    metadatas: Dict[str, float] = eqx.field(static=True)

    def __init__(self, trajectory: Trajectory, case: Case, metadatas: Dict[str, float]=None) -> Obs:
        time = trajectory.time
        steps = time[1:] - time[:-1]
        if not jnp.all(steps == steps[0]):
            raise ValueError('Tunax only handle constant output time-steps')
        self.trajectory = trajectory
        self.case = case
        self.metadatas = metadatas

    @classmethod
    def from_nc_yaml(
            cls,
            nc_path: str,
            yaml_path: str,
            var_names: Dict[str, str],
            eos_tracers: str = 't',
            do_pt: bool = False
        ) -> Data:
        """
        Create an instance from a *netcdf* and a *yaml* files.

        This class method build a trajectory from the :code:`.nc` file :code:`nc_path`, it build the
        physical parameters from the configuration file :code:`yaml_path`. :code:`var_names` is used
        to do the link between Tunax name convention and the one from the used database.
        CHANGE HEERE
        Parameters
        ----------
        nc_path : str
            Path of the *netcdf* file that contains the time-series of the observation trajectory.
            The file should contains at least the three dimensions :attr:`~space.Grid.zr`
            :attr:`~space.Grid.zw` and :attr:`~space.Trajectory.time`. The time-series can be
            created with default values if they are not present in the file. Otherwise, they must
            have the good dimensions described in :class:`~space.Trajectory`.
        yaml_path : str
            Path of the *yaml* file that contains the parameters and forcing that describe the
            observation. The parameters should be float numbers and directly accessible from the
            root of the file with a key. Only the parameters that are described in
            :class:`~case.Case` will be takend in account.
        var_names : Dict[str, str]
            Link between the convention names in Tunax and the ones in the database. The keys are
            the Tunax names and the values are the names in the database. It works for variables of
            the :class:`~space.Trajectory` and fornthe parameters of :class:`~case.Case`. It must at
            least contains entries for :attr:`~space.Grid.zr` :attr:`~space.Grid.zw` and
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
        u_name = var_names['u']
        if u_name == '':
            u = jnp.full((nt, nz), 0.)
        else:
            u = jnp.array(ds[u_name].values)
        v_name = var_names['v']
        if v_name == '':
            v = jnp.full((nt, nz), 0.)
        else:
            v = jnp.array(ds[v_name].values)
        tracers_dict = {}
        for tracer in TRACERS_NAMES:
            if tracer in var_names.keys():
                tracers_dict[tracer] = jnp.array(ds[var_names[tracer]].values)
        u_name = var_names['u']
        # writing trajectory
        trajectory = Trajectory(grid, time, u, v, **tracers_dict)

        with open(yaml_path, 'r', encoding='utf-8') as f:
            metadatas = yaml.safe_load(f)

        case = Case()
        case_attributes = list(vars(case).keys())
        def get_pytree_fun(att: str):
            return lambda t: getattr(t, att)
        for att in case_attributes:
            if att in var_names.keys():
                case = eqx.tree_at(get_pytree_fun(att), case, metadatas[var_names[att]])
        case = eqx.tree_at(lambda t: getattr(t, 'eos_tracers'), case, eos_tracers)
        case = eqx.tree_at(lambda t: getattr(t, 'do_pt'), case, do_pt)

        return cls(trajectory, case, metadatas)

    @classmethod
    def from_jld2(
            cls,
            jl2d_path: str,
            names_mapping: Dict[str, Dict[str, str]], # il faut séparer les groupes avec de /
            # si pas indiqué, il faut qu'il soit dans var_names, ensuite on
            # prend la partie "centrale" de longueur nz ou nz+1 pour chaque
            # variable, avec un shift plus petit du côté profond si jamais
            # c'est pas symétrique (avec un warning)
            # divisé en 3 parties : variables, parameters, metadats
            nz: Optional[int] = None,
            dims: Union[DimsType, Dict[str, DimsType]] = (None),
            eos_tracers: str = 't',
            do_pt: bool = False
        ) -> Data:
        """
        Load from a .jld2 file.
        conditions :
        les timeseries doivent être des arrays spatiaux dans des groupes avec des temps différents
        c'est pareil pour l'array des valeurs des temps
        names_mapping : Dict[str, Dict[str, str]]
            il faut séparer les groupes avec de / si pas indiqué, il faut qu'il soit dans var_names,
            ensuite on prend la partie "centrale" de longueur nz ou nz+1 pour chaque variable, avec
            un shift plus petit du côté profond si jamais c'est pas symétrique (avec un warning)
            divisé en 3 parties : variables, parameters, metadats
        TO CHECK
        """
        var_map = names_mapping['variables']
        par_map = names_mapping['parameters']
        metadata_map = names_mapping['metadatas']
        jl = H5pyFile(jl2d_path, 'r')
        # récupération de la bonne valeur de nz
        if nz is None:
            nz = int(jl[par_map['nz']][()])
        # variables grid et time
        zr = jnp.array(get_var_jl(jl, var_map, 'zr', nz, dims))
        zw = jnp.array(get_var_jl(jl, var_map, 'zw', nz+1, dims))
        time_group = var_map['time']
        time_str_list = list(jl[time_group].keys())
        time_str_list = [int(i) for i in time_str_list]
        time_str_list.sort()
        time_str_list = [str(i) for i in time_str_list]
        time_float_list = []
        for time_str in time_str_list:
            time_val = jl[f'{time_group}/{time_str}'][()]
            time_float_list.append(float(time_val))
        time = jnp.array(time_float_list)
        # variables
        variables_dict = {}
        for var_name in ['u', 'v'] + TRACERS_NAMES:
            if var_name not in var_map:
                continue
            var_list = []
            for time_str in time_str_list:
                var_time = get_var_jl(jl, var_map, var_name, nz, dims, f'/{time_str}')
                var_list.append(var_time)
            variables_dict[var_name] = jnp.vstack(var_list)
        # trajectory
        trajectory = Trajectory(Grid(zr, zw), time, **variables_dict)

        # parameters
        params = {}
        case_params_list = [nom for nom in vars(Case).keys()]
        for par_name, jl_name in par_map.items():
            if par_name in case_params_list:
                params[par_name] = float(jl[jl_name][()])
        case = Case(eos_tracers=eos_tracers, do_pt=do_pt, **params)

        # metadatas
        metadatas = {}
        for metadata_name, jl_name in metadata_map.items():
            jl_val = jl[jl_name][()]
            if isinstance(jl_val, np.floating) or isinstance(jl_val, np.integer):
                metadatas[metadata_name] = float(jl_val)

        return cls(trajectory, case, metadatas)

    def cut(self, out_nt_cut: int) -> List[Data]:
        traj_list = self.trajectory.cut(out_nt_cut)
        return [Data(traj, self.case, self.metadatas) for traj in traj_list]


class Weights(eqx.Module):
    weight_u: float = 0.
    weight_v: float = 0.
    weight_t: float = 0.
    weight_s: float = 0.
    weight_b: float = 0.
    weight_pt: float = 0.

class Obs(eqx.Module):
    """
    i_stop : index of output on which stop to compute the model, comes with a "virtual nt"
    """
    trajectory: Trajectory
    model: SingleColumnModel
    weights: Weights
    i_out_stop: int

    @classmethod
    def from_data(cls, data: Data, dt: float, weights: Weights, stop_pars: Tuple[int, int]=(-1, -1)):
        """
        Create a Obs instance from a Data one adding weights and a dt$
        TODO add warnings
        """
        nt, i_out_stop = stop_pars
        time = data.trajectory.time 
        if nt == -1:
            nt = int((time[-1]-time[0])/dt)
        out_dt = float(time[1] - time[0])
        p_out = int(out_dt/dt)
        if i_out_stop == -1:
            i_out_stop = int(nt/p_out)
        init_state = data.trajectory.extract_state(0)
        start_time = time[0]
        model = SingleColumnModel(nt, dt, p_out, init_state, data.case, 'k-epsilon', start_time)
        return Obs(data.trajectory, model, weights, i_out_stop)

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
        A list of several observations with potentially various forcings, geometry and time
        configuration.

    """

    observations: List[Obs]
    metadatas: Any
