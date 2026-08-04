#!/usr/bin/env python
# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import os
import sys
from numpy import empty
from mpi4py import MPI
from netCDF4 import Dataset
from warnings import warn
from .nc_projections import RotatedNes, PointsNes, PointsNesGHOST, PointsNesProvidentia, LCCNes, LatLonNes, MercatorNes

DIM_VAR_NAMES = ["lat", "latitude", "lat_bnds", "lon", "longitude", "lon_bnds", "time", "time_bnds", "lev", "level",
                 "cell_area", "crs", "rotated_pole", "x", "y", "rlat", "rlon", "Lambert_conformal", "mercator",
                 "climatology_bounds"]


def open_netcdf(path, comm=None, info=False, parallel_method="Y", avoid_first_hours=0, avoid_last_hours=0,
                first_level=0, last_level=None, balanced=False, time_exception=False):
    """
    Open a netCDF file.

    Parameters
    ----------
    path : str
        Path to the NetCDF file to read.
    comm : MPI.COMM
        MPI communicator to use in that netCDF. Default: MPI.COMM_WORLD.
    info : bool
        Indicates if you want to print (stdout) the reading/writing steps.
    avoid_first_hours : int
        Number of hours to remove from first time steps.
    avoid_last_hours : int
        Number of hours to remove from last time steps.
    parallel_method : str
        Indicates the parallelization method that you want. Default: "Y".
        Accepted values: ["X", "Y", "T"]
    balanced : bool
        Indicates if you want a balanced parallelization or not. Balanced dataset cannot be written in chunking mode.
    first_level : int
        Index of the first level to use.
    last_level : int, None
        Index of the last level to use. None if it is the last.
    time_exception : bool
        Indicates if you want to avoid the error when no time in dimensions


    Returns
    -------
    Nes
        A Nes object. Variables read in lazy mode (only metadata).
    """

    if comm is None:
        comm = MPI.COMM_WORLD
    else:
        comm = comm

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    dataset = Dataset(path, format="NETCDF4", mode="r", parallel=False)
    # Parallel is not needed for reading
    # if comm.Get_size() == 1:
    #     dataset = Dataset(path, format="NETCDF4", mode="r", parallel=False)
    # else:
    #     dataset = Dataset(path, format="NETCDF4", mode="r", parallel=True, comm=comm, info=MPI.Info())

    if __is_rotated(dataset):
        # Rotated grids
        nessy = RotatedNes(comm=comm, dataset=dataset, info=info, parallel_method=parallel_method,
                           avoid_first_hours=avoid_first_hours, avoid_last_hours=avoid_last_hours,
                           first_level=first_level, last_level=last_level, create_nes=False, balanced=balanced,
                           time_exception=time_exception)
    elif __is_points(dataset):
        if parallel_method == "Y":
            warn("Parallel method cannot be 'Y' to create points NES. Setting it to 'X'")
            sys.stderr.flush()
            parallel_method = "X"
        if __is_points_ghost(dataset):
            # Points - GHOST
            nessy = PointsNesGHOST(comm=comm, dataset=dataset, info=info,
                                   parallel_method=parallel_method,
                                   avoid_first_hours=avoid_first_hours, avoid_last_hours=avoid_last_hours,
                                   first_level=first_level, last_level=last_level, create_nes=False, balanced=balanced,
                                   time_exception=time_exception)
        elif __is_points_providentia(dataset):
            # Points - Providentia
            nessy = PointsNesProvidentia(comm=comm, dataset=dataset, info=info,
                                         parallel_method=parallel_method,
                                         avoid_first_hours=avoid_first_hours, avoid_last_hours=avoid_last_hours,
                                         first_level=first_level, last_level=last_level, create_nes=False,
                                         balanced=balanced, time_exception=time_exception)
        else:
            # Points - non-GHOST
            nessy = PointsNes(comm=comm, dataset=dataset, info=info, parallel_method=parallel_method,
                              avoid_first_hours=avoid_first_hours, avoid_last_hours=avoid_last_hours,
                              first_level=first_level, last_level=last_level, create_nes=False, balanced=balanced,
                              time_exception=time_exception)
    elif __is_lcc(dataset):
        # Lambert conformal conic grids
        nessy = LCCNes(comm=comm, dataset=dataset, info=info, parallel_method=parallel_method,
                       avoid_first_hours=avoid_first_hours, avoid_last_hours=avoid_last_hours,
                       first_level=first_level, last_level=last_level, create_nes=False, balanced=balanced,
                       time_exception=time_exception)
    elif __is_mercator(dataset):
        # Mercator grids
        nessy = MercatorNes(comm=comm, dataset=dataset, info=info, parallel_method=parallel_method,
                            avoid_first_hours=avoid_first_hours, avoid_last_hours=avoid_last_hours,
                            first_level=first_level, last_level=last_level, create_nes=False, balanced=balanced,
                            time_exception=time_exception)
    else:
        # Regular grids
        nessy = LatLonNes(comm=comm, dataset=dataset, info=info, parallel_method=parallel_method,
                          avoid_first_hours=avoid_first_hours, avoid_last_hours=avoid_last_hours,
                          first_level=first_level, last_level=last_level, create_nes=False, balanced=balanced,
                          time_exception=time_exception)

    return nessy


def __is_rotated(dataset):
    """
    Check if the netCDF is in rotated pole projection or not.

    Parameters
    ----------
    dataset : Dataset
        netcdf4-python open dataset object.

    Returns
    -------
    value : bool
        Indicated if the netCDF is a rotated one.
    """

    if "rotated_pole" in dataset.variables.keys():
        return True
    elif ("rlat" in dataset.dimensions) and ("rlon" in dataset.dimensions):
        return True
    else:
        return False


def __is_points(dataset):
    """
    Check if the netCDF is a points dataset in non-GHOST format or not.

    Parameters
    ----------
    dataset : Dataset
        netcdf4-python open dataset object.

    Returns
    -------
    value : bool
        Indicated if the netCDF is a points non-GHOST one.
    """

    if "station" in dataset.dimensions:
        return True
    else:
        return False


def __is_points_ghost(dataset):
    """
    Check if the netCDF is a points dataset in GHOST format or not.

    Parameters
    ----------
    dataset : Dataset
        netcdf4-python open dataset object.

    Returns
    -------
    value : bool
        Indicated if the netCDF is a points GHOST one.
    """

    if "N_flag_codes" in dataset.dimensions and "N_qa_codes" in dataset.dimensions:
        return True
    else:
        return False


def __is_points_providentia(dataset):
    """
    Check if the netCDF is a points dataset in Providentia format or not.

    Parameters
    ----------
    dataset : Dataset
        netcdf4-python open dataset object.

    Returns
    -------
    value : bool
        Indicated if the netCDF is a points Providentia one.
    """

    if (("grid_edge" in dataset.dimensions) and ("model_latitude" in dataset.dimensions) and
            ("model_longitude" in dataset.dimensions)):
        return True
    else:
        return False


def __is_lcc(dataset):
    """
    Check if the netCDF is in Lambert Conformal Conic (LCC) projection or not.

    Parameters
    ----------
    dataset : Dataset
        netcdf4-python open dataset object.

    Returns
    -------
    value : bool
        Indicated if the netCDF is an LCC one.
    """

    if "Lambert_Conformal" in dataset.variables.keys() or "Lambert_conformal" in dataset.variables.keys():
        return True
    else:
        return False


def __is_mercator(dataset):
    """
    Check if the netCDF is in Mercator projection or not.

    Parameters
    ----------
    dataset : Dataset
        netcdf4-python open dataset object.

    Returns
    -------
    value : bool
        Indicated if the netCDF is a Mercator one.
    """

    if "mercator" in dataset.variables.keys():
        return True
    else:
        return False


def concatenate_netcdfs(nessy_list, comm=None, info=False, parallel_method="Y", avoid_first_hours=0, avoid_last_hours=0,
                        first_level=0, last_level=None, balanced=False):
    """
    Concatenate variables form different sources.

    Parameters
    ----------
    nessy_list : list
        A List of Nes objects or list of paths to concatenate.
    comm : MPI.Comm
        MPI Communicator.
    info: bool
        Indicates if you want to get reading/writing info.
    parallel_method : str
        Indicates the parallelization method that you want. Default: "Y".
        accepted values: ["X", "Y", "T"].
    balanced : bool
        Indicates if you want a balanced parallelization or not.
        Balanced dataset cannot be written in chunking mode.
    avoid_first_hours : int
        Number of hours to remove from first time steps.
    avoid_last_hours : int
        Number of hours to remove from last time steps.
    first_level : int
        Index of the first level to use.
    last_level : int, None
        Index of the last level to use. None if it is the last.

    Returns
    -------
    Nes
        A Nes object with all the variables.
    """
    if not isinstance(nessy_list, list):
        raise AttributeError("You must pass a list of NES objects or paths.")

    if isinstance(nessy_list[0], str):
        nessy_first = open_netcdf(nessy_list[0],
                                  comm=comm,
                                  parallel_method=parallel_method,
                                  info=info,
                                  avoid_first_hours=avoid_first_hours,
                                  avoid_last_hours=avoid_last_hours,
                                  first_level=first_level,
                                  last_level=last_level,
                                  balanced=balanced
                                  )
        nessy_first.load()
    else:
        nessy_first = nessy_list[0]
    for i, aux_nessy in enumerate(nessy_list[1:]):
        if isinstance(aux_nessy, str):
            nc_add = Dataset(filename=aux_nessy, mode="r")
            for var_name, var_info in nc_add.variables.items():
                if var_name not in DIM_VAR_NAMES:
                    nessy_first.variables[var_name] = {}
                    var_dims = var_info.dimensions
                    # Read data in 4 dimensions
                    if len(var_dims) < 2:
                        data = var_info[:]
                    elif len(var_dims) == 2:
                        data = var_info[nessy_first.read_axis_limits["y_min"]:nessy_first.read_axis_limits["y_max"],
                                        nessy_first.read_axis_limits["x_min"]:nessy_first.read_axis_limits["x_max"]]
                        data = data.reshape(1, 1, data.shape[-2], data.shape[-1])
                    elif len(var_dims) == 3:
                        if "strlen" in var_dims:
                            data = var_info[nessy_first.read_axis_limits["y_min"]:nessy_first.read_axis_limits["y_max"],
                                            nessy_first.read_axis_limits["x_min"]:nessy_first.read_axis_limits["x_max"],
                                            :]
                            data_aux = empty(shape=(data.shape[0], data.shape[1]), dtype=object)
                            for lat_n in range(data.shape[0]):
                                for lon_n in range(data.shape[1]):
                                    data_aux[lat_n, lon_n] = "".join(
                                        data[lat_n, lon_n].tobytes().decode("ascii").replace("\x00", ""))
                            data = data_aux.reshape((1, 1, data_aux.shape[-2], data_aux.shape[-1]))
                        else:
                            data = var_info[nessy_first.read_axis_limits["t_min"]:nessy_first.read_axis_limits["t_max"],
                                            nessy_first.read_axis_limits["y_min"]:nessy_first.read_axis_limits["y_max"],
                                            nessy_first.read_axis_limits["x_min"]:nessy_first.read_axis_limits["x_max"]]
                            data = data.reshape(data.shape[-3], 1, data.shape[-2], data.shape[-1])
                    elif len(var_dims) == 4:
                        data = var_info[nessy_first.read_axis_limits["t_min"]:nessy_first.read_axis_limits["t_max"],
                                        nessy_first.read_axis_limits["z_min"]:nessy_first.read_axis_limits["z_max"],
                                        nessy_first.read_axis_limits["y_min"]:nessy_first.read_axis_limits["y_max"],
                                        nessy_first.read_axis_limits["x_min"]:nessy_first.read_axis_limits["x_max"]]
                    else:
                        raise TypeError("{} data shape is nto accepted".format(var_dims))

                    nessy_first.variables[var_name]["data"] = data
                    # Avoid some attributes
                    for attrname in var_info.ncattrs():
                        if attrname not in ["missing_value", "_FillValue"]:
                            value = getattr(var_info, attrname)
                            if value in ["unitless", "-"]:
                                value = ""
                            nessy_first.variables[var_name][attrname] = value
            nc_add.close()

        else:
            nessy_first.concatenate(aux_nessy)

    return nessy_first
