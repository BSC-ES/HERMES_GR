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
from .projections import (
    RotatedNes, PointsNes, PointsNesGHOST, PointsNesProvidentia, LCCNes, LatLonNes,
    MercatorNes, LAEANes,
)
from .projections.laea_nes import laea_mapping
from .projections._coordinates import mapping_name

DIM_VAR_NAMES = ["lat", "latitude", "lat_bnds", "lon", "longitude", "lon_bnds", "time", "time_bnds", "lev", "level",
                 "cell_area", "crs", "rotated_pole", "x", "y", "rlat", "rlon", "Lambert_conformal", "mercator",
                 "climatology_bounds", "laea", "x_bnds", "y_bnds"]


def open_netcdf(path, comm=None, info=False, parallel_method="Y", avoid_first_hours=0, avoid_last_hours=0,
                first_level=0, last_level=None, balanced=False, time_exception=False,
                crs=None, coordinate_names=None, coordinate_units=None):
    """Open a netCDF file.

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
        Allow input without a time dimension and use the NES placeholder time. Default is False.
    crs : str, int or pyproj.CRS, optional
        Explicit source CRS, accepted by pyproj.CRS.from_user_input, for example
        "EPSG:3035". Completes absent projection metadata; existing contradictory
        metadata raise ValueError. Currently supported for Lambert Azimuthal Equal Area.
    coordinate_names : dict, optional
        Source coordinate variable names keyed by "x" and/or "y", for example
        {"x": "LON", "y": "LAT"}. Unspecified axes use CF standard-name discovery.
    coordinate_units : dict, optional
        Native axis units keyed by "x" and/or "y". Completes missing units without
        converting coordinate values. Conflicting existing units raise ValueError.


    Returns
    -------
    Nes
        A Nes object. Variables read in lazy mode (only metadata).

    Raises
    ------
    FileNotFoundError
        The input path does not exist.
    ValueError
        Referenced CF mappings, native units, Earth shape or grid geometry are
        missing, inconsistent or unsupported.

    NotImplementedError
        A referenced structured CF projection is not supported.

    Notes
    -----
    Structured readers follow CF mapping references and preserve explicit Earth
    definitions and native axis bounds. Legacy NES mapping names remain
    supported. Geographical auxiliaries can be recovered from native axes.

    Lambert Azimuthal Equal Area (LAEA) detection follows each data variable's CF ``grid_mapping`` reference
    and checks ``grid_mapping_name = "lambert_azimuthal_equal_area"``.
    Projected coordinates are identified by their CF standard names, rather
    than fixed variable names. One-dimensional axes must use supported length
    units and be the final Y/X data dimensions. Nonuniform or singleton axes
    require explicit projected bounds. Geographical centres and corners are
    transformed from native projected coordinates using the explicit Earth
    shape, without preprocessing regridding.

    Explicit input options interpret missing metadata in memory and never modify
    the source file. Spatial data dimensions are reordered lazily to Y/X while
    reading rank-local slices. Supply a CRS when no mapping exists. Scientific
    units and valid standard names remain the caller's responsibility. Use
    time_exception=True followed by set_time([datetime(...)]) for an untimed file;
    set_time_bnds can describe the annual interval before writing.

    Call collectively on all ranks of ``comm``. Data variables remain lazy;
    call ``load()`` to read their rank-local arrays.
    """

    if comm is None:
        comm = MPI.COMM_WORLD
    else:
        comm = comm

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if comm.Get_size() == 1:
        dataset = Dataset(path, format="NETCDF4", mode="r", parallel=False)
    else:
        dataset = Dataset(path, format="NETCDF4", mode="r", parallel=True, comm=comm, info=MPI.Info())

    try:
        explicit_input = any(value is not None for value in (crs, coordinate_names, coordinate_units))
        mapping = laea_mapping(dataset)
        native_mapping = mapping_name(dataset)
        supported = {'latitude_longitude', 'rotated_latitude_longitude', 'lambert_conformal_conic',
                     'mercator', 'lambert_azimuthal_equal_area'}
        if 'station' not in dataset.dimensions and native_mapping is not None and native_mapping not in supported:
            raise NotImplementedError(f'Unsupported CF grid mapping: {native_mapping}')
    except Exception:
        dataset.close()
        raise
    try:
        if mapping is not None or explicit_input:
            nessy = LAEANes(comm=comm, dataset=dataset, info=info, parallel_method=parallel_method,
                            avoid_first_hours=avoid_first_hours, avoid_last_hours=avoid_last_hours,
                            first_level=first_level, last_level=last_level, create_nes=False, balanced=balanced,
                            time_exception=time_exception, crs=crs, coordinate_names=coordinate_names,
                            coordinate_units=coordinate_units)
        elif __is_rotated(dataset):
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
                                       first_level=first_level, last_level=last_level, create_nes=False,
                                       balanced=balanced, time_exception=time_exception)
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

    except Exception:
        if dataset.isopen():
            dataset.close()
        raise

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

    if mapping_name(dataset) == "rotated_latitude_longitude":
        return True
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

    if mapping_name(dataset) == "lambert_conformal_conic":
        return True
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

    if mapping_name(dataset) == "mercator":
        return True
    if "mercator" in dataset.variables.keys():
        return True
    else:
        return False


def concatenate_netcdfs(nessy_list, comm=None, info=False, parallel_method="Y", avoid_first_hours=0, avoid_last_hours=0,
                        first_level=0, last_level=None, balanced=False):
    """Concatenate variables form different sources.

    Parameters
    ----------
    nessy_list : list
        A List of Nes objects or list of paths to concatenate.
    comm : MPI.Comm
        MPI Communicator.
    info : bool
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

    Notes
    -----
    Concatenate variables for a shared grid and period, not consecutive time
    ranges. LAEA paths use the CF-aware projection reader, so arbitrarily
    named mapping, coordinate and bounds variables are excluded from scientific
    data. All communicator ranks must participate when input objects are
    distributed.
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
        if isinstance(aux_nessy, str) and isinstance(nessy_first, LAEANes):
            # The CF reader excludes arbitrarily named mappings and bounds;
            # the legacy fixed-name exclusion list cannot do that reliably.
            nessy_first.concatenate(aux_nessy)
            continue
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
