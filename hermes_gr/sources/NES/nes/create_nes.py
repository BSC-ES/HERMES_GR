#!/usr/bin/env python
# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import warnings
import sys
from netCDF4 import num2date
from mpi4py import MPI
from .nc_projections import PointsNes, LatLonNes, RotatedNes, RotatedNestedNes, LCCNes, MercatorNes


def create_nes(comm=None, info=False, projection=None, parallel_method="Y", balanced=False,
               times=None, avoid_first_hours=0, avoid_last_hours=0, first_level=0, last_level=None,
               **kwargs):
    """
    Create a Nes class from scratch.

    Parameters
    ----------
    comm : MPI.Comm, optional
        MPI Communicator. If None, uses MPI.COMM_WORLD.
    info : bool, optional
        Indicates if reading/writing info should be provided. Default is False.
    projection : str, optional
        The projection type. Accepted values are None, "regular", "global", "global_monarch", "rotated",
        "rotated-nested", "lcc", "mercator".
    parallel_method : str, optional
        The parallelization method to use. Default is "Y". Accepted values are ["X", "Y", "T"].
    balanced : bool, optional
        Indicates if balanced parallelization is desired. Balanced datasets cannot be written in chunking mode.
        Default is False.
    times : list of datetime, optional
        List of datetime objects representing the time dimension. If None, a default time array is created.
    avoid_first_hours : int, optional
        Number of hours to remove from the start of the time steps. Default is 0.
    avoid_last_hours : int, optional
        Number of hours to remove from the end of the time steps. Default is 0.
    first_level : int, optional
        Index of the first level to use. Default is 0.
    last_level : int or None, optional
        Index of the last level to use. If None, the last level is used. Default is None.
    **kwargs : additional arguments
        Additional parameters required for specific projections.

    Returns
    -------
    nes : Nes
        An instance of the Nes class based on the specified parameters and projection.

    Raises
    ------
    ValueError
        If any required projection-specific parameters are missing or if invalid parameters are provided.
    NotImplementedError
        If an unsupported parallel method or projection type is specified.

    Notes
    -----
    The function dynamically creates an instance of a specific Nes subclass based on the provided projection.
    The required parameters for each projection type are:
        - None: ["lat", "lon"]
        - "regular": ["lat_orig", "lon_orig", "inc_lat", "inc_lon", "n_lat", "n_lon"]
        - "global": ["inc_lat", "inc_lon"]
        - "global_monarch": ["inc_lat", "inc_lon"]
        - "rotated": ["centre_lat", "centre_lon", "west_boundary", "south_boundary", "inc_rlat", "inc_rlon"]
        - "rotated-nested": ["parent_grid_path", "parent_ratio", "i_parent_start", "j_parent_start", "n_rlat", "n_rlon"]
        - "lcc": ["lat_1", "lat_2", "lon_0", "lat_0", "nx", "ny", "inc_x", "inc_y", "x_0", "y_0"]
        - "mercator": ["lat_ts", "lon_0", "nx", "ny", "inc_x", "inc_y", "x_0", "y_0"]

    Example
    -------
    >>> nes = create_nes(projection="regular", lat_orig=0, lon_orig=0, inc_lat=1, inc_lon=1, n_lat=180, n_lon=360)
    """

    if comm is None:
        comm = MPI.COMM_WORLD
    else:
        comm = comm

    # Create time array
    if times is None:
        units = "days since 1996-12-31 00:00:00"
        calendar = "standard"
        times = num2date([0], units=units, calendar=calendar)
        times = [aux.replace(second=0, microsecond=0) for aux in times]
    else:
        if not isinstance(times, list):
            times = list(times)

    # Check if the parameters that are required to create the object have been defined in kwargs
    kwargs_list = []
    for name, value in kwargs.items():
        kwargs_list.append(name)

    if projection is None:
        required_vars = ["lat", "lon"]
    elif projection == "regular":
        required_vars = ["lat_orig", "lon_orig", "inc_lat", "inc_lon", "n_lat", "n_lon"]
    elif projection in ["global", "global_monarch"]:
        required_vars = ["inc_lat", "inc_lon"]
    elif projection == "rotated":
        required_vars = ["centre_lat", "centre_lon", "west_boundary", "south_boundary", "inc_rlat", "inc_rlon"]
    elif projection in ["rotated-nested", "rotated_nested"]:
        required_vars = ["parent_grid_path", "parent_ratio", "i_parent_start", "j_parent_start", "n_rlat", "n_rlon"]
    elif projection == "lcc":
        required_vars = ["lat_1", "lat_2", "lon_0", "lat_0", "nx", "ny", "inc_x", "inc_y", "x_0", "y_0"]
    elif projection == "mercator":
        required_vars = ["lat_ts", "lon_0", "nx", "ny", "inc_x", "inc_y", "x_0", "y_0"]
    else:
        raise ValueError("Unknown projection: {0}".format(projection))

    for var in required_vars:
        if var not in kwargs_list:
            msg = "Variable {0} has not been defined. ".format(var)
            msg += "For a {} projection, it is necessary to define {}".format(projection, required_vars)
            raise ValueError(msg)

    for var in kwargs_list:
        if var not in required_vars:
            msg = "Variable {0} has been defined. ".format(var)
            msg += "For a {} projection, you can only define {}".format(projection, required_vars)
            raise ValueError(msg)

    if projection in ["global", "global_monarch"]:
        kwargs["nes_projection"] = projection

    if projection is None:
        if parallel_method == "Y":
            warnings.warn("Parallel method cannot be 'Y' to create points NES. Setting it to 'X'")
            sys.stderr.flush()
            parallel_method = "X"
        elif parallel_method == "T":
            raise NotImplementedError("Parallel method T not implemented yet")
        nessy = PointsNes(comm=comm, dataset=None, info=info, parallel_method=parallel_method,
                          avoid_first_hours=avoid_first_hours, avoid_last_hours=avoid_last_hours,
                          first_level=first_level, last_level=last_level, balanced=balanced,
                          create_nes=True, times=times, **kwargs)
    elif projection in ["regular", "global", "global_monarch"]:
        nessy = LatLonNes(comm=comm, dataset=None, info=info, parallel_method=parallel_method,
                          avoid_first_hours=avoid_first_hours, avoid_last_hours=avoid_last_hours,
                          first_level=first_level, last_level=last_level, balanced=balanced,
                          create_nes=True, times=times, **kwargs)
    elif projection == "rotated":
        nessy = RotatedNes(comm=comm, dataset=None, info=info, parallel_method=parallel_method,
                           avoid_first_hours=avoid_first_hours, avoid_last_hours=avoid_last_hours,
                           first_level=first_level, last_level=last_level, balanced=balanced,
                           create_nes=True, times=times, **kwargs)
    elif projection == "rotated-nested":
        nessy = RotatedNestedNes(comm=comm, dataset=None, info=info, parallel_method=parallel_method,
                                 avoid_first_hours=avoid_first_hours, avoid_last_hours=avoid_last_hours,
                                 first_level=first_level, last_level=last_level, balanced=balanced,
                                 create_nes=True, times=times, **kwargs)
    elif projection == "lcc":
        nessy = LCCNes(comm=comm, dataset=None, info=info, parallel_method=parallel_method,
                       avoid_first_hours=avoid_first_hours, avoid_last_hours=avoid_last_hours,
                       first_level=first_level, last_level=last_level, balanced=balanced,
                       create_nes=True, times=times, **kwargs)
    elif projection == "mercator":
        nessy = MercatorNes(comm=comm, dataset=None, info=info, parallel_method=parallel_method,
                            avoid_first_hours=avoid_first_hours, avoid_last_hours=avoid_last_hours,
                            first_level=first_level, last_level=last_level, balanced=balanced,
                            create_nes=True, times=times, **kwargs)
    else:
        raise NotImplementedError(projection)

    return nessy


def from_shapefile(path, method=None, parallel_method="Y", **kwargs):
    """
    Create NES from shapefile data.

    1. Create NES grid.
    2. Create shapefile for grid.
    3. Spatial join to add shapefile variables to NES variables.

    Parameters
    ----------
    path : str
        Path to shapefile.
    method : str
        Overlay method. Accepted values: ["nearest", "intersection", None].
    parallel_method : str
        Indicates the parallelization method that you want. Default: "Y".
        accepted values: ["X", "Y", "T"].
    """

    # Create NES
    nessy = create_nes(comm=None, info=False, parallel_method=parallel_method, **kwargs)

    # Create shapefile for grid
    nessy.create_shapefile()

    # Make spatial join
    nessy.spatial_join(path, method=method)

    return nessy
