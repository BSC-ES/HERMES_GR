#!/usr/bin/env python
# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import sys
from copy import copy, deepcopy
from datetime import datetime, timedelta
from gc import collect
from hashlib import md5
from math import isclose
from os import makedirs, path
from shutil import which
from sqlite3 import connect
from time import time as _timer
from typing import Any, Dict, List, Tuple, Union
from warnings import warn

import cftime
import numpy.typing as npt
from dateutil.relativedelta import relativedelta
from geopandas import GeoDataFrame, read_file
from mpi4py import MPI
from netCDF4 import Dataset, date2num, num2date
from numba import boolean, njit
from numpy import (
    abs,
    append,
    arange,
    argsort,
    array,
    character,
    column_stack,
    concatenate,
    cumsum,
    diff,
    dstack,
    empty,
    flipud,
    float32,
    float64,
    full,
    generic,
    histogram,
    hstack,
    inf,
    isnan,
    issubdtype,
    log10,
    logspace,
    ma,
    mean,
    nan,
    nanmax,
    nanmean,
    nanmin,
    nansum,
    nanpercentile,
    ndarray,
    newaxis,
    split,
    stack,
    std,
    take,
    tile,
    unique,
    vstack,
    zeros,
)
from osgeo.gdal import SetConfigOption, UseExceptions
from pandas import Index, concat
from pyproj import Proj, Transformer
from shapely.geometry import Point

from ..methods import (
    cell_measures,
    horizontal_interpolation,
    spatial_join,
    vertical_interpolation,
)
from ..nes_formats import (
    to_chimere_units,
    to_cmaq_units,
    to_mocage_units,
    to_monarch_units,
    to_netcdf_cams_ra,
    to_netcdf_chimere,
    to_netcdf_cmaq,
    to_netcdf_mocage,
    to_netcdf_monarch,
    to_netcdf_wrf_chem,
    to_wrf_chem_units,
)

UseExceptions()
SetConfigOption("CPL_LOG_ERRORS", "ON")
SetConfigOption("GDAL_NUM_THREADS", "1")
SetConfigOption("GDAL_CACHEMAX", "64")
SetConfigOption("OGR_SQLITE_SYNCHRONOUS", "OFF")
SetConfigOption("OGR_SQLITE_JOURNAL", "MEMORY")


class Nes(object):
    """
    A class to handle netCDF data with parallel processing capabilities using MPI.

    Attributes
    ----------
    comm:MPI.Comm
        MPI communicator.
    rank:int
        MPI rank.
    master:bool
        True when rank == 0.
    size:int
        Size of the communicator.
    info:bool
        Indicates if you want to print reading/writing info.
    __ini_path:str
        Path to the original file to read when open_netcdf is called.
    hours_start:int
        Number of hours to avoid from the first original values.
    hours_end:int
        Number of hours to avoid from the last original values.
    dataset:Dataset
        netcdf4-python Dataset.
    variables:Dict[str, Dict[str, Any]]
        Variables information. The dictionary structure is:
        {
            var_name: {
                "data": ndarray or None,  # Array values or None if the variable is not loaded.
                attr_name: attr_value,  # Variable attributes.
                ...
            },
            ...
        }
    _full_time:List[datetime]
        Complete list of original time step values.
    _full_lev:Dict[str, array]
        Vertical level dictionary with the complete "data" key for all the values and the rest of the attributes.
        {
            "data": ndarray,  # Array of vertical level values.
            attr_name: attr_value,  # Vertical level attributes.
            ...
        }
    _full_lat:dict
        Latitudes dictionary with the complete "data" key for all the values and the rest of the attributes.
        {
            "data": ndarray,  # Array of latitude values.
            attr_name: attr_value,  # Latitude attributes.
            ...
        }
    _full_lon:dict
        Longitudes dictionary with the complete "data" key for all the values and the rest of the attributes.
        {
            "data": ndarray,  # Array of longitude values.
            attr_name: attr_value,  # Longitude attributes.
            ...
        }
    _full_lat_bnds:dict
        Latitude bounds dictionary with the complete "data" key for the latitudinal boundaries of each grid and the
        rest of the attributes.
        {
            "data": ndarray,  # Array of latitude bounds.
            attr_name: attr_value,  # Latitude bounds attributes.
            ...
        }
    _full_lon_bnds:dict
        Longitude bounds dictionary with the complete "data" key for the longitudinal boundaries of each grid and the
        rest of the attributes.
        {
            "data": ndarray,  # Array of longitude bounds.
            attr_name: attr_value,  # Longitude bounds attributes.
            ...
        }
    parallel_method:str
        Parallel method to read/write. Can be chosen from any of the following axes to parallelize: "T", "Y", or "X".
    read_axis_limits:dict
        Dictionary with the 4D limits of the rank data to read. Structure:
        {
            "t_min": int, "t_max": int,  # Time axis limits.
            "z_min": int, "z_max": int,  # Vertical axis limits.
            "y_min": int, "y_max": int,  # Latitudinal axis limits.
            "x_min": int, "x_max": int,  # Longitudinal axis limits.
        }
    write_axis_limits:dict
        Dictionary with the 4D limits of the rank data to write. Structure:
        {
            "t_min": int, "t_max": int,  # Time axis limits.
            "z_min": int, "z_max": int,  # Vertical axis limits.
            "y_min": int, "y_max": int,  # Latitudinal axis limits.
            "x_min": int, "x_max": int,  # Longitudinal axis limits.
        }
    time:List[datetime]
        List of time steps of the rank data.
    lev:dict
        Vertical levels dictionary with the portion of "data" corresponding to the rank values. Structure:
        {
            "data": ndarray,  # Array of vertical level values for the rank.
            attr_name: attr_value,  # Vertical level attributes.
            ...
        }
    lat:dict
        Latitudes dictionary with the portion of "data" corresponding to the rank values. Structure:
        {
            "data": ndarray,  # Array of latitude values for the rank.
            attr_name: attr_value,  # Latitude attributes.
            ...
        }
    lon:dict
        Longitudes dictionary with the portion of "data" corresponding to the rank values. Structure:
        {
            "data": ndarray,  # Array of longitude values for the rank.
            attr_name: attr_value,  # Longitude attributes.
            ...
        }
    lat_bnds:dict
        Latitude bounds dictionary with the portion of "data" for the latitudinal boundaries corresponding to the rank
        values.
        Structure:
        {
            "data": ndarray,  # Array of latitude bounds for the rank.
            attr_name: attr_value,  # Latitude bounds attributes.
            ...
        }
    lon_bnds:dict
        Longitude bounds dictionary with the portion of "data" for the longitudinal boundaries corresponding to the
        rank values.
        Structure:
        {
            "data": ndarray,  # Array of longitude bounds for the rank.
            attr_name: attr_value,  # Longitude bounds attributes.
            ...
        }
    global_attrs:dict
        Global attributes with the attribute name as key and data as values. Structure:
        {
            attr_name: attr_value,  # Global attribute name and value.
            ...
        }
    _var_dim:tuple
        Name of the Y and X dimensions for the variables.
    _lat_dim:tuple
        Name of the dimensions of the Latitude values.
    _lon_dim:tuple
        Name of the dimensions of the Longitude values.
    projection:Proj
        Grid projection.
    projection_data:dict
        Dictionary with the projection information. Structure:
        {
            proj_param: proj_value,  # Projection parameters.
            ...
        }
    """

    def __init__(
        self,
        comm: Union[MPI.Comm, None] = None,
        path: Union[str, None] = None,
        info: bool = False,
        dataset: Union[Dataset, None] = None,
        parallel_method: str = "Y",
        avoid_first_hours: int = 0,
        avoid_last_hours: int = 0,
        first_level: int = 0,
        last_level: Union[int, None] = None,
        create_nes: bool = False,
        balanced: bool = False,
        times: Union[List[datetime], None] = None,
        time_exception: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize the Nes class

        Parameters
        ----------
        comm: MPI.COMM
            MPI Communicator.
        path: str
            Path to the NetCDF to initialize the object.
        info: bool
            Indicates if you want to get reading/writing info.
        dataset: Dataset or None
            NetCDF4-python Dataset to initialize the class.
        parallel_method:str
            Indicates the parallelization method that you want. Default over Y axis
            accepted values: ["X", "Y", "T"].
        avoid_first_hours:int
            Number of hours to remove from first time steps.
        avoid_last_hours:int
            Number of hours to remove from last time steps.
        first_level:int
            Index of the first level to use.
        last_level:int or None
            Index of the last level to use. None if it is the last.
        create_nes:bool
            Indicates if you want to create the object from scratch (True) or through an existing file.
        balanced:bool
            Indicates if you want a balanced parallelization or not.
            Balanced dataset cannot be written in chunking mode.
        times:List[datetime] or None
            List of times to substitute the current ones while creation.
        time_exception : bool
            Indicates if you want to avoid the error when no time in dimensions
        """

        # MPI Initialization
        if comm is None:
            self.comm = MPI.COMM_WORLD
        else:
            self.comm = comm
        self.rank = self.comm.Get_rank()
        self.master = self.rank == 0
        self.size = self.comm.Get_size()

        # General info
        self.info = info
        self.__ini_path = path
        self.shapefile = None

        # Selecting info
        self.hours_start = avoid_first_hours
        self.hours_end = avoid_last_hours
        self.first_level = first_level
        self.last_level = last_level
        self.lat_min = None
        self.lat_max = None
        self.lon_min = None
        self.lon_max = None
        self.balanced = balanced

        # Define parallel method
        self.parallel_method = parallel_method
        self.serial_nc = None  # Place to store temporally the serial Nes instance

        # Get minor and major axes of Earth
        self.earth_radius = self.get_earth_radius("WGS84")

        # Time resolution and climatology will be modified, if needed, during the time variable reading
        self._time_resolution = "hours"
        self._climatology = False
        self._climatology_var_name = "climatology_bounds"  # Default var_name but can be changed if the input is dif

        # NetCDF object
        if create_nes:
            self.dataset = None

            # Set string length
            self.strlen = None

            # Initialize variables
            self.variables = {}

            # Projection data This is duplicated due to if it is needed to create the object NES needs that info to
            # create coordinates data.
            self.projection_data = self._get_projection_data(create_nes, **kwargs)
            self.projection = self._get_pyproj_projection()

            # Complete dimensions
            self._full_time = times

            self._full_time_bnds = self.__get_time_bnds(create_nes)
            self._full_lat_bnds, self._full_lon_bnds = self.__get_coordinates_bnds(
                create_nes
            )
            self._full_lev = {"data": array([0]), "units": "", "positive": "up"}
            self._full_lat, self._full_lon = self._create_centre_coordinates(**kwargs)

            # Set axis limits for parallel readingblack
            self.read_axis_limits = self._get_read_axis_limits()
            self.write_axis_limits = self._get_write_axis_limits()

            # Dimensions screening
            self.time = self.get_full_times()[
                self.read_axis_limits["t_min"]: self.read_axis_limits["t_max"]
            ]
            self.time_bnds = self.get_full_time_bnds()
            self.lev = self.get_full_levels()
            self.lat_bnds = self.get_full_latitudes_boundaries()
            self.lon_bnds = self.get_full_longitudes_boundaries()

            # Cell measures screening
            self.cell_measures = self.__get_cell_measures(create_nes)

            # Set NetCDF attributes
            self.global_attrs = self.__get_global_attributes(create_nes)
            self.loaded = True

        else:
            if dataset is not None:
                self.dataset = dataset
            elif self.__ini_path is not None:
                self._open()

            # Get string length
            self.strlen = self._get_strlen()

            # Lazy variables
            self.variables = self._get_lazy_variables()

            # Complete dimensions
            self._full_time = self.__get_time(time_exception=time_exception)
            self._full_time_bnds = self.__get_time_bnds()
            self._full_lev = self._get_coordinate_dimension(
                ["lev", "level", "lm", "plev"]
            )
            self._full_lat = self._get_coordinate_dimension(
                ["lat", "latitude", "latitudes"]
            )
            self._full_lon = self._get_coordinate_dimension(
                ["lon", "longitude", "longitudes"]
            )
            self._full_lat_bnds, self._full_lon_bnds = self.__get_coordinates_bnds()

            # Complete cell measures
            self._cell_measures = self.__get_cell_measures()

            # Set axis limits for parallel reading
            self.read_axis_limits = self._get_read_axis_limits()
            # Set axis limits for parallel writing
            self.write_axis_limits = self._get_write_axis_limits()

            # Dimensions screening
            self.time = self.get_full_times()[
                self.read_axis_limits["t_min"]: self.read_axis_limits["t_max"]
            ]
            self.time_bnds = self.get_full_time_bnds()
            self.lev = self._get_coordinate_values(self.get_full_levels(), "Z")
            self.lat = self._get_coordinate_values(self.get_full_latitudes(), "Y")
            self.lon = self._get_coordinate_values(self.get_full_longitudes(), "X")
            self.lat_bnds = self._get_coordinate_values(
                self.get_full_latitudes_boundaries(), "Y", bounds=True
            )
            self.lon_bnds = self._get_coordinate_values(
                self.get_full_longitudes_boundaries(), "X", bounds=True
            )

            # Cell measures screening
            self.cell_measures = self._get_cell_measures_values(self._cell_measures)

            # Set NetCDF attributes
            self.global_attrs = self.__get_global_attributes()

            # Projection data
            self.projection_data = self._get_projection_data(create_nes, **kwargs)
            self.projection = self._get_pyproj_projection()
            self.loaded = False

        # TODO: Check free vars for this:
        self.free_vars(self._climatology_var_name)

        # Writing options
        self.zip_lvl = 0

        # Dimensions information
        self._var_dim = None
        self._lat_dim = None
        self._lon_dim = None

        self.vertical_var_name = None

        # Filtering (portion of the filter coordinates function)
        idx = self._get_idx_intervals()
        if self.master:
            self.set_full_times(self._full_time[idx["idx_t_min"]: idx["idx_t_max"]])
            self._full_lev["data"] = self._full_lev["data"][
                idx["idx_z_min"]: idx["idx_z_max"]
            ]

        self.hours_start = 0
        self.hours_end = 0
        self.last_level = None
        self.first_level = None

    @staticmethod
    def new(
        comm=None,
        path=None,
        info=False,
        dataset=None,
        parallel_method="Y",
        avoid_first_hours=0,
        avoid_last_hours=0,
        first_level=0,
        last_level=None,
        create_nes=False,
        balanced=False,
        times=None,
        **kwargs,
    ):
        """
        Initialize the Nes class.

        Parameters
        ----------
        comm: MPI.COMM
            MPI Communicator.
        path: str
            Path to the NetCDF to initialize the object.
        info: bool
            Indicates if you want to get reading/writing info.
        dataset: Dataset
            NetCDF4-python Dataset to initialize the class.
        parallel_method:str
            Indicates the parallelization method that you want. Default over Y axis
            accepted values: ["X", "Y", "T"].
        avoid_first_hours:int
            Number of hours to remove from first time steps.
        avoid_last_hours:int
            Number of hours to remove from last time steps.
        first_level:int
            Index of the first level to use.
        last_level:int or None
            Index of the last level to use. None if it is the last.
        create_nes:bool
            Indicates if you want to create the object from scratch (True) or through an existing file.
        balanced:bool
            Indicates if you want a balanced parallelization or not.
            Balanced dataset cannot be written in chunking mode.
        times:List[datetime] or None
            List of times to substitute the current ones while creation.
        """

        new = Nes(
            comm=comm,
            path=path,
            info=info,
            dataset=dataset,
            parallel_method=parallel_method,
            avoid_first_hours=avoid_first_hours,
            avoid_last_hours=avoid_last_hours,
            first_level=first_level,
            last_level=last_level,
            create_nes=create_nes,
            balanced=balanced,
            times=times,
            **kwargs,
        )

        return new

    def _get_strlen(self):
        """
        Get the strlen

        Returns
        -------
        int
            Max length of the string data
        """

        if "strlen" in self.dataset.dimensions:
            strlen = self.dataset.dimensions["strlen"].size
        else:
            return None

        return strlen

    def set_strlen(self, strlen=75):
        """
        Set the strlen

        75 is the standard value used in GHOST data

        Parameters
        ----------
        strlen:int or None
            Max length of the string
        """

        self.strlen = strlen

        return None

    def __del__(self):
        """
        To delete the Nes object and close all the open datasets.
        """

        self.close()
        try:
            self.free_vars(list(self.variables.keys()))
            del self.variables
            del self.time
            del self._full_time
            del self.time_bnds
            del self._full_time_bnds
            del self.lev
            del self._full_lev
            del self.lat
            del self._full_lat
            del self.lon
            del self._full_lon
            del self._full_lat_bnds
            del self.lat_bnds
            del self._full_lon_bnds
            del self.lon_bnds
            del self.strlen
            del self.shapefile
            for cell_measure in self.cell_measures.keys():
                if self.cell_measures[cell_measure]["data"] is not None:
                    del self.cell_measures[cell_measure]["data"]
            del self.cell_measures
        except (AttributeError, KeyError):
            pass

        del self
        collect()

        return None

    def __getstate__(self):
        """
        Read the CSV file that contains all the Reduce variable specifications.

        Returns
        -------
        state:dict
            Dictionary with the class parameters.
        """

        d = self.__dict__
        state = {
            k: d[k]
            for k in d
            if k not in ["comm", "variables", "dataset", "cell_measures"]
        }

        return state

    def __setstate__(self, state):
        """
        Set the state of the class.

        Parameters
        ----------
        state: dict
            Dictionary with the class parameters.
        """

        self.__dict__ = state

        return None

    def __add__(self, other):
        """
        Sum two NES objects

        Parameters
        ----------
        other:Nes
            A Nes to be summed

        Returns
        -------
        Nes
            Summed Nes object
        """
        nessy = self.copy(copy_vars=True)
        for var_name in other.variables.keys():
            if var_name not in nessy.variables.keys():
                # Create New variable
                nessy.variables[var_name] = deepcopy(other.variables[var_name])
            else:
                nessy.variables[var_name]["data"] += other.variables[var_name]["data"]
        return nessy

    def __radd__(self, other):
        if other == 0 or other is None:
            return self
        else:
            return self.__add__(other)

    def __getitem__(self, key: str) -> Union[array, None]:
        """
        Retrieve the data associated with the specified key.

        Parameters
        ----------
        key:str
            The key to retrieve the data for.

        Returns
        -------
        Union[array, None]
            The data associated with the specified key, or None if the key
            does not exist.

        Notes
        -----
        This method allows accessing data in the variables dictionary using
        dictionary-like syntax, e.g., obj[key]["data"].

        """
        return self.variables[key]["data"]

    def copy(self, copy_vars: bool = False, new_comm=None):
        """
        Copy the Nes object.
        The copy avoids duplicating dataset and MPI communicator (unless specified).

        Parameters
        ----------
        copy_vars:bool
            If True, includes variables and cell_measures in the copy.
        new_comm:MPI.Comm, optional
            If provided, sets a new MPI communicator.

        Returns
        -------
        Nes
            A copy of the Nes object.
        """
        # Temporarily remove non-copyable elements
        original_comm = self.comm
        original_dataset = self.dataset

        self.comm = None
        self.dataset = None

        nessy = deepcopy(self)

        # Restore to original
        self.comm = original_comm
        self.dataset = original_dataset

        # Reattach communicator and dataset
        if new_comm is not None:
            nessy.set_communicator(new_comm)
        else:
            nessy.comm = self.comm

        # Reattach dataset manually if needed
        nessy.dataset = None

        if copy_vars:
            nessy.variables = deepcopy(self.variables)
            nessy.cell_measures = deepcopy(self.cell_measures)
        else:
            nessy.variables = {}
            nessy.cell_measures = {}

        return nessy

    def copy2(self, copy_vars: bool = False, new_comm=None):
        """
        Copy the Nes object.
        The copy will avoid to copy the communicator, dataset and variables by default.

        Parameters
        ----------
        copy_vars: bool
            Indicates if you want to copy the variables (in lazy mode).
        new_comm:Comm
            New MPI communicator

        Returns
        -------
        nessy:Nes
            Copy of the Nes object.
        """

        nessy = deepcopy(self)
        nessy.dataset = None

        if copy_vars:
            nessy.variables = deepcopy(self.variables)
            nessy.cell_measures = deepcopy(self.cell_measures)
        else:
            nessy.variables = {}
            nessy.cell_measures = {}
            if new_comm is not None:
                nessy.set_communicator(new_comm)

            return nessy

    def get_full_times(self) -> List[datetime]:
        """
        Retrieve the complete list of original time step values.

        Returns
        -------
        List[datetime]
            The complete list of original time step values from the netCDF data.
        """
        if self.size == 1:
            return self._full_time

        if self.master:
            data = self._full_time
        else:
            data = None
        data = self.comm.bcast(data, root=0)

        if not isinstance(data, list):
            data = list(data)
        return data

    def get_full_time_bnds(self) -> List[datetime]:
        """
        Retrieve the complete list of original time step boundaries.

        Returns
        -------
        List[datetime]
            The complete list of original time step boundary values from the netCDF data.
        """
        if self.size == 1:
            return self._full_time_bnds
        data = self.comm.bcast(self._full_time_bnds, root=0)
        return data

    def get_full_levels(self) -> Dict[str, Any]:
        """
        Retrieve the complete vertical level information.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the complete vertical level data and its attributes.
            The dictionary structure is:
            {
                "data": ndarray,  # Array of vertical level values.
                attr_name: attr_value,  # Vertical level attributes.
                ...
            }
        """
        if self.size == 1:
            return self._full_lev
        data = self.comm.bcast(self._full_lev, root=0)
        return data

    def get_full_latitudes(self) -> Dict[str, Any]:
        """
        Retrieve the complete latitude information.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the complete latitude data and its attributes.
            The dictionary structure is:
            {
                "data": ndarray,  # Array of latitude values.
                attr_name: attr_value,  # Latitude attributes.
                ...
            }
        """
        if self.size == 1:
            return self._full_lat
        data = self.comm.bcast(self._full_lat, root=0)

        return data

    def get_full_longitudes(self) -> Dict[str, Any]:
        """
        Retrieve the complete longitude information.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the complete longitude data and its attributes.
            The dictionary structure is:
            {
                "data": ndarray,  # Array of longitude values.
                attr_name: attr_value,  # Longitude attributes.
                ...
            }
        """
        if self.size == 1:
            return self._full_lon
        data = self.comm.bcast(self._full_lon, root=0)
        return data

    def get_full_latitudes_boundaries(self) -> Dict[str, Any]:
        """
        Retrieve the complete latitude boundaries information.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the complete latitude boundaries data and its attributes.
            The dictionary structure is:
            {
                "data": ndarray,  # Array of latitude boundaries values.
                attr_name: attr_value,  # Latitude boundaries attributes.
                ...
            }
        """
        if self.size == 1:
            return self._full_lat_bnds
        data = self.comm.bcast(self._full_lat_bnds, root=0)
        return data

    def get_full_longitudes_boundaries(self) -> Dict[str, Any]:
        """
        Retrieve the complete longitude boundaries information.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the complete longitude boundaries data and its attributes.
            The dictionary structure is:
            {
                "data": ndarray,  # Array of longitude boundaries values.
                attr_name: attr_value,  # Longitude boundaries attributes.
                ...
            }
        """
        if self.size == 1:
            return self._full_lon_bnds
        data = self.comm.bcast(self._full_lon_bnds, root=0)
        return data

    def set_full_times(self, data: List[datetime]) -> None:
        """
        Set the complete list of original time step values.

        Parameters
        ----------
        data:List[datetime]
            The complete list of original time step values to set.
        """
        if self.master:
            self._full_time = data
        return None

    def set_full_time_bnds(self, data: List[datetime]) -> None:
        """
        Set the complete list of original time step boundaries.

        Parameters
        ----------
        data:List[datetime]
            The complete list of original time step boundary values to set.
        """
        if self.master:
            self._full_time_bnds = data
        return None

    def set_full_levels(self, data: Dict[str, Any]) -> None:
        """
        Set the complete vertical level information.

        Parameters
        ----------
        data:Dict[str, Any]
            A dictionary containing the complete vertical level data and its attributes.
            The dictionary structure is:
            {
                "data": ndarray,  # Array of vertical level values.
                attr_name: attr_value,  # Vertical level attributes.
                ...
            }
        """
        if self.master:
            self._full_lev = data
        return None

    def set_full_latitudes(self, data: Dict[str, Any]) -> None:
        """
        Set the complete latitude information.

        Parameters
        ----------
        data:Dict[str, Any]
            A dictionary containing the complete latitude data and its attributes.
            The dictionary structure is:
            {
                "data": ndarray,  # Array of latitude values.
                attr_name: attr_value,  # Latitude attributes.
                ...
            }
        """
        if self.master:
            self._full_lat = data
        return None

    def set_full_longitudes(self, data: Dict[str, Any]) -> None:
        """
        Set the complete longitude information.

        Parameters
        ----------
        data:Dict[str, Any]
            A dictionary containing the complete longitude data and its attributes.
            The dictionary structure is:
            {
                "data": ndarray,  # Array of longitude values.
                attr_name: attr_value,  # Longitude attributes.
                ...
            }
        """
        if self.master:
            self._full_lon = data
        return None

    def set_full_latitudes_boundaries(self, data: Dict[str, Any]) -> None:
        """
        Set the complete latitude boundaries information.

        Parameters
        ----------
        data:Dict[str, Any]
            A dictionary containing the complete latitude boundaries data and its attributes.
            The dictionary structure is:
            {
                "data": ndarray,  # Array of latitude boundaries values.
                attr_name: attr_value,  # Latitude boundaries attributes.
                ...
            }
        """
        if self.master:
            self._full_lat_bnds = data
        return None

    def set_full_longitudes_boundaries(self, data: Dict[str, Any]) -> None:
        """
        Set the complete longitude boundaries information.

        Parameters
        ----------
        data:Dict[str, Any]
            A dictionary containing the complete longitude boundaries data and its attributes.
            The dictionary structure is:
            {
                "data": ndarray,  # Array of longitude boundaries values.
                attr_name: attr_value,  # Longitude boundaries attributes.
                ...
            }
        """
        if self.master:
            self._full_lon_bnds = data

        return None

    def get_fids(self, use_read=False):
        """
        Obtain the FIDs in a 2D format.

        Parameters
        ----------
        use_read:bool
            Indicate if you want to use the read_axis_limits

        Returns
        -------
        fids: Array
            2D array with the FID data.
        """
        if self.master:
            fids = arange(
                self._full_lat["data"].shape[0] * self._full_lon["data"].shape[-1]
            )
            fids = fids.reshape(
                (self._full_lat["data"].shape[0], self._full_lon["data"].shape[-1])
            )
            if self.size == 1:
                return fids
        else:
            fids = None

        fids = self.comm.bcast(fids, root=0)

        if use_read:
            fids = fids[
                self.read_axis_limits["y_min"]: self.read_axis_limits["y_max"],
                self.read_axis_limits["x_min"]: self.read_axis_limits["x_max"],
            ]
        else:
            try:
                fids = fids[
                    self.write_axis_limits["y_min"]: self.write_axis_limits["y_max"],
                    self.write_axis_limits["x_min"]: self.write_axis_limits["x_max"],
                ]
            except TypeError as e:
                print(self.rank, fids, self.write_axis_limits)
                sys.stdout.flush()
                raise e
        return fids

    def get_full_shape(self):
        """
        Obtain the Full 2D shape of tha data

        Returns
        -------
        tuple
            2D shape of tha data.
        """
        if self.master:
            shape = (self._full_lat["data"].shape[0], self._full_lon["data"].shape[-1])
            if self.size == 1:
                return shape
        else:
            shape = None
        shape = self.comm.bcast(shape, root=0)

        return shape

    def set_level_direction(self, new_direction):
        """
        Set the direction of the vertical level values.

        Parameters
        ----------
        new_direction:str
            The new direction for the vertical levels. Must be either "up" or "down".

        Returns
        -------
        bool
            True if the direction was set successfully.

        Raises
        ------
        ValueError
            If `new_direction` is not "up" or "down".
        """
        if new_direction not in ["up", "down"]:
            raise ValueError(
                f"Level direction mus be up or down. '{new_direction}' is not a valid option"
            )
        if self.master:
            self._full_lev["positive"] = new_direction
        self.lev["positive"] = new_direction

        return True

    def reverse_level_direction(self):
        """
        Reverse the current direction of the vertical level values.

        Returns
        -------
        bool
            True if the direction was reversed successfully.
        """
        if "positive" in self.lev.keys():
            if self.lev["positive"] == "up":
                if self.master:
                    self._full_lev["positive"] = "down"
                self.lev["positive"] = "down"
            else:
                if self.master:
                    self._full_lev["positive"] = "up"
                self.lev["positive"] = "up"
        return True

    def clear_communicator(self):
        """
        Erase the communicator and the parallelization indexes.
        """

        self.comm = None
        self.rank = 0
        self.master = 0
        self.size = 0

        return None

    def set_communicator(self, comm):
        """
        Set a new communicator and the correspondent parallelization indexes.

        Parameters
        ----------
        comm: MPI.COMM
            Communicator to be set.
        """

        self.comm = comm
        self.rank = comm.Get_rank()
        self.master = comm.Get_rank() == 0
        self.size = comm.Get_size()

        self.read_axis_limits = self._get_read_axis_limits()
        self.write_axis_limits = self._get_write_axis_limits()

        return None

    def set_climatology(self, is_climatology):
        """
        Set whether the dataset represents climatological data.

        Parameters
        ----------
        is_climatology:bool
            A boolean indicating if the dataset represents climatological data.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If `is_climatology` is not a boolean.
        """
        if not isinstance(is_climatology, bool):
            raise TypeError("Only boolean values are accepted")
        self._climatology = is_climatology
        return None

    def get_climatology(self):
        """
        Get whether the dataset represents climatological data.

        Returns
        -------
        bool
            True if the dataset represents climatological data, False otherwise.
        """
        return self._climatology

    def set_levels(self, levels):
        """
        Modify the original level values with new ones.

        Parameters
        ----------
        levels:dict
            Dictionary with the new level information to be set.
        """
        self.set_full_levels(deepcopy(levels))
        self.lev = deepcopy(levels)

        return None

    def set_time(self, time_list):
        """
        Modify the original level values with new ones.

        Parameters
        ----------
        time_list:List[datetime]
            List of time steps
        """
        if self.parallel_method == "T":
            raise TypeError("Cannot set time on a 'T' parallel method")
        self.set_full_times(deepcopy(time_list))
        self.time = deepcopy(time_list)

        return None

    def set_time_bnds(self, time_bnds):
        """
        Modify the original time bounds values with new ones.

        Parameters
        ----------
        time_bnds:List
            AList with the new time bounds information to be set.
        """

        correct_format = True
        for time_bnd in array(time_bnds).flatten():
            if not isinstance(time_bnd, datetime):
                print("{0} is not a datetime object".format(time_bnd))
                correct_format = False
        if correct_format:
            if len(self.get_full_times()) == len(time_bnds):
                self.set_full_time_bnds(deepcopy(time_bnds))
                self.time_bnds = deepcopy(time_bnds)
            else:
                msg = "WARNING!!! "
                msg += "The given time bounds list has a different length than the time array. "
                msg += "(time:{0}, bnds:{1}). Time bounds will not be set.".format(
                    len(self.time), len(time_bnds)
                )
                warn(msg)
                sys.stderr.flush()
        else:
            msg = "WARNING!!! "
            msg += "There is at least one element in the time bounds to be set that is not a datetime object. "
            msg += "Time bounds will not be set."
            warn(msg)
            sys.stderr.flush()

        return None

    def set_time_resolution(self, new_resolution):
        """
        Set the time resolution for the dataset.

        Parameters
        ----------
        new_resolution:str
            The new time resolution. Accepted values are "second", "seconds", "minute", "minutes",
            "hour", "hours", "day", "days".

        Returns
        -------
        bool
            True if the time resolution was set successfully.

        Raises
        ------
        ValueError
            If `new_resolution` is not one of the accepted values.
        """
        accepted_resolutions = [
            "second",
            "seconds",
            "minute",
            "minutes",
            "hour",
            "hours",
            "day",
            "days",
        ]
        if new_resolution in accepted_resolutions:
            self._time_resolution = new_resolution
        else:
            raise ValueError(
                f"Time resolution '{new_resolution}' is not accepted. "
                + f"Use one of this: {accepted_resolutions}"
            )
        return True

    @staticmethod
    def _create_single_spatial_bounds(coordinates, inc, spatial_nv=2, inverse=False):
        """
        Calculate the vertices coordinates.

        Parameters
        ----------
        coordinates:Array
            Coordinates in degrees (latitude or longitude).
        inc:float
            Increment between centre values.
        spatial_nv:int
            Non-mandatory parameter that informs the number of vertices that the boundaries must have. Default: 2.
        inverse:bool
            For some grid latitudes.

        Returns
        ----------
        bounds:array
            An Array with as many elements as vertices for each value of coords.
        """

        # Create new arrays moving the centres half increment less and more.
        coords_left = coordinates - inc / 2
        coords_right = coordinates + inc / 2

        # Defining the number of corners needed. 2 to regular grids and 4 for irregular ones.
        if spatial_nv == 2:
            # Create an array of N arrays of 2 elements to store the floor and the ceil values for each cell
            bounds = dstack((coords_left, coords_right))
            bounds = bounds.reshape((len(coordinates), spatial_nv))
        elif spatial_nv == 4:
            # Create an array of N arrays of 4 elements to store the corner values for each cell
            # It can be stored in clockwise starting form the left-top element, or in inverse mode.
            if inverse:
                bounds = dstack((coords_left, coords_left, coords_right, coords_right))
            else:
                bounds = dstack((coords_left, coords_right, coords_right, coords_left))
        else:
            raise ValueError("The number of vertices of the boundaries must be 2 or 4.")

        return bounds

    def create_spatial_bounds(self):
        """
        Calculate longitude and latitude bounds and set them.
        """
        # Latitudes
        full_lat = self.get_full_latitudes()
        inc_lat = abs(mean(diff(full_lat["data"])))
        lat_bnds = self._create_single_spatial_bounds(
            full_lat["data"], inc_lat, spatial_nv=2
        )

        self.set_full_latitudes_boundaries({"data": deepcopy(lat_bnds)})
        self.lat_bnds = {
            "data": lat_bnds[
                self.write_axis_limits["y_min"]: self.write_axis_limits["y_max"], :
            ]
        }

        # Longitudes
        full_lon = self.get_full_longitudes()
        inc_lon = abs(mean(diff(full_lon["data"])))
        lon_bnds = self._create_single_spatial_bounds(
            full_lon["data"], inc_lon, spatial_nv=2
        )

        self.set_full_longitudes_boundaries({"data": deepcopy(lon_bnds)})
        self.lon_bnds = {
            "data": lon_bnds[
                self.write_axis_limits["x_min"]: self.write_axis_limits["x_max"], :
            ]
        }

        return None

    def get_spatial_bounds_mesh_format(self):
        """
        Get the spatial bounds in the pcolormesh format:

        see: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.pcolormesh.html

        Returns
        -------
        lon_bnds_mesh:numpy.ndarray
            Longitude boundaries in the mesh format
        lat_bnds_mesh:numpy.ndarray
            Latitude boundaries in the mesh format
        """
        if self.size > 1:
            raise RuntimeError(
                "NES.get_spatial_bounds_mesh_format() function only works in serial mode."
            )
        if self.lat_bnds is None:
            self.create_spatial_bounds()

        if self.lat_bnds["data"].shape[-1] == 2:
            # get the lat_b and lon_b first rows
            lat_b_0 = append(self.lat_bnds["data"][:, 0], self.lat_bnds["data"][-1, -1])
            lon_b_0 = append(self.lon_bnds["data"][:, 0], self.lon_bnds["data"][-1, -1])
            # expand lat_band lon_b in 2D
            lat_bnds_mesh = tile(lat_b_0, (len(self.lon["data"]) + 1, 1)).transpose()
            lon_bnds_mesh = tile(lon_b_0, (len(self.lat["data"]) + 1, 1))

        elif self.lat_bnds["data"].shape[-1] == 4:
            # Irregular quadrilateral polygon cell definition
            lat_bnds_mesh = empty(
                (self.lat["data"].shape[0] + 1, self.lat["data"].shape[1] + 1)
            )
            lat_bnds_mesh[:-1, :-1] = self.lat_bnds["data"][:, :, 0]
            lat_bnds_mesh[:-1, 1:] = self.lat_bnds["data"][:, :, 1]
            lat_bnds_mesh[1:, 1:] = self.lat_bnds["data"][:, :, 2]
            lat_bnds_mesh[1:, :-1] = self.lat_bnds["data"][:, :, 3]

            lon_bnds_mesh = empty(
                (self.lat["data"].shape[0] + 1, self.lat["data"].shape[1] + 1)
            )
            lon_bnds_mesh[:-1, :-1] = self.lon_bnds["data"][:, :, 0]
            lon_bnds_mesh[:-1, 1:] = self.lon_bnds["data"][:, :, 1]
            lon_bnds_mesh[1:, 1:] = self.lon_bnds["data"][:, :, 2]
            lon_bnds_mesh[1:, :-1] = self.lon_bnds["data"][:, :, 3]
        else:
            raise RuntimeError(
                "Invalid number of vertices: {0}".format(
                    self.lat_bnds["data"].shape[-1]
                )
            )

        return lon_bnds_mesh, lat_bnds_mesh

    def free_vars(self, var_list):
        """
        Erase the selected variables from the variables' information.

        Parameters
        ----------
        var_list:List or str
            List (or single string) of the variables to be loaded.
        """

        if isinstance(var_list, str):
            var_list = [var_list]

        if self.variables is not None:
            for var_name in var_list:
                if var_name in self.variables:
                    if "data" in self.variables[var_name].keys():
                        del self.variables[var_name]["data"]
                    del self.variables[var_name]
        collect()

        return None

    def keep_vars(self, var_list):
        """
        Keep the selected variables and erases the rest.

        Parameters
        ----------
        var_list:List or str
            List (or single string) of the variables to be loaded.
        """

        if isinstance(var_list, str):
            var_list = [var_list]

        to_remove = list(set(self.variables.keys()).difference(set(var_list)))

        self.free_vars(to_remove)

        return None

    @property
    def get_time_interval(self):
        """
        Calculate the interrval of hours between time steps.

        Returns
        -------
        int
            Number of hours between time steps.
        """
        if self.master:
            time_interval = self._full_time[1] - self._full_time[0]
            time_interval = int(time_interval.seconds // 3600)
        else:
            time_interval = None

        return self.comm.bcast(time_interval, root=0)

    def sel_time(self, time, inplace=True):
        """
        To select only one time step.

        Parameters
        ----------
        time:datetime
            Time stamp to select.
        inplace:bool
            Indicates if you want a copy with the selected time step (False) or to modify te existing one (True).

        Returns
        -------
        Nes
            A Nes object with the data (and metadata) of the selected time step.
        """

        if not inplace:
            aux_nessy = self.copy(copy_vars=False)
            aux_nessy.comm = self.comm
        else:
            aux_nessy = self

        aux_nessy.hours_start = 0
        aux_nessy.hours_end = 0

        idx_time = aux_nessy.time.index(time)

        aux_nessy.time = [self.time[idx_time]]
        aux_nessy._full_time = aux_nessy.time
        for var_name, var_info in self.variables.items():
            if copy:
                aux_nessy.variables[var_name] = {}
                for att_name, att_value in var_info.items():
                    if att_name == "data":
                        if att_value is None:
                            raise ValueError("{} data not loaded".format(var_name))
                        aux_nessy.variables[var_name][att_name] = att_value[[idx_time]]
                    else:
                        aux_nessy.variables[var_name][att_name] = att_value
            else:
                aux_nessy.variables[var_name]["data"] = aux_nessy.variables[var_name][
                    "data"
                ][[idx_time]]

        return aux_nessy

    def sel(
        self,
        hours_start=None,
        time_min=None,
        hours_end=None,
        time_max=None,
        lev_min=None,
        lev_max=None,
        lat_min=None,
        lat_max=None,
        lon_min=None,
        lon_max=None,
    ):
        """
        Select a slice of time, vertical level, latitude, or longitude given minimum and maximum limits.

        Parameters
        ----------
        hours_start:int, optional
            The number of hours from the start to begin the selection.
        time_min:datetime, optional
            The minimum datetime for the time selection. Mutually exclusive with `hours_start`.
        hours_end:int, optional
            The number of hours from the end to end the selection.
        time_max:datetime, optional
            The maximum datetime for the time selection. Mutually exclusive with `hours_end`.
        lev_min:int, optional
            The minimum vertical level index for the selection.
        lev_max:int, optional
            The maximum vertical level index for the selection.
        lat_min:float, optional
            The minimum latitude for the selection.
        lat_max:float, optional
            The maximum latitude for the selection.
        lon_min:float, optional
            The minimum longitude for the selection.
        lon_max:float, optional
            The maximum longitude for the selection.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If any variables are already loaded or if mutually exclusive parameters are both provided.

        Notes
        -----
        This method updates the selection criteria for the dataset and recalculates the read and write axis limits
        accordingly. It also updates the time, level, latitude, and longitude slices based on the new criteria.
        """
        full_time = self.get_full_times()
        loaded_vars = False
        for var_info in self.variables.values():
            if var_info["data"] is not None:
                loaded_vars = True
        if loaded_vars:
            raise ValueError(
                "Some variables have been loaded. Use select function before load."
            )

        # First time filter
        if hours_start is not None:
            if time_min is not None:
                raise ValueError(
                    "Choose to select by hours_start or time_min but not both"
                )
            self.hours_start = hours_start
        elif time_min is not None:
            if time_min <= full_time[0]:
                self.hours_start = 0
            else:
                self.hours_start = int(
                    (time_min - full_time[0]).total_seconds() // 3600
                )

        # Last time filter
        if hours_end is not None:
            if time_max is not None:
                raise ValueError(
                    "Choose to select by hours_end or time_max but not both"
                )
            self.hours_end = hours_end
        elif time_max is not None:
            if time_max >= full_time[-1]:
                self.hours_end = 0
            else:
                self.hours_end = int((full_time[-1] - time_max).total_seconds() // 3600)

        # Level filter
        self.first_level = lev_min
        self.last_level = lev_max

        # Coordinate filter
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max

        # New axis limits
        self.read_axis_limits = self._get_read_axis_limits()

        # Dimensions screening
        self.time = self.get_full_times()[
            self.read_axis_limits["t_min"]: self.read_axis_limits["t_max"]
        ]
        self.time_bnds = self.get_full_time_bnds()
        self.lev = self._get_coordinate_values(self.get_full_levels(), "Z")
        self.lat = self._get_coordinate_values(self.get_full_latitudes(), "Y")
        self.lon = self._get_coordinate_values(self.get_full_longitudes(), "X")

        self.lat_bnds = self._get_coordinate_values(
            self.get_full_latitudes_boundaries(), "Y", bounds=True
        )
        self.lon_bnds = self._get_coordinate_values(
            self.get_full_longitudes_boundaries(), "X", bounds=True
        )

        # Filter dimensions
        self._filter_coordinates_selection()

        # Removing complete coordinates
        self.write_axis_limits = self._get_write_axis_limits()

        return None

    def expand(self, n_cells: int = 1) -> None:
        """
        Expands the grid by increasing the number of cells in the spatial dimensions.

        Parameters
        ----------
        n_cells:int, optional
            Number of cells to expand in each direction. Default is 1.

        Returns
        -------
        None
        """
        is_first = self.rank == 0
        is_last = self.rank == self.size - 1

        self._expand_coordinates(n_cells, is_first, is_last)
        self._expand_variables(n_cells, is_first, is_last)
        self._expand_full_coordinates(n_cells)
        self._update_axis_limits(
            expand=True, n_cells=n_cells, is_first=is_first, is_last=is_last
        )
        self._post_expand_processing()

    def contract(self, n_cells: int = 1) -> None:
        """
        Contracts the grid by removing `n_cells` from the spatial dimensions.

        Parameters
        ----------
        n_cells:int, optional
            The number of cells to remove in each direction. Default is 1.

        Returns
        -------
        None
        """
        is_first = self.rank == 0
        is_last = self.rank == self.size - 1

        self._contract_coordinates(n_cells, is_first, is_last)
        self._contract_variables(n_cells, is_first, is_last)
        self._contract_full_coordinates(n_cells)
        self._update_axis_limits(
            expand=False, n_cells=n_cells, is_first=is_first, is_last=is_last
        )
        self._post_expand_processing()

    def _expand_coordinates(self, n_cells: int, is_first: bool, is_last: bool) -> None:
        """
        Expands the latitude and longitude coordinates.

        Parameters
        ----------
        n_cells:int
            Number of cells to expand.
        is_first:bool
            Whether the current rank is the first in parallel processing.
        is_last:bool
            Whether the current rank is the last in parallel processing.

        Returns
        -------
        None
        """
        raise NotImplementedError("It is not possible to expand Default coordinates")

    def _contract_coordinates(
        self, n_cells: int, is_first: bool, is_last: bool
    ) -> None:
        """
        Contracts latitude and longitude coordinates by removing `n_cells` from the borders.

        Parameters
        ----------
        n_cells:int
            The number of cells to remove from each border.
        is_first:bool
            Indicates whether the current rank is the first in parallel processing.
        is_last:bool
            Indicates whether the current rank is the last in parallel processing.

        Returns
        -------
        None
        """
        raise NotImplementedError("It is not possible to contract Default coordinates")

    def _expand_full_coordinates(self, n_cells: int) -> None:
        """
        Expands the full coordinates if `self.master` is True.

        Parameters
        ----------
        n_cells:int
            Number of cells to expand.

        Returns
        -------
        None
        """
        raise NotImplementedError(
            "It is not possible to expand Default full coordinates"
        )

    def _contract_full_coordinates(self, n_cells: int) -> None:
        """
        Contracts full latitude and longitude coordinates if `self.master` is True.

        Parameters
        ----------
        n_cells:int
            The number of cells to remove from each side.

        Returns
        -------
        None
        """
        raise NotImplementedError(
            "It is not possible to contract Default full coordinates"
        )

    def _expand_variables(self, n_cells: int, is_first: bool, is_last: bool) -> None:
        """
        Expands the grid variables.

        Parameters
        ----------
        n_cells:int
            Number of cells to expand.
        is_first:bool
            Whether the current rank is the first in parallel processing.
        is_last:bool
            Whether the current rank is the last in parallel processing.

        Returns
        -------
        None
        """
        expand_x = self.parallel_method in ["X", None]
        expand_y = self.parallel_method in ["Y", None]

        for var_name, var_data in self.variables.items():
            if isinstance(self.variables[var_name]["data"], ndarray):
                self.variables[var_name]["data"] = self._expand_variables_data(
                    var_data["data"],
                    n_cells=n_cells,
                    left=is_first if expand_x else True,
                    right=is_last if expand_x else True,
                    top=is_last if expand_y else True,
                    bottom=is_first if expand_y else True,
                )
            else:
                self.variables[var_name]["data"] = var_data["data"]
        return None

    def _contract_variables(self, n_cells: int, is_first: bool, is_last: bool) -> None:
        """
        Contracts the grid variables by removing `n_cells` from the spatial dimensions.

        Parameters
        ----------
        n_cells:int
            The number of cells to remove.
        is_first:bool
            Indicates whether the current rank is the first in parallel processing.
        is_last:bool
            Indicates whether the current rank is the last in parallel processing.

        Returns
        -------
        None
        """
        contract_x = self.parallel_method in ["X", None]
        contract_y = self.parallel_method in ["Y", None]

        for var_name, var_data in self.variables.items():
            if isinstance(self.variables[var_name]["data"], ndarray):
                self.variables[var_name]["data"] = self._contract_variables_data(
                    var_data["data"],
                    n_cells=n_cells,
                    left=is_first if contract_x else True,
                    right=is_last if contract_x else True,
                    top=is_last if contract_y else True,
                    bottom=is_first if contract_y else True,
                )
            else:
                self.variables[var_name]["data"] = var_data["data"]
        return None

    @staticmethod
    def _expand_1d(
        coordinate: ndarray, n_cells: int, left: bool = True, right: bool = True
    ) -> ndarray or None:
        """
        Expands a 1D or 2D coordinate array by adding `n_cells` values on each side.

        If `coordinate` is 2D (bounds format), each row is expanded by extrapolating the first and last bounds.

        Parameters
        ----------
        coordinate:np.ndarray
            The original 1D or 2D array of coordinates.
        n_cells:int
            The number of cells to expand on each side.
        left:bool, optional
            If True, expand on the left side (default is True).
        right:bool, optional
            If True, expand on the right side (default is True).

        Returns
        -------
        np.ndarray or None
            The expanded coordinate array.

        Raises
        ------
        ValueError
            If `coordinate` is neither 1D nor 2D.

        Notes
        -----
        - Assumes uniform spacing between coordinates.
        - If `coordinate` is `None`, returns `None`.
        - If `n_cells` is 0 or neither `left` nor `right` are True, the array remains unchanged.
        - Supports both 1D (center points) and 2D (bounds) coordinates.

        Examples
        --------
        Expanding a 1D coordinate array:

        coords = np.array([10, 20, 30, 40])
        _expand_1d(None, coords, 2)
        array([ -10,   0,  10,  20,  30,  40,  50,  60])

        Expanding a 2D coordinate array (bounds):

        bounds = np.array([[10, 11], [20, 21], [30, 31], [40, 41]])
        _expand_1d(None, bounds, 2)
        array([[ -10,   -9],
               [   0,    1],
               [  10,   11],
               [  20,   21],
               [  30,   31],
               [  40,   41],
               [  50,   51],
               [  60,   61]])
        """
        if coordinate is None:
            return None

        if n_cells <= 0 or not (left or right):
            return coordinate  # No expansion needed

        if coordinate.ndim == 1:
            # Expansion for 1D coordinates
            step = coordinate[1] - coordinate[0]

            expanded_left = (
                arange(coordinate[0] - n_cells * step, coordinate[0], step)
                if left
                else empty((0,))
            )
            expanded_right = (
                arange(
                    coordinate[-1] + step, coordinate[-1] + (n_cells + 1) * step, step
                )
                if right
                else empty((0,))
            )

            result = concatenate([expanded_left, coordinate, expanded_right])

        elif coordinate.ndim == 2:
            # Expansion for 2D coordinates (bounds)
            step = coordinate[1, 0] - coordinate[0, 0]  # Assuming uniform step size
            expanded_left = (
                array(
                    [
                        [coordinate[0, 0] - i * step, coordinate[0, 1] - i * step]
                        for i in range(n_cells, 0, -1)
                    ]
                )
                if left
                else empty((0, 2))
            )
            expanded_right = (
                array(
                    [
                        [coordinate[-1, 0] + i * step, coordinate[-1, 1] + i * step]
                        for i in range(1, n_cells + 1)
                    ]
                )
                if right
                else empty((0, 2))
            )

            result = concatenate([expanded_left, coordinate, expanded_right])

        else:
            raise ValueError("Input coordinate array must be either 1D or 2D.")

        return result

    @staticmethod
    def _contract_1d(
        coordinate: ndarray, n_cells: int, left: bool = True, right: bool = True
    ) -> ndarray or None:
        """
        Contracts a 1D or 2D coordinate array by removing `n_cells` from each specified side.

        Parameters
        ----------
        coordinate:np.ndarray
            The input 1D or 2D array of coordinates.
        n_cells:int
            The number of cells to remove from each border.
        left:bool, optional
            If True, removes `n_cells` from the left side. Default is True.
        right:bool, optional
            If True, removes `n_cells` from the right side. Default is True.

        Returns
        -------
        np.ndarray or None
            The contracted coordinate array.

        Raises
        ------
        ValueError
            If `coordinate` is neither 1D nor 2D.
        IndexError
            If `n_cells` is too large and would remove all data.
        """
        if coordinate is None:
            return None

        if n_cells <= 0 or not (left or right):
            return coordinate  # No contraction needed

        if coordinate.ndim == 1:
            if n_cells * (left + right) >= coordinate.shape[0]:
                raise IndexError(
                    "n_cells is too large, would remove all elements in the 1D array."
                )
            result = coordinate[n_cells:] if left else coordinate
            result = result[:-n_cells] if right else result

        elif coordinate.ndim == 2:
            if n_cells * (left + right) >= coordinate.shape[0]:
                raise IndexError(
                    "n_cells is too large, would remove all rows in the 2D array."
                )
            result = coordinate[n_cells:] if left else coordinate
            result = result[:-n_cells] if right else result

        else:
            raise ValueError("Input coordinate array must be either 1D or 2D.")

        return result

    @staticmethod
    def _expand_2d(
        coordinate: ndarray,
        n_cells: int,
        left: bool = True,
        right: bool = True,
        top: bool = True,
        bottom: bool = True,
    ) -> ndarray or None:
        """
        Expands a 2D coordinate grid by adding `n_cells` on each specified side.

        Parameters
        ----------
        coordinate:np.ndarray
            A 2D array representing latitude or longitude grid points.
        n_cells:int
            The number of cells to expand on each side.
        left:bool, optional
            If True, expand on the left side (default is True).
        right:bool, optional
            If True, expand on the right side (default is True).
        top:bool, optional
            If True, expand on the top side (default is True).
        bottom:bool, optional
            If True, expand on the bottom side (default is True).

        Returns
        -------
        np.ndarray or None
            The expanded coordinate array with `n_cells` added on each selected side.
        """
        if coordinate is None:
            return None

        if n_cells <= 0 or not (left or right or top or bottom):
            return coordinate  # No expansion needed

        ny, nx = coordinate.shape

        # Expand left and right
        if left:
            left_extension = tile(coordinate[:, [0]], (1, n_cells))
            left_extension -= arange(n_cells, 0, -1).reshape(1, -1) * (
                coordinate[:, [1]] - coordinate[:, [0]]
            )
        else:
            left_extension = empty((ny, 0))

        if right:
            right_extension = tile(coordinate[:, [-1]], (1, n_cells))
            right_extension += arange(1, n_cells + 1).reshape(1, -1) * (
                coordinate[:, [-1]] - coordinate[:, [-2]]
            )
        else:
            right_extension = empty((ny, 0))

        expanded = (
            hstack([left_extension, coordinate, right_extension])
            if left or right
            else coordinate
        )

        # Expand top and bottom
        if top:
            top_extension = tile(expanded[[0], :], (n_cells, 1))
            top_extension -= arange(n_cells, 0, -1).reshape(-1, 1) * (
                expanded[[1], :] - expanded[[0], :]
            )
        else:
            top_extension = empty((0, expanded.shape[1]))

        if bottom:
            bottom_extension = tile(expanded[[-1], :], (n_cells, 1))
            bottom_extension += arange(1, n_cells + 1).reshape(-1, 1) * (
                expanded[[-1], :] - expanded[[-2], :]
            )
        else:
            bottom_extension = empty((0, expanded.shape[1]))

        expanded = (
            vstack([top_extension, expanded, bottom_extension])
            if top or bottom
            else expanded
        )

        return expanded

    @staticmethod
    def _expand_2d_bounds(
        coordinate: ndarray,
        n_cells: int,
        left: bool = True,
        right: bool = True,
        top: bool = True,
        bottom: bool = True,
    ) -> ndarray or None:
        """
        Expands a 3D coordinate grid (bounds format) by adding `n_cells` on each specified side.

        Parameters
        ----------
        coordinate:np.ndarray
            A 3D array where the last dimension contains 4 values representing bounds.
        n_cells:int
            The number of cells to expand on each side.
        left:bool, optional
            If True, expand on the left side (default is True).
        right:bool, optional
            If True, expand on the right side (default is True).
        top:bool, optional
            If True, expand on the top side (default is True).
        bottom:bool, optional
            If True, expand on the bottom side (default is True).

        Returns
        -------
        np.ndarray or None
            The expanded coordinate array with `n_cells` added on each selected side.
        """
        if coordinate is None:
            return None

        if coordinate.ndim != 3 or coordinate.shape[2] != 4:
            raise ValueError(
                "Input coordinate array must be 3D with the last dimension of size 4 (bounds format)."
            )

        if n_cells <= 0 or not (left or right or top or bottom):
            return coordinate  # No expansion needed

        ny, nx, nbounds = coordinate.shape

        # Expand left and right
        if left:
            left_extension = tile(coordinate[:, [0], :], (1, n_cells, 1))
            left_extension -= arange(n_cells, 0, -1).reshape(1, -1, 1) * (
                coordinate[:, [1], :] - coordinate[:, [0], :]
            )
        else:
            left_extension = empty((ny, 0, nbounds))

        if right:
            right_extension = tile(coordinate[:, [-1], :], (1, n_cells, 1))
            right_extension += arange(1, n_cells + 1).reshape(1, -1, 1) * (
                coordinate[:, [-1], :] - coordinate[:, [-2], :]
            )
        else:
            right_extension = empty((ny, 0, nbounds))

        expanded = (
            hstack([left_extension, coordinate, right_extension])
            if left or right
            else coordinate
        )

        # Expand top and bottom
        if top:
            top_extension = tile(expanded[[0], :, :], (n_cells, 1, 1))
            top_extension -= arange(n_cells, 0, -1).reshape(-1, 1, 1) * (
                expanded[[1], :, :] - expanded[[0], :, :]
            )
        else:
            top_extension = empty((0, expanded.shape[1], nbounds))

        if bottom:
            bottom_extension = tile(expanded[[-1], :, :], (n_cells, 1, 1))
            bottom_extension += arange(1, n_cells + 1).reshape(-1, 1, 1) * (
                expanded[[-1], :, :] - expanded[[-2], :, :]
            )
        else:
            bottom_extension = empty((0, expanded.shape[1], nbounds))

        expanded = (
            vstack([top_extension, expanded, bottom_extension])
            if top or bottom
            else expanded
        )

        return expanded

    @staticmethod
    def _contract_2d(
        coordinate: ndarray,
        n_cells: int,
        left: bool = True,
        right: bool = True,
        top: bool = True,
        bottom: bool = True,
    ) -> ndarray or None:
        """
        Contracts a 2D coordinate grid by removing `n_cells` from each specified side.

        Parameters
        ----------
        coordinate:np.ndarray
            A 2D array representing latitude or longitude grid points.
        n_cells:int
            The number of cells to remove from each side.
        left:bool, optional
            If True, remove `n_cells` from the left side (default is True).
        right:bool, optional
            If True, remove `n_cells` from the right side (default is True).
        top:bool, optional
            If True, remove `n_cells` from the top side (default is True).
        bottom:bool, optional
            If True, remove `n_cells` from the bottom side (default is True).

        Returns
        -------
        np.ndarray or None
            The contracted coordinate array with `n_cells` removed from each selected side.
        """
        if coordinate is None:
            return None

        if n_cells <= 0 or not (left or right or top or bottom):
            return coordinate  # No contraction needed

        ny, nx = coordinate.shape

        if left:
            coordinate = coordinate[:, n_cells:]
        if right:
            coordinate = coordinate[:, :-n_cells]
        if top:
            coordinate = coordinate[n_cells:, :]
        if bottom:
            coordinate = coordinate[:-n_cells, :]

        return coordinate

    @staticmethod
    def _contract_2d_bounds(
        coordinate: ndarray,
        n_cells: int,
        left: bool = True,
        right: bool = True,
        top: bool = True,
        bottom: bool = True,
    ) -> ndarray or None:
        """
        Contracts a 3D coordinate grid (bounds format) by removing `n_cells` from each specified side.

        Parameters
        ----------
        coordinate:np.ndarray
            A 3D array where the last dimension contains 4 values representing bounds.
        n_cells:int
            The number of cells to remove from each side.
        left:bool, optional
            If True, remove `n_cells` from the left side (default is True).
        right:bool, optional
            If True, remove `n_cells` from the right side (default is True).
        top:bool, optional
            If True, remove `n_cells` from the top side (default is True).
        bottom:bool, optional
            If True, remove `n_cells` from the bottom side (default is True).

        Returns
        -------
        np.ndarray or None
            The contracted coordinate array with `n_cells` removed from each selected side.
        """
        if coordinate is None:
            return None

        if coordinate.ndim != 3 or coordinate.shape[2] != 4:
            raise ValueError(
                "Input coordinate array must be 3D with the last dimension of size 4 (bounds format)."
            )

        if n_cells <= 0 or not (left or right or top or bottom):
            return coordinate  # No contraction needed

        ny, nx, nbounds = coordinate.shape

        if left:
            coordinate = coordinate[:, n_cells:, :]
        if right:
            coordinate = coordinate[:, :-n_cells, :]
        if top:
            coordinate = coordinate[n_cells:, :, :]
        if bottom:
            coordinate = coordinate[:-n_cells, :, :]

        return coordinate

    @staticmethod
    def _expand_variables_data(
        data: ndarray,
        n_cells: int,
        left: bool = True,
        right: bool = True,
        top: bool = True,
        bottom: bool = True,
    ) -> ndarray or None:
        """
        Expands a 4D data array by adding zero-filled cells along the spatial dimensions (Y and X).

        Parameters
        ----------
        data:np.ndarray
            The original 4D array of shape (time, level, Y, X).
        n_cells:int
            The number of cells to expand along each specified spatial dimension.
        left:bool, optional
            If True, expand along the left (X) dimension (default is True).
        right:bool, optional
            If True, expand along the right (X) dimension (default is True).
        top:bool, optional
            If True, expand along the top (Y) dimension (default is True).
        bottom:bool, optional
            If True, expand along the bottom (Y) dimension (default is True).

        Returns
        -------
        np.ndarray or None
            The expanded 4D data array with added zero-filled cells in the selected directions.

        Raises
        ------
        ValueError
            If `data` is not a 4D array.

        Notes
        -----
        - The function assumes the input array has the shape (time, level, Y, X).
        - Expands only the last two dimensions (Y and X) by adding zeros.
        - If `n_cells` is 0 or all expansion flags are False, the array remains unchanged.

        Examples
        --------
        Expanding a 4D array:

        data = np.ones((2, 3, 4, 5))  # Example 4D array
        expanded_data = _expand_variables(None, data, 1, left=True, right=True, top=True, bottom=True)
        expanded_data.shape
        (2, 3, 6, 7)  # Expanded by 1 cell in all directions
        """
        if data is None:
            return None

        if data.ndim != 4:
            raise ValueError("Input data must be a 4D array (time, level, Y, X).")

        time_dim, level_dim, y_dim, x_dim = data.shape

        if n_cells <= 0 or not (left or right or top or bottom):
            return data  # No expansion needed

        def zero_pad(shape):
            return zeros(shape, dtype=data.dtype)

        # Expand Y dimension (top and bottom)
        if top:
            top_padding = zero_pad((time_dim, level_dim, n_cells, x_dim))
            data = concatenate([top_padding, data], axis=2)
        if bottom:
            bottom_padding = zero_pad((time_dim, level_dim, n_cells, x_dim))
            data = concatenate([data, bottom_padding], axis=2)

        # Expand X dimension (left and right)
        if left:
            left_padding = zero_pad((time_dim, level_dim, data.shape[2], n_cells))
            data = concatenate([left_padding, data], axis=3)
        if right:
            right_padding = zero_pad((time_dim, level_dim, data.shape[2], n_cells))
            data = concatenate([data, right_padding], axis=3)

        return data

    @staticmethod
    def _contract_variables_data(
        data: ndarray,
        n_cells: int,
        left: bool = True,
        right: bool = True,
        top: bool = True,
        bottom: bool = True,
    ) -> ndarray or None:
        """
        Contracts a 4D data array by removing `n_cells` along the spatial dimensions (Y and X).

        Parameters
        ----------
        data:np.ndarray
            The original 4D array of shape (time, level, Y, X).
        n_cells:int
            The number of cells to remove along each specified spatial dimension.
        left:bool, optional
            If True, removes `n_cells` from the left (X) dimension. Default is True.
        right:bool, optional
            If True, removes `n_cells` from the right (X) dimension. Default is True.
        top:bool, optional
            If True, removes `n_cells` from the top (Y) dimension. Default is True.
        bottom:bool, optional
            If True, removes `n_cells` from the bottom (Y) dimension. Default is True.

        Returns
        -------
        np.ndarray or None
            The contracted 4D data array with removed cells in the selected directions.

        Raises
        ------
        ValueError
            If `data` is not a 4D array.
        IndexError
            If `n_cells` is too large and would remove all data.

        Notes
        -----
        - The function assumes the input array has the shape (time, level, Y, X).
        - Contracts only the last two dimensions (Y and X), keeping `time` and `level` unchanged.
        - If `n_cells` is 0 or all contraction flags are False, the function returns the array unchanged.
        """

        if data is None:
            return None

        if data.ndim != 4:
            raise ValueError("Input data must be a 4D array (time, level, Y, X).")

        time_dim, level_dim, y_dim, x_dim = data.shape

        if n_cells <= 0 or not (left or right or top or bottom):
            return data  # No contraction needed

        # Validate that n_cells does not exceed the available dimensions
        if n_cells >= y_dim and (top or bottom):
            raise IndexError(
                "n_cells is too large, would remove all rows in Y dimension."
            )
        if n_cells >= x_dim and (left or right):
            raise IndexError(
                "n_cells is too large, would remove all columns in X dimension."
            )

        # Contract Y dimension (top and bottom)
        if top:
            data = data[:, :, n_cells:, :]
        if bottom:
            data = data[:, :, :-n_cells, :]

        # Contract X dimension (left and right)
        if left:
            data = data[:, :, :, n_cells:]
        if right:
            data = data[:, :, :, :-n_cells]

        return data

    def _post_expand_processing(self) -> None:
        """
        Performs additional processing after expansion, including shapefile update and grid area calculation.

        Returns
        -------
        None
        """
        if self.shapefile is not None:
            self.create_shapefile(overwrite=True)
        if "cell_area" in self.cell_measures:
            self.calculate_grid_area(overwrite=True)

    def _update_axis_limits(
        self, expand: bool = True, n_cells: int = 1, is_first=True, is_last=True
    ) -> None:
        """
        Update the axis limits by expanding or contracting the read and write boundaries.

        Parameters
        ----------
        expand:bool, optional
            If True, expands the limits; if False, contracts them. Default is True.
        n_cells:int, optional
            Number of cells to expand or contract, default is 1.
        is_first:bool
            Whether the current rank is the first in parallel processing.
        is_last:bool
            Whether the current rank is the last in parallel processing.

        Returns
        -------
        None
        """
        self._adjust_read_axis_limits(
            n_cells=n_cells, expand=expand, is_first=is_first, is_last=is_last
        )
        self._adjust_write_axis_limits(
            n_cells=n_cells, expand=expand, is_first=is_first, is_last=is_last
        )

    def _adjust_axis_limits(
        self, axis_limits, n_cells: int, expand: bool, is_first=True, is_last=True
    ) -> dict:
        """
        Adjust read axis limits by expanding or contracting them based on parallelization method.

        Parameters
        ----------
        n_cells:int
            Number of cells to adjust.
        expand:bool
            If True, expands the limits; otherwise, contracts them.
        is_first:bool
            Whether the current rank is the first in parallel processing.
        is_last:bool
            Whether the current rank is the last in parallel processing.

        Returns
        -------
        dict
            Updated axis limits
        """
        sign = 1 if expand else -1

        if self.parallel_method == "Y":
            if is_first:
                axis_limits["y_min"] -= sign * n_cells
            axis_limits["x_min"] -= sign * n_cells
            if is_last:
                if axis_limits["y_max"] is not None:
                    axis_limits["y_max"] += sign * n_cells
            if axis_limits["x_max"] is not None:
                axis_limits["x_max"] += sign * n_cells
        elif self.parallel_method == "X":
            if is_first:
                axis_limits["x_min"] -= sign * n_cells
            axis_limits["y_min"] -= sign * n_cells
            if is_last:
                if axis_limits["x_max"] is not None:
                    axis_limits["x_max"] += sign * n_cells
            if axis_limits["y_max"] is not None:
                axis_limits["y_max"] += sign * n_cells
        else:
            axis_limits["y_min"] -= sign * n_cells
            axis_limits["x_min"] -= sign * n_cells
            if axis_limits["y_max"] is not None:
                axis_limits["y_max"] += sign * n_cells
            if axis_limits["x_max"] is not None:
                axis_limits["x_max"] += sign * n_cells

        if axis_limits["x_min"] < 0:
            if axis_limits["x_max"] is not None:
                axis_limits["x_max"] += abs(axis_limits["x_min"])
            axis_limits["x_min"] = 0
        if axis_limits["y_min"] < 0:
            if axis_limits["y_max"] is not None:
                axis_limits["y_max"] += abs(axis_limits["y_min"])
            axis_limits["y_min"] = 0
        return axis_limits

    def _adjust_read_axis_limits(
        self, n_cells: int, expand: bool, is_first=True, is_last=True
    ) -> None:
        """
        Adjust read axis limits by expanding or contracting them based on parallelization method.

        Parameters
        ----------
        n_cells:int
            Number of cells to adjust.
        expand:bool
            If True, expands the limits; otherwise, contracts them.
        is_first:bool
            Whether the current rank is the first in parallel processing.
        is_last:bool
            Whether the current rank is the last in parallel processing.

        Returns
        -------
        None
        """
        sign = 1 if expand else -1

        if self.parallel_method == "Y":
            if is_first:
                self.read_axis_limits["y_min"] -= sign * n_cells
            self.read_axis_limits["x_min"] -= sign * n_cells
            if is_last:
                self.read_axis_limits["y_max"] += sign * n_cells
            self.read_axis_limits["x_max"] += sign * n_cells
        elif self.parallel_method == "X":
            if is_first:
                self.read_axis_limits["x_min"] -= sign * n_cells
            self.read_axis_limits["y_min"] -= sign * n_cells
            if is_last:
                self.read_axis_limits["x_max"] += sign * n_cells
            self.read_axis_limits["y_max"] += sign * n_cells
        else:
            self.read_axis_limits["y_min"] -= sign * n_cells
            self.read_axis_limits["x_min"] -= sign * n_cells
            self.read_axis_limits["y_max"] += sign * n_cells
            self.read_axis_limits["x_max"] += sign * n_cells
        return None

    def _adjust_write_axis_limits(
        self, n_cells: int, expand: bool, is_first=True, is_last=True
    ) -> None:
        """
        Adjust write axis limits by expanding or contracting them.

        Parameters
        ----------
        n_cells:int
            Number of cells to adjust.
        expand:bool
            If True, expands the limits; otherwise, contracts them.

        is_first:bool
            Whether the current rank is the first in parallel processing.
        is_last:bool
            Whether the current rank is the last in parallel processing.

        Returns
        -------
        None
        """
        sign = 1 if expand else -1
        if self.parallel_method == "X":
            max_axis = ["x_max"]
            min_axis = ["x_min"]
        elif self.parallel_method == "Y":
            max_axis = ["y_max"]
            min_axis = ["y_min"]
        else:
            max_axis = ["y_max", "x_max"]
            min_axis = ["y_min", "x_min"]

        for axis in max_axis:
            if self.write_axis_limits[axis] is not None:
                if is_last:
                    self.write_axis_limits[axis] += sign * n_cells * 2
                else:
                    self.write_axis_limits[axis] += sign * n_cells
        for axis in min_axis:
            if self.write_axis_limits[axis] is not None:
                if is_first:
                    pass
                else:
                    self.write_axis_limits[axis] += sign * n_cells
        return None

    def get_global_limits(self, limits: dict) -> dict:
        """
        Compute the global axis limits across all MPI ranks.

        This function uses MPI reduce operations to determine the minimum and maximum
        values for each axis across all ranks. `None` values are temporarily replaced
        with `inf` or `-inf` to avoid issues during reduction and are restored afterward.

        Parameters
        ----------
        limits:dict
            A dictionary containing local axis limits for the current MPI rank.
            Expected keys:
            - `x_min`, `x_max`
            - `y_min`, `y_max`
            - `z_min`, `z_max`
            - `t_min`, `t_max`
            Values can be numerical or `None` (which will be processed accordingly).

        Returns
        -------
        dict
            A dictionary with the global minimum and maximum values for each axis,
            computed across all MPI ranks. `None` is returned for dimensions that
            were originally `None`.

        Notes
        -----
        - The function assumes `self.comm` is a valid `MPI.COMM_WORLD` communicator.
        - The reduction is performed only for numerical values; `None` values are handled explicitly.
        - The final dictionary is only meaningful in rank 0. Other ranks return an undefined result.
        """
        my_limits = deepcopy(limits)
        for key in my_limits:
            if my_limits[key] is None:
                my_limits[key] = float("inf") if "min" in key else float("-inf")

        # Perform MPI reduction to find global min and max values
        global_limits = {}
        for key in my_limits:
            if "min" in key:
                global_limits[key] = self.comm.reduce(
                    my_limits[key], op=MPI.MIN, root=0
                )
            elif "max" in key:
                global_limits[key] = self.comm.reduce(
                    my_limits[key], op=MPI.MAX, root=0
                )

        # Restore None for dimensions that were originally None
        if self.rank == 0:
            for key in global_limits:
                if global_limits[key] in [float("inf"), float("-inf")]:
                    global_limits[key] = None

        return global_limits

    def get_global_limits_b(self, limits: dict) -> dict:
        """
        Compute the global axis limits across all MPI ranks.

        This function uses MPI reduce operations to determine the minimum and maximum
        values for each axis across all ranks. `None` values are temporarily replaced
        with `inf` or `-inf` to avoid issues during reduction and are restored afterward.

        Parameters
        ----------
        limits:dict
            A dictionary containing local axis limits for the current MPI rank.
            Expected keys:
            - `x_min`, `x_max`
            - `y_min`, `y_max`
            - `z_min`, `z_max`
            - `t_min`, `t_max`
            Values can be numerical or `None` (which will be processed accordingly).

        Returns
        -------
        dict
            A dictionary with the global minimum and maximum values for each axis,
            computed across all MPI ranks. `None` is returned for dimensions that
            were originally `None`.

        Notes
        -----
        - The function assumes `self.comm` is a valid `MPI.COMM_WORLD` communicator.
        - The reduction is performed only for numerical values; `None` values are handled explicitly.
        - The final dictionary is only meaningful in rank 0. Other ranks return an undefined result.
        """
        my_limits = deepcopy(limits)
        for key in my_limits:
            if my_limits[key] is None:
                my_limits[key] = float("inf") if "min" in key else float("-inf")

        # Perform MPI reduction to find global min and max values
        global_limits = {}
        for key in my_limits:
            if "min" in key:
                global_limits[key] = self.comm.reduce(
                    my_limits[key], op=MPI.MIN, root=0
                )
            elif "max" in key:
                global_limits[key] = self.comm.reduce(
                    my_limits[key], op=MPI.MAX, root=0
                )

        # Restore None for dimensions that were originally None
        if self.rank == 0:
            for key in global_limits:
                if global_limits[key] in [float("inf"), float("-inf")]:
                    global_limits[key] = None

        return global_limits

    def _filter_coordinates_selection(self):
        """
        Use the selection limits to filter time, lev, lat, lon, lon_bnds and lat_bnds.
        """
        global_limits = self.get_global_limits(self.read_axis_limits)

        if self.master:
            self._full_time = self._full_time[
                global_limits["t_min"]: global_limits["t_max"]
            ]
            self._full_lev["data"] = self._full_lev["data"][
                global_limits["z_min"]: global_limits["z_max"]
            ]

            if len(self._full_lat["data"].shape) == 1:
                # Regular projection
                self._full_lat["data"] = self._full_lat["data"][
                    global_limits["y_min"]: global_limits["y_max"]
                ]
                if global_limits["x_min"] < 0:
                    self._full_lon["data"] = concatenate(
                        (
                            self._full_lon["data"][global_limits["x_min"]:],
                            self._full_lon["data"][: global_limits["x_max"]],
                        )
                    )
                    self._full_lon["data"][self._full_lon["data"] > 180] -= 360
                else:
                    self._full_lon["data"] = self._full_lon["data"][
                        global_limits["x_min"]: global_limits["x_max"]
                    ]

                if self._full_lat_bnds is not None:
                    self._full_lat_bnds["data"] = self._full_lat_bnds["data"][
                        global_limits["y_min"]: global_limits["y_max"], :
                    ]
                if self._full_lon_bnds is not None:
                    if global_limits["x_min"] < 0:
                        self._full_lon_bnds["data"] = concatenate(
                            (
                                self._full_lon_bnds["data"][
                                    global_limits["x_min"]:, :
                                ],
                                self._full_lon_bnds["data"][
                                    : global_limits["x_max"], :
                                ],
                            )
                        )

                        self._full_lon_bnds["data"][
                            self._full_lon_bnds["data"] > 180
                        ] -= 360
                    else:
                        self._full_lon_bnds["data"] = self._full_lon_bnds["data"][
                            global_limits["x_min"]: global_limits["x_max"], :
                        ]
            else:
                # Irregular projections
                if global_limits["x_min"] < 0:
                    self._full_lat["data"] = concatenate(
                        (
                            self._full_lat["data"][
                                global_limits["y_min"]: global_limits["y_max"],
                                global_limits["x_min"]:,
                            ],
                            self._full_lat["data"][
                                global_limits["y_min"]: global_limits["y_max"],
                                : global_limits["x_max"],
                            ],
                        ),
                        axis=1,
                    )

                    self._full_lon["data"] = concatenate(
                        (
                            self._full_lon["data"][
                                global_limits["y_min"]: global_limits["y_max"],
                                global_limits["x_min"]:,
                            ],
                            self._full_lon["data"][
                                global_limits["y_min"]: global_limits["y_max"],
                                : global_limits["x_max"],
                            ],
                        ),
                        axis=1,
                    )
                    self._full_lon["data"][self._full_lon["data"] > 180] -= 360

                    if self._full_lat_bnds is not None:
                        self._full_lat_bnds["data"] = concatenate(
                            (
                                self._full_lat_bnds["data"][
                                    global_limits["y_min"]: global_limits["y_max"],
                                    global_limits["x_min"]:,
                                    :,
                                ],
                                self._full_lat_bnds["data"][
                                    global_limits["y_min"]: global_limits["y_max"],
                                    : global_limits["x_max"],
                                    :,
                                ],
                            ),
                            axis=1,
                        )
                    if self._full_lon_bnds is not None:
                        self._full_lon_bnds["data"] = concatenate(
                            (
                                self._full_lon_bnds["data"][
                                    global_limits["y_min"]: global_limits["y_max"],
                                    global_limits["x_min"]:,
                                    :,
                                ],
                                self._full_lon_bnds["data"][
                                    global_limits["y_min"]: global_limits["y_max"],
                                    : global_limits["x_max"],
                                    :,
                                ],
                            ),
                            axis=1,
                        )
                        self._full_lon_bnds["data"][
                            self._full_lon_bnds["data"] > 180
                        ] -= 360
                else:
                    self._full_lat["data"] = self._full_lat["data"][
                        global_limits["y_min"]: global_limits["y_max"],
                        global_limits["x_min"]: global_limits["x_max"],
                    ]

                    self._full_lon["data"] = self._full_lon["data"][
                        global_limits["y_min"]: global_limits["y_max"],
                        global_limits["x_min"]: global_limits["x_max"],
                    ]

                    if self._full_lat_bnds is not None:
                        self._full_lat_bnds["data"] = self._full_lat_bnds["data"][
                            global_limits["y_min"]: global_limits["y_max"],
                            global_limits["x_min"]: global_limits["x_max"],
                            :,
                        ]
                    if self._full_lon_bnds is not None:
                        self._full_lon_bnds["data"] = self._full_lon_bnds["data"][
                            global_limits["y_min"]: global_limits["y_max"],
                            global_limits["x_min"]: global_limits["x_max"],
                            :,
                        ]

        self.hours_start = 0
        self.hours_end = 0
        self.last_level = None
        self.first_level = None
        self.lat_min = None
        self.lat_max = None
        self.lon_max = None
        self.lon_min = None

        return None

    def _get_projection_data(self, create_nes, **kwargs):
        """
        Retrieves projection data based on grid details.

        Parameters
        ----------
        create_nes:bool
            Flag indicating whether to create new object (True) or use existing (False).
        **kwargs:dict
            Additional keyword arguments for specifying projection details.
        """

        raise NotImplementedError("Must be implemented on inner class.")

    @staticmethod
    def _get_pyproj_projection():
        """
        Retrieves Pyproj projection data based on grid details.

        """

        raise NotImplementedError("Must be implemented on inner class.")

    def _get_idx_intervals(self):
        """
        Calculate the index intervals

        Returns
        -------
        dict
            Dictionary with the index intervals
        """
        full_lat = self.get_full_latitudes()
        full_lon = self.get_full_longitudes()
        idx = {
            "idx_t_min": self._get_time_id(self.hours_start, first=True),
            "idx_t_max": self._get_time_id(self.hours_end, first=False),
            "idx_z_min": self.first_level,
            "idx_z_max": self.last_level,
        }

        # Axis Y
        if self.lat_min is None:
            idx["idx_y_min"] = 0
        else:
            idx["idx_y_min"] = self._get_coordinate_id(
                full_lat["data"], self.lat_min, axis=0
            )
        if self.lat_max is None:
            idx["idx_y_max"] = full_lat["data"].shape[0]
        else:
            idx["idx_y_max"] = (
                self._get_coordinate_id(full_lat["data"], self.lat_max, axis=0) + 1
            )

        if idx["idx_y_min"] > idx["idx_y_max"]:
            idx_aux = copy(idx["idx_y_min"])
            idx["idx_y_min"] = idx["idx_y_max"]
            idx["idx_y_max"] = idx_aux

        # Axis X
        if self.lon_min is None:
            idx["idx_x_min"] = 0
        else:
            if len(full_lon["data"].shape) == 1:
                axis = 0
            else:
                axis = 1
            if self.lon_min < 0 and full_lon["data"].max() > 180:
                aux_xmin = self._get_coordinate_id(
                    full_lon["data"], self.lon_min + 360, axis=axis
                )
                idx["idx_x_min"] = -(self.lon["data"].shape[-1] - aux_xmin)
            else:
                idx["idx_x_min"] = self._get_coordinate_id(
                    full_lon["data"], self.lon_min, axis=axis
                )
        if self.lon_max is None:
            idx["idx_x_max"] = full_lon["data"].shape[-1]
        else:
            if len(full_lon["data"].shape) == 1:
                axis = 0
            else:
                axis = 1
            idx["idx_x_max"] = (
                self._get_coordinate_id(full_lon["data"], self.lon_max, axis=axis) + 1
            )

        if idx["idx_x_min"] > idx["idx_x_max"]:
            idx_aux = copy(idx["idx_x_min"])
            idx["idx_x_min"] = idx["idx_x_max"]
            idx["idx_x_max"] = idx_aux
        return idx

    # ==================================================================================================================
    #                                                 Statistics
    # ==================================================================================================================

    def last_time_step(self):
        """
        Modify variables to keep only the last time step.
        """

        if self.parallel_method == "T":
            raise NotImplementedError(
                "Statistics are not implemented on time axis parallelization method."
            )
        aux_time = self.get_full_times()[0].replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        self.set_full_times([aux_time])
        self.time = [aux_time]

        for var_name, var_info in self.variables.items():
            if var_info["data"] is None:
                self.load(var_name)
            aux_data = var_info["data"][-1, :]
            if len(aux_data.shape) == 3:
                aux_data = aux_data.reshape(
                    (1, aux_data.shape[0], aux_data.shape[1], aux_data.shape[2])
                )
            self.variables[var_name]["data"] = aux_data
        self.hours_start = 0
        self.hours_end = 0

        return None

    def daily_statistic(self, op, type_op="calendar"):
        """
        Calculate daily statistic.

        Parameters
        ----------
        op:str
            Statistic to perform. Accepted values: "max", "mean" and "min" and his corresponding nanmean, ....
        type_op:str
            Type of statistic to perform. Accepted values: "calendar", "alltsteps", and "withoutt0".
            - "calendar": Calculate the statistic using the time metadata. It will avoid single time step by day
                    calculations
            - "alltsteps": Calculate a single time statistic with all the time steps.
            - "withoutt0": Calculate a single time statistic with all the time steps avoiding the first one.
        """

        if self.parallel_method == "T":
            raise NotImplementedError(
                "Statistics are not implemented on time axis parallel method."
            )
        time_interval = self.get_time_interval
        if type_op == "calendar":
            day_list = [date_aux.day for date_aux in self.time]
            for var_name, var_info in self.variables.items():
                aux_time_bounds = []
                aux_time = []
                if var_info["data"] is None:
                    self.load(var_name)
                stat_data = None
                for day in unique(day_list):
                    idx_first = next(
                        i for i, val in enumerate(day_list, 0) if val == day
                    )
                    idx_last = len(day_list) - next(
                        i for i, val in enumerate(reversed(day_list), 1) if val == day
                    )
                    if idx_first != idx_last:  # To avoid single time step statistic
                        if idx_last != len(day_list):
                            if op == "mean":
                                data_aux = var_info["data"][
                                    idx_first: idx_last + 1, :, :, :
                                ].mean(axis=0)
                            elif op == "max":
                                data_aux = var_info["data"][
                                    idx_first: idx_last + 1, :, :, :
                                ].max(axis=0)
                            elif op == "min":
                                data_aux = var_info["data"][
                                    idx_first: idx_last + 1, :, :, :
                                ].min(axis=0)
                            elif op == "nanmean":
                                data_aux = nanmean(
                                    var_info["data"][idx_first: idx_last + 1, :, :, :],
                                    axis=0,
                                )
                            elif op == "nanmax":
                                data_aux = nanmax(
                                    var_info["data"][idx_first: idx_last + 1, :, :, :],
                                    axis=0,
                                )
                            elif op == "nanmin":
                                data_aux = nanmin(
                                    var_info["data"][idx_first: idx_last + 1, :, :, :],
                                    axis=0,
                                )
                            else:
                                raise NotImplementedError(
                                    f"Statistic operation '{op}' is not implemented."
                                )
                            aux_time_bounds.append(
                                [self.time[idx_first], self.time[idx_last]]
                            )
                        else:
                            if op == "mean":
                                data_aux = var_info["data"][idx_first:, :, :, :].mean(
                                    axis=0
                                )
                            elif op == "max":
                                data_aux = var_info["data"][idx_first:, :, :, :].max(
                                    axis=0
                                )
                            elif op == "min":
                                data_aux = var_info["data"][idx_first:, :, :, :].min(
                                    axis=0
                                )
                            elif op == "nanmean":
                                data_aux = nanmean(
                                    var_info["data"][idx_first:, :, :, :], axis=0
                                )
                            elif op == "nanmax":
                                data_aux = nanmax(
                                    var_info["data"][idx_first:, :, :, :], axis=0
                                )
                            elif op == "nanmin":
                                data_aux = nanmin(
                                    var_info["data"][idx_first:, :, :, :], axis=0
                                )
                            else:
                                raise NotImplementedError(
                                    f"Statistic operation '{op}' is not implemented."
                                )
                            aux_time_bounds.append(
                                [self.time[idx_first], self.time[-1]]
                            )

                        data_aux = data_aux.reshape(
                            (1, data_aux.shape[0], data_aux.shape[1], data_aux.shape[2])
                        )
                        aux_time.append(
                            self.time[idx_first].replace(hour=0, minute=0, second=0)
                        )
                        # Append over time dimension
                        if stat_data is None:
                            stat_data = data_aux.copy()
                        else:
                            stat_data = vstack([stat_data, data_aux])
                self.variables[var_name]["data"] = stat_data
                self.variables[var_name]["cell_methods"] = (
                    "time: {0} (interval: {1}hr)".format(op, time_interval)
                )
            self.time = aux_time
            self.set_full_times(self.time)

            self.set_time_bnds(aux_time_bounds)

        elif type_op == "alltsteps":
            for var_name, var_info in self.variables.items():
                if var_info["data"] is None:
                    self.load(var_name)
                if op == "mean":
                    aux_data = var_info["data"].mean(axis=0)
                elif op == "max":
                    aux_data = var_info["data"].max(axis=0)
                elif op == "min":
                    aux_data = var_info["data"].min(axis=0)
                elif op == "nanmean":
                    aux_data = nanmean(var_info["data"], axis=0)
                elif op == "nanmax":
                    aux_data = nanmax(var_info["data"], axis=0)
                elif op == "nanmin":
                    aux_data = nanmin(var_info["data"], axis=0)
                else:
                    raise NotImplementedError(
                        f"Statistic operation '{op}' is not implemented."
                    )
                if len(aux_data.shape) == 3:
                    aux_data = aux_data.reshape(
                        (1, aux_data.shape[0], aux_data.shape[1], aux_data.shape[2])
                    )
                self.variables[var_name]["data"] = aux_data
                self.variables[var_name]["cell_methods"] = (
                    "time: {0} (interval: {1}hr)".format(op, time_interval)
                )

            aux_time = self.time[0].replace(hour=0, minute=0, second=0, microsecond=0)
            aux_time_bounds = [[self.time[0], self.time[-1]]]
            self.time = [aux_time]
            self.set_full_times(self.time)

            self.set_time_bnds(aux_time_bounds)

        elif type_op == "withoutt0":
            for var_name, var_info in self.variables.items():
                if var_info["data"] is None:
                    self.load(var_name)
                if op == "mean":
                    aux_data = var_info["data"][1:, :].mean(axis=0)
                elif op == "max":
                    aux_data = var_info["data"][1:, :].max(axis=0)
                elif op == "min":
                    aux_data = var_info["data"][1:, :].min(axis=0)
                elif op == "nanmean":
                    aux_data = nanmean(var_info["data"][1:, :], axis=0)
                elif op == "nanmax":
                    aux_data = nanmax(var_info["data"][1:, :], axis=0)
                elif op == "nanmin":
                    aux_data = nanmin(var_info["data"][1:, :], axis=0)
                else:
                    raise NotImplementedError(
                        f"Statistic operation '{op}' is not implemented."
                    )
                if len(aux_data.shape) == 3:
                    aux_data = aux_data.reshape(
                        (1, aux_data.shape[0], aux_data.shape[1], aux_data.shape[2])
                    )
                self.variables[var_name]["data"] = aux_data
                self.variables[var_name]["cell_methods"] = (
                    "time: {0} (interval: {1}hr)".format(op, time_interval)
                )
            full_time = self.get_full_times()
            aux_time = full_time[1].replace(hour=0, minute=0, second=0, microsecond=0)
            aux_time_bounds = [[full_time[1], full_time[-1]]]
            self.time = [aux_time]
            self.set_full_times(self.time)

            self.set_time_bnds(aux_time_bounds)
        else:
            raise NotImplementedError(
                f"Statistic operation type '{type_op}' is not implemented."
            )
        self.hours_start = 0
        self.hours_end = 0

        return None

    def get_totals(self) -> dict:
        """
        Compute per-variable global totals across all MPI ranks.

        This method assumes each variable's 4‑D data array (time, level, Y, X)
        is already loaded in memory (i.e., `self.variables[var]["data"] is not None`).
        It performs the following steps on every rank:

        1. **Local sum**: For each variable, compute the sum of all its data values
           on the current rank (ignoring non‑numeric variables by treating them as 0.0).
        2. **MPI reduction**: Reduce the local sums with `MPI.SUM` to rank 0 so that
           rank 0 holds the global totals per variable.
        3. **Broadcast**: Broadcast the resulting dictionary so that **all ranks**
           receive the same global totals.

        Returns
        -------
        dict
            Mapping from variable name (str) to its global total (float).

        Raises
        ------
        RuntimeError
            If any variable has not been loaded (`data is None`).

        Notes
        -----
        - Non‑numeric variables (e.g., strings) are safely ignored by assigning 0.0
          to their local contribution.
        - The reduction is performed variable‑by‑variable to keep memory usage low
          and avoid gathering large structures at once.
        """
        # 1) Compute local sums per variable (on each rank)
        local_totals = {}
        for var_name, var_info in self.variables.items():
            data = var_info.get("data")
            if data is None:
                # Raise error if data is not loaded for this variable
                raise RuntimeError(f"Data was not loaded for variable '{var_name}'")
            # Try to sum numeric arrays; if not summable (e.g., dtype=object/str), use 0.0
            try:
                local_val = float(nansum(data))
            except Exception:
                # If data is not numeric, treat as 0.0
                local_val = 0.0
            local_totals[var_name] = local_val

        if self.size == 1:
            global_totals = local_totals
        else:
            # 2) Reduce local sums to rank 0 (global totals)
            #    Keep a plain Python dict to avoid unnecessary large communications
            global_totals = {}
            var_names = list(self.variables.keys())
            for name in var_names:
                local_val = float(local_totals.get(name, 0.0))
                # Reduce across all ranks (MPI.SUM), result only valid on root
                reduced_val = self.comm.reduce(local_val, op=MPI.SUM, root=0)
                if self.rank == 0:
                    # On rank 0, store the globally reduced value for this variable
                    global_totals[name] = float(reduced_val)

            # 3) Broadcast the final dictionary so that all ranks return the same result
            global_totals = self.comm.bcast(
                global_totals if self.rank == 0 else None, root=0
            )

        return global_totals

    def get_min(self, var_name: str) -> float:
        """
        Compute the **global minimum** of a variable across all MPI ranks.

        The variable must already be loaded in memory, i.e.,
        `self.variables[var_name]["data"]` is a numeric NumPy array.
        NaN values are ignored when computing the minimum on each rank.

        Parameters
        ----------
        var_name:str
            Name of the variable stored in `self.variables`.

        Returns
        -------
        float
            The global minimum value of the variable across all ranks. If all
            values are NaN (on every rank) or the variable is non‑numeric, the
            function returns `nan`.

        Raises
        ------
        KeyError
            If `var_name` is not present in `self.variables`.
        RuntimeError
            If the variable's data has not been loaded (`data is None`).

        Notes
        -----
        * The reduction is implemented with `MPI.MIN` after converting NaNs to
          `+inf` locally, so NaNs do not contaminate the global minimum.
        * After the reduction on rank 0, the result is broadcast so that all
          ranks receive the same scalar value.
        """
        # --- Existence and load checks ---
        if var_name not in self.variables:
            raise KeyError(f"Variable '{var_name}' not found in self.variables")

        data = self.variables[var_name].get("data")
        if data is None:
            # Keep the error explicit: caller must ensure the data has been loaded
            raise RuntimeError(f"Data was not loaded for variable '{var_name}'")

        # --- Local minimum with NaN-safe handling ---
        # Use nanmin if possible; if data is non-numeric, fall back to NaN
        try:
            local_min = float(nanmin(data))
        except Exception:
            local_min = float("nan")

        # Replace NaN by +inf *before* the MIN reduction so it does not affect
        # the global minimum. If all ranks are NaN => all send +inf and we'll
        # convert back to NaN on root.
        val_for_reduce = local_min

        if isnan(val_for_reduce):
            val_for_reduce = float("inf")

        # --- MPI reduction using explicit NumPy buffers (faster than comm.reduce) ---
        sendbuf = array(val_for_reduce, dtype=float64)
        recvbuf = array(0.0, dtype=float64) if self.rank == 0 else None
        self.comm.Reduce(sendbuf, recvbuf, op=MPI.MIN, root=0)

        if self.rank == 0:
            global_min = recvbuf.item()
            if global_min == float("inf"):
                global_min = float("nan")
            result = float(global_min)
        else:
            result = None

        # --- Broadcast the final scalar to every rank ---
        result = self.comm.bcast(result, root=0)
        return result

    def get_max(self, var_name: str) -> float:
        """
        Compute the **global maximum** of a variable across all MPI ranks.

        The variable must already be loaded in memory, i.e.,
        `self.variables[var_name]["data"]` is a numeric NumPy array.
        NaN values are ignored when computing the maximum on each rank.

        Parameters
        ----------
        var_name:str
            Name of the variable stored in `self.variables`.

        Returns
        -------
        float
            The global maximum value of the variable across all ranks. If all
            values are NaN (on every rank) or the variable is non‑numeric, the
            function returns `nan`.

        Raises
        ------
        KeyError
            If `var_name` is not present in `self.variables`.
        RuntimeError
            If the variable's data has not been loaded (`data is None`).

        Notes
        -----
        * The reduction is implemented with `MPI.MAX` after converting NaNs to
          `-inf` locally, so NaNs do not contaminate the global maximum.
        * After the reduction on rank 0, the result is broadcast so that all
          ranks receive the same scalar value.
        """
        # --- Existence and load checks ---
        if var_name not in self.variables:
            raise KeyError(f"Variable '{var_name}' not found in self.variables")

        data = self.variables[var_name].get("data")
        if data is None:
            # Keep the error explicit: caller must ensure the data has been loaded
            raise RuntimeError(f"Data was not loaded for variable '{var_name}'")

        # --- Local minimum with NaN-safe handling ---
        # Use nanmin if possible; if data is non-numeric, fall back to NaN
        try:
            local_max = float(nanmax(data))
        except Exception:
            local_max = float("nan")

        # Replace NaN by +inf *before* the MIN reduction so it does not affect
        # the global minimum. If all ranks are NaN => all send +inf and we'll
        # convert back to NaN on root.
        val_for_reduce = local_max

        if isnan(val_for_reduce):
            val_for_reduce = float("-inf")

        # --- MPI reduction using explicit NumPy buffers (faster than comm.reduce) ---
        sendbuf = array(val_for_reduce, dtype=float64)
        recvbuf = array(0.0, dtype=float64) if self.rank == 0 else None
        self.comm.Reduce(sendbuf, recvbuf, op=MPI.MIN, root=0)

        if self.rank == 0:
            global_max = recvbuf.item()
            if global_max == float("-inf"):
                global_max = float("nan")
            result = float(global_max)
        else:
            result = None

        # --- Broadcast the final scalar to every rank ---
        result = self.comm.bcast(result, root=0)
        return result

    @staticmethod
    def _get_axis_index_(axis):
        if axis == "T":
            value = 0
        elif axis == "Z":
            value = 1
        elif axis == "Y":
            value = 2
        elif axis == "X":
            value = 3
        else:
            raise ValueError("Unknown axis: {0}".format(axis))

        return value

    def sum_axis(self, axis="Z"):
        if self.parallel_method == axis:
            raise NotImplementedError(
                f"It is not possible to sum the axis with it is parallelized '{self.parallel_method}'"
            )

        for var_name, var_info in self.variables.items():
            if var_info["data"] is not None:
                self.variables[var_name]["data"] = self.variables[var_name]["data"].sum(
                    axis=self._get_axis_index_(axis), keepdims=True
                )
                if axis == "T":
                    self.variables[var_name]["cell_methods"] = (
                        "time: sum (interval: {0}hr)".format(
                            (self.time[-1] - self.time[0]).total_seconds() // 3600
                        )
                    )

        if axis == "T":
            self.set_time_bnds([self.time[0], self.time[-1]])
            self.time = [self.time[0]]
            self.set_full_times([self.time[0]])
        if axis == "Z":
            self.lev["data"] = array([self.lev["data"][0]])
            self.set_full_levels(self.lev)

        return None

    def find_time_id(self, time):
        """
        Find index of time in time array.

        Parameters
        ----------
        time:datetime
            Time element.

        Returns
        -------
        int
            Index of time element.
        """

        if time in self.time:
            return self.time.index(time)

    def rolling_mean(self, var_list=None, hours=8):
        """
        Calculate rolling mean for given hours

        Parameters
        ----------
        var_list:: List, str, None
            List (or single string) of the variables to be loaded.
        hours:int, optional
            Window hours to calculate rolling mean, by default 8

        Returns
        -------
        Nes
             A Nes object
        """

        if self.parallel_method == "T":
            raise NotImplementedError(
                "The rolling mean cannot be calculated using the time axis parallel method."
            )

        aux_nessy = self.copy(copy_vars=False)
        aux_nessy.set_communicator(self.comm)

        if isinstance(var_list, str):
            var_list = [var_list]
        elif var_list is None:
            var_list = list(self.variables.keys())

        for var_name in var_list:
            # Load variables if they have not been loaded previously
            if self.variables[var_name]["data"] is None:
                self.load(var_name)

            # Get original file shape
            nessy_shape = self.variables[var_name]["data"].shape

            # Initialise array
            aux_nessy.variables[var_name] = {}
            aux_nessy.variables[var_name]["data"] = empty(shape=nessy_shape)
            aux_nessy.variables[var_name]["dimensions"] = deepcopy(
                self.variables[var_name]["dimensions"]
            )

            for curr_time in self.time:
                # Get previous time given a set of hours
                prev_time = curr_time - timedelta(hours=(hours - 1))

                # Get time indices
                curr_time_id = self.find_time_id(curr_time)
                prev_time_id = self.find_time_id(prev_time)

                # Get mean if previous time is available
                if prev_time_id is not None:
                    if self.info:
                        print(f"Calculating mean between {prev_time} and {curr_time}.")
                    aux_nessy.variables[var_name]["data"][
                        curr_time_id, :, :, :
                    ] = self.variables[var_name]["data"][
                        prev_time_id:curr_time_id, :, :, :
                    ].mean(
                        axis=0, keepdims=True
                    )
                # Fill with nan if previous time is not available
                else:
                    if self.info:
                        msg = f"Mean between {prev_time} and {curr_time} cannot be calculated "
                        msg += f"because data for {prev_time} is not available."
                        print(msg)
                    aux_nessy.variables[var_name]["data"][curr_time_id, :, :, :] = full(
                        shape=(1, nessy_shape[1], nessy_shape[2], nessy_shape[3]),
                        fill_value=nan,
                    )

        return aux_nessy

    # ==================================================================================================================
    #                                                 Reading
    # ==================================================================================================================

    def _get_read_axis_limits(self):
        """
        Calculate the 4D reading axis limits depending on if them have to balanced or not.

        Returns
        -------
        dict
            Dictionary with the 4D limits of the rank data to read.
            t_min, t_max, z_min, z_max, y_min, y_max, x_min and x_max.
        """

        if self.balanced:
            return self._get_read_axis_limits_balanced()
        else:
            return self._get_read_axis_limits_unbalanced()

    def _get_read_axis_limits_unbalanced(self):
        """
        Calculate the 4D reading axis limits.

        Returns
        -------
        dict
            Dictionary with the 4D limits of the rank data to read.
            t_min, t_max, z_min, z_max, y_min, y_max, x_min and x_max.
        """

        axis_limits = {
            "x_min": None,
            "x_max": None,
            "y_min": None,
            "y_max": None,
            "z_min": None,
            "z_max": None,
            "t_min": None,
            "t_max": None,
        }

        idx = self._get_idx_intervals()
        if self.parallel_method == "Y":
            y_len = idx["idx_y_max"] - idx["idx_y_min"]
            if y_len < self.size:
                raise IndexError(
                    "More processors (size={0}) selected than Y elements (size={1})".format(
                        self.size, y_len
                    )
                )
            axis_limits["y_min"] = ((y_len // self.size) * self.rank) + idx["idx_y_min"]
            if self.rank + 1 < self.size:
                axis_limits["y_max"] = ((y_len // self.size) * (self.rank + 1)) + idx[
                    "idx_y_min"
                ]
            else:
                axis_limits["y_max"] = idx["idx_y_max"]

            # Non parallel filters
            axis_limits["x_min"] = idx["idx_x_min"]
            axis_limits["x_max"] = idx["idx_x_max"]

            axis_limits["t_min"] = idx["idx_t_min"]
            axis_limits["t_max"] = idx["idx_t_max"]

        elif self.parallel_method == "X":
            x_len = idx["idx_x_max"] - idx["idx_x_min"]
            if x_len < self.size:
                raise IndexError(
                    "More processors (size={0}) selected than X elements (size={1})".format(
                        self.size, x_len
                    )
                )
            axis_limits["x_min"] = ((x_len // self.size) * self.rank) + idx["idx_x_min"]
            if self.rank + 1 < self.size:
                axis_limits["x_max"] = ((x_len // self.size) * (self.rank + 1)) + idx[
                    "idx_x_min"
                ]
            else:
                axis_limits["x_max"] = idx["idx_x_max"]

            # Non parallel filters
            axis_limits["y_min"] = idx["idx_y_min"]
            axis_limits["y_max"] = idx["idx_y_max"]

            axis_limits["t_min"] = idx["idx_t_min"]
            axis_limits["t_max"] = idx["idx_t_max"]

        elif self.parallel_method == "T":
            t_len = idx["idx_t_max"] - idx["idx_t_min"]
            if t_len < self.size:
                raise IndexError(
                    "More processors (size={0}) selected than T elements (size={1})".format(
                        self.size, t_len
                    )
                )
            axis_limits["t_min"] = ((t_len // self.size) * self.rank) + idx["idx_t_min"]
            if self.rank + 1 < self.size:
                axis_limits["t_max"] = ((t_len // self.size) * (self.rank + 1)) + idx[
                    "idx_t_min"
                ]

            # Non parallel filters
            axis_limits["y_min"] = idx["idx_y_min"]
            axis_limits["y_max"] = idx["idx_y_max"]

            axis_limits["x_min"] = idx["idx_x_min"]
            axis_limits["x_max"] = idx["idx_x_max"]

        else:
            raise NotImplementedError(
                "Parallel method '{meth}' is not implemented. Use one of these: {accept}".format(
                    meth=self.parallel_method, accept=["X", "Y", "T"]
                )
            )

        # Vertical levels selection:
        axis_limits["z_min"] = self.first_level
        if self.last_level == -1 or self.last_level is None:
            self.last_level = None
        elif self.last_level + 1 == len(self.get_full_levels()["data"]):
            self.last_level = None
        else:
            self.last_level += 1
        axis_limits["z_max"] = self.last_level

        return axis_limits

    def _get_read_axis_limits_balanced(self):
        """
        Calculate the 4D reading balanced axis limits.

        Returns
        -------
        dict
            Dictionary with the 4D limits of the rank data to read.
            t_min, t_max, z_min, z_max, y_min, y_max, x_min and x_max.
        """
        idx = self._get_idx_intervals()

        fid_dist = {}
        if self.parallel_method == "Y":
            len_to_split = idx["idx_y_max"] - idx["idx_y_min"]
            if len_to_split < self.size:
                raise IndexError(
                    "More processors (size={0}) selected than Y elements (size={1})".format(
                        self.size, len_to_split
                    )
                )
            min_axis = "y_min"
            max_axis = "y_max"
            to_add = idx["idx_y_min"]

        elif self.parallel_method == "X":
            len_to_split = idx["idx_x_max"] - idx["idx_x_min"]
            if len_to_split < self.size:
                raise IndexError(
                    "More processors (size={0}) selected than X elements (size={1})".format(
                        self.size, len_to_split
                    )
                )
            min_axis = "x_min"
            max_axis = "x_max"
            to_add = idx["idx_x_min"]
        elif self.parallel_method == "T":
            len_to_split = idx["idx_t_max"] - idx["idx_t_min"]
            if len_to_split < self.size:
                raise IndexError(
                    f"More processors (size={self.size}) selected than T elements (size={len_to_split})"
                )
            min_axis = "t_min"
            max_axis = "t_max"
            to_add = idx["idx_t_min"]
        else:
            raise NotImplementedError(
                "Parallel method '{meth}' is not implemented. Use one of these: {accept}".format(
                    meth=self.parallel_method, accept=["X", "Y", "T"]
                )
            )

        procs_len = len_to_split // self.size
        procs_rows_extended = len_to_split - (procs_len * self.size)

        rows_sum = 0
        for proc in range(self.size):
            fid_dist[proc] = {
                "x_min": 0,
                "x_max": None,
                "y_min": 0,
                "y_max": None,
                "z_min": 0,
                "z_max": None,
                "t_min": 0,
                "t_max": None,
            }
            if proc < procs_rows_extended:
                aux_rows = procs_len + 1
            else:
                aux_rows = procs_len

            len_to_split -= aux_rows
            if len_to_split < 0:
                rows = len_to_split + aux_rows
            else:
                rows = aux_rows

            fid_dist[proc][min_axis] = rows_sum
            fid_dist[proc][max_axis] = rows_sum + rows

            if to_add is not None:
                fid_dist[proc][min_axis] += to_add
                fid_dist[proc][max_axis] += to_add

            # # Last element
            # if len_to_split == 0 and to_add == 0:
            #     fid_dist[proc][max_axis] = None

            rows_sum += rows

        axis_limits = fid_dist[self.rank]

        # Non parallel filters
        if self.parallel_method != "T":
            axis_limits["t_min"] = idx["idx_t_min"]
            axis_limits["t_max"] = idx["idx_t_max"]
        if self.parallel_method != "X":
            axis_limits["x_min"] = idx["idx_x_min"]
            axis_limits["x_max"] = idx["idx_x_max"]
        if self.parallel_method != "Y":
            axis_limits["y_min"] = idx["idx_y_min"]
            axis_limits["y_max"] = idx["idx_y_max"]

        # Vertical levels selection:
        axis_limits["z_min"] = self.first_level
        if self.last_level == -1 or self.last_level is None:
            self.last_level = None
        elif self.last_level + 1 == len(self.get_full_levels()["data"]):
            self.last_level = None
        else:
            self.last_level += 1
        axis_limits["z_max"] = self.last_level

        return axis_limits

    def _get_time_id(self, hours, first=True):
        """
        Get the index of the corresponding time value.

        Parameters
        ----------
        hours:int
            Number of hours to avoid.
        first:bool
            Indicates if you want to avoid from the first hours (True) or from the last (False).
            Default: True.

        Returns
        -------
        int
            Index of the time array.
        """
        full_time = self.get_full_times()

        if first:
            idx = full_time.index(full_time[0] + timedelta(hours=hours))
        else:
            idx = full_time.index(full_time[-1] - timedelta(hours=hours)) + 1

        return idx

    @staticmethod
    def _get_coordinate_id(my_array, value, axis=0):
        """
        Get the index of the corresponding coordinate value.

        Parameters
        ----------
        my_array:array
            An Array with the coordinate data
        value:float
            Coordinate value to search.
        axis:int
            Axis where find the value
            Default: 0.

        Returns
        -------
        int
            Index of the coordinate array.
        """
        idx = (abs(my_array - value)).argmin(axis=axis).min()

        return idx

    def _open(self):
        """
        Open the NetCDF.
        """

        self.dataset = self.__open_netcdf4()

        return None

    def __open_netcdf4(self, mode="r"):
        """
        Open the NetCDF with netcdf4-python.

        Parameters
        ----------
        mode:str
            Inheritance from mode parameter from https://unidata.github.io/netcdf4-python/#Dataset.__init__
            Default: "r" (read-only).
        Returns
        -------
        netcdf:Dataset
            Open dataset.
        """

        if self.size == 1:
            netcdf = Dataset(
                self.__ini_path, format="NETCDF4", mode=mode, parallel=False
            )
        else:
            netcdf = Dataset(
                self.__ini_path,
                format="NETCDF4",
                mode=mode,
                parallel=True,
                comm=self.comm,
                info=MPI.Info(),
            )
        self.dataset = netcdf

        return netcdf

    def close(self):
        """
        Close the NetCDF with netcdf4-python.
        """
        if (hasattr(self, "serial_nc")) and (self.serial_nc is not None):
            if self.master:
                self.serial_nc.close()
            self.serial_nc = None
        if (hasattr(self, "dataset")) and (self.dataset is not None):
            self.dataset.close()
            self.dataset = None

        return None

    @staticmethod
    def __get_dates_from_months(time):
        """
        Calculates the number of days since the first date
        in the "time" list and store in new list:
        This is useful when the units are "months since",
        which cannot be transformed to dates using "num2date".

        Parameter
        ---------
        time: List[datetime]
            Original time.

        Returns
        -------
        time: List
            CF compliant time.
        """

        start_date_str = time.units.split("since")[1].lstrip()
        start_date = datetime(
            int(start_date_str[0:4]),
            int(start_date_str[5:7]),
            int(start_date_str[8:10]),
        )

        new_time_deltas = []

        for month_delta in time[:]:
            # Transform current_date into number of days since base date
            current_date = start_date + relativedelta(months=month_delta)

            # Calculate number of days between base date and the other dates
            n_days = int((current_date - start_date).days)

            # Store in list
            new_time_deltas.append(n_days)

        return new_time_deltas

    @staticmethod
    def __get_dates_from_years(time):
        """
        Calculates the number of days since the first date
        in the "time" list and store in new list:
        This is useful when the units are "years since",
        which cannot be transformed to dates using "num2date".

        Parameter
        ---------
        time: List[datetime]
            Original time.

        Returns
        -------
        time: List
            CF compliant time.
        """

        start_date_str = time.units.split("since")[1].lstrip()
        start_date = datetime(
            int(start_date_str[0:4]),
            int(start_date_str[5:7]),
            int(start_date_str[8:10]),
        )

        new_time_deltas = []

        for year_delta in time[:]:
            # Transform current_date into number of days since base date
            current_date = start_date + relativedelta(years=year_delta)

            # Calculate number of days between base date and the other dates
            n_days = int((current_date - start_date).days)

            # Store in list
            new_time_deltas.append(n_days)

        return new_time_deltas

    def __parse_time(self, time):
        """
        Parses the time to be CF compliant.

        Parameters
        ----------
        time: Namespace
            Original time.

        Returns
        -------
        time:str
            CF compliant time.
        """

        units = self.__parse_time_unit(time.units)

        if not hasattr(time, "calendar"):
            calendar = "standard"
        else:
            calendar = time.calendar

        if "months since" in time.units:
            units = "days since " + time.units.split("since")[1].lstrip()
            time = self.__get_dates_from_months(time)
        elif "years since" in time.units:
            units = "days since " + time.units.split("since")[1].lstrip()
            time = self.__get_dates_from_years(time)

        time_data = time[:]

        if len(time_data) == 1 and isnan(time_data[0]):
            time_data[0] = 0

        return time_data, units, calendar

    @staticmethod
    def __parse_time_unit(t_units):
        """
        Parses the time units to be CF compliant.

        Parameters
        ----------
        t_units:str
            Original time units.

        Returns
        -------
        t_units:str
            CF compliant time units.
        """

        if "h @" in t_units:
            t_units = "hours since {0}-{1}-{2} {3}:{4}:{5} UTC".format(
                t_units[4:8],
                t_units[8:10],
                t_units[10:12],
                t_units[13:15],
                t_units[15:17],
                t_units[17:-4],
            )

        return t_units

    @staticmethod
    def __get_time_resolution_from_units(units):
        """
        Parses the time units to get the time resolution

        Parameters
        ----------
        units:str
            Time variable units

        Returns
        -------
        str
            Time variable resolution
        """
        if "day" in units or "days" in units:
            resolution = "days"
        elif "hour" in units or "hours" in units:
            resolution = "hours"
        elif "minute" in units or "minutes" in units:
            resolution = "minutes"
        elif "second" in units or "seconds" in units:
            resolution = "seconds"
        else:
            # Default resolution is "hours"
            resolution = "hours"
        return resolution

    def __get_time(self, time_exception: bool = False):
        """
        Get the NetCDF file time values.

        Returns
        -------
        time:List[datetime]
            List of times (datetime) of the NetCDF data.
        time_exception : bool
            Indicates if you want to avoid the error when no time in dimensions
        """
        if time_exception:
            time = [
                datetime(
                    year=2000,
                    month=1,
                    day=1,
                    hour=0,
                    minute=0,
                )]
            warn("Overwriting time dimension to [2000-01-01 00:00]. "
                 "Use Nes.set_time(...) to set the correct time values.")
            return time

        time_var_name = "time"
        if self.master:
            try:
                nc_var = self.dataset.variables["time"]
            except KeyError:
                nc_variable_names = self.dataset.variables.keys()
                # Accepted name options for the time variable
                accepted_time_names = ["TIME", "valid_time"]
                # Get name of the time variable of the dataset
                time_var_name = list(
                    set(nc_variable_names).intersection(set(accepted_time_names))
                )[0]
                nc_var = self.dataset.variables[time_var_name]

            time_data, units, calendar = self.__parse_time(nc_var)
            # Extracting time resolution depending on the units
            self._time_resolution = self.__get_time_resolution_from_units(units)
            # Checking if it is a climatology dataset
            if hasattr(nc_var, "climatology"):
                self._climatology = True
                self._climatology_var_name = nc_var.climatology
            time = num2date(time_data, units, calendar=calendar)
            new_time = []
            for idx, dt in enumerate(time):  # type: ignore
                if isinstance(dt, (datetime, cftime.datetime)):
                    new_time.append(
                        datetime(
                            year=dt.year,
                            month=dt.month,
                            day=dt.day,
                            hour=dt.hour,
                            minute=dt.minute,
                        )
                    )
                else:
                    warn(
                        f"Skipping time value '{dt}' of type '{type(dt).__name__}' at index '{idx}'; "
                        "expected datetime.datetime or cftime.datetime.",
                        UserWarning,
                    )
            time = new_time
        else:
            time = None
        # Free the time variable
        self.free_vars(time_var_name)

        return time

    def __get_time_bnds(self, create_nes=False):
        """
        Get the NetCDF time bounds values.

        Parameters
        ----------
        create_nes:bool
            Indicates if you want to create the object from scratch (True) or through an existing file.

        Returns
        -------
        time_bnds:List
            A List of time bounds (datetime) of the NetCDF data.
        """

        if not create_nes:
            if self.master:
                if "time_bnds" in self.dataset.variables.keys() or self._climatology:
                    time = self.dataset.variables["time"]
                    if self._climatology:
                        nc_var = self.dataset.variables[self._climatology_var_name]
                    else:
                        nc_var = self.dataset.variables["time_bnds"]
                    time_bnds = num2date(
                        nc_var[:],
                        self.__parse_time_unit(time.units),
                        calendar=time.calendar,
                    ).tolist()
                    # Iterate over each inner list
                    for inner_list in time_bnds:
                        # Create a new list to store datetime objects
                        new_inner_list = []
                        # Iterate over datetime objects within each inner list
                        for dt in inner_list:
                            # Access year, month, day, hour, and minute attributes of datetime objects
                            new_dt = datetime(
                                year=dt.year,
                                month=dt.month,
                                day=dt.day,
                                hour=dt.hour,
                                minute=dt.minute,
                            )
                            # Append the new datetime object to the new inner list
                            new_inner_list.append(new_dt)
                        # Replace the old inner list with the new one
                        time_bnds[time_bnds.index(inner_list)] = new_inner_list
                else:
                    time_bnds = None
            else:
                time_bnds = None
        else:
            time_bnds = None

        self.free_vars("time_bnds")

        return time_bnds

    def __get_coordinates_bnds(self, create_nes=False):
        """
        Get the NetCDF coordinates bounds values.

        Parameters
        ----------
        create_nes:bool
            Indicates if you want to create the object from scratch (True) or through an existing file.

        Returns
        -------
        lat_bnds:dict
            Latitude bounds of the NetCDF data.
        lon_bnds:dict
            Longitude bounds of the NetCDF data.
        """

        if not create_nes:
            if self.master:
                if "lat_bnds" in self.dataset.variables.keys():
                    lat_bnds = {
                        "data": self._unmask_array(
                            self.dataset.variables["lat_bnds"][:]
                        )
                    }
                else:
                    lat_bnds = None

                if "lon_bnds" in self.dataset.variables.keys():
                    lon_bnds = {
                        "data": self._unmask_array(
                            self.dataset.variables["lon_bnds"][:]
                        )
                    }
                else:
                    lon_bnds = None
            else:
                lat_bnds = None
                lon_bnds = None
        else:
            lat_bnds = None
            lon_bnds = None

        self.free_vars(["lat_bnds", "lon_bnds"])

        return lat_bnds, lon_bnds

    def __get_cell_measures(self, create_nes=False):
        """
        Get the NetCDF cell measures values.

        Parameters
        ----------
        create_nes:bool
            Indicates if you want to create the object from scratch (True) or through an existing file.

        Returns
        -------
        dict
            Dictionary of cell measures of the NetCDF data.
        """

        c_measures = {}
        if self.master:
            if not create_nes:
                if "cell_area" in self.dataset.variables.keys():
                    c_measures["cell_area"] = {}
                    c_measures["cell_area"]["data"] = self._unmask_array(
                        self.dataset.variables["cell_area"][:]
                    )
        c_measures = self.comm.bcast(c_measures, root=0)

        self.free_vars(["cell_area"])

        return c_measures

    def _get_coordinate_dimension(self, possible_names):
        """
        Read the coordinate dimension data.

        This will read the complete data of the coordinate.

        Parameters
        ----------
        possible_names: List, str, list
            A List (or single string) of the possible names of the coordinate (e.g. ["lat", "latitude"]).

        Returns
        -------
        nc_var:dict
            Dictionary with the "data" key with the coordinate variable values. and the attributes as other keys.
        """

        if isinstance(possible_names, str):
            possible_names = [possible_names]

        try:
            dimension_name = (
                set(possible_names).intersection(set(self.variables.keys())).pop()
            )
            if self.master:
                nc_var = self.variables[dimension_name].copy()
                nc_var["data"] = self.dataset.variables[dimension_name][:]
                if hasattr(nc_var, "units"):
                    if nc_var["units"] in ["unitless", "-"]:
                        nc_var["units"] = ""
            else:
                nc_var = None
            self.free_vars(dimension_name)
        except KeyError:
            if self.master:
                nc_var = {"data": array([0]), "units": ""}
            else:
                nc_var = None

        return nc_var

    def _get_coordinate_values(self, coordinate_info, coordinate_axis, bounds=False):
        """
        Get the coordinate data of the current portion.

        Parameters
        ----------
        coordinate_info:dict, list
            Dictionary with the "data" key with the coordinate variable values. and the attributes as other keys.
        coordinate_axis:str
            Name of the coordinate to extract. Accepted values: ["Z", "Y", "X"].
        bounds:bool
            Boolean variable to know if there are coordinate bounds.
        Returns
        -------
        values:dict
            Dictionary with the portion of data corresponding to the rank.
        """

        if coordinate_info is None:
            return None

        if not isinstance(coordinate_info, dict):
            values = {"data": deepcopy(coordinate_info)}
        else:
            values = deepcopy(coordinate_info)

        coordinate_len = len(values["data"].shape)
        if bounds:
            coordinate_len -= 1

        if coordinate_axis == "Y":
            if coordinate_len == 1:
                values["data"] = values["data"][
                    self.read_axis_limits["y_min"]: self.read_axis_limits["y_max"]
                ]
            elif coordinate_len == 2:
                values["data"] = values["data"][
                    self.read_axis_limits["y_min"]: self.read_axis_limits["y_max"],
                    self.read_axis_limits["x_min"]: self.read_axis_limits["x_max"],
                ]
            else:
                raise NotImplementedError(
                    "The coordinate has wrong dimensions: {dim}".format(
                        dim=values["data"].shape
                    )
                )
        elif coordinate_axis == "X":
            if coordinate_len == 1:
                if self.read_axis_limits["x_min"] < 0:
                    # Negative longitudes
                    values["data"] = concatenate(
                        (
                            values["data"][self.read_axis_limits["x_min"]:],
                            values["data"][: self.read_axis_limits["x_max"]],
                        )
                    )
                    values["data"][values["data"] > 180] -= 360
                else:
                    values["data"] = values["data"][
                        self.read_axis_limits["x_min"]: self.read_axis_limits["x_max"]
                    ]
            elif coordinate_len == 2:
                if self.read_axis_limits["x_min"] < 0:
                    # Negative longitudes
                    values["data"] = concatenate(
                        (
                            values["data"][
                                self.read_axis_limits["y_min"]: self.read_axis_limits[
                                    "y_max"
                                ],
                                self.read_axis_limits["x_min"]:,
                            ],
                            values["data"][
                                self.read_axis_limits["y_min"]: self.read_axis_limits[
                                    "y_max"
                                ],
                                : self.read_axis_limits["x_max"],
                            ],
                        ),
                        axis=1,
                    )
                    values["data"][values["data"] > 180] -= 360
                else:
                    values["data"] = values["data"][
                        self.read_axis_limits["y_min"]: self.read_axis_limits["y_max"],
                        self.read_axis_limits["x_min"]: self.read_axis_limits["x_max"],
                    ]
            else:
                raise NotImplementedError(
                    "The coordinate has wrong dimensions: {dim}".format(
                        dim=values["data"].shape
                    )
                )
        elif coordinate_axis == "Z":
            if coordinate_len == 1:
                values["data"] = values["data"][
                    self.read_axis_limits["z_min"]: self.read_axis_limits["z_max"]
                ]
            else:
                raise NotImplementedError(
                    "The coordinate has wrong dimensions: {dim}".format(
                        dim=values["data"].shape
                    )
                )

        return values

    def _get_cell_measures_values(self, cell_measures_info):
        """
        Get the cell measures data of the current portion.

        Parameters
        ----------
        cell_measures_info:dict, list
            Dictionary with the "data" key with the cell measures variable values. and the attributes as other keys.

        Returns
        -------
        values:dict
            Dictionary with the portion of data corresponding to the rank.
        """

        if cell_measures_info is None:
            return None

        cell_measures_values = {}

        for cell_measures_var in cell_measures_info.keys():
            values = deepcopy(cell_measures_info[cell_measures_var])
            coordinate_len = len(values["data"].shape)

            if coordinate_len == 1:
                values["data"] = values["data"][
                    self.read_axis_limits["x_min"]: self.read_axis_limits["x_max"]
                ]
            elif coordinate_len == 2:
                values["data"] = values["data"][
                    self.read_axis_limits["y_min"]: self.read_axis_limits["y_max"],
                    self.read_axis_limits["x_min"]: self.read_axis_limits["x_max"],
                ]
            else:
                raise NotImplementedError(
                    "The coordinate has wrong dimensions: {dim}".format(
                        dim=values["data"].shape
                    )
                )

            cell_measures_values[cell_measures_var] = values

        return cell_measures_values

    def _get_lazy_variables(self):
        """
        Get all the variables' information.

        Returns
        -------
        variables:dict
            Dictionary with the variable name as key and another dictionary as value.
            De value dictionary will have the "data" key with None as value and all the variable attributes as the
            other keys.
            e.g.
            {"var_name_1": {"data": None, "attr_1": value_1_1, "attr_2": value_1_2, ...},
             "var_name_2": {"data": None, "attr_1": value_2_1, "attr_2": value_2_2, ...},
             ...}
        """

        if self.master:
            variables = {}
            # Initialise data
            for var_name, var_info in self.dataset.variables.items():
                variables[var_name] = {}
                variables[var_name]["data"] = None
                variables[var_name]["dimensions"] = var_info.dimensions
                variables[var_name]["dtype"] = var_info.dtype
                if variables[var_name]["dtype"] in [str, object]:
                    if self.strlen is None:
                        self.set_strlen()
                    variables[var_name]["dtype"] = str

                # Avoid some attributes
                for attrname in var_info.ncattrs():
                    if attrname not in [
                        "missing_value",
                        "_FillValue",
                        "add_offset",
                        "scale_factor",
                    ]:
                        value = getattr(var_info, attrname)
                        if str(value) in ["unitless", "-"]:
                            value = ""
                        variables[var_name][attrname] = value
        else:
            variables = None
        variables = self.comm.bcast(variables, root=0)

        return variables

    def _read_variable(self, var_name):
        """
        Read the corresponding variable data according to the current rank.

        Parameters
        ----------
        var_name:str
            Name of the variable to read.

        Returns
        -------
        data: array
            Portion of the variable data corresponding to the rank.
        """

        nc_var = self.dataset.variables[var_name]
        var_dims = nc_var.dimensions

        # Read data in 4 dimensions
        if self.read_axis_limits["x_min"] < 0:
            if len(var_dims) < 2:
                data = nc_var[:]
            elif len(var_dims) == 2:
                data = concatenate(
                    (
                        nc_var[
                            self.read_axis_limits["y_min"]: self.read_axis_limits[
                                "y_max"
                            ],
                            self.read_axis_limits["x_min"]:,
                        ],
                        nc_var[
                            self.read_axis_limits["y_min"]: self.read_axis_limits[
                                "y_max"
                            ],
                            : self.read_axis_limits["x_max"],
                        ],
                    ),
                    axis=1,
                )
                data = data.reshape(1, 1, data.shape[-2], data.shape[-1])
            elif len(var_dims) == 3:
                if "strlen" in var_dims:
                    data = concatenate(
                        (
                            nc_var[
                                self.read_axis_limits["y_min"]: self.read_axis_limits[
                                    "y_max"
                                ],
                                self.read_axis_limits["x_min"]:,
                                :,
                            ],
                            nc_var[
                                self.read_axis_limits["y_min"]: self.read_axis_limits[
                                    "y_max"
                                ],
                                : self.read_axis_limits["x_max"],
                                :,
                            ],
                        ),
                        axis=1,
                    )
                    data_aux = empty(shape=(data.shape[0], data.shape[1]), dtype=object)
                    for lat_n in range(data.shape[0]):
                        for lon_n in range(data.shape[1]):
                            data_aux[lat_n, lon_n] = "".join(
                                data[lat_n, lon_n]
                                .tobytes()
                                .decode("ascii")
                                .replace("\x00", "")
                            )
                    data = data_aux.reshape(
                        (1, 1, data_aux.shape[-2], data_aux.shape[-1])
                    )
                else:
                    data = concatenate(
                        (
                            nc_var[
                                self.read_axis_limits["t_min"]: self.read_axis_limits[
                                    "t_max"
                                ],
                                self.read_axis_limits["y_min"]: self.read_axis_limits[
                                    "y_max"
                                ],
                                self.read_axis_limits["x_min"]:,
                            ],
                            nc_var[
                                self.read_axis_limits["t_min"]: self.read_axis_limits[
                                    "t_max"
                                ],
                                self.read_axis_limits["y_min"]: self.read_axis_limits[
                                    "y_max"
                                ],
                                : self.read_axis_limits["x_max"],
                            ],
                        ),
                        axis=2,
                    )
                    data = data.reshape(
                        data.shape[-3], 1, data.shape[-2], data.shape[-1]
                    )
            elif len(var_dims) == 4:
                data = concatenate(
                    (
                        nc_var[
                            self.read_axis_limits["t_min"]: self.read_axis_limits[
                                "t_max"
                            ],
                            self.read_axis_limits["z_min"]: self.read_axis_limits[
                                "z_max"
                            ],
                            self.read_axis_limits["y_min"]: self.read_axis_limits[
                                "y_max"
                            ],
                            self.read_axis_limits["x_min"]:,
                        ],
                        nc_var[
                            self.read_axis_limits["t_min"]: self.read_axis_limits[
                                "t_max"
                            ],
                            self.read_axis_limits["z_min"]: self.read_axis_limits[
                                "z_max"
                            ],
                            self.read_axis_limits["y_min"]: self.read_axis_limits[
                                "y_max"
                            ],
                            : self.read_axis_limits["x_max"],
                        ],
                    ),
                    axis=3,
                )
            elif len(var_dims) == 5:
                if "strlen" in var_dims:
                    data = concatenate(
                        (
                            nc_var[
                                self.read_axis_limits["t_min"]: self.read_axis_limits[
                                    "t_max"
                                ],
                                self.read_axis_limits["z_min"]: self.read_axis_limits[
                                    "z_max"
                                ],
                                self.read_axis_limits["y_min"]: self.read_axis_limits[
                                    "y_max"
                                ],
                                self.read_axis_limits["x_min"]:,
                                :,
                            ],
                            nc_var[
                                self.read_axis_limits["t_min"]: self.read_axis_limits[
                                    "t_max"
                                ],
                                self.read_axis_limits["z_min"]: self.read_axis_limits[
                                    "z_max"
                                ],
                                self.read_axis_limits["y_min"]: self.read_axis_limits[
                                    "y_max"
                                ],
                                : self.read_axis_limits["x_max"],
                                :,
                            ],
                        ),
                        axis=3,
                    )
                    data_aux = empty(
                        shape=(
                            data.shape[0],
                            data.shape[1],
                            data.shape[2],
                            data.shape[3],
                        ),
                        dtype=object,
                    )
                    for time_n in range(data.shape[0]):
                        for lev_n in range(data.shape[1]):
                            for lat_n in range(data.shape[2]):
                                for lon_n in range(data.shape[3]):
                                    data_aux[time_n, lev_n, lat_n, lon_n] = "".join(
                                        data[time_n, lev_n, lat_n, lon_n]
                                        .tobytes()
                                        .decode("ascii")
                                        .replace("\x00", "")
                                    )
                    data = data_aux
                else:
                    raise NotImplementedError(
                        f"Error with {var_name}. Only can be read netCDF with 4 dimensions or less"
                    )
            else:
                raise NotImplementedError(
                    f"Error with {var_name}. Only can be read netCDF with 4 dimensions or less"
                )
        else:
            if len(var_dims) < 2:
                data = nc_var[:]
            elif len(var_dims) == 2:
                data = nc_var[
                    self.read_axis_limits["y_min"]: self.read_axis_limits["y_max"],
                    self.read_axis_limits["x_min"]: self.read_axis_limits["x_max"],
                ]
                data = data.reshape(1, 1, data.shape[-2], data.shape[-1])
            elif len(var_dims) == 3:
                if "strlen" in var_dims:
                    data = nc_var[
                        self.read_axis_limits["y_min"]: self.read_axis_limits["y_max"],
                        self.read_axis_limits["x_min"]: self.read_axis_limits["x_max"],
                        :,
                    ]
                    data_aux = empty(shape=(data.shape[0], data.shape[1]), dtype=object)
                    for lat_n in range(data.shape[0]):
                        for lon_n in range(data.shape[1]):
                            data_aux[lat_n, lon_n] = "".join(
                                data[lat_n, lon_n]
                                .tobytes()
                                .decode("ascii")
                                .replace("\x00", "")
                            )
                    data = data_aux.reshape(
                        (1, 1, data_aux.shape[-2], data_aux.shape[-1])
                    )
                else:
                    data = nc_var[
                        self.read_axis_limits["t_min"]: self.read_axis_limits["t_max"],
                        self.read_axis_limits["y_min"]: self.read_axis_limits["y_max"],
                        self.read_axis_limits["x_min"]: self.read_axis_limits["x_max"],
                    ]
                    data = data.reshape(
                        data.shape[-3], 1, data.shape[-2], data.shape[-1]
                    )
            elif len(var_dims) == 4:
                data = nc_var[
                    self.read_axis_limits["t_min"]: self.read_axis_limits["t_max"],
                    self.read_axis_limits["z_min"]: self.read_axis_limits["z_max"],
                    self.read_axis_limits["y_min"]: self.read_axis_limits["y_max"],
                    self.read_axis_limits["x_min"]: self.read_axis_limits["x_max"],
                ]
            elif len(var_dims) == 5:
                if "strlen" in var_dims:
                    data = nc_var[
                        self.read_axis_limits["t_min"]: self.read_axis_limits["t_max"],
                        self.read_axis_limits["z_min"]: self.read_axis_limits["z_max"],
                        self.read_axis_limits["y_min"]: self.read_axis_limits["y_max"],
                        self.read_axis_limits["x_min"]: self.read_axis_limits["x_max"],
                        :,
                    ]
                    data_aux = empty(
                        shape=(
                            data.shape[0],
                            data.shape[1],
                            data.shape[2],
                            data.shape[3],
                        ),
                        dtype=object,
                    )
                    for time_n in range(data.shape[0]):
                        for lev_n in range(data.shape[1]):
                            for lat_n in range(data.shape[2]):
                                for lon_n in range(data.shape[3]):
                                    data_aux[time_n, lev_n, lat_n, lon_n] = "".join(
                                        data[time_n, lev_n, lat_n, lon_n]
                                        .tobytes()
                                        .decode("ascii")
                                        .replace("\x00", "")
                                    )
                    data = data_aux
                else:
                    raise NotImplementedError(
                        "Error with {0}. Only can be read netCDF with 4 dimensions or less".format(
                            var_name
                        )
                    )
            else:
                raise NotImplementedError(
                    "Error with {0}. Only can be read netCDF with 4 dimensions or less".format(
                        var_name
                    )
                )

        # Unmask array
        data = self._unmask_array(data)

        return data

    def load(self, var_list=None):
        """
        Load of the selected variables.

        That function will fill the variable "data" key with the corresponding values.

        Parameters
        ----------
        var_list:List, str, None
            List (or single string) of the variables to be loaded.
        """

        if (self.__ini_path is None) and (self.dataset is None):
            raise RuntimeError("Only data from existing files can be loaded.")

        if self.dataset is None:
            self.__open_netcdf4()
            close = True
        else:
            close = False

        if isinstance(var_list, str):
            var_list = [var_list]
        elif var_list is None:
            var_list = list(self.variables.keys())

        for i, var_name in enumerate(var_list):
            if self.info:
                print(
                    "Rank {0:03d}: Loading {1} var ({2}/{3})".format(
                        self.rank, var_name, i + 1, len(var_list)
                    )
                )
            if self.variables[var_name]["data"] is None:
                self.variables[var_name]["data"] = self._read_variable(var_name)
                # Data type changes when joining characters in read_variable (S1 to S+strlen)
                if "strlen" in self.variables[var_name]["dimensions"]:
                    if self.strlen is None:
                        self.set_strlen()
                    self.variables[var_name]["dtype"] = str
                    self.variables[var_name]["dimensions"] = tuple(
                        [
                            x
                            for x in self.variables[var_name]["dimensions"]
                            if x != "strlen"
                        ]
                    )
            else:
                if self.master:
                    print(
                        "Data for {0} was previously loaded. Skipping variable.".format(
                            var_name
                        )
                    )
            if self.info:
                print(
                    "Rank {0:03d}: Loaded {1} var ({2})".format(
                        self.rank, var_name, self.variables[var_name]["data"].shape
                    )
                )

        if close:
            self.close()

        self.loaded = True

        return None

    @staticmethod
    def _unmask_array(data):
        """
        Missing to nan. This operation is done because sometimes the missing value is lost during the calculation.

        Parameters
        ----------
        data:array
            Masked array to unmask.

        Returns
        -------
        array
            Unmasked array.
        """

        if isinstance(data, ma.MaskedArray):
            try:
                data = data.filled(nan)
            except TypeError:
                msg = "Data missing values cannot be converted to nan."
                warn(msg)
                sys.stderr.flush()

        return data

    def to_dtype(self, data_type="float32"):
        """Cast variables data into selected data type.

        Parameters
        ----------
        data_type:str or Type
            Data type, by default "float32"
        """

        for var_name, var_info in self.variables.items():
            if isinstance(var_info["data"], ndarray):
                self.variables[var_name]["data"] = self.variables[var_name][
                    "data"
                ].astype(data_type)
            self.variables[var_name]["dtype"] = data_type

        return None

    def concatenate(self, aux_nessy):
        """
        Concatenate different variables into the same NES object.

        Parameters
        ----------
        aux_nessy:Nes, str
            Nes object or str with the path to the NetCDF file that contains the variables to add.

        Returns
        -------
        list
            A List of var names added.
        """

        if isinstance(aux_nessy, str):
            aux_nessy = self.new(
                path=aux_nessy,
                comm=self.comm,
                parallel_method=self.parallel_method,
                avoid_first_hours=self.hours_start,
                avoid_last_hours=self.hours_end,
                first_level=self.first_level,
                last_level=self.last_level,
            )
            new = True
        else:
            new = False
        for var_name, var_info in aux_nessy.variables.items():
            if var_info["data"] is None:
                aux_nessy.read_axis_limits = self.read_axis_limits
                aux_nessy.load(var_name)

        new_vars_added = []
        for new_var_name, new_var_data in aux_nessy.variables.items():
            if new_var_name not in self.variables.keys():
                self.variables[new_var_name] = deepcopy(new_var_data)
            new_vars_added.append(new_var_name)

        if new:
            del aux_nessy

        return new_vars_added

    def __get_global_attributes(self, create_nes=False):
        """
        Read the netcdf global attributes.

        Parameters
        ----------
        create_nes:bool
            Indicates if you want to create the object from scratch (True) or through an existing file.

        Returns
        -------
        gl_attrs:dict
            Dictionary with the netCDF global attributes.
        """

        gl_attrs = {}

        if not create_nes:
            for attrname in self.dataset.ncattrs():
                gl_attrs[attrname] = getattr(self.dataset, attrname)

        return gl_attrs

    # ==================================================================================================================
    #                                                 Writing
    # ==================================================================================================================

    def _get_write_axis_limits(self):
        """
        Calculate the 4D writing axis limits depending on if them have to balanced or not.

        Returns
        -------
        dict
            Dictionary with the 4D limits of the rank data to write.
            t_min, t_max, z_min, z_max, y_min, y_max, x_min and x_max.
        """

        if self.balanced:
            return self._get_write_axis_limits_balanced()
        else:
            return self._get_write_axis_limits_unbalanced()

    def _get_write_axis_limits_unbalanced(self):
        """
        Calculate the 4D writing axis limits.

        Returns
        -------
        dict
            Dictionary with the 4D limits of the rank data to write.
            t_min, t_max, z_min, z_max, y_min, y_max, x_min and x_max.
        """

        axis_limits = {
            "x_min": None,
            "x_max": None,
            "y_min": None,
            "y_max": None,
            "z_min": None,
            "z_max": None,
            "t_min": None,
            "t_max": None,
        }
        my_shape = self.get_full_shape()
        if self.parallel_method == "Y":
            y_len = my_shape[0]
            axis_limits["y_min"] = (y_len // self.size) * self.rank
            if self.rank + 1 < self.size:
                axis_limits["y_max"] = (y_len // self.size) * (self.rank + 1)
        elif self.parallel_method == "X":
            x_len = my_shape[-1]
            axis_limits["x_min"] = (x_len // self.size) * self.rank
            if self.rank + 1 < self.size:
                axis_limits["x_max"] = (x_len // self.size) * (self.rank + 1)
        elif self.parallel_method == "T":
            t_len = len(self.get_full_times())
            axis_limits["t_min"] = (t_len // self.size) * self.rank
            if self.rank + 1 < self.size:
                axis_limits["t_max"] = (t_len // self.size) * (self.rank + 1)
        else:
            raise NotImplementedError(
                "Parallel method '{meth}' is not implemented. Use one of these: {accept}".format(
                    meth=self.parallel_method, accept=["X", "Y", "T"]
                )
            )

        return axis_limits

    def _get_write_axis_limits_balanced(self):
        """
        Calculate the 4D reading balanced axis limits.

        Returns
        -------
        dict
            Dictionary with the 4D limits of the rank data to read.
            t_min, t_max, z_min, z_max, y_min, y_max, x_min and x_max.
        """
        my_shape = self.get_full_shape()
        fid_dist = {}
        if self.parallel_method == "Y":
            len_to_split = my_shape[0]
            min_axis = "y_min"
            max_axis = "y_max"
        elif self.parallel_method == "X":
            len_to_split = my_shape[-1]
            min_axis = "x_min"
            max_axis = "x_max"
        elif self.parallel_method == "T":
            len_to_split = len(self.get_full_times())
            min_axis = "t_min"
            max_axis = "t_max"
        else:
            raise NotImplementedError(
                "Parallel method '{meth}' is not implemented. Use one of these: {accept}".format(
                    meth=self.parallel_method, accept=["X", "Y", "T"]
                )
            )

        procs_len = len_to_split // self.size
        procs_rows_extended = len_to_split - (procs_len * self.size)

        rows_sum = 0
        for proc in range(self.size):
            fid_dist[proc] = {
                "x_min": 0,
                "x_max": None,
                "y_min": 0,
                "y_max": None,
                "z_min": 0,
                "z_max": None,
                "t_min": 0,
                "t_max": None,
            }
            if proc < procs_rows_extended:
                aux_rows = procs_len + 1
            else:
                aux_rows = procs_len

            len_to_split -= aux_rows
            if len_to_split < 0:
                rows = len_to_split + aux_rows
            else:
                rows = aux_rows

            fid_dist[proc][min_axis] = rows_sum
            fid_dist[proc][max_axis] = rows_sum + rows

            # Last element
            if len_to_split == 0:
                fid_dist[proc][max_axis] = None

            rows_sum += rows

        axis_limits = fid_dist[self.rank]

        return axis_limits

    def _create_dimensions(self, netcdf):
        """
        Create "time", "time_bnds", "lev", "lon" and "lat" dimensions.

        Parameters
        ----------
        netcdf:Dataset
            netcdf4-python open dataset.
        """

        # Create time dimension
        netcdf.createDimension("time", None)

        # Create time_nv (number of vertices) dimension
        full_time_bnds = self.get_full_time_bnds()
        if full_time_bnds is not None:
            netcdf.createDimension("time_nv", 2)

        # Create lev, lon and lat dimensions
        netcdf.createDimension("lev", len(self.lev["data"]))

        # Create string length dimension
        if self.strlen is not None:
            netcdf.createDimension("strlen", self.strlen)

        return None

    def _create_dimension_variables(self, netcdf):
        """
        Create the "time", "time_bnds", "lev", "lat", "lat_bnds", "lon" and "lon_bnds" variables.

        Parameters
        ----------
        netcdf:Dataset
            netcdf4-python open dataset.
        """

        self._create_dimension_variables_64(netcdf)

        return None

    def _create_dimension_variables_32(self, netcdf):
        """
        Create the "time", "time_bnds", "lev", "lat", "lat_bnds", "lon" and "lon_bnds" variables.

        Parameters
        ----------
        netcdf:Dataset
            netcdf4-python open dataset.
        """

        # TIMES
        full_time = self.get_full_times()
        full_time_bnds = self.get_full_time_bnds()
        time_var = netcdf.createVariable(
            "time", float32, ("time",), zlib=self.zip_lvl > 0, complevel=self.zip_lvl
        )
        time_var.units = "{0} since {1}".format(
            self._time_resolution, full_time[0].strftime("%Y-%m-%d %H:%M:%S")
        )
        time_var.standard_name = "time"
        time_var.calendar = "standard"
        time_var.long_name = "time"
        if full_time_bnds is not None:
            if self._climatology:
                time_var.climatology = self._climatology_var_name
            else:
                time_var.bounds = "time_bnds"
        if self.size > 1:
            time_var.set_collective(True)
        time_var[:] = date2num(full_time[:], time_var.units, time_var.calendar)

        # TIME BOUNDS
        if full_time_bnds is not None:
            if self._climatology:
                time_bnds_var = netcdf.createVariable(
                    self._climatology_var_name,
                    float64,
                    (
                        "time",
                        "time_nv",
                    ),
                    zlib=self.zip_lvl,
                    complevel=self.zip_lvl,
                )
            else:
                time_bnds_var = netcdf.createVariable(
                    "time_bnds",
                    float64,
                    (
                        "time",
                        "time_nv",
                    ),
                    zlib=self.zip_lvl,
                    complevel=self.zip_lvl,
                )
            if self.size > 1:
                time_bnds_var.set_collective(True)
            time_bnds_var[:] = date2num(
                full_time_bnds, time_var.units, calendar="standard"
            )

        # LEVELS
        full_lev = self.get_full_levels()
        lev = netcdf.createVariable(
            "lev", float32, ("lev",), zlib=self.zip_lvl > 0, complevel=self.zip_lvl
        )
        if "units" in full_lev.keys():
            lev.units = full_lev["units"]
        else:
            lev.units = ""
        if "positive" in full_lev.keys():
            lev.positive = full_lev["positive"]

        if self.size > 1:
            lev.set_collective(True)
        lev[:] = array(full_lev["data"], dtype=float32)

        # LATITUDES
        full_lat = self.get_full_latitudes()
        full_lat_bnds = self.get_full_latitudes_boundaries()
        lat = netcdf.createVariable(
            "lat", float32, self._lat_dim, zlib=self.zip_lvl > 0, complevel=self.zip_lvl
        )
        lat.units = "degrees_north"
        lat.axis = "Y"
        lat.long_name = "latitude coordinate"
        lat.standard_name = "latitude"
        if full_lat_bnds is not None:
            lat.bounds = "lat_bnds"
        if self.size > 1:
            lat.set_collective(True)
        lat[:] = array(full_lat["data"], dtype=float32)

        # LATITUDES BOUNDS
        if full_lat_bnds is not None:
            lat_bnds_var = netcdf.createVariable(
                "lat_bnds",
                float32,
                self._lat_dim + ("spatial_nv",),
                zlib=self.zip_lvl > 0,
                complevel=self.zip_lvl,
            )
            if self.size > 1:
                lat_bnds_var.set_collective(True)
            lat_bnds_var[:] = array(full_lat_bnds["data"], dtype=float32)

        # LONGITUDES
        full_lon = self.get_full_longitudes()
        full_lon_bnds = self.get_full_longitudes_boundaries()
        lon = netcdf.createVariable(
            "lon", float32, self._lon_dim, zlib=self.zip_lvl > 0, complevel=self.zip_lvl
        )
        lon.units = "degrees_east"
        lon.axis = "X"
        lon.long_name = "longitude coordinate"
        lon.standard_name = "longitude"
        if full_lon_bnds is not None:
            lon.bounds = "lon_bnds"
        if self.size > 1:
            lon.set_collective(True)
        lon[:] = array(full_lon["data"], dtype=float32)

        # LONGITUDES BOUNDS
        if full_lon_bnds is not None:
            lon_bnds_var = netcdf.createVariable(
                "lon_bnds",
                float32,
                self._lon_dim + ("spatial_nv",),
                zlib=self.zip_lvl > 0,
                complevel=self.zip_lvl,
            )
            if self.size > 1:
                lon_bnds_var.set_collective(True)
            lon_bnds_var[:] = array(full_lon_bnds["data"], dtype=float32)

        return None

    def _create_dimension_variables_64(self, netcdf):
        """
        Create the "time", "time_bnds", "lev", "lat", "lat_bnds", "lon" and "lon_bnds" variables.

        Parameters
        ----------
        netcdf:Dataset
            netcdf4-python open dataset.
        """

        # TIMES
        full_time = self.get_full_times()
        full_time_bnds = self.get_full_time_bnds()
        time_var = netcdf.createVariable(
            "time", float64, ("time",), zlib=self.zip_lvl > 0, complevel=self.zip_lvl
        )
        time_var.units = "{0} since {1}".format(
            self._time_resolution, full_time[0].strftime("%Y-%m-%d %H:%M:%S")
        )
        time_var.standard_name = "time"
        time_var.calendar = "standard"
        time_var.long_name = "time"
        if full_time_bnds is not None:
            if self._climatology:
                time_var.climatology = self._climatology_var_name
            else:
                time_var.bounds = "time_bnds"
        if self.size > 1:
            time_var.set_collective(True)
        time_var[:] = date2num(full_time[:], time_var.units, time_var.calendar)

        # TIME BOUNDS
        if full_time_bnds is not None:
            if self._climatology:
                time_bnds_var = netcdf.createVariable(
                    self._climatology_var_name,
                    float64,
                    (
                        "time",
                        "time_nv",
                    ),
                    zlib=self.zip_lvl,
                    complevel=self.zip_lvl,
                )
            else:
                time_bnds_var = netcdf.createVariable(
                    "time_bnds",
                    float64,
                    (
                        "time",
                        "time_nv",
                    ),
                    zlib=self.zip_lvl,
                    complevel=self.zip_lvl,
                )
            if self.size > 1:
                time_bnds_var.set_collective(True)
            time_bnds_var[:] = date2num(
                full_time_bnds, time_var.units, calendar="standard"
            )

        # LEVELS
        full_lev = self.get_full_levels()
        lev = netcdf.createVariable(
            "lev",
            full_lev["data"].dtype,
            ("lev",),
            zlib=self.zip_lvl > 0,
            complevel=self.zip_lvl,
        )
        if "units" in full_lev.keys():
            lev.units = full_lev["units"]
        else:
            lev.units = ""
        if "positive" in full_lev.keys():
            lev.positive = full_lev["positive"]

        if self.size > 1:
            lev.set_collective(True)
        lev[:] = full_lev["data"]

        # LATITUDES
        full_lat = self.get_full_latitudes()
        full_lat_bnds = self.get_full_latitudes_boundaries()
        lat = netcdf.createVariable(
            "lat",
            full_lat["data"].dtype,
            self._lat_dim,
            zlib=self.zip_lvl > 0,
            complevel=self.zip_lvl,
        )
        lat.units = "degrees_north"
        lat.axis = "Y"
        lat.long_name = "latitude coordinate"
        lat.standard_name = "latitude"
        if full_lat_bnds is not None:
            lat.bounds = "lat_bnds"
        if self.size > 1:
            lat.set_collective(True)
        lat[:] = full_lat["data"]

        # LATITUDES BOUNDS
        if full_lat_bnds is not None:
            lat_bnds_var = netcdf.createVariable(
                "lat_bnds",
                full_lat_bnds["data"].dtype,
                self._lat_dim + ("spatial_nv",),
                zlib=self.zip_lvl > 0,
                complevel=self.zip_lvl,
            )
            if self.size > 1:
                lat_bnds_var.set_collective(True)
            lat_bnds_var[:] = full_lat_bnds["data"]

        # LONGITUDES
        full_lon = self.get_full_longitudes()
        full_lon_bnds = self.get_full_longitudes_boundaries()
        lon = netcdf.createVariable(
            "lon",
            full_lon["data"].dtype,
            self._lon_dim,
            zlib=self.zip_lvl > 0,
            complevel=self.zip_lvl,
        )
        lon.units = "degrees_east"
        lon.axis = "X"
        lon.long_name = "longitude coordinate"
        lon.standard_name = "longitude"
        if full_lon_bnds is not None:
            lon.bounds = "lon_bnds"
        if self.size > 1:
            lon.set_collective(True)
        lon[:] = full_lon["data"]

        # LONGITUDES BOUNDS
        if full_lon_bnds is not None:
            lon_bnds_var = netcdf.createVariable(
                "lon_bnds",
                full_lon_bnds["data"].dtype,
                self._lon_dim + ("spatial_nv",),
                zlib=self.zip_lvl > 0,
                complevel=self.zip_lvl,
            )
            if self.size > 1:
                lon_bnds_var.set_collective(True)
            lon_bnds_var[:] = full_lon_bnds["data"]

        return None

    def _create_cell_measures(self, netcdf):
        # CELL AREA
        if "cell_area" in self.cell_measures.keys():
            cell_area = netcdf.createVariable(
                "cell_area",
                self.cell_measures["cell_area"]["data"].dtype,
                self._var_dim,
                zlib=self.zip_lvl > 0,
                complevel=self.zip_lvl,
            )
            if self.size > 1:
                cell_area.set_collective(True)
            cell_area[
                self.write_axis_limits["y_min"]: self.write_axis_limits["y_max"],
                self.write_axis_limits["x_min"]: self.write_axis_limits["x_max"],
            ] = self.cell_measures["cell_area"]["data"]

            cell_area.long_name = "area of grid cell"
            cell_area.standard_name = "cell_area"
            cell_area.units = "m2"

            for var_name in self.variables.keys():
                self.variables[var_name]["cell_measures"] = "area: cell_area"

            if self.info:
                print("Rank {0:03d}: Cell measures done".format(self.rank))
        return None

    def _str2char(self, data):
        if self.strlen is None:
            msg = "String data could not be converted into chars while writing."
            msg += " Please, set the maximum string length (set_strlen) before writing."
            raise RuntimeError(msg)

        # Get final shape by adding strlen at the end
        data_new_shape = data.shape + (self.strlen,)

        # nD (2D, 3D, 4D) data as 1D string array
        data = data.flatten()

        # Normalize all elements to Python str, then create a fixed-width bytes array
        str_data = array([
            v.decode("ascii", "ignore") if isinstance(v, bytes) else str(v)
            for v in data
        ], dtype="S" + str(self.strlen))

        # View as single-byte characters and reshape to (original_shape..., strlen)
        data_aux = str_data.view("S1").reshape(data_new_shape)

        return data_aux

    def _create_variables(self, netcdf, chunking=False):
        """
        Create the netCDF file variables.

        Parameters
        ----------
        netcdf:Dataset
            netcdf4-python open dataset.
        chunking:bool
            Indicates if you want to chunk the output netCDF.
        """

        for i, (var_name, var_dict) in enumerate(self.variables.items()):
            if isinstance(var_dict["data"], int) and var_dict["data"] == 0:
                var_dims = (
                    "time",
                    "lev",
                ) + self._var_dim
                var_dtype = float32
            else:
                # Get dimensions
                if (var_dict["data"] is None) or (len(var_dict["data"].shape) == 4):
                    var_dims = (
                        "time",
                        "lev",
                    ) + self._var_dim
                else:
                    var_dims = self._var_dim

                # Get data type
                if "dtype" in var_dict.keys():
                    var_dtype = var_dict["dtype"]
                    if (var_dict["data"] is not None) and (
                        var_dtype != var_dict["data"].dtype
                    ):
                        msg = "WARNING!!! "
                        msg += "Different data types for variable {0}. ".format(
                            var_name
                        )
                        msg += "Input dtype={0}. Data dtype={1}.".format(
                            var_dtype, var_dict["data"].dtype
                        )
                        warn(msg)
                        sys.stderr.flush()
                        try:
                            var_dict["data"] = var_dict["data"].astype(var_dtype)
                        except Exception as e:  # TODO: Detect exception
                            print(e)
                            raise TypeError(
                                "It was not possible to cast the data to the input dtype."
                            )
                else:
                    var_dtype = var_dict["data"].dtype
                    if var_dtype is object:
                        raise TypeError(
                            "Data dtype is object. Define dtype explicitly as dictionary key 'dtype'"
                        )

                if var_dict["data"] is not None:
                    # Ensure data is of type numpy array (to create NES)
                    if not isinstance(var_dict["data"], (ndarray, generic)):
                        try:
                            var_dict["data"] = array(var_dict["data"])
                        except AttributeError:
                            raise AttributeError(
                                "Data for variable {0} must be a numpy array.".format(
                                    var_name
                                )
                            )

                    # Convert list of strings to chars for parallelization
                    if issubdtype(var_dtype, character):
                        var_dict["data_aux"] = self._str2char(var_dict["data"])
                        var_dims += ("strlen",)
                        var_dtype = "S1"

            if self.info:
                print(
                    "Rank {0:03d}: Writing {1} var ({2}/{3})".format(
                        self.rank, var_name, i + 1, len(self.variables)
                    )
                )

            if not chunking:
                var = netcdf.createVariable(
                    var_name,
                    var_dtype,
                    var_dims,
                    zlib=self.zip_lvl > 0,
                    complevel=self.zip_lvl,
                )
            else:
                if self.balanced:
                    raise NotImplementedError("A balanced data cannot be chunked.")
                if self.master:
                    chunk_size = var_dict["data"].shape
                else:
                    chunk_size = None
                chunk_size = self.comm.bcast(chunk_size, root=0)
                var = netcdf.createVariable(
                    var_name,
                    var_dtype,
                    var_dims,
                    zlib=self.zip_lvl > 0,
                    complevel=self.zip_lvl,
                    chunksizes=chunk_size,
                )
            if self.info:
                print(
                    "Rank {0:03d}: Var {1} created ({2}/{3})".format(
                        self.rank, var_name, i + 1, len(self.variables)
                    )
                )
            if self.size > 1:
                var.set_collective(True)
                if self.info:
                    print(
                        "Rank {0:03d}: Var {1} collective ({2}/{3})".format(
                            self.rank, var_name, i + 1, len(self.variables)
                        )
                    )

            for att_name, att_value in var_dict.items():
                if att_name == "data":
                    if att_value is not None:
                        if self.info:
                            print(
                                "Rank {0:03d}: Filling {1}".format(self.rank, var_name)
                            )
                        if "data_aux" in var_dict.keys():
                            att_value = var_dict["data_aux"]
                        if isinstance(att_value, int) and att_value == 0:
                            var[
                                self.write_axis_limits[
                                    "t_min"
                                ]: self.write_axis_limits["t_max"],
                                self.write_axis_limits[
                                    "z_min"
                                ]: self.write_axis_limits["z_max"],
                                self.write_axis_limits[
                                    "y_min"
                                ]: self.write_axis_limits["y_max"],
                                self.write_axis_limits[
                                    "x_min"
                                ]: self.write_axis_limits["x_max"],
                            ] = 0

                        elif len(att_value.shape) == 5:
                            if "strlen" in var_dims:
                                var[
                                    self.write_axis_limits[
                                        "t_min"
                                    ]: self.write_axis_limits["t_max"],
                                    self.write_axis_limits[
                                        "z_min"
                                    ]: self.write_axis_limits["z_max"],
                                    self.write_axis_limits[
                                        "y_min"
                                    ]: self.write_axis_limits["y_max"],
                                    self.write_axis_limits[
                                        "x_min"
                                    ]: self.write_axis_limits["x_max"],
                                    :,
                                ] = att_value
                            else:
                                raise NotImplementedError(
                                    "It is not possible to write 5D variables."
                                )

                        elif len(att_value.shape) == 4:
                            var[
                                self.write_axis_limits[
                                    "t_min"
                                ]: self.write_axis_limits["t_max"],
                                self.write_axis_limits[
                                    "z_min"
                                ]: self.write_axis_limits["z_max"],
                                self.write_axis_limits[
                                    "y_min"
                                ]: self.write_axis_limits["y_max"],
                                self.write_axis_limits[
                                    "x_min"
                                ]: self.write_axis_limits["x_max"],
                            ] = att_value

                        elif len(att_value.shape) == 3:
                            if "strlen" in var_dims:
                                var[
                                    self.write_axis_limits[
                                        "y_min"
                                    ]: self.write_axis_limits["y_max"],
                                    self.write_axis_limits[
                                        "x_min"
                                    ]: self.write_axis_limits["x_max"],
                                    :,
                                ] = att_value
                            else:
                                raise NotImplementedError(
                                    "It is not possible to write 3D variables."
                                )

                        if self.info:
                            print(
                                "Rank {0:03d}: Var {1} data ({2}/{3})".format(
                                    self.rank, var_name, i + 1, len(self.variables)
                                )
                            )

                elif att_name not in [
                    "chunk_size",
                    "var_dims",
                    "dimensions",
                    "dtype",
                    "data_aux",
                ]:
                    var.setncattr(att_name, att_value)

            if "data_aux" in var_dict.keys():
                del var_dict["data_aux"]

            self._set_var_crs(var)
            if self.info:
                print(
                    "Rank {0:03d}: Var {1} completed ({2}/{3})".format(
                        self.rank, var_name, i + 1, len(self.variables)
                    )
                )

        return None

    def append_time_step_data(self, i_time, out_format="DEFAULT"):
        """
        Fill the netCDF data for the indicated index time.

        Parameters
        ----------
        i_time:int
            index of the time step to write
        out_format:str
            Indicates the output format type to change the units (if needed)
        """
        if self.serial_nc is not None:
            try:
                data = self._gather_data(self.variables)
            except KeyError:
                # Key Error means string data
                data = self.__gather_data_py_object(self.variables)
            if self.master:
                self.serial_nc.variables = data
                self.serial_nc.append_time_step_data(i_time, out_format=out_format)
            self.comm.Barrier()
        else:
            if out_format == "MONARCH":
                self.variables = to_monarch_units(self)
            elif out_format == "CMAQ":
                self.variables = to_cmaq_units(self)
            elif out_format == "WRF_CHEM":
                self.variables = to_wrf_chem_units(self)
            elif out_format == "MOCAGE":
                self.variables = to_mocage_units(self)
            elif out_format == "CHIMERE":
                self.variables = to_chimere_units(self)
            for i, (var_name, var_dict) in enumerate(self.variables.items()):
                for att_name, att_value in var_dict.items():
                    if att_name == "data":
                        if att_value is not None:
                            if self.info:
                                print(
                                    "Rank {0:03d}: Filling {1}".format(
                                        self.rank, var_name
                                    )
                                )
                            var = self.dataset.variables[var_name]
                            if isinstance(att_value, int) and att_value == 0:
                                var[
                                    i_time,
                                    self.write_axis_limits[
                                        "z_min"
                                    ]: self.write_axis_limits["z_max"],
                                    self.write_axis_limits[
                                        "y_min"
                                    ]: self.write_axis_limits["y_max"],
                                    self.write_axis_limits[
                                        "x_min"
                                    ]: self.write_axis_limits["x_max"],
                                ] = 0
                            elif len(att_value.shape) == 4:
                                if len(var.shape) == 3:
                                    # No level info
                                    var[
                                        i_time,
                                        self.write_axis_limits[
                                            "y_min"
                                        ]: self.write_axis_limits["y_max"],
                                        self.write_axis_limits[
                                            "x_min"
                                        ]: self.write_axis_limits["x_max"],
                                    ] = att_value
                                else:
                                    var[
                                        i_time,
                                        self.write_axis_limits[
                                            "z_min"
                                        ]: self.write_axis_limits["z_max"],
                                        self.write_axis_limits[
                                            "y_min"
                                        ]: self.write_axis_limits["y_max"],
                                        self.write_axis_limits[
                                            "x_min"
                                        ]: self.write_axis_limits["x_max"],
                                    ] = att_value

                            elif len(att_value.shape) == 3:
                                raise NotImplementedError(
                                    "It is not possible to write 3D variables."
                                )
                            else:
                                raise NotImplementedError(
                                    "SHAPE APPEND ERROR: {0}".format(att_value.shape)
                                )
                            if self.info:
                                print(
                                    "Rank {0:03d}: Var {1} data ({2}/{3})".format(
                                        self.rank, var_name, i + 1, len(self.variables)
                                    )
                                )
                        else:
                            raise ValueError(
                                "Cannot append None Data for {0}".format(var_name)
                            )
                else:
                    # Metadata already writen
                    pass

        return None

    def _create_centre_coordinates(self, **kwargs):
        """
        Calculate centre latitudes and longitudes from grid details.

        Must be implemented on inner classes

        Returns
        ----------
        centre_lat:dict
            Dictionary with data of centre latitudes in 1D
        centre_lon:dict
            Dictionary with data of centre longitudes in 1D
        """

        return None

    def _create_metadata(self, netcdf):
        """
        Must be implemented on inner class.
        """

        return None

    @staticmethod
    def _set_var_crs(var):
        """
        Must be implemented on inner class.

        Parameters
        ----------
        var:Variable
            netCDF4-python variable object.
        """

        return None

    def __to_netcdf_py(self, path, chunking=False, keep_open=False):
        """
        Create the NetCDF using netcdf4-python methods.

        Parameters
        ----------
        path:str
            Path to the output netCDF file.
        chunking: bool
            Indicates if you want to chunk the output netCDF.
        keep_open:bool
            Indicates if you want to keep open the NetCDH to fill the data by time-step
        """
        # Open NetCDF
        if self.info:
            print("Rank {0:03d}: Creating {1}".format(self.rank, path))
        if self.size > 1:
            netcdf = Dataset(
                path,
                format="NETCDF4",
                mode="w",
                parallel=True,
                comm=self.comm,
                info=MPI.Info(),
            )
        else:
            netcdf = Dataset(path, format="NETCDF4", mode="w", parallel=False)
        if self.info:
            print("Rank {0:03d}: NetCDF ready to write".format(self.rank))

        # Create dimensions
        self._create_dimensions(netcdf)

        # Create dimension variables
        self._create_dimension_variables(netcdf)
        if self.info:
            print("Rank {0:03d}: Dimensions done".format(self.rank))

        # Create cell measures
        self._create_cell_measures(netcdf)

        # Create variables
        self._create_variables(netcdf, chunking=chunking)

        # Create metadata
        self._create_metadata(netcdf)

        # Close NetCDF
        if self.global_attrs is not None:
            for att_name, att_value in self.global_attrs.items():
                netcdf.setncattr(att_name, att_value)
        netcdf.setncattr("Conventions", "CF-1.7")

        if keep_open:
            self.dataset = netcdf
        else:
            netcdf.close()

        return None

    def __to_netcdf_cams_ra(self, path):
        return to_netcdf_cams_ra(self, path)

    def to_netcdf(
        self,
        path,
        compression_level=0,
        serial=False,
        info=False,
        chunking=False,
        nc_type="NES",
        keep_open=False,
    ):
        """
        Write the netCDF output file.

        Parameters
        ----------
        path:str
            Path to the output netCDF file.
        compression_level:int
            Level of compression (0 to 9) Default: 0 (no compression).
        serial:bool
            Indicates if you want to write in serial or not. Default: False.
        info:bool
            Indicates if you want to print the information of each writing step by stdout Default: False.
        chunking:bool
            Indicates if you want a chunked netCDF output. Only available with non-serial writes. Default: False.
        nc_type:str
            Type to NetCDf to write. "CAMS_RA" or "NES", MONARCH, MOCAGE, WRF_CHEM, CMAQ.
        keep_open:bool
            Indicates if you want to keep open the NetCDH to fill the data by time-step
        """
        nc_type = nc_type
        old_info = self.info
        self.info = info
        self.serial_nc = None
        self.zip_lvl = compression_level

        # if serial:
        if serial and self.size > 1:
            try:
                data = self._gather_data(self.variables)
            except KeyError:
                data = self.__gather_data_py_object(self.variables)
            try:
                c_measures = self._gather_data(self.cell_measures)
            except KeyError:
                c_measures = self.__gather_data_py_object(self.cell_measures)
            if self.master:
                new_nc = self.copy(copy_vars=False, new_comm=MPI.COMM_SELF)
                new_nc.variables = data
                new_nc.cell_measures = c_measures
                if nc_type in ["NES", "DEFAULT"]:
                    new_nc.__to_netcdf_py(path, keep_open=keep_open)
                elif nc_type == "CAMS_RA":
                    new_nc.__to_netcdf_cams_ra(path)
                elif nc_type == "MONARCH":
                    to_netcdf_monarch(
                        new_nc, path, chunking=chunking, keep_open=keep_open
                    )
                elif nc_type == "CMAQ":
                    to_netcdf_cmaq(new_nc, path, keep_open=keep_open)
                elif nc_type == "WRF_CHEM":
                    to_netcdf_wrf_chem(new_nc, path, keep_open=keep_open)
                elif nc_type == "MOCAGE":
                    to_netcdf_mocage(new_nc, path, keep_open=keep_open)
                elif nc_type == "CHIMERE":
                    to_netcdf_chimere(new_nc, path, keep_open=keep_open)
                else:
                    msg = f"Unknown NetCDF type '{nc_type}'. "
                    msg += "Use CAMS_RA, MONARCH, CMAQ, WRF_CHEM, MOCAGE or NES (or DEFAULT)"
                    raise ValueError(msg)
                self.serial_nc = new_nc
            else:
                self.serial_nc = True
        else:
            if nc_type in ["NES", "DEFAULT"]:
                self.__to_netcdf_py(path, chunking=chunking, keep_open=keep_open)
            elif nc_type == "CAMS_RA":
                self.__to_netcdf_cams_ra(path)
            elif nc_type == "MONARCH":
                to_netcdf_monarch(self, path, chunking=chunking, keep_open=keep_open)
            elif nc_type == "CMAQ":
                to_netcdf_cmaq(self, path, keep_open=keep_open)
            elif nc_type == "WRF_CHEM":
                to_netcdf_wrf_chem(self, path, keep_open=keep_open)
            elif nc_type == "MOCAGE":
                to_netcdf_mocage(self, path, keep_open=keep_open)
            elif nc_type == "CHIMERE":
                to_netcdf_chimere(self, path, keep_open=keep_open)
            else:
                msg = f"Unknown NetCDF type '{nc_type}''. "
                msg += "Use CAMS_RA, MONARCH, CMAQ, WRF_CHEM, MOCAGE, CHIMERE or NES (or DEFAULT)"
                raise ValueError(msg)

        self.info = old_info

        return None

    def __to_grib2(
        self, path, grib_keys, grib_template_path, lat_flip=True, info=False
    ):
        """
        Private method to write output file with grib2 format.

        Parameters
        ----------
        path:str
            Path to the output file.
        grib_keys:dict
            Dictionary with the grib2 keys.
        grib_template_path:str
            Path to the grib2 file to use as template.
        info:bool
            Indicates if you want to print extra information during the process.
        """

        from eccodes import (
            codes_clone,
            codes_get_string,
            codes_grib_new_from_file,
            codes_keys_iterator_delete,
            codes_keys_iterator_get_name,
            codes_keys_iterator_new,
            codes_keys_iterator_next,
            codes_release,
            codes_set,
            codes_set_values,
            codes_write,
        )

        fout = open(path, "wb")

        # read template
        fin = open(grib_template_path, "rb")

        gid = codes_grib_new_from_file(fin)
        if gid is None:
            sys.exit(1)

        iterid = codes_keys_iterator_new(gid, "ls")
        while codes_keys_iterator_next(iterid):
            keyname = codes_keys_iterator_get_name(iterid)
            keyval = codes_get_string(gid, keyname)
            if info:
                print("%s = %s" % (keyname, keyval))

        codes_keys_iterator_delete(iterid)
        for var_name, var_info in self.variables.items():
            for i_time, time in enumerate(self.time):
                for i_lev, lev in enumerate(self.lev["data"]):
                    clone_id = codes_clone(gid)

                    # Adding grib2 keys to file
                    for key, value in grib_keys.items():
                        if value not in ["", "None", None, nan]:
                            try:
                                codes_set(clone_id, key, value)
                            except Exception as e:
                                print(
                                    f"Something went wrong while writing the Grib key '{key}': {value}"
                                )
                                raise e

                    # Time dependent keys
                    if "dataTime" in grib_keys.keys() and grib_keys["dataTime"] in [
                        "",
                        "None",
                        None,
                        nan,
                    ]:
                        codes_set(clone_id, "dataTime", int(i_time * 100))
                    if "stepRange" in grib_keys.keys() and grib_keys["stepRange"] in [
                        "",
                        "None",
                        None,
                        nan,
                    ]:
                        n_secs = (time - self.get_full_times()[0]).total_seconds()
                        codes_set(clone_id, "stepRange", int(n_secs // 3600))
                    if "forecastTime" in grib_keys.keys() and grib_keys[
                        "forecastTime"
                    ] in ["", "None", None, nan]:
                        n_secs = (time - self.get_full_times()[0]).total_seconds()
                        codes_set(clone_id, "forecastTime", int(n_secs))

                    # Level dependent keys
                    if "typeOfFirstFixedSurface" in grib_keys.keys() and grib_keys[
                        "typeOfFirstFixedSurface"
                    ] in ["", "None", None, nan]:
                        if float(lev) == 0:
                            codes_set(clone_id, "typeOfFirstFixedSurface", 1)
                            # grib_keys["typeOfFirstFixedSurface"] = 1
                        else:
                            codes_set(clone_id, "typeOfFirstFixedSurface", 103)
                            # grib_keys["typeOfFirstFixedSurface"] = 103
                    if "level" in grib_keys.keys() and grib_keys["level"] in [
                        "",
                        "None",
                        None,
                        nan,
                    ]:
                        codes_set(clone_id, "level", float(lev))

                    newval = var_info["data"][i_time, i_lev, :, :]
                    if lat_flip:
                        newval = flipud(newval)

                    # TODO Check default NaN Value
                    newval[isnan(newval)] = 0.0

                    codes_set_values(clone_id, array(newval.ravel(), dtype="float64"))
                    codes_write(clone_id, fout)
                    del newval
        codes_release(gid)
        fout.close()
        fin.close()

        return None

    def to_grib2(self, path, grib_keys, grib_template_path, lat_flip=True, info=False):
        """
        Write output file with grib2 format.

        Parameters
        ----------
        path:str
            Path to the output file.
        grib_keys:dict
            Dictionary with the grib2 keys.
        grib_template_path:str
            Path to the grib2 file to use as template.
        lat_flip:bool
            Indicates if the latitude values (and data) has to be flipped
        info:bool
            Indicates if you want to print extra information during the process.
        """

        # if serial:
        if self.parallel_method in ["X", "Y"] and self.size > 1:
            try:
                data = self._gather_data(self.variables)
            except KeyError:
                data = self.__gather_data_py_object(self.variables)
            try:
                c_measures = self._gather_data(self.cell_measures)
            except KeyError:
                c_measures = self.__gather_data_py_object(self.cell_measures)
            if self.master:
                new_nc = self.copy(copy_vars=False, new_comm=MPI.COMM_SELF)
                new_nc.variables = data
                new_nc.cell_measures = c_measures
                new_nc.__to_grib2(
                    path, grib_keys, grib_template_path, lat_flip=lat_flip, info=info
                )
        else:
            self.__to_grib2(
                path, grib_keys, grib_template_path, lat_flip=lat_flip, info=info
            )

        return None

    def create_shapefile(self, overwrite=False, points=False):
        """
        Create spatial GeoDataFrame (shapefile).

        Parameters
        ----------
        overwrite:bool
            Indicates if you want to overwrite a pre-existent geostructure
        points:bool
            Indicates if you want a Point Geostructure. Default: False -> Polygons

        Returns
        -------
        shapefile:GeoPandasDataFrame
            Shapefile dataframe.
        """

        if self.shapefile is None or overwrite:
            if not points:
                # Create Polygons
                if self.lat_bnds is None or self.lon_bnds is None:
                    self.create_spatial_bounds()

                # Reshape arrays to create geometry
                aux_shape = (
                    self.lat_bnds["data"].shape[0],
                    self.lon_bnds["data"].shape[0],
                    4,
                )
                lon_bnds_aux = empty(aux_shape)
                lon_bnds_aux[:, :, 0] = self.lon_bnds["data"][newaxis, :, 0]
                lon_bnds_aux[:, :, 1] = self.lon_bnds["data"][newaxis, :, 1]
                lon_bnds_aux[:, :, 2] = self.lon_bnds["data"][newaxis, :, 1]
                lon_bnds_aux[:, :, 3] = self.lon_bnds["data"][newaxis, :, 0]

                lon_bnds = lon_bnds_aux
                del lon_bnds_aux

                lat_bnds_aux = empty(aux_shape)
                lat_bnds_aux[:, :, 0] = self.lat_bnds["data"][:, newaxis, 0]
                lat_bnds_aux[:, :, 1] = self.lat_bnds["data"][:, newaxis, 0]
                lat_bnds_aux[:, :, 2] = self.lat_bnds["data"][:, newaxis, 1]
                lat_bnds_aux[:, :, 3] = self.lat_bnds["data"][:, newaxis, 1]

                lat_bnds = lat_bnds_aux
                del lat_bnds_aux

                aux_b_lats = lat_bnds.reshape(
                    (lat_bnds.shape[0] * lat_bnds.shape[1], lat_bnds.shape[2])
                )
                aux_b_lons = lon_bnds.reshape(
                    (lon_bnds.shape[0] * lon_bnds.shape[1], lon_bnds.shape[2])
                )

                # Create all polygons vectorized via Shapely 2.x C-level call
                # Build (N, 5, 2) coordinate array: 5 ring vertices (closed), 2 = (x, y)
                n_cells = aux_b_lons.shape[0]
                coords = empty((n_cells, 5, 2))
                coords[:, 0, 0] = aux_b_lons[:, 0]
                coords[:, 0, 1] = aux_b_lats[:, 0]
                coords[:, 1, 0] = aux_b_lons[:, 1]
                coords[:, 1, 1] = aux_b_lats[:, 1]
                coords[:, 2, 0] = aux_b_lons[:, 2]
                coords[:, 2, 1] = aux_b_lats[:, 2]
                coords[:, 3, 0] = aux_b_lons[:, 3]
                coords[:, 3, 1] = aux_b_lats[:, 3]
                coords[:, 4, 0] = aux_b_lons[:, 0]
                coords[:, 4, 1] = aux_b_lats[:, 0]

                from shapely import linearrings as _linearrings
                from shapely import polygons as _polygons
                rings = _linearrings(coords)
                geometry = _polygons(rings)
            else:
                from numpy import meshgrid
                from geopandas import points_from_xy
                # Create Points
                lon_grid, lat_grid = meshgrid(self.lon["data"], self.lat["data"])
                geometry = points_from_xy(lon_grid.ravel(), lat_grid.ravel())

            fids = self.get_fids()
            gdf = GeoDataFrame(
                index=Index(name="FID", data=fids.ravel()),
                geometry=geometry,
                crs="EPSG:4326",
            )
            self.shapefile = gdf

        else:
            gdf = self.shapefile

        return gdf

    def write_shapefile(self, path):
        """
        Save spatial GeoDataFrame (shapefile).

        Parameters
        ----------
        path:str
            Path to the output file.
        """

        if self.shapefile is None:
            raise ValueError("Shapefile was not created.")

        if self.parallel_method in ["X", "Y"]:
            if self.size == 1:
                # In serial, avoid gather
                self.shapefile.to_file(path)
            else:
                # In parallel
                data = self.comm.gather(self.shapefile, root=0)
                if self.master:
                    data = concat(data)
                    data.to_file(path)

        elif self.parallel_method == "T":
            self.shapefile.to_file(path)

        return None

    def to_shapefile(self, path, time=None, lev=None, var_list=None, info=True):
        """
        Create shapefile from NES data.

        1. Create grid shapefile.
        2. Add variables to shapefile (as independent function).
        3. Write shapefile.

        Parameters
        ----------
        path:str
            Path to the output file.
        time:datetime
            Time stamp to select.
        lev:int
            Vertical level to select.
        var_list:List, str, None
            List (or single string) of the variables to be loaded and saved in the shapefile.
        info: bool
            Flag to allow/suppress warnings when the 'time' or 'lev' parameters are None. Default is True.
        """

        # If list is not defined, get all variables
        if var_list is None:
            var_list = list(self.variables.keys())
        else:
            if isinstance(var_list, str):
                var_list = [var_list]

        # Add warning for unloaded variables
        unloaded_vars = []
        for var_name in var_list:
            if self.variables[var_name]["data"] is None:
                unloaded_vars.append(var_name)
        if len(unloaded_vars) > 0:
            raise ValueError(
                "The variables {0} need to be loaded/created before using to_shapefile.".format(
                    unloaded_vars
                )
            )

        # Select first vertical level (if needed)
        if lev is None:
            if info:
                msg = "No vertical level has been specified. The first one will be selected."
                warn(msg)
                sys.stderr.flush()
            idx_lev = 0
        else:
            if lev not in self.lev["data"]:
                raise ValueError(
                    "Level {} is not available. Choose from {}".format(
                        lev, self.lev["data"]
                    )
                )
            idx_lev = lev

        # Select first time (if needed)
        if time is None:
            if info:
                msg = "No time has been specified. The first one will be selected."
                warn(msg)
                sys.stderr.flush()
            idx_time = 0
        else:
            if time not in self.time:
                raise ValueError(
                    "Time {} is not available. Choose from {}".format(time, self.time)
                )
            idx_time = self.time.index(time)

        # Create shapefile
        self.create_shapefile()

        # Load variables from original file and get data for selected time / level
        self.add_variables_to_shapefile(var_list, idx_lev, idx_time)

        # Write shapefile
        self.write_shapefile(path)

        return None

    def add_variables_to_shapefile(self, var_list, idx_lev=0, idx_time=0):
        """
        Add variables data to shapefile.

        var_list : List or str
           Variables to be loaded and saved in the shapefile.
        idx_lev : int
            Index of vertical level for which the data will be saved in the shapefile.
        idx_time : int
            Index of time for which the data will be saved in the shapefile.
        """

        mask = getattr(self, "_keep_mask", None)

        for var_name in var_list:
            flat_data = self.variables[var_name]["data"][idx_time, idx_lev, :].ravel()
            if mask is not None:
                flat_data = flat_data[mask]
            self.shapefile[var_name] = flat_data

        return None

    def get_centroids_from_coordinates(self):
        """
        Get centroids from geographical coordinates.

        Returns
        -------
        centroids_gdf: GeoPandasDataFrame
            Centroids dataframe.
        """

        # Get centroids from coordinates
        centroids = []
        for lat_ind in range(0, len(self.lat["data"])):
            for lon_ind in range(0, len(self.lon["data"])):
                centroids.append(Point(self.lon["data"][lon_ind], self.lat["data"][lat_ind]))

        # Create dataframe containing all points
        fids = self.get_fids()
        centroids_gdf = GeoDataFrame(
            index=Index(name="FID", data=fids.ravel()),
            geometry=centroids,
            crs="EPSG:4326",
        )

        return centroids_gdf

    def __gather_data_py_object(self, data_to_gather):
        """
        Gather all the variable data into the MPI rank 0 to perform a serial write.

        Returns
        -------
        data_list: dict
            Variables dictionary with all the data from all the ranks.
        """

        data_list = deepcopy(data_to_gather)
        for var_name in data_list.keys():
            try:
                # noinspection PyArgumentList
                data_aux = self.comm.gather(data_list[var_name]["data"], root=0)
                if self.rank == 0:
                    shp_len = len(data_list[var_name]["data"].shape)
                    add_dimension = False  # to Add a dimension
                    if self.parallel_method == "Y":
                        if shp_len == 2:
                            # if is a 2D concatenate over first axis
                            axis = 0
                        elif shp_len == 3:
                            # if is a 3D concatenate over second axis
                            axis = 1
                        else:
                            # if is a 4D concatenate over third axis
                            axis = 2
                    elif self.parallel_method == "X":
                        if shp_len == 2:
                            # if is a 2D concatenate over second axis
                            axis = 1
                        elif shp_len == 3:
                            # if is a 3D concatenate over third axis
                            axis = 2
                        else:
                            # if is a 4D concatenate over forth axis
                            axis = 3
                    elif self.parallel_method == "T":
                        if shp_len == 2:
                            # if is a 2D add dimension
                            add_dimension = True
                            axis = None  # Not used
                        elif shp_len == 3:
                            # if is a 3D concatenate over first axis
                            axis = 0
                        else:
                            # if is a 4D concatenate over second axis
                            axis = 0
                    else:
                        raise NotImplementedError(
                            "Parallel method '{meth}' is not implemented. Use one of these: {accept}".format(
                                meth=self.parallel_method, accept=["X", "Y", "T"]
                            )
                        )
                    if add_dimension:
                        data_list[var_name]["data"] = stack(data_aux)
                    else:
                        data_list[var_name]["data"] = concatenate(data_aux, axis=axis)
            except Exception as e:
                msg = f"**ERROR** an error has occurred while gathering the '{var_name}' variable.\n"
                print(msg)
                raise e

        return data_list

    def _gather_data(self, data_to_gather):
        """
        Gather all the variable data into the MPI rank 0 to perform a serial write.

        Returns
        -------
        data_to_gather: dict
            Variables to gather.
        """

        data_list = deepcopy(data_to_gather)
        for var_name in data_list.keys():
            if self.info and self.master:
                print("Gathering {0}".format(var_name))
            if data_list[var_name]["data"] is None:
                data_list[var_name]["data"] = None
            elif (
                isinstance(data_list[var_name]["data"], int)
                and data_list[var_name]["data"] == 0
            ):
                data_list[var_name]["data"] = 0
            else:
                shp_len = len(data_list[var_name]["data"].shape)
                # Collect local array sizes using the gather communication pattern
                rank_shapes = array(
                    self.comm.gather(data_list[var_name]["data"].shape, root=0)
                )
                sendbuf = data_list[var_name]["data"].flatten()
                sendcounts = array(self.comm.gather(len(sendbuf), root=0))
                if self.master:
                    recvbuf = empty(sum(sendcounts), dtype=type(sendbuf.max()))
                else:
                    recvbuf = None
                self.comm.Gatherv(
                    sendbuf=sendbuf, recvbuf=(recvbuf, sendcounts), root=0
                )
                if self.master:
                    recvbuf = split(recvbuf, cumsum(sendcounts))
                    # TODO ask
                    # I don"t understand why it is giving one more split
                    if len(recvbuf) > len(sendcounts):
                        recvbuf = recvbuf[:-1]
                    for i, shape in enumerate(rank_shapes):
                        recvbuf[i] = recvbuf[i].reshape(shape)
                    add_dimension = False  # to Add a dimension
                    if self.parallel_method == "Y":
                        if shp_len == 2:
                            # if is a 2D concatenate over first axis
                            axis = 0
                        elif shp_len == 3:
                            # if is a 3D concatenate over second axis
                            axis = 1
                        else:
                            # if is a 4D concatenate over third axis
                            axis = 2
                    elif self.parallel_method == "X":
                        if shp_len == 2:
                            # if is a 2D concatenate over second axis
                            axis = 1
                        elif shp_len == 3:
                            # if is a 3D concatenate over third axis
                            axis = 2
                        else:
                            # if is a 4D concatenate over forth axis
                            axis = 3
                    elif self.parallel_method == "T":
                        if shp_len == 2:
                            # if is a 2D add dimension
                            add_dimension = True
                            axis = None  # Not used
                        elif shp_len == 3:
                            # if is a 3D concatenate over first axis
                            axis = 0
                        else:
                            # if is a 4D concatenate over second axis
                            axis = 0
                    else:
                        raise NotImplementedError(
                            "Parallel method '{meth}' is not implemented. Use one of these: {accept}".format(
                                meth=self.parallel_method, accept=["X", "Y", "T"]
                            )
                        )
                    if add_dimension:
                        data_list[var_name]["data"] = stack(recvbuf)
                    else:
                        data_list[var_name]["data"] = concatenate(recvbuf, axis=axis)

        return data_list

    def to_mbtile(
        self,
        mbtiles_path: str,
        domain: str,
        geostructure: str = "",
    ) -> None:
        """Converts a monthly NES object into daily MBTiles

        Parameters
        ----------
        mbtiles_path : str
            Path to the folder where the tiles will be saved in
        domain: str
            Domain for which the MBTiles are being generated
        geostructure: str
            Optional geostructure file to be provided, allowing to skip the manual computation, by default None
        """

        def _set_agg_tiles_hash(mbtiles_path: str) -> str:
            """Compute agg_tiles_hash as MD5 over (z, x, y, tile_data) for all rows
            ordered by z,x,y, and store it in metadata.name='agg_tiles_hash'.

            Parameters
            ----------
            mbtiles_path : str
                Fully formed path to the mbtile

            Returns
            -------
            str
                hex digest
            """
            con = connect(mbtiles_path)
            try:
                cur = con.cursor()
                h = md5()
                for z, x, y, data in cur.execute(
                    "SELECT zoom_level, tile_column, tile_row, tile_data "
                    "FROM tiles ORDER BY zoom_level, tile_column, tile_row"
                ):
                    h.update(str(z).encode("utf-8"))
                    h.update(str(x).encode("utf-8"))
                    h.update(str(y).encode("utf-8"))
                    if data is not None:
                        h.update(data)
                # Empty tile
                digest = h.hexdigest().upper()

                cur.execute(
                    "INSERT OR REPLACE INTO metadata (name, value) VALUES (?, ?)",
                    ("agg_tiles_hash", digest),
                )
                con.commit()
                return digest
            finally:
                con.close()

        @njit(fastmath=True)
        def _filter_by_index_sum(data_arrays: list[ndarray]):
            """
            Filters a list of 1D arrays (same length) by checking each index across all
            arrays. Cells where every sector/pollutant value is zero are filtered out.

            Parameters
            ----------
            data_arrays : list[ndarray]
                List of 1D NumPy arrays representing different data sectors. All arrays must have the same length.

            Returns
            -------
            filtered : list[ndarray]
                List of filtered arrays with values retained only at indices where:
                - no array has a NaN at that index, and
                - at least one array has a non-zero value at that index.
            keep : ndarray[bool]
                Boolean mask of the original length indicating which indices were kept.
            """

            n = data_arrays[0].shape[0]
            num_sectors = len(data_arrays)
            keep = zeros(n, dtype=boolean)  # type: ignore

            # Parallelization with multithreading
            for i in range(n):
                valid = True
                any_nonzero = False
                for j in range(num_sectors):
                    val = data_arrays[j][i]
                    if isnan(val):
                        valid = False
                        break
                    if val != 0.0:
                        any_nonzero = True
                if valid and any_nonzero:
                    keep[i] = True

            # Count kept values
            count = 0
            for i in range(n):
                if keep[i]:
                    count += 1

            # Filter each sector
            filtered = []
            for j in range(num_sectors):
                arr = data_arrays[j]
                out = empty(count, dtype=arr.dtype)
                idx = 0
                for i in range(n):
                    if keep[i]:
                        out[idx] = arr[i]
                        idx += 1
                filtered.append(out)

            return filtered, keep

        if self.parallel_method != "T":
            raise ValueError(
                f"Warning, {self.parallel_method} incompatible with 'to_mbtiles', "
                f"we can only parallelize across the time domain."
            )

        # Snapshot variable names (avoid live dict_keys view)
        sectors = list(self.variables.keys())

        if not path.exists(mbtiles_path):
            # Create output directory if it doesn't exist
            makedirs(mbtiles_path, exist_ok=True)

        if geostructure != "":
            self.shapefile = read_file(geostructure)
        else:
            self.create_shapefile()

        # Keep a reference to the base shapefile — reuse every iteration
        # Instead of deep-copying the entire NES object via sel_time.
        base_shapefile = self.shapefile
        total_tiles = len(self.get_full_times())   # global count across all ranks

        # Cumulative offset so the progress counter reflects a global index
        local_counts = self.comm.allgather(len(self.time))
        rank_offset = sum(local_counts[:self.rank])
        loop_start = _timer()

        # Iterate through each timestep
        for tile_idx, date in enumerate(self.time, start=1):
            tile_start = _timer()
            idx_time = self.time.index(date)

            # Flatten sector data directly from the full array (level 0)
            sector_arrays = [self.variables[sector]["data"][idx_time, 0].ravel() for sector in sectors]

            # Filter 0s and near-0 values6
            filtered_arrays, keep_mask = _filter_by_index_sum(sector_arrays)

            # Build a lightweight filtered GeoDataFrame from the base shapefile
            filtered_shp = base_shapefile[keep_mask].reset_index(drop=True)
            for i, sector in enumerate(sectors):
                filtered_shp[sector] = filtered_arrays[i]

            del filtered_arrays, keep_mask, sector_arrays

            # Format date portion of file name
            date_part = str(date.year) + str(date.month).zfill(2) + str(date.day).zfill(2)

            # Assemble path to mbtiles file
            mbtiles = path.join(mbtiles_path, f"daily_{domain}_{date_part}.mbtiles")
            global_idx = rank_offset + tile_idx
            print(
                f"\t[Rank {self.rank}] Processing daily_{domain}_{date_part}.mbtiles"
                f" [{global_idx}/{total_tiles}]",
                flush=True,
            )
            # Convert geojson to mbtile
            options = {
                "MINZOOM": 5,
                "MAXZOOM": 12,
                "MAX_SIZE": 10000000000,
                "MAX_FEATURES": 10000000000,
            }
            filtered_shp.to_file(mbtiles, driver="MBTiles", layer="layer", **options)

            # Compute agg_tiles_hash and store in metadata
            _set_agg_tiles_hash(mbtiles)

            # Store per-sector total sums as metadata
            con = connect(mbtiles)
            try:
                cur = con.cursor()
                for sector in sectors:
                    total = float(filtered_shp[sector].sum())
                    cur.execute(
                        "INSERT OR REPLACE INTO metadata (name, value) VALUES (?, ?)",
                        (sector, str(total)),
                    )
                con.commit()
            finally:
                con.close()

            tile_elapsed = _timer() - tile_start
            print(
                f"\t[Rank {self.rank}] Tile daily_{domain}_{date_part}.mbtiles"
                f" generated in {tile_elapsed:.2f}s",
                flush=True,
            )

        local_elapsed = _timer() - loop_start

        # Synchronize all ranks and report summary from master
        self.comm.Barrier()
        max_elapsed = self.comm.reduce(local_elapsed, op=MPI.MAX, root=0)
        if self.master:
            print(
                f"\tAll {total_tiles} tiles for domain '{domain}' generated"
                f" in {max_elapsed:.2f}s (wall-clock, {self.size} ranks)",
                flush=True,
            )

    @staticmethod
    def lon_lat_to_cartesian_ecef(lon, lat):
        """
        # Convert observational/model geographic longitude/latitude coordinates to cartesian ECEF (Earth Centred,
        # Earth Fixed) coordinates, assuming WGS84 datum and ellipsoid, and that all heights = 0.
        # ECEF coordinates represent positions (in meters) as X, Y, Z coordinates, approximating the earth surface
        # as an ellipsoid of revolution.
        # This conversion is for the subsequent calculation of Euclidean distances of the model grid cell centres
        # from each observational station.
        # Defining the distance between two points on the earth's surface as simply the Euclidean distance
        # between the two lat/lon pairs could lead to inaccurate results depending on the distance
        # between two points (i.e. 1 deg. of longitude varies with latitude).

        Parameters
        ----------
        lon:Array
            Longitude values.
        lat:Array
            Latitude values.
        """

        lla = Proj(proj="latlong", ellps="WGS84", datum="WGS84")
        ecef = Proj(proj="geocent", ellps="WGS84", datum="WGS84")
        # x, y, z = pyproj.transform(lla, ecef, lon, lat, zeros(lon.shape), radians=False)
        # Deprecated: https://pyproj4.github.io/pyproj/stable/gotchas.html#upgrading-to-pyproj-2-from-pyproj-1
        transformer = Transformer.from_proj(lla, ecef)
        x, y, z = transformer.transform(lon, lat, zeros(lon.shape), radians=False)
        return column_stack([x, y, z])

    def add_4d_vertical_info(self, info_to_add):
        """
        To add the vertical information from other source.

        Parameters
        ----------
        info_to_add:nes.Nes, str
            Nes object with the vertical information as variable or str with the path to the NetCDF file that contains
            the vertical data.
        """

        return vertical_interpolation.add_4d_vertical_info(self, info_to_add)

    def interpolate_vertical(
        self,
        new_levels,
        new_src_vertical=None,
        kind="linear",
        extrapolate=None,
        info=None,
        overwrite=False,
    ):
        """
        Vertical interpolation function.

        Parameters
        ----------
        self:Nes
            Source Nes object.
        new_levels:List
            A List of new vertical levels.
        new_src_vertical:nes.Nes, str
            Nes object with the vertical information as variable or str with the path to the NetCDF file that contains
            the vertical data.
        kind:str
            Vertical methods type.
        extrapolate:bool or tuple or None or number or NaN
            If bool:
                - If True, both extrapolation options are set to "extrapolate".
                - If False, extrapolation options are set to ("bottom", "top").
            If tuple:
                - The first element represents the extrapolation option for the lower bound.
                - The second element represents the extrapolation option for the upper bound.
                - If any element is bool:
                    - If True, it represents "extrapolate".
                    - If False:
                        - If it"s the first element, it represents "bottom".
                        - If it"s the second element, it represents "top".
                - If any element is None, it is replaced with numpy.nan.
                - Other numeric values are kept as they are.
                - If any element is NaN, it is kept as NaN.
            If None:
                - Both extrapolation options are set to (NaN, NaN).
            If number:
                - Both extrapolation options are set to the provided number.
            If NaN:
                - Both extrapolation options are set to NaN.
        info: None, bool
            Indicates if you want to print extra information.
        overwrite: bool
            Indicates if you want to compute the vertical interpolation in the same object or not.
        """

        return vertical_interpolation.interpolate_vertical(
            self,
            new_levels,
            new_src_vertical=new_src_vertical,
            kind=kind,
            extrapolate_options=extrapolate,
            info=info,
            overwrite=overwrite,
        )

    def interpolate_horizontal(
        self,
        dst_grid,
        weight_matrix_path=None,
        kind="NearestNeighbour",
        n_neighbours=4,
        info=False,
        to_providentia=False,
        only_create_wm=False,
        wm=None,
        flux=False,
        keep_nan=False,
        fix_border=False,
    ):
        """
        Horizontal methods from the current grid to another one.

        Parameters
        ----------
        dst_grid:nes.Nes
            Final projection Nes object.
        weight_matrix_path:str, None
            Path to the weight matrix to read/create.
        kind:str
            Kind of horizontal methods. choices = ["NearestNeighbour", "Conservative"].
        n_neighbours: int
            Used if kind == NearestNeighbour. Number of nearest neighbours to interpolate. Default: 4.
        info: bool
            Indicates if you want to print extra info during the methods process.
        to_providentia:bool
            Indicates if we want the interpolated grid in Providentia format.
        only_create_wm:bool
            Indicates if you want to only create the Weight Matrix.
        wm:Nes
            Weight matrix Nes File.
        flux:bool
            Indicates if you want to calculate the weight matrix for flux variables.
        keep_nan:bool
            Indicates if you want to keep nan values after the interpolation
        fix_border:bool
            Indicates if you want to fix the borders on the NN interpolation by expanding it 1 cell and contracting
            it later.
        """

        return horizontal_interpolation.interpolate_horizontal(
            self,
            dst_grid,
            weight_matrix_path=weight_matrix_path,
            kind=kind,
            n_neighbours=n_neighbours,
            info=info,
            to_providentia=to_providentia,
            only_create_wm=only_create_wm,
            wm=wm,
            flux=flux,
            keep_nan=keep_nan,
            fix_border=fix_border,
        )

    def spatial_join(
        self, ext_shp, method=None, var_list=None, info=False, apply_bbox=True
    ):
        """
        Compute overlay intersection of two GeoPandasDataFrames.

        Parameters
        ----------
        ext_shp:GeoPandasDataFrame or str
            File or path from where the data will be obtained on the intersection.
        method:str
            Overlay method. Accepted values: ["nearest", "intersection", "centroid"].
        var_list:List or None
            Variables that will be included in the resulting shapefile.
        info:bool
            Indicates if you want to print the process info.
        apply_bbox:bool
            Indicates if you want to reduce the shapefile to a bbox.
        """

        return spatial_join(
            self,
            ext_shp=ext_shp,
            method=method,
            var_list=var_list,
            info=info,
            apply_bbox=apply_bbox,
        )

    def calculate_grid_area(self, overwrite=True):
        """
        Get coordinate bounds and call function to calculate the area of each cell of a grid.

        Parameters
        ----------
        self:nes.Nes
            Source projection Nes Object.
        overwrite:bool
            Indicates if we want to overwrite the grid area.
        """

        if ("cell_area" not in self.cell_measures.keys()) or overwrite:
            grid_area = cell_measures.calculate_grid_area(self)
            grid_area = grid_area.reshape(
                [self.lat["data"].shape[0], self.lon["data"].shape[-1]]
            )
            self.cell_measures["cell_area"] = {"data": grid_area}
        else:
            grid_area = self.cell_measures["cell_area"]["data"]

        return grid_area

    @staticmethod
    def calculate_geometry_area(
        geometry_list,
        earth_radius_minor_axis=6356752.3142,
        earth_radius_major_axis=6378137.0,
    ):
        """
        Get coordinate bounds and call function to calculate the area of each cell of a set of geometries.

        Parameters
        ----------
        geometry_list:List
            A List with polygon geometries.
        earth_radius_minor_axis:float
            Radius of the minor axis of the Earth.
        earth_radius_major_axis:float
            Radius of the major axis of the Earth.
        """

        return cell_measures.calculate_geometry_area(
            geometry_list,
            earth_radius_minor_axis=earth_radius_minor_axis,
            earth_radius_major_axis=earth_radius_major_axis,
        )

    @staticmethod
    def get_earth_radius(ellps):
        """
        Get minor and major axis of Earth.

        Parameters
        ----------
        ellps:str
            Spatial reference system.
        """

        # WGS84 with radius defined in Cartopy source code
        earth_radius_dict = {"WGS84": [6371000.0, 6371000.0]}

        return earth_radius_dict[ellps]

    def create_providentia_exp_centre_coordinates(self):
        """
        Calculate centre latitudes and longitudes from original coordinates and store as 2D arrays.

        Returns
        ----------
        model_centre_lat:dict
            Dictionary with data of centre coordinates for latitude in 2D (latitude, longitude).
        model_centre_lon:dict
            Dictionary with data of centre coordinates for longitude in 2D (latitude, longitude).
        """

        raise NotImplementedError(
            "create_providentia_exp_centre_coordinates function is not implemented by default"
        )

    # noinspection DuplicatedCode
    def create_providentia_exp_grid_edge_coordinates(self):
        """
        Calculate grid edge latitudes and longitudes and get model grid outline.

        Returns
        ----------
        grid_edge_lat:dict
            Dictionary with data of grid edge latitudes.
        grid_edge_lon:dict
            Dictionary with data of grid edge longitudes.
        """
        raise NotImplementedError(
            "create_providentia_exp_grid_edge_coordinates function is not implemented by default"
        )

    def _detect_longitude_format(self):
        """
        Determines whether longitude values are in the [0, 360] or [-180, 180] range.

        Returns
        ---------
            bool: True if longitudes are in [0, 360], False otherwise.
        """
        longitudes = self.lon["data"]
        longitudes = array(longitudes)
        if all((longitudes >= 0) & (longitudes <= 360)):
            return True
        elif all((longitudes >= -180) & (longitudes <= 180)):
            return False
        else:
            return False

    def _check_if_data_is_loaded(self):
        """
        Verifies that data is loaded for all variables.

        Raises
        -------
            ValueError: If any variable's data is missing.
        """
        for variable in self.variables.keys():
            if self.variables[variable]["data"] is None:
                raise ValueError(
                    f"All variables data must be loaded before using this function. Data for {variable} is not loaded."
                )

    def convert_longitudes(self):
        """
        Converts longitudes from the [0, 360] range to the [-180, 180] range.

        Parameters
        ------------
            path (str): The file path where the converted data will be saved.

        Raises
        --------
            ValueError: If the method is run in parallel processing mode.
            ValueError: If longitudes are already in [-180, 180] format or an unrecognized format.
            ValueError: If data is not fully loaded before conversion.
        """
        if self.comm.size > 1:
            raise ValueError("This method is currently only available in serial.")

        if not self._detect_longitude_format():
            raise ValueError(
                "Longitudes are already in [-180, 180] format or another unrecognised format."
            )

        self._check_if_data_is_loaded()

        # adjust longitude values
        lon = self.lon
        lon_data = lon["data"]
        lon_data = lon_data % 360
        lon_data[lon_data > 180] -= 360
        sorted_indices = argsort(lon_data)
        self.lon["data"] = lon_data[sorted_indices]
        self.set_full_longitudes(self.lon)

        # adjust longitude bounds
        lon_bnds_data = self.lon_bnds["data"]
        lon_bnds_data = lon_bnds_data % 360
        lon_bnds_data[lon_bnds_data > 180] -= 360
        lon_bnds_sorted = lon_bnds_data[sorted_indices]

        if (lon_bnds_sorted[0][0] > lon_bnds_sorted[0][1]) and (
            isclose(lon_bnds_sorted[0][0], 180)
        ):
            lon_bnds_sorted[0][0] = -180
        elif (lon_bnds_sorted[-1][0] > lon_bnds_sorted[-1][1]) and (
            isclose(lon_bnds_sorted[-1][1], -180)
        ):
            lon_bnds_sorted[-1][1] = 180

        self.lon_bnds["data"] = lon_bnds_sorted
        self.set_full_longitudes_boundaries(self.lon_bnds)

        # adjust variables which have longitude in their dimensions
        for name, var in self.variables.items():
            if "longitude" in var["dimensions"] or "lon" in var["dimensions"]:
                data = var["data"]
                reordered_data = take(data, sorted_indices, axis=3)
                self.variables[name]["data"] = reordered_data

        return None

    def get_bbox(self) -> Tuple[float, float, float, float]:
        """
        Return the geographic bounding box of the current grid.

        The bounding box is given in **(min_lon, min_lat, max_lon, max_lat)** order,
        following common GIS conventions (e.g., Shapely, GeoPandas).

        Notes
        -----
        * If latitude/longitude bounds are not yet computed, the method calls
          :py:meth:`create_spatial_bounds` internally.
        * All coordinates are expressed in degrees using the grid's native
          longitude convention (typically ``[-180, 180]`` after any conversion).

        Returns
        -------
        tuple of float
            Tuple containing (min_lon, min_lat, max_lon, max_lat), all in degrees.
        """
        if self.lat_bnds is None or self.lat_bnds is None:
            self.create_spatial_bounds()

        min_lat = float(self.lat_bnds["data"].min())
        max_lat = float(self.lat_bnds["data"].max())
        min_lon = float(self.lon_bnds["data"].min())
        max_lon = float(self.lon_bnds["data"].max())

        return min_lon, min_lat, max_lon, max_lat

    def get_full_bbox(self, only_master=False) -> Tuple[float, float, float, float]:
        """
        Return the geographic bounding box of the current grid.

        The bounding box is given in **(min_lon, min_lat, max_lon, max_lat)** order,
        following common GIS conventions (e.g., Shapely, GeoPandas).

        Notes
        -----
        * If latitude/longitude bounds are not yet computed, the method calls
          :py:meth:`create_spatial_bounds` internally.
        * All coordinates are expressed in degrees using the grid's native
          longitude convention (typically ``[-180, 180]`` after any conversion).

        Parameters
        ----------
        only_master:bool
            Indicates if only master needs to get that info

        Returns
        -------
        tuple of float
            Tuple containing (min_lon, min_lat, max_lon, max_lat), all in degrees.
        """
        if self.lat_bnds is None or self.lat_bnds is None:
            self.create_spatial_bounds()

        if self.master:
            min_lat = float(self._full_lat_bnds["data"].min())
            max_lat = float(self._full_lat_bnds["data"].max())
            min_lon = float(self._full_lon_bnds["data"].min())
            max_lon = float(self._full_lon_bnds["data"].max())
            bbox = (min_lon, min_lat, max_lon, max_lat)
        else:
            bbox = None

        if only_master:
            return bbox

        bbox = self.comm.bcast(bbox, root=0)

        return bbox

    def _get_plot_data(
        self,
        var_name: str,
        level: int,
        time_step: int,
        threshold: float = 1e-20,
        percentile_clip: tuple = (),
    ) -> ndarray:
        """
        Extract and process data for plotting.

        Parameters
        ----------
        var_name : str
            Name of the variable to extract.
        level : int
            Vertical level index for the variable.
        time_step : int
            Time step index for the variable.
        threshold : float, optional
            Threshold below which data values are set to NaN, by default 1e-20
        percentile_clip : tuple, optional
            Tuple containing lower and upper percentiles for clipping, by default ()

        Returns
        -------
        ndarray
            Processed data array ready for plotting.
        """
        import numpy as np

        data = self.variables[var_name]["data"][time_step, level, :, :]
        data[data < threshold] = np.nan

        if percentile_clip != ():
            lower, upper = percentile_clip
            vmin_clip = np.nanpercentile(data, lower)
            vmax_clip = np.nanpercentile(data, upper)
            data = np.clip(data, vmin_clip, vmax_clip)

        return data

    def _get_mesh_coordinates(self, stride: int = 1) -> Tuple[ndarray, ndarray]:
        """
        Get coordinates for plotting, handling 1D/2D and striding.

        Parameters
        ----------
        stride : int, optional
            Stride value for downsampling the coordinates, by default 1

        Returns
        -------
        Tuple[ndarray, ndarray]
            Tuple containing longitude and latitude meshgrid arrays.
        """
        import numpy as np

        lon = self.lon["data"]
        lat = self.lat["data"]

        if stride > 1:
            lon = lon[::stride] if lon.ndim == 1 else lon[::stride, ::stride]
            lat = lat[::stride] if lat.ndim == 1 else lat[::stride, ::stride]

        if lon.ndim == 1 and lat.ndim == 1:
            lon_mesh, lat_mesh = np.meshgrid(lon, lat)
        else:
            lon_mesh = lon
            lat_mesh = lat

        return lon_mesh, lat_mesh

    def _setup_normalization(
        self, data: ndarray, scale: str, vmin: float, vmax: float
    ) -> Tuple:
        """
        Setup normalization for plotting.

        Parameters
        ----------
        data : ndarray
            Data for plotting
        scale : str
            Type of scale to apply to the colorbar ("linear", "log", "power").
        vmin : float
            Minimum data value for color scaling. Default is None.
        vmax : float
            Maximum data value for color scaling. Default is None.

        Returns
        -------
        Tuple
            Tuple containing the normalization object, vmin, and vmax.
        """
        from matplotlib.colors import LogNorm, Normalize, PowerNorm

        data_pos = data.copy()
        data_pos[data_pos <= 0] = nan

        if vmin is None:
            vmin = nanmin(data_pos[data_pos > 0]) if (data_pos > 0).any() else 1e-10
        if vmax is None:
            vmax = nanmax(data_pos)

        if scale == "log":
            norm = LogNorm(vmin=vmin, vmax=vmax)
        elif scale == "power":
            norm = PowerNorm(gamma=0.3, vmin=vmin, vmax=vmax)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)

        return norm, vmin, vmax

    def _set_plot_labels(
        self, ax, var_name: str, level: int, time_step: int, title_suffix: str = ""
    ):
        """
        Set standard labels and title.


        Parameters
        ----------
        ax : plt.ax object
            Matplotlib axes object to set labels on.
        var_name : str
            Name of the variable to plot.
        level : int
            Vertical level index for the variable.
        time_step : int
            Time step index for the variable.
        title_suffix : str, optional
            Additional string to append to the plot title, by default ""
        """

        title_parts = [var_name]
        if "long_name" in self.variables[var_name]:
            title_parts = [self.variables[var_name]["long_name"]]

        title_parts.append(f"Level: {self.lev['data'][level]}")
        if len(self.time) > 0:
            title_parts.append(f"Time: {self.time[time_step]}")

        ax.set_title(" - ".join(title_parts) + title_suffix)

    def _plot_colormesh(
        self,
        var_name: str,
        level: int,
        time_step: int,
        data: ndarray,
        scale: str,
        save: bool,
        ax=None,
        add_colorbar=True,
        anim=False,
        **kwargs,
    ) -> Tuple:
        """Generates a colormesh plot for the specified variable, level, and time step.

        Parameters
        ----------
        var_name : str
            Variable name to plot.
        level : int
            Level index for the variable.
        time_step : int
            Time step index for the variable.
        data : ndarray
            Data array to plot.
        scale : str
            Type of scale to apply to the colorbar ("linear", "log", "power").
        save : bool
            Whether to save the plot to a file.
        ax : Axes, optional
            Matplotlib axes to plot on. If None, a new figure and axes are created.
        add_colorbar : bool, optional
            Whether to add a colorbar to the plot. Default is True.
        anim : bool, optional
            Whether the plot is part of an animation. Default is False.

        Returns
        -------
        tuple[fig, ax, im]
            Matplotlib figure, axes, and image objects.
        """
        import matplotlib.pyplot as plt

        # Get mesh boundaries for pcolormesh
        lon_bnds_mesh, lat_bnds_mesh = self.get_spatial_bounds_mesh_format()

        # Create figure and axes
        if ax is None:
            fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (10, 8)))
        else:
            fig = ax.figure

        # Extract plot parameters
        cmap = kwargs.pop("cmap", "viridis")
        vmin = kwargs.pop("vmin", None)
        vmax = kwargs.pop("vmax", None)

        norm, vmin, vmax = self._setup_normalization(data, scale, vmin, vmax)

        # Create pcolormesh plot
        im = ax.pcolormesh(
            lon_bnds_mesh,
            lat_bnds_mesh,
            data,
            cmap=cmap,
            norm=norm,
            antialiased=False,
            **kwargs,
        )

        # Add colorbar
        if add_colorbar:
            cbar = plt.colorbar(im, ax=ax)
            if "units" in self.variables[var_name]:
                cbar.set_label(self.variables[var_name]["units"])

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        self._set_plot_labels(ax, var_name, level, time_step)

        ax.set_aspect("auto")

        plt.tight_layout()
        if not anim:
            if save:
                plt.savefig(
                    f"{var_name}_level{level}_time{time_step}_{scale}_colormesh.png",
                    dpi=600,
                )
            else:
                plt.show()

        # Return plot object in case further manipulation is needed
        return fig, ax, im

    def _plot_contour(
        self,
        var_name: str,
        level: int,
        time_step: int,
        data: ndarray,
        scale: str,
        save: bool,
        ax=None,
        add_colorbar=True,
        anim=False,
        **kwargs,
    ) -> Tuple:
        """

        Parameters
        ----------
        var_name : str
            Variable name to plot.
        level : int
            Level index for the variable.
        time_step : int
            Time step index for the variable.
        data : ndarray
            Data array to plot.
        scale : str
            Scale type for color normalization (e.g., "linear", "log", "power").
        save : bool
            Whether to save the plot to a file.
        ax : Axes, optional
            Matplotlib axes to plot on. If None, a new figure and axes are created.
        add_colorbar : bool, optional
            Whether to add a colorbar to the plot. Default is True.
        anim : bool, optional
            Whether the plot is part of an animation. Default is False.

        Returns
        -------
        tuple[fig, ax, im]
            Matplotlib figure, axes, and image objects.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (10, 8)))
        else:
            fig = ax.figure

        lon_mesh, lat_mesh = self._get_mesh_coordinates()

        cmap = kwargs.pop("cmap", "viridis")
        vmin = kwargs.pop("vmin", None)
        vmax = kwargs.pop("vmax", None)
        clevels = kwargs.pop("clevels", 15)

        norm, vmin, vmax = self._setup_normalization(data, scale, vmin, vmax)

        if scale == "log" and isinstance(clevels, int):
            clevels = logspace(log10(vmin), log10(vmax), clevels)

        im = ax.contourf(
            lon_mesh,
            lat_mesh,
            data,
            levels=clevels,
            cmap=cmap,
            norm=norm,
            antialiased=False,
            **kwargs,
        )

        if add_colorbar:
            cbar = plt.colorbar(im, ax=ax)
            if "units" in self.variables[var_name]:
                cbar.set_label(self.variables[var_name]["units"])

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        self._set_plot_labels(ax, var_name, level, time_step)

        if not anim:
            plt.tight_layout()
            if save:
                plt.savefig(
                    f"{var_name}_level{level}_time{time_step}_{scale}_contour.png",
                    dpi=600,
                )
            else:
                plt.show()

        return fig, ax, im

    def _plot_histogram(
        self,
        var_name: str,
        level: int,
        time_step: int,
        data: ndarray,
        scale: str,
        save: bool,
        **kwargs,
    ):
        """Plot histogram of data values.

        Parameters
        ----------
        var_name : str
            Name of variable to plot.
        level : int
            Level index to plot.
        time_step : int
            Time step index to plot.
        data : ndarray
            Data array to plot.
        scale : str
            Type of scale to apply to the color normalization ("linear", "log", "power").
        save : bool
            Whether to save the plot to a file.

        Returns
        -------
        tuple[fig, ax, None]
            Matplotlib figure, axes, and None (no image object for histogram).
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (10, 6)))

        data_flat = data.flatten()
        data_flat = data_flat[~isnan(data_flat)]

        bins = kwargs.pop("bins", 50)
        color = kwargs.pop("color", "blue")
        alpha = kwargs.pop("alpha", 0.7)
        include_stats = kwargs.pop("include_stats", True)
        kwargs.pop("cmap", None)

        if scale == "log":
            bins = logspace(log10(data_flat.min()), log10(data_flat.max()), 40)
            counts, bin_edges = histogram(data_flat, bins=bins)
            nonzero_mask = counts > 0
            counts = counts[nonzero_mask]
            bin_left_edges = bin_edges[:-1][nonzero_mask]
            bar_widths = diff(bin_edges)[nonzero_mask]
            ax.bar(
                bin_left_edges,
                counts,
                width=bar_widths,
                align="edge",
                edgecolor="black",
                color=color,
                alpha=alpha,
                **kwargs,
            )
            ax.set_yscale("log")
        elif scale == "power":
            ax.hist(
                data_flat,
                bins=bins,
                color=color,
                alpha=alpha,
                edgecolor="black",
                **kwargs,
            )
            ax.set_yscale("function", functions=(lambda x: x**0.5, lambda x: x**2))
        elif scale == "linear":
            ax.hist(
                data_flat,
                bins=bins,
                color=color,
                alpha=alpha,
                edgecolor="black",
                **kwargs,
            )
        else:
            raise ValueError(
                "scale must be either 'linear', 'log' or 'power' for histogram"
            )

        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")

        self._set_plot_labels(
            ax, var_name, level, time_step, title_suffix=" - Histogram"
        )

        if include_stats:
            stats_text = (
                f"Mean: {mean(data_flat):.2e}\n"
                f"Std: {std(data_flat):.2e}\n"
                f"Min: {min(data_flat):.2e}\n"
                f"Max: {max(data_flat):.2e}"
            )
            ax.text(
                0.95,
                0.95,
                stats_text,
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                fontsize=14,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        plt.tight_layout()
        if save:
            plt.savefig(
                f"{var_name}_level{level}_time{time_step}_{scale}_histogram.png",
                dpi=600,
            )
        else:
            plt.show()

        return fig, ax, None

    def _plot_surface(
        self,
        var_name: str,
        level: int,
        time_step: int,
        data: npt.NDArray,
        scale: str,
        save: bool,
        **kwargs,
    ) -> Tuple:
        """Generate surface plot

        Parameters
        ----------
        var_name : str
            Name of variable to plot.
        level : int
            Level index to plot.
        time_step : int
            Time step index to plot.
        data : npt.NDArray
            Data array to plot.
        scale : str
            Type of scale to apply to the color normalization ("linear", "log", "power").
        save : bool
            Whether to save the plot to a file.

        Returns
        -------
        tuple[fig, ax, surf]
            Matplotlib figure, axes, and surface plot object.
        """

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=kwargs.pop("figsize", (12, 9)))
        ax = fig.add_subplot(111, projection="3d")

        stride = kwargs.pop("stride", 1)
        lon_mesh, lat_mesh = self._get_mesh_coordinates(stride=stride)

        # Downsample data if stride > 1
        if stride > 1:
            data = data[::stride, ::stride]

        cmap = kwargs.pop("cmap", "viridis")
        vmin = kwargs.pop("vmin", None)
        vmax = kwargs.pop("vmax", None)
        rstride = kwargs.pop("rstride", 1)
        cstride = kwargs.pop("cstride", 1)

        norm, vmin, vmax = self._setup_normalization(data, scale, vmin, vmax)

        elev = kwargs.pop("elev", 30)
        azim = kwargs.pop("azim", 45)
        ax.view_init(elev=elev, azim=azim)

        surf = ax.plot_surface(
            lon_mesh,
            lat_mesh,
            data,
            cmap=cmap,
            norm=norm,
            rstride=rstride,
            cstride=cstride,
            **kwargs,
        )

        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        if "units" in self.variables[var_name]:
            cbar.set_label(self.variables[var_name]["units"])

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_zlabel(self.variables[var_name].get("units", var_name))

        self._set_plot_labels(ax, var_name, level, time_step)

        plt.tight_layout()
        if save:
            plt.savefig(
                f"{var_name}_level{level}_time{time_step}_{scale}_surface.png", dpi=300
            )
        else:
            plt.show()

        return fig, ax, surf

    def _plot_animation(
        self,
        var_name: str,
        level: int,
        threshold: float,
        scale: str,
        animation_method: str = "colormesh",
        **kwargs,
    ):
        """Create an animation for a variable over time for a given level.

        Parameters
        ----------
        var_name : str
            Variable name to plot.
        level : int
            Vertical level of the NetCDF to plot
        threshold : float
            Threshold value below which data is considered invalid.
        scale : str
            Scale to use for the color normalization ('linear', 'log', 'power').
        animation_method : str, default 'colormesh'
            Plotting method to use for animation frames. Options are 'colormesh' or 'contour'.

        Common kwargs
        -------
        cmap : str
            Colormap to use for the plot. Default is 'viridis'.
        vmin : float
            Minimum data value for color scaling. Default is None (auto-calculated).
        vmax : float
            maximum data value for color scaling. Default is None (auto-calculated).
        fps : int
            Frames per second for the animation. Default is 10.
        dpi : int
            Dots per inch for the output video. Default is 300.
        percentile_clip : tuple
            Percentile range for clipping data values (lower, upper). Default is None.
        clevels : int or array-like
            Number of contour levels or specific levels for contour method. Default is 15.
        """

        import matplotlib.pyplot as plt
        from matplotlib.animation import FFMpegWriter, FuncAnimation

        # Pop kwargs
        cmap = kwargs.pop("cmap", "viridis")
        vmin = kwargs.pop("vmin", None)
        vmax = kwargs.pop("vmax", None)
        fps = kwargs.pop("fps", 10)
        dpi = kwargs.pop("dpi", 300)
        percentile_clip = kwargs.pop("percentile_clip", ())
        clevels = kwargs.pop("clevels", 15)

        output_file = kwargs.pop(
            "output_file",
            f"{var_name}_level{level}_animation_{animation_method}_{scale}.mp4",
        )

        num_timesteps = self.variables[var_name]["data"].shape[0]

        # Calculate global vmin/vmax
        if vmin is None or vmax is None:
            if percentile_clip != ():
                all_data = []
                for t in range(num_timesteps):
                    data_t = self._get_plot_data(var_name, level, t, threshold)
                    all_data.append(data_t)
                all_data_combined = concatenate([d.ravel() for d in all_data])
                all_data_combined = all_data_combined[~isnan(all_data_combined)]

                lower, upper = percentile_clip
                vmin_global = (
                    nanpercentile(all_data_combined, lower) if vmin is None else vmin
                )
                vmax_global = (
                    nanpercentile(all_data_combined, upper) if vmax is None else vmax
                )
                del all_data
                del all_data_combined
            else:
                # Iterative min/max to save memory
                vmin_global = inf
                vmax_global = -inf
                for t in range(num_timesteps):
                    data_t = self._get_plot_data(var_name, level, t, threshold)
                    vmin_t = nanmin(data_t)
                    vmax_t = nanmax(data_t)
                    if vmin_t < vmin_global:
                        vmin_global = vmin_t
                    if vmax_t > vmax_global:
                        vmax_global = vmax_t

                if vmin is not None:
                    vmin_global = vmin
                if vmax is not None:
                    vmax_global = vmax
        else:
            vmin_global = vmin
            vmax_global = vmax

        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (10, 8)))

        # Initial frame
        data_init = self._get_plot_data(var_name, level, 0, threshold, percentile_clip)

        if animation_method == "colormesh":
            _, _, im = self._plot_colormesh(
                var_name,
                level,
                0,
                data_init,
                scale,
                save=False,
                ax=ax,
                add_colorbar=True,
                cmap=cmap,
                vmin=vmin_global,
                vmax=vmax_global,
                anim=True,
                **kwargs,
            )
        elif animation_method == "contour":
            _, _, im = self._plot_contour(
                var_name,
                level,
                0,
                data_init,
                scale,
                save=False,
                ax=ax,
                add_colorbar=True,
                cmap=cmap,
                vmin=vmin_global,
                vmax=vmax_global,
                clevels=clevels,
                anim=True,
                **kwargs,
            )

        plt.tight_layout()

        def update(frame):
            """Update function for animation.

            Parameters
            ----------
            frame : int
                Current time-step index for the animation.

            Returns
            -------
            list
                Current frame plotted in either contour or colormesh.
            """
            data_frame = self._get_plot_data(
                var_name, level, frame, threshold, percentile_clip
            )

            if animation_method == "colormesh":
                im.set_array(data_frame.ravel())
                self._set_plot_labels(ax, var_name, level, frame)
            elif animation_method == "contour":
                ax.clear()
                self._plot_contour(
                    var_name,
                    level,
                    frame,
                    data_frame,
                    scale,
                    save=False,
                    ax=ax,
                    add_colorbar=False,
                    cmap=cmap,
                    vmin=vmin_global,
                    vmax=vmax_global,
                    clevels=clevels,
                    anim=True,
                    **kwargs,
                )
            return [ax]

        anim = FuncAnimation(
            fig, update, frames=num_timesteps, interval=1000 / fps, blit=False
        )
        writer = FFMpegWriter(fps=fps, bitrate=2000)
        anim.save(output_file, writer=writer, dpi=dpi)
        plt.close(fig)

        return None

    def plot(
        self,
        var_name,
        level,
        time_step,
        method="colormesh",
        save=False,
        threshold=1e-20,
        **kwargs,
    ) -> Tuple:
        """
        Plot a variable from the Nes object.

        Parameters
        ----------
        var_name : str
            Name of the variable to plot.
        level : int
            Vertical level index to plot.
        time_step : int
            Time step index to plot.
        method : str, optional
            Plotting method. Choices:
            - "colormesh": 2D pseudocolor plot
            - "histogram": Histogram of data values
            - "contour": Filled contour plot
            - "surface": 3D surface plot
            - "animation": Create MP4 video animation through all time steps using either colormesh or contour.
              By default "colormesh".
        save : bool, optional
            Whether to save the plot to a file. Default is False (show plot interactively).
        threshold : float, optional
            Threshold value below which data is considered invalid. Default is 1e-20.
        **kwargs : dict
            Additional keyword arguments to pass to matplotlib plotting functions.
            Common kwargs:
            - cmap : str, optional - Matplotlib colormap name (default: 'viridis')
            - vmin, vmax : float, optional - Color scale limits
            - scale : str, optional - Type of scale to use ("linear", "log", "power") (default: "linear")
            - percentile_clip : tuple, optional - Clip data to percentiles (e.g., (5, 95))
            - figsize : tuple, optional - Figure size (default: (10, 8) or (12, 9) for 3D)

        colormesh specific kwargs
        ----------
        shading : str, optional - Shading method for colormesh: 'flat', 'nearest', 'gouraud', 'auto' (default: 'auto')

        histogram specific kwargs
        ----------
        bins : int, optional - Number of histogram bins (default: 50)
        color : str, optional - Histogram bar color (default: 'blue')
        alpha : float, optional - Histogram bar transparency (default: 0.7)
        include_stats : bool, optional - Whether to include statistics box on histogram (default: True)

        contour specific kwargs
        ----------
        clevels : int or array-like, optional - Number of contour levels or specific levels (default: 15)

        surface specific kwargs
        ----------
        stride : int, optional - Downsample data by plotting every nth point (default: 1)
        rstride : int, optional - Row stride for surface plot (default: 1)
        cstride : int, optional - Column stride for surface plot (default: 1)
        elev : float, optional - Elevation angle for 3D view (default: 30)
        azim : float, optional - Azimuth angle for 3D view (default: 45)

        animation specific kwargs
        ----------
        animation_method : str, optional - Plotting method for animation frames: 'colormesh' or 'contour'
            (default: 'colormesh')
        fps : int, optional - Frames per second for animation (default: 10)
        output_file : str, optional - Output filename for animation
            (default: '{var_name}_level{level}_animation.mp4')
        dpi : int, optional - Resolution for animation (default: 300)

        Raises
        ------
        ValueError
            If the variable is not loaded or method is not recognized.
        NotImplementedError
            If trying to plot in parallel mode (only serial plotting is supported).

        Returns
        -------
        Tuple[fig, ax, im]
            Matplotlib figure, axes, and image/surface object for single frame plots.
            For animation method, returns None.
        """
        # Check if running in serial mode
        if self.size > 1:
            raise NotImplementedError("Plotting is only available in serial mode.")

        # Check if variable exists and is loaded
        if var_name not in self.variables:
            raise ValueError(f"Variable '{var_name}' not found in the dataset.")

        if self.variables[var_name]["data"] is None:
            raise ValueError(
                f"Data for variable '{var_name}' is not loaded. Please load it first."
            )

        # Handle log scale
        scale = kwargs.pop("scale", "linear")
        percentile_clip = kwargs.pop("percentile_clip", None)

        if method == "animation":
            # Check that ffmpeg is installed
            if which("ffmpeg") is None:
                # TODO: tell how to install
                raise EnvironmentError(
                    "ffmpeg is not installed or not found in PATH. Please install ffmpeg to use animation plotting."
                )
            self._plot_animation(var_name, level, threshold, scale, **kwargs)
            return None, None, None

        # Extract data for single time step methods
        data = self._get_plot_data(
            var_name, level, time_step, threshold, percentile_clip
        )

        if method == "colormesh":
            fig, ax, im = self._plot_colormesh(
                var_name, level, time_step, data, scale, save, **kwargs
            )
            return fig, ax, im

        elif method == "contour":
            fig, ax, im = self._plot_contour(
                var_name, level, time_step, data, scale, save, **kwargs
            )
            return fig, ax, im

        elif method == "histogram":
            fig, ax, im = self._plot_histogram(
                var_name, level, time_step, data, scale, save, **kwargs
            )
            return fig, ax, im

        elif method == "surface":
            fig, ax, im = self._plot_surface(
                var_name, level, time_step, data, scale, save, **kwargs
            )
            return fig, ax, im
        else:
            raise ValueError(f"Unknown plotting method: '{method}'.")
