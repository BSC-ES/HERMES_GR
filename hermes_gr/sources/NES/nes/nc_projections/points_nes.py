#!/usr/bin/env python
# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import sys
from warnings import warn
from numpy import float64, arange, array, ndarray, generic, issubdtype, character, concatenate
from pandas import Index
from geopandas import GeoDataFrame, points_from_xy
from pyproj import Proj
from copy import deepcopy
from netCDF4 import date2num
from .default_nes import Nes


class PointsNes(Nes):
    """

    Attributes
    ----------
    _var_dim : tuple
        A Tuple with the name of the Y and X dimensions for the variables.
        ("lat", "lon", ) for a points grid.
    _lat_dim : tuple
        A Tuple with the name of the dimensions of the Latitude values.
        ("lat", ) for a points grid.
    _lon_dim : tuple
        A Tuple with the name of the dimensions of the Longitude values.
        ("lon", ) for a points grid.
    _station : tuple
        A Tuple with the name of the dimensions of the station values.
        ("station", ) for a points grid.
    """

    def __init__(self, comm=None, path=None, info=False, dataset=None, parallel_method="X",
                 avoid_first_hours=0, avoid_last_hours=0, first_level=0, last_level=None, create_nes=False,
                 balanced=False, times=None, **kwargs):
        """
        Initialize the PointsNes class.

        Parameters
        ----------
        comm: MPI.Comm
            MPI Communicator.
        path: str
            Path to the NetCDF to initialize the object.
        info: bool
            Indicates if you want to get reading/writing info.
        dataset: Dataset or None
            NetCDF4-python Dataset to initialize the class.
        parallel_method : str
            Indicates the parallelization method that you want. Default: "X".
            accepted values: ["X", "T"].
        strlen: int
            Maximum length of strings in NetCDF. Default: 75.
        avoid_first_hours : int
            Number of hours to remove from first time steps.
        avoid_last_hours : int
            Number of hours to remove from last time steps.
        first_level : int
            Index of the first level to use.
        last_level : int, None
            Index of the last level to use. None if it is the last.
        create_nes : bool
            Indicates if you want to create the object from scratch (True) or through an existing file.
        balanced : bool
            Indicates if you want a balanced parallelization or not.
            Balanced dataset cannot be written in chunking mode.
        times : list, None
            List of times to substitute the current ones while creation.
        """

        super(PointsNes, self).__init__(comm=comm, path=path, info=info, dataset=dataset,
                                        parallel_method=parallel_method,
                                        avoid_first_hours=avoid_first_hours, avoid_last_hours=avoid_last_hours,
                                        first_level=first_level, last_level=last_level, create_nes=create_nes,
                                        times=times, balanced=balanced, **kwargs)

        if create_nes:
            # Dimensions screening
            self.lat = self._get_coordinate_values(self.get_full_latitudes(), "X")
            self.lon = self._get_coordinate_values(self.get_full_longitudes(), "X")

        # Complete dimensions
        self._station = {"data": arange(len(self.get_full_longitudes()["data"]))}

        # Dimensions screening
        self.station = self._get_coordinate_values(self._station, "X")

        # Set axis limits for parallel writing
        self.write_axis_limits = self._get_write_axis_limits()

        self._var_dim = ("station",)
        self._lat_dim = ("station",)
        self._lon_dim = ("station",)

    @staticmethod
    def new(comm=None, path=None, info=False, dataset=None, parallel_method="X",
            avoid_first_hours=0, avoid_last_hours=0, first_level=0, last_level=None,
            create_nes=False, balanced=False, times=None, **kwargs):
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
        parallel_method : str
            Indicates the parallelization method that you want. Default: "X".
            accepted values: ["X", "T"].
        avoid_first_hours : int
            Number of hours to remove from first time steps.
        avoid_last_hours : int
            Number of hours to remove from last time steps.
        first_level : int
            Index of the first level to use.
        last_level : int, None
            Index of the last level to use. None if it is the last.
        create_nes : bool
            Indicates if you want to create the object from scratch (True) or through an existing file.
        balanced : bool
            Indicates if you want a balanced parallelization or not.
            Balanced dataset cannot be written in chunking mode.
        times : list, None
            List of times to substitute the current ones while creation.
        """

        new = PointsNes(comm=comm, path=path, info=info, dataset=dataset,
                        parallel_method=parallel_method, avoid_first_hours=avoid_first_hours,
                        avoid_last_hours=avoid_last_hours, first_level=first_level, last_level=last_level,
                        create_nes=create_nes, balanced=balanced, times=times, **kwargs)

        return new

    def expand(self, n_cells=1):
        raise RuntimeError("It is not possible to expand a Point Dataset")

    def contract(self, n_cells=1):
        raise RuntimeError("It is not possible to contract a Point Dataset")

    @staticmethod
    def _get_pyproj_projection():
        """
        Get projection data as in Pyproj library.

        Returns
        ----------
        projection : pyproj.Proj
            Grid projection.
        """

        projection = Proj(proj="latlong", ellps="WGS84",)

        return projection

    def _get_projection_data(self, create_nes, **kwargs):
        """
        Retrieves projection data based on grid details.

        Parameters
        ----------
        create_nes : bool
            Flag indicating whether to create new object (True) or use existing (False).
        **kwargs : dict
            Additional keyword arguments for specifying projection details.
        """

        return None

    def _create_dimensions(self, netcdf):
        """
        Create "time", "time_nv", "station" and "strlen" dimensions.

        Parameters
        ----------
        netcdf : Dataset
            NetCDF object.
        """

        # Create time dimension
        netcdf.createDimension("time", None)

        # Create time_nv (number of vertices) dimension
        if self.time_bnds is not None:
            netcdf.createDimension("time_nv", 2)

        # Create station dimension
        # The number of longitudes is equal to the number of stations
        netcdf.createDimension("station", len(self.get_full_longitudes()["data"]))

        # Create string length dimension
        if self.strlen is not None:
            netcdf.createDimension("strlen", self.strlen)

        return None

    # noinspection DuplicatedCode
    def _create_dimension_variables(self, netcdf):
        """
        Create the "time", "time_bnds", "station", "lat", "lat_bnds", "lon" and "lon_bnds" variables.

        Parameters
        ----------
        netcdf : Dataset
            NetCDF object.
        """

        # TIMES
        time_var = netcdf.createVariable("time", float64, ("time",), zlib=self.zip_lvl > 0, complevel=self.zip_lvl)
        time_var.units = "hours since {0}".format(
            self.get_full_times()[self._get_time_id(self.hours_start, first=True)].strftime("%Y-%m-%d %H:%M:%S"))
        time_var.standard_name = "time"
        time_var.calendar = "standard"
        time_var.long_name = "time"
        if self.time_bnds is not None:
            time_var.bounds = "time_bnds"
        if self.size > 1:
            time_var.set_collective(True)
        time_var[:] = date2num(self.get_full_times()[self._get_time_id(self.hours_start, first=True):
                                                     self._get_time_id(self.hours_end, first=False)],
                               time_var.units, time_var.calendar)

        # TIME BOUNDS
        if self.time_bnds is not None:
            time_bnds_var = netcdf.createVariable("time_bnds", float64, ("time", "time_nv",), zlib=self.zip_lvl,
                                                  complevel=self.zip_lvl)
            if self.size > 1:
                time_bnds_var.set_collective(True)
            time_bnds_var[:] = date2num(self.get_full_time_bnds(), time_var.units, calendar="standard")

        # STATIONS
        stations = netcdf.createVariable("station", float64, ("station",), zlib=self.zip_lvl > 0,
                                         complevel=self.zip_lvl)
        stations.units = ""
        stations.axis = "X"
        stations.long_name = ""
        stations.standard_name = "station"
        if self.size > 1:
            stations.set_collective(True)
        stations[:] = self._station["data"]

        # LATITUDES
        lat = netcdf.createVariable("lat", float64, self._lat_dim, zlib=self.zip_lvl > 0, complevel=self.zip_lvl)
        lat.units = "degrees_north"
        lat.axis = "Y"
        lat.long_name = "latitude coordinate"
        lat.standard_name = "latitude"
        if self.lat_bnds is not None:
            lat.bounds = "lat_bnds"
        if self.size > 1:
            lat.set_collective(True)
        lat[:] = self.get_full_latitudes()["data"]

        # LONGITUDES
        lon = netcdf.createVariable("lon", float64, self._lon_dim, zlib=self.zip_lvl > 0, complevel=self.zip_lvl)
        lon.units = "degrees_east"
        lon.axis = "X"
        lon.long_name = "longitude coordinate"
        lon.standard_name = "longitude"
        if self.lon_bnds is not None:
            lon.bounds = "lon_bnds"
        if self.size > 1:
            lon.set_collective(True)
        lon[:] = self.get_full_longitudes()["data"]

        return None

    # noinspection DuplicatedCode
    def _get_coordinate_values(self, coordinate_info, coordinate_axis, bounds=False):
        """
        Get the coordinate data of the current portion.

        Parameters
        ----------
        coordinate_info : dict, list
            Dictionary with the "data" key with the coordinate variable values. and the attributes as other keys.
        coordinate_axis : str
            Name of the coordinate to extract. Accepted values: ["X"].
        bounds : bool
            Boolean variable to know if there are coordinate bounds.
        Returns
        -------
        values : dict
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

        if coordinate_axis == "X":
            if coordinate_len == 1:
                values["data"] = values["data"][self.read_axis_limits["x_min"]:self.read_axis_limits["x_max"]]
            elif coordinate_len == 2:
                values["data"] = values["data"][self.read_axis_limits["t_min"]:self.read_axis_limits["t_max"],
                                                self.read_axis_limits["x_min"]:self.read_axis_limits["x_max"]]
            else:
                raise NotImplementedError("The coordinate has wrong dimensions: {dim}".format(
                    dim=values["data"].shape))

        return values

    def _read_variable(self, var_name):
        """
        Read the corresponding variable data according to the current rank.

        Parameters
        ----------
        var_name : str
            Name of the variable to read.

        Returns
        -------
        data: array
            Portion of the variable data corresponding to the rank.
        """

        nc_var = self.dataset.variables[var_name]
        var_dims = nc_var.dimensions

        # Read data in 1 or 2 dimensions
        if len(var_dims) < 2:
            data = nc_var[self.read_axis_limits["x_min"]:self.read_axis_limits["x_max"]]
        elif len(var_dims) == 2:
            if "strlen" in var_dims:
                data = nc_var[self.read_axis_limits["x_min"]:self.read_axis_limits["x_max"], :]
                data = array(["".join(i.tobytes().decode("ascii").replace("\x00", "")) for i in data], dtype=object)
            else:
                data = nc_var[self.read_axis_limits["t_min"]:self.read_axis_limits["t_max"],
                              self.read_axis_limits["x_min"]:self.read_axis_limits["x_max"]]
        else:
            raise NotImplementedError("Error with {0}. Only can be read netCDF with 2 dimensions or less".format(
                var_name))

        # Unmask array
        data = self._unmask_array(data)

        return data

    # noinspection DuplicatedCode
    def _create_variables(self, netcdf, chunking=False):
        """
        Create the netCDF file variables.

        Parameters
        ----------
        netcdf : Dataset
            netcdf4-python open Dataset.
        chunking : bool
            Indicates if you want to chunk the output netCDF.
        """

        if self.variables is not None:
            for i, (var_name, var_dict) in enumerate(self.variables.items()):
                # Get data type
                if "dtype" in var_dict.keys():
                    var_dtype = var_dict["dtype"]
                    if (var_dict["data"] is not None) and (var_dtype != var_dict["data"].dtype):
                        msg = "WARNING!!! "
                        msg += "Different data types for variable {0}. ".format(var_name)
                        msg += "Input dtype={0}. Data dtype={1}.".format(var_dtype, var_dict["data"].dtype)
                        warn(msg)
                        sys.stderr.flush()
                        try:
                            var_dict["data"] = var_dict["data"].astype(var_dtype)
                        except Exception:  # TODO: Detect exception
                            raise TypeError("It was not possible to cast the data to the input dtype.")
                else:
                    var_dtype = var_dict["data"].dtype
                    if var_dtype is object:
                        raise TypeError("Data dtype is object. Define dtype explicitly as dictionary key 'dtype'")

                # Get dimensions when reading datasets
                if "dimensions" in var_dict.keys():
                    var_dims = var_dict["dimensions"]
                # Get dimensions when creating new datasets
                else:
                    if len(var_dict["data"].shape) == 1:
                        # For data that depends only on station (e.g. station_code)
                        var_dims = self._var_dim
                    else:
                        # For data that is dependent on time and station (e.g. PM10)
                        var_dims = ("time",) + self._var_dim

                if var_dict["data"] is not None:

                    # Ensure data is of type numpy array (to create NES)
                    if not isinstance(var_dict["data"], (ndarray, generic)):
                        try:
                            var_dict["data"] = array(var_dict["data"])
                        except AttributeError:
                            raise AttributeError("Data for variable {0} must be a numpy array.".format(var_name))

                    # Convert list of strings to chars for parallelization
                    if issubdtype(var_dtype, character):
                        var_dict["data_aux"] = self._str2char(var_dict["data"])
                        var_dims += ("strlen",)
                        var_dtype = "S1"

                if self.info:
                    print("Rank {0:03d}: Writing {1} var ({2}/{3})".format(self.rank, var_name, i + 1,
                                                                           len(self.variables)))
                if not chunking:
                    var = netcdf.createVariable(var_name, var_dtype, var_dims,
                                                zlib=self.zip_lvl > 0, complevel=self.zip_lvl)
                else:
                    if self.balanced:
                        raise NotImplementedError("A balanced data cannot be chunked.")
                    if self.master:
                        chunk_size = var_dict["data"].shape
                    else:
                        chunk_size = None
                    chunk_size = self.comm.bcast(chunk_size, root=0)
                    var = netcdf.createVariable(var_name, var_dtype, var_dims,
                                                zlib=self.zip_lvl > 0, complevel=self.zip_lvl,
                                                chunksizes=chunk_size)

                if self.info:
                    print("Rank {0:03d}: Var {1} created ({2}/{3})".format(
                        self.rank, var_name, i + 1, len(self.variables)))
                if self.size > 1:
                    var.set_collective(True)
                    if self.info:
                        print("Rank {0:03d}: Var {1} collective ({2}/{3})".format(
                            self.rank, var_name, i + 1, len(self.variables)))

                for att_name, att_value in var_dict.items():
                    if att_name == "data":
                        if self.info:
                            print("Rank {0:03d}: Filling {1})".format(self.rank, var_name))
                        if "data_aux" in var_dict.keys():
                            att_value = var_dict["data_aux"]
                        if len(att_value.shape) == 1:
                            try:
                                var[self.write_axis_limits["x_min"]:self.write_axis_limits["x_max"]] = att_value
                            except IndexError:
                                raise IndexError("Different shapes. out_shape={0}, data_shp={1}".format(
                                    var[self.write_axis_limits["x_min"]:self.write_axis_limits["x_max"]].shape,
                                    att_value.shape))
                            except ValueError:
                                raise ValueError("Axis limits cannot be accessed. out_shape={0}, data_shp={1}".format(
                                    var[self.write_axis_limits["x_min"]:self.write_axis_limits["x_max"]].shape,
                                    att_value.shape))
                        elif len(att_value.shape) == 2:
                            if "strlen" in var_dims:
                                try:
                                    var[self.write_axis_limits["x_min"]:self.write_axis_limits["x_max"], :] = att_value
                                except IndexError:
                                    raise IndexError("Different shapes. out_shape={0}, data_shp={1}".format(
                                        var[self.write_axis_limits["x_min"]:self.write_axis_limits["x_max"], :].shape,
                                        att_value.shape))
                                except ValueError:
                                    raise ValueError(
                                        "Axis limits cannot be accessed. out_shape={0}, data_shp={1}".format(
                                            var[self.write_axis_limits["x_min"]:
                                                self.write_axis_limits["x_max"]].shape,
                                            att_value.shape))
                            else:
                                try:
                                    var[self.write_axis_limits["t_min"]:self.write_axis_limits["t_max"],
                                        self.write_axis_limits["x_min"]:self.write_axis_limits["x_max"]] = att_value
                                except IndexError:
                                    out_shape = var[
                                        self.write_axis_limits["t_min"]:self.write_axis_limits["t_max"],
                                        self.write_axis_limits["x_min"]:self.write_axis_limits["x_max"]].shape
                                    raise IndexError("Different shapes. out_shape={0}, data_shp={1}".format(
                                        out_shape, att_value.shape))
                                except ValueError:
                                    out_shape = var[
                                        self.write_axis_limits["t_min"]:self.write_axis_limits["t_max"],
                                        self.write_axis_limits["x_min"]:self.write_axis_limits["x_max"]].shape
                                    raise ValueError(
                                        f"Axis limits cannot be accessed. out_shape={out_shape}, "
                                        f"data_shp={att_value.shape}")

                        if self.info:
                            print("Rank {0:03d}: Var {1} data ({2}/{3})".format(self.rank, var_name, i + 1,
                                                                                len(self.variables)))
                    elif att_name not in ["chunk_size", "var_dims", "dimensions", "dtype", "data_aux"]:
                        var.setncattr(att_name, att_value)

                if "data_aux" in var_dict.keys():
                    del var_dict["data_aux"]

                self._set_var_crs(var)
                if self.info:
                    print("Rank {0:03d}: Var {1} completed ({2}/{3})".format(self.rank, var_name, i + 1,
                                                                             len(self.variables)))

        return None

    # noinspection DuplicatedCode
    def _gather_data(self, data_to_gather):
        """
        Gather all the variable data into the MPI rank 0 to perform a serial write.

        Returns
        -------
        data_to_gather: dict
            Variables to gather.
        """
        data_list = deepcopy(data_to_gather)
        for var_name, var_info in data_list.items():
            try:
                # noinspection PyArgumentList
                data_aux = self.comm.gather(data_list[var_name]["data"], root=0)
                if self.rank == 0:
                    shp_len = len(data_list[var_name]["data"].shape)
                    if self.parallel_method == "X":
                        # concatenate over station
                        if shp_len == 1:
                            # dimensions = (station)
                            axis = 0
                        elif shp_len == 2:
                            if "strlen" in var_info["dimensions"]:
                                # dimensions = (station, strlen)
                                axis = 0
                            else:
                                # dimensions = (time, station)
                                axis = 1
                        else:
                            msg = "The points NetCDF must have "
                            msg += "surface values (without levels)."
                            raise NotImplementedError(msg)
                    elif self.parallel_method == "T":
                        # concatenate over time
                        if shp_len == 1:
                            # dimensions = (station)
                            axis = None
                        elif shp_len == 2:
                            if "strlen" in var_info["dimensions"]:
                                # dimensions = (station, strlen)
                                axis = None
                            else:
                                # dimensions = (time, station)
                                axis = 0
                        else:
                            msg = "The points NetCDF must only have surface values (without levels)."
                            raise NotImplementedError(msg)
                    else:
                        raise NotImplementedError(
                            "Parallel method '{meth}' is not implemented. Use one of these: {accept}".format(
                                meth=self.parallel_method, accept=["X", "T"]))
                    data_list[var_name]["data"] = concatenate(data_aux, axis=axis)
            except Exception as e:
                msg = f"**ERROR** an error has occurred while gathering the '{var_name}' variable.\n"
                print(msg)
                sys.stderr.write(msg)
                print(e)
                sys.stderr.write(str(e))
                sys.stderr.flush()
                self.comm.Abort(1)

        return data_list

    def _create_centre_coordinates(self, **kwargs):
        """
        Calculate centre latitudes and longitudes from points.

        Parameters
        ----------
        netcdf : Dataset
            NetCDF object.
        """

        # Calculate centre latitudes
        centre_lat = kwargs["lat"]

        # Calculate centre longitudes
        centre_lon = kwargs["lon"]

        return {"data": centre_lat}, {"data": centre_lon}

    def _create_metadata(self, netcdf):
        """
        Create metadata variables

        Parameters
        ----------
        netcdf : Dataset
            NetCDF object.
        """

        return None

    def create_spatial_bounds(self):
        """
        Calculate longitude and latitude bounds and set them.
        """

        raise NotImplementedError("Spatial bounds cannot be created for points datasets.")

    def to_providentia(self, model_centre_lon, model_centre_lat, grid_edge_lon, grid_edge_lat):
        """
        Transform a PointsNes into a PointsNesProvidentia object

        Returns
        ----------
        points_nes_providentia : nes.Nes
            Points Nes Providentia Object
        """

        from .points_nes_providentia import PointsNesProvidentia

        points_nes_providentia = PointsNesProvidentia(comm=self.comm,
                                                      info=self.info,
                                                      balanced=self.balanced,
                                                      parallel_method=self.parallel_method,
                                                      avoid_first_hours=self.hours_start,
                                                      avoid_last_hours=self.hours_end,
                                                      first_level=self.first_level,
                                                      last_level=self.last_level,
                                                      create_nes=True,
                                                      times=self.time,
                                                      model_centre_lon=model_centre_lon,
                                                      model_centre_lat=model_centre_lat,
                                                      grid_edge_lon=grid_edge_lon,
                                                      grid_edge_lat=grid_edge_lat,
                                                      lat=self.lat["data"],
                                                      lon=self.lon["data"]
                                                      )

        # Convert dimensions (time, lev, lat, lon) to (station, time) for interpolated variables and reshape data
        variables = {}
        interpolated_variables = deepcopy(self.variables)
        for var_name, var_info in interpolated_variables.items():
            variables[var_name] = {}
            # ("time", "lev", "lat", "lon") or ("time", "lat", "lon") to ("station", "time")
            if len(var_info["dimensions"]) != len(var_info["data"].shape):
                variables[var_name]["data"] = var_info["data"].T
                variables[var_name]["dimensions"] = ("station", "time")
            else:
                variables[var_name]["data"] = var_info["data"]
                variables[var_name]["dimensions"] = var_info["dimensions"]

        # Set variables
        points_nes_providentia.variables = variables

        return points_nes_providentia

    def to_grib2(self, path, grib_keys, grib_template_path, lat_flip=False, info=False):
        """
        Write output file with grib2 format.

        Parameters
        ----------
        lat_flip : bool
            Indicates if you want to flip the latitude direction.
        path : str
            Path to the output file.
        grib_keys : dict
            Dictionary with the grib2 keys.
        grib_template_path : str
            Path to the grib2 file to use as template.
        info : bool
            Indicates if you want to print extra information during the process.
        """

        raise NotImplementedError("Grib2 format cannot be written with point data.")

    def create_shapefile(self, overwrite=False):
        """
        Create spatial GeoDataFrame (shapefile).

        Returns
        -------
        shapefile : GeoPandasDataFrame
            Shapefile dataframe.
        """

        if self.shapefile is None or overwrite:

            # Create dataframe containing all points
            gdf = self.get_centroids_from_coordinates()
            self.shapefile = gdf

        else:
            gdf = self.shapefile

        return gdf

    def get_centroids_from_coordinates(self):
        """
        Get centroids from geographical coordinates.

        Returns
        -------
        centroids_gdf: GeoPandasDataFrame
            Centroids dataframe.
        """

        # Get centroids from coordinates
        centroids = points_from_xy(self.lon["data"], self.lat["data"])

        # Create dataframe containing all points
        fids = arange(len(self.get_full_longitudes()["data"]))
        fids = fids[self.read_axis_limits["x_min"]:self.read_axis_limits["x_max"]]
        centroids_gdf = GeoDataFrame(index=Index(name="FID", data=fids),
                                     geometry=centroids,
                                     crs="EPSG:4326")

        return centroids_gdf

    def add_variables_to_shapefile(self, var_list, idx_lev=0, idx_time=0):
        """
        Add variables data to shapefile.

        var_list : list, str
            List (or single string) of the variables to be loaded and saved in the shapefile.
        idx_lev : int
            Index of vertical level for which the data will be saved in the shapefile.
        idx_time : int
            Index of time for which the data will be saved in the shapefile.
        """

        if idx_lev != 0:
            msg = "Error: Points dataset has no level (Level: {0}).".format(idx_lev)
            raise ValueError(msg)

        for var_name in var_list:
            # station as dimension
            if len(self.variables[var_name]["dimensions"]) == 1:
                self.shapefile[var_name] = self.variables[var_name]["data"][:].ravel()
            # station and time as dimensions
            else:
                self.shapefile[var_name] = self.variables[var_name]["data"][idx_time, :].ravel()

        return None

    @staticmethod
    def _get_axis_index_(axis):
        if axis == "T":
            value = 0
        elif axis == "X":
            value = 1
        else:
            raise ValueError("Unknown axis: {0}".format(axis))
        return value

    @staticmethod
    def _set_var_crs(var):
        """
        Set the grid_mapping

        Parameters
        ----------
        var : Variable
            netCDF4-python variable object.
        """
        # var.coordinates = "lat lon"

        return None
