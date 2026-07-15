#!/usr/bin/env python
# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import sys
from warnings import warn
from copy import deepcopy
from numpy import ndarray, generic, array, issubdtype, character, concatenate
from .points_nes import PointsNes


class PointsNesProvidentia(PointsNes):
    """

    Attributes
    ----------
    _model_centre_lon : dict
        Model centre longitudes dictionary with the complete "data" key for all the values and the rest of the
        attributes.
    _model_centre_lat : dict
        Model centre latitudes dictionary with the complete "data" key for all the values and the rest of the
        attributes.
    _grid_edge_lon : dict
        Grid edge longitudes dictionary with the complete "data" key for all the values and the rest of the
        attributes.
    _grid_edge_lat : dict
        Grid edge latitudes dictionary with the complete "data" key for all the values and the rest of the
        attributes.
    model_centre_lon : dict
        Model centre longitudes dictionary with the portion of "data" corresponding to the rank values.
    model_centre_lat : dict
        Model centre latitudes dictionary with the portion of "data" corresponding to the rank values.
    grid_edge_lon : dict
        Grid edge longitudes dictionary with the portion of "data" corresponding to the rank values.
    grid_edge_lat : dict
        Grid edge latitudes dictionary with the portion of "data" corresponding to the rank values.
    """
    def __init__(self, comm=None, path=None, info=False, dataset=None, parallel_method="X",
                 avoid_first_hours=0, avoid_last_hours=0, first_level=0, last_level=None, create_nes=False,
                 balanced=False, times=None, model_centre_lon=None, model_centre_lat=None, grid_edge_lon=None,
                 grid_edge_lat=None,
                 **kwargs):
        """
        Initialize the PointsNesProvidentia class

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
            Accepted values: ["X"].
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
        model_centre_lon : dict
            Model centre longitudes dictionary with the portion of "data" corresponding to the rank values.
        model_centre_lat : dict
            Model centre latitudes dictionary with the portion of "data" corresponding to the rank values.
        grid_edge_lon : dict
            Grid edge longitudes dictionary with the portion of "data" corresponding to the rank values.
        grid_edge_lat : dict
            Grid edge latitudes dictionary with the portion of "data" corresponding to the rank values.
        """

        super(PointsNesProvidentia, self).__init__(comm=comm, path=path, info=info, dataset=dataset,
                                                   parallel_method=parallel_method,
                                                   avoid_first_hours=avoid_first_hours,
                                                   avoid_last_hours=avoid_last_hours,
                                                   first_level=first_level, last_level=last_level,
                                                   create_nes=create_nes, times=times, balanced=balanced, **kwargs)

        if create_nes:
            # Complete dimensions
            self._model_centre_lon = model_centre_lon
            self._model_centre_lat = model_centre_lat
            self._grid_edge_lon = grid_edge_lon
            self._grid_edge_lat = grid_edge_lat
        else:
            # Complete dimensions
            self._model_centre_lon = self._get_coordinate_dimension(["model_centre_longitude"])
            self._model_centre_lat = self._get_coordinate_dimension(["model_centre_latitude"])
            self._grid_edge_lon = self._get_coordinate_dimension(["grid_edge_longitude"])
            self._grid_edge_lat = self._get_coordinate_dimension(["grid_edge_latitude"])

        # Dimensions screening
        self.model_centre_lon = self._get_coordinate_values(self._model_centre_lon, "")
        self.model_centre_lat = self._get_coordinate_values(self._model_centre_lat, "")
        self.grid_edge_lon = self._get_coordinate_values(self._grid_edge_lon, "")
        self.grid_edge_lat = self._get_coordinate_values(self._grid_edge_lat, "")

    @staticmethod
    def new(comm=None, path=None, info=False, dataset=None, parallel_method="X",
            avoid_first_hours=0, avoid_last_hours=0, first_level=0, last_level=None,
            create_nes=False, balanced=False, times=None,
            model_centre_lon=None, model_centre_lat=None, grid_edge_lon=None, grid_edge_lat=None,
            **kwargs):
        """
        Initialize the PointsNesProvidentia class.

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
            Accepted values: ["X"].
        avoid_first_hours : int
            Number of hours to remove from first time steps.
        avoid_last_hours : int
            Number of hours to remove from last time steps.
        first_level : int
            Index of the first level to use
        last_level : int, None
            Index of the last level to use. None if it is the last.
        balanced : bool
            Indicates if you want a balanced parallelization or not.
            Balanced dataset cannot be written in chunking mode.
        times : list, None
            List of times to substitute the current ones while creation.
        create_nes : bool
            Indicates if you want to create the object from scratch (True) or through an existing file.
        model_centre_lon : dict
            Model centre longitudes dictionary with the portion of "data" corresponding to the rank values.
        model_centre_lat : dict
            Model centre latitudes dictionary with the portion of "data" corresponding to the rank values.
        grid_edge_lon : dict
            Grid edge longitudes dictionary with the portion of "data" corresponding to the rank values.
        grid_edge_lat : dict
            Grid edge latitudes dictionary with the portion of "data" corresponding to the rank values.
        """

        new = PointsNesProvidentia(comm=comm, path=path, info=info, dataset=dataset,
                                   parallel_method=parallel_method, avoid_first_hours=avoid_first_hours,
                                   avoid_last_hours=avoid_last_hours, first_level=first_level, last_level=last_level,
                                   create_nes=create_nes, balanced=balanced, times=times,
                                   model_centre_lon=model_centre_lon, model_centre_lat=model_centre_lat,
                                   grid_edge_lon=grid_edge_lon, grid_edge_lat=grid_edge_lat, **kwargs)

        return new

    def _create_dimensions(self, netcdf):
        """
        Create "grid_edge", "model_latitude" and "model_longitude" dimensions and the super dimensions
        "time", "time_nv", "station", and "strlen".

        Parameters
        ----------
        netcdf : Dataset
            NetCDF object.
        """

        super(PointsNesProvidentia, self)._create_dimensions(netcdf)

        # Create grid_edge, model_latitude and model_longitude dimensions
        netcdf.createDimension("grid_edge", len(self._grid_edge_lon["data"]))
        netcdf.createDimension("model_latitude", self._model_centre_lon["data"].shape[0])
        netcdf.createDimension("model_longitude", self._model_centre_lon["data"].shape[1])

        return None

    def _create_dimension_variables(self, netcdf):
        """
        Create the "model_centre_lon", model_centre_lat", "grid_edge_lon" and "grid_edge_lat" variables.

        Parameters
        ----------
        netcdf : Dataset
            NetCDF object.
        """

        super(PointsNesProvidentia, self)._create_dimension_variables(netcdf)

        # MODEL CENTRE LONGITUDES
        model_centre_lon = netcdf.createVariable("model_centre_longitude", "f8",
                                                 ("model_latitude", "model_longitude",),
                                                 zlib=self.zip_lvl > 0, complevel=self.zip_lvl)
        model_centre_lon.units = "degrees_east"
        model_centre_lon.axis = "X"
        model_centre_lon.long_name = "model centre longitude"
        model_centre_lon.standard_name = "model centre longitude"
        if self.size > 1:
            model_centre_lon.set_collective(True)
        msg = "2D meshed grid centre longitudes with "
        msg += "{} longitudes in {} bands of latitude".format(self._model_centre_lon["data"].shape[1],
                                                              self._model_centre_lat["data"].shape[0])
        model_centre_lon.description = msg
        model_centre_lon[:] = self._model_centre_lon["data"]

        # MODEL CENTRE LATITUDES
        model_centre_lat = netcdf.createVariable("model_centre_latitude", "f8",
                                                 ("model_latitude", "model_longitude",),
                                                 zlib=self.zip_lvl > 0, complevel=self.zip_lvl)
        model_centre_lat.units = "degrees_north"
        model_centre_lat.axis = "Y"
        model_centre_lat.long_name = "model centre latitude"
        model_centre_lat.standard_name = "model centre latitude"
        if self.size > 1:
            model_centre_lat.set_collective(True)
        msg = "2D meshed grid centre longitudes with "
        msg += "{} longitudes in {} bands of latitude".format(self._model_centre_lon["data"].shape[1],
                                                              self._model_centre_lat["data"].shape[0])
        model_centre_lat[:] = self._model_centre_lat["data"]

        # GRID EDGE DOMAIN LONGITUDES
        grid_edge_lon = netcdf.createVariable("grid_edge_longitude", "f8", "grid_edge")
        grid_edge_lon.units = "degrees_east"
        grid_edge_lon.axis = "X"
        grid_edge_lon.long_name = "grid edge longitude"
        grid_edge_lon.standard_name = "grid edge longitude"
        if self.size > 1:
            grid_edge_lon.set_collective(True)
        msg = "Longitude coordinate along edge of grid domain "
        msg += "(going clockwise around grid boundary from bottom-left corner)."
        grid_edge_lon.description = msg
        grid_edge_lon[:] = self._grid_edge_lon["data"]

        # GRID EDGE DOMAIN LATITUDES
        grid_edge_lat = netcdf.createVariable("grid_edge_latitude", "f8", "grid_edge")
        grid_edge_lat.units = "degrees_north"
        grid_edge_lat.axis = "Y"
        grid_edge_lat.long_name = "grid edge latitude"
        grid_edge_lat.standard_name = "grid edge latitude"
        if self.size > 1:
            grid_edge_lat.set_collective(True)
        msg = "Latitude coordinate along edge of grid domain "
        msg += "(going clockwise around grid boundary from bottom-left corner)."
        grid_edge_lat.description = msg
        grid_edge_lat[:] = self._grid_edge_lat["data"]

        self.free_vars(["model_centre_longitude", "model_centre_latitude", "grid_edge_longitude", "grid_edge_latitude"])

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
                values["data"] = values["data"][self.read_axis_limits["x_min"]:self.read_axis_limits["x_max"],
                                                self.read_axis_limits["t_min"]:self.read_axis_limits["t_max"]]
            elif coordinate_len == 3:
                values["data"] = values["data"][self.read_axis_limits["x_min"]:self.read_axis_limits["x_max"],
                                                self.read_axis_limits["t_min"]:self.read_axis_limits["t_max"], :]
            else:
                raise NotImplementedError("The coordinate has wrong dimensions: {dim}".format(
                    dim=values["data"].shape))
        elif coordinate_axis == "":
            # pass for "model_centre_lon", "model_centre_lat", "grid_edge_lon" and "grid_edge_lat"
            pass

        return values

    # noinspection DuplicatedCode
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

        # Read data in 1, 2 or 3 dimensions
        if len(var_dims) < 2:
            data = nc_var[self.read_axis_limits["x_min"]:self.read_axis_limits["x_max"]]
        elif len(var_dims) == 2:
            data = nc_var[self.read_axis_limits["x_min"]:self.read_axis_limits["x_max"],
                          self.read_axis_limits["t_min"]:self.read_axis_limits["t_max"]]
        elif len(var_dims) == 3:
            data = nc_var[self.read_axis_limits["x_min"]:self.read_axis_limits["x_max"],
                          self.read_axis_limits["t_min"]:self.read_axis_limits["t_max"],
                          :]
        else:
            raise NotImplementedError("Error with {0}. Only can be read netCDF with 3 dimensions or less".format(
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
                        msg += "Input dtype={0}. Data dtype={1}.".format(
                            var_dtype, var_dict["data"].dtype)
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
                        var_dims = self._var_dim + ("time",)

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
                    if self.master:
                        chunk_size = var_dict["data"].shape
                    else:
                        chunk_size = None
                    chunk_size = self.comm.bcast(chunk_size, root=0)
                    var = netcdf.createVariable(var_name, var_dtype, var_dims, zlib=self.zip_lvl > 0,
                                                complevel=self.zip_lvl, chunksizes=chunk_size)

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
                                            var[self.write_axis_limits["x_min"]:self.write_axis_limits["x_max"],
                                                :].shape,
                                            att_value.shape))
                            else:
                                try:
                                    var[self.write_axis_limits["x_min"]:self.write_axis_limits["x_max"],
                                        self.write_axis_limits["t_min"]:self.write_axis_limits["t_max"]] = att_value
                                except IndexError:
                                    raise IndexError("Different shapes. out_shape={0}, data_shp={1}".format(
                                        var[self.write_axis_limits["x_min"]:self.write_axis_limits["x_max"],
                                            self.write_axis_limits["t_min"]:self.write_axis_limits["t_max"]].shape,
                                        att_value.shape))
                                except ValueError:
                                    raise ValueError(
                                        "Axis limits cannot be accessed. out_shape={0}, data_shp={1}".format(
                                            var[self.write_axis_limits["x_min"]:self.write_axis_limits["x_max"],
                                                self.write_axis_limits["t_min"]:self.write_axis_limits["t_max"]].shape,
                                            att_value.shape))
                        elif len(att_value.shape) == 3:
                            try:
                                var[self.write_axis_limits["x_min"]:self.write_axis_limits["x_max"],
                                    self.write_axis_limits["t_min"]:self.write_axis_limits["t_max"],
                                    :] = att_value
                            except IndexError:
                                raise IndexError("Different shapes. out_shape={0}, data_shp={1}".format(
                                    var[self.write_axis_limits["x_min"]:self.write_axis_limits["x_max"],
                                        self.write_axis_limits["t_min"]:self.write_axis_limits["t_max"],
                                        :].shape,
                                    att_value.shape))
                            except ValueError:
                                raise ValueError("Axis limits cannot be accessed. out_shape={0}, data_shp={1}".format(
                                    var[self.write_axis_limits["x_min"]:self.write_axis_limits["x_max"],
                                        self.write_axis_limits["t_min"]:self.write_axis_limits["t_max"],
                                        :].shape,
                                    att_value.shape))

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
                    # concatenate over station
                    if self.parallel_method == "X":
                        if shp_len == 1:
                            # dimensions = (station)
                            axis = 0
                        elif shp_len == 2:
                            # dimensions = (station, strlen) or
                            # dimensions = (station, time)
                            axis = 0
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
                                # dimensions = (station, time)
                                axis = 1
                        else:
                            msg = "The points NetCDF must have "
                            msg += "surface values (without levels)."
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
                # print(e, file=sys.stderr)
                sys.stderr.flush()
                self.comm.Abort(1)

        return data_list

    def to_netcdf(self, path, compression_level=0, serial=False, info=False, chunking=False, nc_type="NES",
                  keep_open=False):
        """
        Write the netCDF output file.

        Parameters
        ----------
        path : str
            Path to the output netCDF file.
        compression_level : int
            Level of compression (0 to 9) Default: 0 (no compression).
        serial : bool
            Indicates if you want to write in serial or not. Default: False.
        info : bool
            Indicates if you want to print the information of each writing step by stdout Default: False.
        chunking : bool
            Indicates if you want a chunked netCDF output. Only available with non-serial writes. Default: False.
        nc_type : str
            Type to NetCDf to write. "CAMS_RA" or "NES"
        keep_open : bool
            Indicates if you want to keep open the NetCDH to fill the data by time-step
        """

        if (not serial) and (self.size > 1):
            msg = "WARNING!!! "
            msg += "Providentia datasets cannot be written in parallel yet. "
            msg += "Changing to serial mode."
            warn(msg)
            sys.stderr.flush()

        super(PointsNesProvidentia, self).to_netcdf(path, compression_level=compression_level,
                                                    serial=True, info=info, chunking=chunking)

        return None

    # noinspection DuplicatedCode
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
                self.shapefile[var_name] = self.variables[var_name]["data"][:, idx_time].ravel()

        return None

    @staticmethod
    def _get_axis_index_(axis):
        if axis == "T":
            value = 1
        elif axis == "X":
            value = 0
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
        return None
