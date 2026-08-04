#!/usr/bin/env python
# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import sys
from warnings import warn
from numpy import float64, empty, ndarray, generic, array, issubdtype, character, concatenate, int64
from netCDF4 import date2num
from copy import deepcopy
from .points_nes import PointsNes


class PointsNesGHOST(PointsNes):
    """

    Attributes
    ----------
    _qa : dict
        Quality flags (GHOST checks) dictionary with the complete "data" key for all the values and the rest of the
        attributes.
    _flag : dict
        Data flags (given by data provider) dictionary with the complete "data" key for all the values and the rest of
        the attributes.
    _qa : dict
        Quality flags (GHOST checks) dictionary with the portion of "data" corresponding to the rank values.
    _flag : dict
        Data flags (given by data provider) dictionary with the portion of "data" corresponding to the rank values.
    """

    def __init__(self, comm=None, path=None, info=False, dataset=None, parallel_method="X",
                 avoid_first_hours=0, avoid_last_hours=0, first_level=0, last_level=None, create_nes=False,
                 balanced=False, times=None, **kwargs):
        """
        Initialize the PointsNesGHOST class.

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
        """

        super(PointsNesGHOST, self).__init__(comm=comm, path=path, info=info, dataset=dataset,
                                             parallel_method=parallel_method,
                                             avoid_first_hours=avoid_first_hours, avoid_last_hours=avoid_last_hours,
                                             first_level=first_level, last_level=last_level, create_nes=create_nes,
                                             times=times, balanced=balanced, **kwargs)

        # Complete dimensions
        self._flag = self._get_coordinate_dimension(["flag"])
        self._qa = self._get_coordinate_dimension(["qa"])

        # Dimensions screening
        self.flag = self._get_coordinate_values(self._flag, "X")
        self.qa = self._get_coordinate_values(self._qa, "X")

    @staticmethod
    def new(comm=None, path=None, info=False, dataset=None, parallel_method="X",
            avoid_first_hours=0, avoid_last_hours=0, first_level=0, last_level=None, create_nes=False,
            balanced=False, times=None, **kwargs):
        """
        Initialize the PointsNesGHOST class.

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
        """

        new = PointsNesGHOST(comm=comm, path=path, info=info, dataset=dataset,
                             parallel_method=parallel_method, avoid_first_hours=avoid_first_hours,
                             avoid_last_hours=avoid_last_hours, first_level=first_level, last_level=last_level,
                             create_nes=create_nes, balanced=balanced, times=times, **kwargs)

        return new

    def _create_dimensions(self, netcdf):
        """
        Create "N_flag_codes" and "N_qa_codes" dimensions and the super dimensions
        "time", "time_nv", "station", and "strlen".

        Parameters
        ----------
        netcdf : Dataset
            NetCDF object.
        """

        super(PointsNesGHOST, self)._create_dimensions(netcdf)

        # Create N_flag_codes and N_qa_codes dimensions
        netcdf.createDimension("N_flag_codes", self._flag["data"].shape[2])
        netcdf.createDimension("N_qa_codes", self._qa["data"].shape[2])

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
        lat = netcdf.createVariable("latitude", float64, self._lat_dim, zlib=self.zip_lvl > 0, complevel=self.zip_lvl)
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
        lon = netcdf.createVariable("longitude", float64, self._lon_dim, zlib=self.zip_lvl > 0, complevel=self.zip_lvl)
        lon.units = "degrees_east"
        lon.axis = "X"
        lon.long_name = "longitude coordinate"
        lon.standard_name = "longitude"
        if self.lon_bnds is not None:
            lon.bounds = "lon_bnds"
        if self.size > 1:
            lon.set_collective(True)
        lon[:] = self.get_full_longitudes()["data"]

    def erase_flags(self):

        first_time_idx = self._get_time_id(self.hours_start, first=True)
        last_time_idx = self._get_time_id(self.hours_end, first=False)
        t_len = last_time_idx - first_time_idx

        self._qa["data"] = empty((len(self.get_full_longitudes()["data"]), t_len, 0))
        self._flag["data"] = empty((len(self.get_full_longitudes()["data"]), t_len, 0))

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
                values["data"] = values["data"][self.read_axis_limits["x_min"]:self.read_axis_limits["x_max"],
                                                self.read_axis_limits["t_min"]:self.read_axis_limits["t_max"]]
            elif coordinate_len == 3:
                values["data"] = values["data"][self.read_axis_limits["x_min"]:self.read_axis_limits["x_max"],
                                                self.read_axis_limits["t_min"]:self.read_axis_limits["t_max"], :]
            else:
                raise NotImplementedError("The coordinate has wrong dimensions: {dim}".format(
                    dim=values["data"].shape))

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

        # Read data in 1 or 2 dimensions
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
                        msg += "Input dtype={0}. Data dtype={1}.".format(var_dtype, var_dict["data"].dtype)
                        warn(msg)
                        sys.stderr.flush()
                        try:
                            var_dict["data"] = var_dict["data"].astype(var_dtype)
                        except Exception:
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
                                    raise IndexError(
                                        "Different shapes. out_shape={0}, data_shp={1}".format(
                                            var[self.write_axis_limits["x_min"]:self.write_axis_limits["x_max"],
                                                :].shape,
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
                                    out_shape = var[
                                        self.write_axis_limits["x_min"]:self.write_axis_limits["x_max"],
                                        self.write_axis_limits["t_min"]:self.write_axis_limits["t_max"]].shape
                                    raise ValueError(
                                        f"Axis limits cannot be accessed. out_shape={out_shape}, "
                                        f"data_shp={att_value.shape}")
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
                sys.stderr.flush()
                self.comm.Abort(1)

        return data_list

    def _create_metadata(self, netcdf):
        """
        Create metadata variables.

        Parameters
        ----------
        netcdf : Dataset
            NetCDF object.
        """

        # N FLAG CODES
        flag = netcdf.createVariable("flag", int64, ("station", "time", "N_flag_codes",),
                                     zlib=self.zip_lvl > 0, complevel=self.zip_lvl)
        flag.units = ""
        flag.axis = ""
        flag.long_name = ""
        flag.standard_name = "flag"
        if self.size > 1:
            flag.set_collective(True)
        flag[:] = self._flag["data"]

        # N QA CODES
        qa = netcdf.createVariable("qa", int64, ("station", "time", "N_qa_codes",),
                                   zlib=self.zip_lvl > 0, complevel=self.zip_lvl)
        qa.units = ""
        qa.axis = ""
        qa.long_name = ""
        qa.standard_name = "N_qa_codes"
        if self.size > 1:
            qa.set_collective(True)
        qa[:] = self._qa["data"]

        return None

    def to_netcdf(self, path, compression_level=0, serial=False, info=False, chunking=False, nc_type="NES",
                  keep_open=False):
        """
        Write the netCDF output file.

        Parameters
        ----------
        keep_open : bool
        nc_type : str
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
        """

        if (not serial) and (self.size > 1):
            msg = "WARNING!!! "
            msg += "GHOST datasets cannot be written in parallel yet. "
            msg += "Changing to serial mode."
            warn(msg)
            sys.stderr.flush()

        super(PointsNesGHOST, self).to_netcdf(
            path, compression_level=compression_level, serial=True, info=info, chunking=chunking)

        return None

    def to_points(self):
        """
        Transform a PointsNesGHOST into a PointsNes object

        Returns
        ----------
        points_nes : nes.Nes
            Points Nes Object (without GHOST metadata variables)
        """

        points_nes = PointsNes(comm=self.comm,
                               info=self.info,
                               balanced=self.balanced,
                               parallel_method=self.parallel_method,
                               avoid_first_hours=self.hours_start,
                               avoid_last_hours=self.hours_end,
                               first_level=self.first_level,
                               last_level=self.last_level,
                               create_nes=True,
                               lat=self.lat["data"],
                               lon=self.lon["data"],
                               times=self.time,
                               )

        # The version attribute in GHOST files prior to 1.3.3 is called data_version, after it is version
        if "version" in self.global_attrs:
            ghost_version = self.global_attrs["version"]
        elif "data_version" in self.global_attrs:
            ghost_version = self.global_attrs["data_version"]
        else:
            ghost_version = "0.0.0"
        metadata_variables = self.get_standard_metadata(ghost_version)
        self.free_vars(metadata_variables)
        self.free_vars("station")
        points_nes.variables = deepcopy(self.variables)

        return points_nes

    @staticmethod
    def get_standard_metadata(ghost_version):
        """
        Get all possible GHOST variables for each version.

        Parameters
        ----------
        ghost_version : str
            Version of GHOST file.

        Returns
        ----------
        metadata_variables[GHOST_version] : list
            A List of metadata variables for a certain GHOST version
        """

        # This metadata variables are
        metadata_variables = {"1.4": ["GHOST_version", "station_reference", "station_timezone", "latitude", "longitude",
                                      "altitude", "sampling_height", "measurement_altitude", "ellipsoid",
                                      "horizontal_datum", "vertical_datum", "projection", "distance_to_building",
                                      "distance_to_kerb", "distance_to_junction", "distance_to_source", "street_width",
                                      "street_type", "daytime_traffic_speed", "daily_passing_vehicles", "data_level",
                                      "climatology", "station_name", "city", "country",
                                      "administrative_country_division_1", "administrative_country_division_2",
                                      "population", "representative_radius", "network", "associated_networks",
                                      "area_classification", "station_classification", "main_emission_source",
                                      "land_use", "terrain", "measurement_scale",
                                      "ESDAC_Iwahashi_landform_classification",
                                      "ESDAC_modal_Iwahashi_landform_classification_5km",
                                      "ESDAC_modal_Iwahashi_landform_classification_25km",
                                      "ESDAC_Meybeck_landform_classification",
                                      "ESDAC_modal_Meybeck_landform_classification_5km",
                                      "ESDAC_modal_Meybeck_landform_classification_25km",
                                      "GHSL_settlement_model_classification",
                                      "GHSL_modal_settlement_model_classification_5km",
                                      "GHSL_modal_settlement_model_classification_25km",
                                      "Joly-Peuch_classification_code", "Koppen-Geiger_classification",
                                      "Koppen-Geiger_modal_classification_5km",
                                      "Koppen-Geiger_modal_classification_25km",
                                      "MODIS_MCD12C1_v6_IGBP_land_use", "MODIS_MCD12C1_v6_modal_IGBP_land_use_5km",
                                      "MODIS_MCD12C1_v6_modal_IGBP_land_use_25km", "MODIS_MCD12C1_v6_UMD_land_use",
                                      "MODIS_MCD12C1_v6_modal_UMD_land_use_5km",
                                      "MODIS_MCD12C1_v6_modal_UMD_land_use_25km", "MODIS_MCD12C1_v6_LAI",
                                      "MODIS_MCD12C1_v6_modal_LAI_5km", "MODIS_MCD12C1_v6_modal_LAI_25km",
                                      "WMO_region", "WWF_TEOW_terrestrial_ecoregion", "WWF_TEOW_biogeographical_realm",
                                      "WWF_TEOW_biome", "UMBC_anthrome_classification",
                                      "UMBC_modal_anthrome_classification_5km",
                                      "UMBC_modal_anthrome_classification_25km",
                                      "EDGAR_v4.3.2_annual_average_BC_emissions",
                                      "EDGAR_v4.3.2_annual_average_CO_emissions",
                                      "EDGAR_v4.3.2_annual_average_NH3_emissions",
                                      "EDGAR_v4.3.2_annual_average_NMVOC_emissions",
                                      "EDGAR_v4.3.2_annual_average_NOx_emissions",
                                      "EDGAR_v4.3.2_annual_average_OC_emissions",
                                      "EDGAR_v4.3.2_annual_average_PM10_emissions",
                                      "EDGAR_v4.3.2_annual_average_biogenic_PM2.5_emissions",
                                      "EDGAR_v4.3.2_annual_average_fossilfuel_PM2.5_emissions",
                                      "EDGAR_v4.3.2_annual_average_SO2_emissions", "ASTER_v3_altitude",
                                      "ETOPO1_altitude", "ETOPO1_max_altitude_difference_5km",
                                      "GHSL_built_up_area_density", "GHSL_average_built_up_area_density_5km",
                                      "GHSL_average_built_up_area_density_25km", "GHSL_max_built_up_area_density_5km",
                                      "GHSL_max_built_up_area_density_25km", "GHSL_population_density",
                                      "GHSL_average_population_density_5km", "GHSL_average_population_density_25km",
                                      "GHSL_max_population_density_5km", "GHSL_max_population_density_25km",
                                      "GPW_population_density", "GPW_average_population_density_5km",
                                      "GPW_average_population_density_25km", "GPW_max_population_density_5km",
                                      "GPW_max_population_density_25km",
                                      "NOAA-DMSP-OLS_v4_nighttime_stable_lights",
                                      "NOAA-DMSP-OLS_v4_average_nighttime_stable_lights_5km",
                                      "NOAA-DMSP-OLS_v4_average_nighttime_stable_lights_25km",
                                      "NOAA-DMSP-OLS_v4_max_nighttime_stable_lights_5km",
                                      "NOAA-DMSP-OLS_v4_max_nighttime_stable_lights_25km",
                                      "OMI_level3_column_annual_average_NO2",
                                      "OMI_level3_column_cloud_screened_annual_average_NO2",
                                      "OMI_level3_tropospheric_column_annual_average_NO2",
                                      "OMI_level3_tropospheric_column_cloud_screened_annual_average_NO2",
                                      "GSFC_coastline_proximity", "primary_sampling_type",
                                      "primary_sampling_instrument_name",
                                      "primary_sampling_instrument_documented_flow_rate",
                                      "primary_sampling_instrument_reported_flow_rate",
                                      "primary_sampling_process_details", "primary_sampling_instrument_manual_name",
                                      "primary_sampling_further_details", "sample_preparation_types",
                                      "sample_preparation_techniques", "sample_preparation_process_details",
                                      "sample_preparation_further_details", "measurement_methodology",
                                      "measuring_instrument_name", "measuring_instrument_sampling_type",
                                      "measuring_instrument_documented_flow_rate",
                                      "measuring_instrument_reported_flow_rate", "measuring_instrument_process_details",
                                      "measuring_instrument_process_details", "measuring_instrument_manual_name",
                                      "measuring_instrument_further_details", "measuring_instrument_reported_units",
                                      "measuring_instrument_reported_lower_limit_of_detection",
                                      "measuring_instrument_documented_lower_limit_of_detection",
                                      "measuring_instrument_reported_upper_limit_of_detection",
                                      "measuring_instrument_documented_upper_limit_of_detection",
                                      "measuring_instrument_reported_uncertainty",
                                      "measuring_instrument_documented_uncertainty",
                                      "measuring_instrument_reported_accuracy",
                                      "measuring_instrument_documented_accuracy",
                                      "measuring_instrument_reported_precision",
                                      "measuring_instrument_documented_precision",
                                      "measuring_instrument_reported_zero_drift",
                                      "measuring_instrument_documented_zero_drift",
                                      "measuring_instrument_reported_span_drift",
                                      "measuring_instrument_documented_span_drift",
                                      "measuring_instrument_reported_zonal_drift",
                                      "measuring_instrument_documented_zonal_drift",
                                      "measuring_instrument_reported_measurement_resolution",
                                      "measuring_instrument_documented_measurement_resolution",
                                      "measuring_instrument_reported_absorption_cross_section",
                                      "measuring_instrument_documented_absorption_cross_section",
                                      "measuring_instrument_inlet_information",
                                      "measuring_instrument_calibration_scale",
                                      "network_provided_volume_standard_temperature",
                                      "network_provided_volume_standard_pressure", "retrieval_algorithm",
                                      "principal_investigator_name", "principal_investigator_institution",
                                      "principal_investigator_email_address", "contact_name",
                                      "contact_institution", "contact_email_address", "meta_update_stamp",
                                      "data_download_stamp", "data_revision_stamp", "network_sampling_details",
                                      "network_uncertainty_details", "network_maintenance_details",
                                      "network_qa_details", "network_miscellaneous_details", "data_licence",
                                      "process_warnings", "temporal_resolution",
                                      "reported_lower_limit_of_detection_per_measurement",
                                      "reported_upper_limit_of_detection_per_measurement",
                                      "reported_uncertainty_per_measurement", "derived_uncertainty_per_measurement",
                                      "day_night_code", "weekday_weekend_code", "season_code",
                                      "hourly_native_representativity_percent", "hourly_native_max_gap_percent",
                                      "daily_native_representativity_percent", "daily_representativity_percent",
                                      "daily_native_max_gap_percent", "daily_max_gap_percent",
                                      "monthly_native_representativity_percent", "monthly_representativity_percent",
                                      "monthly_native_max_gap_percent", "monthly_max_gap_percent",
                                      "annual_native_representativity_percent", "annual_native_max_gap_percent",
                                      "all_representativity_percent", "all_max_gap_percent"],
                              }

        return metadata_variables[ghost_version]

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
