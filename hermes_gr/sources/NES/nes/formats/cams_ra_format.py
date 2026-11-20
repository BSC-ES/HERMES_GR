#!/usr/bin/env python
# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import sys
import nes
from numpy import float64, float32, int32, array
from warnings import warn
from netCDF4 import Dataset
from mpi4py import MPI
from copy import copy


# noinspection DuplicatedCode
def to_netcdf_cams_ra(self, path):
    """
    Horizontal methods from one grid to another one.

    Parameters
    ----------
    self : nes.Nes
        Source projection Nes Object.
    path : str
        Path to the output netCDF file.
    """

    if not isinstance(self, nes.LatLonNes):
        raise TypeError("CAMS Re-Analysis format must have Regular Lat-Lon projection")
    if "<level>" not in path:
        raise ValueError(f"AMS Re-Analysis path must contain '<level>' as pattern; current: '{path}'")

    orig_path = copy(path)

    for i_lev, level in enumerate(self.lev["data"]):
        path = orig_path.replace("<level>", "l{0}".format(i_lev))
        # Open NetCDF
        if self.info:
            print("Rank {0:03d}: Creating {1}".format(self.rank, path))
        if self.size > 1:
            netcdf = Dataset(path, format="NETCDF4", mode="w", parallel=True, comm=self.comm, info=MPI.Info())
        else:
            netcdf = Dataset(path, format="NETCDF4", mode="w", parallel=False)
        if self.info:
            print("Rank {0:03d}: NetCDF ready to write".format(self.rank))
        self.to_dtype(data_type=float32)

        # Create dimensions
        create_dimensions(self, netcdf)

        # Create variables
        create_variables(self, netcdf, i_lev)

        # Create dimension variables
        create_dimension_variables(self, netcdf)
        if self.info:
            print("Rank {0:03d}: Dimensions done".format(self.rank))

        # Close NetCDF
        if self.global_attrs is not None:
            for att_name, att_value in self.global_attrs.items():
                netcdf.setncattr(att_name, att_value)

        netcdf.close()

    return None


def create_dimensions(self, netcdf):
    """
    Create "time", "time_bnds", "lev", "lon" and "lat" dimensions.

    Parameters
    ----------
    self : nes.Nes
        Source projection Nes Object.
    netcdf : Dataset
        netcdf4-python open dataset.
    """

    # Create time dimension
    netcdf.createDimension("time", None)

    # Create lev, lon and lat dimensions
    netcdf.createDimension("lat", len(self.get_full_latitudes()["data"]))
    netcdf.createDimension("lon", len(self.get_full_longitudes()["data"]))

    return None


def create_dimension_variables(self, netcdf):
    """
    Create the "time", "time_bnds", "lev", "lat", "lat_bnds", "lon" and "lon_bnds" variables.

    Parameters
    ----------
    self : nes.Nes
        Source projection Nes Object.
    netcdf : Dataset
        netcdf4-python open dataset.
    """

    # LATITUDES
    lat = netcdf.createVariable("lat", float64, ("lat",))
    lat.standard_name = "latitude"
    lat.long_name = "latitude"
    lat.units = "degrees_north"
    lat.axis = "Y"

    if self.size > 1:
        lat.set_collective(True)
    lat[:] = self.get_full_latitudes()["data"]

    # LONGITUDES
    lon = netcdf.createVariable("lon", float64, ("lon",))
    lon.long_name = "longitude"
    lon.standard_name = "longitude"
    lon.units = "degrees_east"
    lon.axis = "X"
    if self.size > 1:
        lon.set_collective(True)
    lon[:] = self.get_full_longitudes()["data"]

    # TIMES
    time_var = netcdf.createVariable("time", float64, ("time",))
    time_var.standard_name = "time"
    time_var.units = "day as %Y%m%d.%f"
    time_var.calendar = "proleptic_gregorian"
    time_var.axis = "T"
    if self.size > 1:
        time_var.set_collective(True)
    time_var[:] = __date2num(self.get_full_times()[self._get_time_id(self.hours_start, first=True):
                                                   self._get_time_id(self.hours_end, first=False)])

    return None


# noinspection DuplicatedCode
def create_variables(self, netcdf, i_lev):
    """
    Create and write variables to a netCDF file.

    Parameters
    ----------
    self : nes.Nes
        Source projection Nes Object.
    netcdf : Dataset
        netcdf4-python open dataset.
    i_lev : int
        The specific level index to write data for.
    """

    for i, (var_name, var_dict) in enumerate(self.variables.items()):
        if var_dict["data"] is not None:
            if self.info:
                print("Rank {0:03d}: Writing {1} var ({2}/{3})".format(self.rank, var_name, i + 1, len(self.variables)))
            try:
                var = netcdf.createVariable(var_name, float32, ("time", "lat", "lon",),
                                            zlib=True, complevel=7, least_significant_digit=3)

                if self.info:
                    print("Rank {0:03d}: Var {1} created ({2}/{3})".format(
                        self.rank, var_name, i + 1, len(self.variables)))
                if self.size > 1:
                    var.set_collective(True)
                    if self.info:
                        print("Rank {0:03d}: Var {1} collective ({2}/{3})".format(
                            self.rank, var_name, i + 1, len(self.variables)))

                if self.info:
                    print("Rank {0:03d}: Filling {1})".format(self.rank, var_name))
                var[self.write_axis_limits["t_min"]:self.write_axis_limits["t_max"],
                    self.write_axis_limits["y_min"]:self.write_axis_limits["y_max"],
                    self.write_axis_limits["x_min"]:self.write_axis_limits["x_max"]] = var_dict["data"][:, i_lev, :, :]

                if self.info:
                    print("Rank {0:03d}: Var {1} data ({2}/{3})".format(
                        self.rank, var_name, i + 1, len(self.variables)))
                var.long_name = var_dict["long_name"]
                var.units = var_dict["units"]
                var.number_of_significant_digits = int32(3)

                if self.info:
                    print("Rank {0:03d}: Var {1} completed ({2}/{3})".format(self.rank, var_name, i + 1,
                                                                             len(self.variables)))
            except Exception as e:
                print(f"**ERROR** an error has occurred while writing the '{var_name}' variable")
                raise e
        else:
            msg = "WARNING!!! "
            msg += "Variable {0} was not loaded. It will not be written.".format(var_name)
            warn(msg)
            sys.stderr.flush()

    return None


def __date2num(time_array):
    """
    Convert an array of datetime objects to numerical values.

    Parameters
    ----------
    time_array : List[datetime.datetime]
        List of datetime objects to be converted.

    Returns
    -------
    numpy.ndarray
        Array of numerical time values, with each date represented as a float.

    Notes
    -----
    The conversion represents each datetime as a float in the format YYYYMMDD.HH/24.
    """

    time_res = []
    for aux_time in time_array:
        time_res.append(float(aux_time.strftime("%Y%m%d")) + (float(aux_time.strftime("%H")) / 24))
    time_res = array(time_res, dtype=float64)

    return time_res
