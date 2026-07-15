# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

from copy import deepcopy
from typing import Tuple

import netCDF4 as nc4
import numpy as np
from mpi4py import MPI

CHIMERE_GLOBAL_ATTRS = {
    "Title": "",
    "Sub-title": "",
    "Generating_process": "",
    "Conventions": "",
    "Domain": "",
    "history": "",
}
CONSTANTS = {"SpStrLen": 23, "DateStrlen": 19}
AVOGADROS_NUMBER = 6.02214076e23


def to_netcdf_chimere(self, path: str, keep_open=False) -> None:
    """Standardizes an input NetCDF into the requirements from the CHIMERE suite format

    Parameters
    ----------
    self : nes.Nes
        Source projection Nes Object.
    path : str
        Path to the output netCDF file.
    keep_open : bool
        Indicates if you want to keep open the NetCDH to fill the data by time-step.
    """

    self.to_dtype(np.float32)

    print("Change variable attributes", flush=True)
    change_variables_attributes(self)

    print("Opening netcdf", flush=True)
    if self.info:
        print("Rank {0:03d}: Creating {1}".format(self.rank, path))
    if self.size > 1:
        netcdf = nc4.Dataset(
            path,
            format="NETCDF4",
            mode="w",
            parallel=True,
            comm=self.comm,
            info=MPI.Info(),
        )
    else:
        netcdf = nc4.Dataset(path, format="NETCDF4", mode="w", parallel=False)
    if self.info:
        print("Rank {0:03d}: NetCDF ready to write".format(self.rank))

    print("Setting global attributes", flush=True)
    _set_global_attributes(self, netcdf)
    print("Creating dimensions", flush=True)
    lat_arr, lon_arr = _create_dimensions(self, netcdf)
    print("Creating variables", flush=True)
    _create_variables(self, netcdf, lat_arr, lon_arr)
    print("Creating species variables", flush=True)
    _create_species_variables(self, netcdf)

    # Close NetCDF
    if keep_open:
        self.dataset = netcdf
    else:
        netcdf.close()

    return None


def to_chimere_units(self) -> dict:
    """Changes species unit formatting to match the CHIMERE format

    Parameters
    ----------
    self : nes.Nes
        Source projection Nes Object.

    Returns
    ----------
    self.variables : dict
        Dictionary holding the variables of the netcdf

    Raises
    ------
    TypeError
        Error specifying which species had non supported units
    """
    # As input hte emissions are # Kmol.m-2.s-1 to molecule/cm2/s #or kg.m-2.-s1
    self.calculate_grid_area(overwrite=False)
    for var_name in self.variables.keys():
        if isinstance(self.variables[var_name]["data"], np.ndarray):
            if self.variables[var_name]["units"] in [
                "molecules/cm2/s",
                "molecule/cm2/s",
            ]:
                # m-2->cm-2 kmol->mol   mol -> molecules
                self.variables[var_name]["data"] = np.array(
                    self.variables[var_name]["data"]
                    * ((10**-4) * (10**3) * (AVOGADROS_NUMBER)),
                    dtype=np.float32,
                )
            else:
                raise TypeError(
                    "The unit '{0}' of species {1} is not defined correctly. ".format(
                        self.variables[var_name]["units"], var_name
                    )
                    + "Should be 'molecules/cm2/s' or 'molecule/cm2/s'"
                )
        else:
            print(
                f"Something went wrong with {var_name}, data may be empty", flush=True
            )

        self.variables[var_name]["dtype"] = np.float32
    return self.variables


def change_variables_attributes(self) -> None:
    """Change units and long_name (description) variable attributes of each emission species

    Parameters
    ----------
    self : nes.Nes
        Source projection Nes Object.

    Raises
    ------
    TypeError
        Unsupported emission unit error
    """
    for var_name in self.variables.keys():
        if self.variables[var_name]["units"] == "molecule/cm2/s":
            self.variables[var_name]["units"] = "{:<16}".format("molecule/cm2/s")
            self.variables[var_name]["long_name"] = f"{var_name} Emission"

        elif self.variables[var_name]["units"] == "molecules/cm2/s":
            self.variables[var_name]["units"] = "{:<16}".format("molecules/cm2/s")
            self.variables[var_name]["long_name"] = f"{var_name} Emission"
        else:
            raise TypeError(
                "The unit '{0}' of species {1} is not defined correctly. ".format(
                    self.variables[var_name]["units"], var_name
                )
                + "Should be 'molecule/cm2/s' or 'molecule/cm2/s'"
            )
    return None


def _is_nonregular_like(lat_arr: np.ndarray, lon_arr: np.ndarray) -> bool:
    """Checks if NetCDF is rotated

    Parameters
    ----------
    lat_arr : np.ndarray
        Latitude array of source NetCDF
    lon_arr : np.ndarray
        Longitude array of source NetCDF

    Returns
    -------
    bool
        Boolean indicating whether the latitude and longitude data is 2-dimensional, meaning the NetCDF is rotated
    """
    return lat_arr.ndim == 2 and lon_arr.ndim == 2


def _set_global_attributes(self, netcdf: nc4.Dataset) -> None:
    """Sets the global attributes of the NetCDF to match the CHIMERE suite format

    Parameters
    ----------
    self : nes.Nes
        Source projection Nes Object.
    netcdf : Dataset
        netcdf4-python open dataset.
    """
    current_attributes = deepcopy(self.global_attrs)
    del self.global_attrs

    self.global_attributes = CHIMERE_GLOBAL_ATTRS

    for att_name, att_values in current_attributes.items():
        if att_name in CHIMERE_GLOBAL_ATTRS.keys():
            netcdf.setncattr(att_name, att_values)
        else:
            self
    return None


def _create_dimensions(self, netcdf: nc4.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """Creates the Species string length, species, bottom_top (levels), time, date string length,
    south/north and west/east dimensions

    Parameters
    ----------
    self : nes.NES
        Source projection Nes Object.
    netcdf : netcdf4.Dataset
        netcdf4-python open dataset.

    Returns
    -------
    np.ndarray, np.ndarray
        Full latitude and longitude arrays
    """

    lat_arr = np.asanyarray(self.get_full_latitudes()["data"])
    lon_arr = np.asanyarray(self.get_full_longitudes()["data"])

    if _is_nonregular_like(lat_arr, lon_arr):
        # Rotated or otherwise 2D coordinates
        # Trust the shape of the 2D lat array
        sn, we = lat_arr.shape
    else:
        sn, we = lat_arr.size, lon_arr.size

    netcdf.createDimension("Time", None)
    netcdf.createDimension("west_east", we)
    netcdf.createDimension("south_north", sn)
    netcdf.createDimension("bottom_top", len(self.get_full_levels()["data"]))
    netcdf.createDimension("SpStrLen", CONSTANTS["SpStrLen"])
    netcdf.createDimension("DateStrLen", CONSTANTS["DateStrlen"])
    netcdf.createDimension("Species", len(self.variables.keys()))

    return lat_arr, lon_arr


def _create_variables(
    self, netcdf: nc4.Dataset, lat_arr: np.ndarray, lon_arr: np.ndarray
) -> None:
    """Creates the latitude, longitude, times and species variables

    Parameters
    ----------
    self : nes.Nes
        Source projection Nes Object.
    netcdf : netcdf4.Dataset
        netcdf4-python open dataset.
    lat_array : np.ndarray
        Full latitude array
    lon_array : np.ndarray
        Full longitude array
    """

    time_var = netcdf.createVariable("Times", "S1", ("Time", "DateStrLen"))
    if self.master:
        time_var[:] = np.array(
            [
                list(s.ljust(CONSTANTS["DateStrlen"]))
                for s in [t.strftime("%Y-%m-%d_%H:%M:%S") for t in self.time]
            ],
            dtype="S1",
        )

    # Species
    netcdf.createVariable("species", "S1", ("Species", "SpStrLen"))

    # Check if rotated or not
    if _is_nonregular_like(lat_arr, lon_arr):
        lat_2d, lon_2d = lat_arr, lon_arr
    else:
        lon_2d, lat_2d = np.meshgrid(lon_arr, lat_arr)

    # Longitude
    lon = netcdf.createVariable("lon", "f4", ("south_north", "west_east"))
    lon.units = "degrees_east"
    lon.long_name = "Longitude"
    if self.master:
        lon[:] = lon_2d.astype("f4", copy=False)

    # Latitude
    lat = netcdf.createVariable("lat", "f4", ("south_north", "west_east"))
    lat.units = "degrees_north"
    lat.long_name = "Latitude"
    if self.master:
        lat[:] = lat_2d.astype("f4", copy=False)

    return None


def _create_species_variables(self, netcdf: nc4.Dataset) -> None:
    """Creates species' variables for each pollutant in the original file, and writes them to match the
    CHIMERE suite format

    Parameters
    ----------
    self : nes.Nes
        Source projection Nes Object.
    netcdf : netcd4.Dataset
        netcdf4-python open dataset.

    Raises
    ------
    e
        Error informing the user that something went wrong during writing of one of the variables
    """
    for var_name, var_info in self.variables.items():
        var = netcdf.createVariable(
            var_name, "f4", ("Time", "bottom_top", "south_north", "west_east")
        )
        var.units = var_info["units"]
        var.long_name = var_info["long_name"]

        if var_info["data"] is not None:
            if isinstance(var_info["data"], int) and var_info["data"] == 0:
                var[
                    self.write_axis_limits["t_min"]:self.write_axis_limits["t_max"],
                    self.write_axis_limits["z_min"]:self.write_axis_limits["z_max"],
                    self.write_axis_limits["y_min"]:self.write_axis_limits["y_max"],
                    self.write_axis_limits["x_min"]:self.write_axis_limits["x_max"],
                ] = 0

            elif len(var_info["data"].shape) == 4:
                var[
                    self.write_axis_limits["t_min"]:self.write_axis_limits["t_max"],
                    self.write_axis_limits["z_min"]:self.write_axis_limits["z_max"],
                    self.write_axis_limits["y_min"]:self.write_axis_limits["y_max"],
                    self.write_axis_limits["x_min"]:self.write_axis_limits["x_max"],
                ] = var_info["data"]

    return None
