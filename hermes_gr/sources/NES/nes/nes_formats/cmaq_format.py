#!/usr/bin/env python
# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import nes
from numpy import float32, array, ndarray, empty, int32, float64
from netCDF4 import Dataset
from mpi4py import MPI
from copy import deepcopy
from datetime import datetime

GLOBAL_ATTRIBUTES_ORDER = [
    "IOAPI_VERSION", "EXEC_ID", "FTYPE", "CDATE", "CTIME", "WDATE", "WTIME", "SDATE", "STIME", "TSTEP",  "NTHIK",
    "NCOLS", "NROWS", "NLAYS", "NVARS", "GDTYP", "P_ALP", "P_BET", "P_GAM", "XCENT", "YCENT",  "XORIG", "YORIG",
    "XCELL", "YCELL", "VGTYP", "VGTOP", "VGLVLS", "GDNAM", "UPNAM", "FILEDESC", "HISTORY", "VAR-LIST"]


# noinspection DuplicatedCode
def to_netcdf_cmaq(self, path, keep_open=False):
    """
    Create the NetCDF using netcdf4-python methods.

    Parameters
    ----------
    self : nes.Nes
        Source projection Nes Object.
    path : str
        Path to the output netCDF file.
    keep_open : bool
        Indicates if you want to keep open the NetCDH to fill the data by time-step.
    """

    self.to_dtype(float32)

    set_global_attributes(self)
    change_variable_attributes(self)

    # Open NetCDF
    if self.info:
        print("Rank {0:03d}: Creating {1}".format(self.rank, path))
    if self.size > 1:
        netcdf = Dataset(path, format="NETCDF4", mode="w", parallel=True, comm=self.comm, info=MPI.Info())
    else:
        netcdf = Dataset(path, format="NETCDF4", mode="w", parallel=False)
    if self.info:
        print("Rank {0:03d}: NetCDF ready to write".format(self.rank))

    # Create dimensions
    create_dimensions(self, netcdf)

    create_dimension_variables(self, netcdf)
    if self.info:
        print("Rank {0:03d}: Dimensions done".format(self.rank))

    # Create variables
    create_variables(self, netcdf)

    for att_name in GLOBAL_ATTRIBUTES_ORDER:
        netcdf.setncattr(att_name, self.global_attrs[att_name])

    # Close NetCDF
    if keep_open:
        self.dataset = netcdf
    else:
        netcdf.close()

    return None


def change_variable_attributes(self):
    """
    Modify the emission list to be consistent to use the output as input for CMAQ model.

    Parameters
    ----------
    self : nes.Nes
        A Nes Object.
    """

    for var_name in self.variables.keys():

        if self.variables[var_name]["units"] == "mol.s-1":
            self.variables[var_name]["units"] = "{:<16}".format("mole/s")
            self.variables[var_name]["var_desc"] = "{:<80}".format(self.variables[var_name]["long_name"])
            self.variables[var_name]["long_name"] = "{:<16}".format(var_name)
        elif self.variables[var_name]["units"] == "g.s-1":
            self.variables[var_name]["units"] = "{:<16}".format("g/s")
            self.variables[var_name]["var_desc"] = "{:<80}".format(self.variables[var_name]["long_name"])
            self.variables[var_name]["long_name"] = "{:<16}".format(var_name)

        else:
            raise TypeError("The unit '{0}' of specie {1} is not defined correctly. ".format(
                self.variables[var_name]["units"], var_name) + "Should be 'mol.s-1' or 'g.s-1'")

    return None


def to_cmaq_units(self):
    """
    Change the data values according to the CMAQ conventions

    Parameters
    ----------
    self : nes.Nes
        A Nes Object.

    Returns
    -------
    dict
        Variable in the MONARCH units.
    """

    self.calculate_grid_area(overwrite=False)
    for var_name in self.variables.keys():
        if isinstance(self.variables[var_name]["data"], ndarray):
            if self.variables[var_name]["units"] == "mol.s-1":
                # Kmol.m-2.s-1 to mol.s-1
                self.variables[var_name]["data"] = array(
                    self.variables[var_name]["data"] * 1000 * self.cell_measures["cell_area"]["data"], dtype=float32)
            elif self.variables[var_name]["units"] == "g.s-1":
                # Kg.m-2.s-1 to g.s-1
                self.variables[var_name]["data"] = array(
                    self.variables[var_name]["data"] * 1000 * self.cell_measures["cell_area"]["data"], dtype=float32)

            else:
                raise TypeError("The unit '{0}' of specie {1} is not defined correctly. ".format(
                    self.variables[var_name]["units"], var_name) + "Should be 'mol.s-1' or 'g.s-1'")
        self.variables[var_name]["dtype"] = float32

    return self.variables


def create_tflag(self):
    """
    Create the content of the CMAQ variable TFLAG.

    Parameters
    ----------
    self : nes.Nes
        A Nes Object.

    Returns
    -------
    numpy.ndarray
        Array with the content of TFLAG.
    """

    t_flag = empty((len(self.time), len(self.variables), 2))

    for i_d, aux_date in enumerate(self.time):
        y_d = int(aux_date.strftime("%Y%j"))
        hms = int(aux_date.strftime("%H%M%S"))
        for i_p in range(len(self.variables)):
            t_flag[i_d, i_p, 0] = y_d
            t_flag[i_d, i_p, 1] = hms

    return t_flag


def str_var_list(self):
    """
    Transform the list of variable names to a string with the elements with 16 white spaces.

    Parameters
    ----------
    self : nes.Nes
        A Nes Object.

    Returns
    -------
    str
        List of variable names transformed on string.
    """

    str_var_list_aux = ""
    for var in self.variables.keys():
        str_var_list_aux += "{:<16}".format(var)

    return str_var_list_aux


# noinspection DuplicatedCode
def set_global_attributes(self):
    """
    Set the NetCDF global attributes.

    Parameters
    ----------
    self : nes.Nes
        A Nes Object.
    """

    now = datetime.now()
    if len(self.time) > 1:
        tstep = ((self.time[1] - self.time[0]).seconds // 3600) * 10000
    else:
        tstep = 1 * 10000

    current_attributes = deepcopy(self.global_attrs)
    del self.global_attrs

    self.global_attrs = {"IOAPI_VERSION": "None: made only with NetCDF libraries",
                         "EXEC_ID": "{:<80}".format("0.1alpha"),  # Editable
                         "FTYPE": int32(1),  # Editable
                         "CDATE": int32(now.strftime("%Y%j")),
                         "CTIME": int32(now.strftime("%H%M%S")),
                         "WDATE": int32(now.strftime("%Y%j")),
                         "WTIME": int32(now.strftime("%H%M%S")),
                         "SDATE": int32(self.time[0].strftime("%Y%j")),
                         "STIME": int32(self.time[0].strftime("%H%M%S")),
                         "TSTEP": int32(tstep),
                         "NTHIK": int32(1),  # Editable
                         "NCOLS": None,  # Projection dependent
                         "NROWS": None,  # Projection dependent
                         "NLAYS": int32(len(self.lev["data"])),
                         "NVARS": None,  # Projection dependent
                         "GDTYP": None,  # Projection dependent
                         "P_ALP": None,  # Projection dependent
                         "P_BET": None,  # Projection dependent
                         "P_GAM": None,  # Projection dependent
                         "XCENT": None,  # Projection dependent
                         "YCENT": None,  # Projection dependent
                         "XORIG": None,  # Projection dependent
                         "YORIG": None,  # Projection dependent
                         "XCELL": None,  # Projection dependent
                         "YCELL": None,  # Projection dependent
                         "VGTYP": int32(7),  # Editable
                         "VGTOP": float32(5000.),  # Editable
                         "VGLVLS": array([1., 0.], dtype=float32),  # Editable
                         "GDNAM": "{:<16}".format(""),  # Editable
                         "UPNAM": "{:<16}".format("HERMESv3"),
                         "FILEDESC": "",  # Editable
                         "HISTORY": "",  # Editable
                         "VAR-LIST": str_var_list(self)}

    # Editable attributes
    for att_name, att_value in current_attributes.items():
        if att_name == "EXEC_ID":
            self.global_attrs[att_name] = "{:<80}".format(att_value)  # Editable
        elif att_name == "FTYPE":
            self.global_attrs[att_name] = int32(att_value)  # Editable
        elif att_name == "NTHIK":
            self.global_attrs[att_name] = int32(att_value)  # Editable
        elif att_name == "VGTYP":
            self.global_attrs[att_name] = int32(att_value)  # Editable
        elif att_name == "VGTOP":
            self.global_attrs[att_name] = float32(att_value)  # Editable
        elif att_name == "VGLVLS":
            self.global_attrs[att_name] = array(att_value.split(), dtype=float32)  # Editable
        elif att_name == "GDNAM":
            self.global_attrs[att_name] = "{:<16}".format(att_value)  # Editable
        elif att_name == "FILEDESC":
            self.global_attrs[att_name] = att_value  # Editable
        elif att_name == "HISTORY":
            self.global_attrs[att_name] = att_value  # Editable

    # Projection dependent attributes
    if isinstance(self, nes.LCCNes):
        self.global_attrs["NCOLS"] = int32(len(self._full_x["data"]))
        self.global_attrs["NROWS"] = int32(len(self._full_y["data"]))
        self.global_attrs["NVARS"] = int32(len(self.variables))
        self.global_attrs["GDTYP"] = int32(2)

        self.global_attrs["P_ALP"] = float64(self.projection_data["standard_parallel"][0])
        self.global_attrs["P_BET"] = float64(self.projection_data["standard_parallel"][1])
        self.global_attrs["P_GAM"] = float64(self.projection_data["longitude_of_central_meridian"])
        self.global_attrs["XCENT"] = float64(self.projection_data["longitude_of_central_meridian"])
        self.global_attrs["YCENT"] = float64(self.projection_data["latitude_of_projection_origin"])
        self.global_attrs["XORIG"] = float64(
            self._full_x["data"][0]) - (float64(self._full_x["data"][1] - self._full_x["data"][0]) / 2)
        self.global_attrs["YORIG"] = float64(
            self._full_y["data"][0]) - (float64(self._full_y["data"][1] - self._full_y["data"][0]) / 2)
        self.global_attrs["XCELL"] = float64(self._full_x["data"][1] - self._full_x["data"][0])
        self.global_attrs["YCELL"] = float64(self._full_y["data"][1] - self._full_y["data"][0])

    return None


def create_dimensions(self, netcdf):
    """
    Create "time", "time_bnds", "lev", "lon" and "lat" dimensions.

    Parameters
    ----------
    self : nes.Nes
        Nes Object.
    netcdf : Dataset
        netcdf4-python open dataset.
    """

    netcdf.createDimension("TSTEP", len(self.get_full_times()))
    netcdf.createDimension("DATE-TIME", 2)
    netcdf.createDimension("LAY", len(self.get_full_levels()["data"]))
    netcdf.createDimension("VAR", len(self.variables))
    if isinstance(self, nes.LCCNes):
        netcdf.createDimension("COL", len(self._full_x["data"]))
        netcdf.createDimension("ROW", len(self._full_y["data"]))

    return None


def create_dimension_variables(self, netcdf):
    """
    Create the "y" and "x" variables.

    Parameters
    ----------
    self : nes.Nes
        A Nes Object.
    netcdf : Dataset
        NetCDF object.
    """

    tflag = netcdf.createVariable("TFLAG", "i", ("TSTEP", "VAR", "DATE-TIME",))
    tflag.setncatts({"units": "{:<16}".format("<YYYYDDD,HHMMSS>"), "long_name": "{:<16}".format("TFLAG"),
                     "var_desc": "{:<80}".format("Timestep-valid flags:  (1) YYYYDDD or (2) HHMMSS")})
    tflag[:] = create_tflag(self)

    return None


# noinspection DuplicatedCode
def create_variables(self, netcdf):
    """
    Create the netCDF file variables.

    Parameters
    ----------
    self : nes.Nes
        Nes Object.
    netcdf : Dataset
        netcdf4-python open dataset.
    """

    for var_name, var_info in self.variables.items():
        var = netcdf.createVariable(var_name, "f", ("TSTEP", "LAY", "ROW", "COL",),
                                    zlib=self.zip_lvl > 0, complevel=self.zip_lvl)
        var.units = var_info["units"]
        var.long_name = str(var_info["long_name"])
        var.var_desc = str(var_info["var_desc"])
        if var_info["data"] is not None:
            if self.info:
                print("Rank {0:03d}: Filling {1})".format(self.rank, var_name))

            if isinstance(var_info["data"], int) and var_info["data"] == 0:
                var[self.write_axis_limits["t_min"]:self.write_axis_limits["t_max"],
                    self.write_axis_limits["z_min"]:self.write_axis_limits["z_max"],
                    self.write_axis_limits["y_min"]:self.write_axis_limits["y_max"],
                    self.write_axis_limits["x_min"]:self.write_axis_limits["x_max"]] = 0

            elif len(var_info["data"].shape) == 4:
                var[self.write_axis_limits["t_min"]:self.write_axis_limits["t_max"],
                    self.write_axis_limits["z_min"]:self.write_axis_limits["z_max"],
                    self.write_axis_limits["y_min"]:self.write_axis_limits["y_max"],
                    self.write_axis_limits["x_min"]:self.write_axis_limits["x_max"]] = var_info["data"]

    return None
