#!/usr/bin/env python
# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import nes
from numpy import float32, int32, ndarray, array, char
from netCDF4 import Dataset
from mpi4py import MPI
from copy import deepcopy

GLOBAL_ATTRIBUTES_ORDER = [
    "TITLE",
    "START_DATE",
    "WEST-EAST_GRID_DIMENSION",
    "SOUTH-NORTH_GRID_DIMENSION",
    "BOTTOM-TOP_GRID_DIMENSION",
    "DX",
    "DY",
    "GRIDTYPE",
    "DIFF_OPT",
    "KM_OPT",
    "DAMP_OPT",
    "DAMPCOEF",
    "KHDIF",
    "KVDIF",
    "MP_PHYSICS",
    "RA_LW_PHYSICS",
    "RA_SW_PHYSICS",
    "SF_SFCLAY_PHYSICS",
    "SF_SURFACE_PHYSICS",
    "BL_PBL_PHYSICS",
    "CU_PHYSICS",
    "SF_LAKE_PHYSICS",
    "SURFACE_INPUT_SOURCE",
    "SST_UPDATE",
    "GRID_FDDA",
    "GFDDA_INTERVAL_M",
    "GFDDA_END_H",
    "GRID_SFDDA",
    "SGFDDA_INTERVAL_M",
    "SGFDDA_END_H",
    "WEST-EAST_PATCH_START_UNSTAG",
    "WEST-EAST_PATCH_END_UNSTAG",
    "WEST-EAST_PATCH_START_STAG",
    "WEST-EAST_PATCH_END_STAG",
    "SOUTH-NORTH_PATCH_START_UNSTAG",
    "SOUTH-NORTH_PATCH_END_UNSTAG",
    "SOUTH-NORTH_PATCH_START_STAG",
    "SOUTH-NORTH_PATCH_END_STAG",
    "BOTTOM-TOP_PATCH_START_UNSTAG",
    "BOTTOM-TOP_PATCH_END_UNSTAG",
    "BOTTOM-TOP_PATCH_START_STAG",
    "BOTTOM-TOP_PATCH_END_STAG",
    "GRID_ID",
    "PARENT_ID",
    "I_PARENT_START",
    "J_PARENT_START",
    "PARENT_GRID_RATIO",
    "DT",
    "CEN_LAT",
    "CEN_LON",
    "TRUELAT1",
    "TRUELAT2",
    "MOAD_CEN_LAT",
    "STAND_LON",
    "POLE_LAT",
    "POLE_LON",
    "GMT",
    "JULYR",
    "JULDAY",
    "MAP_PROJ",
    "MMINLU",
    "NUM_LAND_CAT",
    "ISWATER",
    "ISLAKE",
    "ISICE",
    "ISURBAN",
    "ISOILWATER",
]


# noinspection DuplicatedCode
def to_netcdf_wrf_chem(self, path, keep_open=False):
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
    Modify the emission list to be consistent to use the output as input for WRF-CHEM model.

    Parameters
    ----------
    self : nes.Nes
        A Nes Object.
    """

    for var_name in self.variables.keys():
        if self.variables[var_name]["units"] == "mol.h-1.km-2":
            self.variables[var_name]["FieldType"] = int32(104)
            self.variables[var_name]["MemoryOrder"] = "XYZ"
            self.variables[var_name]["description"] = "EMISSIONS"
            self.variables[var_name]["units"] = "mol km^-2 hr^-1"
            self.variables[var_name]["stagger"] = ""
            self.variables[var_name]["coordinates"] = "XLONG XLAT"

        elif self.variables[var_name]["units"] == "ug.s-1.m-2":
            self.variables[var_name]["FieldType"] = int32(104)
            self.variables[var_name]["MemoryOrder"] = "XYZ"
            self.variables[var_name]["description"] = "EMISSIONS"
            self.variables[var_name]["units"] = "ug/m3 m/s"
            self.variables[var_name]["stagger"] = ""
            self.variables[var_name]["coordinates"] = "XLONG XLAT"

        else:
            raise TypeError(
                "The unit '{0}' of specie {1} is not defined correctly. ".format(
                    self.variables[var_name]["units"], var_name
                )
                + "Should be 'mol.h-1.km-2' or 'ug.s-1.m-2'"
            )

        if "long_name" in self.variables[var_name].keys():
            del self.variables[var_name]["long_name"]

    return None


def to_wrf_chem_units(self):
    """
    Change the data values according to the WRF-CHEM conventions.

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
            if self.variables[var_name]["units"] == "mol.h-1.km-2":
                # 10**6 -> from m2 to km2
                # 10**3 -> from kmol to mol
                # 3600 -> from s to h
                self.variables[var_name]["data"] = array(
                    self.variables[var_name]["data"] * (10**6 * 10**3 * 3600),
                    dtype=float32,
                )
            elif self.variables[var_name]["units"] == "ug.s-1.m-2":
                # 10**9 -> from kg to ug
                self.variables[var_name]["data"] = array(
                    self.variables[var_name]["data"] * (10**9), dtype=float32
                )

            else:
                raise TypeError(
                    "The unit '{0}' of specie {1} is not defined correctly. ".format(
                        self.variables[var_name]["units"], var_name
                    )
                    + "Should be 'mol.h-1.km-2' or 'ug.s-1.m-2'"
                )
        self.variables[var_name]["dtype"] = float32

    return self.variables


def create_times_var(self):
    """
    Create the content of the WRF-CHEM variable times.

    Parameters
    ----------
    self : nes.Nes
        A Nes Object.

    Returns
    -------
    numpy.ndarray
        Array with the content of TFLAG.
    """

    aux_times = char.chararray((len(self.time), 19), itemsize=1)

    for i_d, aux_date in enumerate(self.time):
        aux_times[i_d] = list(aux_date.strftime("%Y-%m-%d_%H:%M:%S"))

    return aux_times


# noinspection DuplicatedCode
def set_global_attributes(self):
    """
    Set the NetCDF global attributes

    Parameters
    ----------
    self : nes.Nes
        A Nes Object.
    """

    # now = datetime.now()
    # if len(self.time) > 1:
    #     tstep = ((self.time[1] - self.time[0]).seconds // 3600) * 10000
    # else:
    #     tstep = 1 * 10000

    current_attributes = deepcopy(self.global_attrs)
    del self.global_attrs

    self.global_attrs = {
        "TITLE": "",
        "START_DATE": self.time[0].strftime("%Y-%m-%d_%H:%M:%S"),
        "WEST-EAST_GRID_DIMENSION": None,  # Projection dependent attributes
        "SOUTH-NORTH_GRID_DIMENSION": None,  # Projection dependent attributes
        "BOTTOM-TOP_GRID_DIMENSION": int32(45),
        "DX": None,  # Projection dependent attributes
        "DY": None,  # Projection dependent attributes
        "GRIDTYPE": "C",
        "DIFF_OPT": int32(1),
        "KM_OPT": int32(4),
        "DAMP_OPT": int32(3),
        "DAMPCOEF": float32(0.2),
        "KHDIF": float32(0.0),
        "KVDIF": float32(0.0),
        "MP_PHYSICS": int32(6),
        "RA_LW_PHYSICS": int32(4),
        "RA_SW_PHYSICS": int32(4),
        "SF_SFCLAY_PHYSICS": int32(2),
        "SF_SURFACE_PHYSICS": int32(2),
        "BL_PBL_PHYSICS": int32(8),
        "CU_PHYSICS": int32(0),
        "SF_LAKE_PHYSICS": int32(0),
        "SURFACE_INPUT_SOURCE": None,  # Projection dependent attributes
        "SST_UPDATE": int32(0),
        "GRID_FDDA": int32(0),
        "GFDDA_INTERVAL_M": int32(0),
        "GFDDA_END_H": int32(0),
        "GRID_SFDDA": int32(0),
        "SGFDDA_INTERVAL_M": int32(0),
        "SGFDDA_END_H": int32(0),
        "WEST-EAST_PATCH_START_UNSTAG": None,  # Projection dependent attributes
        "WEST-EAST_PATCH_END_UNSTAG": None,  # Projection dependent attributes
        "WEST-EAST_PATCH_START_STAG": None,  # Projection dependent attributes
        "WEST-EAST_PATCH_END_STAG": None,  # Projection dependent attributes
        "SOUTH-NORTH_PATCH_START_UNSTAG": None,  # Projection dependent attributes
        "SOUTH-NORTH_PATCH_END_UNSTAG": None,  # Projection dependent attributes
        "SOUTH-NORTH_PATCH_START_STAG": None,  # Projection dependent attributes
        "SOUTH-NORTH_PATCH_END_STAG": None,  # Projection dependent attributes
        "BOTTOM-TOP_PATCH_START_UNSTAG": None,
        "BOTTOM-TOP_PATCH_END_UNSTAG": None,
        "BOTTOM-TOP_PATCH_START_STAG": None,
        "BOTTOM-TOP_PATCH_END_STAG": None,
        "GRID_ID": int32(1),
        "PARENT_ID": int32(0),
        "I_PARENT_START": int32(1),
        "J_PARENT_START": int32(1),
        "PARENT_GRID_RATIO": int32(1),
        "DT": float32(18.0),
        "CEN_LAT": None,  # Projection dependent attributes
        "CEN_LON": None,  # Projection dependent attributes
        "TRUELAT1": None,  # Projection dependent attributes
        "TRUELAT2": None,  # Projection dependent attributes
        "MOAD_CEN_LAT": None,  # Projection dependent attributes
        "STAND_LON": None,  # Projection dependent attributes
        "POLE_LAT": None,  # Projection dependent attributes
        "POLE_LON": None,  # Projection dependent attributes
        "GMT": float32(self.time[0].hour),
        "JULYR": int32(self.time[0].year),
        "JULDAY": int32(self.time[0].strftime("%j")),
        "MAP_PROJ": None,  # Projection dependent attributes
        "MMINLU": "MODIFIED_IGBP_MODIS_NOAH",
        "NUM_LAND_CAT": int32(41),
        "ISWATER": int32(17),
        "ISLAKE": int32(-1),
        "ISICE": int32(15),
        "ISURBAN": int32(13),
        "ISOILWATER": int32(14),
        "HISTORY": "",  # Editable
    }

    # Editable attributes
    float_atts = ["DAMPCOEF", "KHDIF", "KVDIF", "CEN_LAT", "CEN_LON", "DT"]
    int_atts = [
        "BOTTOM-TOP_GRID_DIMENSION",
        "DIFF_OPT",
        "KM_OPT",
        "DAMP_OPT",
        "MP_PHYSICS",
        "RA_LW_PHYSICS",
        "RA_SW_PHYSICS",
        "SF_SFCLAY_PHYSICS",
        "SF_SURFACE_PHYSICS",
        "BL_PBL_PHYSICS",
        "CU_PHYSICS",
        "SF_LAKE_PHYSICS",
        "SURFACE_INPUT_SOURCE",
        "SST_UPDATE",
        "GRID_FDDA",
        "GFDDA_INTERVAL_M",
        "GFDDA_END_H",
        "GRID_SFDDA",
        "SGFDDA_INTERVAL_M",
        "SGFDDA_END_H",
        "BOTTOM-TOP_PATCH_START_UNSTAG",
        "BOTTOM-TOP_PATCH_END_UNSTAG",
        "BOTTOM-TOP_PATCH_START_STAG",
        "BOTTOM-TOP_PATCH_END_STAG",
        "GRID_ID",
        "PARENT_ID",
        "I_PARENT_START",
        "J_PARENT_START",
        "PARENT_GRID_RATIO",
        "NUM_LAND_CAT",
        "ISWATER",
        "ISLAKE",
        "ISICE",
        "ISURBAN",
        "ISOILWATER",
    ]
    str_atts = ["GRIDTYPE", "MMINLU", "HISTORY"]
    for att_name, att_value in current_attributes.items():
        if att_name in int_atts:
            self.global_attrs[att_name] = int32(att_value)
        elif att_name in float_atts:
            self.global_attrs[att_name] = float32(att_value)
        elif att_name in str_atts:
            self.global_attrs[att_name] = str(att_value)

    # Projection dependent attributes
    if isinstance(self, nes.LCCNes) or isinstance(self, nes.MercatorNes):
        self.global_attrs["WEST-EAST_GRID_DIMENSION"] = int32(
            len(self._full_x["data"]) + 1
        )
        self.global_attrs["SOUTH-NORTH_GRID_DIMENSION"] = int32(
            len(self._full_y["data"]) + 1
        )
        self.global_attrs["DX"] = float32(
            self._full_x["data"][1] - self._full_x["data"][0]
        )
        self.global_attrs["DY"] = float32(
            self._full_y["data"][1] - self._full_y["data"][0]
        )
        self.global_attrs["SURFACE_INPUT_SOURCE"] = int32(1)
        self.global_attrs["WEST-EAST_PATCH_START_UNSTAG"] = int32(1)
        self.global_attrs["WEST-EAST_PATCH_END_UNSTAG"] = int32(
            len(self._full_x["data"])
        )
        self.global_attrs["WEST-EAST_PATCH_START_STAG"] = int32(1)
        self.global_attrs["WEST-EAST_PATCH_END_STAG"] = int32(
            len(self._full_x["data"]) + 1
        )
        self.global_attrs["SOUTH-NORTH_PATCH_START_UNSTAG"] = int32(1)
        self.global_attrs["SOUTH-NORTH_PATCH_END_UNSTAG"] = int32(
            len(self._full_y["data"])
        )
        self.global_attrs["SOUTH-NORTH_PATCH_START_STAG"] = int32(1)
        self.global_attrs["SOUTH-NORTH_PATCH_END_STAG"] = int32(
            len(self._full_y["data"]) + 1
        )

        self.global_attrs["POLE_LAT"] = float32(90)
        self.global_attrs["POLE_LON"] = float32(0)

        if isinstance(self, nes.LCCNes):
            self.global_attrs["MAP_PROJ"] = int32(1)
            self.global_attrs["TRUELAT1"] = float32(
                self.projection_data["standard_parallel"][0]
            )
            self.global_attrs["TRUELAT2"] = float32(
                self.projection_data["standard_parallel"][1]
            )
            self.global_attrs["MOAD_CEN_LAT"] = float32(
                self.projection_data["latitude_of_projection_origin"]
            )
            self.global_attrs["STAND_LON"] = float32(
                self.projection_data["longitude_of_central_meridian"]
            )
            self.global_attrs["CEN_LAT"] = float32(
                self.projection_data["latitude_of_projection_origin"]
            )
            self.global_attrs["CEN_LON"] = float32(
                self.projection_data["longitude_of_central_meridian"]
            )
        elif isinstance(self, nes.MercatorNes):
            self.global_attrs["MAP_PROJ"] = int32(3)
            self.global_attrs["TRUELAT1"] = float32(
                self.projection_data["standard_parallel"]
            )
            self.global_attrs["TRUELAT2"] = float32(0)
            self.global_attrs["MOAD_CEN_LAT"] = float32(
                self.projection_data["standard_parallel"]
            )
            self.global_attrs["STAND_LON"] = float32(
                self.projection_data["longitude_of_projection_origin"]
            )
            self.global_attrs["CEN_LAT"] = float32(
                self.projection_data["standard_parallel"]
            )
            self.global_attrs["CEN_LON"] = float32(
                self.projection_data["longitude_of_projection_origin"]
            )

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

    netcdf.createDimension("Time", len(self.get_full_times()))
    netcdf.createDimension("DateStrLen", 19)
    netcdf.createDimension("emissions_zdim", len(self.get_full_levels()["data"]))
    if isinstance(self, nes.LCCNes):
        netcdf.createDimension("west_east", len(self._full_x["data"]))
        netcdf.createDimension("south_north", len(self._full_y["data"]))

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

    times = netcdf.createVariable(
        "Times",
        "S1",
        (
            "Time",
            "DateStrLen",
        ),
    )
    times[:] = create_times_var(self)

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
        var = netcdf.createVariable(
            var_name,
            "f",
            (
                "Time",
                "emissions_zdim",
                "south_north",
                "west_east",
            ),
            zlib=self.zip_lvl > 0,
            complevel=self.zip_lvl,
        )
        var.FieldType = var_info["FieldType"]
        var.MemoryOrder = var_info["MemoryOrder"]
        var.description = var_info["description"]
        var.units = var_info["units"]
        var.stagger = var_info["stagger"]
        var.coordinates = var_info["coordinates"]

        if var_info["data"] is not None:
            if self.info:
                print("Rank {0:03d}: Filling {1})".format(self.rank, var_name))

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
