#!/usr/bin/env python
# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import nes
from numpy import float32, array, ndarray
from netCDF4 import Dataset
from mpi4py import MPI


# noinspection DuplicatedCode
def to_netcdf_monarch(self, path, chunking=False, keep_open=False):
    """
    Create the NetCDF using netcdf4-python methods.

    Parameters
    ----------
    self : nes.Nes
        Source projection Nes Object.
    path : str
        Path to the output netCDF file.
    chunking: bool
        Indicates if you want to chunk the output netCDF.
    keep_open : bool
        Indicates if you want to keep open the NetCDH to fill the data by time-step.
    """

    self.to_dtype(float32)

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
    self._create_dimensions(netcdf)

    # Create dimension variables
    if self.master:
        self._full_lev["data"] = array(self._full_lev["data"], dtype=float32)
        self._full_lat["data"] = array(self._full_lat["data"], dtype=float32)
        self._full_lat_bnds["data"] = array(self._full_lat_bnds["data"], dtype=float32)
        self._full_lon["data"] = array(self._full_lon["data"], dtype=float32)
        self._full_lon_bnds["data"] = array(self._full_lon_bnds["data"], dtype=float32)

    if isinstance(self, nes.RotatedNes):
        self._full_rlat["data"] = array(self._full_rlat["data"], dtype=float32)
        self._full_rlon["data"] = array(self._full_rlon["data"], dtype=float32)
    if isinstance(self, nes.LCCNes) or isinstance(self, nes.MercatorNes):
        self._full_y["data"] = array(self._full_y["data"], dtype=float32)
        self._full_x["data"] = array(self._full_x["data"], dtype=float32)

    self._create_dimension_variables(netcdf)
    if self.info:
        print("Rank {0:03d}: Dimensions done".format(self.rank))

    # Create cell measures
    if "cell_area" in self.cell_measures.keys():
        self.cell_measures["cell_area"]["data"] = array(self.cell_measures["cell_area"]["data"], dtype=float32)
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


def to_monarch_units(self):
    """
    Change the data values according to the MONARCH conventions.

    Parameters
    ----------
    self : nes.Nes
        A Nes Object.

    Returns
    -------
    dict
        Variable in the MONARCH units.
    """

    for var_name in self.variables.keys():
        if isinstance(self.variables[var_name]["data"], ndarray):
            if self.variables[var_name]["units"] == "mol.s-1.m-2":
                # Kmol to mol
                self.variables[var_name]["data"] = array(self.variables[var_name]["data"] * 1000, dtype=float32)
            elif self.variables[var_name]["units"] == "kg.s-1.m-2":
                # No unit change needed
                self.variables[var_name]["data"] = array(self.variables[var_name]["data"], dtype=float32)

            else:
                raise TypeError("The unit '{0}' of specie {1} is not defined correctly. ".format(
                    self.variables[var_name]["units"], var_name) +
                                "Should be 'mol.s-1.m-2' or 'kg.s-1.m-2'")
        self.variables[var_name]["dtype"] = float32
    return self.variables
