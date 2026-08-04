#!/usr/bin/env python
# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

from numpy import linspace, float64
from netCDF4 import Dataset
from .rotated_nes import RotatedNes


class RotatedNestedNes(RotatedNes):

    def __init__(self, comm=None, path=None, info=False, dataset=None, parallel_method="Y",
                 avoid_first_hours=0, avoid_last_hours=0, first_level=0, last_level=None, create_nes=False,
                 balanced=False, times=None, **kwargs):
        """
        Initialize the RotatedNestedNes class.

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
        parallel_method : str
            Indicates the parallelization method that you want. Default: "Y".
            Accepted values: ["X", "Y", "T"].
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

        super(RotatedNestedNes, self).__init__(comm=comm, path=path,
                                               info=info, dataset=dataset, balanced=balanced,
                                               parallel_method=parallel_method,
                                               avoid_first_hours=avoid_first_hours, avoid_last_hours=avoid_last_hours,
                                               first_level=first_level, last_level=last_level, create_nes=create_nes,
                                               times=times, **kwargs)

    @staticmethod
    def _get_parent_attributes(projection_data):
        """
        Get projection attributes from parent grid.

        Parameters
        ----------
        projection_data : dict
            Dictionary with the projection information.

        Returns
        -------
        projection_data : dict
            Dictionary with the projection information, including parameters from the parent grid.
        """

        # Read variables from parent grid
        netcdf = Dataset(projection_data["parent_grid_path"], mode="r")
        rlat = netcdf.variables["rlat"][:]
        rlon = netcdf.variables["rlon"][:]
        rotated_pole = netcdf.variables["rotated_pole"]

        # j_parent_start starts at index 1, so we must subtract 1
        projection_data["inc_rlat"] = (rlat[1] - rlat[0]) / projection_data["parent_ratio"]
        projection_data["1st_rlat"] = rlat[int(projection_data["j_parent_start"]) - 1]

        # i_parent_start starts at index 1, so we must subtract 1
        projection_data["inc_rlon"] = (rlon[1] - rlon[0]) / projection_data["parent_ratio"]
        projection_data["1st_rlon"] = rlon[int(projection_data["i_parent_start"]) - 1]

        projection_data["grid_north_pole_longitude"] = rotated_pole.grid_north_pole_longitude
        projection_data["grid_north_pole_latitude"] = rotated_pole.grid_north_pole_latitude

        netcdf.close()

        return projection_data

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
        if create_nes:
            projection_data = {"grid_mapping_name": "rotated_latitude_longitude",
                               "parent_grid_path": kwargs["parent_grid_path"],
                               "parent_ratio": kwargs["parent_ratio"],
                               "i_parent_start": kwargs["i_parent_start"],
                               "j_parent_start": kwargs["j_parent_start"],
                               "n_rlat": kwargs["n_rlat"],
                               "n_rlon": kwargs["n_rlon"]
                               }
            projection_data = self._get_parent_attributes(projection_data)
        else:
            projection_data = super()._get_projection_data(create_nes, **kwargs)

        return projection_data

    def _create_rotated_coordinates(self):
        """
        Calculate rotated latitudes and longitudes from grid details.

        Returns
        ----------
        _rlat : dict
            Rotated latitudes dictionary with the "data" key for all the values and the rest of the attributes.
        _rlon : dict
            Rotated longitudes dictionary with the "data" key for all the values and the rest of the attributes.
        """

        # Get grid resolution
        inc_rlon = self.projection_data["inc_rlon"]
        inc_rlat = self.projection_data["inc_rlat"]

        # Get number of rotated coordinates
        n_rlat = self.projection_data["n_rlat"]
        n_rlon = self.projection_data["n_rlon"]

        # Get first coordinates
        first_rlat = self.projection_data["1st_rlat"]
        first_rlon = self.projection_data["1st_rlon"]

        # Calculate rotated latitudes
        rlat = linspace(first_rlat, first_rlat + (inc_rlat * (n_rlat - 1)), n_rlat, dtype=float64)

        # Calculate rotated longitudes
        rlon = linspace(first_rlon, first_rlon + (inc_rlon * (n_rlon - 1)), n_rlon, dtype=float64)

        return {"data": rlat}, {"data": rlon}
