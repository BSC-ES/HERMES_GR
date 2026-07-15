#!/usr/bin/env python
# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import sys

from numpy import (float64, linspace, cos, sin, arcsin, arctan2, array, mean, diff, append, flip, repeat, concatenate,
                   vstack)
from math import pi
from geopandas import GeoDataFrame
from pandas import Index
from pyproj import Proj
from copy import deepcopy
from typing import Dict, Any
from shapely.geometry import Polygon, Point
from .default_nes import Nes


class RotatedNes(Nes):
    """

    Attributes
    ----------
    _full_rlat : dict
        Rotated latitudes dictionary with the complete "data" key for all the values and the rest of the attributes.
    _full_rlon : dict
        Rotated longitudes dictionary with the complete "data" key for all the values and the rest of the attributes.
    rlat : dict
        Rotated latitudes dictionary with the portion of "data" corresponding to the rank values.
    rlon : dict
        Rotated longitudes dictionary with the portion of "data" corresponding to the rank values.
    _var_dim : tuple
        A Tuple with the name of the Y and X dimensions for the variables.
        ("rlat", "rlon") for a rotated projection.
    _lat_dim : tuple
        A Tuple with the name of the dimensions of the Latitude values.
        ("rlat", "rlon") for a rotated projection.
    _lon_dim : tuple
        A Tuple with the name of the dimensions of the Longitude values.
        ("rlat", "rlon") for a rotated projection.
    """
    def __init__(self, comm=None, path=None, info=False, dataset=None, parallel_method="Y",
                 avoid_first_hours=0, avoid_last_hours=0, first_level=0, last_level=None, create_nes=False,
                 balanced=False, times=None, **kwargs):
        """
        Initialize the RotatedNes class.

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
        self._full_rlat = None
        self._full_rlon = None

        super(RotatedNes, self).__init__(comm=comm, path=path,
                                         info=info, dataset=dataset, balanced=balanced,
                                         parallel_method=parallel_method,
                                         avoid_first_hours=avoid_first_hours, avoid_last_hours=avoid_last_hours,
                                         first_level=first_level, last_level=last_level, create_nes=create_nes,
                                         times=times, **kwargs)

        if create_nes:
            # Complete dimensions
            # self._full_rlat, self._full_rlon = self._create_rotated_coordinates()
            # Dimensions screening
            self.lat = self._get_coordinate_values(self.get_full_latitudes(), "Y")
            self.lon = self._get_coordinate_values(self.get_full_longitudes(), "X")
        else:
            # Complete dimensions
            self._full_rlat = self._get_coordinate_dimension("rlat")
            self._full_rlon = self._get_coordinate_dimension("rlon")

        # Dimensions screening
        self.rlat = self._get_coordinate_values(self.get_full_rlat(), "Y")
        self.rlon = self._get_coordinate_values(self.get_full_rlon(), "X")

        # Set axis limits for parallel writing
        self.write_axis_limits = self._get_write_axis_limits()

        self._var_dim = ("rlat", "rlon")
        self._lat_dim = ("rlat", "rlon")
        self._lon_dim = ("rlat", "rlon")

    @staticmethod
    def new(comm=None, path=None, info=False, dataset=None, parallel_method="Y",
            avoid_first_hours=0, avoid_last_hours=0, first_level=0, last_level=None, create_nes=False,
            balanced=False, times=None, **kwargs):
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
            Indicates the parallelization method that you want. Default over Y axis
            accepted values: ["X", "Y", "T"].
        avoid_first_hours : int
            Number of hours to remove from first time steps.
        avoid_last_hours : int
            Number of hours to remove from last time steps.
        first_level : int
            Index of the first level to use.
        last_level : int or None
            Index of the last level to use. None if it is the last.
        create_nes : bool
            Indicates if you want to create the object from scratch (True) or through an existing file.
        balanced : bool
            Indicates if you want a balanced parallelization or not.
            Balanced dataset cannot be written in chunking mode.
        times : List[datetime] or None
            List of times to substitute the current ones while creation.
        """

        new = RotatedNes(comm=comm, path=path, info=info, dataset=dataset,
                         parallel_method=parallel_method, avoid_first_hours=avoid_first_hours,
                         avoid_last_hours=avoid_last_hours, first_level=first_level, last_level=last_level,
                         create_nes=create_nes, balanced=balanced, times=times, **kwargs)

        return new

    def get_full_rlat(self) -> Dict[str, Any]:
        """
        Retrieve the complete rotated latitude information.

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
        data = self.comm.bcast(self._full_rlat, root=0)

        return data

    def get_full_rlon(self) -> Dict[str, Any]:
        """
        Retrieve the complete rotated longitude information.

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
        data = self.comm.bcast(self._full_rlon, root=0)
        return data

    def set_full_rlat(self, data: Dict[str, Any]) -> None:
        """
        Set the complete rotated latitude information.

        Parameters
        ----------
        data : Dict[str, Any]
            A dictionary containing the complete latitude data and its attributes.
            The dictionary structure is:
            {
                "data": ndarray,  # Array of latitude values.
                attr_name: attr_value,  # Latitude attributes.
                ...
            }
        """
        if self.master:
            self._full_rlat = data
        return None

    def set_full_rlon(self, data: Dict[str, Any]) -> None:
        """
        Set the complete rotated longitude information.

        Parameters
        ----------
        data : Dict[str, Any]
            A dictionary containing the complete longitude data and its attributes.
            The dictionary structure is:
            {
                "data": ndarray,  # Array of longitude values.
                attr_name: attr_value,  # Longitude attributes.
                ...
            }
        """
        if self.master:
            self._full_rlon = data
        return None

    def _expand_coordinates(self, n_cells: int, is_first: bool, is_last: bool) -> None:
        """
        Expands the latitude and longitude coordinates.

        Parameters
        ----------
        n_cells : int
            Number of cells to expand.
        is_first : bool
            Whether the current rank is the first in parallel processing.
        is_last : bool
            Whether the current rank is the last in parallel processing.

        Returns
        -------
        None
        """
        for coord in ['rlat', 'rlon']:
            expand_left = is_first if "lat" in coord else True
            expand_right = is_last if "lat" in coord else True
            self.__dict__[coord]['data'] = self._expand_1d(
                self.__dict__[coord]['data'], n_cells=n_cells, left=expand_left, right=expand_right
            )

        if self.parallel_method == "X":
            expand_top = True
            expand_bottom = True
            expand_left = is_first
            expand_right = is_last
        elif self.parallel_method == "Y":
            expand_top = is_last
            expand_bottom = is_first
            expand_left = True
            expand_right = True
        else:
            expand_top = True
            expand_bottom = True
            expand_left = True
            expand_right = True

        for coord in ['lat', 'lon']:
            self.__dict__[coord]['data'] = self._expand_2d(
                self.__dict__[coord]['data'], n_cells=n_cells, left=expand_left, right=expand_right, top=expand_top,
                bottom=expand_bottom)
        for coord in ['lat_bnds', 'lon_bnds']:
            if self.__dict__[coord] is not None:
                self.__dict__[coord]['data'] = self._expand_2d_bounds(
                    self.__dict__[coord]['data'], n_cells=n_cells, left=expand_left, right=expand_right, top=expand_top,
                    bottom=expand_bottom)
        return None

    def _contract_coordinates(self, n_cells: int, is_first: bool, is_last: bool) -> None:
        """
        Contracts latitude and longitude coordinates by removing `n_cells` from the borders.

        Parameters
        ----------
        n_cells : int
            The number of cells to remove from each border.
        is_first : bool
            Indicates whether the current rank is the first in parallel processing.
        is_last : bool
            Indicates whether the current rank is the last in parallel processing.

        Returns
        -------
        None
        """
        for coord in ['rlat', 'rlon']:
            contract_left = is_first if "lat" in coord else True
            contract_right = is_last if "lat" in coord else True
            self.__dict__[coord]['data'] = self._contract_1d(
                self.__dict__[coord]['data'], n_cells=n_cells, left=contract_left, right=contract_right
            )
        if self.parallel_method == "X":
            expand_top = True
            expand_bottom = True
            expand_left = is_first
            expand_right = is_last
        elif self.parallel_method == "Y":
            expand_top = is_last
            expand_bottom = is_first
            expand_left = True
            expand_right = True
        else:
            expand_top = True
            expand_bottom = True
            expand_left = True
            expand_right = True

        for coord in ['lat', 'lon']:
            self.__dict__[coord]['data'] = self._contract_2d(
                self.__dict__[coord]['data'], n_cells=n_cells, left=expand_left, right=expand_right, top=expand_top,
                bottom=expand_bottom)
        for coord in ['lat_bnds', 'lon_bnds']:
            if self.__dict__[coord] is not None:
                self.__dict__[coord]['data'] = self._contract_2d_bounds(
                    self.__dict__[coord]['data'], n_cells=n_cells, left=expand_left, right=expand_right, top=expand_top,
                    bottom=expand_bottom)
        return None

    def _expand_full_coordinates(self, n_cells: int) -> None:
        """
        Expands the full coordinates if `self.master` is True.

        Parameters
        ----------
        n_cells : int
            Number of cells to expand.

        Returns
        -------
        None
        """
        if self.master:
            for coord in ['_full_rlat', '_full_rlon']:
                self.__dict__[coord]['data'] = self._expand_1d(
                    self.__dict__[coord]['data'], n_cells=n_cells, left=True, right=True
                )
            for coord in ['_full_lat', '_full_lon', ]:
                self.__dict__[coord]['data'] = self._expand_2d(
                    self.__dict__[coord]['data'], n_cells=n_cells, left=True, right=True, top=True, bottom=True
                )
            for coord in ['_full_lat_bnds', '_full_lon_bnds']:
                if self.__dict__[coord] is not None:
                    self.__dict__[coord]['data'] = self._expand_2d_bounds(
                        self.__dict__[coord]['data'], n_cells=n_cells, left=True, right=True, bottom=True
                    )

        return None

    def _contract_full_coordinates(self, n_cells: int) -> None:
        """
        Contracts full latitude and longitude coordinates if `self.master` is True.

        Parameters
        ----------
        n_cells : int
            The number of cells to remove from each side.

        Returns
        -------
        None
        """
        if self.master:
            for coord in ['_full_rlat', '_full_rlon']:
                self.__dict__[coord]['data'] = self._contract_1d(
                    self.__dict__[coord]['data'], n_cells=n_cells, left=True, right=True
                )
            for coord in ['_full_lat', '_full_lon']:
                self.__dict__[coord]['data'] = self._contract_2d(
                    self.__dict__[coord]['data'], n_cells=n_cells, left=True, right=True, top=True, bottom=True,
                )
            for coord in ['_full_lat_bnds', '_full_lon_bnds']:
                if self.__dict__[coord] is not None:
                    self.__dict__[coord]['data'] = self._contract_2d_bounds(
                        self.__dict__[coord]['data'], n_cells=n_cells, left=True, right=True, top=True, bottom=True,
                    )
        return None

    # noinspection DuplicatedCode
    def _filter_coordinates_selection(self):
        """
        Use the selection limits to filter rlat, rlon, time, lev, lat, lon, lon_bnds and lat_bnds.
        """

        idx = self._get_idx_intervals()

        full_rlat = self.get_full_rlat()
        full_rlon = self.get_full_rlon()

        self.rlat = self._get_coordinate_values(full_rlat, "Y")
        self.rlon = self._get_coordinate_values(full_rlon, "X")

        if self.master:
            self.set_full_rlat({'data': full_rlat["data"][idx["idx_y_min"]:idx["idx_y_max"]]})
            self.set_full_rlon({'data': full_rlon["data"][idx["idx_x_min"]:idx["idx_x_max"]]})

        super(RotatedNes, self)._filter_coordinates_selection()

        return None

    def _get_pyproj_projection(self):
        """
        Get projection data as in Pyproj library.

        Returns
        ----------
        projection : pyproj.Proj
            Grid projection.
        """

        projection = Proj(proj="ob_tran",
                          o_proj="longlat",
                          ellps="WGS84",
                          R=self.earth_radius[0],
                          o_lat_p=float64(self.projection_data["grid_north_pole_latitude"]),
                          o_lon_p=float64(self.projection_data["grid_north_pole_longitude"]),
                          )

        return projection

    # noinspection DuplicatedCode
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
                               "grid_north_pole_latitude": 90 - kwargs["centre_lat"],
                               "grid_north_pole_longitude": -180 + kwargs["centre_lon"],
                               "inc_rlat": kwargs["inc_rlat"],
                               "inc_rlon": kwargs["inc_rlon"],
                               "south_boundary": kwargs["south_boundary"],
                               "west_boundary": kwargs["west_boundary"],
                               }
        else:
            if "rotated_pole" in self.variables.keys():
                projection_data = self.variables["rotated_pole"]
                self.free_vars("rotated_pole")
            else:
                msg = "There is no variable called rotated_pole, projection has not been defined."
                raise RuntimeError(msg)

            if "dtype" in projection_data.keys():
                del projection_data["dtype"]

            if "data" in projection_data.keys():
                del projection_data["data"]

            if "dimensions" in projection_data.keys():
                del projection_data["dimensions"]

        return projection_data

    def _create_dimensions(self, netcdf):
        """
        Create "rlat", "rlon"  and "spatial_nv" dimensions and the dimensions "lev", "time", "time_nv", "lon" and "lat".

        Parameters
        ----------
        netcdf : Dataset
            NetCDF object.
        """

        super(RotatedNes, self)._create_dimensions(netcdf)

        shape = self.get_full_shape()
        # Create rlat and rlon dimensions
        netcdf.createDimension("rlon", shape[1])
        netcdf.createDimension("rlat", shape[0])

        # Create spatial_nv (number of vertices) dimension
        if (self.lat_bnds is not None) and (self.lon_bnds is not None):
            netcdf.createDimension("spatial_nv", 4)
            pass

        return None

    def _create_dimension_variables(self, netcdf):
        """
        Create the "rlat" and "rlon" variables.

        Parameters
        ----------
        netcdf : Dataset
            NetCDF object.
        """
        super(RotatedNes, self)._create_dimension_variables(netcdf)

        # ROTATED LATITUDES
        full_rlat = self.get_full_rlat()
        rlat = netcdf.createVariable("rlat", full_rlat["data"].dtype, ("rlat",))
        rlat.long_name = "latitude in rotated pole grid"
        if "units" in full_rlat.keys():
            rlat.units = full_rlat["units"]
        else:
            rlat.units = "degrees"
        rlat.standard_name = "grid_latitude"
        if self.size > 1:
            rlat.set_collective(True)
        rlat[:] = full_rlat["data"]

        # ROTATED LONGITUDES
        full_rlon = self.get_full_rlon()
        rlon = netcdf.createVariable("rlon", full_rlon["data"].dtype, ("rlon",))
        rlon.long_name = "longitude in rotated pole grid"
        if "units" in full_rlon.keys():
            rlon.units = full_rlon["units"]
        else:
            rlon.units = "degrees"
        rlon.standard_name = "grid_longitude"
        if self.size > 1:
            rlon.set_collective(True)
        rlon[:] = full_rlon["data"]

        return None

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
        inc_rlon = float64(self.projection_data["inc_rlon"])
        inc_rlat = float64(self.projection_data["inc_rlat"])

        # Get south and west boundaries
        south_boundary = float64(self.projection_data["south_boundary"])
        west_boundary = float64(self.projection_data["west_boundary"])

        # Calculate rotated latitudes
        n_lat = int((abs(south_boundary) / inc_rlat) * 2 + 1)
        rlat = linspace(south_boundary, south_boundary + (inc_rlat * (n_lat - 1)), n_lat, dtype=float64)

        # Calculate rotated longitudes
        n_lon = int((abs(west_boundary) / inc_rlon) * 2 + 1)
        rlon = linspace(west_boundary, west_boundary + (inc_rlon * (n_lon - 1)), n_lon, dtype=float64)

        return {"data": rlat}, {"data": rlon}

    def rotated2latlon(self, lon_deg, lat_deg, lon_min=-180):
        """
        Calculate the unrotated coordinates using the rotated ones.

        Parameters
        ----------
        lon_deg : array
            Rotated longitude coordinate.
        lat_deg : array
            Rotated latitude coordinate.
        lon_min : float
            Minimum value for the longitudes: -180 (-180 to 180) or 0 (0 to 360).

        Returns
        ----------
        almd : array
            Unrotated longitudes.
        aphd : array
            Unrotated latitudes.
        """

        # Get centre coordinates
        centre_lat = 90 - float64(self.projection_data["grid_north_pole_latitude"])
        centre_lon = float64(self.projection_data["grid_north_pole_longitude"]) + 180

        # Convert to radians
        degrees_to_radians = pi / 180.
        tph0 = centre_lat * degrees_to_radians
        tlm = lon_deg * degrees_to_radians
        tph = lat_deg * degrees_to_radians

        tlm0d = -180 + centre_lon
        ctph0 = cos(tph0)
        stph0 = sin(tph0)
        stlm = sin(tlm)
        ctlm = cos(tlm)
        stph = sin(tph)
        ctph = cos(tph)

        # Calculate unrotated latitudes
        sph = (ctph0 * stph) + (stph0 * ctph * ctlm)
        sph[sph > 1.] = 1.
        sph[sph < -1.] = -1.
        aph = arcsin(sph)
        aphd = aph / degrees_to_radians

        # Calculate rotated longitudes
        anum = ctph * stlm
        denom = (ctlm * ctph - stph0 * sph) / ctph0
        relm = arctan2(anum, denom) - pi
        almd = relm / degrees_to_radians + tlm0d
        almd[almd > (lon_min + 360)] -= 360
        almd[almd < lon_min] += 360

        return almd, aphd

    def _create_centre_coordinates(self, **kwargs):
        """
        Calculate centre latitudes and longitudes from grid details.

        Returns
        ----------
        centre_lat : dict
            Dictionary with data of centre coordinates for latitude in 2D (latitude, longitude).
        centre_lon : dict
            Dictionary with data of centre coordinates for longitude in 2D (latitude, longitude).
        """
        if self.master:
            # Complete dimensions
            self._full_rlat, self._full_rlon = self._create_rotated_coordinates()

            # Calculate centre latitudes and longitudes (1D to 2D)
            centre_lon, centre_lat = self.rotated2latlon(
                array([self._full_rlon["data"]] * len(self._full_rlat["data"])),
                array([self._full_rlat["data"]] * len(self._full_rlon["data"])).T)

            return {"data": centre_lat}, {"data": centre_lon}
        else:
            return None, None

    def create_providentia_exp_centre_coordinates(self):
        """
        Calculate centre latitudes and longitudes from original coordinates and store as 2D arrays.

        Returns
        ----------
        model_centre_lat : dict
            Dictionary with data of centre coordinates for latitude in 2D (latitude, longitude).
        model_centre_lon : dict
            Dictionary with data of centre coordinates for longitude in 2D (latitude, longitude).
        """

        # Get centre latitudes
        model_centre_lat = self.lat

        # Get centre longitudes
        model_centre_lon = self.lon

        return model_centre_lat, model_centre_lon

    # noinspection DuplicatedCode
    def create_providentia_exp_grid_edge_coordinates(self):
        """
        Calculate grid edge latitudes and longitudes and get model grid outline.

        Returns
        ----------
        grid_edge_lat : dict
            Dictionary with data of grid edge latitudes.
        grid_edge_lon : dict
            Dictionary with data of grid edge longitudes.
        """

        # Get grid resolution
        inc_rlon = abs(mean(diff(self.rlon["data"])))
        inc_rlat = abs(mean(diff(self.rlat["data"])))

        # Get bounds for rotated coordinates
        rlat_bounds = self._create_single_spatial_bounds(self.rlat["data"], inc_rlat)
        rlon_bounds = self._create_single_spatial_bounds(self.rlon["data"], inc_rlon)

        # Get rotated latitudes for grid edge
        left_edge_rlat = append(rlat_bounds.flatten()[::2], rlat_bounds.flatten()[-1])
        right_edge_rlat = flip(left_edge_rlat, 0)
        top_edge_rlat = repeat(rlat_bounds[-1][-1], len(self.rlon["data"]) - 1)
        bottom_edge_rlat = repeat(rlat_bounds[0][0], len(self.rlon["data"]))
        rlat_grid_edge = concatenate((left_edge_rlat, top_edge_rlat, right_edge_rlat, bottom_edge_rlat))

        # Get rotated longitudes for grid edge
        left_edge_rlon = repeat(rlon_bounds[0][0], len(self.rlat["data"]) + 1)
        top_edge_rlon = rlon_bounds.flatten()[1:-1:2]
        right_edge_rlon = repeat(rlon_bounds[-1][-1], len(self.rlat["data"]) + 1)
        bottom_edge_rlon = flip(rlon_bounds.flatten()[:-1:2], 0)
        rlon_grid_edge = concatenate((left_edge_rlon, top_edge_rlon, right_edge_rlon, bottom_edge_rlon))

        # Get edges for regular coordinates
        grid_edge_lon_data, grid_edge_lat_data = self.rotated2latlon(rlon_grid_edge, rlat_grid_edge)

        # Create grid outline by stacking the edges in both coordinates
        model_grid_outline = vstack((grid_edge_lon_data, grid_edge_lat_data)).T

        grid_edge_lat = {"data": model_grid_outline[:, 1]}
        grid_edge_lon = {"data": model_grid_outline[:, 0]}

        return grid_edge_lat, grid_edge_lon

    # noinspection DuplicatedCode
    def create_spatial_bounds(self):
        """
        Calculate longitude and latitude bounds and set them.
        """

        # Calculate rotated coordinates bounds
        full_rlat = self.get_full_rlat()
        full_rlon = self.get_full_rlon()
        inc_rlat = abs(mean(diff(full_rlat["data"])))
        rlat_bnds = self._create_single_spatial_bounds(array([full_rlat["data"]] * len(full_rlon["data"])).T,
                                                       inc_rlat, spatial_nv=4, inverse=True)

        inc_rlon = abs(mean(diff(full_rlon["data"])))
        rlon_bnds = self._create_single_spatial_bounds(array([full_rlon["data"]] * len(full_rlat["data"])),
                                                       inc_rlon, spatial_nv=4)

        # Transform rotated bounds to regular bounds
        lon_bnds, lat_bnds = self.rotated2latlon(rlon_bnds, rlat_bnds)

        # Obtain regular coordinates bounds
        self.set_full_latitudes_boundaries({"data": deepcopy(lat_bnds)})
        self.lat_bnds = {"data": lat_bnds[self.write_axis_limits["y_min"]:self.write_axis_limits["y_max"],
                                          self.write_axis_limits["x_min"]:self.write_axis_limits["x_max"],
                                          :]}

        self.set_full_longitudes_boundaries({"data": deepcopy(lon_bnds)})
        self.lon_bnds = {"data": lon_bnds[self.write_axis_limits["y_min"]:self.write_axis_limits["y_max"],
                                          self.write_axis_limits["x_min"]:self.write_axis_limits["x_max"],
                                          :]}

        return None

    @staticmethod
    def _set_var_crs(var):
        """
        Set the grid_mapping to "rotated_pole".

        Parameters
        ----------
        var : Variable
            netCDF4-python variable object.
        """

        var.grid_mapping = "rotated_pole"
        var.coordinates = "lat lon"

        return None

    def _create_metadata(self, netcdf):
        """
        Create the "crs" variable for the rotated latitude longitude grid_mapping.

        Parameters
        ----------
        netcdf : Dataset
            netcdf4-python Dataset.
        """

        if self.projection_data is not None:
            mapping = netcdf.createVariable("rotated_pole", "i")
            mapping.grid_mapping_name = self.projection_data["grid_mapping_name"]
            mapping.grid_north_pole_latitude = self.projection_data["grid_north_pole_latitude"]
            mapping.grid_north_pole_longitude = self.projection_data["grid_north_pole_longitude"]

        return None

    def to_grib2(self, path, grib_keys, grib_template_path, lat_flip=False, info=False):
        """
        Write output file with grib2 format.

        Parameters
        ----------
        lat_flip : bool
            Indicates if you want to flip the latitude coordinates.
        path : str
            Path to the output file.
        grib_keys : dict
            Dictionary with the grib2 keys.
        grib_template_path : str
            Path to the grib2 file to use as template.
        info : bool
            Indicates if you want to print extra information during the process.
        """

        raise NotImplementedError("Grib2 format cannot be written in a Rotated pole projection.")

    # noinspection DuplicatedCode
    def create_shapefile(self, overwrite=False):
        """
        Create spatial geodataframe (shapefile).

        Returns
        -------
        shapefile : GeoPandasDataFrame
            Shapefile dataframe.
        """

        if self.shapefile is None or overwrite:

            if self.lat_bnds is None or self.lon_bnds is None:
                self.create_spatial_bounds()

            # Reshape arrays to create geometry
            aux_b_lats = self.lat_bnds["data"].reshape((self.lat_bnds["data"].shape[0] * self.lat_bnds["data"].shape[1],
                                                        self.lat_bnds["data"].shape[2]))
            aux_b_lons = self.lon_bnds["data"].reshape((self.lon_bnds["data"].shape[0] * self.lon_bnds["data"].shape[1],
                                                        self.lon_bnds["data"].shape[2]))

            sys.stdout.flush()
            self.comm.Barrier()
            # Get polygons from bounds
            geometry = []
            for i in range(aux_b_lons.shape[0]):
                geometry.append(Polygon([(aux_b_lons[i, 0], aux_b_lats[i, 0]),
                                         (aux_b_lons[i, 1], aux_b_lats[i, 1]),
                                         (aux_b_lons[i, 2], aux_b_lats[i, 2]),
                                         (aux_b_lons[i, 3], aux_b_lats[i, 3]),
                                         (aux_b_lons[i, 0], aux_b_lats[i, 0])]))

            # Create dataframe containing all polygons
            fids = self.get_fids()
            gdf = GeoDataFrame(index=Index(name="FID", data=fids.ravel()), geometry=geometry, crs="EPSG:4326")
            self.shapefile = gdf

        else:
            gdf = self.shapefile

        return gdf

    # noinspection DuplicatedCode
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
        for lat_ind in range(0, self.lon["data"].shape[0]):
            for lon_ind in range(0, self.lon["data"].shape[1]):
                centroids.append(Point(self.lon["data"][lat_ind, lon_ind],
                                       self.lat["data"][lat_ind, lon_ind]))

        # Create dataframe cointaining all points
        fids = self.get_fids()
        centroids_gdf = GeoDataFrame(index=Index(name="FID", data=fids.ravel()), geometry=centroids, crs="EPSG:4326")

        return centroids_gdf
