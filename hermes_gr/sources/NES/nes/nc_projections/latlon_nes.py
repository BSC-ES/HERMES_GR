#!/usr/bin/env python
# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

from numpy import float64, linspace, meshgrid, mean, diff, append, flip, repeat, concatenate, vstack
from pyproj import Proj
from .default_nes import Nes


class LatLonNes(Nes):
    """

    Attributes
    ----------
    _var_dim : tuple
        A Tuple with the name of the Y and X dimensions for the variables.
        ("lat", "lon") for a regular latitude-longitude projection.
    _lat_dim : tuple
        A Tuple with the name of the dimensions of the Latitude values.
        ("lat", ) for a regular latitude-longitude projection.
    _lon_dim : tuple
        A Tuple with the name of the dimensions of the Longitude values.
        ("lon", ) for a regular latitude-longitude projection.
    """
    def __init__(self, comm=None, path=None, info=False, dataset=None, parallel_method="Y",
                 avoid_first_hours=0, avoid_last_hours=0, first_level=0, last_level=None, create_nes=False,
                 balanced=False, times=None, **kwargs):
        """
        Initialize the LatLonNes class.

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

        super(LatLonNes, self).__init__(comm=comm, path=path, info=info, dataset=dataset,
                                        parallel_method=parallel_method, balanced=balanced,
                                        avoid_first_hours=avoid_first_hours, avoid_last_hours=avoid_last_hours,
                                        first_level=first_level, last_level=last_level, create_nes=create_nes,
                                        times=times, **kwargs)

        if create_nes:
            # Dimensions screening
            self.lat = self._get_coordinate_values(self.get_full_latitudes(), "Y")
            self.lon = self._get_coordinate_values(self.get_full_longitudes(), "X")

        # Set axis limits for parallel writing
        self.write_axis_limits = self._get_write_axis_limits()

        self._var_dim = ("lat", "lon")
        self._lat_dim = ("lat",)
        self._lon_dim = ("lon",)

        self.free_vars("crs")

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

        new = LatLonNes(comm=comm, path=path, info=info, dataset=dataset,
                        parallel_method=parallel_method, avoid_first_hours=avoid_first_hours,
                        avoid_last_hours=avoid_last_hours, first_level=first_level, last_level=last_level,
                        create_nes=create_nes, balanced=balanced, times=times, **kwargs)

        return new

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
        for coord in ['lat', 'lat_bnds', 'lon', 'lon_bnds']:
            expand_left = is_first if "lat" in coord else True
            expand_right = is_last if "lat" in coord else True
            if self.__dict__[coord] is not None:
                self.__dict__[coord]['data'] = self._expand_1d(
                    self.__dict__[coord]['data'], n_cells=n_cells, left=expand_left, right=expand_right
                )
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
        for coord in ['lat', 'lat_bnds', 'lon', 'lon_bnds']:
            contract_left = is_first if "lat" in coord else True
            contract_right = is_last if "lat" in coord else True
            if self.__dict__[coord] is not None:
                self.__dict__[coord]['data'] = self._contract_1d(
                    self.__dict__[coord]['data'], n_cells=n_cells, left=contract_left, right=contract_right
                )
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
            for coord in ['_full_lat', '_full_lat_bnds', '_full_lon', '_full_lon_bnds']:
                if self.__dict__[coord] is not None:
                    self.__dict__[coord]['data'] = self._expand_1d(
                        self.__dict__[coord]['data'], n_cells=n_cells, left=True, right=True
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
            for coord in ['_full_lat', '_full_lat_bnds', '_full_lon', '_full_lon_bnds']:
                if self.__dict__[coord] is not None:
                    self.__dict__[coord]['data'] = self._contract_1d(
                        self.__dict__[coord]['data'], n_cells=n_cells, left=True, right=True
                    )
        return None

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

        Returns
        -------
        Dict[str, Any]
            A dictionary containing projection data with the following keys:
            - "grid_mapping_name" : str
                Type of grid mapping (e.g., "latitude_longitude").
            - "semi_major_axis" : float
                Semi-major axis of the Earth's ellipsoid.
            - "inverse_flattening" : int
                Inverse flattening parameter.
            - "inc_lat" : float
                Increment in latitude.
            - "inc_lon" : float
                Increment in longitude.
            - "lat_orig" : float
                Origin latitude of the grid.
            - "lon_orig" : float
                Origin longitude of the grid.
            - "n_lat" : int
                Number of grid points along latitude.
            - "n_lon" : int
                Number of grid points along longitude.

        Notes
        -----
        Depending on the `create_nes` flag and input `kwargs`, the method constructs
        or retrieves projection data. If `create_nes` is True, the method initializes
        projection details based on provided arguments such as increments (`inc_lat`, `inc_lon`),
        and if additional keyword arguments (`lat_orig`, `lon_orig`, `n_lat`, `n_lon`) are not provided,
        defaults for the global domain are used. If `create_nes` is False, the method checks for
        an existing "crs" variable in `self.variables` and retrieves its data, freeing the "crs" variable
        afterward to optimize memory usage.

        """
        if create_nes:
            projection_data = {"grid_mapping_name": "latitude_longitude",
                               "semi_major_axis": self.earth_radius[1],
                               "inverse_flattening": 0,
                               "inc_lat": kwargs["inc_lat"],
                               "inc_lon": kwargs["inc_lon"],
                               }
            nes_projection = kwargs.get("nes_projection")
            # Global domain
            if nes_projection == "global_monarch":
                lat_cells_each_side = round(90 / float64(projection_data["inc_lat"]))
                lon_cells_each_side = round(180 / float64(projection_data["inc_lon"]))
                if abs(lat_cells_each_side - (90 / float64(projection_data["inc_lat"]))) > 1e-9:
                    raise ValueError("inc_lat must divide 90 exactly to create a global_monarch projection")
                if abs(lon_cells_each_side - (180 / float64(projection_data["inc_lon"]))) > 1e-9:
                    raise ValueError("inc_lon must divide 180 exactly to create a global_monarch projection")

                projection_data["nes_projection"] = nes_projection
                projection_data["lat_orig"] = -90 - (float64(projection_data["inc_lat"]) / 2)
                projection_data["lon_orig"] = -180 - (float64(projection_data["inc_lon"]) / 2)
                projection_data["n_lat"] = int((lat_cells_each_side * 2) + 1)
                projection_data["n_lon"] = int((lon_cells_each_side * 2) + 1)
            elif nes_projection == "global" or len(kwargs) == 2:
                projection_data["lat_orig"] = -90
                projection_data["lon_orig"] = -180
                projection_data["n_lat"] = int(180 // float64(projection_data["inc_lat"]))
                projection_data["n_lon"] = int(360 // float64(projection_data["inc_lon"]))
            # Other domains
            else:
                projection_data["lat_orig"] = kwargs["lat_orig"]
                projection_data["lon_orig"] = kwargs["lon_orig"]
                projection_data["n_lat"] = kwargs["n_lat"]
                projection_data["n_lon"] = kwargs["n_lon"]
        else:
            if "crs" in self.variables.keys():
                projection_data = self.variables["crs"]
                self.free_vars("crs")
            else:
                projection_data = {"grid_mapping_name": "latitude_longitude",
                                   "semi_major_axis": self.earth_radius[1],
                                   "inverse_flattening": 0,
                                   }

            if "dtype" in projection_data.keys():
                del projection_data["dtype"]

            if "data" in projection_data.keys():
                del projection_data["data"]

            if "dimensions" in projection_data.keys():
                del projection_data["dimensions"]

        return projection_data

    def _create_dimensions(self, netcdf):
        """
        Create "spatial_nv" dimensions and the super dimensions "lev", "time", "time_nv", "lon" and "lat".

        Parameters
        ----------
        netcdf : Dataset
            NetCDF object.
        """

        super(LatLonNes, self)._create_dimensions(netcdf)

        netcdf.createDimension("lon", len(self.get_full_longitudes()["data"]))
        netcdf.createDimension("lat", len(self.get_full_latitudes()["data"]))

        # Create spatial_nv (number of vertices) dimension
        if (self.lat_bnds is not None) and (self.lon_bnds is not None):
            netcdf.createDimension("spatial_nv", 2)

        return None

    def _create_centre_coordinates(self, **kwargs):
        """
        Calculate centre latitudes and longitudes from grid details.

        Returns
        ----------
        centre_lat : dict
            Dictionary with data of centre latitudes in 1D
        centre_lon : dict
            Dictionary with data of centre longitudes in 1D
        """

        # Get grid resolution
        inc_lat = float64(self.projection_data["inc_lat"])
        inc_lon = float64(self.projection_data["inc_lon"])

        # Get coordinates origen
        lat_orig = float64(self.projection_data["lat_orig"])
        lon_orig = float64(self.projection_data["lon_orig"])

        # Get number of coordinates
        n_lat = int(self.projection_data["n_lat"])
        n_lon = int(self.projection_data["n_lon"])

        # Calculate centre latitudes
        lat_c_orig = lat_orig + (inc_lat / 2)
        centre_lat = linspace(lat_c_orig, lat_c_orig + (inc_lat * (n_lat - 1)), n_lat, dtype=float64)

        # Calculate centre longitudes
        lon_c_orig = lon_orig + (inc_lon / 2)
        centre_lon = linspace(lon_c_orig, lon_c_orig + (inc_lon * (n_lon - 1)), n_lon, dtype=float64)

        if self.projection_data.get("nes_projection") == "global_monarch":
            centre_lat[0] += inc_lat / 4
            centre_lat[-1] -= inc_lat / 4
            centre_lon[0] += inc_lon / 4
            centre_lon[-1] -= inc_lon / 4

        return {"data": centre_lat}, {"data": centre_lon}

    def create_spatial_bounds(self):
        """
        Calculate longitude and latitude bounds and clip global Monarch edge cells.
        """

        if self.projection_data.get("nes_projection") == "global_monarch":
            full_lat = self.get_full_latitudes()
            lat_bnds = self._create_single_spatial_bounds(
                full_lat["data"], float64(self.projection_data["inc_lat"]), spatial_nv=2
            ).clip(-90, 90)
            self.set_full_latitudes_boundaries({"data": lat_bnds})
            self.lat_bnds = {
                "data": lat_bnds[
                    self.write_axis_limits["y_min"]: self.write_axis_limits["y_max"], :
                ]
            }

            full_lon = self.get_full_longitudes()
            lon_bnds = self._create_single_spatial_bounds(
                full_lon["data"], float64(self.projection_data["inc_lon"]), spatial_nv=2
            ).clip(-180, 180)
            self.set_full_longitudes_boundaries({"data": lon_bnds})
            self.lon_bnds = {
                "data": lon_bnds[
                    self.write_axis_limits["x_min"]: self.write_axis_limits["x_max"], :
                ]
            }
        else:
            super(LatLonNes, self).create_spatial_bounds()

        return None

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

        model_centre_lon_data, model_centre_lat_data = meshgrid(self.lon["data"], self.lat["data"])

        # Calculate centre latitudes
        model_centre_lat = {"data": model_centre_lat_data}

        # Calculate centre longitudes
        model_centre_lon = {"data": model_centre_lon_data}

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
        inc_lon = abs(mean(diff(self.lon["data"])))
        inc_lat = abs(mean(diff(self.lat["data"])))

        # Get bounds
        lat_bounds = self._create_single_spatial_bounds(self.lat["data"], inc_lat)
        lon_bounds = self._create_single_spatial_bounds(self.lon["data"], inc_lon)

        # Get latitudes for grid edge
        left_edge_lat = append(lat_bounds.flatten()[::2], lat_bounds.flatten()[-1])
        right_edge_lat = flip(left_edge_lat, 0)
        top_edge_lat = repeat(lat_bounds[-1][-1], len(self.lon["data"]) - 1)
        bottom_edge_lat = repeat(lat_bounds[0][0], len(self.lon["data"]))
        lat_grid_edge = concatenate((left_edge_lat, top_edge_lat, right_edge_lat, bottom_edge_lat))

        # Get longitudes for grid edge
        left_edge_lon = repeat(lon_bounds[0][0], len(self.lat["data"]) + 1)
        top_edge_lon = lon_bounds.flatten()[1:-1:2]
        right_edge_lon = repeat(lon_bounds[-1][-1], len(self.lat["data"]) + 1)
        bottom_edge_lon = flip(lon_bounds.flatten()[:-1:2], 0)
        lon_grid_edge = concatenate((left_edge_lon, top_edge_lon, right_edge_lon, bottom_edge_lon))

        # Create grid outline by stacking the edges in both coordinates
        model_grid_outline = vstack((lon_grid_edge, lat_grid_edge)).T
        grid_edge_lat = {"data": model_grid_outline[:, 1]}
        grid_edge_lon = {"data": model_grid_outline[:, 0]}

        return grid_edge_lat, grid_edge_lon

    @staticmethod
    def _set_var_crs(var):
        """
        Set the grid_mapping to "crs".

        Parameters
        ----------
        var : Variable
            netCDF4-python variable object.
        """

        var.grid_mapping = "crs"
        # var.coordinates = "lat lon"

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
            mapping = netcdf.createVariable("crs", "i")
            mapping.grid_mapping_name = self.projection_data["grid_mapping_name"]
            mapping.semi_major_axis = self.projection_data["semi_major_axis"]
            if self.projection_data.get("inverse_flattening") is not None:
                mapping.inverse_flattening = self.projection_data["inverse_flattening"]
            if self.projection_data.get("nes_projection") is not None:
                mapping.nes_projection = self.projection_data["nes_projection"]

        return None

    def to_grib2(self, path, grib_keys, grib_template_path, lat_flip=False, info=False):
        """
        Write output file with grib2 format.

        Parameters
        ----------
        lat_flip : bool
            Indicates if the latitudes have to be flipped
        path : str
            Path to the output file.
        grib_keys : dict
            Dictionary with the grib2 keys.
        grib_template_path : str
            Path to the grib2 file to use as template.
        info : bool
            Indicates if you want to print extra information during the process.
        """

        return super(LatLonNes, self).to_grib2(path, grib_keys, grib_template_path, lat_flip=lat_flip, info=info)
