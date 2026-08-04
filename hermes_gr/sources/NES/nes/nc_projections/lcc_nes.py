#!/usr/bin/env python
# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

from numpy import float64, linspace, array, mean, diff, append, flip, repeat, concatenate, vstack
from geopandas import GeoDataFrame
from pandas import Index
from pyproj import Proj
from copy import deepcopy
from typing import Dict, Any
from shapely.geometry import Polygon, Point
from .default_nes import Nes


class LCCNes(Nes):
    """

    Attributes
    ----------
    _full_y : dict
        Y coordinates dictionary with the complete "data" key for all the values and the rest of the attributes.
    _full_x : dict
        X coordinates dictionary with the complete "data" key for all the values and the rest of the attributes.
    y : dict
        Y coordinates dictionary with the portion of "data" corresponding to the rank values.
    x : dict
        X coordinates dictionary with the portion of "data" corresponding to the rank values.
    _var_dim : tuple
        A Tuple with the name of the Y and X dimensions for the variables.
        ("y", "x", ) for an LCC projection.
    _lat_dim : tuple
        A Tuple with the name of the dimensions of the Latitude values.
        ("y", "x", ) for an LCC projection.
    _lon_dim : tuple
        ATuple with the name of the dimensions of the Longitude values.
        ("y", "x") for an LCC projection.
    """
    def __init__(self, comm=None, path=None, info=False, dataset=None, parallel_method="Y",
                 avoid_first_hours=0, avoid_last_hours=0, first_level=0, last_level=None, create_nes=False,
                 balanced=False, times=None, **kwargs):
        """
        Initialize the LCCNes class

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
        self._full_y = None
        self._full_x = None

        super(LCCNes, self).__init__(comm=comm, path=path, info=info, dataset=dataset,
                                     parallel_method=parallel_method, balanced=balanced,
                                     avoid_first_hours=avoid_first_hours, avoid_last_hours=avoid_last_hours,
                                     first_level=first_level, last_level=last_level, create_nes=create_nes,
                                     times=times, **kwargs)

        if create_nes:
            # Dimensions screening
            self.lat = self._get_coordinate_values(self.get_full_latitudes(), "Y")
            self.lon = self._get_coordinate_values(self.get_full_longitudes(), "X")
        else:
            # Complete dimensions
            self._full_y = self._get_coordinate_dimension("y")
            self._full_x = self._get_coordinate_dimension("x")

        # Dimensions screening
        self.y = self._get_coordinate_values(self.get_full_y(), "Y")
        self.x = self._get_coordinate_values(self.get_full_x(), "X")

        # Set axis limits for parallel writing
        self.write_axis_limits = self._get_write_axis_limits()

        self._var_dim = ("y", "x")
        self._lat_dim = ("y", "x")
        self._lon_dim = ("y", "x")

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

        new = LCCNes(comm=comm, path=path, info=info, dataset=dataset,
                     parallel_method=parallel_method, avoid_first_hours=avoid_first_hours,
                     avoid_last_hours=avoid_last_hours, first_level=first_level, last_level=last_level,
                     create_nes=create_nes, balanced=balanced, times=times, **kwargs)

        return new

    def get_full_y(self) -> Dict[str, Any]:
        """
        Retrieve the complete Y information.

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
        data = self.comm.bcast(self._full_y, root=0)

        return data

    def get_full_x(self) -> Dict[str, Any]:
        """
        Retrieve the complete X information.

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
        data = self.comm.bcast(self._full_x, root=0)
        return data

    def set_full_y(self, data: Dict[str, Any]) -> None:
        """
        Set the complete Y information.

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
            self._full_y = data
        return None

    def set_full_x(self, data: Dict[str, Any]) -> None:
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
            self._full_x = data
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
        for coord in ['x', 'y']:
            expand_left = is_first if "y" in coord else True
            expand_right = is_last if "y" in coord else True
            self.__dict__[coord]['data'] = self._expand_1d(
                self.__dict__[coord]['data'], n_cells=n_cells, left=expand_left, right=expand_right
            )
        for coord in ['lat', 'lon']:
            expand_left = is_first if "lat" in coord else True
            expand_right = is_last if "lat" in coord else True
            self.__dict__[coord]['data'] = self._expand_2d(
                self.__dict__[coord]['data'], n_cells=n_cells, left=expand_left, right=expand_right
            )
        for coord in ['lat_bnds', 'lon_bnds']:
            expand_left = is_first if "lat" in coord else True
            expand_right = is_last if "lat" in coord else True
            if self.__dict__[coord] is not None:
                self.__dict__[coord]['data'] = self._expand_2d_bounds(
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
        for coord in ['x', 'y']:
            contract_left = is_first if "y" in coord else True
            contract_right = is_last if "y" in coord else True
            self.__dict__[coord]['data'] = self._contract_1d(
                self.__dict__[coord]['data'], n_cells=n_cells, left=contract_left, right=contract_right
            )
        for coord in ['lat', 'lon']:
            contract_left = is_first if "lat" in coord else True
            contract_right = is_last if "lat" in coord else True
            self.__dict__[coord]['data'] = self._contract_2d(
                self.__dict__[coord]['data'], n_cells=n_cells, left=contract_left, right=contract_right
            )
        for coord in ['lat_bnds', 'lon_bnds']:
            contract_left = is_first if "lat" in coord else True
            contract_right = is_last if "lat" in coord else True
            if self.__dict__[coord] is not None:
                self.__dict__[coord]['data'] = self._contract_2d_bounds(
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
            for coord in ['_full_x', '_full_y']:
                self.__dict__[coord]['data'] = self._expand_1d(
                    self.__dict__[coord]['data'], n_cells=n_cells, left=True, right=True
                )
            for coord in ['_full_lat', '_full_lon', ]:
                self.__dict__[coord]['data'] = self._expand_2d(
                    self.__dict__[coord]['data'], n_cells=n_cells, left=True, right=True
                )
            for coord in ['_full_lat_bnds', '_full_lon_bnds']:
                if self.__dict__[coord] is not None:
                    self.__dict__[coord]['data'] = self._expand_2d_bounds(
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
            for coord in ['_full_x', '_full_y']:
                self.__dict__[coord]['data'] = self._contract_1d(
                    self.__dict__[coord]['data'], n_cells=n_cells, left=True, right=True
                )
            for coord in ['_full_lat', '_full_lon']:
                self.__dict__[coord]['data'] = self._contract_2d(
                    self.__dict__[coord]['data'], n_cells=n_cells, left=True, right=True
                )
            for coord in ['_full_lat_bnds', '_full_lon_bnds']:
                if self.__dict__[coord] is not None:
                    self.__dict__[coord]['data'] = self._contract_2d_bounds(
                        self.__dict__[coord]['data'], n_cells=n_cells, left=True, right=True
                    )
        return None

    # noinspection DuplicatedCode
    def _filter_coordinates_selection(self):
        """
        Use the selection limits to filter y, x, time, lev, lat, lon, lon_bnds and lat_bnds.
        """

        idx = self._get_idx_intervals()

        self.y = self._get_coordinate_values(self.get_full_y(), "Y")
        self.x = self._get_coordinate_values(self.get_full_x(), "X")

        self.set_full_y({'data': self.y["data"][idx["idx_y_min"]:idx["idx_y_max"]]})
        self.set_full_x({'data': self.x["data"][idx["idx_x_min"]:idx["idx_x_max"]]})

        super(LCCNes, self)._filter_coordinates_selection()

        return None

    def _get_pyproj_projection(self):
        """
        Get projection data as in Pyproj library.

        Returns
        ----------
        projection : pyproj.Proj
            Grid projection.
        """

        projection = Proj(proj="lcc",
                          ellps="WGS84",
                          R=self.earth_radius[0],
                          lat_1=float64(self.projection_data["standard_parallel"][0]),
                          lat_2=float64(self.projection_data["standard_parallel"][1]),
                          lon_0=float64(self.projection_data["longitude_of_central_meridian"]),
                          lat_0=float64(self.projection_data["latitude_of_projection_origin"]),
                          to_meter=1,
                          x_0=0,
                          y_0=0,
                          a=self.earth_radius[1],
                          k_0=1.0,
                          )

        return projection

    def _get_projection_data(self, create_nes, **kwargs):
        """
        Retrieves projection data based on grid details.

        Parameters
        ----------
        create_nes : bool
            Flag indicating whether to create new object (True) or use existing (False).
        **kwargs : dict
            Additional keyword arguments for specifying projection details.        """
        if create_nes:
            projection_data = {"grid_mapping_name": "lambert_conformal_conic",
                               "standard_parallel": [kwargs["lat_1"], kwargs["lat_2"]],
                               "longitude_of_central_meridian": kwargs["lon_0"],
                               "latitude_of_projection_origin": kwargs["lat_0"],
                               "x_0": kwargs["x_0"], "y_0": kwargs["y_0"],
                               "inc_x": kwargs["inc_x"], "inc_y": kwargs["inc_y"],
                               "nx": kwargs["nx"], "ny": kwargs["ny"],
                               }
        else:
            if "Lambert_Conformal" in self.variables.keys():
                projection_data = self.variables["Lambert_Conformal"]
                self.free_vars("Lambert_Conformal")
            elif "Lambert_conformal" in self.variables.keys():
                projection_data = self.variables["Lambert_conformal"]
                self.free_vars("Lambert_conformal")
            else:
                # We will never have this condition since the LCC grid will never be correctly detected
                # since the function __is_lcc in load_nes only detects LCC grids when there is Lambert_conformal
                msg = "There is no variable called Lambert_Conformal, projection has not been defined."
                raise RuntimeError(msg)

            if "dtype" in projection_data.keys():
                del projection_data["dtype"]

            if "data" in projection_data.keys():
                del projection_data["data"]

            if "dimensions" in projection_data.keys():
                del projection_data["dimensions"]

            if isinstance(projection_data["standard_parallel"], str):
                projection_data["standard_parallel"] = [projection_data["standard_parallel"].split(", ")[0],
                                                        projection_data["standard_parallel"].split(", ")[1]]

        return projection_data

    # noinspection DuplicatedCode
    def _create_dimensions(self, netcdf):
        """
        Create "y", "x" and "spatial_nv" dimensions and the super dimensions "lev", "time", "time_nv", "lon" and "lat"

        Parameters
        ----------
        netcdf : Dataset
            NetCDF object.
        """

        super(LCCNes, self)._create_dimensions(netcdf)

        # Create y and x dimensions
        netcdf.createDimension("y", len(self.get_full_y()["data"]))
        netcdf.createDimension("x", len(self.get_full_x()["data"]))

        # Create spatial_nv (number of vertices) dimension
        if (self.lat_bnds is not None) and (self.lon_bnds is not None):
            netcdf.createDimension("spatial_nv", 4)

        return None

    # noinspection DuplicatedCode
    def _create_dimension_variables(self, netcdf):
        """
        Create the "y" and "x" variables.

        Parameters
        ----------
        netcdf : Dataset
            NetCDF object.
        """

        super(LCCNes, self)._create_dimension_variables(netcdf)

        # LCC Y COORDINATES
        full_y = self.get_full_y()
        y = netcdf.createVariable("y", full_y["data"].dtype, ("y",))
        y.long_name = "y coordinate of projection"
        if "units" in full_y.keys():
            y.units = full_y["units"]
        else:
            y.units = "m"
        y.standard_name = "projection_y_coordinate"
        if self.size > 1:
            y.set_collective(True)
        y[:] = full_y["data"]

        # LCC X COORDINATES
        full_x = self.get_full_x()
        x = netcdf.createVariable("x", full_x["data"].dtype, ("x",))
        x.long_name = "x coordinate of projection"
        if "units" in full_x.keys():
            x.units = full_x["units"]
        else:
            x.units = "m"
        x.standard_name = "projection_x_coordinate"
        if self.size > 1:
            x.set_collective(True)
        x[:] = full_x["data"]

        return None

    # noinspection DuplicatedCode
    def _create_centre_coordinates(self, **kwargs):
        """
        Calculate centre latitudes and longitudes from grid details.

        Parameters
        ----------
        netcdf : Dataset
            NetCDF object.
        """
        if self.master:
            # Get projection details on x
            x_0 = float64(self.projection_data["x_0"])
            inc_x = float64(self.projection_data["inc_x"])
            nx = int(self.projection_data["nx"])

            # Get projection details on y
            y_0 = float64(self.projection_data["y_0"])
            inc_y = float64(self.projection_data["inc_y"])
            ny = int(self.projection_data["ny"])

            # Create a regular grid in metres (1D)
            self._full_x = {"data": linspace(x_0 + (inc_x / 2), x_0 + (inc_x / 2) + (inc_x * (nx - 1)), nx,
                                             dtype=float64)}
            self._full_y = {"data": linspace(y_0 + (inc_y / 2), y_0 + (inc_y / 2) + (inc_y * (ny - 1)), ny,
                                             dtype=float64)}

            # Create a regular grid in metres (1D to 2D)
            x = array([self._full_x["data"]] * len(self._full_y["data"]))
            y = array([self._full_y["data"]] * len(self._full_x["data"])).T

            # Calculate centre latitudes and longitudes (UTM to LCC)
            centre_lon, centre_lat = self.projection(x, y, inverse=True)

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
        inc_x = abs(mean(diff(self.x["data"])))
        inc_y = abs(mean(diff(self.y["data"])))

        # Get bounds for rotated coordinates
        y_bnds = self._create_single_spatial_bounds(self.y["data"], inc_y)
        x_bnds = self._create_single_spatial_bounds(self.x["data"], inc_x)

        # Get rotated latitudes for grid edge
        left_edge_y = append(y_bnds.flatten()[::2], y_bnds.flatten()[-1])
        right_edge_y = flip(left_edge_y, 0)
        top_edge_y = repeat(y_bnds[-1][-1], len(self.x["data"]) - 1)
        bottom_edge_y = repeat(y_bnds[0][0], len(self.x["data"]))
        y_grid_edge = concatenate((left_edge_y, top_edge_y, right_edge_y, bottom_edge_y))

        # Get rotated longitudes for grid edge
        left_edge_x = repeat(x_bnds[0][0], len(self.y["data"]) + 1)
        top_edge_x = x_bnds.flatten()[1:-1:2]
        right_edge_x = repeat(x_bnds[-1][-1], len(self.y["data"]) + 1)
        bottom_edge_x = flip(x_bnds.flatten()[:-1:2], 0)
        x_grid_edge = concatenate((left_edge_x, top_edge_x, right_edge_x, bottom_edge_x))

        # Get edges for regular coordinates
        grid_edge_lon_data, grid_edge_lat_data = self.projection(x_grid_edge, y_grid_edge, inverse=True)

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

        # Calculate LCC coordinates bounds
        full_x = self.get_full_x()
        full_y = self.get_full_y()
        inc_x = abs(mean(diff(full_x["data"])))
        x_bnds = self._create_single_spatial_bounds(array([full_x["data"]] * len(full_y["data"])),
                                                    inc_x, spatial_nv=4)

        inc_y = abs(mean(diff(full_y["data"])))
        y_bnds = self._create_single_spatial_bounds(array([full_y["data"]] * len(full_x["data"])).T,
                                                    inc_y, spatial_nv=4, inverse=True)

        # Transform LCC bounds to regular bounds
        lon_bnds, lat_bnds = self.projection(x_bnds, y_bnds, inverse=True)

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
        Set the grid_mapping to "Lambert_Conformal".

        Parameters
        ----------
        var : Variable
            netCDF4-python variable object.
        """

        var.grid_mapping = "Lambert_Conformal"
        var.coordinates = "lat lon"

        return None

    def _create_metadata(self, netcdf):
        """
        Create the "crs" variable for the lambert conformal grid_mapping.

        Parameters
        ----------
        netcdf : Dataset
            netcdf4-python Dataset
        """

        if self.projection_data is not None:
            mapping = netcdf.createVariable("Lambert_Conformal", "i")
            mapping.grid_mapping_name = self.projection_data["grid_mapping_name"]
            mapping.standard_parallel = self.projection_data["standard_parallel"]
            mapping.longitude_of_central_meridian = self.projection_data["longitude_of_central_meridian"]
            mapping.latitude_of_projection_origin = self.projection_data["latitude_of_projection_origin"]

        return None

    def to_grib2(self, path, grib_keys, grib_template_path, lat_flip=False, info=False):
        """
        Write output file with grib2 format.

        Parameters
        ----------
        lat_flip : bool
            Indicates if the latitudes need to be flipped Up-Down or Down-Up. Default False.
        path : str
            Path to the output file.
        grib_keys : dict
            Dictionary with the grib2 keys.
        grib_template_path : str
            Path to the grib2 file to use as template.
        info : bool
            Indicates if you want to print extra information during the process.
        """

        raise NotImplementedError("Grib2 format cannot be written in a Lambert Conformal Conic projection.")

    # noinspection DuplicatedCode
    def create_shapefile(self, overwrite=False):
        """
        Create spatial GeoDataFrame (shapefile).

        Returns
        -------
        shapefile : GeoPandasDataFrame
            Shapefile dataframe.
        """

        if self.shapefile is None or overwrite:

            # Get latitude and longitude cell boundaries
            if self.lat_bnds is None or self.lon_bnds is None:
                self.create_spatial_bounds()

            # Reshape arrays to create geometry
            aux_b_lat = self.lat_bnds["data"].reshape((self.lat_bnds["data"].shape[0] * self.lat_bnds["data"].shape[1],
                                                       self.lat_bnds["data"].shape[2]))
            aux_b_lon = self.lon_bnds["data"].reshape((self.lon_bnds["data"].shape[0] * self.lon_bnds["data"].shape[1],
                                                       self.lon_bnds["data"].shape[2]))

            # Get polygons from bounds
            geometry = []
            for i in range(aux_b_lon.shape[0]):
                geometry.append(Polygon([(aux_b_lon[i, 0], aux_b_lat[i, 0]),
                                         (aux_b_lon[i, 1], aux_b_lat[i, 1]),
                                         (aux_b_lon[i, 2], aux_b_lat[i, 2]),
                                         (aux_b_lon[i, 3], aux_b_lat[i, 3]),
                                         (aux_b_lon[i, 0], aux_b_lat[i, 0])]))

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

        # Create dataframe containing all points
        fids = self.get_fids()
        centroids_gdf = GeoDataFrame(index=Index(name="FID", data=fids.ravel()), geometry=centroids, crs="EPSG:4326")

        return centroids_gdf
