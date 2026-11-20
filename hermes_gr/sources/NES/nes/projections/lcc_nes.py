#!/usr/bin/env python
# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

"""Lambert Conformal Conic structured grids with CF metadata and native cell geometry."""

from numpy import float64, linspace, array, mean, diff, append, flip, repeat, concatenate, vstack
from geopandas import GeoDataFrame
from pandas import Index
from pyproj import Proj
from copy import deepcopy
from typing import Dict, Any
from shapely.geometry import Polygon, Point
from ..default_nes import Nes

from ._coordinates import (CF_FIELDS, EARTH_FIELDS, build_projection, read_mapping,
                           structured_bounds, select_native_axes, write_native_bounds, resize_native_axes)

from ._coordinates import grid_perimeter

from ._coordinates import validate_creation


class LCCNes(Nes):
    """Represent structured Lambert Conformal Conic grids.

    Notes
    -----

    Creation uses lat_1/lat_2 standard parallels, lon_0/lat_0 projection origins in degrees, nx/ny counts and
    x_0/y_0 cell-edge origins with inc_x/inc_y increments in metres. False offsets are separate optional CF
    parameters. Explicit Earth axes/radius override the legacy default.

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
    _native_axis_names = ('y', 'x')

    def __init__(self, comm=None, path=None, info=False, dataset=None, parallel_method="Y",
                 avoid_first_hours=0, avoid_last_hours=0, first_level=0, last_level=None, create_nes=False,
                 balanced=False, times=None, **kwargs):
        """Initialize the LCCNes class.

        Parameters
        ----------
        comm : mpi4py.MPI.Comm or None, optional
            Communicator for collective operations. None selects MPI.COMM_WORLD.
        path : str or None, optional
            Input/output NetCDF path, as required by the operation.
        info : bool, optional
            Print per-rank processing diagnostics. Default is False.
        dataset : netCDF4.Dataset or None, optional
            Already open input dataset; takes precedence over path when reading.
        parallel_method : {'X', 'Y', 'T'}, optional
            Domain decomposition axis. Structured grids default to Y; station datasets default to X and retain
            their format-specific restrictions.
        avoid_first_hours : int, optional
            Number of hours excluded from the beginning of the input. Default is zero.
        avoid_last_hours : int, optional
            Number of hours excluded from the end of the input. Default is zero.
        first_level : int, optional
            First retained vertical index. Default is zero.
        last_level : int or None, optional
            Last-level selection limit, following base NES slicing. None retains remaining levels.
        create_nes : bool, optional
            Create from geographical parameters if True; otherwise read an existing dataset. Default is False.
        balanced : bool, optional
            Balance the distributed domain. Default is False; chunked output does not support balanced decomposition.
        times : sequence of datetime or None, optional
            Creation time coordinates. Defaults to 1996-12-31 when omitted.
        **kwargs : dict
            Projection-specific creation arguments or documented options forwarded to the underlying operation.

        Notes
        -----
        All ranks of comm must participate in initialization.

        Creation uses lat_1/lat_2 standard parallels, lon_0/lat_0 projection origins in degrees, nx/ny counts
        and x_0/y_0 cell-edge origins with inc_x/inc_y increments in metres. False offsets are separate optional
        CF parameters. Explicit Earth axes/radius override the legacy default.
        """
        if create_nes:
            validate_creation('lcc', kwargs)
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

    @classmethod
    def new(cls, comm=None, path=None, info=False, dataset=None, parallel_method="Y",
            avoid_first_hours=0, avoid_last_hours=0, first_level=0, last_level=None, create_nes=False,
            balanced=False, times=None, **kwargs):
        """Construct a new LCCNes object, preserving subclasses.

        Parameters
        ----------
        comm : mpi4py.MPI.Comm or None, optional
            Communicator for collective operations. None selects MPI.COMM_WORLD.
        path : str or None, optional
            Input/output NetCDF path, as required by the operation.
        info : bool, optional
            Print per-rank processing diagnostics. Default is False.
        dataset : netCDF4.Dataset or None, optional
            Already open input dataset; takes precedence over path when reading.
        parallel_method : {'X', 'Y', 'T'}, optional
            Domain decomposition axis. Structured grids default to Y; station datasets default to X and retain
            their format-specific restrictions.
        avoid_first_hours : int, optional
            Number of hours excluded from the beginning of the input. Default is zero.
        avoid_last_hours : int, optional
            Number of hours excluded from the end of the input. Default is zero.
        first_level : int, optional
            First retained vertical index. Default is zero.
        last_level : int or None, optional
            Last-level selection limit, following base NES slicing. None retains remaining levels.
        create_nes : bool, optional
            Create from geographical parameters if True; otherwise read an existing dataset. Default is False.
        balanced : bool, optional
            Balance the distributed domain. Default is False; chunked output does not support balanced decomposition.
        times : sequence of datetime or None, optional
            Creation time coordinates. Defaults to 1996-12-31 when omitted.
        **kwargs : dict
            Projection-specific creation arguments or documented options forwarded to the underlying operation.

        Returns
        -------
        instance : LCCNes
            New object of the calling class.

        Notes
        -----
        All ranks of comm must participate in initialization.

        Creation uses lat_1/lat_2 standard parallels, lon_0/lat_0 projection origins in degrees, nx/ny counts
        and x_0/y_0 cell-edge origins with inc_x/inc_y increments in metres. False offsets are separate optional
        CF parameters. Explicit Earth axes/radius override the legacy default.
        """

        new = cls(comm=comm, path=path, info=info, dataset=dataset,
                  parallel_method=parallel_method, avoid_first_hours=avoid_first_hours,
                  avoid_last_hours=avoid_last_hours, first_level=first_level, last_level=last_level,
                  create_nes=create_nes, balanced=balanced, times=times, **kwargs)

        return new

    def get_full_y(self) -> Dict[str, Any]:
        """Retrieve the complete Y information.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the complete latitude data and its attributes.
            The dictionary structure is::

                {
                    "data": ndarray,  # Array of latitude values.
                    attr_name: attr_value,  # Latitude attributes.
                    ...
                }
        """
        data = self.comm.bcast(self._full_y, root=0)

        return data

    def get_full_x(self) -> Dict[str, Any]:
        """Retrieve the complete X information.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the complete longitude data and its attributes.
            The dictionary structure is::

                {
                    "data": ndarray,  # Array of longitude values.
                    attr_name: attr_value,  # Longitude attributes.
                    ...
                }
        """
        data = self.comm.bcast(self._full_x, root=0)
        return data

    def set_full_y(self, data: Dict[str, Any]) -> None:
        """Set the complete Y information.

        Parameters
        ----------
        data : Dict[str, Any]
            A dictionary containing the complete latitude data and its attributes.
            The dictionary structure is::

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
        """Set the complete rotated longitude information.

        Parameters
        ----------
        data : Dict[str, Any]
            A dictionary containing the complete longitude data and its attributes.
            The dictionary structure is::

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
        """Expands the latitude and longitude coordinates.

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
        resize_native_axes(self, n_cells, True, is_first=is_first, is_last=is_last)
        return None

    def _contract_coordinates(self, n_cells: int, is_first: bool, is_last: bool) -> None:
        """Contracts latitude and longitude coordinates by removing `n_cells` from the borders.

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
        resize_native_axes(self, n_cells, False, is_first=is_first, is_last=is_last)
        return None

    def _expand_full_coordinates(self, n_cells: int) -> None:
        """Expands the full coordinates if `self.master` is True.

        Parameters
        ----------
        n_cells : int
            Number of cells to expand.

        Returns
        -------
        None
        """
        resize_native_axes(self, n_cells, True, full=True)
        return None

    def _contract_full_coordinates(self, n_cells: int) -> None:
        """Contracts full latitude and longitude coordinates if `self.master` is True.

        Parameters
        ----------
        n_cells : int
            The number of cells to remove from each side.

        Returns
        -------
        None
        """
        resize_native_axes(self, n_cells, False, full=True)
        return None

    # noinspection DuplicatedCode
    def _filter_coordinates_selection(self):
        """Use the selection limits to filter y, x, time, lev, lat, lon, lon_bnds and lat_bnds.

        Notes
        -----

        Collective on the grid communicator: all participating MPI ranks must enter this operation in the same order.
        """

        select_native_axes(self, 'y', 'x')
        super()._filter_coordinates_selection()
        return None

    def _get_pyproj_projection(self):
        """Get projection data as in Pyproj library.

        Returns
        -------
        projection : pyproj.Proj
            Grid projection.
        """

        return build_projection(self)

    def _get_projection_data(self, create_nes, **kwargs):
        """Retrieves projection data based on grid details.

        Parameters
        ----------
        create_nes : bool
            Flag indicating whether to create new object (True) or use existing (False).
        **kwargs : dict
            Additional keyword arguments for specifying projection details.

        Returns
        -------
        projection_data : dict or None
            CF mapping attributes for structured grids; None for the base or station-only layouts.
        """
        if create_nes:
            projection_data = {
                "grid_mapping_name": "lambert_conformal_conic",
                "standard_parallel": [kwargs["lat_1"], kwargs["lat_2"]],
                "longitude_of_central_meridian": kwargs["lon_0"],
                "latitude_of_projection_origin": kwargs["lat_0"],
                "x_0": kwargs["x_0"],
                "y_0": kwargs["y_0"],
                "inc_x": kwargs["inc_x"],
                "inc_y": kwargs["inc_y"],
                "nx": kwargs["nx"],
                "ny": kwargs["ny"],
            }
            earth = {key: kwargs[key] for key in EARTH_FIELDS if key in kwargs}
            if any(
                key in earth for key in ("earth_radius", "semi_major_axis", "semi_minor_axis", "inverse_flattening")
            ):
                for key in ("earth_radius", "semi_major_axis", "semi_minor_axis", "inverse_flattening"):
                    projection_data.pop(key, None)
            projection_data.update(earth)
            for key in ("false_easting", "false_northing", "north_pole_grid_longitude"):
                if key in kwargs:
                    projection_data[key] = kwargs[key]
        else:
            projection_data = read_mapping(
                self, "lambert_conformal_conic", ("Lambert_Conformal", "Lambert_conformal"), required=True
            )
        return projection_data

    # noinspection DuplicatedCode
    def _create_dimensions(self, netcdf):
        """Create "y", "x" and "spatial_nv" dimensions and the super dimensions "lev", "time", "time_nv", "lon"
        and "lat".

        Parameters
        ----------
        netcdf : Dataset
            NetCDF object.

        Notes
        -----

        Collective on the grid communicator: all participating MPI ranks must enter this operation in the same order.
        """

        self.create_spatial_bounds()
        super(LCCNes, self)._create_dimensions(netcdf)
        netcdf.createDimension('y', len(self.get_full_y()['data']))
        netcdf.createDimension('x', len(self.get_full_x()['data']))
        if self.lat_bnds is not None and self.lon_bnds is not None:
            netcdf.createDimension('spatial_nv', 4)
        netcdf.createDimension('native_nv', 2)
        return None

    # noinspection DuplicatedCode
    def _create_dimension_variables(self, netcdf):
        """Create the "y" and "x" variables.

        Parameters
        ----------
        netcdf : Dataset
            NetCDF object.

        Notes
        -----

        Collective on the grid communicator: all participating MPI ranks must enter this operation in the same order.
        """

        super(LCCNes, self)._create_dimension_variables(netcdf)
        full_y = self.get_full_y()
        y = netcdf.createVariable('y', full_y['data'].dtype, ('y',))
        y.long_name = 'y coordinate of projection'
        if 'units' in full_y.keys():
            y.units = full_y['units']
        else:
            y.units = 'm'
        y.standard_name = 'projection_y_coordinate'
        if self.size > 1:
            y.set_collective(True)
        y[:] = full_y['data']
        full_x = self.get_full_x()
        x = netcdf.createVariable('x', full_x['data'].dtype, ('x',))
        x.long_name = 'x coordinate of projection'
        if 'units' in full_x.keys():
            x.units = full_x['units']
        else:
            x.units = 'm'
        x.standard_name = 'projection_x_coordinate'
        if self.size > 1:
            x.set_collective(True)
        x[:] = full_x['data']
        write_native_bounds(self, netcdf, ('y', 'x'))
        return None

    # noinspection DuplicatedCode
    def _create_centre_coordinates(self, **kwargs):
        """Calculate centre latitudes and longitudes from grid details.

        Parameters
        ----------
        **kwargs : dict
            Projection-specific creation arguments or documented options forwarded to the underlying operation.

        Notes
        -----

        Return full centre coordinates on rank zero; native projected geometry is retained separately where applicable.
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
        """Calculate centre latitudes and longitudes from original coordinates and store as 2D arrays.

        Returns
        -------
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
        """Calculate grid edge latitudes and longitudes and get model grid outline.

        Returns
        -------
        grid_edge_lat : dict
            Dictionary with data of grid edge latitudes.
        grid_edge_lon : dict
            Dictionary with data of grid edge longitudes.
        """
        # Get grid resolution
        return grid_perimeter(self, 'y', 'x')

    # noinspection DuplicatedCode
    def create_spatial_bounds(self):
        """Calculate longitude and latitude bounds and set them.

        Notes
        -----

        Collective on the grid communicator: all participating MPI ranks must enter this operation in the same order.
        """

        # Calculate LCC coordinates bounds
        structured_bounds(self, 'y', 'x')
        return None

    @staticmethod
    def _set_var_crs(var):
        """Set the grid_mapping to "Lambert_Conformal".

        Parameters
        ----------
        var : Variable
            netCDF4-python variable object.
        """

        var.grid_mapping = "Lambert_Conformal"
        var.coordinates = "lat lon"

        return None

    def _create_metadata(self, netcdf):
        """Create the "crs" variable for the lambert conformal grid_mapping.

        Parameters
        ----------
        netcdf : Dataset
            netcdf4-python Dataset
        """

        if self.projection_data is not None:
            mapping = netcdf.createVariable('Lambert_Conformal', 'i4')
            mapping.setncatts({key: value for key, value in self.projection_data.items()
                              if key in CF_FIELDS or key == 'nes_projection'})
        return None

    def to_grib2(self, path, grib_keys, grib_template_path, lat_flip=False, info=False):
        """Write output file with grib2 format.

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
        """Create spatial GeoDataFrame (shapefile).

        Parameters
        ----------
        overwrite : bool, optional
            Rebuild cached geometry if True. Default is False.

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
            gdf = GeoDataFrame(index=Index(name="FID", data=fids.ravel()), geometry=geometry, crs=self._geographic_crs)
            self.shapefile = gdf

        else:
            gdf = self.shapefile

        return gdf

    # noinspection DuplicatedCode
    def get_centroids_from_coordinates(self):
        """Get centroids from geographical coordinates.

        Returns
        -------
        centroids_gdf : GeoPandasDataFrame
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
        centroids_gdf = GeoDataFrame(index=Index(name="FID", data=fids.ravel()),
                                     geometry=centroids, crs=self._geographic_crs)

        return centroids_gdf
