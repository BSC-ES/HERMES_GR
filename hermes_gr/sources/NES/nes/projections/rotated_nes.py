#!/usr/bin/env python
# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

"""Rotated latitude-longitude grids with CF pole metadata and native angular axes."""

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
from ..default_nes import Nes

from ._coordinates import (CF_FIELDS, EARTH_FIELDS, build_projection, read_mapping,
                           structured_bounds, select_native_axes, write_native_bounds, resize_native_axes)

from ._coordinates import grid_perimeter

from ._coordinates import validate_creation


class RotatedNes(Nes):
    """Represent structured grids in rotated latitude-longitude coordinates.

    Notes
    -----

    Creation requires centre_lat/centre_lon, south_boundary/west_boundary and inc_rlat/inc_rlon in degrees.
    Optional north_pole_grid_longitude and explicit Earth attributes are supported.

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
    _native_axis_names = ('rlat', 'rlon')

    def __init__(self, comm=None, path=None, info=False, dataset=None, parallel_method="Y",
                 avoid_first_hours=0, avoid_last_hours=0, first_level=0, last_level=None, create_nes=False,
                 balanced=False, times=None, **kwargs):
        """Initialize the RotatedNes class.

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

        Creation requires centre_lat/centre_lon, south_boundary/west_boundary and inc_rlat/inc_rlon in degrees.
        Optional north_pole_grid_longitude and explicit Earth attributes are supported.
        """
        if create_nes:
            validate_creation('rotated-nested' if 'parent_grid_path' in kwargs else 'rotated', kwargs)
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

    @classmethod
    def new(cls, comm=None, path=None, info=False, dataset=None, parallel_method="Y",
            avoid_first_hours=0, avoid_last_hours=0, first_level=0, last_level=None, create_nes=False,
            balanced=False, times=None, **kwargs):
        """Construct a new RotatedNes object, preserving subclasses.

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
        instance : RotatedNes
            New object of the calling class.

        Notes
        -----
        All ranks of comm must participate in initialization.

        Creation requires centre_lat/centre_lon, south_boundary/west_boundary and inc_rlat/inc_rlon in degrees.
        Optional north_pole_grid_longitude and explicit Earth attributes are supported.
        """

        new = cls(comm=comm, path=path, info=info, dataset=dataset,
                  parallel_method=parallel_method, avoid_first_hours=avoid_first_hours,
                  avoid_last_hours=avoid_last_hours, first_level=first_level, last_level=last_level,
                  create_nes=create_nes, balanced=balanced, times=times, **kwargs)

        return new

    def get_full_rlat(self) -> Dict[str, Any]:
        """Retrieve the complete rotated latitude information.

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
        data = self.comm.bcast(self._full_rlat, root=0)

        return data

    def get_full_rlon(self) -> Dict[str, Any]:
        """Retrieve the complete rotated longitude information.

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
        data = self.comm.bcast(self._full_rlon, root=0)
        return data

    def set_full_rlat(self, data: Dict[str, Any]) -> None:
        """Set the complete rotated latitude information.

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
            self._full_rlat = data
        return None

    def set_full_rlon(self, data: Dict[str, Any]) -> None:
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
            self._full_rlon = data
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
        """Use the selection limits to filter rlat, rlon, time, lev, lat, lon, lon_bnds and lat_bnds.

        Notes
        -----

        Collective on the grid communicator: all participating MPI ranks must enter this operation in the same order.
        """

        select_native_axes(self, 'rlat', 'rlon')
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

    # noinspection DuplicatedCode
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
                "grid_mapping_name": "rotated_latitude_longitude",
                "grid_north_pole_latitude": 90 - kwargs["centre_lat"],
                "grid_north_pole_longitude": -180 + kwargs["centre_lon"],
                "inc_rlat": kwargs["inc_rlat"],
                "inc_rlon": kwargs["inc_rlon"],
                "south_boundary": kwargs["south_boundary"],
                "west_boundary": kwargs["west_boundary"],
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
            projection_data = read_mapping(self, "rotated_latitude_longitude", ("rotated_pole",), required=True)
        return projection_data

    def _create_dimensions(self, netcdf):
        """Create "rlat", "rlon"  and "spatial_nv" dimensions and the dimensions "lev", "time", "time_nv", "lon"
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
        super(RotatedNes, self)._create_dimensions(netcdf)
        shape = self.get_full_shape()
        netcdf.createDimension('rlon', shape[1])
        netcdf.createDimension('rlat', shape[0])
        if self.lat_bnds is not None and self.lon_bnds is not None:
            netcdf.createDimension('spatial_nv', 4)
            pass
        netcdf.createDimension('native_nv', 2)
        return None

    def _create_dimension_variables(self, netcdf):
        """Create the "rlat" and "rlon" variables.

        Parameters
        ----------
        netcdf : Dataset
            NetCDF object.

        Notes
        -----

        Collective on the grid communicator: all participating MPI ranks must enter this operation in the same order.
        """
        super(RotatedNes, self)._create_dimension_variables(netcdf)
        full_rlat = self.get_full_rlat()
        rlat = netcdf.createVariable('rlat', full_rlat['data'].dtype, ('rlat',))
        rlat.long_name = 'latitude in rotated pole grid'
        if 'units' in full_rlat.keys():
            rlat.units = full_rlat['units']
        else:
            rlat.units = 'degrees'
        rlat.standard_name = 'grid_latitude'
        if self.size > 1:
            rlat.set_collective(True)
        rlat[:] = full_rlat['data']
        full_rlon = self.get_full_rlon()
        rlon = netcdf.createVariable('rlon', full_rlon['data'].dtype, ('rlon',))
        rlon.long_name = 'longitude in rotated pole grid'
        if 'units' in full_rlon.keys():
            rlon.units = full_rlon['units']
        else:
            rlon.units = 'degrees'
        rlon.standard_name = 'grid_longitude'
        if self.size > 1:
            rlon.set_collective(True)
        rlon[:] = full_rlon['data']
        write_native_bounds(self, netcdf, ('rlat', 'rlon'))
        return None

    def _create_rotated_coordinates(self):
        """Calculate rotated latitudes and longitudes from grid details.

        Returns
        -------
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
        """Calculate the unrotated coordinates using the rotated ones.

        Parameters
        ----------
        lon_deg : array
            Rotated longitude coordinate.
        lat_deg : array
            Rotated latitude coordinate.
        lon_min : float
            Minimum value for the longitudes: -180 (-180 to 180) or 0 (0 to 360).

        Returns
        -------
        almd : array
            Unrotated longitudes.
        aphd : array
            Unrotated latitudes.
        """

        # Get centre coordinates
        lon, lat = self._to_geographical.transform(lon_deg, lat_deg)
        return (lon - lon_min) % 360 + lon_min, lat

    def _create_centre_coordinates(self, **kwargs):
        """Calculate centre latitudes and longitudes from grid details.

        Parameters
        ----------
        **kwargs : dict
            Projection-specific creation arguments or documented options forwarded to the underlying operation.

        Returns
        -------
        centre_lat : dict
            Dictionary with data of centre coordinates for latitude in 2D (latitude, longitude).
        centre_lon : dict
            Dictionary with data of centre coordinates for longitude in 2D (latitude, longitude).

        Notes
        -----

        Return full centre coordinates on rank zero; native projected geometry is retained separately where applicable.
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
        return grid_perimeter(self, 'rlat', 'rlon')

    # noinspection DuplicatedCode
    def create_spatial_bounds(self):
        """Calculate longitude and latitude bounds and set them.

        Notes
        -----

        Collective on the grid communicator: all participating MPI ranks must enter this operation in the same order.
        """

        # Calculate rotated coordinates bounds
        structured_bounds(self, 'rlat', 'rlon')
        return None

    @staticmethod
    def _set_var_crs(var):
        """Set the grid_mapping to "rotated_pole".

        Parameters
        ----------
        var : Variable
            netCDF4-python variable object.
        """

        var.grid_mapping = "rotated_pole"
        var.coordinates = "lat lon"

        return None

    def _create_metadata(self, netcdf):
        """Create the "crs" variable for the rotated latitude longitude grid_mapping.

        Parameters
        ----------
        netcdf : Dataset
            netcdf4-python Dataset.
        """

        if self.projection_data is not None:
            mapping = netcdf.createVariable('rotated_pole', 'i4')
            mapping.setncatts({key: value for key, value in self.projection_data.items()
                              if key in CF_FIELDS or key == 'nes_projection'})
        return None

    def to_grib2(self, path, grib_keys, grib_template_path, lat_flip=False, info=False):
        """Write output file with grib2 format.

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
        """Create spatial geodataframe (shapefile).

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

        # Create dataframe cointaining all points
        fids = self.get_fids()
        centroids_gdf = GeoDataFrame(index=Index(name="FID", data=fids.ravel()),
                                     geometry=centroids, crs=self._geographic_crs)

        return centroids_gdf
