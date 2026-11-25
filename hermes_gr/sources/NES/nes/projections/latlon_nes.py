#!/usr/bin/env python
# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

"""Regular latitude-longitude, global and Global Monarch structured grids."""

from numpy import isclose, empty, float64, linspace, meshgrid, mean, diff, append, flip, repeat, concatenate, vstack
import json
from pyproj import Proj
from ..default_nes import Nes

from ._coordinates import (CF_FIELDS, EARTH_FIELDS, build_projection, read_mapping,
                           structured_bounds, select_native_axes, write_native_bounds)

from ._coordinates import validate_creation


class LatLonNes(Nes):
    """Represent regular latitude-longitude, global and Global Monarch grids.

    Notes
    -----

    Creation requires angular increments and regular/global domain arguments accepted by create_nes. Explicit
    Earth attributes are optional; legacy spherical defaults remain available.

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
        """Initialize the LatLonNes class.

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
            Creation time coordinates. None uses the same default time as the public create_nes factory.
        **kwargs : dict
            Projection-specific creation arguments or documented options forwarded to the underlying operation.

        Notes
        -----
        All ranks of comm must participate in initialization.

        Creation requires angular increments and regular/global domain arguments accepted by create_nes.
        Explicit Earth attributes are optional; legacy spherical defaults remain available.
        """

        if create_nes:
            kind = kwargs.get('nes_projection', 'regular' if 'n_lat' in kwargs else 'global')
            validate_creation(kind, kwargs)
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

    @classmethod
    def new(cls, comm=None, path=None, info=False, dataset=None, parallel_method="Y",
            avoid_first_hours=0, avoid_last_hours=0, first_level=0, last_level=None, create_nes=False,
            balanced=False, times=None, **kwargs):
        """Construct a new LatLonNes object, preserving subclasses.

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
            Creation time coordinates. None uses the same default time as the public create_nes factory.
        **kwargs : dict
            Projection-specific creation arguments or documented options forwarded to the underlying operation.

        Returns
        -------
        instance : LatLonNes
            New object of the calling class.

        Notes
        -----
        All ranks of comm must participate in initialization.

        Creation requires angular increments and regular/global domain arguments accepted by create_nes.
        Explicit Earth attributes are optional; legacy spherical defaults remain available.
        """

        new = cls(comm=comm, path=path, info=info, dataset=dataset,
                  parallel_method=parallel_method, avoid_first_hours=avoid_first_hours,
                  avoid_last_hours=avoid_last_hours, first_level=first_level, last_level=last_level,
                  create_nes=create_nes, balanced=balanced, times=times, **kwargs)

        return new

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
        for coord in ['lat', 'lat_bnds', 'lon', 'lon_bnds']:
            expand_left = is_first if "lat" in coord else True
            expand_right = is_last if "lat" in coord else True
            if self.__dict__[coord] is not None:
                self.__dict__[coord]['data'] = self._expand_1d(
                    self.__dict__[coord]['data'], n_cells=n_cells, left=expand_left, right=expand_right
                )
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
        for coord in ['lat', 'lat_bnds', 'lon', 'lon_bnds']:
            contract_left = is_first if "lat" in coord else True
            contract_right = is_last if "lat" in coord else True
            if self.__dict__[coord] is not None:
                self.__dict__[coord]['data'] = self._contract_1d(
                    self.__dict__[coord]['data'], n_cells=n_cells, left=contract_left, right=contract_right
                )
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
        if self.master:
            for coord in ['_full_lat', '_full_lat_bnds', '_full_lon', '_full_lon_bnds']:
                if self.__dict__[coord] is not None:
                    self.__dict__[coord]['data'] = self._expand_1d(
                        self.__dict__[coord]['data'], n_cells=n_cells, left=True, right=True
                    )
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
        if self.master:
            for coord in ['_full_lat', '_full_lat_bnds', '_full_lon', '_full_lon_bnds']:
                if self.__dict__[coord] is not None:
                    self.__dict__[coord]['data'] = self._contract_1d(
                        self.__dict__[coord]['data'], n_cells=n_cells, left=True, right=True
                    )
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
        projection_data : dict
            CF latitude/longitude mapping, Earth definition and available creation geometry.

        Notes
        -----
        Preserve regular/global/Global Monarch geometry during creation. Read referenced CF latitude_longitude
        mappings, retaining the historical crs variable-name fallback. Explicit Earth attributes are preserved;
        only files without an Earth definition use the established NES sphere.
        """
        if create_nes:
            projection_data = {
                "grid_mapping_name": "latitude_longitude",
                "semi_major_axis": self.earth_radius[1],
                "inverse_flattening": 0,
                "inc_lat": kwargs["inc_lat"],
                "inc_lon": kwargs["inc_lon"],
            }
            nes_projection = kwargs.get("nes_projection")
            if nes_projection == "global_monarch":
                lat_cells_each_side = round(90 / float64(projection_data["inc_lat"]))
                lon_cells_each_side = round(180 / float64(projection_data["inc_lon"]))
                if abs(lat_cells_each_side - 90 / float64(projection_data["inc_lat"])) > 1e-09:
                    raise ValueError("inc_lat must divide 90 exactly to create a global_monarch projection")
                if abs(lon_cells_each_side - 180 / float64(projection_data["inc_lon"])) > 1e-09:
                    raise ValueError("inc_lon must divide 180 exactly to create a global_monarch projection")
                projection_data["nes_projection"] = nes_projection
                projection_data["lat_orig"] = -90 - float64(projection_data["inc_lat"]) / 2
                projection_data["lon_orig"] = -180 - float64(projection_data["inc_lon"]) / 2
                projection_data["n_lat"] = int(lat_cells_each_side * 2 + 1)
                projection_data["n_lon"] = int(lon_cells_each_side * 2 + 1)
            elif nes_projection == "global" or not any(
                key in kwargs for key in ("lat_orig", "lon_orig", "n_lat", "n_lon")
            ):
                projection_data["lat_orig"] = -90
                projection_data["lon_orig"] = -180
                projection_data["n_lat"] = int(180 // float64(projection_data["inc_lat"]))
                projection_data["n_lon"] = int(360 // float64(projection_data["inc_lon"]))
            else:
                projection_data["lat_orig"] = kwargs["lat_orig"]
                projection_data["lon_orig"] = kwargs["lon_orig"]
                projection_data["n_lat"] = kwargs["n_lat"]
                projection_data["n_lon"] = kwargs["n_lon"]
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
            projection_data = read_mapping(self, "latitude_longitude", ("crs",), required=False)
            if projection_data is None:
                projection_data = {
                    "grid_mapping_name": "latitude_longitude",
                    "semi_major_axis": self.earth_radius[1],
                    "inverse_flattening": 0.0,
                }
        return projection_data

    def _create_dimensions(self, netcdf):
        """Create "spatial_nv" dimensions and the super dimensions "lev", "time", "time_nv", "lon" and "lat".

        Parameters
        ----------
        netcdf : Dataset
            NetCDF object.

        Notes
        -----

        Collective on the grid communicator: all participating MPI ranks must enter this operation in the same order.
        """

        super(LatLonNes, self)._create_dimensions(netcdf)

        netcdf.createDimension("lon", len(self.get_full_longitudes()["data"]))
        netcdf.createDimension("lat", len(self.get_full_latitudes()["data"]))

        # Create spatial_nv (number of vertices) dimension
        if (self.lat_bnds is not None) and (self.lon_bnds is not None):
            netcdf.createDimension("spatial_nv", 2)

        return None

    def _create_centre_coordinates(self, **kwargs):
        """Calculate centre latitudes and longitudes from grid details.

        Parameters
        ----------
        **kwargs : dict
            Projection-specific creation arguments or documented options forwarded to the underlying operation.

        Returns
        -------
        centre_lat : dict
            Dictionary with data of centre latitudes in 1D
        centre_lon : dict
            Dictionary with data of centre longitudes in 1D

        Notes
        -----

        Return full centre coordinates on rank zero; native projected geometry is retained separately where applicable.
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
        """Calculate longitude and latitude bounds and clip global Monarch edge cells.

        Notes
        -----
        Collective on the grid communicator: all participating MPI ranks must enter this operation in the same order.

        Global Monarch end cells have half the nominal increment, matching their quarter-step centres. Prefer
        existing CF bounds and recover missing increments from creation or persisted canonical metadata. The
        resulting global intervals meet at shared edges without overlaps.
        """

        if self.projection_data.get('nes_projection') != 'global_monarch':
            return super().create_spatial_bounds()
        saved = json.loads(self.global_attrs.get('nes_grid_configuration', '{}')) if self.global_attrs else {}
        for name, lower, upper, axis in (('lat', -90., 90., 'Y'), ('lon', -180., 180., 'X')):
            full = self.get_full_latitudes() if name == 'lat' else self.get_full_longitudes()
            existing = self.get_full_latitudes_boundaries() if name == 'lat' else self.get_full_longitudes_boundaries()
            if existing is not None:
                bounds = existing['data']
            else:
                values = full['data']
                step = self.projection_data.get('inc_' + name, saved.get('axes', {}).get(name, {}).get('step'))
                if step is None:
                    step = 4 * (float(values[0]) - lower)
                if step <= 0:
                    raise ValueError('Global Monarch bounds require a recoverable positive native increment')
                bounds = empty((len(values), 2), dtype=float64)
                bounds[:, 0], bounds[:, 1] = values - step / 2, values + step / 2
                # Polar/end cells have half the nominal width, matching their quarter-step centres.
                if isclose(values[0], lower + step / 4):
                    bounds[0] = (lower, lower + step / 2)
                if isclose(values[-1], upper - step / 4):
                    bounds[-1] = (upper - step / 2, upper)
            info = {'data': bounds}
            if name == 'lat':
                self.set_full_latitudes_boundaries(info)
                self.lat_bnds = self._get_coordinate_values(info, axis, bounds=True)
            else:
                self.set_full_longitudes_boundaries(info)
                self.lon_bnds = self._get_coordinate_values(info, axis, bounds=True)
        return None

    def create_providentia_exp_centre_coordinates(self):
        """Calculate centre latitudes and longitudes from original coordinates and store as 2D arrays.

        Returns
        -------
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
        """Calculate grid edge latitudes and longitudes and get model grid outline.

        Returns
        -------
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
        """Set the grid_mapping to "crs".

        Parameters
        ----------
        var : Variable
            netCDF4-python variable object.
        """

        var.grid_mapping = "crs"
        # var.coordinates = "lat lon"

        return None

    def _create_metadata(self, netcdf):
        """Create the "crs" variable for the rotated latitude longitude grid_mapping.

        Parameters
        ----------
        netcdf : Dataset
            netcdf4-python Dataset.
        """

        if self.projection_data is not None:
            mapping = netcdf.createVariable('crs', 'i4')
            mapping.setncatts({key: value for key, value in self.projection_data.items()
                              if key in CF_FIELDS or key == 'nes_projection'})
        return None

    def to_grib2(self, path, grib_keys, grib_template_path, lat_flip=False, info=False):
        """Write output file with grib2 format.

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
