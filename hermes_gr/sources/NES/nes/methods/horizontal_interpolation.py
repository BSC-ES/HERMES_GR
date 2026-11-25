# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import os
import sys
from copy import deepcopy
from datetime import datetime
from math import ceil, floor
from warnings import warn

import geopandas as gpd
import rasterio
import shapely
from filelock import FileLock
from mpi4py import MPI
from numpy import (
    arange,
    array,
    array_split,
    asarray,
    bincount,
    column_stack,
    concatenate,
    cos,
    diff,
    dtype,
    empty,
    expand_dims,
    float32,
    float64,
    floating,
    full,
    int32,
    int64,
    isfinite,
    isin,
    isnan,
    issubdtype,
    ma,
    maximum,
    meshgrid,
    nan,
    nansum,
    pad,
    radians,
    reciprocal,
    searchsorted,
    sin,
    sum,
    spacing,
    uint8,
    uint32,
    unique,
    where,
    zeros,
)
from pandas import DataFrame, concat
from pyproj import CRS, Geod, Proj, Transformer
from rasterio import features as rasterio_features
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds
from scipy import spatial

import nes

# CONSTANTS
NEAREST_OPTS = ["NearestNeighbour", "NearestNeighbours", "nn", "NN"]
CONSERVATIVE_OPTS = ["Conservative", "Area_Conservative", "cons", "conservative", "area"]
# Sphere radius (m) of the cylindrical equal-area plane used to measure the conservative areas
CONSERVATIVE_AREA_RADIUS = 6371000.0
PROXY_TYPES = (
    "raster",
    "raster_categorical",
    "raster_categorical_numerical",
    "points",
    "lines",
    "polygons",
    "multigeometry",
)
# Global attribute recording how point proxies were assigned to cells in a
# cached weight matrix. Matrices without the current value are rebuilt.
POINT_OWNERSHIP_ATTRIBUTE = "nes_point_proxy_ownership"
POINT_OWNERSHIP_POLICY = "half_open_native_cells_v1"
COORDINATE_ATTRS = (
    "lat",
    "lon",
    "lat_bnds",
    "lon_bnds",
    "_full_lat",
    "_full_lon",
    "_full_lat_bnds",
    "_full_lon_bnds",
    "rlat",
    "rlon",
    "x",
    "y",
    "_full_rlat",
    "_full_rlon",
    "_full_x",
    "_full_y",
)


def __uses_point_proxies(proxies):
    """Tell whether a proxy chain can assign point features to cells.

    Parameters
    ----------
    proxies : list or None
        Priority-ordered proxy tuples whose first element is the proxy type.

    Returns
    -------
    bool
        True for points and multigeometry proxies.
    """
    return proxies is not None and any(proxy[0] in ("points", "multigeometry") for proxy in proxies)


def __lowest_float_dtype(arrays):
    """
    Get the least precise floating dtype among some arrays.

    Parameters
    ----------
    arrays : iterable
        Arrays (or None) to inspect. Non floating arrays are ignored.

    Returns
    -------
    numpy.dtype
        Floating dtype with the smallest itemsize, float64 if none of the arrays is floating.
    """
    float_dtypes = [
        dtype(arr.dtype)
        for arr in arrays
        if arr is not None and hasattr(arr, "dtype") and issubdtype(arr.dtype, floating)
    ]
    if not float_dtypes:
        return dtype(float64)
    return min(float_dtypes, key=lambda dt: dt.itemsize)


def __lowest_coordinate_dtype(*grids, comm=None):
    """
    Get the least precise floating dtype of the horizontal coordinates and bounds of some grids.

    Grids stored in single precision (e.g. CAMS-REG lat/lon bounds) have edges that differ slightly from
    the same edges computed in double precision (float32(41.7) = 41.70000076), which creates sliver
    overlaps that leak tiny non-zero weights into the neighbouring destination cells. Rounding every
    geometry (grids, proxies and nearest neighbour cell centres) to the least precise input dtype makes
    the shared edges coincide exactly.

    Parameters
    ----------
    *grids : nes.Nes
        Grids whose rank-local, full, geographic, and projected coordinates are inspected.
    comm : MPI.Comm, None
        If given, the dtype is reduced among all its ranks (collective call), so that every rank rounds
        the geometries the same way even if only some of them hold the full coordinates.

    Returns
    -------
    numpy.dtype
        Least precise floating coordinate dtype, float64 if no floating coordinate is found.
    """
    arrays = []
    for grid in grids:
        for attr in COORDINATE_ATTRS:
            coordinate = getattr(grid, attr, None)
            if isinstance(coordinate, dict):
                arrays.append(coordinate.get("data"))
    coord_dtype = __lowest_float_dtype(arrays)
    if comm is not None:
        coord_dtype = min(comm.allgather(coord_dtype), key=lambda dt: dt.itemsize)
    return coord_dtype


def __lowest_interpolation_dtype(data, *grids):
    """Get the least precise floating dtype across a source variable and its input grid coordinates."""
    arrays = [data]
    for grid in grids:
        for attr in COORDINATE_ATTRS:
            coordinate = getattr(grid, attr, None)
            if isinstance(coordinate, dict):
                arrays.append(coordinate.get("data"))
    return __lowest_float_dtype(arrays)


def __round_coordinates(coords, coord_dtype):
    """
    Round lon/lat coordinates to a lower floating precision.

    Values closer to 0 than 1e-9 degrees are set to 0: float32 keeps them (e.g. the 0 degree edges of
    CAMS-REG are stored as -5e-12), so they would not coincide with an exact 0 edge of the other grid.

    Parameters
    ----------
    coords : numpy.ndarray
        (N, 2) array of lon/lat coordinates.
    coord_dtype : numpy.dtype
        Floating dtype to round to.

    Returns
    -------
    numpy.ndarray
        Rounded coordinates, as float64.
    """
    coords = coords.astype(coord_dtype).astype(float64)
    coords[abs(coords) < 1e-9] = 0.0
    return coords


def __apply_coordinate_precision(gdf, coord_dtype=float64):
    """
    Round in place the coordinates of the geometries of a GeoDataFrame to the given floating dtype.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame to round.
    coord_dtype : numpy.dtype
        Floating dtype to round to. Nothing is done for float64 (or wider).

    Returns
    -------
    gpd.GeoDataFrame
        The same GeoDataFrame with its geometries rounded.
    """
    coord_dtype = dtype(coord_dtype)
    if coord_dtype.itemsize >= dtype(float64).itemsize or len(gdf) == 0:
        return gdf
    orig = array(gdf.geometry.values)
    geoms = shapely.transform(orig, lambda coords: __round_coordinates(coords, coord_dtype))
    # Rounding can break tiny or very thin polygons. Only those are repaired, with buffer(0) so that they stay
    # polygonal (make_valid could turn them into collections with lines or points, which the multigeometry
    # proxy would then weight as lines or points). Points, lines and input geometries that were already
    # invalid are left as they are, like in float64.
    broken = (
        shapely.is_valid(orig) & ~shapely.is_valid(geoms) & isin(shapely.get_type_id(geoms), [3, 6])
    )  # Polygon, MultiPolygon
    if broken.any():
        geoms[broken] = shapely.buffer(geoms[broken], 0)
    gdf[gdf.geometry.name] = gpd.GeoSeries(geoms, index=gdf.index, crs=gdf.crs)
    return gdf


def __normalize_proxies(proxy: str, proxy_type: str, proxy_categories=None, proxy_numerical=None) -> list:
    """
    Normalize the proxy arguments into a list of (proxy_type, path, categories, column) tuples.

    The list keeps the user-specified priority order only between separate proxy datasets.
    A multigeometry proxy is treated as an atomic proxy: all point, line and polygon
    geometries inside that file are applied together, with no priority among those geometry
    types. The order of separate proxy datasets still matters, so a list like
    [('polygons', ...), ('points', ...), ('raster', ...)] is applied in that order.

    Parameters
    ----------
    proxy : str, list, None
        Path to a single proxy file or priority-ordered list of (proxy_type, path) or
        (proxy_type, path, categories) or (proxy_type, path, categories, column) tuples.
    proxy_type : str
        Type of the proxy when a single path is given.
    proxy_categories : list, None
        Categories to keep when a single categorical ("raster_categorical" or
        "raster_categorical_numerical") path is given. Ignored otherwise.
    proxy_numerical : str, None
        Path to the numerical raster when a single "raster_categorical_numerical" path is given.
        Ignored otherwise.

    Returns
    -------
    list, None
        List of (proxy_type, path, categories, column) tuples, or None if no proxy is used. The categories
        element is None for every non-categorical proxy type, and for a "raster_categorical_numerical"
        proxy the path element is a (categorical_path, numerical_path) pair.
    """
    if proxy is None:
        return None
    if isinstance(proxy, str):
        if proxy_type == "raster_categorical_numerical":
            path = (proxy, proxy_numerical)
        else:
            path = proxy
        proxies = [(proxy_type, path, proxy_categories, None)]
    else:
        if proxy_type != "raster":
            warn("proxy_type is ignored when proxy is a list of (proxy_type, path) tuples.")
            sys.stderr.flush()
        proxies = []
        for aux_proxy in proxy:
            if not (isinstance(aux_proxy, (tuple, list)) and len(aux_proxy) in (2, 3, 4)):
                raise ValueError(
                    "Each element of the proxy list must be a (proxy_type, path) or "
                    "(proxy_type, path, categories[, column]) tuple."
                )
            aux_categories = aux_proxy[2] if len(aux_proxy) == 3 else None
            if len(aux_proxy) == 4:
                aux_categories = aux_proxy[2]
            aux_column = aux_proxy[3] if len(aux_proxy) == 4 else None
            proxies.append((aux_proxy[0], aux_proxy[1], aux_categories, aux_column))
    for aux_type, aux_path, aux_categories, aux_column in proxies:
        if aux_type not in PROXY_TYPES:
            raise NotImplementedError(
                f"Proxy type {aux_type} is not implemented yet for horizontal interpolations. "
                + "Use 'raster', 'raster_categorical', 'raster_categorical_numerical', 'points', 'lines', "
                "'polygons' or 'multigeometry'"
            )
        if aux_type in ("raster_categorical", "raster_categorical_numerical"):
            if aux_categories is None or len(aux_categories) == 0:
                raise ValueError(f"A '{aux_type}' proxy requires a non-empty list of categories to keep.")
        elif aux_categories is not None:
            warn(f"Categories are only used with categorical proxies; they are ignored for the '{aux_type}' proxy.")
            sys.stderr.flush()
        if aux_column is not None and (
            aux_type not in ("points", "lines", "polygons", "multigeometry")
            or not isinstance(aux_column, str)
            or not aux_column
        ):
            raise ValueError("A per-proxy column must be a non-empty string on a vector proxy.")
        if aux_type == "raster_categorical_numerical" and not (
            isinstance(aux_path, (tuple, list)) and len(aux_path) == 2 and all(aux_path)
        ):
            raise ValueError(
                "A 'raster_categorical_numerical' proxy requires both a categorical mask raster "
                "and a numerical raster path (as proxy + proxy_numerical, or a (cat_path, "
                "num_path) pair in the priority list)."
            )

    if any(aux_type == "multigeometry" for aux_type, _, _, _ in proxies):
        warn(
            "When a multigeometry proxy is used, its point, line and polygon parts are applied "
            "together without any priority between geometry types. Priorities still apply only "
            "between separate proxy datasets in the provided list."
        )
        sys.stderr.flush()

    return proxies


def interpolate_horizontal(
    self,
    dst_grid,
    weight_matrix_path=None,
    kind="NearestNeighbour",
    n_neighbours=4,
    info=False,
    to_providentia=False,
    only_create_wm=False,
    wm=None,
    flux=False,
    keep_nan=False,
    fix_border=False,
    proxy=None,
    proxy_type="raster",
    proxy_categories=None,
    proxy_numerical=None,
    column_name="weight",
):
    """
    Horizontal methods from one grid to another one.

    Parameters
    ----------
    self : nes.Nes
        Source projection Nes Object.
    dst_grid : nes.Nes
        Final projection Nes object.
    weight_matrix_path : str, None
        Path to the weight matrix to read/create.
    kind : str
        Kind of horizontal interpolation. Accepted values: ["NearestNeighbour", "Conservative"].
    n_neighbours : int
        Used if kind == NearestNeighbour. Number of nearest neighbours to interpolate. Default: 4.
    info : bool
        Indicates if you want to print extra info during the methods process.
    to_providentia : bool
        Indicates if we want the interpolated grid in Providentia format.
    only_create_wm : bool
        Indicates if you want to only create the Weight Matrix.
    wm : Nes
        Weight matrix Nes File.
    flux : bool
        Indicates if you want to calculate the weight matrix for flux (per unit area) variables. Only used
        by the Conservative interpolation, with or without proxies: the mass weights are converted into
        flux weights with the cell areas, w_flux = w_mass * A_src / A_dst.
    keep_nan : bool
        Indicates if you want to keep nan values after the interpolation
    fix_border : bool
        Indicates if you want to fix the borders on the NN interpolation by expanding it 1 cell and contracting
        it later.
    proxy : str, list, None
        Proxy to use for the interpolation. Either a single path to a proxy file (its type given by
        proxy_type) or a priority-ordered list of (proxy_type, path) tuples: each source cell uses the
        first proxy that covers it and the cells covered by none fall back to the plain area
        conservative weights. If None, no proxy is used. A "raster_categorical" proxy given inside the
        list carries its categories as a third tuple element: (proxy_type, path, categories).
        Vector entries may carry a fourth element for their own weight column:
        (proxy_type, path, None, column_name).
    proxy_type : str
        Type of proxy to use for interpolation. Accepted values: ["raster", "raster_categorical",
        "raster_categorical_numerical", "points", "lines", "polygons", "multigeometry"]. A "multigeometry"
        proxy is a vector file combining point, line and polygon features: the features of a source cell
        are applied together, with no priority between geometry types. A "raster_categorical" proxy is a
        raster of discrete category values: only the cells whose value is in proxy_categories are kept as
        a presence mask and the mass is redistributed conservatively over them. A
        "raster_categorical_numerical" proxy uses the categorical raster only as a keep-mask and applies
        the values of a second numerical raster (proxy_numerical) as the weights within the kept cells; both
        rasters must have exactly the same grid (CRS, geotransform and shape), otherwise a ValueError is
        raised. Only used when proxy is a single path. Default is "raster".
    proxy_categories : list, None
        Categories to keep when proxy is a single categorical ("raster_categorical" or
        "raster_categorical_numerical") path. Ignored for other proxy types. Required (non-empty) for a
        categorical proxy.
    proxy_numerical : str, None
        Path to the numerical raster when proxy is a single "raster_categorical_numerical" path. Required
        for that proxy type, ignored otherwise.
    column_name : str
        Name of the column holding the feature weights in the vector proxies ("points", "lines",
        "polygons" and "multigeometry"). If the column is not found, all the features get a weight of
        1.0. Ignored for raster proxies. Default is "weight".

    Notes
    -----
    The coordinates of every geometry (grids, proxies and nearest neighbour cell centres) are rounded to the
    least precise floating dtype of the source and destination horizontal coordinates and bounds, so that
    the cell edges of a grid stored in single precision coincide with the destination ones and no sliver
    overlaps are created. A weight matrix read from weight_matrix_path is reused as it is, whatever the
    precision it was created with. The interpolated data are computed in double precision and stored with
    the least precise floating dtype of each source variable and the source/destination horizontal
    coordinates (float64 if none are floating).
    """
    proxies = __normalize_proxies(proxy, proxy_type, proxy_categories, proxy_numerical)

    if info and self.master:
        print("Creating Weight Matrix")

    # Obtain weight matrix
    if self.parallel_method == "T":
        weights, idx = __get_weights_idx_t_axis(
            self,
            dst_grid,
            weight_matrix_path,
            kind,
            n_neighbours,
            only_create_wm,
            wm,
            flux,
            fix_border,
            proxies=proxies,
            column_name=column_name,
            info=info,
        )
    elif self.parallel_method in ["Y", "X"]:
        weights, idx = __get_weights_idx_xy_axis(
            self,
            dst_grid,
            weight_matrix_path,
            kind,
            n_neighbours,
            only_create_wm,
            wm,
            flux,
            fix_border,
            proxies=proxies,
            column_name=column_name,
            info=info,
        )
    else:
        raise NotImplementedError(
            f"Parallel method {self.parallel_method} is not implemented yet for horizontal interpolations. Use 'T'"
        )

    if info and self.master:
        print("Weight Matrix done!")
    if only_create_wm:
        # weights for only_create is the WM NES object
        return weights

    # idx[idx < 0] = nan
    idx = ma.masked_array(idx, mask=idx == -999)
    # idx = array(idx, dtype=float)
    # idx[idx < 0] = nan
    # weights[weights < 0] = nan
    # Empty slots are -999 in a newly created weight matrix and NaN in one read from file
    weights = ma.masked_array(weights, mask=(weights == -999) | isnan(weights))
    # weights = array(weights, dtype=float)
    # weights[weights < 0] = nan

    # Copy NES
    final_dst = dst_grid.copy()

    sys.stdout.flush()
    final_dst.set_communicator(dst_grid.comm)

    # Remove original file information
    final_dst.__ini_path = None
    final_dst.netcdf = None
    final_dst.dataset = None

    # Return final_dst
    final_dst.lev = self.lev
    final_dst.set_full_levels(self.get_full_levels())
    final_dst.time = self.time
    final_dst.set_full_times(self.get_full_times())
    final_dst.hours_start = self.hours_start
    final_dst.hours_end = self.hours_end

    if info and self.master:
        print("Applying weights")
    # Apply weights
    for var_name, var_info in self.variables.items():
        if info and self.master:
            print(f"\t{var_name} horizontal interpolation")
            sys.stdout.flush()
        src_shape = var_info["data"].shape
        if isinstance(dst_grid, nes.PointsNes):
            dst_shape = (src_shape[0], src_shape[1], idx.shape[-1])
        else:
            dst_shape = (src_shape[0], src_shape[1], idx.shape[-2], idx.shape[-1])
        # Creating new variable without data
        final_dst.variables[var_name] = {
            attr_name: attr_value for attr_name, attr_value in var_info.items() if attr_name != "data"
        }
        # Creating empty data
        output_dtype = __lowest_interpolation_dtype(var_info["data"], self, dst_grid)
        final_dst.variables[var_name]["data"] = empty(dst_shape, dtype=output_dtype)
        final_dst.variables[var_name]["dtype"] = output_dtype

        # src_data = var_info["data"].reshape((src_shape[0], src_shape[1], src_shape[2] * src_shape[3]))
        for time in range(dst_shape[0]):
            for lev in range(dst_shape[1]):
                src_aux = __get_src_data(self.comm, var_info["data"][time, lev], idx, self.parallel_method)
                if keep_nan:
                    final_dst.variables[var_name]["data"][time, lev] = sum(weights * src_aux, axis=1)
                else:
                    final_dst.variables[var_name]["data"][time, lev] = nansum(weights * src_aux, axis=1)

        if isinstance(dst_grid, nes.PointsNes):
            # Removing level axis
            if src_shape[1] != 1:
                raise IndexError("Data with vertical levels cannot be interpolated to points")
            final_dst.variables[var_name]["data"] = final_dst.variables[var_name]["data"].reshape(
                (src_shape[0], idx.shape[-1])
            )
            if isinstance(dst_grid, nes.PointsNesGHOST) and not to_providentia:
                final_dst = final_dst.to_points()

    final_dst.global_attrs = self.global_attrs

    if info and self.master:
        print("Formatting")

    if to_providentia:
        # self = experiment to interpolate (regular, rotated, etc.)
        # final_dst = interpolated experiment (points)
        if isinstance(final_dst, nes.PointsNes):
            model_centre_lat, model_centre_lon = self.create_providentia_exp_centre_coordinates()
            grid_edge_lat, grid_edge_lon = self.create_providentia_exp_grid_edge_coordinates()
            final_dst = final_dst.to_providentia(
                model_centre_lon=model_centre_lon,
                model_centre_lat=model_centre_lat,
                grid_edge_lon=grid_edge_lon,
                grid_edge_lat=grid_edge_lat,
            )
        else:
            msg = "The final projection must be points to interpolate an experiment and get it in Providentia format."
            warn(msg)
            sys.stderr.flush()
    else:
        # Convert dimensions (time, lev, lat, lon) or (time, lat, lon) to (time, station) for interpolated variables
        # and reshape data
        if isinstance(final_dst, nes.PointsNes):
            for var_name, var_info in final_dst.variables.items():
                if len(var_info["dimensions"]) != len(var_info["data"].shape):
                    final_dst.variables[var_name]["dimensions"] = ("time", "station")

    return final_dst


def __get_src_data(comm, var_data, idx, parallel_method):
    """
    To obtain the needed src data to interpolate.

    Parameters
    ----------
    comm : MPI.Comm.
        MPI communicator.
    var_data : array
        Rank source data.
    idx : array
        Index of the needed data in a 2D flatten way.
    parallel_method: str
        Source parallel method.

    Returns
    -------
    array
        Flatten source needed data.
    """

    if parallel_method == "T":
        var_data = var_data.flatten()
    else:
        var_data = comm.gather(var_data, root=0)
        if comm.Get_rank() == 0:
            if parallel_method == "Y":
                axis = 0
            elif parallel_method == "X":
                axis = 1
            else:
                raise NotImplementedError(parallel_method)
            var_data = concatenate(var_data, axis=axis)
            var_data = var_data.flatten()

        var_data = comm.bcast(var_data, root=0)

    var_data = pad(var_data, [1, 1], "constant", constant_values=nan).take(idx + 1, mode="clip")

    return var_data


def __get_weights_idx_t_axis(
    self,
    dst_grid,
    weight_matrix_path,
    kind,
    n_neighbours,
    only_create,
    wm,
    flux,
    fix_border,
    proxies=None,
    column_name="weight",
    info=False,
):
    """Build or read horizontal interpolation weights for time decomposition.

    Parameters
    ----------
    self : nes.Nes
        Source grid distributed along time.
    dst_grid : nes.Nes
        Destination grid with matching time decomposition.
    weight_matrix_path : str or None
        Optional path for reading or caching a weight matrix.
    kind : str
        Supported nearest-neighbour or conservative interpolation selector.
    n_neighbours : int
        Number of neighbours for nearest-neighbour interpolation.
    only_create : bool
        Return the weight-matrix object without extracting weight/index arrays.
    wm : nes.Nes or None
        Previously prepared weight-matrix object, if supplied.
    flux : bool
        Apply the source/destination area ratio for conservative intensive quantities.
    fix_border : bool
        Temporarily extend nearest-neighbour source borders when creating weights.
    proxies : list, None
        Priority-ordered list of (proxy_type, path) tuples to use for the interpolation.
        If None, no proxy is used.
    column_name : str
        Name of the column holding the feature weights in the vector proxies.
    info : bool
        Indicates if you want to print progress info during the weight matrix creation.

    Returns
    -------
    weights : numpy.ndarray or nes.Nes or bool
        Full weights on each rank, or the weight-matrix object when only_create is True. In
        the latter case non-master ranks may receive the existing True sentinel.
    idx : numpy.ndarray or None
        Source-cell indices matching weights, or None when only_create is True.

    Raises
    ------
    NotImplementedError
        The requested interpolation selector is unsupported.

    Notes
    -----
    Collective operation on the original source communicator. Nearest-neighbour weights are
    constructed on rank zero only, using MPI.COMM_SELF copies to isolate geometry helper
    collectives. Conservative weights are constructed collectively by all ranks on the original
    communicator. Arrays are then broadcast through the original communicator. The source and
    destination communicators remain unchanged.
    """
    weight_matrix = None
    weight_src, weight_dst = self, dst_grid
    if self.master and self.size > 1 and kind in NEAREST_OPTS:
        # NN weights are built on rank zero only. Geometry helpers use collectives,
        # so they need a local communicator rather than the distribution one.
        weight_src = self.copy(new_comm=MPI.COMM_SELF)
        weight_dst = dst_grid.copy(new_comm=MPI.COMM_SELF)

    if wm is not None:
        weight_matrix = wm

    elif weight_matrix_path is not None:
        with FileLock(weight_matrix_path + f"{self.rank:03d}.lock"):
            if os.path.isfile(weight_matrix_path):
                if self.master:
                    weight_matrix = __read_weight_matrix(weight_matrix_path, comm=MPI.COMM_SELF)
                else:
                    weight_matrix = True
                stale = None
                if kind in CONSERVATIVE_OPTS and __uses_point_proxies(proxies):
                    if self.master:
                        attrs = weight_matrix.global_attrs or {}
                        stale = attrs.get(POINT_OWNERSHIP_ATTRIBUTE) != POINT_OWNERSHIP_POLICY
                    stale = self.comm.bcast(stale, root=0)
                if stale:
                    if self.master:
                        warn(
                            f"The weight matrix {weight_matrix_path} was created with an older point proxy "
                            + "ownership policy. Re-calculating it without overwriting the file; delete the file "
                            + "to cache the new weights."
                        )
                        sys.stderr.flush()
                    # All ranks must call it: the intersections are computed in parallel with MPI
                    weight_matrix = __create_area_conservative_weight_matrix(
                        self, dst_grid, flux=flux, proxies=proxies, column_name=column_name, info=info
                    )
                if kind in NEAREST_OPTS:
                    if self.master:
                        if len(weight_matrix.lev["data"]) != n_neighbours:
                            warn(
                                "The selected weight matrix does not have the same number of nearest neighbours."
                                + "Re-calculating again but not saving it."
                            )
                            sys.stderr.flush()
                            weight_matrix = __create_nn_weight_matrix(
                                weight_src, weight_dst, n_neighbours=n_neighbours, fix_border=fix_border, info=info
                            )
                    else:
                        weight_matrix = True

            else:
                if kind in NEAREST_OPTS:
                    if self.master:
                        weight_matrix = __create_nn_weight_matrix(
                            weight_src,
                            weight_dst,
                            n_neighbours=n_neighbours,
                            wm_path=weight_matrix_path,
                            fix_border=fix_border,
                            info=info,
                        )
                    else:
                        weight_matrix = True
                elif kind in CONSERVATIVE_OPTS:
                    # All ranks must call it: the intersections are computed in parallel with MPI
                    weight_matrix = __create_area_conservative_weight_matrix(
                        self,
                        dst_grid,
                        wm_path=weight_matrix_path,
                        flux=flux,
                        proxies=proxies,
                        column_name=column_name,
                        info=info,
                    )
                else:
                    raise NotImplementedError(kind)

        if os.path.exists(weight_matrix_path + f"{self.rank:03d}.lock"):
            os.remove(weight_matrix_path + f"{self.rank:03d}.lock")
    else:
        if kind in NEAREST_OPTS:
            if self.master:
                weight_matrix = __create_nn_weight_matrix(
                    weight_src, weight_dst, n_neighbours=n_neighbours, fix_border=fix_border, info=info
                )
            else:
                weight_matrix = True
        elif kind in CONSERVATIVE_OPTS:
            # All ranks must call it: the intersections are computed in parallel with MPI
            weight_matrix = __create_area_conservative_weight_matrix(
                self, dst_grid, flux=flux, proxies=proxies, column_name=column_name, info=info
            )
        else:
            raise NotImplementedError(kind)

    if only_create:
        return weight_matrix, None

    if self.master:
        if kind in NEAREST_OPTS:
            # Normalize to 1
            weights = array(
                array(weight_matrix.variables["weight"]["data"], dtype=float64)
                / array(weight_matrix.variables["weight"]["data"], dtype=float64).sum(axis=1),
                dtype=float64,
            )
        else:
            weights = array(weight_matrix.variables["weight"]["data"], dtype=float64)
        idx = array(weight_matrix.variables["idx"]["data"][0], dtype=int)
    else:
        weights = None
        idx = None

    weights = self.comm.bcast(weights, root=0)
    idx = self.comm.bcast(idx, root=0)

    return weights, idx


# noinspection DuplicatedCode
def __get_weights_idx_xy_axis(
    self,
    dst_grid,
    weight_matrix_path,
    kind,
    n_neighbours,
    only_create,
    wm,
    flux,
    fix_border,
    proxies=None,
    column_name="weight",
    info=False,
):
    """
    To obtain the weights and source data index through the X or Y axis.

    Parameters
    ----------
    self : nes.Nes
        Source projection Nes Object.
    dst_grid : nes.Nes
        Final projection Nes object.
    weight_matrix_path : str, None
        Path to the weight matrix to read/create.
    kind : str
        Kind of horizontal interpolation. Accepted values: ["NearestNeighbour", "Conservative"].
    n_neighbours : int
        Used if kind == NearestNeighbour. Number of nearest neighbours to interpolate. Default: 4.
    only_create : bool
        Indicates if you want to only create the Weight Matrix.
    wm : Nes
        Weight matrix Nes File.
    flux : bool
        Indicates if you want to calculate the weight matrix for flux variables.
    fix_border : bool
        Indicates if you want to fix the borders on the NN interpolation by expanding it 1 cell and contracting
        it later.
    proxies : list, None
        Priority-ordered list of (proxy_type, path) tuples to use for the interpolation.
        If None, no proxy is used.
    column_name : str
        Name of the column holding the feature weights in the vector proxies.
    info : bool
        Indicates if you want to print progress info during the weight matrix creation.

    Returns
    -------
    tuple
        Weights and source data index.
    """
    weight_matrix = None

    if isinstance(dst_grid, nes.PointsNes) and weight_matrix_path is not None:
        if self.master:
            warn("To point weight matrix cannot be saved.")
            sys.stderr.flush()
        weight_matrix_path = None

    if wm is not None:
        weight_matrix = wm

    elif weight_matrix_path is not None:
        with FileLock(weight_matrix_path + f"{self.rank:03d}.lock"):
            if os.path.isfile(weight_matrix_path):
                if self.master:
                    weight_matrix = __read_weight_matrix(weight_matrix_path, comm=MPI.COMM_SELF)
                else:
                    weight_matrix = True
                stale = None
                if kind in CONSERVATIVE_OPTS and __uses_point_proxies(proxies):
                    if self.master:
                        attrs = weight_matrix.global_attrs or {}
                        stale = attrs.get(POINT_OWNERSHIP_ATTRIBUTE) != POINT_OWNERSHIP_POLICY
                    stale = self.comm.bcast(stale, root=0)
                if stale:
                    if self.master:
                        warn(
                            f"The weight matrix {weight_matrix_path} was created with an older point proxy "
                            + "ownership policy. Re-calculating it without overwriting the file; delete the file "
                            + "to cache the new weights."
                        )
                        sys.stderr.flush()
                    # All ranks must call it: the intersections are computed in parallel with MPI
                    weight_matrix = __create_area_conservative_weight_matrix(
                        self, dst_grid, flux=flux, proxies=proxies, column_name=column_name, info=info
                    )
                if kind in NEAREST_OPTS:
                    if self.master:
                        if len(weight_matrix.lev["data"]) != n_neighbours:
                            warn(
                                "The selected weight matrix does not have the same number of nearest neighbours."
                                + "Re-calculating again but not saving it."
                            )
                            sys.stderr.flush()
                            weight_matrix = __create_nn_weight_matrix(
                                self, dst_grid, n_neighbours=n_neighbours, fix_border=fix_border, info=info
                            )
                    else:
                        weight_matrix = True
            else:
                if kind in NEAREST_OPTS:
                    if self.master:
                        weight_matrix = __create_nn_weight_matrix(
                            self,
                            dst_grid,
                            n_neighbours=n_neighbours,
                            wm_path=weight_matrix_path,
                            fix_border=fix_border,
                            info=info,
                        )
                    else:
                        weight_matrix = True
                elif kind in CONSERVATIVE_OPTS:
                    weight_matrix = __create_area_conservative_weight_matrix(
                        self,
                        dst_grid,
                        wm_path=weight_matrix_path,
                        flux=flux,
                        proxies=proxies,
                        column_name=column_name,
                        info=info,
                    )
                else:
                    raise NotImplementedError(kind)

        if os.path.exists(weight_matrix_path + f"{self.rank:03d}.lock"):
            os.remove(weight_matrix_path + f"{self.rank:03d}.lock")
    else:
        if kind in NEAREST_OPTS:
            if self.master:
                weight_matrix = __create_nn_weight_matrix(
                    self, dst_grid, n_neighbours=n_neighbours, fix_border=fix_border, info=info
                )
            else:
                weight_matrix = True
        elif kind in CONSERVATIVE_OPTS:
            weight_matrix = __create_area_conservative_weight_matrix(
                self, dst_grid, flux=flux, proxies=proxies, column_name=column_name, info=info
            )
        else:
            raise NotImplementedError(kind)

    if only_create:
        return weight_matrix, None

    # Normalize to 1
    if self.master:
        if kind in NEAREST_OPTS:
            weights = array(
                array(weight_matrix.variables["weight"]["data"], dtype=float64)
                / array(weight_matrix.variables["weight"]["data"], dtype=float64).sum(axis=1),
                dtype=float64,
            )
        else:
            weights = array(weight_matrix.variables["weight"]["data"], dtype=float64)
        idx = array(weight_matrix.variables["idx"]["data"][0], dtype=int64)
    else:
        weights = None
        idx = None

    weights = dst_grid.comm.bcast(weights, root=0)
    idx = dst_grid.comm.bcast(idx, root=0)

    # if isinstance(dst_grid, nes.PointsNes):
    # print("weights 1 ->", weights.shape)
    # print("idx 1 ->", idx.shape)
    #     weights = weights[:, dst_grid.write_axis_limits["x_min"]:dst_grid.write_axis_limits["x_max"]]
    #     idx = idx[dst_grid.write_axis_limits["x_min"]:dst_grid.write_axis_limits["x_max"]]
    # else:
    weights = weights[
        :,
        :,
        dst_grid.write_axis_limits["y_min"]:dst_grid.write_axis_limits["y_max"],
        dst_grid.write_axis_limits["x_min"]:dst_grid.write_axis_limits["x_max"],
    ]
    idx = idx[
        :,
        dst_grid.write_axis_limits["y_min"]:dst_grid.write_axis_limits["y_max"],
        dst_grid.write_axis_limits["x_min"]:dst_grid.write_axis_limits["x_max"],
    ]
    # print("weights 2 ->", weights.shape)
    # print("idx 2 ->", idx.shape)

    return weights, idx


def __read_weight_matrix(weight_matrix_path, comm=None, parallel_method="T"):
    """
    Read weight matrix.

    Parameters
    ----------
    weight_matrix_path : str
        Path of the weight matrix.
    comm : MPI.Comm
        A Communicator to read the weight matrix.
    parallel_method : str
        Nes parallel method to read the weight matrix.

    Returns
    -------
    nes.Nes
        Weight matrix.
    """

    weight_matrix = nes.open_netcdf(path=weight_matrix_path, comm=comm, parallel_method=parallel_method, balanced=True)
    weight_matrix.load()

    # In previous versions of NES weight was called inverse_dists
    if "inverse_dists" in weight_matrix.variables:
        weight_matrix.variables["weight"] = weight_matrix.variables["inverse_dists"]

    weight_matrix.variables["weight"]["data"][weight_matrix.variables["weight"]["data"] <= 0] = nan
    # idx holds 0-based source cell indices (0 is a valid cell); empty slots are filled with -999
    weight_matrix.variables["weight"]["data"][weight_matrix.variables["idx"]["data"] < 0] = nan

    return weight_matrix


# noinspection DuplicatedCode,PyProtectedMember
def __create_nn_weight_matrix(self, dst_grid, n_neighbours=4, wm_path=None, info=False, fix_border=False):
    """
    To create the weight matrix with the nearest neighbours method.

    Parameters
    ----------
    self : nes.Nes
        Source projection Nes Object.
    dst_grid : nes.Nes
        Final projection Nes object.
    n_neighbours : int
        Used if kind == NearestNeighbour. Number of nearest neighbours to interpolate. Default: 4.
    wm_path : str
        Path where write the weight matrix.
    info: bool
        Indicates if you want to print extra info during the methods process.
    fix_border : bool
        Indicates if you want to fix the borders on the NN interpolation by expanding it 1 cell and contracting
        it later.

    Returns
    -------
    nes.Nes
        Weight matrix.
    """
    # Inspected before expanding the destination, which recomputes its coordinates
    coord_dtype = __lowest_coordinate_dtype(self, dst_grid)
    if fix_border:
        dst_grid.expand(n_cells=1)
    # Only master is here.
    if info and self.master:
        print(f"\tCreating Nearest Neighbour Weight Matrix with {n_neighbours} neighbours")
        sys.stdout.flush()
    # Source
    src_lat = array(self._full_lat["data"], dtype=coord_dtype)
    src_lon = array(self._full_lon["data"], dtype=coord_dtype)

    # 1D to 2D coordinates
    if len(src_lon.shape) == 1:
        src_lon, src_lat = meshgrid(src_lon, src_lat)

    # Destination
    dst_lat = array(dst_grid._full_lat["data"], dtype=coord_dtype)
    dst_lon = array(dst_grid._full_lon["data"], dtype=coord_dtype)

    if isinstance(dst_grid, nes.PointsNes):
        dst_lat = expand_dims(dst_lat, axis=0)
        dst_lon = expand_dims(dst_lon, axis=0)
    else:
        # 1D to 2D coordinates
        if len(dst_lon.shape) == 1:
            dst_lon, dst_lat = meshgrid(dst_lon, dst_lat)

    # calculate N nearest neighbour inverse distance weights (and indices)
    # from gridcells centres of model 1 to each grid cell centre of model 2
    # model geographic longitude/latitude coordinates are first converted
    # to cartesian ECEF (Earth Centred, Earth Fixed) coordinates, before
    # calculating distances.

    # src_mod_xy = lon_lat_to_cartesian(src_lon.flatten(), src_lat.flatten())
    # dst_mod_xy = lon_lat_to_cartesian(dst_lon.flatten(), dst_lat.flatten())

    src_mod_xy = __lon_lat_to_cartesian_ecef(src_lon.flatten(), src_lat.flatten())
    dst_mod_xy = __lon_lat_to_cartesian_ecef(dst_lon.flatten(), dst_lat.flatten())

    # generate KDtree using model 1 coordinates (i.e. the model grid you are
    # interpolating from)
    src_tree = spatial.cKDTree(src_mod_xy)

    # get n-neighbour nearest distances/indices (ravel form) of model 1 grid cell
    # centres from each model 2 grid cell centre

    dists, idx = src_tree.query(dst_mod_xy, k=n_neighbours)
    # self.nearest_neighbour_inds = \
    #     column_stack(unravel_index(idx, lon.shape))

    weight_matrix = dst_grid.copy(new_comm=MPI.COMM_SELF)
    weight_matrix.time = [datetime(year=2000, month=1, day=1, hour=0, second=0, microsecond=0)]
    weight_matrix._full_time = [datetime(year=2000, month=1, day=1, hour=0, second=0, microsecond=0)]
    weight_matrix._full_time_bnds = None
    weight_matrix.time_bnds = None
    weight_matrix.last_level = None
    weight_matrix.first_level = 0
    weight_matrix.hours_start = 0
    weight_matrix.hours_end = 0

    # take the reciprocals of the nearest neighbours distances
    dists[dists < 1] = 1
    inverse_dists = reciprocal(dists)

    inverse_dists_transf = inverse_dists.T.reshape((1, n_neighbours, dst_lon.shape[0], dst_lon.shape[1]))
    weight_matrix.variables["weight"] = {"data": inverse_dists_transf, "units": "m"}
    idx_transf = idx.T.reshape((1, n_neighbours, dst_lon.shape[0], dst_lon.shape[1]))
    weight_matrix.variables["idx"] = {"data": idx_transf, "units": ""}
    weight_matrix.lev = {"data": arange(inverse_dists_transf.shape[1]), "units": ""}
    weight_matrix._full_lev = {"data": arange(inverse_dists_transf.shape[1]), "units": ""}

    if fix_border:
        weight_matrix.contract(n_cells=1)

    if wm_path is not None:
        weight_matrix.to_netcdf(wm_path, serial=True)

    return weight_matrix


def __pixel_window(dataset, min_x, min_y, max_x, max_y):
    """
    Reading window of whole raster pixels covering the given bounds, clipped to the raster extent.

    rasterio reads a fractional window resampled to whole pixels and crops a window exceeding the raster
    extent, while window_transform() keeps the requested origin: the polygonized pixels would then be
    shifted. Snapping the window outwards to the pixel edges and clipping it keeps every pixel in place.

    Parameters
    ----------
    dataset : rasterio.io.DatasetReader
        Open raster.
    min_x, min_y, max_x, max_y : float
        Bounds to cover, in the raster CRS.

    Returns
    -------
    rasterio.windows.Window, None
        Window of whole pixels, or None if the bounds do not overlap the raster.
    """
    window = from_bounds(min_x, min_y, max_x, max_y, dataset.transform)
    col_start, row_start = max(floor(window.col_off), 0), max(floor(window.row_off), 0)
    col_end = min(ceil(window.col_off + window.width), dataset.width)
    row_end = min(ceil(window.row_off + window.height), dataset.height)
    if col_end <= col_start or row_end <= row_start:
        return None
    return Window(col_start, row_start, col_end - col_start, row_end - row_start)


def __reading_bounds(dst_shp, src_shp, crs):
    """
    Bounds (in crs) of the proxy region to read: the destination grid extent, widened to the full
    rank-local source cells that it touches.

    Reading the complete source cells keeps the proxy normalization of the cells cut by the destination
    domain independent of the domain size. The source cells are selected in the grids CRS (shared by
    dst_shp and src_shp) and only the resulting box is transformed to crs. An empty box is returned when
    no source cell touches the destination grid.
    """
    bounds = dst_shp.total_bounds
    if src_shp is not None:
        touching = src_shp.iloc[src_shp.sindex.query(shapely.geometry.box(*bounds), predicate="intersects")]
        # NaN-skipping min/max: a single invalid cell must not invalidate the whole box
        cells = touching.bounds
        bounds = [cells["minx"].min(), cells["miny"].min(), cells["maxx"].max(), cells["maxy"].max()]
    if not isfinite(bounds).all():
        return 0.0, 0.0, 0.0, 0.0
    return transform_bounds(dst_shp.crs, crs, *bounds)


def __read_raster_proxy_as_gdf(path, dst_shp, src_shp=None, coord_dtype=float64):
    """
    Read the proxy raster and polygonize it (one polygon per pixel) into a GeoDataFrame.

    The raster is windowed with the destination grid bounds and, if src_shp is provided, also with
    the rank-local source grid bounds, so that each process only reads and polygonizes the portion
    of the raster that its source cells can intersect. Only the pixels with a valid positive value are
    returned: nodata, NaN and non-positive pixels carry no proxy weight.

    Parameters
    ----------
    path : str
        Path to the proxy file.
    dst_shp : geopandas.GeoDataFrame
        Destination shapefile.
    src_shp : geopandas.GeoDataFrame, optional
        Source shapefile. Default is None.
    coord_dtype : numpy.dtype
        Floating dtype the geometry coordinates are rounded to (see __lowest_coordinate_dtype).

    Returns
    -------
    geopandas.GeoDataFrame
        Polygonized proxy raster.
    """

    dst_crs = dst_shp.crs
    with rasterio.open(path) as src:
        # Compute the reading window bounds (in the raster CRS)
        min_x, min_y, max_x, max_y = __reading_bounds(dst_shp, src_shp, src.crs)

        if min_x >= max_x or min_y >= max_y:
            # The rank-local source cells do not overlap the destination grid
            return gpd.GeoDataFrame({"value": []}, geometry=[], crs=src.crs).to_crs(dst_crs)

        my_window = __pixel_window(src, min_x, min_y, max_x, max_y)
        if my_window is None:
            # The reading window falls outside the raster extent
            return gpd.GeoDataFrame({"value": []}, geometry=[], crs=src.crs).to_crs(dst_crs)

        # Masked read so that the raster nodata pixels (e.g. land in a ship density map) are identified
        data = src.read(1, window=my_window, masked=True)
        if data.size == 0:
            # The reading window falls outside the raster extent
            return gpd.GeoDataFrame({"value": []}, geometry=[], crs=src.crs).to_crs(dst_crs)

        values = data.astype(float64).filled(nan)
        # Only the pixels with a positive value carry proxy weight. Nodata, NaN and non-positive pixels are
        # not polygonized, so a source cell without valid pixels falls back to the lower priority proxies
        # or to the plain area weights instead of spreading its mass over the pixels without proxy.
        keep_mask = isfinite(values) & (values > 0)
        if not keep_mask.any():
            return gpd.GeoDataFrame({"value": []}, geometry=[], crs=src.crs).to_crs(dst_crs)

        # Polygonize a unique id per pixel (one polygon per pixel) and look up the original value by id
        pixel_id = arange(values.size, dtype=int32).reshape(values.shape)
        shape_gen = (
            (shapely.geometry.shape(s), int(v))
            for s, v in rasterio_features.shapes(pixel_id, mask=keep_mask, transform=src.window_transform(my_window))
        )
        proxy_shp = gpd.GeoDataFrame(dict(zip(["geometry", "pixel_id"], zip(*shape_gen))), crs=src.crs)
        proxy_shp["value"] = array(values.ravel()[proxy_shp["pixel_id"].values], dtype=float32)
        proxy_shp.drop(columns="pixel_id", inplace=True)

    # Reproject data
    proxy_shp = proxy_shp.to_crs(dst_crs)
    return __apply_coordinate_precision(proxy_shp, coord_dtype)


def __read_raster_categorical_proxy_as_gdf(path, dst_shp, category_list, src_shp=None, coord_dtype=float64):
    """
    Read a categorical proxy raster and polygonize the cells of the requested categories into a GeoDataFrame.

    Only the pixels whose value is in category_list are kept: they are collapsed into a single presence
    value (1) and dissolved into polygons, so the proxy acts as a binary mask. The raster is windowed
    with the destination grid bounds and, if src_shp is provided, also with the rank-local source grid
    bounds, so that each process only reads and polygonizes the portion it can intersect.

    Parameters
    ----------
    path : str
        Path to the categorical proxy raster file.
    dst_shp : geopandas.GeoDataFrame
        Destination shapefile.
    category_list : list
        Category values to keep. Every other value is discarded.
    src_shp : geopandas.GeoDataFrame, optional
        Source shapefile. Default is None.
    coord_dtype : numpy.dtype
        Floating dtype the geometry coordinates are rounded to (see __lowest_coordinate_dtype).

    Returns
    -------
    geopandas.GeoDataFrame
        Polygonized presence mask of the requested categories.
    """
    dst_crs = dst_shp.crs
    with rasterio.open(path) as src:
        # Compute the reading window bounds (in the raster CRS)
        min_x, min_y, max_x, max_y = __reading_bounds(dst_shp, src_shp, src.crs)

        if min_x >= max_x or min_y >= max_y:
            # The rank-local source cells do not overlap the destination grid
            return gpd.GeoDataFrame({"value": []}, geometry=[], crs=src.crs).to_crs(dst_crs)

        my_window = __pixel_window(src, min_x, min_y, max_x, max_y)
        if my_window is None:
            # The reading window falls outside the raster extent
            return gpd.GeoDataFrame({"value": []}, geometry=[], crs=src.crs).to_crs(dst_crs)
        data = src.read(1, window=my_window, masked=False)
        if data.size == 0:
            # The reading window falls outside the raster extent
            return gpd.GeoDataFrame({"value": []}, geometry=[], crs=src.crs).to_crs(dst_crs)

        # Keep only the requested categories and collapse them into a single presence value
        keep_mask = isin(data, category_list)
        if not keep_mask.any():
            # None of the requested categories are present in this window
            return gpd.GeoDataFrame({"value": []}, geometry=[], crs=src.crs).to_crs(dst_crs)

        binary = where(keep_mask, 1, 0).astype(uint8)
        # Dissolve the kept pixels into polygons; the mask restricts the polygonization to them
        shape_gen = (
            (shapely.geometry.shape(s), v)
            for s, v in rasterio_features.shapes(binary, mask=keep_mask, transform=src.window_transform(my_window))
        )
        proxy_shp = gpd.GeoDataFrame(dict(zip(["geometry", "value"], zip(*shape_gen))), crs=src.crs)
        proxy_shp["value"] = array(proxy_shp["value"], dtype=float32)

    # Reproject data
    proxy_shp = proxy_shp.to_crs(dst_crs)
    return __apply_coordinate_precision(proxy_shp, coord_dtype)


def __check_matching_rasters(cat_path, num_path):
    """
    Check that the numerical raster has exactly the same grid as the categorical raster.

    The numerical raster is never reprojected or resampled: both rasters must have the same CRS, geotransform
    (origin, pixel size and orientation) and shape, so that every numerical pixel matches one categorical pixel.

    Parameters
    ----------
    cat_path : str
        Path to the categorical mask raster file.
    num_path : str
        Path to the numerical raster file.

    Raises
    ------
    ValueError
        If the CRS, geotransform or shape of the rasters differ.
    """
    with rasterio.open(cat_path) as cat_src, rasterio.open(num_path) as num_src:
        cat_grid = (cat_src.crs, cat_src.transform, cat_src.shape)
        num_grid = (num_src.crs, num_src.transform, num_src.shape)
    if cat_grid != num_grid:
        raise ValueError(
            f"The numerical raster {num_path} must have exactly the same grid (CRS, geotransform and shape) as "
            f"the categorical raster {cat_path}: no reprojection or resampling is applied. "
            f"Categorical: {cat_grid}; numerical: {num_grid}."
        )


def __read_raster_categorical_numerical_proxy_as_gdf(
    cat_path, num_path, dst_shp, category_list, src_shp=None, coord_dtype=float64
):
    """
    Read a categorical mask raster and a numerical raster, keeping the numerical values only where the
    categorical raster matches the requested categories and the numerical values are valid and positive,
    and polygonize one polygon per pixel into a GeoDataFrame.

    The categorical raster selects which pixels to keep (its value must be in category_list); the kept
    pixels then take the positive, non-nodata value of the numerical raster, which drives the conservative
    redistribution. The numerical raster must have exactly the same grid as the categorical raster (see
    __check_matching_rasters): it is read over the same window, without any resampling or reprojection.

    Parameters
    ----------
    cat_path : str
        Path to the categorical mask raster file.
    num_path : str
        Path to the numerical raster file.
    dst_shp : geopandas.GeoDataFrame
        Destination shapefile.
    category_list : list
        Categorical values to keep. Every other value is discarded.
    src_shp : geopandas.GeoDataFrame, optional
        Source shapefile. Default is None.
    coord_dtype : numpy.dtype
        Floating dtype the geometry coordinates are rounded to (see __lowest_coordinate_dtype).

    Returns
    -------
    geopandas.GeoDataFrame
        Polygonized numerical values of the kept categorical cells with valid positive weights.
    """
    dst_crs = dst_shp.crs
    with rasterio.open(cat_path) as cat_src:
        # Compute the reading window bounds (in the categorical raster CRS)
        min_x, min_y, max_x, max_y = __reading_bounds(dst_shp, src_shp, cat_src.crs)

        if min_x >= max_x or min_y >= max_y:
            # The rank-local source cells do not overlap the destination grid
            return gpd.GeoDataFrame({"value": []}, geometry=[], crs=cat_src.crs).to_crs(dst_crs)

        cat_window = __pixel_window(cat_src, min_x, min_y, max_x, max_y)
        if cat_window is None:
            # The reading window falls outside the raster extent
            return gpd.GeoDataFrame({"value": []}, geometry=[], crs=cat_src.crs).to_crs(dst_crs)
        cat_data = cat_src.read(1, window=cat_window, masked=False)
        if cat_data.size == 0:
            # The reading window falls outside the raster extent
            return gpd.GeoDataFrame({"value": []}, geometry=[], crs=cat_src.crs).to_crs(dst_crs)

        # Keep only the requested categories
        keep_mask = isin(cat_data, category_list)
        if not keep_mask.any():
            # None of the requested categories are present in this window
            return gpd.GeoDataFrame({"value": []}, geometry=[], crs=cat_src.crs).to_crs(dst_crs)

        transform = cat_src.window_transform(cat_window)
        src_crs = cat_src.crs

    # Both rasters share the same grid (see __check_matching_rasters): read the same window
    with rasterio.open(num_path) as num_src:
        num_data = num_src.read(1, window=cat_window, masked=True)

    values = num_data.astype(float64).filled(nan)
    # Categorical matches without a valid positive numerical value carry no proxy weight.
    keep_mask &= isfinite(values) & (values > 0)
    if not keep_mask.any():
        return gpd.GeoDataFrame({"value": []}, geometry=[], crs=src_crs).to_crs(dst_crs)

    # Polygonize a unique id per pixel and look up its numerical weight.
    pixel_id = arange(values.size, dtype=int32).reshape(values.shape)
    shape_gen = (
        (shapely.geometry.shape(s), int(v))
        for s, v in rasterio_features.shapes(pixel_id, mask=keep_mask, transform=transform)
    )
    proxy_shp = gpd.GeoDataFrame(dict(zip(["geometry", "pixel_id"], zip(*shape_gen))), crs=src_crs)
    proxy_shp["value"] = array(values.ravel()[proxy_shp["pixel_id"].values], dtype=float32)
    proxy_shp.drop(columns="pixel_id", inplace=True)

    # Reproject data
    proxy_shp = proxy_shp.to_crs(dst_crs)
    return __apply_coordinate_precision(proxy_shp, coord_dtype)


def __read_proxy_vector_as_gdf(
    path: str,
    dst_shp: gpd.GeoDataFrame,
    src_shp: gpd.GeoDataFrame = None,
    column_name: str = "weight",
    coord_dtype=float64,
):
    """
    Read a proxy vector file (e.g. GeoJSON with points or lines) into a GeoDataFrame.

    The file is windowed with the destination grid extent, widened to the full rank-local source cells
    that it touches (see __reading_bounds), so that each process only reads the features that its source
    cells can intersect while the cells cut by the destination domain keep all their features.

    The returned GeoDataFrame has a 'weight' column, read from the column_name column of the file, used
    to split the mass of a source cell among the features it contains. If the file has no column_name
    column, all the features get a weight of 1.0.
    """
    dst_crs = dst_shp.crs

    # Compute the reading window bounds (all the grids share the same CRS)
    min_x, min_y, max_x, max_y = __reading_bounds(dst_shp, src_shp, dst_crs)

    if min_x >= max_x or min_y >= max_y:
        # The rank-local source cells do not overlap the destination grid
        return gpd.GeoDataFrame({"weight": []}, geometry=[], crs=dst_crs)

    if dtype(coord_dtype).itemsize < dtype(float64).itemsize:
        # Read points that round onto an edge even if their file coordinates
        # lie just outside the rounded rank-local reading window.
        margins = abs(spacing(array([min_x, min_y, max_x, max_y], dtype=coord_dtype))).astype(float64)
        min_x, min_y, max_x, max_y = (
            min_x - margins[0], min_y - margins[1], max_x + margins[2], max_y + margins[3]
        )

    # The bbox is passed as a GeoSeries so that CRS mismatches with the file are resolved automatically
    bbox = gpd.GeoSeries([shapely.geometry.box(min_x, min_y, max_x, max_y)], crs=dst_crs)
    # Only the weight column is read; columns that are not in the file are silently skipped by read_file
    proxy_gdf = gpd.read_file(path, bbox=bbox, columns=[column_name])
    if len(proxy_gdf) == 0:
        return gpd.GeoDataFrame({"weight": []}, geometry=[], crs=dst_crs)

    # Ensure the weight column: if the file has no weight column, all the features get 1.0
    if column_name not in proxy_gdf.columns:
        warn(f"Weight column '{column_name}' not found in the proxy file {path}. All its features get a weight of 1.0.")
        sys.stderr.flush()
        proxy_gdf["weight"] = 1.0
    else:
        proxy_gdf["weight"] = proxy_gdf[column_name]
        if column_name != "weight":
            proxy_gdf = proxy_gdf.drop(columns=column_name)
    proxy_gdf["weight"] = array(proxy_gdf["weight"], dtype=float64)
    # Features with a missing, non-finite or non-positive weight carry no proxy information. They are
    # dropped so that a source cell containing only such features is not considered covered and falls
    # back to the lower priority proxies or to the plain area conservative weights (otherwise its
    # weights would be normalized by a zero sum and its mass would be lost).
    proxy_gdf = proxy_gdf.loc[isfinite(proxy_gdf["weight"].values) & (proxy_gdf["weight"].values > 0)]
    if len(proxy_gdf) == 0:
        return gpd.GeoDataFrame({"weight": []}, geometry=[], crs=dst_crs)

    # Reproject data
    proxy_gdf = proxy_gdf.to_crs(dst_crs)
    return __apply_coordinate_precision(proxy_gdf.reset_index(drop=True), coord_dtype)


def __geodesic_line_lengths(geometries, crs) -> "ndarray":
    """
    Compute the geodesic length, in meters, of each (multi)line geometry.

    Coordinates in projected or derived (e.g. rotated) CRSs are first transformed to the underlying
    geodetic CRS. Each segment is measured as the geodesic between its vertices on the CRS ellipsoid
    (WGS84 if the ellipsoid is unknown). Non-line parts (e.g. points from touching geometries) have zero length.

    Parameters
    ----------
    geometries : array-like of shapely.Geometry
        Line, multiline or geometry collection geometries.
    crs : pyproj.CRS or None
        CRS of the geometries. If None, planar lengths are returned.

    Returns
    -------
    numpy.ndarray
        Float64 lengths (meters if crs is not None), one per geometry.
    """
    geoms = asarray(geometries, dtype=object)
    if crs is None:
        return array(shapely.length(geoms), dtype=float64)
    lengths = zeros(len(geoms), dtype=float64)
    if len(geoms) == 0:
        return lengths

    # Flattening multi-part geometries and collections down to their single parts
    parts, owner = shapely.get_parts(geoms, return_index=True)
    while len(parts) > 0 and (shapely.get_type_id(parts) >= 4).any():
        parts, idx = shapely.get_parts(parts, return_index=True)
        owner = owner[idx]
    is_line = isin(shapely.get_type_id(parts), (1, 2))
    parts, owner = parts[is_line], owner[is_line]
    if len(parts) == 0:
        return lengths

    coords, coord_part = shapely.get_coordinates(parts, return_index=True)
    crs = CRS.from_user_input(crs)
    geodetic_crs = crs.geodetic_crs if crs.geodetic_crs is not None else CRS.from_epsg(4326)
    if not crs.is_geographic or crs.is_derived:
        lon, lat = Transformer.from_crs(crs, geodetic_crs, always_xy=True).transform(coords[:, 0], coords[:, 1])
    else:
        lon, lat = coords[:, 0], coords[:, 1]
    geod = geodetic_crs.get_geod() or Geod(ellps="WGS84")

    # Consecutive vertices belonging to the same part form a segment
    same_part = coord_part[1:] == coord_part[:-1]
    _, _, seg_len = geod.inv(lon[:-1][same_part], lat[:-1][same_part], lon[1:][same_part], lat[1:][same_part])
    lengths += bincount(owner[coord_part[:-1][same_part]], weights=seg_len, minlength=len(geoms))
    return lengths


def __split_line_portions_among_dst(
    line_parts: gpd.GeoDataFrame, dst_grid: gpd.GeoDataFrame, geodesic: bool = False
) -> DataFrame:
    """
    Split the normalized weight of each line portion among the destination cells it crosses,
    proportionally to the length of the sub-portion lying inside each of them.

    Parameters
    ----------
    line_parts : gpd.GeoDataFrame
        Line portions (already clipped with their source cell) with the FID_src and normalized pxy_value columns.
    dst_grid : gpd.GeoDataFrame
        Complete destination grid shapefile.
    geodesic : bool
        If True, lengths are geodesic lengths in meters (see __geodesic_line_lengths); otherwise they are
        planar lengths in CRS units.

    Returns
    -------
    DataFrame
        Intersection DataFrame with the FID_src, FID_dst and weight columns (not aggregated).
    """
    fid_part, fid_dst = dst_grid.sindex.query(line_parts.geometry, predicate="intersects")
    intersection_df = DataFrame(
        {
            "FID_src": array(line_parts.loc[fid_part, "FID_src"], dtype=uint32),
            "FID_dst": array(dst_grid.loc[fid_dst, "FID_dst"], dtype=uint32),
            "part": array(fid_part, dtype=int64),
        }
    )
    if len(intersection_df) == 0:
        intersection_df["weight"] = array([], dtype=float64)
        return intersection_df.drop(columns="part")

    geom_part = line_parts.geometry.values[fid_part]
    inter_geom = shapely.intersection(geom_part, dst_grid.geometry.values[fid_dst])
    if geodesic:
        inter_len = __geodesic_line_lengths(inter_geom, line_parts.crs)
        part_len = __geodesic_line_lengths(geom_part, line_parts.crs)
    else:
        inter_len = array(shapely.length(inter_geom), dtype=float64)
        part_len = array(shapely.length(geom_part), dtype=float64)
    # A portion lying on the border between destination cells is matched by all of them with its full
    # length: dividing by the summed matched length when it exceeds the portion length avoids counting
    # it more than once (the portion is split evenly among those cells)
    matched_len = DataFrame({"part": fid_part, "len": inter_len}).groupby("part")["len"].transform("sum").values
    intersection_df["weight"] = (
        array(line_parts.loc[fid_part, "pxy_value"], dtype=float64) * inter_len / maximum(part_len, matched_len)
    )
    return intersection_df.drop(columns="part")


def __owner_values(values, coord_dtype):
    """Round ownership coordinates exactly like the proxy geometries.

    Parameters
    ----------
    values : array-like
        Native coordinates (degrees or metres).
    coord_dtype : numpy.dtype
        Common coordinate precision (see __lowest_coordinate_dtype).

    Returns
    -------
    numpy.ndarray
        Float64 values. Lower precisions use __round_coordinates so that
        nested source and destination edges coincide; float64 is unchanged.
    """
    values = array(values, dtype=float64)
    if dtype(coord_dtype).itemsize >= dtype(float64).itemsize:
        return values
    return __round_coordinates(values, coord_dtype)


def __axis_owner_edges(bounds, coord_dtype):
    """Build ascending half-open cell edges for one native grid axis.

    Parameters
    ----------
    bounds : numpy.ndarray
        Cell intervals with shape (n, 2), in native axis order.
    coord_dtype : numpy.dtype
        Common coordinate precision.

    Returns
    -------
    tuple of numpy.ndarray
        Ascending edges (n + 1 values) and, for each sorted interval, its
        original axis index.

    Raises
    ------
    NotImplementedError
        The intervals are not one-dimensional (n, 2) bounds.
    ValueError
        The intervals are not monotonic along the axis.
    """
    bounds = array(bounds, dtype=float64)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise NotImplementedError("Point proxies require grids with one-dimensional native axes")
    lower, upper = bounds.min(axis=1), bounds.max(axis=1)
    if len(lower) == 1 or (diff(lower) > 0).all():
        order = arange(len(lower))
    elif (diff(lower) < 0).all():
        order = arange(len(lower))[::-1]
    else:
        raise ValueError("Point proxy ownership requires monotonic grid axes")
    # Interior edges are the lower bounds of the following cells, so
    # neighbours share one value and rounding cannot create gaps or overlaps.
    edges = concatenate([lower[order], upper[order][-1:]])
    return __owner_values(edges, coord_dtype), order


def __point_ownership_axes(grid, coord_dtype):
    """Collect the global native axes used to assign points to grid cells.

    Parameters
    ----------
    grid : nes.Nes
        Structured grid (regular, rotated, LCC, Mercator or LAEA).
    coord_dtype : numpy.dtype
        Common coordinate precision.

    Returns
    -------
    dict
        Sorted edges and orders per axis, the full column count used by the
        NES FIDs (row * n_columns + column) and, for projected or rotated
        grids, the geographic CRS and native-to-geographic transformer.

    Raises
    ------
    NotImplementedError
        The grid has no structured native axes (e.g. a points grid).

    Notes
    -----
    Collective on the grid communicator: the full-axis getters broadcast.
    Native axes are in degrees (regular, rotated) or metres (projected).
    """
    if isinstance(grid, nes.PointsNes):
        raise NotImplementedError("Point proxies require structured source and destination grids")
    from nes.projections._coordinates import angular_factor, length_factor, native_bounds

    if isinstance(grid, nes.LAEANes):
        from nes.projections.laea_nes import _metres

        x_axis, y_axis = grid.get_full_x(), grid.get_full_y()
        x_bounds = x_axis["bounds_data"] * _metres(x_axis["units"])
        y_bounds = y_axis["bounds_data"] * _metres(y_axis["units"])
    elif isinstance(grid, (nes.RotatedNes, nes.LCCNes, nes.MercatorNes)):
        if isinstance(grid, nes.RotatedNes):
            names, factor, default_units = ("rlat", "rlon"), angular_factor, "degrees"
        else:
            names, factor, default_units = ("y", "x"), length_factor, "m"
        y_bounds, x_bounds = [
            native_bounds(axis, grid.projection_data.get("inc_" + name))
            * factor(axis.get("units", default_units))
            for name, axis in ((name, getattr(grid, "get_full_" + name)()) for name in names)
        ]
    else:
        y_bounds = grid.get_full_latitudes_boundaries()
        x_bounds = grid.get_full_longitudes_boundaries()
        if y_bounds is None or x_bounds is None:
            grid.create_spatial_bounds()
            y_bounds = grid.get_full_latitudes_boundaries()
            x_bounds = grid.get_full_longitudes_boundaries()
        y_bounds, x_bounds = y_bounds["data"], x_bounds["data"]
    x_edges, x_order = __axis_owner_edges(x_bounds, coord_dtype)
    y_edges, y_order = __axis_owner_edges(y_bounds, coord_dtype)
    native = isinstance(grid, (nes.LAEANes, nes.RotatedNes, nes.LCCNes, nes.MercatorNes))
    return {
        "x": (x_edges, x_order),
        "y": (y_edges, y_order),
        "n_columns": len(x_order),
        "coord_dtype": coord_dtype,
        "geographic_crs": grid._geographic_crs if native else None,
        "to_geographical": grid._to_geographical if native else None,
    }


def __axis_owner(values, edges, order):
    """Locate values in half-open intervals [edges[i], edges[i + 1]).

    Parameters
    ----------
    values : numpy.ndarray
        Coordinates in the axis units.
    edges : numpy.ndarray
        Ascending cell edges.
    order : numpy.ndarray
        Original axis index of each sorted interval.

    Returns
    -------
    numpy.ndarray
        Original axis index per value, -1 outside the axis. The outer upper
        edge belongs to the last cell so that no domain point is lost.
    """
    position = searchsorted(edges, values, side="right") - 1
    position[values == edges[-1]] = len(edges) - 2
    inside = (position >= 0) & (position < len(edges) - 1)
    owner = full(len(values), -1, dtype=int64)
    owner[inside] = order[position[inside]]
    return owner


def __point_owner_fids(points, axes):
    """Assign each point to exactly one grid cell using half-open intervals.

    Parameters
    ----------
    points : gpd.GeoDataFrame
        Single-part point geometries, rounded to the common precision.
    axes : dict
        Output of __point_ownership_axes.

    Returns
    -------
    numpy.ndarray
        Global NES FID (row * n_columns + column) per point, -1 when the
        point is outside the grid.

    Notes
    -----
    Each cell owns [lower, upper) along both native axes, so a point on a
    shared edge or corner belongs to the cell with the larger native x and
    y (the east/north cell on usual grids). Only global axes are used: the
    owner is independent of MPI decomposition, cell order and local
    polygons, and needs no communication. On projected and rotated grids
    points are first transformed to native coordinates, so ownership is
    exact in native coordinates, not in longitude/latitude.
    """
    x = array(shapely.get_x(points.geometry.values), dtype=float64)
    y = array(shapely.get_y(points.geometry.values), dtype=float64)
    if axes["to_geographical"] is not None and len(x):
        lon, lat = Transformer.from_crs(points.crs, axes["geographic_crs"], always_xy=True).transform(x, y)
        x, y = axes["to_geographical"].transform(lon, lat, direction="INVERSE")
        x, y = __owner_values(x, axes["coord_dtype"]), __owner_values(y, axes["coord_dtype"])
    column = __axis_owner(x, *axes["x"])
    row = __axis_owner(y, *axes["y"])
    return where((column >= 0) & (row >= 0), row * axes["n_columns"] + column, -1)


def __reject_non_point_geometries(points, comm):
    """Fail on every rank if a points proxy holds non-Point geometries.

    Parameters
    ----------
    points : gpd.GeoDataFrame or None
        Rank-local proxy features.
    comm : MPI.Comm
        Source-grid communicator; all ranks must call this function.

    Raises
    ------
    ValueError
        Any rank read a MultiPoint or another non-Point geometry. The check
        is collective, so no rank is left waiting in a later MPI call.
    """
    local = [] if points is None else sorted(set(points.geom_type.dropna()) - {"Point"})
    found = sorted({kind for part in comm.allgather(local) for kind in part})
    if found:
        raise ValueError(
            f"Points proxies must contain only Point geometries (found {found}). "
            "Split MultiPoint features into separate Point features, each with its own weight."
        )


def __proxy_points_intersections(
    self,
    proxy_path: str,
    src_grid: gpd.GeoDataFrame,
    dst_grid: gpd.GeoDataFrame,
    dst_nes,
    column_name="weight",
    info=False,
    coord_dtype=float64,
):
    """
    To compute the source to destination weights driven by a points proxy.

    Parameters
    ----------
    self : nes.Nes
        Source projection Nes Object.
    proxy_path : str
        Path to the proxy points vector file.
    src_grid : gpd.GeoDataFrame
        Rank-local portion of the source grid shapefile.
    dst_grid : gpd.GeoDataFrame
        Complete destination grid shapefile.
    dst_nes : nes.Nes
        Destination grid, whose global native axes define point ownership.
    column_name : str
        Name of the column holding the feature weights in the proxy file.
    info : bool
        Indicates if you want to print extra info during the methods process.
    coord_dtype : numpy.dtype
        Floating dtype the geometry coordinates are rounded to (see __lowest_coordinate_dtype).

    Returns
    -------
    tuple
        Intersection DataFrame with the FID_src, FID_dst and weight columns, and the array of source
        cell FIDs covered by the proxy.

    Raises
    ------
    ValueError
        The proxy holds MultiPoint or other non-Point geometries.

    Notes
    -----
    Collective on the source communicator. Each point has exactly one source
    owner and one destination owner, given by half-open native grid cells
    (see __point_owner_fids). A point whose source owner is not one of the
    rank-local remaining cells is ignored on this rank. Other source cells
    keep their own emissions and use fallback if they own no point.
    """
    if info and self.master:
        print("\t\t\tReading proxy points")
        sys.stdout.flush()
    # Window the point file with the region covered by both the local source cells and the destination grid
    pxy_points = (
        __read_proxy_vector_as_gdf(
            proxy_path, dst_grid, src_shp=src_grid, column_name=column_name, coord_dtype=coord_dtype
        )
        if len(src_grid) > 0
        else None
    )
    if info and self.master:
        print(f"\t\t\tProxy points read (master process: {0 if pxy_points is None else len(pxy_points)} points)")
        sys.stdout.flush()

    # Collective steps first, so that ranks without source cells still take part
    __reject_non_point_geometries(pxy_points, self.comm)
    src_axes = __point_ownership_axes(self, coord_dtype)
    dst_axes = __point_ownership_axes(dst_nes, coord_dtype)

    # Each point belongs to exactly one source cell; keep the ones owned by this rank's cells
    if pxy_points is None:
        pxy_points = gpd.GeoDataFrame({"weight": []}, geometry=[], crs=src_grid.crs)
    owner_src = __point_owner_fids(pxy_points, src_axes)
    fid_pxy = where(isin(owner_src, src_grid["FID_src"].values))[0]
    if self.comm.allreduce(len(fid_pxy)) == 0 and self.master:
        warn(
            "No proxy points found inside the source grid. "
            + "The source cells will fall back to the lower priority proxies or to the plain area conservative "
            "weights."
        )
        sys.stderr.flush()
    if info and self.master:
        print(f"\t\t\tSource to points: {len(fid_pxy)} matches found (master process)")
        sys.stdout.flush()

    if len(fid_pxy) == 0:
        # The source cells of this rank do not contain any proxy point
        covered_fids = array([], dtype=uint32)
        intersection_df = DataFrame(
            {"FID_src": array([], dtype=uint32), "FID_dst": array([], dtype=uint32), "weight": array([], dtype=float64)}
        )
        return intersection_df, covered_fids

    # Filling the proxy GeoDataFrame with the matched FID values, point weights and point geometries
    proxy_df = gpd.GeoDataFrame(
        {
            "FID_src": array(owner_src[fid_pxy], dtype=uint32),
            "FID_pxy": array(fid_pxy, dtype=uint32),
            "pxy_value": array(pxy_points.loc[fid_pxy, "weight"], dtype=float64),
        },
        geometry=pxy_points.loc[fid_pxy, "geometry"].values,
        crs=pxy_points.crs,
    )

    proxy_df.reset_index(drop=True, inplace=True)
    covered_fids = proxy_df["FID_src"].unique()

    # Normalizing the point weights by dividing each weight by the sum of the weights of the points
    # inside the same source cell. All the points of a source cell live on the same rank, so this is
    # a local operation.
    proxy_df["pxy_value"] = proxy_df["pxy_value"].div(proxy_df.groupby("FID_src")["pxy_value"].transform("sum"))

    if info and self.master:
        print("\t\t\tPoints to destination")
        sys.stdout.flush()

    # Each point belongs to exactly one destination cell (points outside the destination grid are dropped)
    owner_dst = __point_owner_fids(proxy_df, dst_axes)
    fid_src_pxy = where(isin(owner_dst, dst_grid["FID_dst"].values))[0]
    if info and self.master:
        print(f"\t\t\tPoints to destination: {len(fid_src_pxy)} matches found (master process)")
        sys.stdout.flush()

    # A point is fully contained in one destination cell, so its normalized weight is transferred as is
    intersection_df = DataFrame(
        {
            "FID_src": array(proxy_df.loc[fid_src_pxy, "FID_src"], dtype=uint32),
            "FID_pxy": array(proxy_df.loc[fid_src_pxy, "FID_pxy"], dtype=uint32),
            "FID_dst": array(owner_dst[fid_src_pxy], dtype=uint32),
            "weight": array(proxy_df.loc[fid_src_pxy, "pxy_value"], dtype=float64),
        }
    )

    intersection_df = intersection_df.groupby(["FID_src", "FID_dst"])["weight"].sum().reset_index()
    if info and self.master:
        print(f"\t\t\tPoint weights aggregated (master process: {len(intersection_df)} source-destination pairs)")
        sys.stdout.flush()

    return intersection_df, covered_fids


def __proxy_lines_intersections(
    self,
    proxy_path: str,
    src_grid: gpd.GeoDataFrame,
    dst_grid: gpd.GeoDataFrame,
    column_name="weight",
    info=False,
    coord_dtype=float64,
):
    """
    To compute the source to destination weights driven by a lines proxy.

    Parameters
    ----------
    self : nes.Nes
        Source projection Nes Object.
    proxy_path : str
        Path to the proxy lines vector file.
    src_grid : gpd.GeoDataFrame
        Rank-local portion of the source grid shapefile.
    dst_grid : gpd.GeoDataFrame
        Complete destination grid shapefile.
    column_name : str
        Name of the column holding the feature weights in the proxy file.
    info : bool
        Indicates if you want to print extra info during the methods process.
    coord_dtype : numpy.dtype
        Floating dtype the geometry coordinates are rounded to (see __lowest_coordinate_dtype).

    Returns
    -------
    tuple
        Intersection DataFrame with the FID_src, FID_dst and weight columns, and the array of source
        cell FIDs covered by the proxy.
    """
    if info and self.master:
        print("\t\t\tReading proxy lines")
        sys.stdout.flush()
    # Window the line file with the region covered by both the local source cells and the destination grid
    pxy_lines = (
        __read_proxy_vector_as_gdf(
            proxy_path, dst_grid, src_shp=src_grid, column_name=column_name, coord_dtype=coord_dtype
        )
        if len(src_grid) > 0
        else None
    )
    if info and self.master:
        print(f"\t\t\tProxy lines read (master process: {0 if pxy_lines is None else len(pxy_lines)} lines)")
        sys.stdout.flush()

    # Querying the proxy lines to find the ones crossing each source cell
    if pxy_lines is None or len(pxy_lines) == 0:
        fid_src, fid_pxy = array([], dtype=int64), array([], dtype=int64)
    else:
        fid_src, fid_pxy = pxy_lines.sindex.query(src_grid.geometry, predicate="intersects")
    if self.comm.allreduce(len(fid_src)) == 0 and self.master:
        warn(
            "No proxy lines found inside the source grid. "
            + "The source cells will fall back to the lower priority proxies or to the plain area conservative "
            "weights."
        )
        sys.stderr.flush()
    if info and self.master:
        print(f"\t\t\tSource to lines: {len(fid_src)} matches found (master process)")
        sys.stdout.flush()

    if len(fid_src) == 0:
        # The source cells of this rank do not intersect any proxy line
        covered_fids = array([], dtype=uint32)
        intersection_df = DataFrame(
            {"FID_src": array([], dtype=uint32), "FID_dst": array([], dtype=uint32), "weight": array([], dtype=float64)}
        )
        return intersection_df, covered_fids

    # Filling the proxy GeoDataFrame with the matched FID values, line weights and line geometries
    proxy_df = gpd.GeoDataFrame(
        {
            "FID_src": array(src_grid.loc[fid_src, "FID_src"], dtype=uint32),
            "FID_pxy": array(fid_pxy, dtype=uint32),
            "pxy_value": array(pxy_lines.loc[fid_pxy, "weight"], dtype=float64),
        },
        geometry=pxy_lines.loc[fid_pxy, "geometry"].values,
        crs=pxy_lines.crs,
    )
    proxy_df["geometry_src"] = src_grid.loc[fid_src, "geometry"].values

    # Clipping each line with the source cell: a line crossing several source cells contributes
    # to each of them only with the portion lying inside it
    proxy_df["geometry"] = proxy_df.apply(lambda x: x["geometry"].intersection(x["geometry_src"]), axis=1)
    proxy_df.drop(columns="geometry_src", inplace=True)

    # Removing zero-length portions (e.g. lines that only touch the border of a source cell)
    portion_len = __geodesic_line_lengths(proxy_df.geometry.values, proxy_df.crs)
    keep = portion_len > 0
    proxy_df = proxy_df.loc[keep].reset_index(drop=True)
    covered_fids = proxy_df["FID_src"].unique()

    # The mass of a source cell is split among its line portions proportionally to their geodesic
    # length (meters) multiplied by the line weight
    proxy_df["pxy_value"] = proxy_df["pxy_value"] * portion_len[keep]

    # Normalizing the portion weights by dividing each weight by the sum of the weights of the
    # portions inside the same source cell. All the portions of a source cell live on the same
    # rank, so this is a local operation.
    proxy_df["pxy_value"] = proxy_df["pxy_value"].div(proxy_df.groupby("FID_src")["pxy_value"].transform("sum"))

    if info and self.master:
        print("\t\t\tLines to destination")
        sys.stdout.flush()

    # Querying the destination grid to find the destination cells crossed by each line portion, and
    # splitting each portion among them proportionally to the length lying inside each of them
    intersection_df = __split_line_portions_among_dst(proxy_df, dst_grid, geodesic=True)
    if info and self.master:
        print(f"\t\t\tLines to destination: {len(intersection_df)} matches found (master process)")
        sys.stdout.flush()

    intersection_df = intersection_df.groupby(["FID_src", "FID_dst"])["weight"].sum().reset_index()
    if info and self.master:
        print(f"\t\t\tLine weights aggregated (master process: {len(intersection_df)} source-destination pairs)")
        sys.stdout.flush()

    return intersection_df, covered_fids


def __proxy_polygons_intersections(
    self, proxy_path, src_grid, dst_grid, column_name="weight", info=False, coord_dtype=float64
):
    """
    To compute the source to destination weights driven by a polygons proxy.

    Parameters
    ----------
    self : nes.Nes
        Source projection Nes Object.
    proxy_path : str
        Path to the proxy polygons vector file.
    src_grid : GeoDataFrame
        Rank-local portion of the source grid shapefile.
    dst_grid : GeoDataFrame
        Complete destination grid shapefile.
    column_name : str
        Name of the column holding the feature weights in the proxy file.
    info : bool
        Indicates if you want to print extra info during the methods process.
    coord_dtype : numpy.dtype
        Floating dtype the geometry coordinates are rounded to (see __lowest_coordinate_dtype).

    Returns
    -------
    tuple
        Intersection DataFrame with the FID_src, FID_dst and weight columns, and the array of source
        cell FIDs covered by the proxy.
    """
    if info and self.master:
        print("\t\t\tReading proxy polygons")
        sys.stdout.flush()
    # Window the polygon file with the region covered by both the local source cells and the destination grid
    pxy_polys = (
        __read_proxy_vector_as_gdf(
            proxy_path, dst_grid, src_shp=src_grid, column_name=column_name, coord_dtype=coord_dtype
        )
        if len(src_grid) > 0
        else None
    )
    if info and self.master:
        print(f"\t\t\tProxy polygons read (master process: {0 if pxy_polys is None else len(pxy_polys)} polygons)")
        sys.stdout.flush()

    # Querying the proxy polygons to find the ones overlapping each source cell
    if pxy_polys is None or len(pxy_polys) == 0:
        fid_src, fid_pxy = array([], dtype=int64), array([], dtype=int64)
    else:
        fid_src, fid_pxy = pxy_polys.sindex.query(src_grid.geometry, predicate="intersects")
    if self.comm.allreduce(len(fid_src)) == 0 and self.master:
        warn(
            "No proxy polygons found inside the source grid. "
            + "The source cells will fall back to the lower priority proxies or to the plain area conservative "
            "weights."
        )
        sys.stderr.flush()
    if info and self.master:
        print(f"\t\t\tSource to polygons: {len(fid_src)} matches found (master process)")
        sys.stdout.flush()

    if len(fid_src) == 0:
        # The source cells of this rank do not overlap any proxy polygon
        covered_fids = array([], dtype=uint32)
        intersection_df = DataFrame(
            {"FID_src": array([], dtype=uint32), "FID_dst": array([], dtype=uint32), "weight": array([], dtype=float64)}
        )
        return intersection_df, covered_fids

    # Filling the proxy GeoDataFrame with the matched FID values, polygon weights and geometries
    proxy_df = gpd.GeoDataFrame(
        {
            "FID_src": array(src_grid.loc[fid_src, "FID_src"], dtype=uint32),
            "FID_pxy": array(fid_pxy, dtype=uint32),
            "pxy_value": array(pxy_polys.loc[fid_pxy, "weight"], dtype=float64),
        },
        geometry=pxy_polys.loc[fid_pxy, "geometry"].values,
        crs=pxy_polys.crs,
    )
    proxy_df["geometry_src"] = src_grid.loc[fid_src, "geometry"].values

    # Clipping each polygon with the source cell: a polygon overlapping several source cells contributes
    # to each of them only with the part lying inside it
    proxy_df["geometry"] = proxy_df.apply(lambda x: x["geometry"].intersection(x["geometry_src"]).buffer(0), axis=1)
    proxy_df.drop(columns="geometry_src", inplace=True)

    # Removing zero-area parts (e.g. polygons that only touch the border of a source cell)
    # ! Store area
    proxy_df = proxy_df.loc[proxy_df.geometry.area > 0].reset_index(drop=True)
    covered_fids = proxy_df["FID_src"].unique()

    # The mass of a source cell is split among its polygon parts proportionally to their area
    # multiplied by the polygon weight
    proxy_df["pxy_value"] = proxy_df["pxy_value"] * proxy_df.geometry.area

    # Normalizing the part weights by dividing each weight by the sum of the weights of the
    # parts inside the same source cell. All the parts of a source cell live on the same
    # rank, so this is a local operation.
    proxy_df["pxy_value"] = proxy_df["pxy_value"].div(proxy_df.groupby("FID_src")["pxy_value"].transform("sum"))

    if info and self.master:
        print("\t\t\tPolygons to destination")
        sys.stdout.flush()

    # Querying the destination grid to find the destination cells overlapped by each polygon part
    fid_src_pxy, fid_dst = dst_grid.sindex.query(proxy_df.geometry, predicate="intersects")
    if info and self.master:
        print(f"\t\t\tPolygons to destination: {len(fid_dst)} matches found (master process)")
        sys.stdout.flush()

    # A polygon part is split among the destination cells proportionally to the area of the
    # sub-part lying inside each of them
    intersection_df = DataFrame(
        {
            "FID_src": array(proxy_df.loc[fid_src_pxy, "FID_src"], dtype=uint32),
            "FID_dst": array(dst_grid.loc[fid_dst, "FID_dst"], dtype=uint32),
            "pxy_value": array(proxy_df.loc[fid_src_pxy, "pxy_value"], dtype=float64),
        }
    )
    intersection_df["geometry_pxy"] = proxy_df.loc[fid_src_pxy, "geometry"].values
    intersection_df["geometry_dst"] = dst_grid.loc[fid_dst, "geometry"].values

    if len(intersection_df) > 0:
        intersection_df["weight"] = array(
            intersection_df.apply(
                lambda x: (
                    x["pxy_value"]
                    * (x["geometry_pxy"].intersection(x["geometry_dst"]).buffer(0).area / x["geometry_pxy"].area)
                ),
                axis=1,
            ),
            dtype=float64,
        )
    else:
        intersection_df["weight"] = array([], dtype=float64)
    intersection_df.drop(columns=["pxy_value", "geometry_pxy", "geometry_dst"], inplace=True)

    intersection_df = intersection_df.groupby(["FID_src", "FID_dst"])["weight"].sum().reset_index()
    if info and self.master:
        print(f"\t\t\tPolygon weights aggregated (master process: {len(intersection_df)} source-destination pairs)")
        sys.stdout.flush()

    return intersection_df, covered_fids


def __proxy_multigeometry_intersections(
    self, proxy_path, src_grid, dst_grid, dst_nes, column_name="weight", info=False, coord_dtype=float64
):
    """
    To compute the source to destination weights driven by a multigeometry (points, lines and polygons) proxy.

    There is no priority between geometry types: all the features of a source cell share the same
    normalization pool, where points contribute with their weight, line portions with their weight
    multiplied by their length inside the cell and polygon parts with their weight multiplied by
    their area inside the cell. Multi-part geometries are split into their single parts.

    Parameters
    ----------
    self : nes.Nes
        Source projection Nes Object.
    proxy_path : str
        Path to the proxy vector file combining point, line and polygon geometries.
    src_grid : GeoDataFrame
        Rank-local portion of the source grid shapefile.
    dst_grid : GeoDataFrame
        Complete destination grid shapefile.
    dst_nes : nes.Nes
        Destination grid, whose global native axes define point ownership.
    column_name : str
        Name of the column holding the feature weights in the proxy file.
    info : bool
        Indicates if you want to print extra info during the methods process.
    coord_dtype : numpy.dtype
        Floating dtype the geometry coordinates are rounded to (see __lowest_coordinate_dtype).

    Returns
    -------
    tuple
        Intersection DataFrame with the FID_src, FID_dst and weight columns, and the array of source
        cell FIDs covered by the proxy.

    Notes
    -----
    Collective on the source communicator. Point parts use the same
    half-open native-cell ownership as points proxies (see
    __point_owner_fids), so each point has one source and one destination
    cell. Structured grids are only required when the file has points.
    """
    if info and self.master:
        print("\t\t\tReading multigeometry proxy")
        sys.stdout.flush()
    # Window the vector file with the region covered by both the local source cells and the destination grid
    pxy_feats = (
        __read_proxy_vector_as_gdf(
            proxy_path, dst_grid, src_shp=src_grid, column_name=column_name, coord_dtype=coord_dtype
        )
        if len(src_grid) > 0
        else None
    )
    if pxy_feats is not None and len(pxy_feats) > 0:
        # Split multi-part geometries so that each part is handled with its own geometry logic
        pxy_feats = pxy_feats.explode(index_parts=False).reset_index(drop=True)
        supported = pxy_feats.geometry.geom_type.isin(["Point", "LineString", "LinearRing", "Polygon"])
        if (~supported).any() and self.master:
            warn(
                f"Unsupported geometry types {sorted(pxy_feats.loc[~supported].geometry.geom_type.unique())} "
                + "found in the multigeometry proxy. Only point, line and polygon geometries are used."
            )
            sys.stderr.flush()
        pxy_feats = pxy_feats.loc[supported].reset_index(drop=True)
    if info and self.master:
        print(f"\t\t\tMultigeometry proxy read (master process: {0 if pxy_feats is None else len(pxy_feats)} features)")
        sys.stdout.flush()

    # Point ownership needs collective axis broadcasts, so all ranks decide together whether it is used
    local_points = pxy_feats is not None and bool((pxy_feats.geometry.geom_type == "Point").any())
    if self.comm.allreduce(local_points, op=MPI.LOR):
        src_axes = __point_ownership_axes(self, coord_dtype)
        dst_axes = __point_ownership_axes(dst_nes, coord_dtype)

    # Querying the proxy features to find the ones intersecting each source cell
    if pxy_feats is None or len(pxy_feats) == 0:
        fid_src, fid_pxy = array([], dtype=int64), array([], dtype=int64)
    else:
        fid_src, fid_pxy = pxy_feats.sindex.query(src_grid.geometry, predicate="intersects")
    if self.comm.allreduce(len(fid_src)) == 0 and self.master:
        warn(
            "No proxy features found inside the source grid. "
            + "The source cells will fall back to the lower priority proxies or to the plain area conservative "
            "weights."
        )
        sys.stderr.flush()
    if info and self.master:
        print(f"\t\t\tSource to features: {len(fid_src)} matches found (master process)")
        sys.stdout.flush()

    if len(fid_src) == 0:
        # The source cells of this rank do not intersect any proxy feature
        covered_fids = array([], dtype=uint32)
        intersection_df = DataFrame(
            {"FID_src": array([], dtype=uint32), "FID_dst": array([], dtype=uint32), "weight": array([], dtype=float64)}
        )
        return intersection_df, covered_fids

    # Filling the proxy GeoDataFrame with the matched FID values, feature weights and geometries
    proxy_df = gpd.GeoDataFrame(
        {
            "FID_src": array(src_grid.loc[fid_src, "FID_src"], dtype=uint32),
            "FID_pxy": array(fid_pxy, dtype=uint32),
            "pxy_value": array(pxy_feats.loc[fid_pxy, "weight"], dtype=float64),
        },
        geometry=pxy_feats.loc[fid_pxy, "geometry"].values,
        crs=pxy_feats.crs,
    )
    proxy_df["geometry_src"] = src_grid.loc[fid_src, "geometry"].values
    # Explicit kind column: clipping can turn a geometry into a collection, hiding its original type
    geom_types = proxy_df.geometry.geom_type
    proxy_df["pxy_kind"] = "line"
    proxy_df.loc[geom_types == "Point", "pxy_kind"] = "point"
    proxy_df.loc[geom_types == "Polygon", "pxy_kind"] = "polygon"

    # A point lying on the border of several source cells is kept only for its owner cell
    point_df = proxy_df.loc[proxy_df["pxy_kind"] == "point"].drop(columns="geometry_src")
    if len(point_df) > 0:
        point_df = point_df.loc[__point_owner_fids(point_df, src_axes) == point_df["FID_src"].values]

    # Clipping each line with the source cell: only the portion lying inside it contributes,
    # proportionally to its length multiplied by the line weight
    line_df = proxy_df.loc[proxy_df["pxy_kind"] == "line"].reset_index(drop=True)
    if len(line_df) > 0:
        line_df["geometry"] = line_df.apply(lambda x: x["geometry"].intersection(x["geometry_src"]), axis=1)
        # Removing zero-length portions (e.g. lines that only touch the border of a source cell)
        line_df = line_df.loc[line_df.geometry.length > 0].reset_index(drop=True)
        line_df["pxy_value"] = line_df["pxy_value"] * line_df.geometry.length
    line_df = line_df.drop(columns="geometry_src")

    # Clipping each polygon with the source cell: only the part lying inside it contributes,
    # proportionally to its area multiplied by the polygon weight
    poly_df = proxy_df.loc[proxy_df["pxy_kind"] == "polygon"].reset_index(drop=True)
    if len(poly_df) > 0:
        poly_df["geometry"] = poly_df.apply(lambda x: x["geometry"].intersection(x["geometry_src"]).buffer(0), axis=1)
        # Removing zero-area parts (e.g. polygons that only touch the border of a source cell)
        poly_df = poly_df.loc[poly_df.geometry.area > 0].reset_index(drop=True)
        poly_df["pxy_value"] = poly_df["pxy_value"] * poly_df.geometry.area
    poly_df = poly_df.drop(columns="geometry_src")

    # Points, line portions and polygon parts share the same normalization pool: no priority
    # between geometry types
    parts_df = concat([point_df, line_df, poly_df], ignore_index=True)
    covered_fids = parts_df["FID_src"].unique()
    parts_df["pxy_value"] = parts_df["pxy_value"].div(parts_df.groupby("FID_src")["pxy_value"].transform("sum"))

    if info and self.master:
        print("\t\t\tFeatures to destination")
        sys.stdout.flush()

    pnt_parts = parts_df.loc[parts_df["pxy_kind"] == "point"].reset_index(drop=True)
    lin_parts = parts_df.loc[parts_df["pxy_kind"] == "line"].reset_index(drop=True)
    ply_parts = parts_df.loc[parts_df["pxy_kind"] == "polygon"].reset_index(drop=True)

    # A point belongs to exactly one destination cell, so its normalized weight is transferred as is
    owner_dst = __point_owner_fids(pnt_parts, dst_axes) if len(pnt_parts) > 0 else array([], dtype=int64)
    fid_pnt = where(isin(owner_dst, dst_grid["FID_dst"].values))[0]
    point_int = DataFrame(
        {
            "FID_src": array(pnt_parts.loc[fid_pnt, "FID_src"], dtype=uint32),
            "FID_pxy": array(pnt_parts.loc[fid_pnt, "FID_pxy"], dtype=uint32),
            "FID_dst": array(owner_dst[fid_pnt], dtype=uint32),
            "weight": array(pnt_parts.loc[fid_pnt, "pxy_value"], dtype=float64),
        }
    )
    point_int.drop(columns="FID_pxy", inplace=True)

    # A line portion is split among the destination cells proportionally to the length of the
    # sub-portion lying inside each of them
    line_int = __split_line_portions_among_dst(lin_parts, dst_grid)

    # A polygon part is split among the destination cells proportionally to the area of the
    # sub-part lying inside each of them
    fid_ply, fid_dst = dst_grid.sindex.query(ply_parts.geometry, predicate="intersects")
    poly_int = DataFrame(
        {
            "FID_src": array(ply_parts.loc[fid_ply, "FID_src"], dtype=uint32),
            "FID_dst": array(dst_grid.loc[fid_dst, "FID_dst"], dtype=uint32),
            "pxy_value": array(ply_parts.loc[fid_ply, "pxy_value"], dtype=float64),
        }
    )
    poly_int["geometry_pxy"] = ply_parts.loc[fid_ply, "geometry"].values
    poly_int["geometry_dst"] = dst_grid.loc[fid_dst, "geometry"].values
    if len(poly_int) > 0:
        poly_int["weight"] = array(
            poly_int.apply(
                lambda x: (
                    x["pxy_value"]
                    * (x["geometry_pxy"].intersection(x["geometry_dst"]).buffer(0).area / x["geometry_pxy"].area)
                ),
                axis=1,
            ),
            dtype=float64,
        )
    else:
        poly_int["weight"] = array([], dtype=float64)
    poly_int.drop(columns=["pxy_value", "geometry_pxy", "geometry_dst"], inplace=True)

    intersection_df = concat([point_int, line_int, poly_int], ignore_index=True)
    intersection_df = intersection_df.groupby(["FID_src", "FID_dst"])["weight"].sum().reset_index()
    if info and self.master:
        print(
            f"\t\t\tMultigeometry weights aggregated (master process: {len(intersection_df)} source-destination pairs)"
        )
        sys.stdout.flush()

    return intersection_df, covered_fids


def __proxy_raster_intersections(
    self,
    proxy_path: str,
    src_grid: gpd.GeoDataFrame,
    dst_grid: gpd.GeoDataFrame,
    info: bool = False,
    category_list=None,
    numerical_path=None,
    coord_dtype=float64,
):
    """
    To compute the source to destination weights driven by a raster proxy.

    Parameters
    ----------
    self : nes.Nes
        Source projection Nes Object.
    proxy_path : str
        Path to the proxy raster file (the categorical mask raster when numerical_path is given).
    src_grid : gpd.GeoDataFrame
        Rank-local portion of the source grid shapefile.
    dst_grid : gpd.GeoDataFrame
        Complete destination grid shapefile.
    info : bool
        Indicates if you want to print extra info during the methods process.
    category_list : list, None
        When provided, the raster is treated as categorical: only the cells whose value is in this
        list are kept. When None, the raster is treated as continuous.
    numerical_path : str, None
        When provided (together with category_list), proxy_path is used as a keep-mask and the values
        of this numerical raster drive the redistribution within the kept cells.
    coord_dtype : numpy.dtype
        Floating dtype the geometry coordinates are rounded to (see __lowest_coordinate_dtype).

    Returns
    -------
    tuple
        Intersection DataFrame with the FID_src, FID_dst and weight columns, and the array of source
        cell FIDs covered by the proxy.
    """
    if info and self.master:
        print("\t\t\tCreating proxy shapefile from raster")
        sys.stdout.flush()
    if numerical_path is not None:
        # Checked on every process (before skipping the empty ones) so that all of them raise together
        __check_matching_rasters(proxy_path, numerical_path)
    # Window the raster with the region covered by both the local source cells and the destination grid
    if len(src_grid) == 0:
        pxy_grid = None
    elif numerical_path is not None:
        pxy_grid = __read_raster_categorical_numerical_proxy_as_gdf(
            proxy_path, numerical_path, dst_grid, category_list, src_shp=src_grid, coord_dtype=coord_dtype
        )
    elif category_list is not None:
        pxy_grid = __read_raster_categorical_proxy_as_gdf(
            proxy_path, dst_grid, category_list, src_shp=src_grid, coord_dtype=coord_dtype
        )
    else:
        pxy_grid = __read_raster_proxy_as_gdf(proxy_path, dst_grid, src_shp=src_grid, coord_dtype=coord_dtype)
    if pxy_grid is not None:
        pxy_grid["FID_pxy"] = pxy_grid.index
    if info and self.master:
        print(f"\t\t\tProxy shapefile created (master process: {0 if pxy_grid is None else len(pxy_grid)} polygons)")
        sys.stdout.flush()

    # Querying the proxy shapefile to find intersections with the source grid
    if pxy_grid is None or len(pxy_grid) == 0:
        fid_src, fid_proxy = array([], dtype=int64), array([], dtype=int64)
    else:
        fid_src, fid_proxy = pxy_grid.sindex.query(src_grid.geometry, predicate="intersects")
    if self.comm.allreduce(len(fid_src)) == 0 and self.master:
        warn(
            "No intersections found between the source grid and the proxy raster. "
            + "The source cells will fall back to the lower priority proxies or to the plain area conservative "
            "weights."
        )
        sys.stderr.flush()
    if info and self.master:
        print(f"\t\t\tSource to proxy: {len(fid_src)} intersections found (master process)")
        sys.stdout.flush()

    if len(fid_src) == 0:
        # The source cells of this rank do not intersect the proxy raster
        covered_fids = array([], dtype=uint32)
        intersection_df = DataFrame(
            {"FID_src": array([], dtype=uint32), "FID_dst": array([], dtype=uint32), "weight": array([], dtype=float64)}
        )
        return intersection_df, covered_fids

    # Calculate intersected areas and fractions
    proxy_df = DataFrame(columns=["FID_src", "FID_pxy"])

    # Filling the proxy DataFrame with the intersected FID values
    proxy_df["FID_src"] = array(src_grid.loc[fid_src, "FID_src"], dtype=uint32)
    proxy_df["FID_pxy"] = array(pxy_grid.loc[fid_proxy, "FID_pxy"], dtype=uint32)

    # Filling the proxy DataFrame with the intersected geometries
    proxy_df["geometry_src"] = src_grid.loc[fid_src, "geometry"].values
    proxy_df["geometry_pxy"] = pxy_grid.loc[fid_proxy, "geometry"].values

    if info and self.master:
        print("\t\t\tSource to proxy weight", flush=True)

    # Calculating the weight for each intersection based on the area of intersection and the area of the
    # source geometry
    proxy_df["weight"] = array(
        proxy_df.apply(
            lambda x: x["geometry_src"].intersection(x["geometry_pxy"]).buffer(0).area / x["geometry_src"].area, axis=1
        ),
        dtype=float64,
    )

    # Drop border-only touches (zero overlap area): they would divide by zero later and must not mark
    # a source cell as covered when a proxy pixel merely shares an edge with it
    proxy_df = proxy_df.loc[proxy_df["weight"] > 0].reset_index(drop=True)
    covered_fids = proxy_df["FID_src"].unique()

    if len(proxy_df) == 0:
        intersection_df = DataFrame(
            {"FID_src": array([], dtype=uint32), "FID_dst": array([], dtype=uint32), "weight": array([], dtype=float64)}
        )
        return intersection_df, covered_fids

    # Calculating the proxy value for each intersection based on the weight and the value of the proxy geometry
    proxy_df["pxy_value"] = pxy_grid.loc[proxy_df["FID_pxy"], "value"].values
    proxy_df["pxy_value"] = proxy_df["weight"] * proxy_df["pxy_value"]

    # Updating the proxy geometry to be the intersection of the source and proxy geometries
    proxy_df["geometry_pxy"] = proxy_df.apply(lambda x: x["geometry_pxy"].intersection(x["geometry_src"]), axis=1)
    # Creating a GeoDataFrame from the proxy DataFrame with the updated geometries and the same CRS as the
    # source grid
    proxy_df = gpd.GeoDataFrame(proxy_df, geometry=proxy_df["geometry_pxy"], crs=src_grid.crs)

    # Dropping the weight column as it is no longer needed
    proxy_df.drop(columns="weight", inplace=True)

    # Normalizing the proxy values by dividing each proxy value by the sum of proxy values for each source
    # geometry. All the pixels of a source cell live on the same rank, so this is a local operation.
    proxy_df["pxy_value"] = proxy_df["pxy_value"].div(proxy_df.groupby("FID_src")["pxy_value"].transform("sum"))

    if info and self.master:
        print("\t\t\tProxy to destination")
        sys.stdout.flush()

    # Querying the destination grid to find intersections with the proxy geometries
    fid_src_pxy, fid_dst = dst_grid.sindex.query(proxy_df.geometry, predicate="intersects")
    if info and self.master:
        print(f"\t\t\tProxy to destination: {len(fid_dst)} intersections found (master process)")
        sys.stdout.flush()

    # Calculate intersected areas and fractions
    intersection_df = DataFrame(columns=["FID_src", "FID_pxy", "FID_dst"])

    intersection_df["FID_src"] = array(proxy_df.loc[fid_src_pxy, "FID_src"], dtype=uint32)
    intersection_df["FID_pxy"] = array(proxy_df.loc[fid_src_pxy, "FID_pxy"], dtype=uint32)
    intersection_df["FID_dst"] = array(dst_grid.loc[fid_dst, "FID_dst"], dtype=uint32)

    intersection_df["geometry_src"] = proxy_df.loc[fid_src_pxy, "geometry_src"].values
    intersection_df["geometry_pxy"] = proxy_df.loc[fid_src_pxy, "geometry_pxy"].values
    intersection_df["geometry_dst"] = dst_grid.loc[fid_dst, "geometry"].values

    intersection_df["pxy_value"] = proxy_df.loc[fid_src_pxy, "pxy_value"].values

    if info and self.master:
        print("\t\t\tProxy to destination weight")
        sys.stdout.flush()

    # Calculating the weight for each intersection based on the area of intersection and the area of the
    # proxy geometry
    if len(intersection_df) > 0:
        intersection_df["pxy_value"] = array(
            intersection_df.apply(
                lambda x: (
                    x["pxy_value"]
                    * (x["geometry_pxy"].intersection(x["geometry_dst"]).buffer(0).area / x["geometry_pxy"].area)
                ),
                axis=1,
            ),
            dtype=float64,
        )

    intersection_df = intersection_df.groupby(["FID_src", "FID_dst"])["pxy_value"].sum().reset_index()
    intersection_df.rename(columns={"pxy_value": "weight"}, inplace=True)
    if info and self.master:
        print(f"\t\t\tProxy weights aggregated (master process: {len(intersection_df)} source-destination pairs)")
        sys.stdout.flush()

    return intersection_df, covered_fids


def __to_equal_area(geometries):
    """
    Project lon/lat (degree) geometries onto the spherical cylindrical equal-area plane (m).

    x = R * lon and y = R * sin(lat), with R = CONSERVATIVE_AREA_RADIUS. Parallels and meridians stay
    straight, so shapely areas are the exact spherical areas of the lat/lon cells and are additive.
    Unlike pyproj, longitudes are not wrapped to [-180, 180], so the cells of 0-360 grids that touch or
    cross 180 keep their shape instead of turning into strips spanning the whole latitude band.

    Parameters
    ----------
    geometries : array-like of shapely.Geometry
        Geometries with lon/lat coordinates in degrees.

    Returns
    -------
    numpy.ndarray of shapely.Geometry
        Geometries in the equal-area plane, in m.
    """
    return shapely.transform(
        asarray(geometries),
        lambda coords: column_stack(
            [
                CONSERVATIVE_AREA_RADIUS * radians(coords[:, 0]),
                CONSERVATIVE_AREA_RADIUS * sin(radians(coords[:, 1])),
            ]
        ),
    )


def __plain_area_intersections(self, src_grid, dst_grid, info=False):
    """
    Compute mass weights using overlap and source areas in m2.

    Intersections are computed in the same equal-area projection used for
    the flux conversion, so subdivided cell areas remain additive.

    Parameters
    ----------
    self : nes.Nes
        Source projection Nes Object.
    src_grid : GeoDataFrame
        Rank-local portion of the source grid shapefile.
    dst_grid : GeoDataFrame
        Complete destination grid shapefile.
    info : bool
        Indicates if you want to print extra info during the methods process.

    Returns
    -------
    DataFrame
        Intersection DataFrame with the FID_src, FID_dst and weight columns.
    """
    src_geoms = __to_equal_area(src_grid.geometry.values)
    dst_geoms = __to_equal_area(dst_grid.geometry.values)

    fid_src, fid_dst = shapely.STRtree(dst_geoms).query(src_geoms, predicate="intersects")
    if info and self.master:
        print(f"\t\t\tSource to destination: {len(fid_dst)} intersections found (master process)")
        sys.stdout.flush()

    intersection_df = DataFrame(
        {
            "FID_src": array(src_grid["FID_src"].values[fid_src], dtype=uint32),
            "FID_dst": array(dst_grid["FID_dst"].values[fid_dst], dtype=uint32),
        }
    )

    if len(intersection_df) > 0:
        geom_src = src_geoms[fid_src]
        geom_dst = dst_geoms[fid_dst]
        intersections = shapely.intersection(geom_src, geom_dst)
        intersection_df["weight"] = array(
            shapely.area(intersections) / shapely.area(geom_src),
            dtype=float64,
        )
    else:
        intersection_df["weight"] = array([], dtype=float64)

    return intersection_df


def __mass_to_flux_weights(
    intersection_df: DataFrame,
    src_grid: gpd.GeoDataFrame,
    dst_grid: gpd.GeoDataFrame,
) -> DataFrame:
    """
    Convert mass weights into flux (per unit area) weights: w_flux = w_mass * A_src / A_dst.

    The mass weights are the fractions of the source cell mass sent to each destination cell, whatever the
    proxy that produced them, so this conversion is valid for every proxy type and for the plain area
    fallback. Cell areas are measured in m2 in the same spherical cylindrical equal-area
    projection as the plain-area intersections.

    Parameters
    ----------
    intersection_df : DataFrame
        Intersection DataFrame with the FID_src, FID_dst and (mass) weight columns.
    src_grid : GeoDataFrame
        Rank-local portion of the source grid shapefile.
    dst_grid : GeoDataFrame
        Complete destination grid shapefile.

    Returns
    -------
    DataFrame
        Intersection DataFrame with the flux weights.
    """
    if len(intersection_df) == 0:
        return intersection_df
    # Only the cells that appear in the intersections are measured
    fid_src = intersection_df["FID_src"].values
    fid_dst = intersection_df["FID_dst"].values
    src_ids, src_inv = unique(fid_src, return_inverse=True)
    dst_ids, dst_inv = unique(fid_dst, return_inverse=True)
    src_area = shapely.area(__to_equal_area(src_grid.set_index("FID_src").loc[src_ids, "geometry"].values))
    dst_area = shapely.area(__to_equal_area(dst_grid.set_index("FID_dst").loc[dst_ids, "geometry"].values))
    intersection_df["weight"] = intersection_df["weight"].values * src_area[src_inv] / dst_area[dst_inv]
    return intersection_df


# noinspection DuplicatedCode
def __create_area_conservative_weight_matrix(
    self, dst_nes, wm_path=None, flux=False, info=False, proxies=None, column_name="weight"
):
    """
    To create the weight matrix with the area conservative method.

    Parameters
    ----------
    self : nes.Nes
        Source projection Nes Object.
    dst_nes : nes.Nes
        Final projection Nes object.
    wm_path : str
        Path where write the weight matrix.
    flux : bool
        Indicates if you want to calculate the weight matrix for flux variables. The (mass) weights of every
        proxy and of the plain area fallback are converted with w_flux = w_mass * A_src / A_dst, where the
        cell areas are measured in m2 in the same spherical cylindrical equal-area projection
        as the plain-area intersections.
    info: bool
        Indicates if you want to print extra info during the methods process.
    proxies : list, None
        Priority-ordered list of (proxy_type, path) tuples to use for the interpolation: each source
        cell uses the first proxy that covers it and the uncovered cells fall back to the plain area
        conservative weights. If None, no proxy is used.
    column_name : str
        Name of the column holding the feature weights in the vector proxies.

    Notes
    -----
    The intersections are computed in parallel with MPI: every rank works on a disjoint portion of
    the source grid against the complete destination grid, and the master rank gathers the results
    to build the weight matrix. Therefore, all ranks must call this function.

    Returns
    -------
    nes.Nes
        Weight matrix.
    """
    if info and self.master:
        print("\tCreating area conservative Weight Matrix")
        sys.stdout.flush()

    my_crs = CRS.from_proj4("+proj=latlon")  # Common projection for both shapefiles
    coord_dtype = __lowest_coordinate_dtype(self, dst_nes, comm=self.comm)
    if info and self.master:
        print(f"\t\tCoordinate dtype: {coord_dtype}")
        sys.stdout.flush()

    if info and self.master:
        print("\t\t[1/5] Preparing source and destination shapefiles")
        sys.stdout.flush()

    # Get a portion of the destiny grid
    if dst_nes.shapefile is None:
        dst_nes.create_shapefile()
    dst_grid: gpd.geodataframe.GeoDataFrame = deepcopy(dst_nes.shapefile)

    # Formatting Destination grid
    dst_grid.to_crs(crs=my_crs, inplace=True)
    __apply_coordinate_precision(dst_grid, coord_dtype)
    dst_grid["FID_dst"] = dst_grid.index

    # Preparing Source grid
    if self.shapefile is None:
        self.create_shapefile()
    src_grid: gpd.geodataframe.GeoDataFrame = deepcopy(self.shapefile)

    # Formatting Source grid
    src_grid.to_crs(crs=my_crs, inplace=True)
    __apply_coordinate_precision(src_grid, coord_dtype)
    src_grid["FID_src"] = src_grid.index

    # Distribute the intersection workload by source cells: every rank keeps a disjoint portion of
    # the source grid and the complete destination grid.
    if self.size > 1:
        if info and self.master:
            print("\t\t[2/5] Distributing the workload among the processes")
            sys.stdout.flush()
        if self.parallel_method == "T":
            # Every rank has the full spatial domain: keep only a disjoint chunk of the source cells
            src_grid = gpd.GeoDataFrame(array_split(src_grid, self.size)[self.rank])
        else:
            # The source cells are already distributed (X/Y): replicate the destination grid on all ranks
            dst_grid = gpd.GeoDataFrame(concat(self.comm.allgather(dst_grid)))

    src_grid = src_grid.reset_index(drop=True)
    dst_grid = dst_grid.reset_index(drop=True)

    if info and self.master:
        print(
            f"\t\t[3/5] Computing intersections "
            f"(master process: {len(src_grid)} source cells, {len(dst_grid)} destination cells)"
        )
        sys.stdout.flush()

    # Priority chain: separate proxy datasets are applied in the user-defined order. A
    # multigeometry proxy is an atomic dataset, so the point/line/polygon parts inside it are
    # handled together with no internal priority between geometry types. Any cells left after all
    # proxy steps fall back to the plain area conservative weights.
    intersection_parts = []
    remaining_grid = src_grid
    if proxies is not None:
        for pxy_pos, (pxy_type, pxy_path, pxy_categories, pxy_column) in enumerate(proxies):
            vector_column = pxy_column if pxy_column is not None else column_name
            if info and self.master:
                print(
                    f"\t\t\tProxy {pxy_pos + 1}/{len(proxies)} ({pxy_type}): "
                    f"{len(remaining_grid)} source cells to cover (master process)"
                )
                sys.stdout.flush()
            if pxy_type == "points":
                partial_df, covered_fids = __proxy_points_intersections(
                    self,
                    pxy_path,
                    remaining_grid,
                    dst_grid,
                    dst_nes,
                    column_name=vector_column,
                    info=info,
                    coord_dtype=coord_dtype,
                )
            elif pxy_type == "lines":
                partial_df, covered_fids = __proxy_lines_intersections(
                    self,
                    pxy_path,
                    remaining_grid,
                    dst_grid,
                    column_name=vector_column,
                    info=info,
                    coord_dtype=coord_dtype,
                )
            elif pxy_type == "polygons":
                partial_df, covered_fids = __proxy_polygons_intersections(
                    self,
                    pxy_path,
                    remaining_grid,
                    dst_grid,
                    column_name=vector_column,
                    info=info,
                    coord_dtype=coord_dtype,
                )
            elif pxy_type == "multigeometry":
                partial_df, covered_fids = __proxy_multigeometry_intersections(
                    self,
                    pxy_path,
                    remaining_grid,
                    dst_grid,
                    dst_nes,
                    column_name=vector_column,
                    info=info,
                    coord_dtype=coord_dtype,
                )
            elif pxy_type == "raster_categorical":
                partial_df, covered_fids = __proxy_raster_intersections(
                    self,
                    pxy_path,
                    remaining_grid,
                    dst_grid,
                    info=info,
                    category_list=pxy_categories,
                    coord_dtype=coord_dtype,
                )
            elif pxy_type == "raster_categorical_numerical":
                cat_path, num_path = pxy_path
                partial_df, covered_fids = __proxy_raster_intersections(
                    self,
                    cat_path,
                    remaining_grid,
                    dst_grid,
                    info=info,
                    category_list=pxy_categories,
                    numerical_path=num_path,
                    coord_dtype=coord_dtype,
                )
            else:
                partial_df, covered_fids = __proxy_raster_intersections(
                    self, pxy_path, remaining_grid, dst_grid, info=info, coord_dtype=coord_dtype
                )
            intersection_parts.append(partial_df)
            # Lower-priority proxies only see the cells not covered by this one
            remaining_grid = remaining_grid.loc[~remaining_grid["FID_src"].isin(covered_fids)].reset_index(drop=True)

        if info and self.master:
            print(
                f"\t\t\tSource cells covered by no proxy falling back to plain area conservative "
                f"weights (master process: {len(remaining_grid)} cells)"
            )
            sys.stdout.flush()

    # The cells covered by no proxy get the plain area conservative weights, so that their mass is not lost
    fallback_df = __plain_area_intersections(self, remaining_grid, dst_grid, info=info)
    intersection_parts.append(fallback_df)
    intersection_df = concat(intersection_parts, ignore_index=True)

    if flux:
        # Every part holds mass weights (fraction of the source cell mass sent to each destination cell),
        # so the flux weights are obtained in a single step for all the proxy types and the fallback:
        # f_dst = sum(w * f_src * A_src) / A_dst
        intersection_df = __mass_to_flux_weights(intersection_df, src_grid, dst_grid)
    del src_grid, dst_grid, remaining_grid, intersection_parts

    if info and self.master:
        print("\t\t[4/5] Intersections computed and weights calculated")
        sys.stdout.flush()

    # Format & Clean
    if info and self.master:
        print("\t\t[5/5] Weights calculated. Formatting weight matrix")
        sys.stdout.flush()

    # Gather all the intersections on the master process to initialise the weight matrix
    intersection_df = self.comm.gather(intersection_df, root=0)
    if self.master:
        intersection_df = concat(intersection_df, ignore_index=True)
        intersection_df = (
            intersection_df.set_index(["FID_dst", intersection_df.groupby("FID_dst").cumcount()])
            .rename_axis(("FID", "level"))
            .sort_index()
        )
        intersection_df.rename(columns={"FID_src": "idx"}, inplace=True)
        weight_matrix = dst_nes.copy()
        weight_matrix.time = [datetime(year=2000, month=1, day=1, hour=0, second=0, microsecond=0)]
        weight_matrix._time = [datetime(year=2000, month=1, day=1, hour=0, second=0, microsecond=0)]
        weight_matrix._time_bnds = None
        weight_matrix.time_bnds = None
        weight_matrix.last_level = None
        weight_matrix.first_level = 0
        weight_matrix.hours_start = 0
        weight_matrix.hours_end = 0

        weight_matrix.set_communicator(MPI.COMM_SELF)

        weight_matrix.set_levels(
            {
                "data": arange(intersection_df.index.get_level_values("level").max() + 1),
                "dimensions": ("lev",),
                "units": "",
                "positive": "up",
            }
        )

        if info:
            print(f"\t\t\tFilling weight matrix variables ({len(weight_matrix.lev['data'])} levels)")
            sys.stdout.flush()

        # Creating Weight matrix empty variables
        if len(weight_matrix._full_lat["data"].shape) == 1:
            shape = (
                1,
                len(weight_matrix.lev["data"]),
                weight_matrix._full_lat["data"].shape[0],
                weight_matrix._full_lon["data"].shape[0],
            )
            shape_flat = (
                1,
                len(weight_matrix.lev["data"]),
                weight_matrix._full_lat["data"].shape[0] * weight_matrix._full_lon["data"].shape[0],
            )
        else:
            shape = (
                1,
                len(weight_matrix.lev["data"]),
                weight_matrix._full_lat["data"].shape[0],
                weight_matrix._full_lat["data"].shape[1],
            )
            shape_flat = (
                1,
                len(weight_matrix.lev["data"]),
                weight_matrix._full_lat["data"].shape[0] * weight_matrix._full_lat["data"].shape[1],
            )

        weight_matrix.variables["weight"] = {"data": empty(shape_flat), "units": "-"}
        weight_matrix.variables["weight"]["data"][:] = -999
        weight_matrix.variables["idx"] = {"data": empty(shape_flat), "units": "-"}
        weight_matrix.variables["idx"]["data"][:] = -999

        # Filling Weight matrix variables
        for aux_lev in weight_matrix.lev["data"]:
            aux_data = intersection_df.xs(level="level", key=aux_lev)
            weight_matrix.variables["weight"]["data"][0, aux_lev, aux_data.index] = aux_data.loc[:, "weight"].values
            weight_matrix.variables["idx"]["data"][0, aux_lev, aux_data.index] = aux_data.loc[:, "idx"].values

        # Re-shaping
        weight_matrix.variables["weight"]["data"] = weight_matrix.variables["weight"]["data"].reshape(shape)
        weight_matrix.variables["idx"]["data"] = weight_matrix.variables["idx"]["data"].reshape(shape)
        if __uses_point_proxies(proxies):
            weight_matrix.global_attrs = dict(weight_matrix.global_attrs or {})
            weight_matrix.global_attrs[POINT_OWNERSHIP_ATTRIBUTE] = POINT_OWNERSHIP_POLICY
        if wm_path is not None:
            if info and self.master:
                print(f"\t\tWeight matrix saved at {wm_path}")
                sys.stdout.flush()
            weight_matrix.to_netcdf(wm_path)
    else:
        weight_matrix = True
    return weight_matrix


# noinspection DuplicatedCode
def __lon_lat_to_cartesian(lon, lat, radius=6378137.0):
    """
    Calculate lon, lat coordinates of a point on a sphere.

    DEPRECATED!!!!

    Parameters
    ----------
    lon : array
        Longitude values.
    lat : array
        Latitude values.
    radius : float
        Radius of the sphere to get the distances.
    """

    lon_r = radians(lon)
    lat_r = radians(lat)

    x = radius * cos(lat_r) * cos(lon_r)
    y = radius * cos(lat_r) * sin(lon_r)
    z = radius * sin(lat_r)

    return column_stack([x, y, z])


def __lon_lat_to_cartesian_ecef(lon, lat):
    """
    Convert observational/model geographic longitude/latitude coordinates to cartesian ECEF (Earth Centred,
    Earth Fixed) coordinates, assuming WGS84 datum and ellipsoid, and that all heights = 0.
    ECEF coordinates represent positions (in meters) as X, Y, Z coordinates, approximating the earth surface
    as an ellipsoid of revolution.
    This conversion is for the subsequent calculation of Euclidean distances of the model grid cell centres
    from each observational station.
    Defining the distance between two points on the earth's surface as simply the Euclidean distance
    between the two lat/lon pairs could lead to inaccurate results depending on the distance
    between two points (i.e. 1 deg. of longitude varies with latitude).

    Parameters
    ----------
    lon : array
        Longitude values.
    lat : array
        Latitude values.
    """

    lla = Proj(proj="latlong", ellps="WGS84", datum="WGS84")
    ecef = Proj(proj="geocent", ellps="WGS84", datum="WGS84")

    # x, y, z = pyproj.transform(lla, ecef, lon, lat, zeros(lon.shape), radians=False)
    # Deprecated: https://pyproj4.github.io/pyproj/stable/gotchas.html#upgrading-to-pyproj-2-from-pyproj-1
    transformer = Transformer.from_proj(lla, ecef)
    x, y, z = transformer.transform(lon, lat, zeros(lon.shape), radians=False)
    return column_stack([x, y, z])
