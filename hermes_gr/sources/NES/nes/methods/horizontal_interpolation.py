#!/usr/bin/env python
# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import sys
import os
import nes
from warnings import warn, filterwarnings
from numpy import (ma, empty, nansum, sum, concatenate, pad, nan, array, float64, int64, float32, meshgrid, expand_dims,
                   reciprocal, arange, uint32, array_split, radians, cos, sin, column_stack, zeros)
from pandas import concat, DataFrame
from mpi4py import MPI
from scipy import spatial
from filelock import FileLock
from datetime import datetime
from copy import deepcopy
from pyproj import Proj, Transformer, CRS
import gc

# CONSTANTS
NEAREST_OPTS = ["NearestNeighbour", "NearestNeighbours", "nn", "NN"]
CONSERVATIVE_OPTS = ["Conservative", "Area_Conservative", "cons", "conservative", "area"]


def interpolate_horizontal(self, dst_grid, weight_matrix_path=None, kind="NearestNeighbour", n_neighbours=4,
                           info=False, to_providentia=False, only_create_wm=False, wm=None, flux=False, keep_nan=False,
                           fix_border=False):
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
        Indicates if you want to calculate the weight matrix for flux variables.
    keep_nan : bool
        Indicates if you want to keep nan values after the interpolation
    fix_border : bool
        Indicates if you want to fix the borders on the NN interpolation by expanding it 1 cell and contracting
        it later.
    """
    if info and self.master:
        print("Creating Weight Matrix")

    # Obtain weight matrix
    if self.parallel_method == "T":
        weights, idx = __get_weights_idx_t_axis(self, dst_grid, weight_matrix_path, kind, n_neighbours,
                                                only_create_wm, wm, flux, fix_border)
    elif self.parallel_method in ["Y", "X"]:

        weights, idx = __get_weights_idx_xy_axis(self, dst_grid, weight_matrix_path, kind, n_neighbours,
                                                 only_create_wm, wm, flux, fix_border)
    else:
        raise NotImplementedError("Parallel method {0} is not implemented yet for horizontal interpolations.".format(
            self.parallel_method) + "Use 'T'")

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
    weights = ma.masked_array(weights, mask=weights == -999)
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
            print("\t{var} horizontal interpolation".format(var=var_name))
            sys.stdout.flush()
        src_shape = var_info["data"].shape
        if isinstance(dst_grid, nes.PointsNes):
            dst_shape = (src_shape[0], src_shape[1], idx.shape[-1])
        else:
            dst_shape = (src_shape[0], src_shape[1], idx.shape[-2], idx.shape[-1])
        # Creating new variable without data
        final_dst.variables[var_name] = {attr_name: attr_value for attr_name, attr_value in var_info.items()
                                         if attr_name != "data"}
        # Creating empty data
        final_dst.variables[var_name]["data"] = empty(dst_shape)

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
                (src_shape[0], idx.shape[-1]))
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
            final_dst = final_dst.to_providentia(model_centre_lon=model_centre_lon,
                                                 model_centre_lat=model_centre_lat,
                                                 grid_edge_lon=grid_edge_lon,
                                                 grid_edge_lat=grid_edge_lat)
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


# noinspection DuplicatedCode
def __get_weights_idx_t_axis(self, dst_grid, weight_matrix_path, kind, n_neighbours, only_create, wm, flux, fix_border):
    """
    To obtain the weights and source data index through the T axis.

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
    Returns
    -------
    tuple
        Weights and source data index.
    """
    weight_matrix = None

    if wm is not None:
        weight_matrix = wm

    elif weight_matrix_path is not None:
        with FileLock(weight_matrix_path + "{0:03d}.lock".format(self.rank)):
            if os.path.isfile(weight_matrix_path):
                if self.master:
                    weight_matrix = __read_weight_matrix(weight_matrix_path, comm=MPI.COMM_SELF)
                else:
                    weight_matrix = True
                if kind in NEAREST_OPTS:
                    if self.master:
                        if len(weight_matrix.lev["data"]) != n_neighbours:
                            warn("The selected weight matrix does not have the same number of nearest neighbours." +
                                 "Re-calculating again but not saving it.")
                            sys.stderr.flush()
                            weight_matrix = __create_nn_weight_matrix(self, dst_grid, n_neighbours=n_neighbours,
                                                                      fix_border=fix_border)
                    else:
                        weight_matrix = True

            else:
                if self.master:
                    if kind in NEAREST_OPTS:
                        weight_matrix = __create_nn_weight_matrix(self, dst_grid, n_neighbours=n_neighbours,
                                                                  wm_path=weight_matrix_path, fix_border=fix_border)
                    elif kind in CONSERVATIVE_OPTS:
                        weight_matrix = __create_area_conservative_weight_matrix(
                            self, dst_grid, wm_path=weight_matrix_path, flux=flux)
                    else:
                        raise NotImplementedError(kind)
                else:
                    weight_matrix = True

        if os.path.exists(weight_matrix_path + "{0:03d}.lock".format(self.rank)):
            os.remove(weight_matrix_path + "{0:03d}.lock".format(self.rank))
    else:
        if self.master:
            if kind in NEAREST_OPTS:
                weight_matrix = __create_nn_weight_matrix(self, dst_grid, n_neighbours=n_neighbours,
                                                          fix_border=fix_border)
            elif kind in CONSERVATIVE_OPTS:
                weight_matrix = __create_area_conservative_weight_matrix(self, dst_grid, flux=flux)
            else:
                raise NotImplementedError(kind)
        else:
            weight_matrix = True

    if only_create:
        return weight_matrix, None

    if self.master:
        if kind in NEAREST_OPTS:
            # Normalize to 1
            weights = array(array(weight_matrix.variables["weight"]["data"], dtype=float64) /
                            array(weight_matrix.variables["weight"]["data"], dtype=float64).sum(axis=1),
                            dtype=float64)
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
def __get_weights_idx_xy_axis(self, dst_grid, weight_matrix_path, kind, n_neighbours, only_create, wm, flux,
                              fix_border):
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
        with FileLock(weight_matrix_path + "{0:03d}.lock".format(self.rank)):
            if os.path.isfile(weight_matrix_path):
                if self.master:
                    weight_matrix = __read_weight_matrix(weight_matrix_path, comm=MPI.COMM_SELF)
                else:
                    weight_matrix = True
                if kind in NEAREST_OPTS:
                    if self.master:
                        if len(weight_matrix.lev["data"]) != n_neighbours:
                            warn("The selected weight matrix does not have the same number of nearest neighbours." +
                                 "Re-calculating again but not saving it.")
                            sys.stderr.flush()
                            weight_matrix = __create_nn_weight_matrix(self, dst_grid, n_neighbours=n_neighbours,
                                                                      fix_border=fix_border)
                    else:
                        weight_matrix = True
            else:
                if kind in NEAREST_OPTS:
                    if self.master:
                        weight_matrix = __create_nn_weight_matrix(self, dst_grid, n_neighbours=n_neighbours,
                                                                  wm_path=weight_matrix_path, fix_border=fix_border)
                    else:
                        weight_matrix = True
                elif kind in CONSERVATIVE_OPTS:
                    weight_matrix = __create_area_conservative_weight_matrix(
                        self, dst_grid, wm_path=weight_matrix_path, flux=flux)
                else:
                    raise NotImplementedError(kind)

        if os.path.exists(weight_matrix_path + "{0:03d}.lock".format(self.rank)):
            os.remove(weight_matrix_path + "{0:03d}.lock".format(self.rank))
    else:
        if kind in NEAREST_OPTS:
            if self.master:
                weight_matrix = __create_nn_weight_matrix(self, dst_grid, n_neighbours=n_neighbours,
                                                          fix_border=fix_border)
            else:
                weight_matrix = True
        elif kind in CONSERVATIVE_OPTS:
            weight_matrix = __create_area_conservative_weight_matrix(self, dst_grid, flux=flux)
        else:
            raise NotImplementedError(kind)

    if only_create:
        return weight_matrix, None

    # Normalize to 1
    if self.master:
        if kind in NEAREST_OPTS:
            weights = array(array(weight_matrix.variables["weight"]["data"], dtype=float64) /
                            array(weight_matrix.variables["weight"]["data"], dtype=float64).sum(axis=1),
                            dtype=float64)
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
    weights = weights[:, :, dst_grid.write_axis_limits["y_min"]:dst_grid.write_axis_limits["y_max"],
                      dst_grid.write_axis_limits["x_min"]:dst_grid.write_axis_limits["x_max"]]
    idx = idx[:, dst_grid.write_axis_limits["y_min"]:dst_grid.write_axis_limits["y_max"],
              dst_grid.write_axis_limits["x_min"]:dst_grid.write_axis_limits["x_max"]]
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
    if "inverse_dists" in weight_matrix.variables.keys():
        weight_matrix.variables["weight"] = weight_matrix.variables["inverse_dists"]

    weight_matrix.variables["weight"]["data"][weight_matrix.variables["weight"]["data"] <= 0] = nan
    weight_matrix.variables["weight"]["data"][weight_matrix.variables["idx"]["data"] <= 0] = nan

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
    if fix_border:
        dst_grid.expand(n_cells=1)
    # Only master is here.
    if info and self.master:
        print("\tCreating Nearest Neighbour Weight Matrix with {0} neighbours".format(n_neighbours))
        sys.stdout.flush()
    # Source
    src_lat = array(self._full_lat["data"], dtype=float32)
    src_lon = array(self._full_lon["data"], dtype=float32)

    # 1D to 2D coordinates
    if len(src_lon.shape) == 1:
        src_lon, src_lat = meshgrid(src_lon, src_lat)

    # Destination
    dst_lat = array(dst_grid._full_lat["data"], dtype=float32)
    dst_lon = array(dst_grid._full_lon["data"], dtype=float32)

    if isinstance(dst_grid, nes.PointsNes):
        dst_lat = expand_dims(dst_grid._full_lat["data"], axis=0)
        dst_lon = expand_dims(dst_grid._full_lon["data"], axis=0)
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


# noinspection DuplicatedCode
def __create_area_conservative_weight_matrix(self, dst_nes, wm_path=None, flux=False, info=False):
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
        Indicates if you want to calculate the weight matrix for flux variables.
    info: bool
        Indicates if you want to print extra info during the methods process.

    Returns
    -------
    nes.Nes
        Weight matrix.
    """

    if info and self.master:
        print("\tCreating area conservative Weight Matrix")
        sys.stdout.flush()

    my_crs = CRS.from_proj4("+proj=latlon")  # Common projection for both shapefiles

    # Get a portion of the destiny grid
    if dst_nes.shapefile is None:
        dst_nes.create_shapefile()
    dst_grid = deepcopy(dst_nes.shapefile)

    # Formatting Destination grid
    dst_grid.to_crs(crs=my_crs, inplace=True)
    dst_grid["FID_dst"] = dst_grid.index

    # Preparing Source grid
    if self.shapefile is None:
        self.create_shapefile()
    src_grid = deepcopy(self.shapefile)

    # Formatting Source grid
    src_grid.to_crs(crs=my_crs, inplace=True)

    # Serialize index intersection function to avoid memory problems
    if self.size > 1 and self.parallel_method != "T":
        src_grid = self.comm.gather(src_grid, root=0)
        dst_grid = self.comm.gather(dst_grid, root=0)
        if self.master:
            src_grid = concat(src_grid)
            dst_grid = concat(dst_grid)
    if self.master:
        src_grid["FID_src"] = src_grid.index
        src_grid = src_grid.reset_index()
        dst_grid = dst_grid.reset_index()
        fid_src, fid_dst = dst_grid.sindex.query(src_grid.geometry, predicate="intersects")

        # Calculate intersected areas and fractions
        intersection_df = DataFrame(columns=["FID_src", "FID_dst"])

        intersection_df["FID_src"] = array(src_grid.loc[fid_src, "FID_src"], dtype=uint32)
        intersection_df["FID_dst"] = array(dst_grid.loc[fid_dst, "FID_dst"], dtype=uint32)

        intersection_df["geometry_src"] = src_grid.loc[fid_src, "geometry"].values
        intersection_df["geometry_dst"] = dst_grid.loc[fid_dst, "geometry"].values
        del src_grid, dst_grid, fid_src, fid_dst
        # Split the array into smaller arrays in order to scatter the data among the processes
        intersection_df = array_split(intersection_df, self.size)
    else:
        intersection_df = None

    intersection_df = self.comm.scatter(intersection_df, root=0)

    if info and self.master:
        print("\t\tGrids created and ready to interpolate")
        sys.stdout.flush()
    if True:
        # No Warnings Zone
        filterwarnings("ignore")
        # intersection_df["weight"] = array(intersection_df.apply(
        #     lambda x: x["geometry_src"].intersection(x["geometry_dst"]).buffer(0).area / x["geometry_src"].area,
        #     axis=1), dtype=float64)
        if flux:
            intersection_df["weight"] = array(intersection_df.apply(
                lambda x: (x["geometry_src"].intersection(x["geometry_dst"]).buffer(0).area / x["geometry_src"].area) *
                          (nes.Nes.calculate_geometry_area([x["geometry_src"]])[0] /
                           nes.Nes.calculate_geometry_area([x["geometry_dst"]])[0]),
                axis=1), dtype=float64)
        else:
            intersection_df["weight"] = array(intersection_df.apply(
                lambda x: x["geometry_src"].intersection(x["geometry_dst"]).buffer(0).area / x["geometry_src"].area,
                axis=1), dtype=float64)

        intersection_df.drop(columns=["geometry_src", "geometry_dst"], inplace=True)
        gc.collect()
        filterwarnings("default")

    # Format & Clean
    if info and self.master:
        print("\t\tWeights calculated. Formatting weight matrix.")
        sys.stdout.flush()

    # Initialising weight matrix
    if self.parallel_method != "T":
        intersection_df = self.comm.gather(intersection_df, root=0)
    if self.master:
        if self.parallel_method != "T":
            intersection_df = concat(intersection_df)
        intersection_df = intersection_df.set_index(
            ["FID_dst", intersection_df.groupby("FID_dst").cumcount()]).rename_axis(("FID", "level")).sort_index()
        intersection_df.rename(columns={"FID_src": "idx"}, inplace=True)

        weight_matrix = dst_nes.copy(new_comm=MPI.COMM_SELF)
        weight_matrix.time = [datetime(year=2000, month=1, day=1, hour=0, second=0, microsecond=0)]
        weight_matrix._full_time = [datetime(year=2000, month=1, day=1, hour=0, second=0, microsecond=0)]
        weight_matrix._full_time_bnds = None
        weight_matrix.time_bnds = None
        weight_matrix.last_level = None
        weight_matrix.first_level = 0
        weight_matrix.hours_start = 0
        weight_matrix.hours_end = 0

        weight_matrix.set_levels({"data": arange(intersection_df.index.get_level_values("level").max() + 1),
                                  "dimensions": ("lev",),
                                  "units": "",
                                  "positive": "up"})

        # Creating Weight matrix empty variables
        wm_shape = weight_matrix.get_full_shape()
        shape = (1, len(weight_matrix.lev["data"]), wm_shape[0], wm_shape[1],)
        shape_flat = (1, len(weight_matrix.lev["data"]), wm_shape[0] * wm_shape[1],)

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
        if wm_path is not None:
            if info and self.master:
                print("\t\tWeight matrix saved at {0}".format(wm_path))
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
