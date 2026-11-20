#!/usr/bin/env python
# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import sys
from warnings import warn, filterwarnings
from geopandas import sjoin_nearest, sjoin, read_file
from pandas import DataFrame
from numpy import array, uint32, nan
from shapely.errors import TopologicalError


def spatial_join(self, ext_shp, method=None, var_list=None, info=False, apply_bbox=True):
    """
    Compute overlay intersection of two GeoPandasDataFrames.

    Parameters
    ----------
    self : nes.Nes
        A Nes Object.
    ext_shp : GeoPandasDataFrame or str
        File or path from where the data will be obtained on the intersection.
    method : str
        Overlay method. Accepted values: ["nearest", "intersection", "centroid"].
    var_list : List or None or str
        Variables that will be included in the resulting shapefile.
    info : bool
        Indicates if you want to print the process info.
    apply_bbox : bool
        Indicates if you want to reduce the shapefile to a bbox.
    """

    if self.master and info:
        print("Starting spatial join")
    if isinstance(var_list, str):
        # Transforming string (variable name) to a list with length 0
        var_list = [var_list]

    # Create source shapefile if it does not exist
    if self.shapefile is None:
        if self.master and info:
            print("\tCreating shapefile")
            sys.stdout.flush()
        self.create_shapefile()

    if method == "nearest":
        # Nearest method needs the entire shapefile due that the nearest geometry may not be
        # inside the domain decomposition bounding box
        apply_bbox = False

    ext_shp = __prepare_external_shapefile(self, ext_shp=ext_shp, var_list=var_list, info=info,
                                           apply_bbox=apply_bbox)

    if method == "nearest":
        # Nearest centroids to the shapefile polygons
        __spatial_join_nearest(self, ext_shp=ext_shp, info=info)
    elif method == "intersection":
        # Intersect the areas of the shapefile polygons, outside the shapefile there will be NaN
        __spatial_join_intersection(self, ext_shp=ext_shp, info=info)
    elif method == "centroid":
        # Centroids that fall on the shapefile polygons, outside the shapefile there will be NaN
        __spatial_join_centroid(self, ext_shp=ext_shp, info=info)

    else:
        accepted_values = ["nearest", "intersection", "centroid"]
        raise NotImplementedError("{0} is not implemented. Choose from: {1}".format(method, accepted_values))

    return None


def __prepare_external_shapefile(self, ext_shp, var_list, info=False, apply_bbox=True):
    """
    Prepare the external shapefile.

    It is high recommended to pass ext_shp parameter as string because it will clip the external shapefile to the rank.

    1. Read if it is not already read
    2. Filter variables list
    3. Standardize projections

    Parameters
    ----------
    self : nes.Nes
        A Nes Object.
    ext_shp : geopandas.GeoDataFrame or str
        External shapefile or path to it.
    var_list : List[str] or None
        External shapefile variables to be computed.
    info : bool
        Indicates if you want to print the information.
    apply_bbox : bool
        Indicates if you want to reduce the shapefile to a bbox.

    Returns
    -------
    GeoDataFrame
        External shapefile.
    """
    if isinstance(var_list, str):
        var_list = [var_list]

    if isinstance(ext_shp, str):
        # Reading external shapefile
        if self.master and info:
            print("\tReading external shapefile")
        if apply_bbox:
            ext_shp = read_file(ext_shp, bbox=__get_bbox(self))
        else:
            ext_shp = read_file(ext_shp)

        if var_list is not None:
            ext_shp = ext_shp[var_list + ["geometry"]]
    else:
        msg = "WARNING!!! "
        msg += "External shapefile already read. If you pass the path to the shapefile instead of the opened shapefile "
        msg += "a best usage of memory is performed because the external shape will be clipped while reading."
        warn(msg)
        sys.stderr.flush()
        ext_shp.reset_index(inplace=True)
        if var_list is not None:
            ext_shp = ext_shp.loc[:, var_list + ["geometry"]]

    self.comm.Barrier()
    if self.master and info:
        print("\t\tReading external shapefile done!")

    # Standardizing projection
    ext_shp = ext_shp.to_crs(self.shapefile.crs)

    return ext_shp


def __get_bbox(self):
    """
    Obtain the bounding box of the rank data (lon_min, lat_min, lon_max, lat_max).

    Parameters
    ----------
    self : nes.Nes
        A Nes Object.

    Returns
    -------
    tuple
        Bounding box
    """

    bbox = (self.lon_bnds["data"].min(), self.lat_bnds["data"].min(),
            self.lon_bnds["data"].max(), self.lat_bnds["data"].max(), )

    return bbox


# noinspection DuplicatedCode
def __spatial_join_nearest(self, ext_shp, info=False):
    """
    Perform the spatial join using the nearest method.

    Parameters
    ----------
    self : nes.Nes
        A Nes Object.
    ext_shp : GeoDataFrame
        External shapefile.
    info : bool
        Indicates if you want to print the information.
    """

    if self.master and info:
        print("\tNearest spatial joint")
        sys.stdout.flush()
    grid_shp = self.get_centroids_from_coordinates()

    # From geodetic coordinates (e.g. 4326) to meters (e.g. 4328) to use sjoin_nearest
    # TODO: Check if the projection 4328 does not distort the coordinates too much
    # https://gis.stackexchange.com/questions/372564/
    #   userwarning-when-trying-to-get-centroid-from-a-polygon-geopandas
    # ext_shp = ext_shp.to_crs("EPSG:4328")
    # grid_shp = grid_shp.to_crs("EPSG:4328")

    # Calculate spatial joint by distance
    aux_grid = sjoin_nearest(grid_shp, ext_shp, distance_col="distance")

    # Get data from closest shapes to centroids
    del aux_grid["geometry"], aux_grid["index_right"]
    self.shapefile.loc[aux_grid.index, aux_grid.columns] = aux_grid

    var_list = list(ext_shp.columns)
    var_list.remove("geometry")
    for var_name in var_list:
        self.shapefile.loc[:, var_name] = array(self.shapefile.loc[:, var_name], dtype=ext_shp[var_name].dtype)

    return None


# noinspection DuplicatedCode
def __spatial_join_centroid(self, ext_shp, info=False):
    """
    Perform the spatial join using the centroid method.

    Parameters
    ----------
    self : nes.Nes
        A Nes Object.
    ext_shp : GeoDataFrame
        External shapefile.
    info : bool
        Indicates if you want to print the information.
    """

    if self.master and info:
        print("\tCentroid spatial join")
        sys.stdout.flush()
    if info and self.master:
        print("\t\tCalculating centroids")
        sys.stdout.flush()

    # Get centroids
    grid_shp = self.get_centroids_from_coordinates()

    # Calculate spatial joint
    if info and self.master:
        print("\t\tCalculating centroid spatial join")
        sys.stdout.flush()
    aux_grid = sjoin(grid_shp, ext_shp, predicate="within")

    # Get data from shapes where there are centroids, rest will be NaN
    del aux_grid["geometry"], aux_grid["index_right"]
    self.shapefile.loc[aux_grid.index, aux_grid.columns] = aux_grid

    var_list = list(ext_shp.columns)
    var_list.remove("geometry")
    for var_name in var_list:
        self.shapefile.loc[:, var_name] = array(self.shapefile.loc[:, var_name], dtype=ext_shp[var_name].dtype)

    return None


def __spatial_join_intersection(self, ext_shp, info=False):
    """
    Perform the spatial join using the intersection method.

    Parameters
    ----------
    self : nes.Nes
        A Nes Object.
    ext_shp : GeoDataFrame
        External shapefile.
    info : bool
        Indicates if you want to print the information.
    """

    var_list = list(ext_shp.columns)
    var_list.remove("geometry")

    grid_shp = self.shapefile
    grid_shp["FID_grid"] = grid_shp.index
    grid_shp = grid_shp.reset_index()

    # Get intersected areas
    # inp, res = ext_shp.sindex.query(grid_shp.geometry, predicate="intersects")
    inp, res = grid_shp.sindex.query(ext_shp.geometry, predicate="intersects")

    if info:
        print("\t\tRank {0:03d}: {1} intersected areas found".format(self.rank, len(inp)))
        sys.stdout.flush()

    # Calculate intersected areas and fractions
    intersection = DataFrame(columns=["FID", "ext_shp_id", "weight"])
    intersection["FID"] = array(grid_shp.loc[res, "FID_grid"], dtype=uint32)
    intersection["ext_shp_id"] = array(inp, dtype=uint32)

    if len(intersection) > 0:
        if True:
            # No Warnings Zone
            counts = intersection["FID"].value_counts()
            filterwarnings("ignore")
            intersection.loc[:, "weight"] = 1.

            for i, row in intersection.iterrows():
                if isinstance(i, int) and i % 1000 == 0 and info:
                    print("\t\t\tRank {0:03d}: {1:.3f} %".format(self.rank, i * 100 / len(intersection)))
                    sys.stdout.flush()
                # Filter to do not calculate percentages over 100% grid cells spatial joint
                if counts[row["FID"]] > 1:
                    try:
                        intersection.loc[i, "weight"] = grid_shp.loc[res[i], "geometry"].intersection(
                            ext_shp.loc[inp[i], "geometry"]).area / grid_shp.loc[res[i], "geometry"].area
                    except TopologicalError:
                        # If for some reason the geometry is corrupted it should work with the buffer function
                        ext_shp.loc[[inp[i]], "geometry"] = ext_shp.loc[[inp[i]], "geometry"].buffer(0)
                        intersection.loc[i, "weight"] = grid_shp.loc[res[i], "geometry"].intersection(
                            ext_shp.loc[inp[i], "geometry"]).area / grid_shp.loc[res[i], "geometry"].area
            # intersection["intersect_area"] = intersection.apply(
            #     lambda x: x["geometry_grid"].intersection(x["geometry_ext"]).area, axis=1)
            intersection.drop(intersection[intersection["weight"] <= 0].index, inplace=True)

            filterwarnings("default")

        # Choose the biggest area from intersected areas with multiple options
        intersection.sort_values("weight", ascending=False, inplace=True)
        intersection = intersection.drop_duplicates(subset="FID", keep="first")
        intersection = intersection.sort_values("FID").set_index("FID")

        for var_name in var_list:
            self.shapefile.loc[intersection.index, var_name] = array(
                ext_shp.loc[intersection["ext_shp_id"], var_name])

    else:
        for var_name in var_list:
            self.shapefile.loc[:, var_name] = nan

    for var_name in var_list:
        self.shapefile.loc[:, var_name] = array(self.shapefile.loc[:, var_name], dtype=ext_shp[var_name].dtype)

    return None
