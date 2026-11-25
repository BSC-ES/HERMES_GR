#!/usr/bin/env python
# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

from numpy import empty, newaxis, array, arcsin, tan, fabs, arctan, sqrt, radians, cos, sin, column_stack
from copy import deepcopy


def calculate_grid_area(self):
    """
    Get coordinate bounds and call function to calculate the area of each cell of a grid.

    Parameters
    ----------
    self : nes.Nes
        Source projection Nes Object.
    """

    # Create bounds if they do not exist
    if self.lat_bnds is None or self.lon_bnds is None:
        self.create_spatial_bounds()

    # Get spatial number of vertices
    spatial_nv = self.lat_bnds["data"].shape[-1]

    # Reshape bounds
    if spatial_nv == 2:

        aux_shape = (self.lat_bnds["data"].shape[0], self.lon_bnds["data"].shape[0], 4)
        lon_bnds_aux = empty(aux_shape)
        lon_bnds_aux[:, :, 0] = self.lon_bnds["data"][newaxis, :, 0]
        lon_bnds_aux[:, :, 1] = self.lon_bnds["data"][newaxis, :, 1]
        lon_bnds_aux[:, :, 2] = self.lon_bnds["data"][newaxis, :, 1]
        lon_bnds_aux[:, :, 3] = self.lon_bnds["data"][newaxis, :, 0]

        lon_bnds = lon_bnds_aux
        del lon_bnds_aux

        lat_bnds_aux = empty(aux_shape)
        lat_bnds_aux[:, :, 0] = self.lat_bnds["data"][:, newaxis, 0]
        lat_bnds_aux[:, :, 1] = self.lat_bnds["data"][:, newaxis, 0]
        lat_bnds_aux[:, :, 2] = self.lat_bnds["data"][:, newaxis, 1]
        lat_bnds_aux[:, :, 3] = self.lat_bnds["data"][:, newaxis, 1]

        lat_bnds = lat_bnds_aux
        del lat_bnds_aux

    else:
        lon_bnds = self.lon_bnds["data"]
        lat_bnds = self.lat_bnds["data"]

    # Reshape bounds and assign as grid corner coordinates
    grid_corner_lon = deepcopy(lon_bnds).reshape(lon_bnds.shape[0]*lon_bnds.shape[1],
                                                 lon_bnds.shape[2])
    grid_corner_lat = deepcopy(lat_bnds).reshape(lat_bnds.shape[0]*lat_bnds.shape[1],
                                                 lat_bnds.shape[2])

    # Calculate cell areas
    grid_area = calculate_cell_area(grid_corner_lon, grid_corner_lat,
                                    earth_radius_minor_axis=self.earth_radius[0],
                                    earth_radius_major_axis=self.earth_radius[1])

    return grid_area


def calculate_geometry_area(geometry_list, earth_radius_minor_axis=6371000.0, earth_radius_major_axis=6371000.0):
    """
    Get coordinate bounds and call function to calculate the area of each cell of a set of geometries.

    Parameters
    ----------
    geometry_list : List
        A List with polygon geometries.
    earth_radius_minor_axis : float
        Radius of the minor axis of the Earth.
    earth_radius_major_axis : float
        Radius of the major axis of the Earth.
    """

    geometry_area = empty(shape=(len(geometry_list),))

    for geom_ind in range(0, len(geometry_list)):

        # Calculate the area of each geometry in multipolygon and collection objects
        if geometry_list[geom_ind].geom_type in ["MultiPolygon", "GeometryCollection"]:
            multi_geom_area = 0
            for multi_geom_ind in range(0, len(geometry_list[geom_ind].geoms)):
                if geometry_list[geom_ind].geoms[multi_geom_ind].geom_type == "Point":
                    continue
                geometry_corner_lon, geometry_corner_lat = (
                    geometry_list[geom_ind].geoms[multi_geom_ind].exterior.coords.xy)
                geometry_corner_lon = array(geometry_corner_lon)
                geometry_corner_lat = array(geometry_corner_lat)
                geom_area = __mod_huiliers_area(geometry_corner_lon, geometry_corner_lat)
                multi_geom_area += geom_area
            geometry_area[geom_ind] = multi_geom_area * earth_radius_minor_axis * earth_radius_major_axis

        # Calculate the area of each geometry
        else:
            geometry_corner_lon, geometry_corner_lat = geometry_list[geom_ind].exterior.coords.xy
            geometry_corner_lon = array(geometry_corner_lon)
            geometry_corner_lat = array(geometry_corner_lat)
            geom_area = __mod_huiliers_area(geometry_corner_lon, geometry_corner_lat)
            geometry_area[geom_ind] = geom_area * earth_radius_minor_axis * earth_radius_major_axis

    return geometry_area


def calculate_cell_area(grid_corner_lon, grid_corner_lat,
                        earth_radius_minor_axis=6371000.0, earth_radius_major_axis=6371000.0):
    """
    Calculate the area of each cell of a grid.

    Parameters
    ----------
    grid_corner_lon : array
        An Array with longitude bounds of grid.
    grid_corner_lat : array
        An Array with longitude bounds of grid.
    earth_radius_minor_axis : float
        Radius of the minor axis of the Earth.
    earth_radius_major_axis : float
        Radius of the major axis of the Earth.
    """

    # Calculate area for each grid cell
    n_cells = grid_corner_lon.shape[0]
    area = empty(shape=(n_cells,))
    for i in range(0, n_cells):
        area[i] = __mod_huiliers_area(grid_corner_lon[i], grid_corner_lat[i])

    return area*earth_radius_minor_axis*earth_radius_major_axis


def __mod_huiliers_area(cell_corner_lon, cell_corner_lat):
    """
    Calculate the area of each cell according to Huilier's theorem.
    Reference: CDO (https://earth.bsc.es/gitlab/ces/cdo/).

    Parameters
    ----------
    cell_corner_lon : array
        Longitude boundaries of each cell.
    cell_corner_lat : array
        Latitude boundaries of each cell.
    """

    my_sum = 0

    # Get points 0 (bottom left) and 1 (bottom right) in Earth coordinates
    point_0 = __lon_lat_to_cartesian(cell_corner_lon[0], cell_corner_lat[0], earth_radius_major_axis=1)
    point_1 = __lon_lat_to_cartesian(cell_corner_lon[1], cell_corner_lat[1], earth_radius_major_axis=1)
    point_0, point_1 = point_0[0], point_1[0]

    # Get number of vertices
    if cell_corner_lat[0] == cell_corner_lat[-1]:
        spatial_nv = len(cell_corner_lon) - 1
    else:
        spatial_nv = len(cell_corner_lon)

    for i in range(2, spatial_nv):

        # Get point 2 (top right) in Earth coordinates
        point_2 = __lon_lat_to_cartesian(cell_corner_lon[i], cell_corner_lat[i], earth_radius_major_axis=1)
        point_2 = point_2[0]

        # Calculate area of triangle between points 0, 1 and 2
        my_sum += __tri_area(point_0, point_1, point_2)

        # # Copy to calculate area of next triangle
        # if i == (spatial_nv - 1):
        #     point_1 = deepcopy(point_2)
        point_1 = point_2

    return my_sum


def __tri_area(point_0, point_1, point_2):
    """
    Calculate area between three points that form a triangle.
    Reference: CDO (https://earth.bsc.es/gitlab/ces/cdo/).

    Parameters
    ----------
    point_0 : array
        Position of first point in cartesian coordinates.
    point_1 : array
        Position of second point in cartesian coordinates.
    point_2 : array
        Position of third point in cartesian coordinates.
    """

    # Get length of side a (between point 0 and 1)
    tmp_vec = __cross_product(point_0, point_1)
    sin_a = __norm(tmp_vec)
    a = arcsin(sin_a)

    # Get length of side b (between point 0 and 2)
    tmp_vec = __cross_product(point_0, point_2)
    sin_b = __norm(tmp_vec)
    b = arcsin(sin_b)

    # Get length of side c (between point 1 and 2)
    tmp_vec = __cross_product(point_2, point_1)
    sin_c = __norm(tmp_vec)
    c = arcsin(sin_c)

    # Calculate area
    s = 0.5*(a+b+c)
    t = tan(s*0.5) * tan((s - a)*0.5) * tan((s - b)*0.5) * tan((s - c)*0.5)
    area = fabs(4.0 * arctan(sqrt(fabs(t))))

    return area


def __cross_product(a, b):
    """
    Calculate cross product between two points.

    Parameters
    ----------
    a : array
        Position of point A in cartesian coordinates.
    b : array
        Position of point B in cartesian coordinates.
    """

    return [a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]]


def __norm(cp):
    """
    Normalize the result of the cross product operation.

    Parameters
    ----------
    cp : array
        Cross product between two points.
    """

    return sqrt(cp[0]*cp[0] + cp[1]*cp[1] + cp[2]*cp[2])


# noinspection DuplicatedCode
def __lon_lat_to_cartesian(lon, lat, earth_radius_major_axis=6371000.0):
    """
    Calculate lon, lat coordinates of a point on a sphere.

    Parameters
    ----------
    lon : array
        Longitude values.
    lat : array
        Latitude values.
    earth_radius_major_axis : float
        Radius of the major axis of the Earth.
    """

    lon_r = radians(lon)
    lat_r = radians(lat)

    x = earth_radius_major_axis * cos(lat_r) * cos(lon_r)
    y = earth_radius_major_axis * cos(lat_r) * sin(lon_r)
    z = earth_radius_major_axis * sin(lat_r)

    return column_stack([x, y, z])
