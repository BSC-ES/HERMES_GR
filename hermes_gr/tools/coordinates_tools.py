#!/usr/bin/env python

# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.


def get_grid_area(filename):
    """
    Calculate the area for each cell of the grid using CDO

    :param filename: Path to the file to calculate the cell area
    :type filename: str

    :return: Area of each cell of the grid.
    :rtype: numpy.array
    """
    from cdo import Cdo
    from netCDF4 import Dataset

    cdo = Cdo()
    s = cdo.gridarea(input=filename)
    nc_aux = Dataset(s, mode='r')
    grid_area = nc_aux.variables['cell_area'][:]
    nc_aux.close()

    return grid_area


def latlon2rotated(lon_pole_deg, lat_pole_deg, lon_deg, lat_deg, lon_min=-180):
    # TODO Documentation
    """
    Transform lat lon degrees into the rotated coordinates.

    :param lon_pole_deg:
    :param lat_pole_deg:
    :param lon_deg:
    :param lat_deg:
    :param lon_min:
    :return:
    """
    import math

    degrees_to_radians = math.pi / 180.
    radians_to_degrees = 180. / math.pi

    # lon_max = lon_min + 360

    #   stlm=sin(tlm)
    sin_lat_pole_rad = math.sin(lat_pole_deg * degrees_to_radians)
    #   ctlm=cos(tlm)
    cos_lat_pole_rad = math.cos(lat_pole_deg * degrees_to_radians)
    #   stph=sin(tph)
    # sin_lon_pole_rad = math.sin(lon_pole_deg * degrees_to_radians)
    #   ctph=cos(tph)
    # cos_lon_pole_rad = math.cos(lon_pole_deg * degrees_to_radians)

    #   relm=(xlon-tlm0d)*dtr !distance from the centre lon (in rad)
    distance_from_center_lon = (lon_deg - lon_pole_deg) * degrees_to_radians
    #   crlm=cos(relm) !cos of this distance
    cos_distance_from_center_lon = math.cos(distance_from_center_lon)
    #   srlm=sin(relm) !sin of this distance
    sin_distance_from_center_lon = math.sin(distance_from_center_lon)
    #   aph=xlat*dtr !lat in rad
    lat_rad = lat_deg * degrees_to_radians
    #   cph=cos(aph) !cos of lat
    cos_lat_rad = math.cos(lat_rad)
    #   sph=sin(aph) !sin of lat
    sin_lat_rad = math.sin(lat_rad)

    #   cc=cph*crlm !cos of lat times cos of lon distance
    cycdx = cos_lat_rad * cos_distance_from_center_lon
    #   anum=cph*srlm !cos of lat times sin of lon distance
    #   denom=ctph0*cc+stph0*sph !cos of the centre lat times cc plus sin of the centre lat times sin of lat
    #   tlm=atan2(anum,denom)
    rotated_lon = math.atan2(cos_lat_rad * sin_distance_from_center_lon,
                             cos_lat_pole_rad * cycdx + sin_lat_pole_rad * sin_lat_rad)
    #   tph=asin(ctph0*sph-stph0*cc)
    sin_rotated_lat = cos_lat_pole_rad * sin_lat_rad - sin_lat_pole_rad * cycdx
    if sin_rotated_lat > 1.:
        sin_rotated_lat = 1.
    if sin_rotated_lat < -1.:
        sin_rotated_lat = -1.

    rotated_lat = math.asin(sin_rotated_lat)

    return rotated_lon * radians_to_degrees, rotated_lat * radians_to_degrees


def rotated2latlon(lon_pole_deg, lat_pole_deg, lon_deg, lat_deg, lon_min=-180):
    # TODO Documentation
    """
    Transform rotated coordinates into lat lon degrees.

    :param lon_pole_deg:
    :param lat_pole_deg:
    :param lon_deg:
    :param lat_deg:
    :param lon_min:
    :return:
    """
    import numpy as np
    import math

    degrees_to_radians = math.pi / 180.
    # radians_to_degrees = 180. / math.pi

    # Positive east to negative east
    lon_pole_deg -= 180

    tph0 = lat_pole_deg * degrees_to_radians
    tlm = lon_deg * degrees_to_radians
    tph = lat_deg * degrees_to_radians
    tlm0d = lon_pole_deg
    ctph0 = np.cos(tph0)
    stph0 = np.sin(tph0)

    stlm = np.sin(tlm)
    ctlm = np.cos(tlm)
    stph = np.sin(tph)
    ctph = np.cos(tph)

    # Latitude
    sph = (ctph0 * stph) + (stph0 * ctph * ctlm)
    # if sph > 1.:
    #     sph = 1.
    # if sph < -1.:
    #     sph = -1.
    # print type(sph)
    sph[sph > 1.] = 1.
    sph[sph < -1.] = -1.

    aph = np.arcsin(sph)
    aphd = aph / degrees_to_radians

    # Longitude
    anum = ctph * stlm
    denom = (ctlm * ctph - stph0 * sph) / ctph0
    relm = np.arctan2(anum, denom) - math.pi
    almd = relm / degrees_to_radians + tlm0d

    # if almd < min_lon:
    #     almd += 360
    # elif almd > max_lon:
    #     almd -= 360

    almd[almd > (lon_min + 360)] -= 360
    almd[almd < lon_min] += 360

    return almd, aphd


def rotated2latlon_single(lon_pole_deg, lat_pole_deg, lon_deg, lat_deg, lon_min=-180):
    # TODO Docuemtnation
    """

    :param lon_pole_deg:
    :param lat_pole_deg:
    :param lon_deg:
    :param lat_deg:
    :param lon_min:
    :return:
    """
    import math

    degrees_to_radians = math.pi / 180.
    # radians_to_degrees = 180. / math.pi

    # lon_max = lon_min + 360

    # Positive east to negative east
    lon_pole_deg -= 180

    tph0 = lat_pole_deg * degrees_to_radians
    tlm = lon_deg * degrees_to_radians
    tph = lat_deg * degrees_to_radians
    tlm0d = lon_pole_deg
    ctph0 = math.cos(tph0)
    stph0 = math.sin(tph0)

    stlm = math.sin(tlm)
    ctlm = math.cos(tlm)
    stph = math.sin(tph)
    ctph = math.cos(tph)

    # Latitude
    sph = (ctph0 * stph) + (stph0 * ctph * ctlm)
    # if sph > 1.:
    #     sph = 1.
    # if sph < -1.:
    #     sph = -1.

    aph = math.asin(sph)
    aphd = aph / degrees_to_radians

    # Longitude
    anum = ctph * stlm
    denom = (ctlm * ctph - stph0 * sph) / ctph0
    relm = math.atan2(anum, denom) - math.pi
    almd = relm / degrees_to_radians + tlm0d

    if almd > (lon_min + 360):
        almd -= 360
    elif almd < lon_min:
        almd += 360

    return almd, aphd


def create_bounds(coords, number_vertices=2):
    """
    Calculate the vertices coordinates.

    :param coords: Coordinates in degrees (latitude or longitude)
    :type coords: numpy.array

    :param number_vertices: Non mandatory parameter that informs the number of vertices that must have the boundaries.
            (by default 2)
    :type number_vertices: int

    :return: Array with as many elements as vertices for each value of coords.
    :rtype: numpy.array
    """
    import numpy as np

    interval = coords[1] - coords[0]

    coords_left = coords - interval / 2
    coords_right = coords + interval / 2
    if number_vertices == 2:
        bound_coords = np.dstack((coords_left, coords_right))
    elif number_vertices == 4:
        bound_coords = np.dstack((coords_left, coords_right, coords_right, coords_left))
    else:
        raise ValueError('The number of vertices of the boudaries must be 2 or 4')

    return bound_coords


def create_bounds_esmpy(coords, spheric=False):
    # TODO Documentation
    """

    :param coords:
    :param spheric:
    :return:
    """
    import numpy as np

    interval = coords[1] - coords[0]

    bound_coords = coords - interval / 2
    if not spheric:
        bound_coords = np.append(bound_coords, [bound_coords[-1] + interval])

    return bound_coords


def create_regular_rotated(lat_origin, lon_origin, lat_inc, lon_inc, n_lat, n_lon):
    # TODO Documentation
    """

    :param lat_origin:
    :param lon_origin:
    :param lat_inc:
    :param lon_inc:
    :param n_lat:
    :param n_lon:
    :return:
    """
    import numpy as np

    center_latitudes = np.linspace(lat_origin, lat_origin + (lat_inc * (n_lat - 1)), int(n_lat), dtype=np.float64)
    center_longitudes = np.linspace(lon_origin, lon_origin + (lon_inc * (n_lon - 1)), int(n_lon), dtype=np.float64)

    corner_latitudes = create_bounds_esmpy(center_latitudes)
    corner_longitudes = create_bounds_esmpy(center_longitudes)

    return center_latitudes, center_longitudes, corner_latitudes, corner_longitudes


if __name__ == '__main__':
    import numpy as np
    new_pole_lon_d = 20.0  # lonpole tlm0d
    new_pole_lat_d = 35.0  # latpole tph0d
    print(latlon2rotated(new_pole_lon_d, new_pole_lat_d, 20.0, 35.0))
    print(latlon2rotated(new_pole_lon_d, new_pole_lat_d, -20.2485, -9.9036))
    print(rotated2latlon_single(new_pole_lon_d, new_pole_lat_d, 0, 0))
    print(rotated2latlon_single(new_pole_lon_d, new_pole_lat_d, -51., -35.))
    print(rotated2latlon(new_pole_lon_d, new_pole_lat_d, np.array([-51., -51., -51., -51.]),
                         np.array([-35., -34.9, -34.8, -34.7])))
