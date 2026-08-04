#!/usr/bin/env python

# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.


import sys
from netCDF4 import Dataset
from mpi4py import MPI
from functools import reduce


def open_netcdf(netcdf_path):
    """
    Open a netCDF file.

    :param netcdf_path: Path to the netCDF file.
    :type netcdf_path: str

    :return: netCDF
    :rtype: Dataset
    """
    netcdf = Dataset(netcdf_path, mode='a')
    return netcdf


def close_netcdf(netcdf):
    """
    Close the netCDF.

    :param netcdf: netCDF
    :type netcdf: Dataset
    :return:
    """
    netcdf.close()


def get_grid_area(filename):
    """
    Calculate the area of each cell.

    :param filename: Full path to the NetCDF to calculate the cell areas.
    :type filename: str

    :return: Returns the area of each cell.
    :rtype: numpy.array
    """
    from cdo import Cdo

    cdo = Cdo()
    src = cdo.gridarea(input=filename)
    netcdf = Dataset(src, mode='r')
    grid_area = netcdf.variables['cell_area'][:]
    netcdf.close()

    return grid_area


def extract_vars(netcdf_path, variables_list, attributes_list=()):
    """
    Get the data from the list of variabbles.

    :param netcdf_path: Path to the netCDF file
    :type netcdf_path: str

    :param variables_list: List of the names of the variables to get.
    :type variables_list: list

    :param attributes_list: List of the names of the variable attributes to get.
    :type attributes_list: list

    :return: List of the variables from the netCDF as a dictionary with data as values and with the other keys their
             attributes.
    :rtype: list.
    """
    data_list = []
    netcdf = Dataset(netcdf_path, mode='r')
    for var in variables_list:
        if var == 'emi_nox_no2':
            var1 = var
            var2 = 'emi_nox'
        else:
            var1 = var2 = var
        dict_aux = \
            {
                'name': var1,
                'data': netcdf.variables[var2][:],
            }
        for attribute in attributes_list:
            dict_aux.update({attribute: netcdf.variables[var2].getncattr(attribute)})
        data_list.append(dict_aux)
    netcdf.close()

    return data_list


def write_netcdf(netcdf_path, center_latitudes, center_longitudes, data_list,
                 levels=None, date=None, hours=None,
                 boundary_latitudes=None, boundary_longitudes=None, cell_area=None, global_attributes=None,
                 regular_latlon=False,
                 rotated=False, rotated_lats=None, rotated_lons=None, north_pole_lat=None, north_pole_lon=None,
                 lcc=False, lcc_x=None, lcc_y=None, lat_1_2=None, lon_0=None, lat_0=None,
                 mercator=False, lat_ts=None):
    # TODO Documentation
    """

    :param netcdf_path:
    :param center_latitudes:
    :param center_longitudes:
    :param data_list:
    :param levels:
    :param date:
    :param hours:
    :param boundary_latitudes:
    :param boundary_longitudes:
    :param cell_area:
    :param global_attributes:
    :param regular_latlon:
    :param rotated:
    :param rotated_lats:
    :param rotated_lons:
    :param north_pole_lat:
    :param north_pole_lon:
    :param lcc:
    :param lcc_x:
    :param lcc_y:
    :param lat_1_2:
    :param lon_0:
    :param lat_0:
    :param mercator:
    :param lat_ts:
    :return:
    """

    from cf_units import Unit, encode_time

    if not (regular_latlon or lcc or rotated or mercator):
        regular_latlon = True
    netcdf = Dataset(netcdf_path, mode='w', format="NETCDF4")

    # ===== Dimensions =====
    if regular_latlon:
        var_dim = ('lat', 'lon',)

        # Latitude
        if len(center_latitudes.shape) == 1:
            netcdf.createDimension('lat', center_latitudes.shape[0])
            lat_dim = ('lat',)
        elif len(center_latitudes.shape) == 2:
            netcdf.createDimension('lat', center_latitudes.shape[0])
            lat_dim = ('lat', 'lon', )
        else:
            print('ERROR: Latitudes must be on a 1D or 2D array instead of {0}'.format(len(center_latitudes.shape)))
            sys.exit(1)

        # Longitude
        if len(center_longitudes.shape) == 1:
            netcdf.createDimension('lon', center_longitudes.shape[0])
            lon_dim = ('lon',)
        elif len(center_longitudes.shape) == 2:
            netcdf.createDimension('lon', center_longitudes.shape[1])
            lon_dim = ('lat', 'lon', )
        else:
            print('ERROR: Longitudes must be on a 1D or 2D array instead of {0}'.format(len(center_longitudes.shape)))
            sys.exit(1)
    elif rotated:
        var_dim = ('rlat', 'rlon',)

        # Rotated Latitude
        if rotated_lats is None:
            print('ERROR: For rotated grids is needed the rotated latitudes.')
            sys.exit(1)
        netcdf.createDimension('rlat', len(rotated_lats))
        lat_dim = ('rlat', 'rlon',)

        # Rotated Longitude
        if rotated_lons is None:
            print('ERROR: For rotated grids is needed the rotated longitudes.')
            sys.exit(1)
        netcdf.createDimension('rlon', len(rotated_lons))
        lon_dim = ('rlat', 'rlon',)
    elif lcc or mercator:
        var_dim = ('y', 'x',)

        netcdf.createDimension('y', len(lcc_y))
        lat_dim = ('y', 'x', )

        netcdf.createDimension('x', len(lcc_x))
        lon_dim = ('y', 'x', )
    else:
        lat_dim = None
        lon_dim = None
        var_dim = None

    # Levels
    if levels is not None:
        netcdf.createDimension('lev', len(levels))

    # Bounds
    if boundary_latitudes is not None:
        try:
            netcdf.createDimension('nv', len(boundary_latitudes[0, 0]))
        except TypeError:
            netcdf.createDimension('nv', boundary_latitudes.shape[1])

    # Time
    netcdf.createDimension('time', None)

    # ===== Variables =====
    # Time
    if date is None:
        time = netcdf.createVariable('time', 'd', ('time',), zlib=True)
        time.units = "months since 2000-01-01 00:00:00"
        time.standard_name = "time"
        time.calendar = "gregorian"
        time.long_name = "time"
        time[:] = [0.]
    else:
        time = netcdf.createVariable('time', 'd', ('time',), zlib=True)
        u = Unit('hours')
        # print u.offset_by_time(encode_time(date.year, date.month, date.day, date.hour, date.minute, date.second))
        # Unit('hour since 1970-01-01 00:00:00.0000000 UTC')
        time.units = str(u.offset_by_time(encode_time(date.year, date.month, date.day, date.hour, date.minute,
                                                      date.second)))
        time.standard_name = "time"
        time.calendar = "gregorian"
        time.long_name = "time"
        time[:] = hours

    # Latitude
    lats = netcdf.createVariable('lat', 'f', lat_dim, zlib=True)
    lats.units = "degrees_north"
    lats.axis = "Y"
    lats.long_name = "latitude coordinate"
    lats.standard_name = "latitude"
    lats[:] = center_latitudes

    if boundary_latitudes is not None:
        lats.bounds = "lat_bnds"
        lat_bnds = netcdf.createVariable('lat_bnds', 'f', lat_dim + ('nv',), zlib=True)
        # print lat_bnds[:].shape, boundary_latitudes.shape
        lat_bnds[:] = boundary_latitudes

    # Longitude
    lons = netcdf.createVariable('lon', 'f', lon_dim, zlib=True)

    lons.units = "degrees_east"
    lons.axis = "X"
    lons.long_name = "longitude coordinate"
    lons.standard_name = "longitude"
    # print 'lons:', lons[:].shape, center_longitudes.shape
    lons[:] = center_longitudes
    if boundary_longitudes is not None:
        lons.bounds = "lon_bnds"
        lon_bnds = netcdf.createVariable('lon_bnds', 'f', lon_dim + ('nv',), zlib=True)
        lon_bnds[:] = boundary_longitudes

    if rotated:
        # Rotated Latitude
        rlat = netcdf.createVariable('rlat', 'f', ('rlat',), zlib=True)
        rlat.long_name = "latitude in rotated pole grid"
        rlat.units = Unit("degrees").symbol
        rlat.standard_name = "grid_latitude"
        rlat[:] = rotated_lats

        # Rotated Longitude
        rlon = netcdf.createVariable('rlon', 'f', ('rlon',), zlib=True)
        rlon.long_name = "longitude in rotated pole grid"
        rlon.units = Unit("degrees").symbol
        rlon.standard_name = "grid_longitude"
        rlon[:] = rotated_lons
    if lcc or mercator:
        x = netcdf.createVariable('x', 'd', ('x',), zlib=True)
        x.units = Unit("km").symbol
        x.long_name = "x coordinate of projection"
        x.standard_name = "projection_x_coordinate"
        x[:] = lcc_x

        y = netcdf.createVariable('y', 'd', ('y',), zlib=True)
        y.units = Unit("km").symbol
        y.long_name = "y coordinate of projection"
        y.standard_name = "projection_y_coordinate"
        y[:] = lcc_y

    cell_area_dim = var_dim
    # Levels
    if levels is not None:
        var_dim = ('lev',) + var_dim
        lev = netcdf.createVariable('lev', 'f', ('lev',), zlib=True)
        lev.units = Unit("m").symbol
        lev.positive = 'up'
        lev[:] = levels

    # All variables
    if len(data_list) == 0:
        var = netcdf.createVariable('aux_var', 'f', ('time',) + var_dim, zlib=True)
        var[:] = 0
    for variable in data_list:
        # print ('time',) + var_dim
        var = netcdf.createVariable(variable['name'], 'f', ('time',) + var_dim, zlib=True)
        var.units = Unit(variable['units']).symbol
        if 'long_name' in variable:
            var.long_name = str(variable['long_name'])
        if 'standard_name' in variable:
            var.standard_name = str(variable['standard_name'])
        if 'cell_method' in variable:
            var.cell_method = str(variable['cell_method'])
        var.coordinates = "lat lon"
        if cell_area is not None:
            var.cell_measures = 'area: cell_area'
        if regular_latlon:
            var.grid_mapping = 'crs'
        elif rotated:
            var.grid_mapping = 'rotated_pole'
        elif lcc:
            var.grid_mapping = 'Lambert_conformal'
        elif mercator:
            var.grid_mapping = 'mercator'
        try:
            var[:] = variable['data']
        except ValueError:
            print('VAR ERROR, netcdf shape: {0}, variable shape: {1}'.format(var[:].shape, variable['data'].shape))

    # Grid mapping
    if regular_latlon:
        # CRS
        mapping = netcdf.createVariable('crs', 'i')
        mapping.grid_mapping_name = "latitude_longitude"
        mapping.semi_major_axis = 6371000.0
        mapping.inverse_flattening = 0
    elif rotated:
        # Rotated pole
        mapping = netcdf.createVariable('rotated_pole', 'c')
        mapping.grid_mapping_name = 'rotated_latitude_longitude'
        mapping.grid_north_pole_latitude = north_pole_lat
        mapping.grid_north_pole_longitude = north_pole_lon
    elif lcc:
        # CRS
        mapping = netcdf.createVariable('Lambert_conformal', 'i')
        mapping.grid_mapping_name = "lambert_conformal_conic"
        mapping.standard_parallel = lat_1_2
        mapping.longitude_of_central_meridian = lon_0
        mapping.latitude_of_projection_origin = lat_0
    elif mercator:
        # Mercator
        mapping = netcdf.createVariable('mercator', 'i')
        mapping.grid_mapping_name = "mercator"
        mapping.longitude_of_projection_origin = lon_0
        mapping.standard_parallel = lat_ts

    # Cell area
    if cell_area is not None:
        c_area = netcdf.createVariable('cell_area', 'f', cell_area_dim)
        c_area.long_name = "area of the grid cell"
        c_area.standard_name = "cell_area"
        c_area.units = Unit("m2").symbol
        # print c_area[:].shape, cell_area.shape
        c_area[:] = cell_area

    if global_attributes is not None:
        netcdf.setncatts(global_attributes)

    netcdf.close()


def create_netcdf(netcdf_path, center_latitudes, center_longitudes, data_list,
                  levels=None, date=None, hours=None,
                  boundary_latitudes=None, boundary_longitudes=None, cell_area=None, global_attributes=None,
                  regular_latlon=False,
                  rotated=False, rotated_lats=None, rotated_lons=None, north_pole_lat=None, north_pole_lon=None,
                  lcc=False, lcc_x=None, lcc_y=None, lat_1_2=None, lon_0=None, lat_0=None):
    # TODO Documentation
    """

    :param netcdf_path:
    :param center_latitudes:
    :param center_longitudes:
    :param data_list:
    :param levels:
    :param date:
    :param hours:
    :param boundary_latitudes:
    :param boundary_longitudes:
    :param cell_area:
    :param global_attributes:
    :param regular_latlon:
    :param rotated:
    :param rotated_lats:
    :param rotated_lons:
    :param north_pole_lat:
    :param north_pole_lon:
    :param lcc:
    :param lcc_x:
    :param lcc_y:
    :param lat_1_2:
    :param lon_0:
    :param lat_0:
    :return:
    """
    from cf_units import Unit, encode_time
    import numpy as np

    if not (regular_latlon or lcc or rotated):
        regular_latlon = True

    netcdf = Dataset(netcdf_path, mode='w', format="NETCDF4")

    # ===== Dimensions =====
    if regular_latlon:
        var_dim = ('lat', 'lon',)

        # Latitude
        if len(center_latitudes.shape) == 1:
            netcdf.createDimension('lat', center_latitudes.shape[0])
            lat_dim = ('lat',)
        elif len(center_latitudes.shape) == 2:
            netcdf.createDimension('lat', center_latitudes.shape[0])
            lat_dim = ('lat', 'lon', )
        else:
            print('ERROR: Latitudes must be on a 1D or 2D array instead of {0}'.format(len(center_latitudes.shape)))
            sys.exit(1)

        # Longitude
        if len(center_longitudes.shape) == 1:
            netcdf.createDimension('lon', center_longitudes.shape[0])
            lon_dim = ('lon',)
        elif len(center_longitudes.shape) == 2:
            netcdf.createDimension('lon', center_longitudes.shape[1])
            lon_dim = ('lat', 'lon', )
        else:
            print('ERROR: Longitudes must be on a 1D or 2D array instead of {0}'.format(len(center_longitudes.shape)))
            sys.exit(1)
    elif rotated:
        var_dim = ('rlat', 'rlon',)

        # Rotated Latitude
        if rotated_lats is None:
            print('ERROR: For rotated grids is needed the rotated latitudes.')
            sys.exit(1)
        netcdf.createDimension('rlat', len(rotated_lats))
        lat_dim = ('rlat', 'rlon',)

        # Rotated Longitude
        if rotated_lons is None:
            print('ERROR: For rotated grids is needed the rotated longitudes.')
            sys.exit(1)
        netcdf.createDimension('rlon', len(rotated_lons))
        lon_dim = ('rlat', 'rlon',)

    elif lcc:
        var_dim = ('y', 'x',)

        netcdf.createDimension('y', len(lcc_y))
        lat_dim = ('y', 'x', )

        netcdf.createDimension('x', len(lcc_x))
        lon_dim = ('y', 'x', )
    else:
        lat_dim = None
        lon_dim = None
        var_dim = None

    # Levels
    if levels is not None:
        netcdf.createDimension('lev', len(levels))

    # Bounds
    if boundary_latitudes is not None:
        # print boundary_latitudes.shape
        # print len(boundary_latitudes[0, 0])
        netcdf.createDimension('nv', len(boundary_latitudes[0, 0]))
        # sys.exit()

    # Time
    netcdf.createDimension('time', None)

    # ===== Variables =====
    # Time
    if date is None:
        time = netcdf.createVariable('time', 'd', ('time',), zlib=True)
        time.units = "months since 2000-01-01 00:00:00"
        time.standard_name = "time"
        time.calendar = "gregorian"
        time.long_name = "time"
        time[:] = [0.]
    else:
        time = netcdf.createVariable('time', 'd', ('time',), zlib=True)
        u = Unit('hours')
        # print u.offset_by_time(encode_time(date.year, date.month, date.day, date.hour, date.minute, date.second))
        # Unit('hour since 1970-01-01 00:00:00.0000000 UTC')
        time.units = str(u.offset_by_time(encode_time(date.year, date.month, date.day, date.hour, date.minute,
                                                      date.second)))
        time.standard_name = "time"
        time.calendar = "gregorian"
        time.long_name = "time"
        time[:] = hours

    # Latitude
    lats = netcdf.createVariable('lat', 'f', lat_dim, zlib=True)
    lats.units = "degrees_north"
    lats.axis = "Y"
    lats.long_name = "latitude coordinate"
    lats.standard_name = "latitude"
    lats[:] = center_latitudes

    if boundary_latitudes is not None:
        lats.bounds = "lat_bnds"
        lat_bnds = netcdf.createVariable('lat_bnds', 'f', lat_dim + ('nv',), zlib=True)
        # print lat_bnds[:].shape, boundary_latitudes.shape
        lat_bnds[:] = boundary_latitudes

    # Longitude
    lons = netcdf.createVariable('lon', 'f', lon_dim, zlib=True)

    lons.units = "degrees_east"
    lons.axis = "X"
    lons.long_name = "longitude coordinate"
    lons.standard_name = "longitude"
    lons[:] = center_longitudes
    if boundary_longitudes is not None:
        lons.bounds = "lon_bnds"
        lon_bnds = netcdf.createVariable('lon_bnds', 'f', lon_dim + ('nv',), zlib=True)
        # print lon_bnds[:].shape, boundary_longitudes.shape
        lon_bnds[:] = boundary_longitudes

    if rotated:
        # Rotated Latitude
        rlat = netcdf.createVariable('rlat', 'f', ('rlat',), zlib=True)
        rlat.long_name = "latitude in rotated pole grid"
        rlat.units = Unit("degrees").symbol
        rlat.standard_name = "grid_latitude"
        rlat[:] = rotated_lats

        # Rotated Longitude
        rlon = netcdf.createVariable('rlon', 'f', ('rlon',), zlib=True)
        rlon.long_name = "longitude in rotated pole grid"
        rlon.units = Unit("degrees").symbol
        rlon.standard_name = "grid_longitude"
        rlon[:] = rotated_lons
    if lcc:
        x = netcdf.createVariable('x', 'd', ('x',), zlib=True)
        x.units = Unit("km").symbol
        x.long_name = "x coordinate of projection"
        x.standard_name = "projection_x_coordinate"
        x[:] = lcc_x

        y = netcdf.createVariable('y', 'd', ('y',), zlib=True)
        y.units = Unit("km").symbol
        y.long_name = "y coordinate of projection"
        y.standard_name = "projection_y_coordinate"
        y[:] = lcc_y

    cell_area_dim = var_dim
    # Levels
    if levels is not None:
        var_dim = ('lev',) + var_dim
        lev = netcdf.createVariable('lev', 'f', ('lev',), zlib=True)
        lev.units = Unit("m").symbol
        lev.positive = 'up'
        lev[:] = levels

    # All variables
    if len(data_list) == 0:
        var = netcdf.createVariable('aux_var', 'f', ('time',) + var_dim, zlib=True)
        var[:] = 0
    for variable in data_list:
        # print ('time',) + var_dim
        # print variable
        var = netcdf.createVariable(variable['name'], 'f', ('time',) + var_dim, zlib=True)
        var.units = Unit(variable['units']).symbol
        if 'long_name' in variable:
            var.long_name = str(variable['long_name'])
        if 'standard_name' in variable:
            var.standard_name = str(variable['standard_name'])
        if 'cell_method' in variable:
            var.cell_method = str(variable['cell_method'])
        var.coordinates = "lat lon"
        if cell_area is not None:
            var.cell_measures = 'area: cell_area'
        if regular_latlon:
            var.grid_mapping = 'crs'
        elif rotated:
            var.grid_mapping = 'rotated_pole'
        elif lcc:
            var.grid_mapping = 'Lambert_conformal'
        # print 'HOURSSSSSSSSSSSSSSSSSSSSS:', hours
        # if variable['data'] is not 0:
        #     print var[:].shape, variable['data'].shape, variable['data'].max()
        shape = tuple()
        exec("shape = (len(hours), {0}.size, {1}.size, {2}.size)".format(var_dim[0], var_dim[1], var_dim[2]))

        var[:] = np.zeros(shape)

    # Grid mapping
    if regular_latlon:
        # CRS
        mapping = netcdf.createVariable('crs', 'i')
        mapping.grid_mapping_name = "latitude_longitude"
        mapping.semi_major_axis = 6371000.0
        mapping.inverse_flattening = 0
    elif rotated:
        # Rotated pole
        mapping = netcdf.createVariable('rotated_pole', 'c')
        mapping.grid_mapping_name = 'rotated_latitude_longitude'
        mapping.grid_north_pole_latitude = north_pole_lat
        mapping.grid_north_pole_longitude = north_pole_lon
    elif lcc:
        # CRS
        mapping = netcdf.createVariable('Lambert_conformal', 'i')
        mapping.grid_mapping_name = "lambert_conformal_conic"
        mapping.standard_parallel = lat_1_2
        mapping.longitude_of_central_meridian = lon_0
        mapping.latitude_of_projection_origin = lat_0

    # Cell area
    if cell_area is not None:
        c_area = netcdf.createVariable('cell_area', 'f', cell_area_dim)
        c_area.long_name = "area of the grid cell"
        c_area.standard_name = "cell_area"
        c_area.units = Unit("m2").symbol
        # print c_area[:].shape, cell_area.shape
        c_area[:] = cell_area

    if global_attributes is not None:
        netcdf.setncatts(global_attributes)
    return netcdf


def tuple_to_index(tuple_list, bidimensional=False):
    # TODO Documentation
    """

    :param tuple_list:
    :param bidimensional:
    :return:
    """
    from operator import mul
    new_list = []
    for my_tuple in tuple_list:
        if bidimensional:
            new_list.append(my_tuple[-1] * my_tuple[-2])
        else:
            new_list.append(reduce(mul, my_tuple))
    return new_list


def calculate_displacements(counts):
    # TODO Documentation
    """

    :param counts:
    :return:
    """
    new_list = [0]
    accum = 0
    for counter in counts[:-1]:
        accum += counter
        new_list.append(accum)
    return new_list


def netcdf2shp(comm, netcdf_path):
    from netCDF4 import Dataset
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    from shapely import wkt

    if comm.Get_rank() == 0:
        nc_in = Dataset(netcdf_path, mode='r')
        lat_bnds = nc_in.variables['lat_bnds'][:]
        lon_bnds = nc_in.variables['lon_bnds'][:]
        nc_in.close()

        lat_bnds_0 = np.array([lat_bnds[:, 0]] * lon_bnds.shape[0]).T.flatten()
        lat_bnds_1 = np.array([lat_bnds[:, 1]] * lon_bnds.shape[0]).T.flatten()
        lon_bnds_0 = np.array([lon_bnds[:, 0]] * lat_bnds.shape[0]).flatten()
        lon_bnds_1 = np.array([lon_bnds[:, 1]] * lat_bnds.shape[0]).flatten()

        coordinates = pd.DataFrame(lat_bnds_0, columns=['lat_bnds_0'])
        coordinates.index.name = 'FID'
        coordinates['lat_bnds_1'] = lat_bnds_1
        coordinates['lon_bnds_0'] = lon_bnds_0
        coordinates['lon_bnds_1'] = lon_bnds_1
    else:
        coordinates = None
    coordinates = split_shapefile(comm, coordinates)
    coordinates['Coordinates'] = \
        'POLYGON ((' + coordinates['lon_bnds_0'].astype(str) + ' ' + coordinates['lat_bnds_0'].astype(str) + ', ' + \
        coordinates['lon_bnds_0'].astype(str) + ' ' + coordinates['lat_bnds_1'].astype(str) + ', ' + \
        coordinates['lon_bnds_1'].astype(str) + ' ' + coordinates['lat_bnds_1'].astype(str) + ', ' + \
        coordinates['lon_bnds_1'].astype(str) + ' ' + coordinates['lat_bnds_0'].astype(str) + ', ' + \
        coordinates['lon_bnds_0'].astype(str) + ' ' + coordinates['lat_bnds_0'].astype(str) + '))'

    coordinates['Coordinates'] = coordinates['Coordinates'].apply(wkt.loads)

    shapefile = gpd.GeoDataFrame(coordinates, geometry='Coordinates', crs={'init': 'epsg:4326'})
    return shapefile


def split_shapefile(comm, data, rank=0):
    import numpy as np

    if comm.Get_size() == 1:
        data = data
    else:
        if comm.Get_rank() == rank:
            data = np.array_split(data, comm.Get_size())
        else:
            data = None
        data = comm.scatter(data, root=rank)

    return data


def gather_shapefile(comm, data, rank=0):
    import pandas as pd

    if comm.Get_size() == 1:
        data = data
    else:
        data = comm.gather(data, root=rank)
        if comm.Get_rank() == rank:
            data = pd.concat(data)
        else:
            data = None
    return data


def write_shapefile_parallel(comm, data, path, rank=0):
    import os
    import pandas as pd
    data = comm.gather(data, root=rank)
    if comm.Get_rank() == rank:
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        data = pd.concat(data)
        data.to_file(path)

    comm.Barrier()

    return True


if __name__ == '__main__':
    pass
