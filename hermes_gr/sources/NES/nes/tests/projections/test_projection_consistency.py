# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
# Licensed under the Apache License, Version 2.0. See LICENSE for details.
"""Common reconstruction, CF metadata and native-geometry projection contracts."""

from datetime import datetime
import re

import numpy as np
import pytest
from mpi4py import MPI
from netCDF4 import Dataset
from pyproj import CRS, Transformer

import nes


KINDS = ('regular', 'rotated', 'lcc', 'mercator', 'laea')


def parameters(kind):
    """Return small deterministic creation parameters for a structured grid.

    Parameters
    ----------
    kind : str
        Structured projection selector.

    Returns
    -------
    dict
        Native geometry and required projection parameters.
    """
    if kind == 'regular':
        return dict(lat_orig=40., lon_orig=5., inc_lat=.1, inc_lon=.1, n_lat=4, n_lon=4)
    if kind == 'rotated':
        return dict(centre_lat=40., centre_lon=5., south_boundary=-2., west_boundary=-2.,
                    inc_rlat=1., inc_rlon=1.)
    args = dict(x_0=0., y_0=0., inc_x=1000., inc_y=1000., nx=4, ny=4, lon_0=5.)
    if kind == 'lcc':
        args.update(lat_0=40., lat_1=30., lat_2=60.)
    elif kind == 'mercator':
        args.update(lat_ts=40.)
    else:
        args.update(lat_0=40., earth_radius=6371000.)
    return args


def make_grid(kind, comm=MPI.COMM_SELF, **overrides):
    """Create a small populated grid for serial or collective projection checks.

    Parameters
    ----------
    kind : str
        Structured projection selector.
    comm : mpi4py.MPI.Comm, optional
        Grid communicator. Default is MPI.COMM_SELF.
    **overrides : dict
        Geometry, projection or decomposition argument overrides.

    Returns
    -------
    nes.Nes
        Grid containing a rank-local extensive field.
    """
    args = dict(parameters(kind), **overrides)
    grid = nes.create_nes(projection=kind, comm=comm, **args)
    shape = grid.lat['data'].shape if grid.lat['data'].ndim == 2 else (len(grid.lat['data']), len(grid.lon['data']))
    grid.variables['field'] = dict(data=np.ones((len(grid.time), 1) + shape), units='kg', long_name='cell total')
    grid.global_attrs.update(title='Structured projection contract', history='Created for NES regression tests')
    return grid


@pytest.mark.parametrize('kind', KINDS)
def test_reconstruction_copy_and_configuration(kind, tmp_path):
    """Retain class, physical configuration and data across construction and copies."""
    grid = make_grid(kind)
    recreated = type(grid).new(comm=MPI.COMM_SELF, create_nes=True, times=[datetime(2022, 1, 1)], **parameters(kind))
    assert type(recreated) is type(grid)
    assert recreated.get_grid_configuration() == grid.get_grid_configuration()
    for operation in (grid.copy, grid.copy2):
        copied = operation(copy_vars=True)
        assert type(copied) is type(grid)
        assert copied.comm is grid.comm
        np.testing.assert_array_equal(copied.variables['field']['data'], grid.variables['field']['data'])
    path = str(tmp_path / (kind + '.nc'))
    grid.to_netcdf(path)
    opened = type(grid).new(comm=MPI.COMM_SELF, path=path)
    assert type(opened) is type(grid)
    assert opened.get_grid_configuration() == grid.get_grid_configuration()
    assert opened.validate_grid_configuration(projection=kind, **parameters(kind))
    opened.close()


def test_nested_reconstruction_preserves_subtype_and_parent_earth(tmp_path):
    """Reconstruct a nested grid as its own subtype with the parent ellipsoid."""
    parent = make_grid('rotated', semi_major_axis=6378137., inverse_flattening=298.257222101)
    path = str(tmp_path / 'parent.nc')
    parent.to_netcdf(path)
    args = dict(parent_grid_path=path, parent_ratio=2, i_parent_start=2, j_parent_start=2, n_rlat=4, n_rlon=4)
    child = nes.RotatedNestedNes.new(comm=MPI.COMM_SELF, create_nes=True,
                                     times=[datetime(2022, 1, 1)], **args)
    assert type(child) is nes.RotatedNestedNes
    np.testing.assert_allclose(child.earth_radius, parent.earth_radius)
    assert child.validate_grid_configuration(projection='rotated-nested', **args)
    with pytest.raises(ValueError, match='parent'):
        nes.create_nes(projection='rotated-nested', comm=MPI.COMM_SELF, **args, semi_major_axis=1.)


@pytest.mark.parametrize('kind', KINDS)
def test_explicit_earth_and_precision(kind, tmp_path):
    """Use explicit Earth axes consistently and preserve native spatial precision."""
    args = dict(semi_major_axis=6378137., inverse_flattening=298.257222101)
    if kind == 'laea':
        args['earth_radius'] = 6378137.
        args.pop('semi_major_axis')
        args.pop('inverse_flattening')
    grid = make_grid(kind, **args)
    path = str(tmp_path / 'precision.nc')
    grid.to_netcdf(path)
    with Dataset(path) as ds:
        for axis in ('lat', 'lon'):
            assert ds.variables[axis].dtype == np.dtype('float64')
            np.testing.assert_array_equal(ds.variables[axis][:], getattr(grid, 'get_full_' + (
                'latitudes' if axis == 'lat' else 'longitudes'))()['data'])
        mapping = ds.variables[ds.variables['field'].grid_mapping]
        assert any(key in mapping.ncattrs() for key in ('earth_radius', 'semi_major_axis'))
    opened = nes.open_netcdf(path, comm=MPI.COMM_SELF)
    np.testing.assert_allclose(opened.earth_radius, grid.earth_radius)
    assert opened.get_grid_configuration() == grid.get_grid_configuration()
    opened.close()


@pytest.mark.parametrize('kind', ('lcc', 'mercator', 'rotated', 'laea'))
def test_native_geometry_survives_selection_and_resize(kind, tmp_path):
    """Keep local/native axes, metadata and true projected corners aligned."""
    grid = make_grid(kind)
    y_name, x_name = ('rlat', 'rlon') if kind == 'rotated' else ('y', 'x')
    originals = {name: getattr(grid, name)['data'].copy() for name in (y_name, x_name)}
    grid.create_spatial_bounds()
    original_bounds = grid.lat_bnds['data'].copy()
    grid.expand()
    grid.contract()
    for name in originals:
        np.testing.assert_allclose(getattr(grid, name)['data'], originals[name])
    np.testing.assert_allclose(grid.lat_bnds['data'], original_bounds, rtol=0, atol=1e-10)
    path = str(tmp_path / 'selection.nc')
    grid.to_netcdf(path)
    opened = nes.open_netcdf(path, comm=MPI.COMM_SELF)
    data = opened.lat['data']
    opened.sel(lat_min=float(data[1, 0]), lat_max=float(data[-2, -1]))
    assert len(getattr(opened, y_name)['data']) == opened.lat['data'].shape[0]
    assert len(getattr(opened, x_name)['data']) == opened.lat['data'].shape[1]
    assert 'units' in getattr(opened, y_name)
    assert getattr(opened, y_name)['bounds_data'].shape == (len(getattr(opened, y_name)['data']), 2)
    opened.close()


def write_cf_input(path, kind, units='m'):
    """Write independent CF input without geographical auxiliary coordinates.

    Parameters
    ----------
    path : str
        Destination NetCDF file.
    kind : {'lcc', 'mercator', 'rotated'}
        Structured mapping with arbitrary coordinate/dimension/mapping names.
    units : {'m', 'km'}, optional
        Linear units for projected grids; rotated grids always use degrees.

    Returns
    -------
    tuple
        Independent reference CRS and native axis arrays.
    """
    cf = dict(semi_major_axis=6378137., inverse_flattening=298.257222101)
    if kind == 'lcc':
        cf.update(grid_mapping_name='lambert_conformal_conic', standard_parallel=40.,
                  latitude_of_projection_origin=40., longitude_of_central_meridian=10.)
        reference = CRS.from_proj4('+proj=lcc +lat_1=40 +lat_0=40 +lon_0=10 +a=6378137 +rf=298.257222101 '
                                   '+x_0=10000 +y_0=-20000')
    elif kind == 'mercator':
        cf.update(grid_mapping_name='mercator', scale_factor_at_projection_origin=.9,
                  longitude_of_projection_origin=10.)
        reference = CRS.from_proj4('+proj=merc +k_0=.9 +lon_0=10 +a=6378137 +rf=298.257222101 '
                                   '+x_0=10000 +y_0=-20000')
    else:
        cf.update(grid_mapping_name='rotated_latitude_longitude', grid_north_pole_latitude=35.,
                  grid_north_pole_longitude=-170., north_pole_grid_longitude=5.)
        reference = CRS.from_proj4('+proj=ob_tran +o_proj=longlat +o_lat_p=35 +o_lon_p=5 +lon_0=10 '
                                   '+a=6378137 +rf=298.257222101')
    factor = 1000 if units == 'km' else 1
    if kind == 'rotated':
        x, y = np.array([1., 0., -1.]), np.array([0., 1.])
    else:
        cf.update(false_easting=10000. / factor, false_northing=-20000. / factor)
        x, y = np.array([11500., 10500., 9500.]) / factor, np.array([-19500., -18500.]) / factor
    with Dataset(path, 'w') as ds:
        ds.createDimension('eastings', len(x))
        ds.createDimension('northings', len(y))
        ds.createDimension('vertices', 2)
        ds.Conventions = 'CF-1.7'
        for name, values, standard in (
            ('eastings', x, 'grid_longitude' if kind == 'rotated' else 'projection_x_coordinate'),
            ('northings', y, 'grid_latitude' if kind == 'rotated' else 'projection_y_coordinate'),
        ):
            variable = ds.createVariable(name, 'f8', (name,))
            variable.standard_name = standard
            variable.units = 'degrees' if kind == 'rotated' else units
            variable.bounds = name + '_edges'
            variable[:] = values
            bounds = ds.createVariable(variable.bounds, 'f8', (name, 'vertices'))
            width = 1 if kind == 'rotated' else 1000. / factor
            bounds[:] = np.column_stack((values - .4 * width, values + .6 * width))
        mapping = ds.createVariable('arbitrary_mapping', 'i4')
        mapping.setncatts(cf)
        field = ds.createVariable('field', 'f8', ('northings', 'eastings'))
        field.grid_mapping = 'arbitrary_mapping'
        field.units = 'kg'
        field.long_name = 'cell total'
        field[:] = 1.
    return reference, x, y


@pytest.mark.parametrize('kind,units', [('lcc', 'm'), ('lcc', 'km'), ('mercator', 'km'), ('rotated', 'm')])
def test_cf_mapping_axes_units_and_actual_bounds(kind, units, tmp_path):
    """Recover centres/corners from arbitrary CF names, units and true cell intervals."""
    path = str(tmp_path / 'cf.nc')
    reference, x, y = write_cf_input(path, kind, units)
    grid = nes.open_netcdf(path, comm=MPI.COMM_SELF, time_exception=True)
    grid.load()
    assert list(grid.variables) == ['field']
    factor = 1000 if units == 'km' and kind != 'rotated' else 1
    transformer = Transformer.from_crs(
        reference, CRS.from_proj4('+proj=longlat +a=6378137 +rf=298.257222101'), always_xy=True)
    lon, lat = transformer.transform(*np.meshgrid(x * factor, y * factor))
    np.testing.assert_allclose(grid.lat['data'], lat, rtol=0, atol=1e-10)
    np.testing.assert_allclose(grid.lon['data'], lon, rtol=0, atol=1e-10)
    grid.create_spatial_bounds()
    x0, y0 = x[0] * factor, y[0] * factor
    width = 1 if kind == 'rotated' else 1000
    lon, lat = transformer.transform([x0 - .4 * width, x0 + .6 * width, x0 + .6 * width, x0 - .4 * width],
                                     [y0 - .4 * width, y0 - .4 * width, y0 + .6 * width, y0 + .6 * width])
    np.testing.assert_allclose(grid.lat_bnds['data'][0, 0], lat, rtol=0, atol=1e-10)
    output = str(tmp_path / 'cf_output.nc')
    grid.to_netcdf(output)
    reopened = nes.open_netcdf(output, comm=MPI.COMM_SELF)
    assert reopened.get_grid_configuration() == grid.get_grid_configuration()
    np.testing.assert_allclose(reopened.lat_bnds['data'], grid.lat_bnds['data'], atol=1e-10)
    grid.close()
    reopened.close()


@pytest.mark.parametrize('kind', ('lcc', 'mercator', 'laea'))
def test_singleton_and_descending_native_bounds(kind):
    """Keep finite corners for singleton grids and signed projected increments."""
    grid = make_grid(kind, nx=1, ny=2, inc_x=-1000., inc_y=-1000.)
    grid.create_spatial_bounds()
    assert grid.lat_bnds['data'].shape == (2, 1, 4)
    assert np.isfinite(grid.lat_bnds['data']).all()
    assert grid.get_grid_configuration()['axes']['x']['step'] == -1000.


def test_copy_restores_resources_on_failure():
    """Restore communicator/dataset references even if deepcopy cannot copy a member."""
    class Uncopyable:
        """Deliberately reject copying to exercise resource restoration."""

        def __deepcopy__(self, memo):
            """Raise the failure expected by the resource restoration regression."""
            raise RuntimeError('cannot copy')
    grid = make_grid('regular')
    grid.uncopyable = Uncopyable()
    communicator, dataset = grid.comm, grid.dataset
    with pytest.raises(RuntimeError, match='cannot copy'):
        grid.copy()
    assert grid.comm is communicator
    assert grid.dataset is dataset


def write_observations(path, kind):
    """Write small station/time fixtures for specialized observation readers.

    Parameters
    ----------
    path : str
        Destination NetCDF path.
    kind : {'ghost', 'providentia'}
        Observation layout to construct.
    """
    with Dataset(path, 'w') as ds:
        ds.createDimension('station', 4)
        ds.createDimension('time', 2)
        ds.version = '1.4'
        time = ds.createVariable('time', 'f8', ('time',))
        time.units = 'hours since 2022-01-01 00:00:00'
        time[:] = [0, 1]
        for name, values, units in [('lat', [40., 41., 42., 43.], 'degrees_north'),
                                    ('lon', [1., 2., 3., 4.], 'degrees_east')]:
            variable = ds.createVariable(name, 'f8', ('station',))
            variable.units = units
            variable[:] = values
        field = ds.createVariable('field', 'f8', ('station', 'time'))
        field.units = 'kg'
        field.long_name = 'station measurement'
        field[:] = np.arange(8.).reshape(4, 2)
        if kind == 'ghost':
            for name in ('flag', 'qa'):
                dim = 'N_' + name + '_codes'
                ds.createDimension(dim, 1)
                variable = ds.createVariable(name, 'i4', ('station', 'time', dim))
                variable[:] = 1
        else:
            ds.createDimension('model_latitude', 2)
            ds.createDimension('model_longitude', 2)
            ds.createDimension('grid_edge', 5)
            for name, values in [('model_centre_latitude', [[40, 40], [41, 41]]),
                                 ('model_centre_longitude', [[1, 2], [1, 2]])]:
                variable = ds.createVariable(name, 'f8', ('model_latitude', 'model_longitude'))
                variable[:] = values
            for name, values in [('grid_edge_latitude', [39, 39, 42, 42, 39]),
                                 ('grid_edge_longitude', [0, 3, 3, 0, 0])]:
                variable = ds.createVariable(name, 'f8', ('grid_edge',))
                variable[:] = values


@pytest.mark.parametrize('kind', ('ghost', 'providentia'))
def test_observation_metadata_roundtrip_and_writer_options(kind, tmp_path):
    """Preserve specialized metadata and forward common native writer options."""
    path = str(tmp_path / (kind + '.nc'))
    write_observations(path, kind)
    grid = nes.open_netcdf(path, comm=MPI.COMM_SELF, parallel_method='X')
    grid.load()
    expected = nes.PointsNesGHOST if kind == 'ghost' else nes.PointsNesProvidentia
    assert type(grid) is expected
    recreated = expected.new(comm=MPI.COMM_SELF, path=path)
    assert type(recreated) is expected
    recreated.close()
    output = str(tmp_path / 'observations.nc')
    grid.to_netcdf(output, keep_open=True, parallel_io_mode='collective')
    handle = grid.serial_nc.dataset if grid.serial_nc is not None else grid.dataset
    assert handle is not None and handle.isopen()
    assert handle.filepath() == output
    grid.close()
    opened = nes.open_netcdf(output, comm=MPI.COMM_SELF, parallel_method='X')
    opened.load()
    np.testing.assert_array_equal(opened.variables['field']['data'], np.arange(8.).reshape(4, 2))
    if kind == 'ghost':
        opened.erase_flags()
        assert opened.flag['data'].shape == (4, 2, 0)
        assert opened.get_full_qa()['data'].shape == (4, 2, 0)
        points = opened.to_points()
        assert points.variables['field']['dimensions'] == ('time', 'station')
        np.testing.assert_array_equal(points.variables['field']['data'], np.arange(8.).reshape(4, 2).T)
        assert 'field' in opened.variables
    else:
        assert opened.model_centre_lon['data'].shape == (2, 2)
    opened.close()


def test_points_to_providentia_retains_units_and_data(tmp_path):
    """Convert point data to station/time ordering without discarding scientific metadata."""
    source = nes.create_nes(comm=MPI.COMM_SELF, lat=np.array([40., 41., 42., 43.]),
                            lon=np.array([1., 2., 3., 4.]),
                            times=[datetime(2022, 1, 1), datetime(2022, 1, 1, 1)])
    data = np.arange(8.).reshape(2, 4)
    source.variables['field'] = dict(data=data, dimensions=('time', 'station'), units='kg', long_name='measurement')
    grid = source.to_providentia(model_centre_lon={'data': np.array([[1., 2.], [1., 2.]])},
                                 model_centre_lat={'data': np.array([[40., 40.], [41., 41.]])},
                                 grid_edge_lon={'data': np.array([0., 3., 3., 0., 0.])},
                                 grid_edge_lat={'data': np.array([39., 39., 42., 42., 39.])})
    assert grid.variables['field']['units'] == 'kg'
    assert grid.variables['field']['long_name'] == 'measurement'
    assert grid.variables['field']['dimensions'] == ('station', 'time')
    np.testing.assert_array_equal(grid.variables['field']['data'], data.T)
    output = str(tmp_path / 'providentia.nc')
    grid.to_netcdf(output)
    reopened = nes.open_netcdf(output, comm=MPI.COMM_SELF, parallel_method='X')
    reopened.load()
    np.testing.assert_array_equal(reopened.variables['field']['data'], data.T)
    assert reopened.variables['field']['units'] == 'kg'
    reopened.close()


@pytest.mark.parametrize('kind,field', [('lcc', 'longitude_of_central_meridian'),
                                        ('mercator', 'longitude_of_projection_origin'),
                                        ('rotated', 'grid_north_pole_latitude')])
def test_missing_required_cf_parameter_is_rejected(kind, field, tmp_path):
    """Reject missing mandatory CF parameters instead of relying on projection defaults."""
    path = str(tmp_path / 'invalid.nc')
    write_cf_input(path, kind)
    with Dataset(path, 'a') as ds:
        ds.variables['arbitrary_mapping'].delncattr(field)
    with pytest.raises(ValueError, match=field):
        nes.open_netcdf(path, comm=MPI.COMM_SELF, time_exception=True)


def test_unknown_cf_projection_is_not_read_as_latlon(tmp_path):
    """Report unsupported CF mappings rather than misclassifying them as regular grids."""
    path = str(tmp_path / 'unknown.nc')
    write_cf_input(path, 'mercator')
    with Dataset(path, 'a') as ds:
        ds.variables['arbitrary_mapping'].grid_mapping_name = 'unsupported_projection'
    with pytest.raises(NotImplementedError, match='Unsupported CF grid mapping'):
        nes.open_netcdf(path, comm=MPI.COMM_SELF, time_exception=True)


def test_laea_configuration_detects_different_cell_bounds(tmp_path):
    """Compare projected intervals as well as centres and Earth shape in canonical metadata."""
    from nes.tests.projections.test_laea import synthetic
    first, second = str(tmp_path / 'first.nc'), str(tmp_path / 'second.nc')
    synthetic(first)
    synthetic(second)
    with Dataset(second, 'a') as ds:
        ds.variables['easting_edges'][:] += 10.
    left = nes.open_netcdf(first, comm=MPI.COMM_SELF, time_exception=True)
    right = nes.open_netcdf(second, comm=MPI.COMM_SELF, time_exception=True)
    with pytest.raises(ValueError, match='bounds_offsets'):
        right.validate_grid_configuration(expected=left.get_grid_configuration())
    left.close()
    right.close()


def test_global_monarch_bounds_are_contiguous_and_recoverable(tmp_path):
    """Use true half-width outer cells rather than overlapping centre-based intervals."""
    grid = nes.create_nes(projection='global_monarch', comm=MPI.COMM_SELF, inc_lat=45., inc_lon=60.)
    path = str(tmp_path / 'monarch.nc')
    # Recover increments from persisted canonical metadata even without prewritten bounds.
    grid.to_netcdf(path)
    opened = nes.open_netcdf(path, comm=MPI.COMM_SELF)
    for source in (grid, opened):
        source.create_spatial_bounds()
        for name, lower, upper in [('lat_bnds', -90., 90.), ('lon_bnds', -180., 180.)]:
            values = getattr(source, name)['data']
            assert values[0, 0] == lower
            assert values[-1, 1] == upper
            np.testing.assert_allclose(values[:-1, 1], values[1:, 0])
            np.testing.assert_allclose(np.sum(values[:, 1] - values[:, 0]), upper - lower)
    opened.close()


@pytest.mark.parametrize('kind', ('regular', 'lcc', 'mercator', 'rotated'))
def test_creation_validation_runs_before_native_coordinate_allocation(kind):
    """Reject invalid scalar geometry uniformly instead of truncating or allocating invalid shapes."""
    key = 'n_lat' if kind == 'regular' else 'inc_rlat' if kind == 'rotated' else 'nx'
    value = 0 if kind == 'rotated' else 1.5
    with pytest.raises(ValueError, match=key):
        make_grid(kind, **{key: value})


def test_point_creation_accepts_lists_and_rejects_mismatched_axes():
    """Normalize valid array-like station coordinates and reject incompatible lengths."""
    grid = nes.create_nes(comm=MPI.COMM_SELF, lat=[40., 41.], lon=[1., 2.], parallel_method='X')
    assert grid.lat['data'].shape == (2,)
    with pytest.raises(ValueError, match='corresponding'):
        nes.create_nes(comm=MPI.COMM_SELF, lat=[40., 41.], lon=[1.], parallel_method='X')


def test_geographical_radian_units_are_converted_with_bounds(tmp_path):
    """Keep geographical coordinate/bounds units consistent when valid CF input uses radians."""
    path = str(tmp_path / 'radians.nc')
    with Dataset(path, 'w') as ds:
        for name, values in [('lat', [40., 41.]), ('lon', [1., 2., 3.])]:
            ds.createDimension(name, len(values))
            v = ds.createVariable(name, 'f8', (name,))
            v.standard_name = 'latitude' if name == 'lat' else 'longitude'
            v.units = 'radians'
            v[:] = np.deg2rad(values)
        ds.createDimension('vertices', 2)
        for name, values in [('lat', [40., 41.]), ('lon', [1., 2., 3.])]:
            ds.variables[name].bounds = name + '_edges'
            b = ds.createVariable(name + '_edges', 'f8', (name, 'vertices'))
            b[:] = np.deg2rad(np.column_stack((np.array(values) - .5, np.array(values) + .5)))
        field = ds.createVariable('field', 'f8', ('lat', 'lon'))
        field[:] = 1.
    grid = nes.open_netcdf(path, comm=MPI.COMM_SELF, time_exception=True)
    np.testing.assert_allclose(grid.lat['data'], [40., 41.])
    np.testing.assert_allclose(grid.lon['data'], [1., 2., 3.])
    np.testing.assert_allclose(grid.lat_bnds['data'], [[39.5, 40.5], [40.5, 41.5]])
    grid.close()


@pytest.mark.parametrize('kind', ('regular', 'rotated', 'lcc', 'mercator'))
def test_conflicting_earth_metadata_is_rejected(kind):
    """Reject contradictory explicit Earth definitions consistently across structured grids."""
    with pytest.raises(ValueError, match='cannot be combined'):
        make_grid(kind, earth_radius=6371000., semi_major_axis=6378137., inverse_flattening=298.257222101)


@pytest.mark.parametrize('kind', ('lcc', 'mercator', 'rotated', 'laea'))
def test_projected_perimeter_uses_native_units_and_clockwise_order(kind):
    """Produce closed clockwise outlines directly from the original native cell edges."""
    grid = make_grid(kind)
    latitude, longitude = grid.create_providentia_exp_grid_edge_coordinates()
    assert latitude['data'][0] == latitude['data'][-1]
    assert longitude['data'][0] == longitude['data'][-1]
    target = grid.projection.crs
    transformer = Transformer.from_crs(grid._geographic_crs, target, always_xy=True)
    x, y = transformer.transform(longitude['data'], latitude['data'])
    area = np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) / 2
    assert area < 0


def test_ghost_time_filter_keeps_full_and_local_flags_consistent(tmp_path):
    """Trim complete GHOST flags to the retained time range before writing."""
    path = str(tmp_path / 'ghost.nc')
    write_observations(path, 'ghost')
    source = nes.open_netcdf(path, comm=MPI.COMM_SELF, parallel_method='X', avoid_first_hours=1)
    source.load()
    assert source.get_full_flags()['data'].shape == (4, 1, 1)
    assert source.flag['data'].shape == (4, 1, 1)
    output = str(tmp_path / 'ghost_trimmed.nc')
    source.to_netcdf(output)
    reopened = nes.open_netcdf(output, comm=MPI.COMM_SELF, parallel_method='X')
    reopened.load()
    np.testing.assert_array_equal(reopened.variables['field']['data'], np.arange(8.).reshape(4, 2)[:, 1:])
    reopened.close()
    source.close()


@pytest.mark.parametrize('kind', KINDS)
def test_independent_cf_validation_of_structured_writers(kind, tmp_path):
    """Check native structured output with an optional independent CF-1.7 checker."""
    pytest.importorskip('compliance_checker')
    from compliance_checker.runner import ComplianceChecker, CheckSuite
    grid = make_grid(kind)
    path = str(tmp_path / (kind + '.nc'))
    grid.to_netcdf(path)
    CheckSuite.load_all_available_checkers()
    passed, errors = ComplianceChecker.run_checker(path, ['cf:1.7'], 0, 'normal',
                                                   output_filename=str(tmp_path / 'cf_report.txt'))
    report = (tmp_path / 'cf_report.txt').read_text()
    messages = [line.strip() for line in report.splitlines() if line.strip().startswith('* ')]
    if kind == 'mercator' and not passed and not errors and messages and all(
        re.fullmatch(r'\* . is a required attribute for grid mapping mercator', message)
        for message in messages
    ):
        with Dataset(path) as dataset:
            assert 'longitude_of_projection_origin' in dataset.variables['mercator'].ncattrs()
        pytest.xfail('IOOS checker treats Mercator longitude_of_projection_origin as characters, not one field')
    assert passed and not errors, report


@pytest.mark.parametrize('kind', KINDS)
def test_direct_projection_constructor_has_default_times(kind):
    """Use the factory's default time coordinate through every direct structured constructor."""
    expected = make_grid(kind)
    actual = type(expected).new(comm=MPI.COMM_SELF, create_nes=True, **parameters(kind))
    assert actual.time == expected.time


def test_ghost_scratch_creation_reports_missing_format_metadata():
    """Report the unsupported format-specific creation path before making partial flag arrays."""
    with pytest.raises(NotImplementedError, match='existing observation file'):
        nes.PointsNesGHOST.new(comm=MPI.COMM_SELF, create_nes=True, lat=[40.], lon=[1.])


@pytest.mark.parametrize('nc_type', ['DEFAULT', 'MONARCH'])
@pytest.mark.parametrize('descending', [False, True])
def test_rotated_float32_axes_write_native_bounds(tmp_path, nc_type, descending):
    """Write quantized legacy axes without rejecting regular grids or changing data."""
    grid = make_grid('rotated', west_boundary=-6.95, south_boundary=-6.95, inc_rlat=.15, inc_rlon=.15)
    axes = {}
    for name in ('rlat', 'rlon'):
        info = getattr(grid, 'get_full_' + name)().copy()
        values = info['data'].astype(np.float32)
        if descending:
            values = values[::-1].copy()
        info['data'] = values
        info.pop('bounds_data', None)
        getattr(grid, 'set_full_' + name)(info)
        axes[name] = values.copy()
    shape = (1, 1, len(axes['rlat']), len(axes['rlon']))
    grid.variables = {'test': {'data': np.ones(shape), 'units': 'kg.m-2.s-1'}}
    filename = tmp_path / f'float32-{nc_type}-{descending}.nc'
    grid.to_netcdf(str(filename), serial=True, nc_type=nc_type)
    grid.close()
    with Dataset(filename) as dataset:
        for name, values in axes.items():
            coordinate = dataset.variables[name]
            np.testing.assert_array_equal(coordinate[:], values)
            bounds = dataset.variables[coordinate.bounds][:]
            np.testing.assert_allclose(bounds.mean(axis=1), values, rtol=0, atol=1e-9)
        np.testing.assert_array_equal(dataset.variables['test'][:], np.ones(shape))
