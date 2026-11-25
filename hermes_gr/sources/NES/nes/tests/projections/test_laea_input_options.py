# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
# Licensed under the Apache License, Version 2.0. See LICENSE for details.
"""Explicit source metadata, lazy axis adaptation and CF output without archive dependencies."""

from datetime import datetime
from hashlib import sha256

import numpy as np
import pytest
from mpi4py import MPI
from netCDF4 import Dataset, date2num
from pyproj import CRS, Transformer

import nes


OPTIONS = dict(crs='EPSG:3035', coordinate_names={'x': 'LON', 'y': 'LAT'},
               coordinate_units={'x': 'm', 'y': 'm'})


def write_input(path, dimensions=('LON', 'LAT'), descending=False, cf=False, units=None):
    """Write an independent asymmetric raster, optionally with complete CF geometry.

    Parameters
    ----------
    path : pathlib.Path or str
        Destination file.
    dimensions : tuple of str, optional
        Source data dimension order. Optional time and lev dimensions have two values.
    descending : bool, optional
        Reverse projected coordinate order while retaining index-coded data.
    cf : bool, optional
        Supply CF mapping and axis standard names if True.
    units : str or None, optional
        Coordinate unit label; None omits the attribute. km also scales axis values.

    Returns
    -------
    numpy.ndarray
        Expected data in canonical time/level/Y/X order, with a missing value as NaN.
    """
    sizes = {'LON': 5, 'LAT': 3, 'time': 2, 'lev': 2}
    full = np.arange(2 * 2 * 3 * 5, dtype=float).reshape(2, 2, 3, 5) + 1
    full[:, :, 1, 2] = np.nan
    full = full[:2 if 'time' in dimensions else 1, :2 if 'lev' in dimensions else 1]
    canonical = tuple(d for d in ('time', 'lev', 'LAT', 'LON') if d in dimensions)
    data = full.reshape(tuple(sizes[d] for d in canonical)).transpose(tuple(canonical.index(d) for d in dimensions))
    with Dataset(path, 'w') as ds:
        ds.Conventions = 'CF-1.0' if not cf else 'CF-1.7'
        ds.title = 'Independent projected source'
        ds.history = 'Created for explicit-source regression'
        for name in dimensions:
            ds.createDimension(name, sizes[name])
        for axis, name, first in (('x', 'LON', 4320500.), ('y', 'LAT', 3210500.)):
            coordinate = ds.createVariable(name, 'f4', (name,))
            coordinate.standard_name = 'projection_' + axis + '_coordinate' if cf else name
            if units is not None:
                coordinate.units = units
            values = first + np.arange(sizes[name]) * 1000.
            coordinate[:] = (values[::-1] if descending else values) / (1000 if units == 'km' else 1)
        if 'time' in dimensions:
            time = ds.createVariable('time', 'f8', ('time',))
            time.units = 'hours since 2022-01-01'
            time.calendar = 'standard'
            time.standard_name = 'time'
            time[:] = date2num([datetime(2022, 1, 1), datetime(2022, 1, 1, 1)], time.units)
        if 'lev' in dimensions:
            lev = ds.createVariable('lev', 'i4', ('lev',))
            lev.units = '1'
            lev[:] = [1, 2]
        variable = ds.createVariable('emission', 'f8', dimensions, fill_value=-9999.)
        variable.long_name = 'Mass emitted per cell'
        variable.units = 'kg'
        variable[:] = np.ma.masked_invalid(data)
        if cf:
            mapping = ds.createVariable('source_mapping', 'i4')
            attributes = CRS.from_epsg(3035).to_cf()
            if units == 'km':
                attributes['false_easting'] /= 1000
                attributes['false_northing'] /= 1000
            mapping.setncatts(attributes)
            variable.grid_mapping = 'source_mapping'
    return full


def open_input(path, **kwargs):
    """Open an explicitly described projected raster on a supplied communicator.

    Parameters
    ----------
    path : pathlib.Path or str
        Source input path.
    **kwargs : dict
        Overrides for explicit source options and base reader arguments.

    Returns
    -------
    nes.LAEANes
        Lazy input object with canonical spatial geometry.
    """
    options = dict(OPTIONS, comm=MPI.COMM_SELF, time_exception=True)
    options.update(kwargs)
    return nes.open_netcdf(str(path), **options)


@pytest.mark.parametrize('dimensions', [('LON', 'LAT'), ('LAT', 'LON'), ('time', 'LON', 'LAT'),
                                        ('LON', 'time', 'LAT'), ('time', 'LON', 'lev', 'LAT')])
@pytest.mark.parametrize('descending', [False, True])
def test_explicit_metadata_roundtrip_without_changing_source(tmp_path, dimensions, descending):
    """Preserve source bytes, masked values and index orientation through lazy reads and CF output."""
    path = tmp_path / 'original.nc'
    expected = write_input(path, dimensions, descending)
    original_hash = sha256(path.read_bytes()).hexdigest()
    grid = open_input(path, time_exception='time' not in dimensions)
    assert isinstance(grid, nes.LAEANes)
    assert grid.variables['emission']['data'] is None
    assert grid.lat['data'].shape == (3, 5)
    transformer = Transformer.from_crs(3035, 4258, always_xy=True)
    xx, yy = np.meshgrid(grid.x['data'], grid.y['data'])
    longitude, latitude = transformer.transform(xx, yy)
    np.testing.assert_allclose(grid.lat['data'], latitude, atol=1e-10)
    np.testing.assert_allclose(grid.lon['data'], longitude, atol=1e-10)
    if 'time' not in dimensions:
        grid.set_time([datetime(2022, 1, 1)])
        grid.set_time_bnds([[datetime(2022, 1, 1), datetime(2023, 1, 1)]])
    grid.load()
    np.testing.assert_allclose(grid.variables['emission']['data'], expected, equal_nan=True)
    output = str(tmp_path / 'formatted.nc')
    grid.to_netcdf(output)
    reopened = nes.open_netcdf(output, comm=MPI.COMM_SELF)
    reopened.load()
    np.testing.assert_allclose(reopened.variables['emission']['data'], expected, equal_nan=True)
    assert reopened.time == grid.time
    assert reopened.get_grid_configuration() == grid.get_grid_configuration()
    if 'time' not in dimensions:
        assert reopened.time_bnds == grid.time_bnds
    with Dataset(output) as ds:
        assert ds.variables['emission'].dimensions == ('time', 'lev', 'y', 'x')
        assert ds.variables['laea'].grid_mapping_name == 'lambert_azimuthal_equal_area'
        assert ds.variables['x'].units == 'm'
        assert ds.Conventions == 'CF-1.7'
    reopened.close()
    grid.close()
    assert sha256(path.read_bytes()).hexdigest() == original_hash


@pytest.mark.parametrize('cf', [False, True])
def test_native_kilometres_and_offsets_are_preserved(tmp_path, cf):
    """Express EPSG false offsets in native kilometres without scaling source data or centres."""
    path = tmp_path / 'kilometres.nc'
    expected = write_input(path, cf=cf, units='km')
    grid = open_input(path, coordinate_units={'x': 'km', 'y': 'km'})
    assert grid.projection_data['false_easting'] == pytest.approx(4321.)
    np.testing.assert_allclose(grid.x['data'], np.array([4320.5, 4321.5, 4322.5, 4323.5, 4324.5]))
    grid.load()
    np.testing.assert_allclose(grid.variables['emission']['data'], expected, equal_nan=True)
    output = str(tmp_path / 'formatted.nc')
    grid.to_netcdf(output)
    reopened = nes.open_netcdf(output, comm=MPI.COMM_SELF)
    np.testing.assert_allclose(reopened.lat['data'], grid.lat['data'])
    assert reopened.get_grid_configuration() == grid.get_grid_configuration()
    reopened.close()
    grid.close()


@pytest.mark.parametrize('options,match', [
    ({'crs': None}, 'explicit crs'),
    ({'coordinate_names': {'x': 'absent', 'y': 'LAT'}}, 'does not exist'),
    ({'coordinate_names': {'x': 'LON', 'y': 'LON'}}, 'separate'),
    ({'coordinate_names': {'longitude': 'LON'}}, 'coordinate_names'),
    ({'coordinate_units': {'x': 'degrees', 'y': 'm'}}, 'length units'),
    ({'coordinate_units': ['m', 'm']}, 'coordinate_units'),
])
def test_invalid_explicit_metadata_is_rejected(tmp_path, options, match):
    """Fail explicitly on incomplete or ambiguous source descriptions before reading emissions."""
    path = tmp_path / 'invalid.nc'
    write_input(path)
    with pytest.raises(ValueError, match=match):
        open_input(path, **options)


def test_explicit_non_laea_crs_is_rejected(tmp_path):
    """Do not reinterpret a geographic CRS as a projected source."""
    path = tmp_path / 'invalid.nc'
    write_input(path)
    with pytest.raises(NotImplementedError, match='Lambert Azimuthal Equal Area'):
        open_input(path, crs='EPSG:4326')


@pytest.mark.parametrize('field,value', [('latitude_of_projection_origin', 50.), ('semi_major_axis', 6371000.),
                                         ('false_easting', 0.)])
def test_existing_cf_projection_conflicts_are_rejected(tmp_path, field, value):
    """Existing numerical CF metadata cannot silently be replaced by an explicit CRS."""
    path = tmp_path / 'conflict.nc'
    write_input(path, cf=True, units='m')
    with Dataset(path, 'a') as ds:
        ds.variables['source_mapping'].setncattr(field, value)
    with pytest.raises(ValueError):
        open_input(path)


def test_existing_coordinate_units_conflict_is_rejected(tmp_path):
    """An explicit unit label must not rescale an already described source axis."""
    path = tmp_path / 'conflict.nc'
    write_input(path, units='m')
    with pytest.raises(ValueError, match='units conflict'):
        open_input(path, coordinate_units={'x': 'km', 'y': 'm'})


def test_existing_cf_axes_can_be_discovered_with_only_crs(tmp_path):
    """Reuse CF axis metadata while adapting a reversed data dimension order."""
    path = tmp_path / 'cf.nc'
    expected = write_input(path, cf=True, units='m')
    grid = open_input(path, coordinate_names=None, coordinate_units=None)
    grid.load()
    np.testing.assert_allclose(grid.variables['emission']['data'], expected, equal_nan=True)
    grid.close()


def test_independent_cf_validation_of_normalized_file(tmp_path):
    """Check normalized geometry, annual time bounds and supplied scientific metadata independently."""
    pytest.importorskip('compliance_checker')
    from compliance_checker.runner import ComplianceChecker, CheckSuite
    source = tmp_path / 'original.nc'
    write_input(source)
    grid = open_input(source)
    grid.set_time([datetime(2022, 1, 1)])
    grid.set_time_bnds([[datetime(2022, 1, 1), datetime(2023, 1, 1)]])
    grid.load()
    output = str(tmp_path / 'formatted.nc')
    grid.to_netcdf(output)
    grid.close()
    CheckSuite.load_all_available_checkers()
    passed, errors = ComplianceChecker.run_checker(output, ['cf:1.7'], 0, 'normal',
                                                   output_filename=str(tmp_path / 'cf_report.txt'))
    assert passed and not errors, (tmp_path / 'cf_report.txt').read_text()


@pytest.mark.parametrize('crs', [3035, CRS.from_epsg(3035), CRS.from_epsg(3035).to_wkt()])
def test_explicit_crs_accepts_authority_code_object_and_wkt(tmp_path, crs):
    """Resolve equivalent explicit CRS representations to the same native geometry."""
    path = tmp_path / 'crs.nc'
    write_input(path)
    grid = open_input(path, crs=crs)
    assert grid.projection_data['latitude_of_projection_origin'] == 52.
    grid.close()


def test_explicit_crs_completes_partial_source_mapping(tmp_path):
    """Retain existing compatible attributes while completing missing projection and Earth metadata."""
    path = tmp_path / 'partial.nc'
    write_input(path, cf=True, units='m')
    with Dataset(path, 'a') as ds:
        mapping = ds.variables['source_mapping']
        for name in ('crs_wkt', 'semi_major_axis', 'semi_minor_axis', 'inverse_flattening',
                     'latitude_of_projection_origin'):
            mapping.delncattr(name)
    grid = open_input(path)
    assert grid.projection_data['semi_major_axis'] == 6378137.
    assert grid.projection_data['latitude_of_projection_origin'] == 52.
    grid.close()


def test_invalid_explicit_crs_reports_clear_error(tmp_path):
    """Reject malformed CRS identifiers with a reader-level error."""
    path = tmp_path / 'crs.nc'
    write_input(path)
    with pytest.raises(ValueError, match='Invalid explicit crs'):
        open_input(path, crs='definitely-not-a-crs')


def test_explicit_input_rejects_unknown_nonspatial_dimension(tmp_path):
    """Do not silently treat a scenario axis as time or discard its extra values."""
    path = tmp_path / 'scenario.nc'
    write_input(path, dimensions=('time', 'LON', 'LAT'))
    with Dataset(path, 'a') as ds:
        ds.renameVariable('time', 'scenario')
    with pytest.raises(ValueError, match='Non-spatial source dimensions'):
        open_input(path)


def test_explicit_crs_does_not_ignore_an_existing_different_mapping(tmp_path):
    """Reject contradictory source projection types rather than silently relabelling the raster."""
    path = tmp_path / 'other_mapping.nc'
    write_input(path, cf=True, units='m')
    with Dataset(path, 'a') as ds:
        ds.variables['source_mapping'].grid_mapping_name = 'lambert_conformal_conic'
    with pytest.raises(ValueError, match='existing source grid mapping'):
        open_input(path)
