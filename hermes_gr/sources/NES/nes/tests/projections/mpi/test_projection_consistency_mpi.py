# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
# Licensed under the Apache License, Version 2.0. See LICENSE for details.
"""Two-process metadata, native geometry and serialization contracts for every grid layout."""

from datetime import datetime

import numpy as np
import pytest
from mpi4py import MPI
from netCDF4 import __has_parallel4_support__

import nes
from nes.tests.projections.test_projection_consistency import make_grid, write_observations


@pytest.mark.parametrize('kind', ('regular', 'global', 'global_monarch', 'rotated',
                                  'rotated-nested', 'lcc', 'mercator', 'laea'))
@pytest.mark.parametrize('method', ('X', 'Y', 'T'))
def test_structured_mpi_contracts(kind, method, tmp_path):
    """Match collective configurations, cell totals and serialized native geometry."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip('Requires mpiexec -n 2')
    times = [datetime(2022, 1, 1), datetime(2022, 1, 1, 1)]
    options = dict(parallel_method=method, times=times)
    if kind in ('global', 'global_monarch'):
        grid = nes.create_nes(comm=comm, projection=kind, inc_lat=45., inc_lon=60., **options)
    elif kind == 'rotated-nested':
        parent_path = comm.bcast(str(tmp_path / 'parent.nc') if comm.rank == 0 else None, root=0)
        if comm.rank == 0:
            make_grid('rotated').to_netcdf(parent_path)
        comm.Barrier()
        grid = nes.create_nes(comm=comm, projection=kind, parent_grid_path=parent_path, parent_ratio=2,
                              i_parent_start=2, j_parent_start=2, n_rlat=4, n_rlon=4, **options)
    else:
        grid = make_grid(kind, comm=comm, **options)
    definition = grid.get_grid_configuration()
    assert all(item == definition for item in comm.allgather(definition))
    fids = grid.get_fids()
    values = np.stack([fids + 1 + time.hour * 100 for time in grid.time])[:, None, ...].astype(float)
    grid.variables = {'field': dict(data=values, units='kg', long_name='cell total')}
    grid.create_spatial_bounds()
    assert grid.lat_bnds is not None
    result = grid.interpolate_horizontal(grid.copy(), kind='Conservative')
    np.testing.assert_allclose(result.variables['field']['data'], values, rtol=1e-9, atol=1e-8)
    np.testing.assert_allclose(comm.allreduce(np.sum(result.variables['field']['data'])),
                               comm.allreduce(np.sum(values)), rtol=1e-9)
    if not __has_parallel4_support__:
        pytest.skip('Geometry and configuration passed; netCDF4 lacks parallel I/O support')
    path = comm.bcast(str(tmp_path / 'structured.nc') if comm.rank == 0 else None, root=0)
    grid.to_netcdf(path)
    comm.Barrier()
    reopened = nes.open_netcdf(path, comm=comm, parallel_method=method)
    reopened.load()
    assert reopened.validate_grid_configuration(expected=definition)
    np.testing.assert_array_equal(reopened.variables['field']['data'], values)
    np.testing.assert_allclose(reopened.lat_bnds['data'], grid.lat_bnds['data'])
    reopened.close()


@pytest.mark.parametrize('kind', ('ghost', 'providentia'))
def test_observation_mpi_metadata_and_serial_writer(kind, tmp_path):
    """Share complete observation metadata while keeping station data distributed."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip('Requires mpiexec -n 2')
    path = comm.bcast(str(tmp_path / (kind + '.nc')) if comm.rank == 0 else None, root=0)
    if comm.rank == 0:
        write_observations(path, kind)
    comm.Barrier()
    grid = nes.open_netcdf(path, comm=comm, parallel_method='X')
    grid.load()
    local = grid.station['data']
    expected = np.arange(8.).reshape(4, 2)[local]
    np.testing.assert_array_equal(grid.variables['field']['data'], expected)
    if kind == 'ghost':
        assert grid.flag['data'].shape == (len(local), 2, 1)
        assert grid.get_full_flags()['data'].shape == (4, 2, 1)
        grid.erase_flags()
        assert grid.qa['data'].shape == (len(local), 2, 0)
        points = grid.to_points()
        assert len(points.lat['data']) == len(local)
        np.testing.assert_array_equal(points.variables['field']['data'], expected.T)
    else:
        assert grid.model_centre_lon['data'].shape == (2, 2)
        assert grid._grid_edge_lon['data'].shape == (5,)
    output = comm.bcast(str(tmp_path / 'observations_out.nc') if comm.rank == 0 else None, root=0)
    grid.to_netcdf(output, serial=True)
    comm.Barrier()
    reopened = nes.open_netcdf(output, comm=comm, parallel_method='X')
    reopened.load()
    np.testing.assert_array_equal(reopened.variables['field']['data'], expected)
    reopened.close()
    grid.close()


def test_points_time_decomposition_and_native_write(tmp_path):
    """Create and write station data with the same temporal decomposition supported by reading."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip('Requires mpiexec -n 2')
    grid = nes.create_nes(comm=comm, parallel_method='T', lat=[40., 41., 42., 43.], lon=[1., 2., 3., 4.],
                          times=[datetime(2022, 1, 1), datetime(2022, 1, 1, 1)])
    assert len(grid.time) == 1
    assert grid.lat['data'].shape == (4,)
    data = np.array([[time.hour * 10 + station for station in range(4)] for time in grid.time], dtype=float)
    grid.variables = {'field': dict(data=data, dimensions=('time', 'station'), units='kg')}
    path = comm.bcast(str(tmp_path / 'points_t.nc') if comm.rank == 0 else None, root=0)
    grid.to_netcdf(path, parallel_io_mode='independent')
    comm.Barrier()
    reopened = nes.open_netcdf(path, comm=comm, parallel_method='T')
    reopened.load()
    np.testing.assert_array_equal(reopened.variables['field']['data'], data)
    reopened.close()
