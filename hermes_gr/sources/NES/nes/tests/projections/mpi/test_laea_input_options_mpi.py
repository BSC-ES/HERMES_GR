# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
# Licensed under the Apache License, Version 2.0. See LICENSE for details.
"""Two-rank lazy source transposition, annual metadata and normalized NetCDF round trips."""

from datetime import datetime

import numpy as np
import pytest
from mpi4py import MPI
from netCDF4 import __has_parallel4_support__

import nes
from nes.tests.projections.test_laea_input_options import open_input, write_input


@pytest.mark.parametrize('method', ['X', 'Y', 'T'])
def test_explicit_source_axis_order_and_roundtrip_on_two_ranks(tmp_path, method):
    """Distribute asymmetric source slices without swapping indices or duplicating cell totals."""
    comm = MPI.COMM_WORLD
    if comm.size != 2:
        pytest.skip('Requires mpiexec -n 2')
    source = comm.bcast(str(tmp_path / 'original.nc') if comm.rank == 0 else None, root=0)
    dimensions = ('LON', 'time', 'LAT') if method == 'T' else ('LON', 'LAT')
    expected = write_input(source, dimensions=dimensions) if comm.rank == 0 else None
    expected = comm.bcast(expected, root=0)
    comm.Barrier()
    grid = open_input(source, comm=comm, parallel_method=method, time_exception=method != 'T')
    if method != 'T':
        grid.set_time([datetime(2022, 1, 1)])
        grid.set_time_bnds([[datetime(2022, 1, 1), datetime(2023, 1, 1)]])
    grid.load()
    limits = grid.read_axis_limits
    selection = tuple(slice(limits[axis + '_min'], limits[axis + '_max']) for axis in ('t', 'z', 'y', 'x'))
    np.testing.assert_allclose(grid.variables['emission']['data'], expected[selection], equal_nan=True)
    total = comm.allreduce(np.nansum(grid.variables['emission']['data']))
    np.testing.assert_allclose(total, np.nansum(expected))
    if not __has_parallel4_support__:
        grid.close()
        pytest.skip('Distributed source read passed; parallel NetCDF writes unavailable')
    output = comm.bcast(str(tmp_path / 'formatted.nc') if comm.rank == 0 else None, root=0)
    grid.to_netcdf(output)
    comm.Barrier()
    reopened = nes.open_netcdf(output, comm=comm, parallel_method=method)
    reopened.load()
    np.testing.assert_allclose(reopened.variables['emission']['data'], expected[selection], equal_nan=True)
    assert reopened.get_grid_configuration() == grid.get_grid_configuration()
    assert reopened.time == grid.time
    reopened.close()
    grid.close()
