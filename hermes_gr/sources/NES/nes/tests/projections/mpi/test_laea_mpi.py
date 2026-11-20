# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
# Licensed under the Apache License, Version 2.0. See LICENSE for details.
"""Run with mpiexec -n 2 python -m pytest nes/tests/projections/test_laea_mpi.py."""

import numpy as np
import pytest
from mpi4py import MPI
from netCDF4 import Dataset, __has_parallel4_support__

import nes
from nes.tests.projections.test_laea import grid, SPHERE


@pytest.mark.parametrize("method", ["X", "Y", "T"])
def test_mpi_geometry_conservation_and_io(tmp_path, method):
    """Match serial geometry, conservative results and NetCDF data on two MPI ranks.
    """
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("Requires mpiexec -n 2")
    if method == "T":
        from datetime import datetime, timedelta
        times = [datetime(2022, 1, 1) + timedelta(hours=i) for i in range(comm.size)]
        src = grid(SPHERE, comm=comm, x_0=0, y_0=0, parallel_method=method, times=times)
        src.variables["emission"]["data"] = np.ones((len(src.time), 1) + src.lat["data"].shape)
    else:
        src = grid(SPHERE, comm=comm, x_0=0, y_0=0, parallel_method=method)
    serial = grid(SPHERE, x_0=0, y_0=0)
    limits = src.read_axis_limits
    section = (slice(limits["y_min"], limits["y_max"]), slice(limits["x_min"], limits["x_max"]))
    np.testing.assert_allclose(src.lat["data"], serial.lat["data"][section], atol=1e-11)
    np.testing.assert_allclose(src.lat_bnds["data"], serial.lat_bnds["data"][section], atol=1e-11)
    dst_args = dict(projection="regular", lat_orig=-30.1, lon_orig=134.9,
                    inc_lat=.02, inc_lon=.02, n_lat=12, n_lon=12)
    dst = nes.create_nes(comm=comm, parallel_method=method, times=src.get_full_times(), **dst_args)
    result = src.interpolate_horizontal(dst, kind="Conservative")
    serial_dst = nes.create_nes(comm=MPI.COMM_SELF, **dst_args)
    serial_result = serial.interpolate_horizontal(serial_dst, kind="Conservative")
    limits = result.read_axis_limits
    section = (slice(limits["y_min"], limits["y_max"]), slice(limits["x_min"], limits["x_max"]))
    np.testing.assert_allclose(result.variables["emission"]["data"][0, 0],
                               serial_result.variables["emission"]["data"][0, 0][section], equal_nan=True)
    total = comm.allreduce(np.nansum(result.variables["emission"]["data"]))
    np.testing.assert_allclose(total, 16 * (comm.size if method == "T" else 1), rtol=1e-9)
    if not __has_parallel4_support__:
        pytest.skip("Geometry/conservation passed; netCDF4 lacks parallel I/O support")
    path = comm.bcast(str(tmp_path / "mpi_laea.nc") if comm.rank == 0 else None, root=0)
    src.to_netcdf(path)
    comm.Barrier()
    reopened = nes.open_netcdf(path, comm=comm, parallel_method=method)
    reopened.load()
    np.testing.assert_allclose(reopened.lat["data"], src.lat["data"])
    np.testing.assert_allclose(reopened.variables["emission"]["data"], src.variables["emission"]["data"])
    reopened.close()
    if comm.rank == 0:
        with Dataset(path) as ds:
            assert ds.variables["emission"].grid_mapping == "laea"
    comm.Barrier()
