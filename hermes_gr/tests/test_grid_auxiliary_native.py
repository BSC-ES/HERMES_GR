# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

"""Native GR auxiliary roundtrips, runnable under pytest with one or two MPI ranks."""

import hashlib
from datetime import datetime
import logging
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

import pytest

try:
    from mpi4py import MPI
    from netCDF4 import Dataset
    import geopandas as gpd
    import numpy as np
    from nes import create_nes
    import hermes_gr
    from hermes_gr.config import logger_config
    from hermes_gr.modules import grid as grid_module
    from hermes_gr.tests.test_grid_auxiliary_validation import PARAMETERS
except ImportError as error:
    pytest.skip(f"Native grid tests require the complete NES/MPI environment: {error}", allow_module_level=True)


@pytest.fixture
def native_grid(monkeypatch):
    """Isolate native grid I/O and the logger normally initialized by HermesGr.

    Each MPI rank receives a real logger without file handlers. Monkeypatch
    restores the previous logger and timing records after the test so direct
    select_grid calls neither depend on test order nor affect later tests.
    """
    comm = MPI.COMM_WORLD
    root = Path(comm.bcast(tempfile.mkdtemp(prefix="gr-grid-native-") if comm.rank == 0 else None, root=0))
    monkeypatch.setattr(hermes_gr, "DEBUG", False)
    logger = logging.Logger(f"native-grid-test-{comm.rank}")
    logger.addHandler(logging.NullHandler())
    monkeypatch.setattr(logger_config, "logger", logger)
    monkeypatch.setattr(logger_config, "time_records", logger_config.time_records.copy(deep=True))
    monkeypatch.setattr(logger_config, "memory_records", logger_config.memory_records.copy(deep=True))
    # Isolate unrelated country/timezone datasets. Grid creation, persistence,
    # canonical configuration recovery, spatial geometry and validation are real.

    def create_with_timezones(**kwargs):
        grid = create_nes(**kwargs)
        grid.variables["tzid"] = {"data": np.full(grid.get_fids().shape, "UTC")}
        grid.set_strlen(32)
        return grid

    monkeypatch.setattr(grid_module, "create_nes", create_with_timezones)
    monkeypatch.setattr(grid_module, "add_country_iso2_to_grid", lambda grid, countries, write_grid: write_grid)
    monkeypatch.setattr(grid_module, "add_time_info_to_grid", lambda grid, **kwargs: grid)
    try:
        yield comm, root
    finally:
        comm.Barrier()
        if comm.rank == 0:
            shutil.rmtree(root)
        comm.Barrier()


def options_for(root, kind="regular", **changes):
    values = dict(domain_type=kind, auxiliary_files_path=str(root), timezones_shapefile="unused", **PARAMETERS[kind])
    values.update(changes)
    return SimpleNamespace(**values)


def select(comm, options):
    grid = grid_module.select_grid(comm, options, [100., 200.], [datetime(2020, 1, 1)])
    # GeoJSON writing is master-side after gathering; synchronize test file assertions.
    comm.Barrier()
    return grid


@pytest.mark.parametrize("kind", PARAMETERS)
def test_native_creation_and_reuse(native_grid, monkeypatch, kind):
    comm, root = native_grid
    options = options_for(root, kind)
    if kind == "rotated_nested":
        parent = create_nes(comm=comm, projection="rotated", **PARAMETERS["rotated"])
        parent.to_netcdf(str(root / "parent.nc"), serial=True)
        parent.close()
        comm.Barrier()
        options.parent_grid_path = str(root / "parent.nc")
    created = select(comm, options)
    expected = created.get_grid_configuration()
    assert (root / "grid.nc").is_file()
    assert (root / "grid.geojson").is_file()
    original = hashlib.sha256((root / "grid.nc").read_bytes()).hexdigest()
    monkeypatch.setattr(grid_module, "create_nes", lambda **kwargs: pytest.fail("Reference grid must not be created"))
    monkeypatch.setattr(hermes_gr, "DEBUG", True)
    reused = select(comm, options)
    assert reused.get_grid_configuration() == expected
    assert hashlib.sha256((root / "grid.nc").read_bytes()).hexdigest() == original


@pytest.mark.parametrize("change", ["origin", "resolution", "projection"])
def test_native_equal_size_incompatibility_is_collective(native_grid, change):
    comm, root = native_grid
    select(comm, options_for(root))
    if change == "projection":
        requested = options_for(root, "lcc")
    else:
        requested = options_for(root, **({"lat_orig": 41.} if change == "origin" else {"inc_lon": .2}))
    original = (root / "grid.nc").read_bytes()
    error = None
    try:
        select(comm, requested)
    except ValueError as exception:
        error = str(exception)
    assert error is not None
    assert "grid.nc" in error and "expected" in error
    assert len(set(comm.allgather(error))) == 1
    assert (root / "grid.nc").read_bytes() == original


def test_native_legacy_metadata_recovery(native_grid):
    comm, root = native_grid
    options = options_for(root)
    created = select(comm, options)
    expected = created.get_grid_configuration()
    comm.Barrier()
    if comm.rank == 0:
        with Dataset(str(root / "grid.nc"), "r+") as dataset:
            dataset.delncattr("nes_grid_configuration")
    comm.Barrier()
    assert select(comm, options).get_grid_configuration() == expected


def test_native_missing_paired_geojson_is_not_regenerated(native_grid):
    comm, root = native_grid
    options = options_for(root)
    select(comm, options)
    original = (root / "grid.nc").read_bytes()
    if comm.rank == 0:
        (root / "grid.geojson").unlink()
    comm.Barrier()
    with pytest.raises(ValueError, match="Missing paired grid GeoJSON"):
        select(comm, options)
    assert not (root / "grid.geojson").exists()
    assert (root / "grid.nc").read_bytes() == original


@pytest.mark.parametrize("damage", ["geometry", "FID", "unreadable"])
def test_native_debug_checks_paired_geometry(native_grid, monkeypatch, damage):
    comm, root = native_grid
    options = options_for(root)
    select(comm, options)
    if comm.rank == 0:
        path = root / "grid.geojson"
        if damage == "unreadable":
            path.write_text("invalid GeoJSON")
        else:
            frame = gpd.read_file(path)
            if damage == "geometry":
                frame.geometry = frame.geometry.translate(xoff=.5)
            else:
                frame.loc[0, "FID"] = -1
            frame.to_file(path, driver="GeoJSON")
    comm.Barrier()
    # Normal validation is metadata/availability only, without geometry reads.
    select(comm, options)
    monkeypatch.setattr(hermes_gr, "DEBUG", True)
    with pytest.raises(ValueError, match="Invalid paired grid GeoJSON"):
        select(comm, options)


def test_native_rotated_monarch_output(native_grid):
    """Exercise MONARCH's float32 axis conversion on the actual GR output grid."""
    comm, root = native_grid
    options = options_for(root, "rotated", west_boundary=-6.95, south_boundary=-6.95, inc_rlat=.15, inc_rlon=.15)
    grid = select(comm, options)
    expected_axes = {name: getattr(grid, "get_full_" + name)()["data"].astype(np.float32) for name in ("rlat", "rlon")}
    grid.variables = {"test": {"data": np.full((1, 2) + grid.get_fids().shape, 2.5), "units": "kg.m-2.s-1"}}
    grid.global_attrs = {}
    output = root / "monarch-output.nc"
    grid.to_netcdf(str(output), serial=True, nc_type="MONARCH")
    grid.close()
    comm.Barrier()
    with Dataset(str(output)) as dataset:
        for name, values in expected_axes.items():
            coordinate = dataset.variables[name]
            np.testing.assert_array_equal(coordinate[:], values)
            np.testing.assert_allclose(dataset.variables[coordinate.bounds][:].mean(axis=1), values, rtol=0, atol=1e-9)
        np.testing.assert_array_equal(dataset.variables["test"][:], 2.5)
