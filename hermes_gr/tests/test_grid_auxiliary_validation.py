# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

"""Isolate GR cache lifecycle and collective error handling from native grid I/O."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
PARAMETERS = {
    "global": dict(inc_lat=45., inc_lon=60.),
    "global_monarch": dict(inc_lat=45., inc_lon=60.),
    "regular": dict(lat_orig=40., lon_orig=0., inc_lat=.1, inc_lon=.1, n_lat=4, n_lon=4),
    "rotated": dict(centre_lat=40., centre_lon=0., south_boundary=-2., west_boundary=-2., inc_rlat=1., inc_rlon=1.),
    "rotated_nested": dict(parent_grid_path="parent.nc", parent_ratio=2, i_parent_start=2,
                           j_parent_start=2, n_rlat=4, n_rlon=4),
    "lcc": dict(lat_1=30., lat_2=60., lat_0=40., lon_0=0., nx=4, ny=4, inc_x=1000., inc_y=1000., x_0=0., y_0=0.),
    "mercator": dict(lat_ts=40., lon_0=0., nx=4, ny=4, inc_x=1000., inc_y=1000., x_0=0., y_0=0.),
}


def load_grid_functions(namespace):
    """Execute unchanged GR functions without importing native MPI/NES modules."""
    path = Path(__file__).resolve().parents[1] / "modules/grid.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    names = {"_grid_configuration_arguments", "_validate_grid_auxiliary", "_validate_grid_geojson_debug", "select_grid"}
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.fixture
def runtime(tmp_path):
    comm = Mock()
    comm.Get_rank.return_value = 0
    comm.bcast.side_effect = lambda value, root: value
    grid = SimpleNamespace(
        comm=comm, master=True, size=1, parallel_method="Y", shapefile={},
        lat={"data": np.zeros(1)}, lon={"data": np.zeros(1)},
        variables={"tzid": {"data": np.array([["UTC"]])}}, cell_measures={"cell_area": {}},
    )
    for name in ["validate_grid_configuration", "load", "close", "create_shapefile", "set_levels",
                 "to_netcdf", "write_shapefile", "calculate_grid_area"]:
        setattr(grid, name, Mock())
    namespace = load_grid_functions(dict(
        os=os, pd=pd, hermes_gr=SimpleNamespace(DEBUG=False), BALANCED=True,
        log_message=Mock(), get_time_stamp=lambda: 0, record_time=Mock(),
        open_netcdf=Mock(return_value=grid), create_nes=Mock(return_value=grid),
        add_country_iso2_to_grid=lambda grid, countries, write_grid: write_grid,
        add_time_info_to_grid=lambda grid, **kwargs: grid,
    ))
    options = SimpleNamespace(domain_type="regular", auxiliary_files_path=str(tmp_path),
                              timezones_shapefile="unused", **PARAMETERS["regular"])
    return SimpleNamespace(ns=namespace, grid=grid, comm=comm, options=options,
                           nc=tmp_path / "grid.nc", geo=tmp_path / "grid.geojson")


@pytest.mark.parametrize("kind", PARAMETERS)
@pytest.mark.parametrize("reuse", [False, True], ids=["create", "reuse"])
def test_same_geographical_arguments_for_creation_and_validation(runtime, kind, reuse):
    options = SimpleNamespace(domain_type=kind, auxiliary_files_path=runtime.options.auxiliary_files_path,
                              timezones_shapefile="unused", **PARAMETERS[kind])
    if reuse:
        runtime.nc.write_bytes(b"existing grid")
        runtime.geo.write_text("existing GeoJSON")
    result = runtime.ns["select_grid"](runtime.comm, options, levels=[100], time_step_list=[])
    assert result is runtime.grid
    projection = "rotated-nested" if kind == "rotated_nested" else kind
    if reuse:
        runtime.grid.validate_grid_configuration.assert_called_once_with(projection=projection, **PARAMETERS[kind])
        runtime.ns["create_nes"].assert_not_called()
        runtime.grid.to_netcdf.assert_not_called()
        runtime.grid.write_shapefile.assert_not_called()
    else:
        runtime.ns["create_nes"].assert_called_once_with(
            comm=runtime.comm, info=False, times=[], projection=projection, balanced=True, **PARAMETERS[kind])
        runtime.grid.validate_grid_configuration.assert_not_called()
        runtime.grid.to_netcdf.assert_called_once()
        runtime.grid.write_shapefile.assert_called_once()


def test_missing_geojson_rejected_before_geometry_or_writes(runtime):
    runtime.nc.write_bytes(b"unchanged grid")
    with pytest.raises(ValueError, match="Missing paired grid GeoJSON"):
        runtime.ns["select_grid"](runtime.comm, runtime.options, [100])
    assert runtime.nc.read_bytes() == b"unchanged grid"
    runtime.grid.create_shapefile.assert_not_called()
    runtime.grid.to_netcdf.assert_not_called()
    runtime.grid.close.assert_called_once()


@pytest.mark.parametrize("property_name", ["axes.x.origin", "axes.y.step", "projection"])
def test_nes_mismatch_preserved_and_no_recreation(runtime, property_name):
    runtime.nc.write_bytes(b"unchanged grid")
    runtime.geo.write_text("unchanged GeoJSON")
    runtime.grid.validate_grid_configuration.side_effect = ValueError(f"{property_name}: expected 1, found 2")
    with pytest.raises(ValueError, match=property_name) as raised:
        runtime.ns["select_grid"](runtime.comm, runtime.options, [100])
    assert str(runtime.nc) in str(raised.value)
    assert "expected 1, found 2" in str(raised.value)
    assert runtime.nc.read_bytes() == b"unchanged grid"
    assert runtime.geo.read_text() == "unchanged GeoJSON"
    runtime.ns["create_nes"].assert_not_called()
    runtime.grid.load.assert_not_called()
    runtime.grid.create_shapefile.assert_not_called()
    runtime.grid.to_netcdf.assert_not_called()


def test_installed_nes_without_api_has_actionable_error(runtime):
    runtime.grid.validate_grid_configuration = None
    with pytest.raises(RuntimeError, match="update NES"):
        runtime.ns["_validate_grid_auxiliary"](runtime.grid, "regular", {}, str(runtime.nc), str(runtime.geo))


def test_peer_receives_master_missing_file_error(runtime):
    runtime.grid.master = False
    runtime.comm.bcast.side_effect = None
    runtime.comm.bcast.return_value = "Missing paired grid GeoJSON: shared failure"
    with pytest.raises(ValueError, match="shared failure"):
        runtime.ns["_validate_grid_auxiliary"](runtime.grid, "regular", {}, str(runtime.nc), str(runtime.geo))
    runtime.comm.bcast.assert_called_once_with(None, root=0)


def test_normal_mode_does_not_collect_or_read_geometries(runtime):
    runtime.grid.shapefile = None
    runtime.ns["_validate_grid_geojson_debug"](runtime.grid, str(runtime.geo))
    runtime.comm.gather.assert_not_called()
    runtime.comm.bcast.assert_not_called()


@pytest.mark.parametrize("damage", [None, "equivalentCRS", "geometry", "FID", "CRS", "unreadable"])
def test_debug_geojson_validation(runtime, monkeypatch, damage):
    import sys

    class Geometry:
        def __init__(self, value):
            self.value = value

        def equals(self, other):
            return self.value == other.value

        def equals_exact(self, other, tolerance):
            return abs(self.value - other.value) <= tolerance

    class Frame(pd.DataFrame):
        _metadata = ["crs"]

        @property
        def _constructor(self):
            return Frame

        def to_crs(self, crs):
            result = self.copy()
            result.crs = crs
            if damage == "CRS":
                result["geometry"] = [Geometry(99)]
            return result

    expected = Frame({"geometry": [Geometry(1)]}, index=[0])
    expected.crs = "EPSG:4326"
    actual = Frame({"FID": [-1 if damage == "FID" else 0], "geometry": [Geometry(2 if damage == "geometry" else 1)]})
    actual.crs = "different encoding" if damage in ("CRS", "equivalentCRS") else expected.crs
    read_file = Mock(return_value=actual)
    if damage == "unreadable":
        read_file.side_effect = ValueError("Invalid GeoJSON")
    monkeypatch.setitem(sys.modules, "geopandas", SimpleNamespace(read_file=read_file))
    runtime.ns["hermes_gr"].DEBUG = True
    runtime.grid.shapefile = expected
    if damage not in (None, "equivalentCRS"):
        with pytest.raises(ValueError, match="Invalid paired grid GeoJSON"):
            runtime.ns["_validate_grid_geojson_debug"](runtime.grid, str(runtime.geo))
    else:
        runtime.ns["_validate_grid_geojson_debug"](runtime.grid, str(runtime.geo))
    read_file.assert_called_once_with(str(runtime.geo))
    runtime.grid.to_netcdf.assert_not_called()


def test_debug_peer_gathers_partition_and_receives_shared_error(runtime):
    runtime.ns["hermes_gr"].DEBUG = True
    runtime.grid.master, runtime.grid.size = False, 2
    runtime.comm.bcast.side_effect = None
    runtime.comm.bcast.return_value = "Invalid paired grid GeoJSON: shared failure"
    with pytest.raises(ValueError, match="shared failure"):
        runtime.ns["_validate_grid_geojson_debug"](runtime.grid, str(runtime.geo))
    runtime.comm.gather.assert_called_once_with(runtime.grid.shapefile, root=0)
