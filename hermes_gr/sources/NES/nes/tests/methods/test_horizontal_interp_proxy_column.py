# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import json

import numpy as np
import pytest

import nes

# pyright: reportAttributeAccessIssue=false


def make_grid(inc, n):
    grid = nes.create_nes(comm=None, info=False, projection="regular", lat_orig=0.0, lon_orig=0.0,
                          inc_lat=inc, inc_lon=inc, n_lat=n, n_lon=n)
    grid.create_shapefile()
    return grid


def interpolate(**kwargs):
    src = make_grid(1.0, 2)
    dst = make_grid(0.5, 4)
    src.variables["var"] = {"data": np.ones((1, 1, 2, 2)), "units": "-"}
    out = src.interpolate_horizontal(dst_grid=dst, kind="Conservative", **kwargs)
    return out.variables["var"]["data"][0, 0]


@pytest.fixture
def points_path(tmp_path):
    # Two points in the source cell [0,1)x[0,1): "emis" weights 1:3, "weight" weights 3:1
    features = [{"type": "Feature", "properties": {"emis": emis, "weight": weight},
                 "geometry": {"type": "Point", "coordinates": coords}}
                for emis, weight, coords in ((1.0, 3.0, [0.25, 0.25]), (3.0, 1.0, [0.75, 0.75]))]
    path = tmp_path / "points.geojson"
    with open(path, "w") as f:
        json.dump({"type": "FeatureCollection",
                   "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
                   "features": features}, f)
    return str(path)


class TestProxyColumnName:

    def test_default_weight_column(self, points_path):
        data = interpolate(proxy=points_path, proxy_type="points")
        np.testing.assert_allclose([data[0, 0], data[1, 1]], [0.75, 0.25])

    def test_custom_column(self, points_path):
        data = interpolate(proxy=points_path, proxy_type="points", column_name="emis")
        np.testing.assert_allclose([data[0, 0], data[1, 1]], [0.25, 0.75])
        np.testing.assert_allclose(np.sum(data), 4.0)

    def test_existing_two_element_proxy_list_uses_global_column(self, points_path):
        data = interpolate(proxy=[("points", points_path)], column_name="emis")
        np.testing.assert_allclose([data[0, 0], data[1, 1]], [0.25, 0.75])

    def test_missing_column_defaults_to_one(self, points_path):
        with pytest.warns(UserWarning, match="not found"):
            data = interpolate(proxy=points_path, proxy_type="points", column_name="missing")
        np.testing.assert_allclose([data[0, 0], data[1, 1]], [0.5, 0.5])

    def test_mixed_proxies_use_individual_weight_columns(self, points_path, tmp_path):
        second = tmp_path / "second.geojson"
        features = [{"type": "Feature", "properties": {"emis": emis, "weight": weight},
                     "geometry": {"type": "Point", "coordinates": coords}}
                    for emis, weight, coords in ((3.0, 1.0, [1.25, 0.25]),
                                                 (1.0, 3.0, [1.75, 0.75]))]
        with open(second, "w") as f:
            json.dump({"type": "FeatureCollection", "features": features}, f)
        data = interpolate(proxy=[("points", points_path, None, "emis"),
                                  ("points", str(second), None, "weight")])
        np.testing.assert_allclose([data[0, 0], data[1, 1]], [0.25, 0.75])
        np.testing.assert_allclose([data[0, 2], data[1, 3]], [0.25, 0.75])
        np.testing.assert_allclose(np.sum(data), 4.0)
