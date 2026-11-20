# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

"""Check GeoPandas API selection without requiring native geospatial dependencies."""

import ast
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture
def union_geometries():
    path = Path(__file__).resolve().parents[1] / "modules/rescaling/rescaling.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    definition = next(node for node in tree.body if getattr(node, "name", None) == "_union_geometries")
    namespace = {}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_union_geometries"]


def test_current_api_does_not_access_deprecated_property(union_geometries):
    result = object()

    class CurrentSeries:
        union_all = Mock(return_value=result)

        @property
        def unary_union(self):
            raise AssertionError("Deprecated property accessed")

    series = CurrentSeries()
    assert union_geometries(series) is result
    series.union_all.assert_called_once_with()


def test_legacy_api_still_supported(union_geometries):
    result = object()

    class LegacySeries:
        @property
        def unary_union(self):
            return result

    assert union_geometries(LegacySeries()) is result


def test_real_geometry_union_preserves_domain(union_geometries):
    gpd = pytest.importorskip("geopandas")
    geometry = pytest.importorskip("shapely.geometry")
    # Overlapping cells and a detached part exercise the original unary algorithm.
    cells = gpd.GeoSeries([geometry.box(0, 0, 2, 1), geometry.box(1, 0, 3, 1), geometry.box(5, 0, 6, 1)])
    expected = geometry.MultiPolygon([geometry.box(0, 0, 3, 1), geometry.box(5, 0, 6, 1)])
    assert union_geometries(cells).equals(expected)
