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

# High latitude: the true areas of the destination subcells of a source cell differ
LAT0 = 60.0


def make_grid(inc, n, lat0=LAT0):
    grid = nes.create_nes(comm=None, info=False, projection="regular", lat_orig=lat0, lon_orig=0.0,
                          inc_lat=inc, inc_lon=inc, n_lat=n, n_lon=n)
    grid.create_shapefile()
    return grid


def cell_areas(grid):
    """Analytic m2 areas of regular cells bounded by meridians and parallels."""
    bounds = grid.shapefile.geometry.bounds
    areas = (
        6371000.0**2
        * np.radians(bounds["maxx"] - bounds["minx"])
        * (np.sin(np.radians(bounds["maxy"])) - np.sin(np.radians(bounds["miny"])))
    )
    return areas.to_numpy().reshape(len(grid.lat["data"]), len(grid.lon["data"]))


def flux_field(grid):
    lon, lat = np.meshgrid(grid.lon["data"], grid.lat["data"])
    return 1.0 + (lat - LAT0) + 0.5 * lon


def interpolate(data, flux, **kwargs):
    src = make_grid(1.0, 2)
    dst = make_grid(0.5, 4)
    src.variables["var"] = {"data": data[None, None], "units": "-"}
    out = src.interpolate_horizontal(dst_grid=dst, kind="Conservative", flux=flux, **kwargs)
    return out.variables["var"]["data"][0, 0]


def write_geojson(path, features):
    with open(path, "w") as f:
        json.dump({"type": "FeatureCollection",
                   "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
                   "features": [{"type": "Feature", "properties": {"weight": weight},
                                 "geometry": {"type": geom_type, "coordinates": coords}}
                                for weight, geom_type, coords in features]}, f)
    return str(path)


@pytest.fixture(scope="module")
def areas():
    return cell_areas(make_grid(1.0, 2)), cell_areas(make_grid(0.5, 4))


@pytest.fixture(scope="module")
def proxies(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("horizontal_interp_flux")
    return {
        "points": dict(proxy=write_geojson(tmp_dir / "points.geojson", [
            (1.0, "Point", [0.25, LAT0 + 0.25]), (3.0, "Point", [0.75, LAT0 + 0.75])]), proxy_type="points"),
        "lines": dict(proxy=write_geojson(tmp_dir / "lines.geojson", [
            (1.0, "LineString", [[0.1, LAT0 + 0.25], [0.9, LAT0 + 0.25]]),
            (2.0, "LineString", [[1.1, LAT0 + 1.1], [1.9, LAT0 + 1.6]])]), proxy_type="lines"),
        "polygons": dict(proxy=write_geojson(tmp_dir / "polygons.geojson", [
            (1.0, "Polygon", [[[0.1, LAT0 + 0.1], [0.4, LAT0 + 0.1], [0.4, LAT0 + 0.9], [0.1, LAT0 + 0.9],
                               [0.1, LAT0 + 0.1]]]),
            (3.0, "Polygon", [[[1.6, LAT0 + 0.6], [1.9, LAT0 + 0.6], [1.9, LAT0 + 0.9], [1.6, LAT0 + 0.9],
                               [1.6, LAT0 + 0.6]]])]), proxy_type="polygons"),
        "no_proxy": dict(proxy=None),
    }


class TestHorizontalInterpFlux:

    @pytest.mark.parametrize("lat0", [40.0, 65.0])
    @pytest.mark.parametrize("flux", [False, True])
    def test_plain_area_nested_cells(self, lat0, flux):
        """Nested cells preserve source flux or receive their physical mass share."""
        src = make_grid(1.0, 2, lat0=lat0)
        dst = make_grid(0.5, 4, lat0=lat0)
        values = np.arange(1.0, 5.0).reshape(2, 2)
        src.variables["var"] = {"data": values[None, None], "units": "-"}
        out = src.interpolate_horizontal(dst_grid=dst, kind="Conservative", flux=flux)
        result = out.variables["var"]["data"][0, 0]
        parent = np.repeat(np.repeat(values, 2, axis=0), 2, axis=1)
        src_area = cell_areas(src)
        dst_area = cell_areas(dst)
        if flux:
            np.testing.assert_allclose(result, parent, rtol=1e-12)
            np.testing.assert_allclose(np.sum(result * dst_area), np.sum(values * src_area), rtol=1e-12)
        else:
            parent_area = np.repeat(np.repeat(src_area, 2, axis=0), 2, axis=1)
            np.testing.assert_allclose(result, parent * dst_area / parent_area, rtol=1e-12)
            np.testing.assert_allclose(np.sum(result), np.sum(values), rtol=1e-12)

    @pytest.mark.parametrize("flux", [False, True])
    def test_plain_area_0_360_longitudes(self, flux):
        """Cells of 0-360 grids touching 180 must not wrap: nested flux and total mass are preserved."""
        def global_grid(inc, n_lat):
            grid = nes.create_nes(comm=None, info=False, projection="regular", lat_orig=40.0, lon_orig=0.0,
                                  inc_lat=inc, inc_lon=inc, n_lat=n_lat, n_lon=round(360 / inc))
            grid.create_shapefile()
            return grid

        src = global_grid(1.0, 2)
        dst = global_grid(0.5, 4)
        assert src.shapefile.geometry.bounds["maxx"].max() > 180.0
        values = np.arange(1.0, 1.0 + 2 * 360).reshape(2, 360)
        src.variables["var"] = {"data": values[None, None], "units": "-"}
        out = src.interpolate_horizontal(dst_grid=dst, kind="Conservative", flux=flux)
        result = out.variables["var"]["data"][0, 0]
        parent = np.repeat(np.repeat(values, 2, axis=0), 2, axis=1)
        if flux:
            np.testing.assert_allclose(result, parent, rtol=1e-12)
            np.testing.assert_allclose(np.sum(result * cell_areas(dst)), np.sum(values * cell_areas(src)),
                                       rtol=1e-12)
        else:
            src_area = cell_areas(src)
            parent_area = np.repeat(np.repeat(src_area, 2, axis=0), 2, axis=1)
            np.testing.assert_allclose(result, parent * cell_areas(dst) / parent_area, rtol=1e-12)
            np.testing.assert_allclose(np.sum(result), np.sum(values), rtol=1e-12)

    @pytest.mark.parametrize("name", ["no_proxy", "points", "lines", "polygons"])
    def test_flux_consistent_with_mass(self, name, areas, proxies):
        """f_dst * A_dst equals the interpolation of the source mass, and the total mass is conserved."""
        a_src, a_dst = areas
        src_grid = make_grid(1.0, 2)
        f_src = flux_field(src_grid)
        f_dst = interpolate(f_src, flux=True, **proxies[name])
        m_dst = interpolate(f_src * a_src, flux=False, **proxies[name])
        np.testing.assert_allclose(f_dst * a_dst, m_dst, rtol=1e-10)
        np.testing.assert_allclose(np.sum(f_dst * a_dst), np.sum(f_src * a_src), rtol=1e-10)

    def test_points_flux_hand_computed(self, areas, proxies):
        """The points keep 1/4 and 3/4 of the source mass in their destination cells."""
        a_src, a_dst = areas
        f_dst = interpolate(np.full((2, 2), 2.0), flux=True, **proxies["points"])
        src_mass = 2.0 * a_src[0, 0]
        np.testing.assert_allclose(f_dst[0, 0], 0.25 * src_mass / a_dst[0, 0], rtol=1e-10)
        np.testing.assert_allclose(f_dst[1, 1], 0.75 * src_mass / a_dst[1, 1], rtol=1e-10)
        assert f_dst[0, 1] == 0.0 and f_dst[1, 0] == 0.0
