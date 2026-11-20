# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

"""Synthetic flux / no-flux conservation matrix of the proxy conservative interpolation at high latitude.

- src: 4x4 regular grid of 1 deg cells over lon [0,4] x lat [50,54] (the cell areas differ with latitude),
  field v = (1 + (lat - 50) + 0.5 * lon) * (1 + time step), one time step per process.
- dst: 8x8 regular grid of 0.5 deg cells over the same domain.
- For every proxy configuration (each proxy type, boundary geometries, a projected raster, a resampled numerical
  raster, a priority list with every tuple form and no proxy) check_flux_and_mass() checks, for every time step,
  that the no-flux mass and the flux emissions both add up to sum(v * A_src) and agree cell by cell.
- Hand-computed checks: the 1:3 points split, the raster nodata pixel and the all-zero raster src cell.

Run: python test_flux_proxy_synthetic.py [Y|T]
     mpirun -np 4 python test_flux_proxy_synthetic.py [Y|T]
"""

import sys

import numpy as np
from pyproj import Transformer
from rasterio.transform import from_origin

from nes.tests.methods.horizontal_interpolation.utils import (
    RANK,
    SIZE,
    assert_shares,
    cell_areas,
    check_flux_and_mass,
    feature,
    plain_area_shares,
    rect,
    shares,
    write_geojson,
    write_raster,
)

PARALLEL_METHOD = sys.argv[1] if __name__ == "__main__" and len(sys.argv) > 1 else "Y"
LAT0 = 50.0


def write_proxies():
    paths = {
        "points": write_geojson(
            "points.geojson",
            [
                # Src cell lon [0,1) x lat [50,51): weights 1:3 -> 25% to dst (0, 0) and 75% to dst (1, 1)
                feature(1.0, "Point", [0.25, LAT0 + 0.25]),
                feature(3.0, "Point", [0.75, LAT0 + 0.75]),
                feature(2.0, "Point", [3.25, LAT0 + 2.25]),
            ],
        ),
        "lines": write_geojson(
            "lines.geojson",
            [
                feature(1.0, "LineString", [[0.1, LAT0 + 0.25], [0.9, LAT0 + 0.25]]),
                # Crossing src cells lon [1,2) and [2,3) at lat 53.25
                feature(2.0, "LineString", [[1.6, LAT0 + 3.25], [2.4, LAT0 + 3.25]]),
                # Diagonal line crossing 3 dst cells of src cell lon [3,4) x lat [51,52)
                feature(1.0, "LineString", [[3.1, LAT0 + 1.1], [3.9, LAT0 + 1.6]]),
            ],
        ),
        "boundary": write_geojson(
            "boundary.geojson",
            [
                # Point on the corner shared by 4 src cells, point on a dst edge, line on a src edge
                feature(1.0, "Point", [2.0, LAT0 + 2.0]),
                feature(1.0, "Point", [0.5, LAT0 + 3.25]),
                feature(1.0, "LineString", [[1.0, LAT0 + 0.2], [1.0, LAT0 + 0.8]]),
                # Line along the outer domain edge
                feature(1.0, "LineString", [[3.2, LAT0 + 4.0], [3.8, LAT0 + 4.0]]),
            ],
        ),
        "polygons": write_geojson(
            "polygons.geojson",
            [
                feature(1.0, "Polygon", rect(0.1, LAT0 + 0.1, 0.4, LAT0 + 0.9)),
                feature(3.0, "Polygon", rect(3.1, LAT0 + 0.1, 3.4, LAT0 + 0.4)),
                feature(1.0, "Polygon", rect(3.6, LAT0 + 0.6, 3.9, LAT0 + 0.9)),
                # Polygon crossing src cells and partly outside the domain
                feature(5.0, "Polygon", rect(1.6, LAT0 + 3.1, 2.4, LAT0 + 4.4)),
            ],
        ),
        "multi": write_geojson(
            "multi.geojson",
            [
                # All three geometry types share the normalization pool of src cell lon [1,2) x lat [51,52)
                feature(1.0, "Point", [1.25, LAT0 + 1.25]),
                feature(1.0, "LineString", [[1.55, LAT0 + 1.75], [1.95, LAT0 + 1.75]]),
                feature(2.0, "Polygon", rect(1.6, LAT0 + 1.1, 1.9, LAT0 + 1.4)),
                feature(4.0, "Point", [2.75, LAT0 + 0.25]),
            ],
        ),
        "polygons_column": write_geojson(
            "polygons_column.geojson",
            [
                feature(2.0, "Polygon", rect(2.1, LAT0 + 2.1, 2.4, LAT0 + 2.4), column="pw"),
            ],
        ),
    }

    # 8x8 pixels of 0.5 deg over the domain; raster row 7 is the bottom (lat 50-50.5)
    transform = from_origin(0.0, LAT0 + 4.0, 0.5, 0.5)
    rng = np.random.default_rng(42)
    raster = rng.uniform(1.0, 10.0, size=(8, 8)).astype(np.float32)
    raster[0:2, 0:2] = 0.0  # zero valued pixels carry no weight: that source cell falls back to plain area
    raster[2, 2] = -9999.0  # nodata pixel: excluded, its source cell is split among its other pixels
    cats = np.zeros((8, 8), dtype=np.uint8)
    cats[7, 0:2] = 5
    cats[4:6, 4] = 7
    nums = rng.uniform(1.0, 5.0, size=(8, 8)).astype(np.float32)
    paths["raster"] = write_raster("raster.tif", raster, transform, nodata=-9999.0)
    paths["raster_int16"] = write_raster("raster_int16.tif", np.rint(raster).astype(np.int16), transform, nodata=-9999)
    paths["uniform"] = write_raster("uniform.tif", np.full((8, 8), 2.5, dtype=np.float32), transform)
    paths["cats"] = write_raster("cats.tif", cats, transform)
    paths["nums"] = write_raster("nums.tif", nums, transform, nodata=-9999.0)
    # Web Mercator raster of 20 km pixels covering the domain (not aligned with the lat-lon grids)
    west, north = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform(-0.2, LAT0 + 4.2)
    paths["mercator"] = write_raster(
        "mercator.tif",
        rng.uniform(1.0, 10.0, size=(40, 25)).astype(np.float32),
        from_origin(west, north, 20000.0, 20000.0),
        crs="EPSG:3857",
    )
    return paths


def configs(paths):
    return {
        "points": {"proxy": paths["points"], "proxy_type": "points"},
        "lines": {"proxy": paths["lines"], "proxy_type": "lines"},
        "boundary multigeometry": {"proxy": paths["boundary"], "proxy_type": "multigeometry"},
        "polygons": {"proxy": paths["polygons"], "proxy_type": "polygons"},
        "multigeometry": {"proxy": paths["multi"], "proxy_type": "multigeometry"},
        "raster": {"proxy": paths["raster"], "proxy_type": "raster"},
        "raster_int16": {"proxy": paths["raster_int16"], "proxy_type": "raster"},
        "raster_uniform": {"proxy": paths["uniform"], "proxy_type": "raster"},
        "raster_mercator": {"proxy": paths["mercator"], "proxy_type": "raster"},
        "raster_categorical": {"proxy": paths["cats"], "proxy_type": "raster_categorical", "proxy_categories": [5, 7]},
        "raster_categorical_numerical": {
            "proxy": paths["cats"],
            "proxy_type": "raster_categorical_numerical",
            "proxy_categories": [5, 7],
            "proxy_numerical": paths["nums"],
        },
        "mixed": {
            "proxy": [
                ("lines", paths["lines"]),
                ("raster_categorical", paths["cats"], [7]),
                ("polygons", paths["polygons_column"], None, "pw"),
                ("raster_categorical_numerical", (paths["cats"], paths["nums"]), [5]),
                ("points", paths["points"]),
                ("polygons", paths["polygons"]),
            ]
        },
        "no_proxy": {"proxy": None},
    }


def source_values():
    """(time, lev, lat, lon) source field, one time step per process (the T method needs n_times >= SIZE)."""
    lat = LAT0 + 0.5 + np.arange(4)
    lon = 0.5 + np.arange(4)
    lon2d, lat2d = np.meshgrid(lon, lat)
    base = 1.0 + (lat2d - LAT0) + 0.5 * lon2d
    return np.stack([base * (1.0 + t) for t in range(SIZE)])[:, None]


def main():
    paths = write_proxies()
    values = source_values()
    a_dst = cell_areas(0.5, 8, lat0=LAT0) if RANK == 0 else None
    if RANK == 0:
        # High latitude: the destination subcells of a source cell have different areas
        assert not np.isclose(a_dst[0, 0], a_dst[1, 0], rtol=1e-6)
    for name, cfg in configs(paths).items():
        result = check_flux_and_mass(values, name, lat0=LAT0, parallel_method=PARALLEL_METHOD, **cfg)
        if RANK != 0:
            continue
        mass_out, flux_out, src_mass = result
        share = shares(mass_out, src_mass)
        if name == "points":
            assert_shares(share, [((0, 0), 0.25), ((1, 1), 0.75), ((0, 1), 0.0), ((1, 0), 0.0)], name)
        if name in ("raster", "raster_int16"):
            # The nodata pixel (dst (5, 2)) gets nothing, the rest of its source cell gets all the mass; the all
            # zero source cell (lon [0,1) x lat [53,54)) falls back to plain area weights
            assert_shares(share, [((5, 2), 0.0), (np.s_[6:8, 0:2], plain_area_shares(lat0=LAT0)[6:8, 0:2])], name)
            assert np.isclose(share[4:6, 2:4].sum(), 1.0, rtol=1e-10)
        if name == "raster_uniform":
            assert_shares(share, [(np.s_[:, :], 0.25)], name)
        if name == "no_proxy":
            assert_shares(share, [(np.s_[:, :], plain_area_shares(lat0=LAT0))], name)
            parent = np.repeat(np.repeat(values[0, 0], 2, axis=0), 2, axis=1)
            np.testing.assert_allclose(flux_out[0, 0], parent, rtol=1e-10)
    if RANK == 0:
        print(f"ALL CHECKS PASSED (parallel_method={PARALLEL_METHOD}, {SIZE} processes)")


def test_flux_and_mass_conservation():
    main()


if __name__ == "__main__":
    main()
