# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

"""Synthetic test for the mixed (prioritized) proxy conservative interpolation and the proxy list forms.

- src: 4x4 regular grid of 1 deg cells over [0,4]x[0,4] with distinct values per cell.
- dst: 8x8 regular grid of 0.5 deg cells over the same domain (dst index = [lat, lon]).
- proxies, in priority order:
  1) ("lines", path, None, "lw"): 4-tuple with its own weight column.
  2) ("points", path): 2-tuple, weight column from the global column_name="pw".
  3) ("raster_categorical", path, [5]): 3-tuple with the categories to keep.
  4) ("polygons", path): 2-tuple without the global column -> every polygon gets weight 1.
- Expected:
  * Src cell [0,1)x[0,1): line at lat 0.25 -> 50%/50% between dst (0, 0) and (0, 1). The point, the kept
    raster pixel and the polygon of that cell are IGNORED (lower priority).
  * Line crossing src cells [1,2)x[3,4) and [2,3)x[3,4): each cell sends 100% to one dst cell.
  * Src cell [3,4)x[2,3): point -> 100% to dst (4, 6); its polygon is ignored.
  * Src cell [1,2)x[1,2): kept raster pixel -> 100% to dst (3, 3).
  * Src cell [3,4)x[3,4): polygon only -> 100% to dst (7, 7).
  * All the other src cells -> plain area conservative fallback (0.25 to each of their 4 dst subcells).
- Mass (flux=False) and emissions (flux=True) are conserved, and both modes agree cell by cell.

Run: python test_mixed_proxy_synthetic.py
     mpirun -np 4 python test_mixed_proxy_synthetic.py
"""

import numpy as np
from rasterio.transform import from_origin

from nes.tests.methods.horizontal_interpolation.utils import (
    RANK,
    assert_shares,
    check_flux_and_mass,
    default_values,
    feature,
    plain_area_shares,
    rect,
    shares,
    write_geojson,
    write_raster,
)


def main():
    lines = write_geojson(
        "lines.geojson",
        [
            feature(1.0, "LineString", [[0.1, 0.25], [0.9, 0.25]], column="lw"),
            feature(2.0, "LineString", [[1.5, 3.25], [2.5, 3.25]], column="lw"),
        ],
    )
    points = write_geojson(
        "points.geojson",
        [
            feature(7.0, "Point", [0.25, 0.75], column="pw"),
            feature(5.0, "Point", [3.25, 2.25], column="pw"),
        ],
    )
    polygons = write_geojson(
        "polygons.geojson",
        [
            feature(2.0, "Polygon", rect(0.6, 0.6, 0.9, 0.9)),
            feature(2.0, "Polygon", rect(3.6, 2.6, 3.9, 2.9)),
            feature(2.0, "Polygon", rect(3.6, 3.6, 3.9, 3.9)),
        ],
    )
    # Raster row 7 is the southern row (lat 0-0.5)
    cats = np.zeros((8, 8), dtype=np.uint8)
    cats[6, 1] = 5  # dst (1, 1): src cell [0,1)x[0,1), covered by the line
    cats[4, 3] = 5  # dst (3, 3): src cell [1,2)x[1,2)
    cats[4, 2] = 9  # not kept
    raster = write_raster("cats.tif", cats, from_origin(0.0, 4.0, 0.5, 0.5))

    proxy = [
        ("lines", lines, None, "lw"),
        ("points", points),
        ("raster_categorical", raster, [5]),
        ("polygons", polygons),
    ]
    result = check_flux_and_mass(default_values(), "mixed", proxy=proxy, column_name="pw")
    if RANK != 0:
        return
    share = shares(result[0], result[2])
    assert_shares(
        share,
        [
            ((0, 0), 0.5),
            ((0, 1), 0.5),
            ((1, 0), 0.0),
            ((1, 1), 0.0),
            ((6, 3), 1.0),
            ((6, 2), 0.0),
            ((7, 2), 0.0),
            ((7, 3), 0.0),
            ((6, 4), 1.0),
            ((6, 5), 0.0),
            ((7, 4), 0.0),
            ((7, 5), 0.0),
            ((4, 6), 1.0),
            ((4, 7), 0.0),
            ((5, 6), 0.0),
            ((5, 7), 0.0),
            ((3, 3), 1.0),
            ((2, 2), 0.0),
            ((2, 3), 0.0),
            ((3, 2), 0.0),
            ((7, 7), 1.0),
            ((6, 6), 0.0),
            ((6, 7), 0.0),
            ((7, 6), 0.0),
            # No proxy: plain area fallback
            (np.s_[0:2, 2:4], plain_area_shares()[0:2, 2:4]),
        ],
        "mixed",
    )
    print("ALL CHECKS PASSED")


def test_mixed_proxy_conservation():
    main()


if __name__ == "__main__":
    main()
