# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

"""Synthetic test for the polygons-proxy conservative interpolation.

- src: 4x4 regular grid of 1 deg cells over [0,4]x[0,4] with distinct values per cell.
- dst: 8x8 regular grid of 0.5 deg cells over the same domain (dst index = [lat, lon]).
- proxy: polygons GeoJSON:
  * Src cell [0,1)x[0,1): one rectangle with equal areas in dst (0, 0) and (1, 0) -> 50%/50%.
  * Src cell [3,4)x[0,1): two equal-area rectangles with weights 3 and 1 -> 75%/25%.
  * One rectangle crossing src cells [1,2)x[3,4) and [2,3)x[3,4): each cell sends 100% of its
    mass to the single dst cell holding its part.
  * Src cell [2,3)x[0,1): a MultiPolygon with two equal-area parts -> 50%/50%.
  * Src cell [1,2)x[1,2): a rectangle touching src cell [0,1)x[1,2) only along their shared edge -> 100% to
    dst (2, 2); the touched cell is not covered (zero overlap area) and falls back to plain area weights.
  * Src cell [2,3)x[1,2): only a zero-weight polygon -> plain area fallback.
  * All the other src cells -> plain area conservative fallback (0.25 to each of their 4 dst subcells).
- Mass (flux=False) and emissions (flux=True) are conserved, and both modes agree cell by cell.

Run: python test_polygons_proxy_synthetic.py
     mpirun -np 4 python test_polygons_proxy_synthetic.py
"""

import numpy as np

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
)


def main():
    path = write_geojson(
        "polygons.geojson",
        [
            feature(1.0, "Polygon", rect(0.1, 0.1, 0.4, 0.9)),
            feature(3.0, "Polygon", rect(3.1, 0.1, 3.4, 0.4)),
            feature(1.0, "Polygon", rect(3.6, 0.6, 3.9, 0.9)),
            feature(5.0, "Polygon", rect(1.6, 3.1, 2.4, 3.4)),
            feature(2.0, "MultiPolygon", [rect(2.1, 0.1, 2.35, 0.35), rect(2.6, 0.6, 2.85, 0.85)]),
            feature(1.0, "Polygon", rect(1.0, 1.1, 1.3, 1.4)),
            feature(0.0, "Polygon", rect(2.1, 1.1, 2.9, 1.9)),
        ],
    )
    result = check_flux_and_mass(default_values(), "polygons", proxy=path, proxy_type="polygons")
    if RANK != 0:
        return
    share = shares(result[0], result[2])
    assert_shares(
        share,
        [
            ((0, 0), 0.5),
            ((1, 0), 0.5),
            ((0, 1), 0.0),
            ((1, 1), 0.0),
            ((0, 6), 0.75),
            ((1, 7), 0.25),
            ((0, 7), 0.0),
            ((1, 6), 0.0),
            ((6, 3), 1.0),
            ((6, 2), 0.0),
            ((7, 2), 0.0),
            ((7, 3), 0.0),
            ((6, 4), 1.0),
            ((6, 5), 0.0),
            ((7, 4), 0.0),
            ((7, 5), 0.0),
            ((0, 4), 0.5),
            ((1, 5), 0.5),
            ((0, 5), 0.0),
            ((1, 4), 0.0),
            ((2, 2), 1.0),
            ((2, 3), 0.0),
            ((3, 2), 0.0),
            ((3, 3), 0.0),
            # Edge-touched cell and zero-weight polygon: plain area fallback
            (np.s_[2:4, 0:2], plain_area_shares()[2:4, 0:2]),
            (np.s_[2:4, 4:6], plain_area_shares()[2:4, 4:6]),
            # No proxy: plain area fallback
            (np.s_[0:2, 2:4], plain_area_shares()[0:2, 2:4]),
        ],
        "polygons",
    )
    print("ALL CHECKS PASSED")


def test_polygons_proxy_conservation():
    main()


if __name__ == "__main__":
    main()
