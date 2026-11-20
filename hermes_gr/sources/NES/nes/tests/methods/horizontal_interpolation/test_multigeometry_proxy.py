# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

"""Synthetic test for the multigeometry-proxy conservative interpolation.

- src: 4x4 regular grid of 1 deg cells over [0,4]x[0,4] with distinct values per cell.
- dst: 8x8 regular grid of 0.5 deg cells over the same domain (dst index = [lat, lon]).
- proxy: one GeoJSON combining points, lines and polygons (no priority between geometry types):
  * Src cell [0,1)x[0,1): a point (weight 4) and a line (weight 5 x length 0.8 deg = 4) share the
    pool 50%/50%; the line then splits its half by length between dst (1, 0) and (1, 1).
  * Src cell [2,3)x[1,2): a MultiPoint feature (weight 3) exploded into two points -> 50%/50%.
  * Src cell [1,2)x[2,3): a point (weight 1) and a polygon (weight 10 x area 0.1 deg2 = 1) -> 50%/50%.
  * All the other src cells -> plain area conservative fallback (0.25 to each of their 4 dst subcells).
- Mass (flux=False) and emissions (flux=True) are conserved, and both modes agree cell by cell.

Run: python test_multigeometry_proxy_synthetic.py
     mpirun -np 4 python test_multigeometry_proxy_synthetic.py
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
        "multigeometry.geojson",
        [
            feature(4.0, "Point", [0.25, 0.25]),
            feature(5.0, "LineString", [[0.1, 0.75], [0.9, 0.75]]),
            feature(3.0, "MultiPoint", [[2.25, 1.25], [2.75, 1.75]]),
            feature(1.0, "Point", [1.25, 2.25]),
            feature(10.0, "Polygon", rect(1.6, 2.55, 1.85, 2.95)),
        ],
    )
    result = check_flux_and_mass(default_values(), "multigeometry", proxy=path, proxy_type="multigeometry")
    if RANK != 0:
        return
    share = shares(result[0], result[2])
    assert_shares(
        share,
        [
            ((0, 0), 0.5),
            ((1, 0), 0.25),
            ((1, 1), 0.25),
            ((0, 1), 0.0),
            ((2, 4), 0.5),
            ((3, 5), 0.5),
            ((2, 5), 0.0),
            ((3, 4), 0.0),
            ((4, 2), 0.5),
            ((5, 3), 0.5),
            ((4, 3), 0.0),
            ((5, 2), 0.0),
            # No proxy: plain area fallback
            (np.s_[0:2, 2:4], plain_area_shares()[0:2, 2:4]),
        ],
        "multigeometry",
    )
    print("ALL CHECKS PASSED")


def test_multigeometry_proxy_conservation():
    main()


if __name__ == "__main__":
    main()
