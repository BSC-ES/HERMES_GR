# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

"""Synthetic test for the points-proxy conservative interpolation.

- src: 4x4 regular grid of 1 deg cells over [0,4]x[0,4] with distinct values per cell.
- dst: 8x8 regular grid of 0.5 deg cells over the same domain (dst index = [lat, lon]).
- proxy: points GeoJSON:
  * Src cell [0,1)x[0,1): weights 1 and 3 -> 25% to dst (0, 0) and 75% to dst (1, 1).
  * Src cell [3,4)x[2,3): one point (3.25, 2.25) -> 100% to dst (4, 6).
  * Src cell [1,2)x[1,2): a point with weight 0, one with a negative weight and one without weight
    (null): they carry no proxy information, so the cell falls back to plain area weights.
  * Source-edge points (2.0, 0.25), (0.25, 2.0) and corner (2.0, 3.0):
    the east/north source and destination own each point.
  * Destination-only edge points (1.5, 2.25), (3.25, 1.5): choose east/north.
  * Outer corner (4.0, 4.0): the last source and destination retain it.
  * All the other src cells -> plain area conservative fallback (0.25 to each of their 4 dst subcells).
- Mass (flux=False) and emissions (flux=True) are conserved, and both modes agree cell by cell.

Run: python -m nes.tests.methods.horizontal_interpolation.test_points_proxy
     mpirun -np 4 python -m pytest -q \
         nes/tests/methods/horizontal_interpolation/test_points_proxy.py
"""

import numpy as np
import pytest

from nes.tests.methods.horizontal_interpolation.utils import (
    RANK,
    SIZE,
    TOTAL_RTOL,
    assert_shares,
    check_flux_and_mass,
    default_values,
    feature,
    plain_area_shares,
    shares,
    write_geojson,
)


def main(parallel_method="Y"):
    path = write_geojson(
        "points.geojson",
        [
            feature(1.0, "Point", [0.25, 0.25]),
            feature(3.0, "Point", [0.75, 0.75]),
            feature(5.0, "Point", [3.25, 2.25]),
            feature(0.0, "Point", [1.25, 1.25]),
            feature(-2.0, "Point", [1.75, 1.75]),
            feature(None, "Point", [1.75, 1.25]),
            feature(1.0, "Point", [2.0, 3.0]),
            feature(1.0, "Point", [2.0, 0.25]),
            feature(1.0, "Point", [0.25, 2.0]),
            feature(1.0, "Point", [1.5, 2.25]),
            feature(1.0, "Point", [3.25, 1.5]),
            feature(1.0, "Point", [4.0, 4.0]),
        ],
    )
    result = check_flux_and_mass(
        default_values(),
        "points",
        proxy=path,
        proxy_type="points",
        parallel_method=parallel_method,
    )
    if RANK != 0:
        return
    share = shares(result[0], result[2])
    expected = plain_area_shares()
    # Clear the complete footprint of each point-owning source: any fallback
    # there is a failure, even if the global mass total is conserved.
    for row, column in [(0, 0), (2, 3), (3, 2), (0, 2), (2, 0), (2, 1), (1, 3), (3, 3)]:
        expected[2 * row:2 * row + 2, 2 * column:2 * column + 2] = 0.0
    expected[0, 0], expected[1, 1] = 0.25, 0.75
    for row, column in [(4, 6), (6, 4), (0, 4), (4, 0), (4, 3), (3, 6), (7, 7)]:
        expected[row, column] = 1.0
    assert_shares(share, [(np.s_[:, :], expected)], "points")
    print("ALL CHECKS PASSED")


@pytest.mark.parametrize("parallel_method", ["X", "Y"])
def test_points_proxy_conservation(parallel_method):
    """Conserve mass and route boundary points identically for X/Y splits."""
    main(parallel_method)


@pytest.mark.parametrize("parallel_method", ["T", "Y", "X"])
@pytest.mark.parametrize("dst", [(0.8, 5), (2.0, 2)])
def test_points_and_fallback_add_in_destination_cells(parallel_method, dst):
    """Sum source-exclusive methods in shared destinations without mass loss."""
    path = write_geojson(
        "points_with_fallback.geojson",
        [feature(1.0, "Point", [0.25, 0.25]), feature(3.0, "Point", [0.9, 0.9])],
    )
    values = np.repeat(default_values()[None, None], SIZE, axis=0)
    result = check_flux_and_mass(
        values,
        "points plus area fallback",
        dst=dst,
        proxy=path,
        proxy_type="points",
        parallel_method=parallel_method,
    )
    if RANK != 0:
        return
    mass, _, source_mass = result
    increment, count = dst
    expected = np.zeros_like(mass)
    for row in range(4):
        for column in range(4):
            if (row, column) == (0, 0):
                continue
            for dst_row in range(count):
                south = max(row, dst_row * increment)
                north = min(row + 1, (dst_row + 1) * increment)
                if north <= south:
                    continue
                latitude_share = (
                    (np.sin(np.radians(north)) - np.sin(np.radians(south)))
                    / (np.sin(np.radians(row + 1)) - np.sin(np.radians(row)))
                )
                for dst_column in range(count):
                    west = max(column, dst_column * increment)
                    east = min(column + 1, (dst_column + 1) * increment)
                    if east > west:
                        expected[:, :, dst_row, dst_column] += (
                            source_mass[:, :, row, column] * latitude_share * (east - west)
                        )
    for coordinate, share in ((0.25, 0.25), (0.9, 0.75)):
        index = int(coordinate / increment)
        expected[:, :, index, index] += source_mass[:, :, 0, 0] * share
    np.testing.assert_allclose(mass, expected, rtol=TOTAL_RTOL, atol=0.0)


if __name__ == "__main__":
    main()
