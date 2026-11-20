# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

"""Synthetic test for the lines-proxy conservative interpolation.

- src: 4x4 regular grid of 1 deg cells over [0,4]x[0,4] with distinct values per cell.
- dst: 8x8 regular grid of 0.5 deg cells over the same domain (dst index = [lat, lon]).
- proxy: lines GeoJSON:
  Line lengths are geodesic lengths in meters (WGS84), so expected shares are computed with pyproj.Geod.
  * Src cell [0,1)x[0,1): line A (weight 1) at lat 0.25 from lon 0.1 to 0.7 (0.4 deg in dst (0, 0) and 0.2 in
    dst (0, 1)) and line B (weight 3) at lat 0.75 from lon 0.6 to 0.9 (dst (1, 1)). Pool: A + 3 * B.
  * Src cell [2,3)x[0,1): MultiLineString with parts of 0.2 deg at lat 0.25 (dst (0, 4)) and 0.3 deg at lat 0.75
    (dst (1, 5)) -> ~40%/60%, proportional to their geodesic lengths.
  * Line crossing src cells [1,2)x[3,4) and [2,3)x[3,4) at lat 3.25: each cell sends 100% to one dst cell.
  * Src cell [3,4)x[0,1): line lying on the edge between dst rows 0 and 1 (lat 0.5, lon 3.1 to 3.4): it is
    split evenly between dst (0, 6) and (1, 6), not counted twice.
  * Line on the shared edge of src cells [0,1)x[2,3) and [1,2)x[2,3) (lon 1.0, lat 2.1 to 2.9): both cells
    are covered by it and each sends its mass to dst (4, 1), (5, 1), (4, 2) and (5, 2), split by row according
    to the geodesic length of each half (~25% each) and evenly between columns.
  * Src cell [2,3)x[1,2): only a zero-weight line -> plain area fallback.
  * All the other src cells -> plain area conservative fallback (0.25 to each of their 4 dst subcells).
- Mass (flux=False) and emissions (flux=True) are conserved, and both modes agree cell by cell.

High-latitude check (test_lines_proxy_geodesic_high_latitude): same grids shifted to lat 60. Degree lengths
would split the mass of a cell evenly between a zonal and a meridional line of 0.3 deg, while geodesic lengths give
the zonal line about half the length (cos 60). A diagonal line crossing a dst corner is split among dst cells by
geodesic length, and the corner-touching dst cells get nothing.

Run: python test_lines_proxy_synthetic.py
     mpirun -np 4 python test_lines_proxy_synthetic.py
"""

import numpy as np
from pyproj import Geod

from nes.tests.methods.horizontal_interpolation.utils import (
    RANK,
    SHARE_RTOL,
    assert_shares,
    check_flux_and_mass,
    default_values,
    feature,
    plain_area_shares,
    shares,
    write_geojson,
)

GEOD = Geod(ellps="WGS84")


def geo_len(*coords):
    """Geodesic length (m) of the polyline through the given (lon, lat) vertices."""
    lons, lats = zip(*coords)
    return GEOD.line_length(lons, lats)


def main():
    path = write_geojson(
        "lines.geojson",
        [
            feature(1.0, "LineString", [[0.1, 0.25], [0.7, 0.25]]),
            feature(3.0, "LineString", [[0.6, 0.75], [0.9, 0.75]]),
            feature(1.0, "MultiLineString", [[[2.1, 0.25], [2.3, 0.25]], [[2.6, 0.75], [2.9, 0.75]]]),
            feature(2.0, "LineString", [[1.6, 3.25], [2.4, 3.25]]),
            feature(1.0, "LineString", [[3.1, 0.5], [3.4, 0.5]]),
            feature(1.0, "LineString", [[1.0, 2.1], [1.0, 2.9]]),
            feature(0.0, "LineString", [[2.1, 1.5], [2.9, 1.5]]),
        ],
    )
    result = check_flux_and_mass(default_values(), "lines", proxy=path, proxy_type="lines")
    if RANK != 0:
        return
    mass_out, src_mass = result[0][0, 0], result[2][0, 0]
    share = shares(result[0], result[2])
    a_00, a_01 = geo_len((0.1, 0.25), (0.5, 0.25)), geo_len((0.5, 0.25), (0.7, 0.25))
    b_11 = 3.0 * geo_len((0.6, 0.75), (0.9, 0.75))
    pool = a_00 + a_01 + b_11
    m_04, m_15 = geo_len((2.1, 0.25), (2.3, 0.25)), geo_len((2.6, 0.75), (2.9, 0.75))
    assert_shares(
        share,
        [
            ((0, 0), a_00 / pool),
            ((0, 1), a_01 / pool),
            ((1, 1), b_11 / pool),
            ((1, 0), 0.0),
            ((0, 4), m_04 / (m_04 + m_15)),
            ((1, 5), m_15 / (m_04 + m_15)),
            ((0, 5), 0.0),
            ((1, 4), 0.0),
            ((6, 3), 1.0),
            ((6, 2), 0.0),
            ((7, 2), 0.0),
            ((7, 3), 0.0),
            ((6, 4), 1.0),
            ((6, 5), 0.0),
            ((7, 4), 0.0),
            ((7, 5), 0.0),
            ((0, 6), 0.5),
            ((1, 6), 0.5),
            ((0, 7), 0.0),
            ((1, 7), 0.0),
            ((4, 0), 0.0),
            ((5, 0), 0.0),
            ((4, 3), 0.0),
            ((5, 3), 0.0),
            # Zero-weight line: plain area fallback
            (np.s_[2:4, 4:6], plain_area_shares()[2:4, 4:6]),
            # No proxy: plain area fallback
            (np.s_[0:2, 2:4], plain_area_shares()[0:2, 2:4]),
        ],
        "lines",
    )
    # Line on the shared src edge: both src cells send their mass to dst rows 4 and 5 proportionally to the
    # geodesic length of each half (meridian degrees lengthen with latitude), split evenly between columns 1 and 2
    low, high = geo_len((1.0, 2.1), (1.0, 2.5)), geo_len((1.0, 2.5), (1.0, 2.9))
    edge_mass = src_mass[2, 0] + src_mass[2, 1]
    shared_edge = 0.5 * edge_mass * np.array([[low], [high]]) / (low + high) * np.ones((2, 2))
    assert np.allclose(mass_out[4:6, 1:3], shared_edge, rtol=SHARE_RTOL), (
        f"[lines] shared src edge:\n{mass_out[4:6, 1:3]} != {shared_edge}"
    )
    print("ALL CHECKS PASSED")


def high_latitude():
    lat0 = 60.0
    path = write_geojson(
        "lines_high_lat.geojson",
        [
            # Src cell (0, 0): zonal and meridional lines of the same length in degrees
            feature(1.0, "LineString", [[0.1, 60.25], [0.4, 60.25]]),
            feature(1.0, "LineString", [[0.75, 60.6], [0.75, 60.9]]),
            # Src cell (0, 2): diagonal line through the dst corner (2.5, 60.5)
            feature(1.0, "LineString", [[2.1, 60.1], [2.9, 60.9]]),
        ],
    )
    result = check_flux_and_mass(default_values(), "lines high latitude", lat0=lat0, proxy=path, proxy_type="lines")
    if RANK != 0:
        return
    share = shares(result[0], result[2])
    zonal = geo_len((0.1, 60.25), (0.4, 60.25))
    meridional = geo_len((0.75, 60.6), (0.75, 60.9))
    assert zonal / (zonal + meridional) < 0.4, "zonal line should weigh about cos(60) of the meridional one"
    low, high = geo_len((2.1, 60.1), (2.5, 60.5)), geo_len((2.5, 60.5), (2.9, 60.9))
    assert_shares(
        share,
        [
            ((0, 0), zonal / (zonal + meridional)),
            ((1, 1), meridional / (zonal + meridional)),
            ((0, 1), 0.0),
            ((1, 0), 0.0),
            ((0, 4), low / (low + high)),
            ((1, 5), high / (low + high)),
            ((0, 5), 0.0),
            ((1, 4), 0.0),
            (np.s_[0:2, 2:4], plain_area_shares(lat0=lat0)[0:2, 2:4]),
        ],
        "lines high latitude",
    )
    print("ALL HIGH LATITUDE CHECKS PASSED")


def test_lines_proxy_conservation():
    main()


def test_lines_proxy_geodesic_high_latitude():
    high_latitude()


if __name__ == "__main__":
    main()
    high_latitude()
