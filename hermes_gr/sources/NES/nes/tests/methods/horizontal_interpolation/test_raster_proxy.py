# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

"""Synthetic test for the (numerical) raster-proxy conservative interpolation.

- src: 4x4 regular grid of 1 deg cells over [0,4]x[0,4] with distinct values per cell.
- dst: 8x8 regular grid of 0.5 deg cells over the same domain (dst index = [lat, lon]).
- Aligned proxy: 8x8 float32 raster of 0.5 deg pixels (one pixel per dst cell, nodata -9999):
  * Src cell [0,1)x[0,1): pixels 1 (dst (0, 0)) and 3 (dst (1, 1)), the others 0 -> 25%/75%.
  * Src cell [1,2)x[0,1): one nodata pixel (dst (0, 2)) and three pixels of 2 -> 0 and 1/3 each.
  * Src cell [2,3)x[0,1): NaN, negative and zero pixels only -> plain area fallback.
  * All the other src cells only have zero pixels -> plain area fallback (0.25 to each dst subcell).
- Shifted proxy: 5x5 raster of 1 deg pixels shifted half a pixel (origin -0.5, 4.5), so the reading window does
  not fall on the pixel edges: every dst cell lies in a single pixel, and the 4 dst cells of a src cell lie in
  4 different pixels with equal overlaps, so every src cell is split proportionally to those 4 pixel values.
- Shifted partial proxy: the same with a 3x3 raster (origin -0.5, 2.5) ending inside the domain, so the reading
  window exceeds the raster extent; the src cells without pixels fall back to plain area weights.
- Mass (flux=False) and emissions (flux=True) are conserved, and both modes agree cell by cell.

Run: python test_raster_proxy_synthetic.py
     mpirun -np 4 python test_raster_proxy_synthetic.py
"""

import numpy as np
from rasterio.transform import from_origin

from nes.tests.methods.horizontal_interpolation.utils import (
    RANK,
    assert_shares,
    check_flux_and_mass,
    default_values,
    plain_area_shares,
    shares,
    shifted_pixels,
    shifted_pixels_shares,
    write_raster,
)


def aligned():
    # Raster row 7 is the southern row (lat 0-0.5)
    data = np.zeros((8, 8), dtype=np.float32)
    data[7, 0], data[6, 1] = 1.0, 3.0
    data[7, 2], data[7, 3], data[6, 2], data[6, 3] = -9999.0, 2.0, 2.0, 2.0
    data[7, 4], data[7, 5] = np.nan, -1.0
    path = write_raster("aligned.tif", data, from_origin(0.0, 4.0, 0.5, 0.5), nodata=-9999.0)
    result = check_flux_and_mass(default_values(), "raster aligned", proxy=path, proxy_type="raster")
    if RANK != 0:
        return
    share = shares(result[0], result[2])
    assert_shares(
        share,
        [
            ((0, 0), 0.25),
            ((1, 1), 0.75),
            ((0, 1), 0.0),
            ((1, 0), 0.0),
            ((0, 2), 0.0),
            ((0, 3), 1 / 3),
            ((1, 2), 1 / 3),
            ((1, 3), 1 / 3),
            (np.s_[0:2, 4:6], plain_area_shares()[0:2, 4:6]),
            (np.s_[2:8, :], plain_area_shares()[2:8, :]),
            (np.s_[0:2, 6:8], plain_area_shares()[0:2, 6:8]),
        ],
        "raster aligned",
    )


def shifted(n_pixels, label):
    pixels, transform = shifted_pixels(n_pixels)
    path = write_raster(f"shifted_{n_pixels}.tif", pixels, transform)
    result = check_flux_and_mass(default_values(), label, proxy=path, proxy_type="raster")
    if RANK != 0:
        return
    assert_shares(shares(result[0], result[2]), [(np.s_[:, :], shifted_pixels_shares(pixels))], label)


def main():
    aligned()
    shifted(5, "raster shifted")
    shifted(3, "raster shifted partial")
    if RANK == 0:
        print("ALL CHECKS PASSED")


def test_raster_proxy_conservation():
    main()


if __name__ == "__main__":
    main()
