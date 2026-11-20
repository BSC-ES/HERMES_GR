# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

"""Synthetic test for the raster_categorical-proxy conservative interpolation.

- src: 4x4 regular grid of 1 deg cells over [0,4]x[0,4] with distinct values per cell.
- dst: 8x8 regular grid of 0.5 deg cells over the same domain (dst index = [lat, lon]).
- proxy: 16x16 categorical raster of 0.25 deg pixels (finer than the dst grid), keeping categories [5, 7]:
  * Src cell [0,1)x[0,1): one kept pixel (category 5) in dst (0, 0) and one (category 7) in dst (1, 1)
    -> 50%/50% (the categorical proxy is a presence mask: all kept pixels weigh the same per unit area).
  * Src cell [1,2)x[0,1): three kept pixels in dst (0, 2) and one in dst (1, 3) -> 75%/25%.
  * Src cell [2,3)x[0,1): only pixels of a category that is not kept (9) -> plain area fallback.
  * Src cell [3,4)x[0,1): every pixel kept -> 25% to each dst subcell.
  * All the other src cells (category 0) -> plain area fallback (0.25 to each dst subcell).
- Raster shifted half a pixel (5x5 and, ending inside the domain, 3x3 pixels of 1 deg with origin
  (-0.5, n - 0.5)), keeping the pixels of category 5: every src cell is split proportionally to the area of its
  kept pixels.
- Mass (flux=False) and emissions (flux=True) are conserved, and both modes agree cell by cell.

Run: python test_raster_categorical_proxy_synthetic.py
     mpirun -np 4 python test_raster_categorical_proxy_synthetic.py
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


def main():
    # Raster row 15 is the southern row (lat 0-0.25); every dst cell holds 2x2 pixels
    cats = np.zeros((16, 16), dtype=np.uint8)
    cats[15, 0] = 5  # dst (0, 0)
    cats[12, 3] = 7  # dst (1, 1)
    cats[14:16, 4] = 5  # dst (0, 2)
    cats[15, 5] = 7  # dst (0, 2)
    cats[13, 7] = 5  # dst (1, 3)
    cats[12:16, 8:12] = 9  # not kept
    cats[12:16, 12:16] = 7
    path = write_raster("categorical.tif", cats, from_origin(0.0, 4.0, 0.25, 0.25))
    result = check_flux_and_mass(
        default_values(), "raster_categorical", proxy=path, proxy_type="raster_categorical", proxy_categories=[5, 7]
    )
    shifted = {}
    for n_pixels in (5, 3):
        pixels, transform = shifted_pixels(n_pixels)
        shifted_cats = np.where(pixels % 3 == 0, 5, 1).astype(np.uint8)
        label = f"raster_categorical shifted {n_pixels}"
        shifted[label] = (
            (shifted_cats == 5).astype(np.float64),
            check_flux_and_mass(
                default_values(),
                label,
                proxy=write_raster(f"shifted_cats_{n_pixels}.tif", shifted_cats, transform),
                proxy_type="raster_categorical",
                proxy_categories=[5],
            ),
        )
    if RANK != 0:
        return
    for label, (weights, shifted_result) in shifted.items():
        assert_shares(
            shares(shifted_result[0], shifted_result[2]), [(np.s_[:, :], shifted_pixels_shares(weights))], label
        )
    share = shares(result[0], result[2])
    assert_shares(
        share,
        [
            ((0, 0), 0.5),
            ((1, 1), 0.5),
            ((0, 1), 0.0),
            ((1, 0), 0.0),
            ((0, 2), 0.75),
            ((1, 3), 0.25),
            ((0, 3), 0.0),
            ((1, 2), 0.0),
            (np.s_[0:2, 4:6], plain_area_shares()[0:2, 4:6]),
            (np.s_[0:2, 6:8], 0.25),
            (np.s_[2:8, :], plain_area_shares()[2:8, :]),
        ],
        "raster_categorical",
    )
    print("ALL CHECKS PASSED")


def test_raster_categorical_proxy_conservation():
    main()


if __name__ == "__main__":
    main()
