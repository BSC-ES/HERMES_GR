# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

"""Synthetic test for the raster_categorical_numerical-proxy conservative interpolation.

- src: 4x4 regular grid of 1 deg cells over [0,4]x[0,4] with distinct values per cell.
- dst: 8x8 regular grid of 0.5 deg cells over the same domain (dst index = [lat, lon]).
- categorical raster (8x8, 0.5 deg), keeping categories [5]; numerical raster (nodata 65535):
  * Src cell [0,1)x[0,1): kept pixels with values 3 (dst (0, 0)), 1 (dst (0, 1)) and nodata (dst (1, 0));
    dst (1, 1) has value 50 but category 0 -> 75%/25%/0/0.
  * Src cell [1,2)x[0,1): kept pixels with nodata values only -> plain area fallback.
  * Src cell [2,3)x[0,1): kept pixels with zero values only -> plain area fallback.
  * Src cell [3,4)x[0,1): kept pixels with negative and NaN values only -> plain area fallback.
  * All the other src cells have no kept pixel -> plain area fallback.
- The proxy given inside a priority list as ("raster_categorical_numerical", (cat, num), [5]) gives the same
  result.
- A numerical raster whose grid differs from the categorical one (finer resolution, smaller extent, other CRS
  or shifted origin) is rejected with a ValueError: it is never reprojected or resampled.
- Both rasters shifted half a pixel (5x5 and, ending inside the domain, 3x3 pixels of 1 deg with origin
  (-0.5, n - 0.5)), keeping the even valued pixels: every src cell is split proportionally to its kept pixel values.
- Mass (flux=False) and emissions (flux=True) are conserved, and both modes agree cell by cell.

Run: python test_raster_categorical_numerical_proxy_synthetic.py
     mpirun -np 4 python test_raster_categorical_numerical_proxy_synthetic.py
"""

import numpy as np
import pytest
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
    # Raster row 7 is the southern row (lat 0-0.5)
    cats = np.zeros((8, 8), dtype=np.uint8)
    cats[7, 0:8] = 5
    cats[6, 0] = 5
    nums = np.zeros((8, 8), dtype=np.float32)
    nums[7, 0], nums[7, 1], nums[6, 0], nums[6, 1] = 3.0, 1.0, 65535.0, 50.0
    nums[7, 2:4] = 65535.0
    nums[7, 6], nums[7, 7] = -1.0, np.nan
    cat_path = write_raster("categorical.tif", cats, from_origin(0.0, 4.0, 0.5, 0.5))
    num_path = write_raster("numerical.tif", nums, from_origin(0.0, 4.0, 0.5, 0.5), nodata=65535.0)

    values = default_values()
    results = {
        "aligned": check_flux_and_mass(
            values,
            "raster_categorical_numerical aligned",
            proxy=cat_path,
            proxy_type="raster_categorical_numerical",
            proxy_numerical=num_path,
            proxy_categories=[5],
        ),
        "list": check_flux_and_mass(
            values,
            "raster_categorical_numerical list",
            proxy=[("raster_categorical_numerical", (cat_path, num_path), [5])],
        ),
    }
    # A numerical raster whose grid differs from the categorical one is rejected: it is never resampled
    mismatched = {
        "fine": write_raster(
            "numerical_fine.tif",
            np.repeat(np.repeat(nums, 2, axis=0), 2, axis=1),
            from_origin(0.0, 4.0, 0.25, 0.25),
            nodata=65535.0,
        ),
        "partial": write_raster(
            "numerical_partial.tif", np.ascontiguousarray(nums[:, 0:1]), from_origin(0.0, 4.0, 0.5, 0.5), nodata=65535.0
        ),
        "crs": write_raster(
            "numerical_crs.tif", nums, from_origin(0.0, 4.0, 0.5, 0.5), nodata=65535.0, crs="EPSG:4258"
        ),
        "shifted": write_raster("numerical_shifted.tif", nums, from_origin(0.5, 4.0, 0.5, 0.5), nodata=65535.0),
    }
    for name, path in mismatched.items():
        with pytest.raises(ValueError, match="exactly the same grid"):
            check_flux_and_mass(
                values,
                f"raster_categorical_numerical {name} mismatch",
                proxy=cat_path,
                proxy_type="raster_categorical_numerical",
                proxy_numerical=path,
                proxy_categories=[5],
            )

    # Both rasters shifted half a pixel (the reading window is not on the pixel edges); in the partial case they
    # end inside the domain. Pixels with odd values are not kept by the categorical raster.
    shifted = {}
    for n_pixels in (5, 3):
        pixels, transform = shifted_pixels(n_pixels)
        shifted_cats = np.where(pixels % 2 == 0, 5, 1).astype(np.uint8)
        label = f"raster_categorical_numerical shifted {n_pixels}"
        shifted[label] = (
            pixels * (shifted_cats == 5),
            check_flux_and_mass(
                values,
                label,
                proxy=write_raster(f"shifted_cats_{n_pixels}.tif", shifted_cats, transform),
                proxy_type="raster_categorical_numerical",
                proxy_categories=[5],
                proxy_numerical=write_raster(f"shifted_nums_{n_pixels}.tif", pixels, transform),
            ),
        )
    if RANK != 0:
        return
    for name, result in results.items():
        assert_shares(
            shares(result[0], result[2]),
            [
                ((0, 0), 0.75),
                ((0, 1), 0.25),
                ((1, 0), 0.0),
                ((1, 1), 0.0),
                (np.s_[0:2, 2:8], plain_area_shares()[0:2, 2:8]),
                (np.s_[2:8, :], plain_area_shares()[2:8, :]),
            ],
            f"raster_categorical_numerical {name}",
        )
    for label, (weights, result) in shifted.items():
        assert_shares(shares(result[0], result[2]), [(np.s_[:, :], shifted_pixels_shares(weights))], label)
    print("ALL CHECKS PASSED")


def test_raster_categorical_numerical_proxy_conservation():
    main()


if __name__ == "__main__":
    main()
