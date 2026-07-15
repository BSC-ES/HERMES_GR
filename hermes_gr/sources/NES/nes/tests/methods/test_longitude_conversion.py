# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import numpy as np
import pytest

import nes
from nes.tests.utils import DummyData, ExpectedData, load_fresh_nes

FILENAME = "test_lon_conversion.nc"
LonConvData = DummyData.LongitudeConversion
ExpLonConvData = ExpectedData.LongitudeConversion


@pytest.fixture(scope="module")
def tmp_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("longitude_conversion")


@pytest.fixture(scope="module")
def filepath(tmp_dir):
    return str(tmp_dir / FILENAME)


@pytest.fixture(scope="module", autouse=True)
def setup_test_file(filepath):
    """Create the test netCDF file before any tests run."""

    nessy = nes.create_nes(
        comm=None,
        info=False,
        projection=LonConvData["projection"],
        lat_orig=LonConvData["lat_orig"],
        lon_orig=LonConvData["lon_orig"],
        inc_lat=LonConvData["inc_lat"],
        inc_lon=LonConvData["inc_lon"],
        n_lat=LonConvData["n_lat"],
        n_lon=LonConvData["n_lon"],
        times=LonConvData["times"],
    )

    # Create spatial bounds to ensure longitude conversion works correctly
    nessy.create_spatial_bounds()

    # Add dummy variable data with deterministic values
    nessy.variables = {
        "var1": {
            "data": LonConvData["var1"]["data"],
            "dimensions": LonConvData["var1"]["dimensions"],
            "units": LonConvData["var1"]["units"],
        }
    }
    nessy.to_netcdf(filepath)


def test_longitude_conversion(filepath):
    """Test that convert_longitudes correctly converts longitude values and bounds to the range [-180, 180]."""

    nessy = load_fresh_nes(filepath)

    try:
        nessy.convert_longitudes()
    except Exception as e:
        pytest.fail(f"Longitude conversion failed with error: {e}")

    assert nessy.lon is not None and nessy.lon_bnds is not None, (
        "Longitude variables should not be None after conversion."
    )
    assert np.allclose(nessy.lon["data"], ExpLonConvData["lon"]), (
        "Longitude values do not match expected values after conversion."
    )
    assert np.min(nessy.lon_bnds["data"]) >= -180, (
        "Minimum longitude bound should be greater than or equal to -180 after conversion."
    )
    assert np.max(nessy.lon_bnds["data"]) <= 180, (
        "Maximum longitude bound should be less than or equal to 180 after conversion."
    )
