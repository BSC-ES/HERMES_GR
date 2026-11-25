# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import numpy as np
import pytest

import nes
from nes.tests.utils import DummyData, ExpectedData, load_fresh_nes

FILENAME = "test_grid_area.nc"
GridAreaData = DummyData.GridArea
ExpGridAreaData = ExpectedData.GridArea


@pytest.fixture(scope="module")
def tmp_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("grid_area")


@pytest.fixture(scope="module")
def filepath(tmp_dir):
    return str(tmp_dir / FILENAME)


@pytest.fixture(scope="module", autouse=True)
def setup_test_file(filepath):
    """Create the test netCDF file before any tests run."""

    nessy = nes.create_nes(
        comm=None,
        info=False,
        projection=GridAreaData["projection"],
        lat_orig=GridAreaData["lat_orig"],
        lon_orig=GridAreaData["lon_orig"],
        inc_lat=GridAreaData["inc_lat"],
        inc_lon=GridAreaData["inc_lon"],
        n_lat=GridAreaData["n_lat"],
        n_lon=GridAreaData["n_lon"],
        times=GridAreaData["times"],
    )

    # Add dummy variable data with deterministic values
    nessy.variables = {
        "var1": {
            "data": GridAreaData["var1"]["data"],
            "dimensions": GridAreaData["var1"]["dimensions"],
            "units": GridAreaData["var1"]["units"],
        }
    }
    nessy.to_netcdf(filepath)


def test_grid_area_calculation(filepath):
    """Test the grid area calculation method of the NES class."""

    # Open
    nessy = load_fresh_nes(filepath)
    # Calculate grid area
    try:
        print("HOLAaaaaaaa")
        nessy.calculate_grid_area()
    except Exception as e:
        pytest.fail(f"Grid area calculation failed with error: {e}")

    assert nessy.cell_measures is not None, "cell_measures attribute should not be None after calculating grid area."
    assert nessy.cell_measures["cell_area"]["data"] is not None, (
        "cell_area data should not be None after calculating grid area."
    )
    print("Calculated: ", nessy.cell_measures["cell_area"]["data"])
    print("Expected: ", ExpGridAreaData["cell_area"])
    assert np.allclose(nessy.cell_measures["cell_area"]["data"], ExpGridAreaData["cell_area"]), (
        "Calculated cell area does not match expected values."
    )

    # ! CHECK WHY GRID AREA DOES NOT GIVE EXPECTED VALUES FOR LCC AND MERCATOR
