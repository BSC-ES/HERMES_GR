# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import pytest

import nes
from nes.tests.utils import DummyData, load_fresh_nes

FILENAME1 = "test_sum1.nc"
FILENAME2 = "test_sum2.nc"
SumData1 = DummyData.Sum


@pytest.fixture(scope="module")
def tmp_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("summing")


@pytest.fixture(scope="module")
def filepath1(tmp_dir):
    return str(tmp_dir / FILENAME1)


@pytest.fixture(scope="module")
def filepath2(tmp_dir):
    return str(tmp_dir / FILENAME2)


@pytest.fixture(scope="module", autouse=True)
def setup_test_file(filepath1, filepath2):
    """Create the test netCDF file before any tests run."""

    nessy1 = nes.create_nes(
        comm=None,
        info=False,
        projection=SumData1["projection"],
        lat_orig=SumData1["lat_orig"],
        lon_orig=SumData1["lon_orig"],
        inc_lat=SumData1["inc_lat"],
        inc_lon=SumData1["inc_lon"],
        n_lat=SumData1["n_lat"],
        n_lon=SumData1["n_lon"],
        times=SumData1["times"],
    )

    # Add dummy variable data with deterministic values
    nessy1.variables = {
        "var1": {
            "data": SumData1["var1"]["data"],
            "dimensions": SumData1["var1"]["dimensions"],
            "units": SumData1["var1"]["units"],
        }
    }
    nessy1.to_netcdf(filepath1)

    nessy2 = nes.create_nes(
        comm=None,
        info=False,
        projection=SumData1["projection"],
        lat_orig=SumData1["lat_orig"],
        lon_orig=SumData1["lon_orig"],
        inc_lat=SumData1["inc_lat"],
        inc_lon=SumData1["inc_lon"],
        n_lat=SumData1["n_lat"],
        n_lon=SumData1["n_lon"],
        times=SumData1["times"],
    )

    # Add dummy variable data with deterministic values
    nessy2.variables = {
        "var1": {
            "data": SumData1["var2"]["data"],
            "dimensions": SumData1["var2"]["dimensions"],
            "units": SumData1["var2"]["units"],
        }
    }
    nessy2.to_netcdf(filepath2)


def test_sum(filepath1, filepath2):
    """Test the sum method of the NES class."""

    # Load the two test files
    nessy1 = load_fresh_nes(filepath1)
    nessy2 = load_fresh_nes(filepath2)

    # Perform the sum operation
    try:
        nessy_sum = nessy1 + nessy2
    except Exception as e:
        pytest.fail(f"Sum operation failed: {e}")

    # Verify that the sum is correct (var1 should be 3 everywhere)
    assert (nessy_sum.variables["var1"]["data"] == 3.0).all(), (
        "The sum of var1 from both NES objects should be 3.0 everywhere."
    )
    assert nessy_sum.variables["var1"]["data"].shape == nessy1.variables["var1"]["data"].shape, (
        "The shape of the summed variable should match the shape of the input variables."
    )
    assert nessy_sum.variables["var1"]["data"].shape == nessy2.variables["var1"]["data"].shape, (
        "The shape of the summed variable should match the shape of the input variables."
    )
