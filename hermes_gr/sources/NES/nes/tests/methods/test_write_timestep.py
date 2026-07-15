# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

from datetime import datetime, timedelta

import numpy as np
import pytest

import nes
from nes.tests.utils import DummyData

FILENAME = "test_write_timestep.nc"
WriteTimestep = DummyData.WriteTimestep


@pytest.fixture(scope="module")
def tmp_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("write_timestep")


@pytest.fixture(scope="module")
def filepath(tmp_dir):
    return str(tmp_dir / FILENAME)


def setup_test_file():
    """Create the test netCDF file before any tests run."""

    nessy = nes.create_nes(
        comm=None,
        info=False,
        projection=WriteTimestep["projection"],
        lat_orig=WriteTimestep["lat_orig"],
        lon_orig=WriteTimestep["lon_orig"],
        inc_lat=WriteTimestep["inc_lat"],
        inc_lon=WriteTimestep["inc_lon"],
        n_lat=WriteTimestep["n_lat"],
        n_lon=WriteTimestep["n_lon"],
    )

    # Add dummy variable data with deterministic values
    nessy.variables = {
        "var1": {
            "data": WriteTimestep["var1"]["data"],
            "units": WriteTimestep["var1"]["units"],
            "dtype": WriteTimestep["var1"]["dtype"],
        }
    }
    return nessy


def test_write_timestep(filepath):
    """Test that the append_time_step_data method correctly appends
    data for each time step and updates the time variable.
    """

    # Set up the nessy object
    nessy = setup_test_file()

    n_hours = DummyData.n_hours_timestep
    dates = [datetime(2020, 1, 1) + timedelta(hours=i) for i in range(n_hours)]

    # Set the time variable with the generated dates
    try:
        nessy.set_time(dates)
        nessy.to_netcdf(filepath, keep_open=True, info=False, serial=True)
    except Exception as e:
        pytest.fail(f"Setting time with daily intervals failed: {e}")

    assert nessy.lat and nessy.lon

    for idx in range(n_hours):
        # Update variable data with deterministic values based on the index
        nessy.variables["var1"]["data"] = np.ones((1, 1, nessy.lat["data"].shape[0], nessy.lon["data"].shape[-1])) * (
            idx
        )
        # Append the time step data
        try:
            nessy.append_time_step_data(idx)
        except Exception as e:
            pytest.fail(f"Appending time step data for daily intervals failed at index {idx}: {e}")

        if idx == len(dates) - 1:
            nessy.close()

    # Reopen the file to check the final shape of the variable data
    nessy = nes.open_netcdf(filepath)
    nessy.load()

    # For linter
    assert nessy.lat and nessy.lon

    # Check that data and date match
    for idx, date in enumerate(dates):
        assert (nessy.variables["var1"]["data"][idx, 0, :, :] == idx).all()
        assert nessy.time[idx] == date

    assert nessy.variables["var1"]["data"].shape == (
        n_hours,
        1,
        nessy.lat["data"].shape[0],
        nessy.lon["data"].shape[-1],
    )
