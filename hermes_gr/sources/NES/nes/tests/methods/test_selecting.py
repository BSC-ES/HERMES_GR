# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import numpy as np
import pytest

import nes
from nes.tests.utils import DummyData, ExpectedData

FILENAME = "test_sel.nc"
SelData = DummyData.Selecting
ExpSelData = ExpectedData.Selecting


@pytest.fixture(scope="module")
def tmp_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("selecting")


@pytest.fixture(scope="module")
def filepath(tmp_dir):
    return str(tmp_dir / FILENAME)


@pytest.fixture(scope="module", autouse=True)
def setup_test_file(filepath):
    """Create the test netCDF file before any tests run."""

    nessy = nes.create_nes(
        comm=None,
        info=False,
        projection=SelData["projection"],
        lat_orig=SelData["lat_orig"],
        lon_orig=SelData["lon_orig"],
        inc_lat=SelData["inc_lat"],
        inc_lon=SelData["inc_lon"],
        n_lat=SelData["n_lat"],
        n_lon=SelData["n_lon"],
        times=SelData["times"],
    )

    # Add dummy variable data with deterministic values
    nessy.variables = {
        "var1": {
            "data": SelData["var1"]["data"],
            "dimensions": SelData["var1"]["dimensions"],
            "units": SelData["var1"]["units"],
        }
    }
    nessy.to_netcdf(filepath)


class TestDimensionalSelection:
    """Tests for selecting latitudes, longitudes and combination of both using the sel method of the NES class."""

    def test_lat_selection(self, filepath):
        """Test selecting latitudes within specified bounds."""

        lower_lat = ExpSelData["lat"]["lower_bound"]
        upper_lat = ExpSelData["lat"]["upper_bound"]

        # Open nes object from netcdf and select latitudes
        nessy = nes.open_netcdf(filepath)
        try:
            nessy.sel(lat_min=lower_lat, lat_max=upper_lat)
            nessy.load()
        except Exception as e:
            pytest.fail(f"Latitude selection failed with error: {e}")

        # Check lat selection for None, that the data is within bounds and values match
        assert nessy.lat is not None, "Latitude coordinate should not be None after selection."
        assert np.allclose(np.array(nessy.lat["data"]), ExpSelData["lat"]["data"]), (
            "Latitude values after selection do not match expected values."
        )
        assert np.min(nessy.lat["data"]) >= lower_lat, "Minimum latitude value is below the specified lower bound."
        assert np.max(nessy.lat["data"]) <= upper_lat, "Maximum latitude value is above the specified upper bound."

    def test_lon_selection(self, filepath):
        """Test selecting longitudes within specified bounds."""

        lower_lon = ExpSelData["lon"]["lower_bound"]
        upper_lon = ExpSelData["lon"]["upper_bound"]

        # # Open nes object from netcdf and select latitudes
        nessy = nes.open_netcdf(filepath)
        try:
            nessy.sel(lon_min=lower_lon, lon_max=upper_lon)
            nessy.load()
        except Exception as e:
            pytest.fail(f"Longitude selection failed with error: {e}")

        # Check lon selection for None, that the data is within bounds and values match
        assert nessy.lon is not None, "Longitude coordinate should not be None after selection."
        assert np.allclose(np.array(nessy.lon["data"]), ExpSelData["lon"]["data"]), (
            "Longitude values after selection do not match expected values."
        )
        assert np.min(nessy.lon["data"]) >= lower_lon, "Minimum longitude value is below the specified lower bound."
        assert np.max(nessy.lon["data"]) <= upper_lon, "Maximum longitude value is above the specified upper bound."

    def test_lat_lon_selection(self, filepath):
        """Test selecting latitudes and longitudes within specified bounds."""

        lower_lat = ExpSelData["lat"]["lower_bound"]
        upper_lat = ExpSelData["lat"]["upper_bound"]
        lower_lon = ExpSelData["lon"]["lower_bound"]
        upper_lon = ExpSelData["lon"]["upper_bound"]

        # Open nes object from netcdf and select latitudes and longitudes
        nessy = nes.open_netcdf(filepath)
        try:
            nessy.sel(lat_min=lower_lat, lat_max=upper_lat, lon_min=lower_lon, lon_max=upper_lon)
            nessy.load()
        except Exception as e:
            pytest.fail(f"Latitude and Longitude selection failed with error: {e}")

        # Check lat selection for None, that the data is within bounds and values match
        assert nessy.lat is not None, "Latitude coordinate should not be None after selection."
        assert np.allclose(np.array(nessy.lat["data"]), ExpSelData["lat"]["data"]), (
            "Latitude values after selection do not match expected values."
        )
        assert np.min(nessy.lat["data"]) >= lower_lat, "Minimum latitude value is below the specified lower bound."
        assert np.max(nessy.lat["data"]) <= upper_lat, "Maximum latitude value is above the specified upper bound."

        assert nessy.lon is not None, "Longitude coordinate should not be None after selection."
        assert np.allclose(np.array(nessy.lon["data"]), ExpSelData["lon"]["data"]), (
            "Longitude values after selection do not match expected values."
        )
        assert np.min(nessy.lon["data"]) >= lower_lon, "Minimum longitude value is below the specified lower bound."
        assert np.max(nessy.lon["data"]) <= upper_lon, "Maximum longitude value is above the specified upper bound."


class TestTemporalSelection:
    """Tests for selecting time steps using the sel method of the NES class."""

    def test_time_selection(self, filepath):
        """Test selecting time steps within specified bounds."""

        times = SelData["times"]
        grid_size = SelData["n_lat"] * SelData["n_lon"]
        for i, time in enumerate(times):
            # Open nes object from netcdf and select times
            nessy = nes.open_netcdf(filepath)
            nessy.keep_vars(["var1"])
            try:
                nessy.sel(time_min=time, time_max=time)
                nessy.load()
            except Exception as e:
                pytest.fail(f"Time selection failed with error: {e}")

            # Calculate expected data range for this timestep
            start = i * grid_size + 1
            end = (i + 1) * grid_size + 1
            expected_data = np.arange(start, end).reshape(1, 1, SelData["n_lat"], SelData["n_lon"])

            assert nessy.time is not None, "Time variable should not be None after selection."
            assert len(nessy.time) == 1, "Exactly one time step should be selected."
            assert nessy.time[0] == time, "Selected time step does not match expected time."
            assert np.array_equal(nessy.variables["var1"]["data"], expected_data), (
                "Variable data after time selection does not match expected values."
            )

    def test_hours_selection(self, filepath):
        """Test selecting time steps based on hours since the start of the simulation."""

        times = SelData["times"]

        # Test selecting hours in increments of 24 hours, which should select the corresponding time steps
        for idx in range(len(times)):
            nessy = nes.open_netcdf(filepath)
            nessy.keep_vars(["var1"])
            try:
                nessy.sel(hours_start=0, hours_end=(24 * (len(times) - 1)) - idx * 24)
                nessy.load()
            except Exception as e:
                pytest.fail(f"Hours selection failed with error: {e}")

            assert nessy.time is not None, "Time variable should not be None after selection."
            assert len(nessy.time) == idx + 1, "Incorrect number of time steps selected."
            assert nessy.time == times[: idx + 1], "Selected time steps do not match expected times."

            expected_data = np.arange(1, (idx + 1) * SelData["n_lat"] * SelData["n_lon"] + 1).reshape(
                idx + 1, 1, SelData["n_lat"], SelData["n_lon"]
            )
            assert np.array_equal(nessy.variables["var1"]["data"], expected_data)

        # Test selecting individual time steps by specifying hours that correspond to each time step
        for idx, time in enumerate(times):
            nessy = nes.open_netcdf(filepath)
            nessy.keep_vars(["var1"])
            try:
                nessy.sel(hours_start=idx * 24, hours_end=(24 * (len(times) - 1)) - idx * 24)
                nessy.load()
            except Exception as e:
                pytest.fail(f"Hours selection for individual time steps failed with error: {e}")

            assert nessy.time is not None, "Time variable should not be None after selection."
            assert len(nessy.time) == 1, "Exactly one time step should be selected."
            assert nessy.time[0] == time, "Selected time step does not match expected time."

            expected_data = np.arange(
                idx * SelData["n_lat"] * SelData["n_lon"] + 1,
                (idx + 1) * SelData["n_lat"] * SelData["n_lon"] + 1,
            ).reshape(1, 1, SelData["n_lat"], SelData["n_lon"])

            assert np.array_equal(nessy.variables["var1"]["data"], expected_data), (
                "Variable data after hours selection does not match expected values."
            )
