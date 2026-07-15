# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import numpy as np
import pytest

import nes
from nes.tests.utils import DummyData, compute_expected_rolling_mean, load_fresh_nes

FILENAME = "test_rolling_mean.nc"

RollingMean = DummyData.RollingMean


@pytest.fixture(scope="module")
def tmp_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("rolling_mean")


@pytest.fixture(scope="module")
def filepath(tmp_dir):
    return str(tmp_dir / FILENAME)


@pytest.fixture(scope="module", autouse=True)
def setup_test_file(filepath):
    """Create the test netCDF file before any tests run."""

    nessy = nes.create_nes(
        comm=None,
        info=False,
        projection=RollingMean["projection"],
        lat_orig=RollingMean["lat_orig"],
        lon_orig=RollingMean["lon_orig"],
        inc_lat=RollingMean["inc_lat"],
        inc_lon=RollingMean["inc_lon"],
        n_lat=RollingMean["n_lat"],
        n_lon=RollingMean["n_lon"],
        times=RollingMean["times"],
    )

    # Add dummy variable data with deterministic values
    nessy.variables = {
        "var1": {
            "data": RollingMean["var1"]["data"],
            "dimensions": RollingMean["var1"]["dimensions"],
            "units": RollingMean["var1"]["units"],
        }
    }
    nessy.to_netcdf(filepath)


class TestRollingMean:
    """Tests for the rolling_mean method of the NES class."""

    HOURS = 4

    def test_does_not_raise(self, filepath):
        """Test that rolling_mean does not raise an exception."""

        nessy = load_fresh_nes(filepath)
        try:
            nessy.rolling_mean(var_list="var1", hours=self.HOURS)
        except Exception as e:
            pytest.fail(f"rolling_mean raised an exception: {e}")

    def test_output_shape_matches_input(self, filepath):
        """Test that the output shape matches the input shape."""

        nessy = load_fresh_nes(filepath)
        result = nessy.rolling_mean(var_list="var1", hours=self.HOURS)

        assert result.variables["var1"]["data"].shape == nessy.variables["var1"]["data"].shape, (
            "Output shape should match input shape."
        )

    def test_returns_new_object(self, filepath):
        """Test that rolling_mean returns a new NES object and does not modify the original."""

        nessy = load_fresh_nes(filepath)
        original_data = nessy.variables["var1"]["data"].copy()
        result = nessy.rolling_mean(var_list="var1", hours=self.HOURS)

        assert np.array_equal(nessy.variables["var1"]["data"], original_data), (
            "Original NES data should not be modified by rolling_mean."
        )
        assert result is not nessy, "rolling_mean should return a new NES object, not modify the original."

    def test_nan_count(self, filepath):
        """Test that the first (hours - 1) timesteps should be NaN, the rest should not."""

        nessy = load_fresh_nes(filepath)
        result = nessy.rolling_mean(var_list="var1", hours=self.HOURS)
        data = result.variables["var1"]["data"]

        # First (hours - 1) timesteps: all NaN
        for t in range(self.HOURS - 1):
            assert np.all(np.isnan(data[t])), f"Expected all NaN at timestep {t}"

        # Remaining timesteps: no NaN
        for t in range(self.HOURS - 1, data.shape[0]):
            assert not np.any(np.isnan(data[t])), f"Unexpected NaN at timestep {t}"

    def test_values(self, filepath):
        """Test that the rolling mean values are correct."""

        nessy = load_fresh_nes(filepath)
        result = nessy.rolling_mean(var_list="var1", hours=self.HOURS)
        expected = compute_expected_rolling_mean(nessy.variables["var1"]["data"], self.HOURS)

        assert np.allclose(result.variables["var1"]["data"], expected, equal_nan=True), (
            "Rolling mean values do not match expected values."
        )

    def test_dimensions_preserved(self, filepath):
        """Test that the dimensions of the variable are preserved in the output."""

        nessy = load_fresh_nes(filepath)
        result = nessy.rolling_mean(var_list="var1", hours=self.HOURS)
        assert result.variables["var1"]["dimensions"] == nessy.variables["var1"]["dimensions"], (
            "Dimensions should be preserved in the output."
        )

    def test_time_unchanged(self, filepath):
        """Test that the time variable is unchanged in the output."""

        nessy = load_fresh_nes(filepath)
        result = nessy.rolling_mean(var_list="var1", hours=self.HOURS)
        assert result.time == nessy.time, "Time variable should be unchanged in the output."


class TestRollingMeanDefaultHours:
    """Test that the default number of hours used for the rolling mean is 8."""

    HOURS = 8

    def test_nan_count(self, filepath):
        """Test that the first (hours - 1) timesteps should be NaN, the rest should not."""

        nessy = load_fresh_nes(filepath)
        result = nessy.rolling_mean(var_list="var1")
        data = result.variables["var1"]["data"]

        for t in range(self.HOURS - 1):
            assert np.all(np.isnan(data[t])), f"Expected all NaN at timestep {t}"

        for t in range(self.HOURS - 1, data.shape[0]):
            assert not np.any(np.isnan(data[t])), f"Unexpected NaN at timestep {t}"

    def test_values(self, filepath):
        """Test that the rolling mean values are correct when using the default number of hours."""

        nessy = load_fresh_nes(filepath)
        result = nessy.rolling_mean(var_list="var1")
        expected = compute_expected_rolling_mean(nessy.variables["var1"]["data"], self.HOURS)

        assert np.allclose(result.variables["var1"]["data"], expected, equal_nan=True), (
            "Rolling mean values do not match expected values when using default hours."
        )


class TestRollingMeanVarList:
    """Test that the var_list argument can be provided as a string, list, or None."""

    HOURS = 4

    def test_var_list_as_string(self, filepath):
        """Test that var_list can be provided as a string."""

        nessy = load_fresh_nes(filepath)
        try:
            result = nessy.rolling_mean(var_list="var1", hours=self.HOURS)
        except Exception as e:
            pytest.fail(f"rolling_mean raised an exception when var_list is a string: {e}")

        assert "var1" in result.variables, (
            "Variable 'var1' should be in the output NES when var_list is provided as a string."
        )

    def test_var_list_as_list(self, filepath):
        """Test that var_list can be provided as a list."""

        nessy = load_fresh_nes(filepath)
        try:
            result = nessy.rolling_mean(var_list=["var1"], hours=self.HOURS)
        except Exception as e:
            pytest.fail(f"rolling_mean raised an exception when var_list is a list: {e}")

        assert "var1" in result.variables, (
            "Variable 'var1' should be in the output NES when var_list is provided as a list."
        )

    def test_var_list_none(self, filepath):
        """Test that var_list can be None."""

        nessy = load_fresh_nes(filepath)
        try:
            result = nessy.rolling_mean(var_list=None, hours=self.HOURS)
        except Exception as e:
            pytest.fail(f"rolling_mean raised an exception when var_list is None: {e}")

        assert "var1" in result.variables, "Variable 'var1' should be in the output NES when var_list is None."
