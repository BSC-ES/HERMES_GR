# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import numpy as np
import pytest

import nes
from nes.tests.utils import DummyData, load_fresh_nes

FILENAME = "test_daily_statistics.nc"
FILENAME_NAN = "test_daily_statistics_nan.nc"
FILENAME_25 = "test_daily_statistics_25.nc"
FILENAME_NAN_25 = "test_daily_statistics_nan_25.nc"

StatsData = DummyData.DailyStatistics


@pytest.fixture(scope="module")
def tmp_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("daily_statistics")


@pytest.fixture(scope="module")
def filepath(tmp_dir):
    return str(tmp_dir / FILENAME)


@pytest.fixture(scope="module")
def filepath_nan(tmp_dir):
    return str(tmp_dir / FILENAME_NAN)


@pytest.fixture(scope="module")
def filepath_25(tmp_dir):
    return str(tmp_dir / FILENAME_25)


@pytest.fixture(scope="module")
def filepath_nan_25(tmp_dir):
    return str(tmp_dir / FILENAME_NAN_25)


@pytest.fixture(scope="module", autouse=True)
def setup_test_file(filepath, filepath_nan, filepath_25, filepath_nan_25):
    """Create the test netCDF files before any tests run."""

    # regular no-nan file
    nessy = nes.create_nes(
        comm=None,
        info=False,
        projection=StatsData["no_nan"]["projection"],
        lat_orig=StatsData["no_nan"]["lat_orig"],
        lon_orig=StatsData["no_nan"]["lon_orig"],
        inc_lat=StatsData["no_nan"]["inc_lat"],
        inc_lon=StatsData["no_nan"]["inc_lon"],
        n_lat=StatsData["no_nan"]["n_lat"],
        n_lon=StatsData["no_nan"]["n_lon"],
        times=StatsData["no_nan"]["times"],
    )

    # Add dummy variable data with deterministic values
    nessy.variables = {
        "var1": {
            "data": StatsData["no_nan"]["var1"]["data"],
            "dimensions": StatsData["no_nan"]["var1"]["dimensions"],
            "units": StatsData["no_nan"]["var1"]["units"],
        }
    }
    nessy.to_netcdf(filepath)

    # NaN variant
    nessy_nan = nes.create_nes(
        comm=None,
        info=False,
        projection=StatsData["nan"]["projection"],
        lat_orig=StatsData["nan"]["lat_orig"],
        lon_orig=StatsData["nan"]["lon_orig"],
        inc_lat=StatsData["nan"]["inc_lat"],
        inc_lon=StatsData["nan"]["inc_lon"],
        n_lat=StatsData["nan"]["n_lat"],
        n_lon=StatsData["nan"]["n_lon"],
        times=StatsData["nan"]["times"],
    )
    nessy_nan.variables = {
        "var1": {
            "data": StatsData["nan"]["var1"]["data"],
            "dimensions": StatsData["nan"]["var1"]["dimensions"],
            "units": StatsData["nan"]["var1"]["units"],
        }
    }
    nessy_nan.to_netcdf(filepath_nan)

    # 25‑hour no-nan file
    nessy_25 = nes.create_nes(
        comm=None,
        info=False,
        projection=StatsData["no_nan_25"]["projection"],
        lat_orig=StatsData["no_nan_25"]["lat_orig"],
        lon_orig=StatsData["no_nan_25"]["lon_orig"],
        inc_lat=StatsData["no_nan_25"]["inc_lat"],
        inc_lon=StatsData["no_nan_25"]["inc_lon"],
        n_lat=StatsData["no_nan_25"]["n_lat"],
        n_lon=StatsData["no_nan_25"]["n_lon"],
        times=StatsData["no_nan_25"]["times"],
    )
    nessy_25.variables = {
        "var1": {
            "data": StatsData["no_nan_25"]["var1"]["data"],
            "dimensions": StatsData["no_nan_25"]["var1"]["dimensions"],
            "units": StatsData["no_nan_25"]["var1"]["units"],
        }
    }
    nessy_25.to_netcdf(filepath_25)

    # 25‑hour NaN variant
    nessy_nan_25 = nes.create_nes(
        comm=None,
        info=False,
        projection=StatsData["nan_25"]["projection"],
        lat_orig=StatsData["nan_25"]["lat_orig"],
        lon_orig=StatsData["nan_25"]["lon_orig"],
        inc_lat=StatsData["nan_25"]["inc_lat"],
        inc_lon=StatsData["nan_25"]["inc_lon"],
        n_lat=StatsData["nan_25"]["n_lat"],
        n_lon=StatsData["nan_25"]["n_lon"],
        times=StatsData["nan_25"]["times"],
    )
    nessy_nan_25.variables = {
        "var1": {
            "data": StatsData["nan_25"]["var1"]["data"],
            "dimensions": StatsData["nan_25"]["var1"]["dimensions"],
            "units": StatsData["nan_25"]["var1"]["units"],
        }
    }
    nessy_nan_25.to_netcdf(filepath_nan_25)


class TestMean:
    """Test the daily_statistic method for mean operation with different type_op options."""

    def test_calendar(self, filepath):
        """Test the calendar type_op option for mean operation."""

        nessy_mean = load_fresh_nes(filepath)
        # Manually calculate mean for each day (there s 3, so 72 time steps, final array should have 3 time steps)
        expected = (
            nessy_mean.variables["var1"]["data"]
            .reshape(3, 24, *nessy_mean.variables["var1"]["data"].shape[1:])
            .mean(axis=1)
        )
        # Get number of days
        expected_days = len(nessy_mean.time) // 24

        try:
            # Calculate daily mean using the calendar method
            nessy_mean.daily_statistic(op="mean", type_op="calendar")
        except Exception as e:
            pytest.fail(f"daily_statistic raised an exception: {e}")

        # Check data checks out
        assert np.array_equal(nessy_mean.variables["var1"]["data"], expected), (
            "The calculated daily mean does not match the expected values."
        )
        # Time dimensions should be amount of days
        assert np.shape(nessy_mean.variables["var1"]["data"])[0] == expected_days, (
            "The time dimension of the output does not match the expected number of days."
        )
        # Check cell methods attribute
        assert nessy_mean.variables["var1"]["cell_methods"] == "time: mean (interval: 1hr)", (
            "The cell_methods attribute was not set correctly after calculating the daily mean."
        )

    def test_calendar_25(self, filepath_25):
        """Test the calendar type_op option for mean operation when there are 25 time steps (1 extra hour)."""

        nessy_mean = load_fresh_nes(filepath_25)
        # only the first 24 hours should be aggregated, the extra hour is ignored
        expected = (
            nessy_mean.variables["var1"]["data"][:24]
            .reshape(1, 24, *nessy_mean.variables["var1"]["data"].shape[1:])
            .mean(axis=1)
        )
        expected_days = len(nessy_mean.time) // 24  # floor division yields 1

        try:
            nessy_mean.daily_statistic(op="mean", type_op="calendar")
        except Exception as e:
            pytest.fail(f"daily_statistic raised an exception: {e}")

        assert np.array_equal(nessy_mean.variables["var1"]["data"], expected), (
            "The calculated daily mean does not match the expected values."
        )
        assert np.shape(nessy_mean.variables["var1"]["data"])[0] == expected_days, (
            "The time dimension of the output does not match the expected number of days."
        )
        assert nessy_mean.variables["var1"]["cell_methods"] == "time: mean (interval: 1hr)", (
            "The cell_methods attribute was not set correctly after calculating the daily mean."
        )

    def test_all_t_steps(self, filepath):
        """
        Test the alltsteps type_op option for mean operation,
        which should calculate the mean across all time steps without grouping by day.
        """

        nessy_mean = load_fresh_nes(filepath)

        # sum along time dimension and divide by number of time steps)
        expected = nessy_mean.variables["var1"]["data"].sum(axis=0) / len(nessy_mean.time)
        try:
            nessy_mean.daily_statistic(op="mean", type_op="alltsteps")
        except Exception as e:
            pytest.fail(f"daily_statistic raised an exception: {e}")

        assert np.array_equal(
            nessy_mean.variables["var1"]["data"].flatten(),
            expected.flatten(),
        ), "The calculated daily mean does not match the expected values when using all time steps."
        assert nessy_mean.variables["var1"]["cell_methods"] == "time: mean (interval: 1hr)", (
            "The cell_methods attribute was not set correctly after calculating the daily mean."
        )

    def test_withoutt0(self, filepath):
        """
        Test the withoutt0 type_op option for mean operation,
        which should calculate the mean across all time steps except the first one.
        """
        nessy_mean = load_fresh_nes(filepath)

        # sum along time dimension and divide by number of time steps minus one since we avoid the first time step)
        expected = nessy_mean.variables["var1"]["data"][1:, :, :, :].sum(axis=0) / (len(nessy_mean.time) - 1)

        try:
            nessy_mean.daily_statistic(op="mean", type_op="withoutt0")
        except Exception as e:
            pytest.fail(f"daily_statistic raised an exception: {e}")

        assert np.array_equal(
            nessy_mean.variables["var1"]["data"].flatten(),
            expected.flatten(),
        ), "The calculated daily mean does not match the expected values when excluding the first time step."
        assert nessy_mean.variables["var1"]["cell_methods"] == "time: mean (interval: 1hr)", (
            "The cell_methods attribute was not set correctly after calculating the daily mean."
        )


class TestMin:
    """Test the daily_statistic method for min operation with different type_op options."""

    def test_calendar(self, filepath):
        """Test the calendar type_op option for min operation."""

        nessy = load_fresh_nes(filepath)
        # Manually calculate min for each day (3 days x 24 hours = 72 time steps)
        expected = (
            nessy.variables["var1"]["data"].reshape(3, 24, *nessy.variables["var1"]["data"].shape[1:]).min(axis=1)
        )
        expected_days = len(nessy.time) // 24

        try:
            nessy.daily_statistic(op="min", type_op="calendar")
        except Exception as e:
            pytest.fail(f"daily_statistic raised an exception: {e}")

        assert np.array_equal(nessy.variables["var1"]["data"], expected), (
            "The calculated daily minimum does not match the expected values."
        )
        assert np.shape(nessy.variables["var1"]["data"])[0] == expected_days, (
            "The time dimension of the output does not match the expected number of days."
        )
        assert nessy.variables["var1"]["cell_methods"] == "time: min (interval: 1hr)", (
            "The cell_methods attribute was not set correctly after calculating the daily minimum."
        )

    def test_calendar_25(self, filepath_25):
        """Test the calendar type_op option for min operation when there are 25 time steps (1 extra hour)."""

        nessy = load_fresh_nes(filepath_25)
        expected = (
            nessy.variables["var1"]["data"][:24].reshape(1, 24, *nessy.variables["var1"]["data"].shape[1:]).min(axis=1)
        )
        expected_days = len(nessy.time) // 24

        try:
            nessy.daily_statistic(op="min", type_op="calendar")
        except Exception as e:
            pytest.fail(f"daily_statistic raised an exception: {e}")

        assert np.array_equal(nessy.variables["var1"]["data"], expected), (
            "The calculated daily minimum does not match the expected values."
        )
        assert np.shape(nessy.variables["var1"]["data"])[0] == expected_days, (
            "The time dimension of the output does not match the expected number of days."
        )
        assert nessy.variables["var1"]["cell_methods"] == "time: min (interval: 1hr)", (
            "The cell_methods attribute was not set correctly after calculating the daily minimum."
        )

    def test_all_t_steps(self, filepath):
        """
        Test the alltsteps type_op option for min operation,
        which should calculate the minimum across all time steps without grouping by day.
        """

        nessy = load_fresh_nes(filepath)

        expected = nessy.variables["var1"]["data"].min(axis=0)

        try:
            nessy.daily_statistic(op="min", type_op="alltsteps")
        except Exception as e:
            pytest.fail(f"daily_statistic raised an exception: {e}")

        assert np.array_equal(
            nessy.variables["var1"]["data"].flatten(),
            expected.flatten(),
        ), "The calculated daily minimum does not match the expected values when using all time steps."
        assert nessy.variables["var1"]["cell_methods"] == "time: min (interval: 1hr)", (
            "The cell_methods attribute was not set correctly after calculating the daily minimum."
        )

    def test_withoutt0(self, filepath):
        """
        Test the withoutt0 type_op option for min operation,
        which should calculate the minimum across all time steps except the first one.
        """

        nessy = load_fresh_nes(filepath)

        expected = nessy.variables["var1"]["data"][1:, :, :, :].min(axis=0)

        nessy.daily_statistic(op="min", type_op="withoutt0")

        assert np.array_equal(
            nessy.variables["var1"]["data"].flatten(),
            expected.flatten(),
        ), "The calculated daily minimum does not match the expected values when excluding the first time step."
        assert nessy.variables["var1"]["cell_methods"] == "time: min (interval: 1hr)", (
            "The cell_methods attribute was not set correctly after calculating the daily minimum."
        )


class TestMax:
    """Test the daily_statistic method for max operation with different type_op options."""

    def test_calendar(self, filepath):
        """Test the calendar type_op option for max operation."""

        nessy = load_fresh_nes(filepath)
        # Manually calculate max for each day (3 days x 24 hours = 72 time steps)
        expected = (
            nessy.variables["var1"]["data"].reshape(3, 24, *nessy.variables["var1"]["data"].shape[1:]).max(axis=1)
        )
        expected_days = len(nessy.time) // 24

        try:
            nessy.daily_statistic(op="max", type_op="calendar")
        except Exception as e:
            pytest.fail(f"daily_statistic raised an exception: {e}")

        assert np.array_equal(nessy.variables["var1"]["data"], expected), (
            "The calculated daily maximum does not match the expected values."
        )
        assert np.shape(nessy.variables["var1"]["data"])[0] == expected_days, (
            "The time dimension of the output does not match the expected number of days."
        )
        assert nessy.variables["var1"]["cell_methods"] == "time: max (interval: 1hr)", (
            "The cell_methods attribute was not set correctly after calculating the daily maximum."
        )

    def test_calendar_25(self, filepath_25):
        """Test the calendar type_op option for max operation when there are 25 time steps (1 extra hour)."""

        nessy = load_fresh_nes(filepath_25)
        expected = (
            nessy.variables["var1"]["data"][:24].reshape(1, 24, *nessy.variables["var1"]["data"].shape[1:]).max(axis=1)
        )
        expected_days = len(nessy.time) // 24

        try:
            nessy.daily_statistic(op="max", type_op="calendar")
        except Exception as e:
            pytest.fail(f"daily_statistic raised an exception: {e}")

        assert np.array_equal(nessy.variables["var1"]["data"], expected), (
            "The calculated daily maximum does not match the expected values."
        )
        assert np.shape(nessy.variables["var1"]["data"])[0] == expected_days, (
            "The time dimension of the output does not match the expected number of days."
        )
        assert nessy.variables["var1"]["cell_methods"] == "time: max (interval: 1hr)", (
            "The cell_methods attribute was not set correctly after calculating the daily maximum."
        )

    def test_all_t_steps(self, filepath):
        """
        Test the alltsteps type_op option for max operation,
        which should calculate the maximum across all time steps without grouping by day.
        """

        nessy = load_fresh_nes(filepath)

        expected = nessy.variables["var1"]["data"].max(axis=0)

        try:
            nessy.daily_statistic(op="max", type_op="alltsteps")
        except Exception as e:
            pytest.fail(f"daily_statistic raised an exception: {e}")

        assert np.array_equal(
            nessy.variables["var1"]["data"].flatten(),
            expected.flatten(),
        ), "The calculated daily maximum does not match the expected values when using all time steps."
        assert nessy.variables["var1"]["cell_methods"] == "time: max (interval: 1hr)", (
            "The cell_methods attribute was not set correctly after calculating the daily maximum."
        )

    def test_withoutt0(self, filepath):
        """
        Test the withoutt0 type_op option for max operation,
        which should calculate the maximum across all time steps except the first one.
        """

        nessy = load_fresh_nes(filepath)

        expected = nessy.variables["var1"]["data"][1:, :, :, :].max(axis=0)

        try:
            nessy.daily_statistic(op="max", type_op="withoutt0")
        except Exception as e:
            pytest.fail(f"daily_statistic raised an exception: {e}")

        assert np.array_equal(
            nessy.variables["var1"]["data"].flatten(),
            expected.flatten(),
        ), "The calculated daily maximum does not match the expected values when excluding the first time step."
        assert nessy.variables["var1"]["cell_methods"] == "time: max (interval: 1hr)", (
            "The cell_methods attribute was not set correctly after calculating the daily maximum."
        )


class TestNanMean:
    """Test the daily_statistic method for nanmean operation with different type_op options."""

    def test_calendar(self, filepath_nan):
        """Test the calendar type_op option for nanmean operation."""

        nessy = load_fresh_nes(filepath_nan)
        expected = np.nanmean(
            nessy.variables["var1"]["data"].reshape(3, 24, *nessy.variables["var1"]["data"].shape[1:]),
            axis=1,
        )
        expected_days = len(nessy.time) // 24

        try:
            nessy.daily_statistic(op="nanmean", type_op="calendar")
        except Exception as e:
            pytest.fail(f"daily_statistic raised an exception: {e}")

        assert np.allclose(nessy.variables["var1"]["data"], expected, equal_nan=True), (
            "The calculated daily nanmean does not match the expected values."
        )
        assert np.shape(nessy.variables["var1"]["data"])[0] == expected_days, (
            "The time dimension of the output does not match the expected number of days."
        )
        assert nessy.variables["var1"]["cell_methods"] == "time: nanmean (interval: 1hr)", (
            "The cell_methods attribute was not set correctly after calculating the daily nanmean."
        )

    def test_calendar_25(self, filepath_nan_25):
        """Test the calendar type_op option for nanmean operation when there are 25 time steps (1 extra hour)."""

        nessy = load_fresh_nes(filepath_nan_25)
        expected = np.nanmean(
            nessy.variables["var1"]["data"][:24].reshape(1, 24, *nessy.variables["var1"]["data"].shape[1:]),
            axis=1,
        )
        expected_days = len(nessy.time) // 24

        try:
            nessy.daily_statistic(op="nanmean", type_op="calendar")
        except Exception as e:
            pytest.fail(f"daily_statistic raised an exception: {e}")

        assert np.allclose(nessy.variables["var1"]["data"], expected, equal_nan=True), (
            "The calculated daily nanmean does not match the expected values."
        )
        assert np.shape(nessy.variables["var1"]["data"])[0] == expected_days, (
            "The time dimension of the output does not match the expected number of days."
        )
        assert nessy.variables["var1"]["cell_methods"] == "time: nanmean (interval: 1hr)", (
            "The cell_methods attribute was not set correctly after calculating the daily nanmean."
        )

    def test_all_t_steps(self, filepath_nan):
        """
        Test the alltsteps type_op option for nanmean operation,
        which should calculate the nanmean across all time steps without grouping by day.
        """

        nessy = load_fresh_nes(filepath_nan)

        expected = np.nanmean(nessy.variables["var1"]["data"], axis=0)

        try:
            nessy.daily_statistic(op="nanmean", type_op="alltsteps")
        except Exception as e:
            pytest.fail(f"daily_statistic raised an exception: {e}")

        assert np.allclose(
            nessy.variables["var1"]["data"].flatten(),
            expected.flatten(),
            equal_nan=True,
        ), "The calculated daily nanmean does not match the expected values when using all time steps."
        assert nessy.variables["var1"]["cell_methods"] == "time: nanmean (interval: 1hr)", (
            "The cell_methods attribute was not set correctly after calculating the daily nanmean."
        )

    def test_withoutt0(self, filepath_nan):
        """
        Test the withoutt0 type_op option for nanmean operation,
        which should calculate the nanmean across all time steps except the first one.
        """

        nessy = load_fresh_nes(filepath_nan)

        expected = np.nanmean(nessy.variables["var1"]["data"][1:, :, :, :], axis=0)

        try:
            nessy.daily_statistic(op="nanmean", type_op="withoutt0")
        except Exception as e:
            pytest.fail(f"daily_statistic raised an exception: {e}")

        assert np.allclose(
            nessy.variables["var1"]["data"].flatten(),
            expected.flatten(),
            equal_nan=True,
        ), "The calculated daily nanmean does not match the expected values when excluding the first time step."
        assert nessy.variables["var1"]["cell_methods"] == "time: nanmean (interval: 1hr)", (
            "The cell_methods attribute was not set correctly after calculating the daily nanmean."
        )


class TestNanMin:
    """Test the daily_statistic method for nanmin operation with different type_op options."""

    def test_calendar(self, filepath_nan):
        """Test the calendar type_op option for nanmin operation."""

        nessy = load_fresh_nes(filepath_nan)
        expected = np.nanmin(
            nessy.variables["var1"]["data"].reshape(3, 24, *nessy.variables["var1"]["data"].shape[1:]),
            axis=1,
        )
        expected_days = len(nessy.time) // 24

        try:
            nessy.daily_statistic(op="nanmin", type_op="calendar")
        except Exception as e:
            pytest.fail(f"daily_statistic raised an exception: {e}")

        assert np.allclose(nessy.variables["var1"]["data"], expected, equal_nan=True), (
            "The calculated daily minimum does not match the expected values."
        )
        assert np.shape(nessy.variables["var1"]["data"])[0] == expected_days, (
            "The time dimension of the output does not match the expected number of days."
        )
        assert nessy.variables["var1"]["cell_methods"] == "time: nanmin (interval: 1hr)", (
            "The cell_methods attribute was not set correctly after calculating the daily minimum."
        )

    def test_calendar_25(self, filepath_nan_25):
        """Test the calendar type_op option for nanmin operation when there are 25 time steps (1 extra hour)."""

        nessy = load_fresh_nes(filepath_nan_25)
        expected = np.nanmin(
            nessy.variables["var1"]["data"][:24].reshape(1, 24, *nessy.variables["var1"]["data"].shape[1:]),
            axis=1,
        )
        expected_days = len(nessy.time) // 24

        try:
            nessy.daily_statistic(op="nanmin", type_op="calendar")
        except Exception as e:
            pytest.fail(f"daily_statistic raised an exception: {e}")

        assert np.allclose(nessy.variables["var1"]["data"], expected, equal_nan=True), (
            "The calculated daily minimum does not match the expected values."
        )
        assert np.shape(nessy.variables["var1"]["data"])[0] == expected_days, (
            "The time dimension of the output does not match the expected number of days."
        )
        assert nessy.variables["var1"]["cell_methods"] == "time: nanmin (interval: 1hr)", (
            "The cell_methods attribute was not set correctly after calculating the daily minimum."
        )

    def test_all_t_steps(self, filepath_nan):
        """
        Test the alltsteps type_op option for nanmin operation,
        which should calculate the nanmin across all time steps without grouping by day.
        """

        nessy = load_fresh_nes(filepath_nan)

        expected = np.nanmin(nessy.variables["var1"]["data"], axis=0)

        try:
            nessy.daily_statistic(op="nanmin", type_op="alltsteps")
        except Exception as e:
            pytest.fail(f"daily_statistic raised an exception: {e}")

        assert np.allclose(
            nessy.variables["var1"]["data"].flatten(),
            expected.flatten(),
            equal_nan=True,
        ), "The calculated daily minimum does not match the expected values when using all time steps."
        assert nessy.variables["var1"]["cell_methods"] == "time: nanmin (interval: 1hr)", (
            "The cell_methods attribute was not set correctly "
            "after calculating the daily minimum when using all time steps."
        )

    def test_withoutt0(self, filepath_nan):
        """
        Test the withoutt0 type_op option for nanmin operation,
        which should calculate the nanmin across all time steps except the first one.
        """

        nessy = load_fresh_nes(filepath_nan)

        expected = np.nanmin(nessy.variables["var1"]["data"][1:, :, :, :], axis=0)

        try:
            nessy.daily_statistic(op="nanmin", type_op="withoutt0")
        except Exception as e:
            pytest.fail(f"daily_statistic raised an exception: {e}")

        assert np.allclose(
            nessy.variables["var1"]["data"].flatten(),
            expected.flatten(),
            equal_nan=True,
        ), "The calculated daily minimum does not match the expected values when excluding the first time step."
        assert nessy.variables["var1"]["cell_methods"] == "time: nanmin (interval: 1hr)", (
            "The cell_methods attribute was not set correctly "
            "after calculating the daily minimum when excluding the first time step."
        )


class TestNanMax:
    """Test the daily_statistic method for nanmax operation with different type_op options."""

    def test_calendar(self, filepath_nan):
        """Test the calendar type_op option for nanmax operation."""

        nessy = load_fresh_nes(filepath_nan)
        expected = np.nanmax(
            nessy.variables["var1"]["data"].reshape(3, 24, *nessy.variables["var1"]["data"].shape[1:]),
            axis=1,
        )
        expected_days = len(nessy.time) // 24

        try:
            nessy.daily_statistic(op="nanmax", type_op="calendar")
        except Exception as e:
            pytest.fail(f"daily_statistic raised an exception: {e}")

        assert np.allclose(nessy.variables["var1"]["data"], expected, equal_nan=True), (
            "The calculated daily nanmax does not match the expected values."
        )
        assert np.shape(nessy.variables["var1"]["data"])[0] == expected_days, (
            "The time dimension of the output does not match the expected number of days."
        )
        assert nessy.variables["var1"]["cell_methods"] == "time: nanmax (interval: 1hr)", (
            "The cell_methods attribute was not set correctly after calculating the daily nanmax."
        )

    def test_calendar_25(self, filepath_nan_25):
        """Test the calendar type_op option for nanmax operation when there are 25 time steps (1 extra hour)."""

        nessy = load_fresh_nes(filepath_nan_25)
        expected = np.nanmax(
            nessy.variables["var1"]["data"][:24].reshape(1, 24, *nessy.variables["var1"]["data"].shape[1:]),
            axis=1,
        )
        expected_days = len(nessy.time) // 24

        try:
            nessy.daily_statistic(op="nanmax", type_op="calendar")
        except Exception as e:
            pytest.fail(f"daily_statistic raised an exception: {e}")

        assert np.allclose(nessy.variables["var1"]["data"], expected, equal_nan=True), (
            "The calculated daily nanmax does not match the expected values."
        )
        assert np.shape(nessy.variables["var1"]["data"])[0] == expected_days, (
            "The time dimension of the output does not match the expected number of days."
        )
        assert nessy.variables["var1"]["cell_methods"] == "time: nanmax (interval: 1hr)", (
            "The cell_methods attribute was not set correctly after calculating the daily nanmax."
        )

    def test_all_t_steps(self, filepath_nan):
        """
        Test the alltsteps type_op option for nanmax operation,
        which should calculate the nanmax across all time steps without grouping by day.
        """

        nessy = load_fresh_nes(filepath_nan)

        expected = np.nanmax(nessy.variables["var1"]["data"], axis=0)

        try:
            nessy.daily_statistic(op="nanmax", type_op="alltsteps")
        except Exception as e:
            pytest.fail(f"daily_statistic raised an exception: {e}")

        assert np.allclose(
            nessy.variables["var1"]["data"].flatten(),
            expected.flatten(),
            equal_nan=True,
        ), "The calculated daily nanmax does not match the expected values when using all time steps."
        assert nessy.variables["var1"]["cell_methods"] == "time: nanmax (interval: 1hr)", (
            "The cell_methods attribute was not set correctly "
            "after calculating the daily nanmax when using all time steps."
        )

    def test_withoutt0(self, filepath_nan):
        """
        Test the withoutt0 type_op option for nanmax operation,
        which should calculate the nanmax across all time steps except the first one.
        """

        nessy = load_fresh_nes(filepath_nan)

        expected = np.nanmax(nessy.variables["var1"]["data"][1:, :, :, :], axis=0)

        try:
            nessy.daily_statistic(op="nanmax", type_op="withoutt0")
        except Exception as e:
            pytest.fail(f"daily_statistic raised an exception: {e}")

        assert np.allclose(
            nessy.variables["var1"]["data"].flatten(),
            expected.flatten(),
            equal_nan=True,
        ), "The calculated daily nanmax does not match the expected values when excluding the first time step."
        assert nessy.variables["var1"]["cell_methods"] == "time: nanmax (interval: 1hr)", (
            "The cell_methods attribute was not set correctly after "
            "calculating the daily nanmax when excluding the first time step."
        )
