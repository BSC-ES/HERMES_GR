# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import datetime

import netCDF4 as nc4
import numpy as np
import pytest

import nes
from nes.tests.utils import DummyData

FILENAME_IN = "dummy_nessy.nc"
FILENAME_OUT = "dummy_nessy_out.nc"


@pytest.fixture(scope="module")
def tmp_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("raw_io")


@pytest.fixture(scope="module")
def filepath_in(tmp_dir):
    return str(tmp_dir / FILENAME_IN)


@pytest.fixture(scope="module")
def filepath_out(tmp_dir):
    return str(tmp_dir / FILENAME_OUT)


def create_dummy_nessy(filepath_in):
    """Create a dummy NetCDF file using raw NetCDF4 API."""

    dataset = nc4.Dataset(filepath_in, "w", format="NETCDF4")

    DIMS = DummyData.DIMS
    TYPES = DummyData.TYPES
    UNITS = DummyData.UNITS
    DATA_SHAPE = DummyData.DATA_SHAPE

    # Create dimensions
    dataset.createDimension("lon", DIMS["lon"])
    dataset.createDimension("lat", DIMS["lat"])
    dataset.createDimension("lev", DIMS["lev"])
    dataset.createDimension("time", DIMS["time"])

    # Create variables
    lon = dataset.createVariable("lon", TYPES["lon"], ("lon",))
    lat = dataset.createVariable("lat", TYPES["lat"], ("lat",))
    lev = dataset.createVariable("lev", TYPES["lev"], ("lev",))
    time = dataset.createVariable("time", TYPES["time"], ("time",))
    variable = dataset.createVariable(
        "variable",
        TYPES["variable"],
        (
            "time",
            "lev",
            "lat",
            "lon",
        ),
    )

    # Populate variables with dummy data
    lon[:] = np.linspace(1, DIMS["lon"], DIMS["lon"])
    lat[:] = np.linspace(1, DIMS["lat"], DIMS["lat"])
    lev[:] = np.array([0], dtype=TYPES["lev"])
    time[:] = np.array([0], dtype=TYPES["time"])  # represents 1970-01-01 00:00:00
    variable[:, :, :, :] = np.random.rand(DATA_SHAPE[0], DATA_SHAPE[1], DATA_SHAPE[2], DATA_SHAPE[3])

    # Assign variable units
    lon.units = UNITS["lon"]
    lat.units = UNITS["lat"]
    lev.units = UNITS["lev"]
    time.units = UNITS["time"]
    variable.units = UNITS["variable"]
    dataset.close()


@pytest.fixture(params=["read", "write_read"])
def nessy(request, filepath_in, filepath_out):
    """Fixture that provides both the original read nessy and the write-read nessy."""

    try:
        create_dummy_nessy(filepath_in)
    except Exception as e:
        pytest.fail(f"Failed to create dummy NES file with error: {e}")

    try:
        nessy = nes.open_netcdf(filepath_in)
    except Exception as e:
        pytest.fail(f"Failed to open dummy NES file with error: {e}")

    try:
        nessy.load()
    except Exception as e:
        pytest.fail(f"Failed to load dummy NES data with error: {e}")

    if request.param == "read":
        return nessy
    elif request.param == "write_read":
        try:
            nessy.to_netcdf(filepath_out)
        except Exception as e:
            pytest.fail(f"Failed to write dummy NES data with error: {e}")
        try:
            nessy_out = nes.open_netcdf(filepath_out)
        except Exception as e:
            pytest.fail(f"Failed to open written NES file with error: {e}")
        try:
            nessy_out.load()
        except Exception as e:
            pytest.fail(f"Failed to load written NES data with error: {e}")
        return nessy_out
    else:
        pytest.fail(f"Invalid fixture parameter: {request.param}")


class TestRawIO:
    """Test class that runs tests on both read and write-read nessy objects."""

    DIMS = DummyData.DIMS
    TYPES = DummyData.TYPES
    UNITS = DummyData.UNITS
    DATA_SHAPE = DummyData.DATA_SHAPE

    def test_projection_data(self, nessy):
        """Test that projection data is correctly read from the file."""

        assert nessy.projection_data["grid_mapping_name"] == "latitude_longitude", (
            "Projection data should contain correct grid_mapping_name."
        )

    def test_dimensions(self, nessy):
        """Test that dimensions are correctly read from the file."""

        assert nessy.lat is not None and nessy.lon is not None and nessy.lev is not None, (
            "Latitude, longitude, and level variables should not be None."
        )
        assert np.shape(nessy.lat["data"]) == (self.DIMS["lat"],), "Latitude data should have correct shape."
        assert np.shape(nessy.lon["data"]) == (self.DIMS["lon"],), "Longitude data should have correct shape."
        assert np.shape(nessy.lev["data"]) == (self.DIMS["lev"],), "Level data should have correct shape."
        assert len(nessy.time) == self.DIMS["time"], "Time data should have correct length."
        assert nessy.variables["variable"]["data"].shape == self.DATA_SHAPE, "Variable data should have correct shape."

    def test_datatypes(self, nessy):
        """Test that datatypes are correctly read from the file."""

        # dtype label
        assert nessy.lat["dtype"] == np.dtype(self.TYPES["lat"]), "Latitude variable should have correct dtype label."
        assert nessy.lon["dtype"] == np.dtype(self.TYPES["lon"]), "Longitude variable should have correct dtype label."
        assert nessy.lev["dtype"] == np.dtype(self.TYPES["lev"]), "Level variable should have correct dtype label."
        assert nessy.variables["variable"]["data"].dtype == np.dtype(self.TYPES["variable"]), (
            "Variable should have correct dtype label."
        )

        # data dtype label
        assert nessy.lat["data"].dtype == np.dtype(self.TYPES["lat"]), "Latitude data should have correct dtype."
        assert nessy.lon["data"].dtype == np.dtype(self.TYPES["lon"]), "Longitude data should have correct dtype."
        assert nessy.lev["data"].dtype == np.dtype(self.TYPES["lev"]), "Level data should have correct dtype."
        assert nessy.variables["variable"]["data"].dtype == np.dtype(self.TYPES["variable"]), (
            "Variable data should have correct dtype."
        )

        # actual data type
        assert isinstance(nessy.lat["data"][0], np.floating), "Latitude data should be of floating point type."
        assert isinstance(nessy.lon["data"][0], np.floating), "Longitude data should be of floating point type."
        assert isinstance(nessy.lev["data"][0], np.integer), "Level data should be of integer type."
        assert isinstance(nessy.time[0], datetime.datetime), "Time data should be of datetime type."

    def test_units(self, nessy):
        """Test that units are correctly read from the file."""

        assert nessy.lat["units"] == self.UNITS["lat"], "Latitude variable should have correct units."
        assert nessy.lon["units"] == self.UNITS["lon"], "Longitude variable should have correct units."
        assert nessy.lev["units"] == self.UNITS["lev"], "Level variable should have correct units."
        assert nessy.variables["variable"]["units"] == self.UNITS["variable"], "Variable should have correct units."

    def test_write_read_equality(self, filepath_in, filepath_out):
        """Test that write-read produces identical data to the original."""

        try:
            create_dummy_nessy(filepath_in)
        except Exception as e:
            pytest.fail(f"Failed to create dummy NES file with error: {e}")

        try:
            nessy = nes.open_netcdf(filepath_in)
        except Exception as e:
            pytest.fail(f"Failed to open dummy NES file with error: {e}")
        try:
            nessy.load()
        except Exception as e:
            pytest.fail(f"Failed to load NES file with error: {e}")

        try:
            nessy.to_netcdf(filepath_out)
        except Exception as e:
            pytest.fail(f"Failed to write NES file with error: {e}")
        try:
            nessy_out = nes.open_netcdf(filepath_out)
        except Exception as e:
            pytest.fail(f"Failed to open output NES file with error: {e}")
        try:
            nessy_out.load()
        except Exception as e:
            pytest.fail(f"Failed to load output NES file with error: {e}")

        assert nessy_out is not None, "Output NES should not be None after write-read."
        assert nessy_out.lat is not None and nessy_out.lon is not None and nessy_out.lev is not None, (
            "Latitude, longitude, and level variables in output NES should not be None."
        )
        assert nessy_out.time is not None, "Time variable in output NES should not be None."

        assert nessy is not None, "Original NES should not be None after write-read."
        assert nessy.lat is not None and nessy.lon is not None and nessy.lev is not None, (
            "Latitude, longitude, and level variables in original NES should not be None."
        )
        assert nessy.time is not None, "Time variable in original NES should not be None."

        # Compare data
        assert np.array_equal(nessy.lat["data"], nessy_out.lat["data"]), (
            "Latitude data should be equal after write-read."
        )
        assert np.array_equal(nessy.lon["data"], nessy_out.lon["data"]), (
            "Longitude data should be equal after write-read."
        )
        assert np.array_equal(nessy.lev["data"], nessy_out.lev["data"]), "Level data should be equal after write-read."
        assert nessy.time == nessy_out.time, "Time data should be equal after write-read."
        assert np.array_equal(
            nessy.variables["variable"]["data"],
            nessy_out.variables["variable"]["data"],
        ), "Variable data should be equal after write-read."
