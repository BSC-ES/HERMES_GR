# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import os

import numpy as np
import pytest

import nes
from nes.tests.utils import EXTENSIONS, ProjectionParameters, SampleData

FILENAME = "global_nessy"


@pytest.fixture(scope="module")
def tmp_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("global")


@pytest.fixture(scope="module")
def filepath(tmp_dir):
    return str(tmp_dir / FILENAME)


def create_global_grid():
    """Helper function to create a global grid NES object with sample data."""

    params = ProjectionParameters.GLOBAL
    global_nessy = nes.create_nes(
        comm=None,
        info=False,
        projection="global",
        inc_lat=params["inc_lat"],
        inc_lon=params["inc_lon"],
        times=params["times"],
        first_level=params["levels"][0],
        last_level=params["levels"][-1],
    )

    global_nessy.variables = {
        "var1": {
            "data": SampleData.GLOBAL["data"],
            "dimensions": SampleData.GLOBAL["dimensions"],
            "units": SampleData.GLOBAL["units"],
        }
    }
    return global_nessy


class TestGlobalNes:
    """Tests for creating and working with a global grid NES object."""

    p = ProjectionParameters.GLOBAL

    def test_object_creation(self):
        """Test that a global grid NES object can be created without errors and has the expected attributes."""

        try:
            global_nessy = create_global_grid()
        except Exception as e:
            pytest.fail(f"Creation of Global Nes object failed: {e}")

        assert global_nessy is not None, "Global Nes object should not be None after creation."
        assert isinstance(global_nessy, nes.LatLonNes), "Created object should be an instance of nes.LatLonNes."

    def test_projection_data(self):
        """Test that the projection data of the global grid NES object matches expected values."""

        global_nessy = create_global_grid()

        # Projection data checks
        projection_data = global_nessy.projection_data
        assert projection_data["grid_mapping_name"] == self.p["grid_mapping_name"], (
            "Grid mapping name does not match expected value."
        )
        assert projection_data["inc_lat"] == self.p["inc_lat"], "Latitude increment does not match expected value."
        assert projection_data["inc_lon"] == self.p["inc_lon"], "Longitude increment does not match expected value."
        assert projection_data["lat_orig"] == self.p["lat_orig"], "Latitude origin does not match expected value."
        assert projection_data["lon_orig"] == self.p["lon_orig"], "Longitude origin does not match expected value."
        assert projection_data["n_lat"] == np.shape(SampleData.GLOBAL["data"])[2], (
            "Number of latitude points does not match expected value."
        )
        assert projection_data["n_lon"] == np.shape(SampleData.GLOBAL["data"])[3], (
            "Number of longitude points does not match expected value."
        )

    def test_shape(self):
        """Test that the shape of the latitude, longitude, time, level, and variable data matches expected values."""

        global_nessy = create_global_grid()

        assert isinstance(global_nessy, nes.LatLonNes)
        assert (
            global_nessy.lat is not None
            and global_nessy.lon is not None
            and global_nessy.time is not None
            and global_nessy.lev is not None
        ), "NES object is missing one or more of the required attributes: lat, lon, time, lev."

        assert np.shape(global_nessy.lat["data"]) == (np.shape(SampleData.GLOBAL["data"])[2],), (
            "Latitude data shape does not match expected shape."
        )
        assert np.shape(global_nessy.lon["data"]) == (np.shape(SampleData.GLOBAL["data"])[3],), (
            "Longitude data shape does not match expected shape."
        )
        assert len(global_nessy.time) == (len(self.p["times"])), "Time dimension length does not match expected length."
        assert np.shape(global_nessy.lev["data"]) == (len(self.p["levels"]),), (
            "Level data shape does not match expected shape."
        )
        assert global_nessy.variables["var1"]["data"].shape == (
            len(self.p["times"]),
            len(self.p["levels"]),
            np.shape(SampleData.GLOBAL["data"])[2],
            np.shape(SampleData.GLOBAL["data"])[3],
        ), "Variable 'var1' data shape does not match expected shape."

    def test_variable_data(self):
        """Test that the variable data in the global grid NES object matches expected values."""

        global_nessy = create_global_grid()
        var1 = global_nessy.variables["var1"]

        assert var1 is not None, "Variable 'var1' should be present in the NES object."
        assert np.array_equal(var1["data"], SampleData.GLOBAL["data"]), (
            "Variable 'var1' data does not match expected values."
        )
        assert var1["units"] == SampleData.GLOBAL["units"], "Variable 'var1' units do not match expected values."
        assert var1["dimensions"] == SampleData.GLOBAL["dimensions"], (
            "Variable 'var1' dimensions do not match expected values."
        )

    def test_create_shapefile(self, filepath):
        """
        Test that the create_shapefile method creates shapefile
        and geojson without errors and that the files are created.
        """

        global_nessy = create_global_grid()

        # Create shapefile and geojson, check for exceptions
        try:
            global_nessy.create_shapefile()
        except Exception as e:
            pytest.fail(f"Creation of shapefile failed with exception: {e}")

        try:
            global_nessy.to_shapefile(f"{filepath}.shp", time=self.p["times"][0], lev=0)
        except Exception as e:
            pytest.fail(f"Saving of shapefile (.shp) failed with exception: {e}")

        try:
            global_nessy.to_shapefile(f"{filepath}.geojson", time=self.p["times"][0], lev=0)
        except Exception as e:
            pytest.fail(f"Saving of shapefile (.geojson) failed with exception: {e}")

        for ext in EXTENSIONS:
            assert os.path.exists(f"{filepath}{ext}"), f"Expected file {filepath}{ext} was not created."

    def test_write_and_read(self, filepath):
        """Test that writing the global grid NES to NetCDF
        and reading it back produces an equivalent NES object.
        """

        # Creation is already tested in first test
        global_nessy = create_global_grid()

        # Writing
        try:
            global_nessy.to_netcdf(f"{filepath}.nc")
            assert os.path.exists(f"{filepath}.nc")
        except Exception as e:
            pytest.fail(f"Writing to NetCDF failed with exception: {e}")

        # Reading
        try:
            global_nessy_read = nes.open_netcdf(f"{filepath}.nc")
        except Exception as e:
            pytest.fail(f"Reading from NetCDF failed with exception: {e}")

        # Loading
        try:
            global_nessy_read.load()
        except Exception as e:
            pytest.fail(f"Loading NES data failed with exception: {e}")

        assert global_nessy_read is not None, "Global Nes object should not be None after reading from NetCDF."
        assert isinstance(global_nessy, nes.LatLonNes), "Created object should be an instance of nes.LatLonNes."

        # Check that all necessary attributes are present
        assert global_nessy.projection_data and global_nessy_read.projection_data is not None, (
            "Projection data should be present in both original and read NES objects."
        )
        assert global_nessy.lat and global_nessy_read.lat is not None, (
            "Latitude data should be present in both original and read NES objects."
        )
        assert global_nessy.lon and global_nessy_read.lon is not None, (
            "Longitude data should be present in both original and read NES objects."
        )
        assert global_nessy.time and global_nessy_read.time is not None, (
            "Time data should be present in both original and read NES objects."
        )
        assert global_nessy.lev and global_nessy_read.lev is not None, (
            "Level data should be present in both original and read NES objects."
        )
        assert global_nessy.variables and global_nessy_read.variables is not None, (
            "Variables should be present in both original and read NES objects."
        )

        # Check that the grid mapping name matches after read
        projection_data = global_nessy.projection_data
        projection_data_read = global_nessy_read.projection_data

        # Equivalence checks
        assert projection_data["grid_mapping_name"] == projection_data_read["grid_mapping_name"], (
            "Grid mapping name does not match after reading from NetCDF."
        )
        assert projection_data["semi_major_axis"] == projection_data_read["semi_major_axis"], (
            "Semi-major axis does not match after reading from NetCDF."
        )
        assert projection_data["inverse_flattening"] == projection_data_read["inverse_flattening"], (
            "Inverse flattening does not match after reading from NetCDF."
        )
        assert np.array_equal(global_nessy.lat["data"], global_nessy_read.lat["data"]), (
            "Latitude data does not match after reading from NetCDF."
        )
        assert np.array_equal(global_nessy.lon["data"], global_nessy_read.lon["data"]), (
            "Longitude data does not match after reading from NetCDF."
        )
        assert global_nessy.time == global_nessy_read.time, "Time data does not match after reading from NetCDF."
        assert np.array_equal(global_nessy.lev["data"], global_nessy_read.lev["data"]), (
            "Level data does not match after reading from NetCDF."
        )
        assert np.array_equal(
            global_nessy.variables["var1"]["data"],
            global_nessy_read.variables["var1"]["data"],
        ), "Variable 'var1' data does not match after reading from NetCDF."
        assert global_nessy.variables["var1"]["dimensions"] == global_nessy_read.variables["var1"]["dimensions"], (
            "Variable 'var1' dimensions do not match after reading from NetCDF."
        )
        assert global_nessy.variables["var1"]["units"] == global_nessy_read.variables["var1"]["units"], (
            "Variable 'var1' units do not match after reading from NetCDF."
        )
