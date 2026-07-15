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

FILENAME = "lcc_nessy"


@pytest.fixture(scope="module")
def tmp_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("lcc")


@pytest.fixture(scope="module")
def filepath(tmp_dir):
    return str(tmp_dir / FILENAME)


def create_lcc_grid():
    """Helper function to create an LCC NES object for testing."""

    params = ProjectionParameters.LCC
    lcc_nessy = nes.create_nes(
        comm=None,
        info=False,
        projection="lcc",
        lat_1=params["lat_1"],
        lat_2=params["lat_2"],
        lon_0=params["lon_0"],
        lat_0=params["lat_0"],
        nx=params["nx"],
        ny=params["ny"],
        inc_x=params["inc_x"],
        inc_y=params["inc_y"],
        x_0=params["x_0"],
        y_0=params["y_0"],
        times=params["times"],
        first_level=params["levels"][0],
        last_level=params["levels"][-1],
    )

    lcc_nessy.variables = {
        "var1": {
            "data": SampleData.LCC["data"],
            "dimensions": SampleData.LCC["dimensions"],
            "units": SampleData.LCC["units"],
        }
    }
    return lcc_nessy


class TestLCCNes:
    """Tests for the LCCNes class and its methods."""

    p = ProjectionParameters.LCC

    def test_object_creation(self):
        """Test that an LCC NES object can be created without errors and has the correct type."""

        try:
            lcc_nessy = create_lcc_grid()
        except Exception as e:
            pytest.fail(f"Creation of LCC Nes object failed: {e}")

        assert lcc_nessy is not None, "LCC Nes object should not be None after creation."
        assert isinstance(lcc_nessy, nes.LCCNes), "Created object should be an instance of nes.LCCNes."

    def test_projection_data(self):
        """Test that the projection data of the LCC NES object matches the expected parameters."""

        lcc_nessy = create_lcc_grid()

        # Projection data checks
        projection_data = lcc_nessy.projection_data
        assert projection_data["grid_mapping_name"] == self.p["grid_mapping_name"], (
            f"Expected grid_mapping_name '{self.p['grid_mapping_name']}'"
            f" but got '{projection_data['grid_mapping_name']}'"
        )
        assert projection_data["standard_parallel"] == [
            self.p["lat_1"],
            self.p["lat_2"],
        ], "Standard parallels do not match expected values."
        assert projection_data["longitude_of_central_meridian"] == self.p["lon_0"], (
            "Longitude of central meridian does not match expected value."
        )
        assert projection_data["latitude_of_projection_origin"] == self.p["lat_0"], (
            "Latitude of projection origin does not match expected value."
        )

    def test_shape(self):
        """Test that the shapes of the coordinates and variable data match expected values."""

        lcc_nessy = create_lcc_grid()

        assert isinstance(lcc_nessy, nes.LCCNes)  # to suppress warnings
        assert (
            lcc_nessy.x is not None
            and lcc_nessy.y is not None
            and lcc_nessy.lat is not None
            and lcc_nessy.lon is not None
            and lcc_nessy.time is not None
            and lcc_nessy.lev is not None
        ), "One or more of the necessary attributes (x, y, lat, lon, time, lev) is None."

        assert np.shape(lcc_nessy.x["data"]) == (self.p["nx"],), "Shape of x coordinate does not match expected value."
        assert np.shape(lcc_nessy.y["data"]) == (self.p["ny"],), "Shape of y coordinate does not match expected value."
        assert np.shape(lcc_nessy.lat["data"]) == (self.p["ny"], self.p["nx"]), (
            "Shape of latitude data does not match expected value."
        )
        assert np.shape(lcc_nessy.lon["data"]) == (self.p["ny"], self.p["nx"]), (
            "Shape of longitude data does not match expected value."
        )
        assert len(lcc_nessy.time) == len(self.p["times"]), "Length of time dimension does not match expected value."
        assert np.shape(lcc_nessy.lev["data"]) == (len(self.p["levels"]),), (
            "Shape of level data does not match expected value."
        )
        assert np.shape(lcc_nessy.variables["var1"]["data"]) == (
            len(self.p["times"]),
            len(self.p["levels"]),
            self.p["ny"],
            self.p["nx"],
        ), "Shape of variable data does not match expected value."

    def test_variable_data(self):
        """Test that the variable data in the LCC NES object matches the expected values."""

        lcc_nessy = create_lcc_grid()
        var1 = lcc_nessy.variables["var1"]

        assert var1 is not None
        assert np.array_equal(var1["data"], SampleData.LCC["data"]), "Variable data does not match expected values."
        assert var1["units"] == SampleData.LCC["units"], "Variable units do not match expected values."
        assert var1["dimensions"] == SampleData.LCC["dimensions"], "Variable dimensions do not match expected values."

    def test_create_shapefile(self, filepath):
        """Test that the create_shapefile method creates shapefiles without errors and that the files are created."""

        lcc_nessy = create_lcc_grid()

        try:
            lcc_nessy.create_shapefile()
        except Exception as e:
            pytest.fail(f"Creation of shapefile failed with exception: {e}")

        try:
            lcc_nessy.to_shapefile(f"{filepath}.shp", time=self.p["times"][0], lev=0)
        except Exception as e:
            pytest.fail(f"Saving of shapefile failed with exception: {e}")

        try:
            lcc_nessy.to_shapefile(f"{filepath}.geojson", time=self.p["times"][0], lev=0)
        except Exception as e:
            pytest.fail(f"Saving of shapefile (.geojson) failed with exception: {e}")

        for ext in EXTENSIONS:
            assert os.path.exists(f"{filepath}{ext}"), f"Expected file {filepath}{ext} was not created."

    def test_read_write(self, filepath):
        """Test that writing the LCC NES to NetCDF and reading it back produces an equivalent NES object."""

        lcc_nessy = create_lcc_grid()

        # Writing
        try:
            lcc_nessy.to_netcdf(f"{filepath}.nc")
        except Exception as e:
            pytest.fail(f"Writing to NetCDF failed with exception: {e}")

        assert os.path.exists(f"{filepath}.nc"), "NetCDF file was not created after writing."

        # Reading
        try:
            lcc_nessy_read = nes.open_netcdf(f"{filepath}.nc")
        except Exception as e:
            pytest.fail(f"Reading from NetCDF failed with exception: {e}")

        # Loading
        try:
            lcc_nessy_read.load()
        except Exception as e:
            pytest.fail(f"Loading NES data failed with exception: {e}")

        # Suppress warnings
        assert lcc_nessy_read is not None
        assert isinstance(lcc_nessy, nes.LCCNes)

        # Check correct projection type
        assert isinstance(lcc_nessy_read, nes.LCCNes), "Read NES object should be an instance of nes.LCCNes."

        # Check that all necessary attributes are present
        assert lcc_nessy.projection_data and lcc_nessy_read.projection_data is not None, "Projection data is missing."
        assert lcc_nessy.x and lcc_nessy_read.x is not None, "X coordinate data is missing."
        assert lcc_nessy.y and lcc_nessy_read.y is not None, "Y coordinate data is missing."
        assert lcc_nessy.lat and lcc_nessy_read.lat is not None, "Latitude data is missing."
        assert lcc_nessy.lon and lcc_nessy_read.lon is not None, "Longitude data is missing."
        assert lcc_nessy.time and lcc_nessy_read.time is not None, "Time data is missing."
        assert lcc_nessy.lev and lcc_nessy_read.lev is not None, "Level data is missing."
        assert lcc_nessy.variables and lcc_nessy_read.variables is not None, "Variable data is missing."

        # Equivalence checks
        projection_data = lcc_nessy.projection_data
        projection_data_read = lcc_nessy_read.projection_data
        assert projection_data["grid_mapping_name"] == projection_data_read["grid_mapping_name"], (
            "Grid mapping names do not match after reading from NetCDF."
        )
        assert (
            projection_data["longitude_of_central_meridian"] == projection_data_read["longitude_of_central_meridian"]
        ), "Longitude of central meridian does not match after reading from NetCDF."
        assert (
            projection_data["latitude_of_projection_origin"] == projection_data_read["latitude_of_projection_origin"]
        ), "Latitude of projection origin does not match after reading from NetCDF."
        assert np.array_equal(
            projection_data["standard_parallel"],
            projection_data_read["standard_parallel"],
        ), "Standard parallels do not match after reading from NetCDF."
        assert np.array_equal(lcc_nessy.x["data"], lcc_nessy_read.x["data"]), (
            "X coordinates do not match after reading from NetCDF."
        )
        assert np.array_equal(lcc_nessy.y["data"], lcc_nessy_read.y["data"]), (
            "Y coordinates do not match after reading from NetCDF."
        )
        assert np.array_equal(lcc_nessy.lat["data"], lcc_nessy_read.lat["data"]), (
            "Latitude data does not match after reading from NetCDF."
        )
        assert np.array_equal(lcc_nessy.lon["data"], lcc_nessy_read.lon["data"]), (
            "Longitude data does not match after reading from NetCDF."
        )
        assert lcc_nessy.time == lcc_nessy_read.time, "Time data does not match after reading from NetCDF."
        assert np.array_equal(lcc_nessy.lev["data"], lcc_nessy_read.lev["data"]), (
            "Level data does not match after reading from NetCDF."
        )
        assert np.array_equal(
            lcc_nessy.variables["var1"]["data"],
            lcc_nessy_read.variables["var1"]["data"],
        ), "Variable data does not match after reading from NetCDF."
        assert lcc_nessy.variables["var1"]["dimensions"] == lcc_nessy_read.variables["var1"]["dimensions"], (
            "Variable dimensions do not match after reading from NetCDF."
        )
        assert lcc_nessy.variables["var1"]["units"] == lcc_nessy_read.variables["var1"]["units"], (
            "Variable units do not match after reading from NetCDF."
        )
