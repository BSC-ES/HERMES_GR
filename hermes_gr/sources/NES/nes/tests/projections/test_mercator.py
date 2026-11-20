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

FILENAME = "mercator_nessy"


@pytest.fixture(scope="module")
def tmp_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("mercator")


@pytest.fixture(scope="module")
def filepath(tmp_dir):
    return str(tmp_dir / FILENAME)


def create_mercator_grid():
    """Helper function to create a Mercator NES object for testing."""

    params = ProjectionParameters.MERCATOR
    mercator_nessy = nes.create_nes(
        comm=None,
        info=False,
        projection=params["grid_mapping_name"],
        lon_0=params["lon_0"],
        lat_ts=params["lat_ts"],
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

    mercator_nessy.variables = {
        "var1": {
            "data": SampleData.MERCATOR["data"],
            "dimensions": SampleData.MERCATOR["dimensions"],
            "units": SampleData.MERCATOR["units"],
        }
    }

    return mercator_nessy


class TestMercatorNes:
    """Tests for the MercatorNes class and its methods."""

    p = ProjectionParameters.MERCATOR
    testData = SampleData.MERCATOR

    def test_object_creation(self):
        """Test that a Mercator NES object can be created without errors and has the expected attributes."""

        try:
            mercator_nessy = create_mercator_grid()
        except Exception as e:
            pytest.fail(f"Creation of Mercator Nes object failed: {e}")

        assert mercator_nessy is not None, "Mercator Nes object should not be None after creation."
        assert isinstance(mercator_nessy, nes.MercatorNes), "Created object should be an instance of nes.MercatorNes."

    def test_projection_data(self):
        """Test that the projection data in the Mercator NES object matches the expected values."""

        mercator_nessy = create_mercator_grid()
        projection_data = mercator_nessy.projection_data

        # Projection data checks
        assert projection_data["grid_mapping_name"] == self.p["grid_mapping_name"], (
            "Grid mapping name does not match expected value."
        )
        assert projection_data["standard_parallel"] == self.p["lat_ts"], (
            "Standard parallel does not match expected value."
        )
        assert projection_data["longitude_of_projection_origin"] == self.p["lon_0"], (
            "Longitude of projection origin does not match expected value."
        )
        assert projection_data["x_0"] == self.p["x_0"], "X origin does not match expected value."
        assert projection_data["y_0"] == self.p["y_0"], "Y origin does not match expected value."
        assert projection_data["inc_x"] == self.p["inc_x"], "X increment does not match expected value."
        assert projection_data["inc_y"] == self.p["inc_y"], "Y increment does not match expected value."
        assert projection_data["nx"] == self.p["nx"], "Number of x points does not match expected value."
        assert projection_data["ny"] == self.p["ny"], "Number of y points does not match expected value."

    def test_shape(self):
        """
        Test that the shapes of the coordinates and variable
        data in the Mercator NES object match expected values.
        """

        mercator_nessy = create_mercator_grid()

        assert isinstance(mercator_nessy, nes.MercatorNes)  # suppress warnings about type checking
        assert (
            mercator_nessy.x is not None
            and mercator_nessy.y is not None
            and mercator_nessy.lat is not None
            and mercator_nessy.lon is not None
            and mercator_nessy.time is not None
            and mercator_nessy.lev is not None
        ), "One or more of the necessary attributes (x, y, lat, lon, time, lev) are None."

        assert np.shape(mercator_nessy.x["data"]) == (self.p["nx"],), (
            "Shape of x coordinate does not match expected value."
        )
        assert np.shape(mercator_nessy.y["data"]) == (self.p["ny"],), (
            "Shape of y coordinate does not match expected value."
        )
        assert np.shape(mercator_nessy.lat["data"]) == (self.p["ny"], self.p["nx"]), (
            "Shape of latitude coordinate does not match expected value."
        )
        assert np.shape(mercator_nessy.lon["data"]) == (self.p["ny"], self.p["nx"]), (
            "Shape of longitude coordinate does not match expected value."
        )
        assert len(mercator_nessy.time) == len(self.p["times"]), (
            "Length of time dimension does not match expected value."
        )
        assert np.shape(mercator_nessy.lev["data"]) == (len(self.p["levels"]),), (
            "Shape of level coordinate does not match expected value."
        )
        assert np.shape(mercator_nessy.variables["var1"]["data"]) == (
            len(self.p["times"]),
            len(self.p["levels"]),
            self.p["ny"],
            self.p["nx"],
        ), "Shape of variable data does not match expected value."

    def test_variable_data(self):
        """Test that the variable data in the Mercator NES object matches the expected values."""

        mercator_nessy = create_mercator_grid()
        var1 = mercator_nessy.variables["var1"]

        assert var1 is not None, "Variable 'var1' should be present in the NES object."
        assert np.array_equal(var1["data"], SampleData.MERCATOR["data"]), (
            "Variable 'var1' data does not match expected values."
        )
        assert var1["units"] == SampleData.MERCATOR["units"], "Variable 'var1' units do not match expected values."
        assert var1["dimensions"] == SampleData.MERCATOR["dimensions"], (
            "Variable 'var1' dimensions do not match expected values."
        )

    def test_create_shapefile(self, filepath):
        """Test that the create_shapefile method creates shapefiles without errors and that the files are created."""

        mercator_nessy = create_mercator_grid()

        try:
            mercator_nessy.create_shapefile()
        except Exception as e:
            pytest.fail(f"Creation of shapefile failed with exception: {e}")

        try:
            mercator_nessy.to_shapefile(f"{filepath}.shp", time=self.p["times"][0], lev=0)
        except Exception as e:
            pytest.fail(f"Saving of shapefile failed with exception: {e}")

        try:
            mercator_nessy.to_shapefile(f"{filepath}.geojson", time=self.p["times"][0], lev=0)
        except Exception as e:
            pytest.fail(f"Saving of shapefile (.geojson) failed with exception: {e}")

        for ext in EXTENSIONS:
            assert os.path.exists(f"{filepath}{ext}"), f"Expected file {filepath}{ext} was not created."

    def test_read_write(self, filepath):
        """Test that writing the Mercator NES to NetCDF and reading it back produces an equivalent NES object."""

        mercator_nessy = create_mercator_grid()

        # Writing
        try:
            mercator_nessy.to_netcdf(f"{filepath}.nc")
        except Exception as e:
            pytest.fail(f"Writing to NetCDF failed with exception: {e}")

        assert os.path.exists(f"{filepath}.nc"), f"Expected file {filepath}.nc was not created."

        # Reading
        try:
            mercator_nessy_read = nes.open_netcdf(f"{filepath}.nc")
        except Exception as e:
            pytest.fail(f"Reading from NetCDF failed with exception: {e}")

        # Loading
        try:
            mercator_nessy_read.load()
        except Exception as e:
            pytest.fail(f"Loading NES data failed with exception: {e}")

        # Suppress warnings
        assert mercator_nessy_read is not None
        assert isinstance(mercator_nessy, nes.MercatorNes)

        # Check correct projection type
        assert isinstance(mercator_nessy_read, nes.MercatorNes), (
            "Read NES object should be an instance of nes.MercatorNes."
        )

        # Check that all necessary attributes are present
        assert mercator_nessy.projection_data and mercator_nessy_read.projection_data is not None, (
            "Projection data is missing."
        )
        assert mercator_nessy.x and mercator_nessy_read.x is not None, "X coordinate data is missing."
        assert mercator_nessy.y and mercator_nessy_read.y is not None, "Y coordinate data is missing."
        assert mercator_nessy.lat and mercator_nessy_read.lat is not None, "Latitude data is missing."
        assert mercator_nessy.lon and mercator_nessy_read.lon is not None, "Longitude data is missing."
        assert mercator_nessy.time and mercator_nessy_read.time is not None, "Time data is missing."
        assert mercator_nessy.lev and mercator_nessy_read.lev is not None, "Level data is missing."
        assert mercator_nessy.variables and mercator_nessy_read.variables is not None, "Variable data is missing."

        # Projection data
        projection_data = mercator_nessy.projection_data
        projection_data_read = mercator_nessy_read.projection_data

        # Equivalence checks
        assert projection_data["grid_mapping_name"] == projection_data_read["grid_mapping_name"], (
            "Grid mapping name does not match between original and read NES objects."
        )
        assert projection_data["standard_parallel"] == projection_data_read["standard_parallel"], (
            "Standard parallel does not match between original and read NES objects."
        )
        assert (
            projection_data["longitude_of_projection_origin"] == projection_data_read["longitude_of_projection_origin"]
        ), "Longitude of projection origin does not match between original and read NES objects."
        assert np.array_equal(mercator_nessy.x["data"], mercator_nessy_read.x["data"]), (
            "X coordinate data does not match between original and read NES objects."
        )
        assert np.array_equal(mercator_nessy.y["data"], mercator_nessy_read.y["data"]), (
            "Y coordinate data does not match between original and read NES objects."
        )
        assert np.array_equal(mercator_nessy.lat["data"], mercator_nessy_read.lat["data"]), (
            "Latitude data does not match between original and read NES objects."
        )
        assert np.array_equal(mercator_nessy.lon["data"], mercator_nessy_read.lon["data"]), (
            "Longitude data does not match between original and read NES objects."
        )
        assert mercator_nessy.time == mercator_nessy_read.time, (
            "Time data does not match between original and read NES objects."
        )
        assert np.array_equal(mercator_nessy.lev["data"], mercator_nessy_read.lev["data"]), (
            "Level data does not match between original and read NES objects."
        )
        assert np.array_equal(
            mercator_nessy.variables["var1"]["data"],
            mercator_nessy_read.variables["var1"]["data"],
        ), "Variable 'var1' data does not match between original and read NES objects."
        assert mercator_nessy.variables["var1"]["dimensions"] == mercator_nessy_read.variables["var1"]["dimensions"], (
            "Variable 'var1' dimensions do not match between original and read NES objects."
        )
        assert mercator_nessy.variables["var1"]["units"] == mercator_nessy_read.variables["var1"]["units"], (
            "Variable 'var1' units do not match between original and read NES objects."
        )
