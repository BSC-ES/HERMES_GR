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

FILENAME = "points_nessy"


@pytest.fixture(scope="module")
def tmp_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("points")


@pytest.fixture(scope="module")
def filepath(tmp_dir):
    return str(tmp_dir / FILENAME)


def create_points_grid():
    """Helper function to create a PointsNes object with sample data for testing."""

    params = ProjectionParameters.POINTS

    # Create points NES object (projection=None for points)
    points_nessy = nes.create_nes(
        comm=None,
        info=False,
        parallel_method="X",
        projection=params["projection"],
        lat=params["lat"],
        lon=params["lon"],
        times=params["times"],
    )

    # Add variable data (similar to CSIC NH3 data structure)
    points_nessy.variables = {
        "stations": {
            "data": SampleData.POINTS["stations"]["data"],
            "dimensions": SampleData.POINTS["stations"]["dimensions"],
        },
        "var1": {
            "data": SampleData.POINTS["var1"]["data"],
            "dimensions": SampleData.POINTS["var1"]["dimensions"],
            "units": SampleData.POINTS["var1"]["units"],
        },
    }

    points_nessy.set_strlen(75)
    return points_nessy


class TestPointsNes:
    """Tests for the PointsNes class and its methods."""

    p = ProjectionParameters.POINTS

    def test_object_creation(self):
        """Test that a PointsNes object can be created without errors and has the expected attributes."""

        try:
            points_nessy = create_points_grid()
        except Exception as e:
            pytest.fail(f"Object creation failed with exception: {e}")

        assert points_nessy is not None, "Points Nes object should not be None after creation."
        assert isinstance(points_nessy, nes.PointsNes), "Created object should be an instance of nes.PointsNes."

    def test_projection_data(self):
        """
        Test that the projection_data attribute of the PointsNes object is None,
        as expected for a points projection.
        """

        points_nessy = create_points_grid()
        assert points_nessy.projection_data is None, "Projection data should be None for points projection."

    def test_coordinates(self):
        """Test that the latitude and longitude coordinates in the
        PointsNes object match the input values.
        """

        points_nessy = create_points_grid()

        # Check lat/lon arrays match input
        assert points_nessy.lat is not None
        assert points_nessy.lon is not None
        assert np.array_equal(points_nessy.lat["data"], self.p["lat"])
        assert np.array_equal(points_nessy.lon["data"], self.p["lon"])

    def test_shape(self):
        """Test that the shapes of the coordinates and variable data
        in the PointsNes object match expected values.
        """

        points_nessy = create_points_grid()

        assert (
            points_nessy.lat is not None
            and points_nessy.lon is not None
            and points_nessy.lev is not None
            and points_nessy.time is not None
        ), "Latitude, longitude, level, and time should not be None."

        # Shape checks
        assert np.shape(points_nessy.lat["data"]) == (self.p["n_stations"],), (
            "Latitude shape does not match expected shape."
        )
        assert np.shape(points_nessy.lon["data"]) == (self.p["n_stations"],), (
            "Longitude shape does not match expected shape."
        )
        assert len(points_nessy.time) == len(self.p["times"]), "Time length does not match expected length."

        # Variable shape checks
        assert np.shape(points_nessy.variables["var1"]["data"]) == np.shape(SampleData.POINTS["var1"]["data"]), (
            "Variable 'var1' shape does not match expected shape."
        )
        # Stations name variable shape checks
        assert np.shape(points_nessy.variables["stations"]["data"]) == np.shape(
            SampleData.POINTS["stations"]["data"]
        ), "Variable 'stations' shape does not match expected shape."

    def test_variable_data(self):
        """Test that the variable data in the PointsNes object matches the expected values."""

        points_nessy = create_points_grid()

        # Check var1
        var1 = points_nessy.variables["var1"]
        assert var1 is not None, "Variable 'var1' should be present in the NES object."
        assert np.array_equal(var1["data"], SampleData.POINTS["var1"]["data"]), (
            "Variable 'var1' data does not match expected data."
        )
        assert var1["units"] == SampleData.POINTS["var1"]["units"], "Variable 'var1' units do not match expected units."
        assert var1["dimensions"] == SampleData.POINTS["var1"]["dimensions"], (
            "Variable 'var1' dimensions do not match expected dimensions."
        )

        # Check stations variable
        stations = points_nessy.variables["stations"]
        assert stations is not None, "Variable 'stations' should be present in the NES object."
        assert np.array_equal(stations["data"], SampleData.POINTS["stations"]["data"]), (
            "Variable 'stations' data does not match expected data."
        )
        assert stations["dimensions"] == SampleData.POINTS["stations"]["dimensions"], (
            "Variable 'stations' dimensions do not match expected dimensions."
        )

    def test_create_shapefile(self, filepath):
        """Test that the create_shapefile method creates shapefiles without errors and that the files are created."""

        points_nessy = create_points_grid()

        try:
            points_nessy.create_shapefile()
        except Exception as e:
            pytest.fail(f"Creation of shapefile failed with exception: {e}")

        try:
            points_nessy.to_shapefile(f"{filepath}.shp", time=self.p["times"][0], lev=0)
        except Exception as e:
            pytest.fail(f"Saving of shapefile failed with exception: {e}")

        try:
            points_nessy.to_shapefile(f"{filepath}.geojson", time=self.p["times"][0], lev=0)
        except Exception as e:
            pytest.fail(f"Saving of shapefile (.geojson) failed with exception: {e}")

        # Ensure files were created
        for ext in EXTENSIONS:
            assert os.path.exists(f"{filepath}{ext}")

    def test_times(self):
        """Test that the time attribute of the PointsNes object matches the input time values."""

        points_nessy = create_points_grid()

        assert points_nessy.time is not None, "Time attribute should not be None."
        assert points_nessy.time == self.p["times"], "Time values do not match expected values."

    def test_write_netcdf(self, filepath):
        """Test that writing the PointsNes object to NetCDF works without errors and creates the expected file."""

        points_nessy = create_points_grid()

        # Writing
        try:
            points_nessy.set_strlen(75)
            points_nessy.to_netcdf(f"{filepath}.nc")

        except Exception as e:
            pytest.fail(f"Writing NetCDF failed with exception: {e}")

        assert os.path.exists(f"{filepath}.nc"), "NetCDF file was not created after writing."

        # Reading
        try:
            points_nessy_read = nes.open_netcdf(f"{filepath}.nc", parallel_method="X")
        except Exception as e:
            pytest.fail(f"Reading NetCDF failed with exception: {e}")

        # Loading
        try:
            points_nessy_read.load()
        except Exception as e:
            pytest.fail(f"Loading NetCDF data failed with exception: {e}")

        # Suppress warnings
        assert points_nessy_read is not None, "Points Nes object should not be None after reading from NetCDF."
        assert isinstance(points_nessy_read, nes.PointsNes), "Read object should be an instance of nes.PointsNes."

        # Check that all necessary attributes are present
        assert points_nessy.lat and points_nessy_read.lat is not None, (
            "Latitude data should be present in both original and read NES objects."
        )
        assert points_nessy.lon and points_nessy_read.lon is not None, (
            "Longitude data should be present in both original and read NES objects."
        )
        assert points_nessy.time and points_nessy_read.time is not None, (
            "Time data should be present in both original and read NES objects."
        )
        assert points_nessy.lev and points_nessy_read.lev is not None, (
            "Level data should be present in both original and read NES objects."
        )
        assert points_nessy.variables and points_nessy_read.variables is not None, (
            "Variable data should be present in both original and read NES objects."
        )

        # Equivalence checks
        # Data
        assert np.array_equal(points_nessy.lat["data"], points_nessy_read.lat["data"]), (
            "Latitude data does not match between original and read NES objects."
        )
        assert np.array_equal(points_nessy.lon["data"], points_nessy_read.lon["data"]), (
            "Longitude data does not match between original and read NES objects."
        )
        assert points_nessy.time == points_nessy_read.time, (
            "Time data does not match between original and read NES objects."
        )
        assert np.array_equal(points_nessy.lev["data"], points_nessy_read.lev["data"]), (
            "Level data does not match between original and read NES objects."
        )
        assert np.array_equal(
            points_nessy.variables["var1"]["data"],
            points_nessy_read.variables["var1"]["data"],
        ), "Variable 'var1' data does not match between original and read NES objects."
        assert points_nessy.variables["var1"]["units"] == points_nessy_read.variables["var1"]["units"], (
            "Variable 'var1' units do not match between original and read NES objects."
        )
        assert points_nessy.variables["var1"]["dimensions"] == points_nessy_read.variables["var1"]["dimensions"], (
            "Variable 'var1' dimensions do not match between original and read NES objects."
        )
        assert np.array_equal(
            points_nessy.variables["stations"]["data"],
            points_nessy_read.variables["stations"]["data"],
        ), "Variable 'stations' data does not match between original and read NES objects."
