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

FILENAME = "rotated_nessy"


@pytest.fixture(scope="module")
def tmp_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("rotated")


@pytest.fixture(scope="module")
def filepath(tmp_dir):
    return str(tmp_dir / FILENAME)


def create_rotated_grid():
    """Helper function to create a RotatedNes object with the specified parameters and sample data."""

    params = ProjectionParameters.ROTATED

    # Create rotated NES object
    rotated_nessy = nes.create_nes(
        comm=None,
        info=False,
        projection=params["projection"],
        centre_lat=params["centre_lat"],
        centre_lon=params["centre_lon"],
        west_boundary=params["west_boundary"],
        south_boundary=params["south_boundary"],
        inc_rlat=params["inc_rlat"],
        inc_rlon=params["inc_rlon"],
        first_level=params["levels"][0],
        last_level=params["levels"][-1],
        times=params["times"],
    )

    # Add dummy variable data
    rotated_nessy.variables = {
        "var1": {
            "data": SampleData.ROTATED["data"],
            "dimensions": SampleData.ROTATED["dimensions"],
            "units": SampleData.ROTATED["units"],
        }
    }
    return rotated_nessy


class TestRotatedNes:
    """Tests for the NES class with a rotated lat-lon projection."""

    p = ProjectionParameters.ROTATED

    def test_object_creation(self):
        """Test that a RotatedNes object can be created with the specified parameters and sample data."""

        try:
            rotated_nessy = create_rotated_grid()
        except Exception as e:
            pytest.fail(f"Object creation failed with exception: {e}")

        assert rotated_nessy is not None, "Rotated Nes object should not be None after creation."
        assert isinstance(rotated_nessy, nes.RotatedNes), "Created object should be an instance of nes.RotatedNes."

    def test_projection_data(self):
        """
        Test that the projection data in the RotatedNes object
        matches the expected values based on the input parameters.
        """

        rotated_nessy = create_rotated_grid()

        # Projection data checks
        projection_data = rotated_nessy.projection_data
        assert projection_data["grid_mapping_name"] == self.p["grid_mapping_name"], (
            "Grid mapping name in projection data should match the expected value."
        )
        assert projection_data["inc_rlat"] == self.p["inc_rlat"], (
            "Increment in rotated latitude should match the expected value."
        )
        assert projection_data["inc_rlon"] == self.p["inc_rlon"], (
            "Increment in rotated longitude should match the expected value."
        )
        assert projection_data["west_boundary"] == self.p["west_boundary"], (
            "West boundary should match the expected value."
        )
        assert projection_data["south_boundary"] == self.p["south_boundary"], (
            "South boundary should match the expected value."
        )
        assert projection_data["grid_north_pole_latitude"] == 90 - self.p["centre_lat"], (
            "Grid north pole latitude should be 90 minus the centre latitude."
        )
        assert projection_data["grid_north_pole_longitude"] == self.p["centre_lon"] - 180, (
            "Grid north pole longitude should be the centre longitude minus 180."
        )

    def test_shape(self):
        """Test that the shapes of the coordinates and variable data in the RotatedNes object match expected values."""

        rotated_nessy = create_rotated_grid()
        assert isinstance(rotated_nessy, nes.RotatedNes)  # to suppress warnings
        assert (
            rotated_nessy.rlat is not None
            and rotated_nessy.rlon is not None
            and rotated_nessy.lat is not None
            and rotated_nessy.lon is not None
            and rotated_nessy.time is not None
            and rotated_nessy.lev is not None
        ), "NES object is missing one or more of the required attributes: rlat, rlon, lat, lon, time, lev."

        # Shape checks
        assert np.shape(rotated_nessy.rlat["data"]) == (np.shape(SampleData.ROTATED["data"])[2],), (
            "Shape of rotated latitude data does not match expected shape."
        )
        assert np.shape(rotated_nessy.rlon["data"]) == (np.shape(SampleData.ROTATED["data"])[3],), (
            "Shape of rotated longitude data does not match expected shape."
        )
        assert np.shape(rotated_nessy.lat["data"]) == (
            np.shape(SampleData.ROTATED["data"])[2],
            np.shape(SampleData.ROTATED["data"])[3],
        ), "Shape of latitude data does not match expected shape."
        assert np.shape(rotated_nessy.lon["data"]) == (
            np.shape(SampleData.ROTATED["data"])[2],
            np.shape(SampleData.ROTATED["data"])[3],
        ), "Shape of longitude data does not match expected shape."
        assert len(rotated_nessy.time) == len(self.p["times"]), "Time dimension length does not match expected length."
        assert np.shape(rotated_nessy.lev["data"]) == (len(self.p["levels"]),), (
            "Level data shape does not match expected shape."
        )
        assert rotated_nessy.variables["var1"]["data"].shape == np.shape(SampleData.ROTATED["data"]), (
            "Variable 'var1' data shape does not match expected shape."
        )

    def test_variable_data(self):
        """Test that the variable data in the RotatedNes object matches the expected values."""

        rotated_nessy = create_rotated_grid()
        var1 = rotated_nessy.variables["var1"]
        assert var1 is not None, "Variable 'var1' should be present in the NES object."
        assert np.array_equal(var1["data"], SampleData.ROTATED["data"]), (
            "Variable 'var1' data does not match expected values."
        )
        assert var1["units"] == SampleData.ROTATED["units"], "Variable 'var1' units do not match expected values."
        assert var1["dimensions"] == SampleData.ROTATED["dimensions"], (
            "Variable 'var1' dimensions do not match expected values."
        )

    def test_create_shapefile(self, filepath):
        """Test that the create_shapefile method creates shapefiles without errors and that the files are created."""

        rotated_nessy = create_rotated_grid()

        try:
            rotated_nessy.create_shapefile()
        except Exception as e:
            pytest.fail(f"Creation of shapefile failed with exception: {e}")

        try:
            rotated_nessy.to_shapefile(f"{filepath}.shp", time=self.p["times"][0], lev=0)
        except Exception as e:
            pytest.fail(f"Saving of shapefile failed with exception: {e}")

        try:
            rotated_nessy.to_shapefile(f"{filepath}.geojson", time=self.p["times"][0], lev=0)
        except Exception as e:
            pytest.fail(f"Saving of shapefile (.geojson) failed with exception: {e}")

        # Ensure files were created
        for ext in EXTENSIONS:
            assert os.path.exists(f"{filepath}{ext}")

    def test_write_and_read(self, filepath):
        """Test that writing the rotated grid NES to NetCDF and reading it back produces an equivalent NES object."""

        # Creation is already tested in first test
        rotated_nessy = create_rotated_grid()

        # Writing
        try:
            rotated_nessy.to_netcdf(f"{filepath}.nc")

        except Exception as e:
            pytest.fail(f"Writing to NetCDF failed with exception: {e}")

        assert os.path.exists(f"{filepath}.nc"), "NetCDF file was not created after writing."

        # Reading
        try:
            rotated_nessy_read = nes.open_netcdf(f"{filepath}.nc")
        except Exception as e:
            pytest.fail(f"Reading from NetCDF failed with exception: {e}")

        # Loading
        try:
            rotated_nessy_read.load()
        except Exception as e:
            pytest.fail(f"Loading NES data failed with exception: {e}")

        # Suppress warnings
        assert isinstance(rotated_nessy, nes.RotatedNes)

        # Check that the read object is also a RotatedNes
        assert isinstance(rotated_nessy_read, nes.RotatedNes), (
            "Read NES object should be an instance of nes.RotatedNes."
        )

        # Check that all necessary attributes are present
        assert rotated_nessy.projection_data and rotated_nessy_read.projection_data is not None, (
            "Projection data should not be None in both original and read NES objects."
        )
        assert rotated_nessy.rlat is not None and rotated_nessy_read.rlat is not None, (
            "Rotated latitude should not be None in both original and read NES objects."
        )
        assert rotated_nessy.rlon is not None and rotated_nessy_read.rlon is not None, (
            "Rotated longitude should not be None in both original and read NES objects."
        )
        assert rotated_nessy.lat is not None and rotated_nessy_read.lat is not None, (
            "Latitude should not be None in both original and read NES objects."
        )
        assert rotated_nessy.lon is not None and rotated_nessy_read.lon is not None, (
            "Longitude should not be None in both original and read NES objects."
        )
        assert rotated_nessy.time is not None and rotated_nessy_read.time is not None, (
            "Time should not be None in both original and read NES objects."
        )
        assert rotated_nessy.lev is not None and rotated_nessy_read.lev is not None, (
            "Level should not be None in both original and read NES objects."
        )
        assert rotated_nessy.variables and rotated_nessy_read.variables is not None, (
            "Variables should not be None in both original and read NES objects."
        )

        # Equivalence checks
        projection_data = rotated_nessy.projection_data
        projection_data_read = rotated_nessy_read.projection_data
        assert projection_data["grid_mapping_name"] == projection_data_read["grid_mapping_name"], (
            "Grid mapping name in projection data should match after reading from NetCDF."
        )
        assert projection_data_read["grid_north_pole_latitude"] == projection_data["grid_north_pole_latitude"], (
            "Grid north pole latitude should match after reading from NetCDF."
        )
        assert np.array_equal(rotated_nessy_read.rlat["data"], rotated_nessy.rlat["data"]), (
            "Rotated latitude data should match after reading from NetCDF."
        )
        assert np.array_equal(rotated_nessy_read.rlon["data"], rotated_nessy.rlon["data"]), (
            "Rotated longitude data should match after reading from NetCDF."
        )
        assert np.array_equal(rotated_nessy_read.lat["data"], rotated_nessy.lat["data"]), (
            "Latitude data should match after reading from NetCDF."
        )
        assert np.array_equal(rotated_nessy_read.lon["data"], rotated_nessy.lon["data"]), (
            "Longitude data should match after reading from NetCDF."
        )
        assert rotated_nessy_read.time == rotated_nessy.time, "Time data should match after reading from NetCDF."
        assert np.array_equal(rotated_nessy_read.lev["data"], rotated_nessy.lev["data"]), (
            "Level data should match after reading from NetCDF."
        )
        assert np.array_equal(
            rotated_nessy_read.variables["var1"]["data"],
            rotated_nessy.variables["var1"]["data"],
        ), "Variable 'var1' data should match after reading from NetCDF."
        assert rotated_nessy_read.variables["var1"]["dimensions"] == rotated_nessy.variables["var1"]["dimensions"], (
            "Variable 'var1' dimensions should match after reading from NetCDF."
        )
        assert rotated_nessy_read.variables["var1"]["units"] == rotated_nessy.variables["var1"]["units"], (
            "Variable 'var1' units should match after reading from NetCDF."
        )
