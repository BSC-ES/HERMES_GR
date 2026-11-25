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

FILENAME = "regular_nessy"


@pytest.fixture(scope="module")
def tmp_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("regular")


@pytest.fixture(scope="module")
def filepath(tmp_dir):
    return str(tmp_dir / FILENAME)


def create_regular_grid():
    """Helper function to create a regular grid NES object with sample data for testing."""

    params = ProjectionParameters.REGULAR
    regular_nessy = nes.create_nes(
        comm=None,
        info=False,
        projection=params["projection"],
        lat_orig=params["lat_orig"],
        lon_orig=params["lon_orig"],
        inc_lat=params["inc_lat"],
        inc_lon=params["inc_lon"],
        n_lat=params["n_lat"],
        n_lon=params["n_lon"],
        times=params["times"],
        first_level=params["levels"][0],
        last_level=params["levels"][-1],
    )

    # Add dummy variable data
    regular_nessy.variables = {
        "var1": {
            "data": SampleData.REGULAR["data"],
            "dimensions": SampleData.REGULAR["dimensions"],
            "units": SampleData.REGULAR["units"],
        }
    }
    return regular_nessy


class TestLatLonRegular:
    """Tests for the NES class with a regular lat-lon projection."""

    p = ProjectionParameters.REGULAR

    def test_object_creation(self):
        """Test that a regular grid NES object can be created without errors and has the expected attributes."""

        try:
            regular_nessy = create_regular_grid()
        except Exception as e:
            pytest.fail(f"Creation of Regular Nes object failed: {e}")

        assert regular_nessy is not None, "Regular Nes object should not be None after creation."
        assert isinstance(regular_nessy, nes.LatLonNes), "Created object should be an instance of nes.LatLonNes."

    def test_projection_data(self):
        """Test that the projection data of the regular grid NES object matches the expected values."""

        regular_nessy = create_regular_grid()

        # Projection data checks
        projection_data = regular_nessy.projection_data
        assert projection_data["grid_mapping_name"] == self.p["grid_mapping_name"], (
            f"Expected grid_mapping_name '{self.p['grid_mapping_name']}'"
            f" but got '{projection_data['grid_mapping_name']}'"
        )
        assert projection_data["lat_orig"] == self.p["lat_orig"], (
            f"Expected lat_orig '{self.p['lat_orig']}' but got '{projection_data['lat_orig']}'"
        )
        assert projection_data["lon_orig"] == self.p["lon_orig"], (
            f"Expected lon_orig '{self.p['lon_orig']}' but got '{projection_data['lon_orig']}'"
        )
        assert projection_data["inc_lat"] == self.p["inc_lat"], (
            f"Expected inc_lat '{self.p['inc_lat']}' but got '{projection_data['inc_lat']}'"
        )
        assert projection_data["inc_lon"] == self.p["inc_lon"], (
            f"Expected inc_lon '{self.p['inc_lon']}' but got '{projection_data['inc_lon']}'"
        )
        assert projection_data["n_lat"] == self.p["n_lat"], (
            f"Expected n_lat '{self.p['n_lat']}' but got '{projection_data['n_lat']}'"
        )
        assert projection_data["n_lon"] == self.p["n_lon"], (
            f"Expected n_lon '{self.p['n_lon']}' but got '{projection_data['n_lon']}'"
        )

    def test_shape(self):
        """
        Test that the shapes of the coordinates and variable data in
        the regular grid NES object match expected values.
        """

        regular_nessy = create_regular_grid()

        assert (
            regular_nessy.lat is not None
            and regular_nessy.lon is not None
            and regular_nessy.time is not None
            and regular_nessy.lev is not None
        ), "NES object is missing one or more of the required attributes: lat, lon, time, lev."

        assert np.shape(regular_nessy.lat["data"]) == (self.p["n_lat"],), (
            "Latitude data shape does not match expected shape."
        )
        assert np.shape(regular_nessy.lon["data"]) == (self.p["n_lon"],), (
            "Longitude data shape does not match expected shape."
        )
        assert len(regular_nessy.time) == (len(self.p["times"])), (
            "Time dimension length does not match expected length."
        )
        assert np.shape(regular_nessy.lev["data"]) == (len(self.p["levels"]),), (
            "Level data shape does not match expected shape."
        )
        assert regular_nessy.variables["var1"]["data"].shape == (
            len(self.p["times"]),
            len(self.p["levels"]),
            self.p["n_lat"],
            self.p["n_lon"],
        ), "Variable 'var1' data shape does not match expected shape."

    def test_variable_data(self):
        """Test that the variable data in the regular grid NES object matches the expected values."""

        regular_nessy = create_regular_grid()
        var1 = regular_nessy.variables["var1"]

        assert var1 is not None, "Variable 'var1' should not be None."
        assert np.array_equal(var1["data"], SampleData.REGULAR["data"]), (
            "Variable 'var1' data does not match expected values."
        )
        assert var1["units"] == SampleData.REGULAR["units"], "Variable 'var1' units do not match expected values."
        assert var1["dimensions"] == SampleData.REGULAR["dimensions"], (
            "Variable 'var1' dimensions do not match expected values."
        )

    def test_create_shapefile(self, filepath):
        """Test that the create_shapefile method creates shapefiles without errors and that the files are created."""

        regular_nessy = create_regular_grid()

        try:
            regular_nessy.create_shapefile()
        except Exception as e:
            pytest.fail(f"Creation of shapefile failed with exception: {e}")

        try:
            regular_nessy.to_shapefile(f"{filepath}.shp", time=self.p["times"][0], lev=0)
        except Exception as e:
            pytest.fail(f"Saving of shapefile failed with exception: {e}")

        try:
            regular_nessy.to_shapefile(f"{filepath}.geojson", time=self.p["times"][0], lev=0)
        except Exception as e:
            pytest.fail(f"Saving of shapefile (.geojson) failed with exception: {e}")

        # Ensure files were created
        for ext in EXTENSIONS:
            assert os.path.exists(f"{filepath}{ext}"), f"Expected file {filepath}{ext} was not created."

    def test_write_and_read(self, filepath):
        """Test that writing the regular grid NES to NetCDF and reading it back produces an equivalent NES object."""

        # Creation is already tested in first test
        regular_nessy = create_regular_grid()

        # Writing
        try:
            regular_nessy.to_netcdf(f"{filepath}.nc")

        except Exception as e:
            pytest.fail(f"Writing to NetCDF failed with exception: {e}")

        assert os.path.exists(f"{filepath}.nc"), f"Expected file {filepath}.nc was not created."

        # Reading
        try:
            regular_nessy_read = nes.open_netcdf(f"{filepath}.nc")
        except Exception as e:
            pytest.fail(f"Reading from NetCDF failed with exception: {e}")

        # Loading
        try:
            regular_nessy_read.load()
        except Exception as e:
            pytest.fail(f"Loading NES data failed with exception: {e}")

        # Check correct instance type
        assert isinstance(regular_nessy_read, nes.LatLonNes), "Read NES object should be an instance of nes.LatLonNes."

        # Check that all necessary attributes are present
        assert regular_nessy.projection_data and regular_nessy_read.projection_data is not None, (
            "Projection data should not be None in both original and read NES objects."
        )
        assert regular_nessy.lat and regular_nessy_read.lat is not None, (
            "Latitude data should be present in both original and read NES objects."
        )
        assert regular_nessy.lon and regular_nessy_read.lon is not None, (
            "Longitude data should be present in both original and read NES objects."
        )
        assert regular_nessy.time and regular_nessy_read.time is not None, (
            "Time data should be present in both original and read NES objects."
        )
        assert regular_nessy.lev and regular_nessy_read.lev is not None, (
            "Level data should be present in both original and read NES objects."
        )
        assert regular_nessy.variables and regular_nessy_read.variables is not None, (
            "Variable data should be present in both original and read NES objects."
        )

        # Equivalence checks
        projection_data = regular_nessy.projection_data
        projection_data_read = regular_nessy_read.projection_data
        assert projection_data["grid_mapping_name"] == projection_data_read["grid_mapping_name"], (
            f"Expected grid_mapping_name '{projection_data['grid_mapping_name']}' "
            f"but got '{projection_data_read['grid_mapping_name']}' after reading from NetCDF."
        )
        assert projection_data["semi_major_axis"] == projection_data_read["semi_major_axis"], (
            f"Expected semi_major_axis '{projection_data['semi_major_axis']}' "
            f"but got '{projection_data_read['semi_major_axis']}' after reading from NetCDF."
        )
        assert projection_data["inverse_flattening"] == projection_data_read["inverse_flattening"], (
            f"Expected inverse_flattening '{projection_data['inverse_flattening']}'"
            f" but got '{projection_data_read['inverse_flattening']}' after reading from NetCDF."
        )
        assert np.array_equal(regular_nessy.lat["data"], regular_nessy_read.lat["data"]), (
            "Latitude data does not match after reading from NetCDF."
        )
        assert np.array_equal(regular_nessy.lon["data"], regular_nessy_read.lon["data"]), (
            "Longitude data does not match after reading from NetCDF."
        )
        assert regular_nessy.time == regular_nessy_read.time, "Time data does not match after reading from NetCDF."
        assert np.array_equal(regular_nessy.lev["data"], regular_nessy_read.lev["data"]), (
            "Level data does not match after reading from NetCDF."
        )
        assert np.array_equal(
            regular_nessy.variables["var1"]["data"],
            regular_nessy_read.variables["var1"]["data"],
        ), "Variable 'var1' data does not match after reading from NetCDF."
        assert regular_nessy.variables["var1"]["dimensions"] == regular_nessy_read.variables["var1"]["dimensions"], (
            "Variable 'var1' dimensions do not match after reading from NetCDF."
        )
        assert regular_nessy.variables["var1"]["units"] == regular_nessy_read.variables["var1"]["units"], (
            "Variable 'var1' units do not match after reading from NetCDF."
        )
