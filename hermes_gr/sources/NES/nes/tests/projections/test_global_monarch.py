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

FILENAME = "global_monarch_nessy"


@pytest.fixture(scope="module")
def tmp_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("global_monarch")


@pytest.fixture(scope="module")
def filepath(tmp_dir):
    return str(tmp_dir / FILENAME)


def create_global_monarch_grid():
    """Helper function to create a global_monarch grid NES object with sample data."""

    params = ProjectionParameters.GLOBAL_MONARCH
    global_monarch_nessy = nes.create_nes(
        comm=None,
        info=False,
        projection="global_monarch",
        inc_lat=params["inc_lat"],
        inc_lon=params["inc_lon"],
        times=params["times"],
        first_level=params["levels"][0],
        last_level=params["levels"][-1],
    )

    global_monarch_nessy.variables = {
        "var1": {
            "data": SampleData.GLOBAL_MONARCH["data"],
            "dimensions": SampleData.GLOBAL_MONARCH["dimensions"],
            "units": SampleData.GLOBAL_MONARCH["units"],
        }
    }
    return global_monarch_nessy


class TestGlobalMonarchNes:
    """Tests for creating and working with a global_monarch grid NES object."""

    p = ProjectionParameters.GLOBAL_MONARCH

    def test_object_creation(self):
        """Test that a global monarch grid NES object can be created without errors and has the expected attributes."""

        try:
            global_monarch_nessy = create_global_monarch_grid()
        except Exception as e:
            pytest.fail(f"Creation of Global Monarch Nes object failed: {e}")

        assert global_monarch_nessy is not None, "Global Nes object should not be None after creation."
        assert isinstance(global_monarch_nessy, nes.LatLonNes), "Created object should be an instance of nes.LatLonNes."

    def test_projection_data(self):
        """Test that the projection data of the global monarch grid NES object matches expected values."""

        global_monarch_nessy = create_global_monarch_grid()

        # Projection data checks
        projection_data = global_monarch_nessy.projection_data
        assert projection_data["grid_mapping_name"] == self.p["grid_mapping_name"], (
            "Grid mapping name does not match expected value."
        )
        assert projection_data["nes_projection"] == self.p["projection"], (
            "nes_projection does not match expected value."
        )
        assert projection_data["inc_lat"] == self.p["inc_lat"], "Latitude increment does not match expected value."
        assert projection_data["inc_lon"] == self.p["inc_lon"], "Longitude increment does not match expected value."
        assert projection_data["lat_orig"] == self.p["lat_orig"], "Latitude origin does not match expected value."
        assert projection_data["lon_orig"] == self.p["lon_orig"], "Longitude origin does not match expected value."
        assert projection_data["n_lat"] == np.shape(SampleData.GLOBAL_MONARCH["data"])[2], (
            "Number of latitude points does not match expected value."
        )
        assert projection_data["n_lon"] == np.shape(SampleData.GLOBAL_MONARCH["data"])[3], (
            "Number of longitude points does not match expected value."
        )

    def test_shape(self):
        """Test that the shape of the latitude, longitude, time, level, and variable data matches expected values."""

        global_monarch_nessy = create_global_monarch_grid()

        assert isinstance(global_monarch_nessy, nes.LatLonNes)
        assert (
            global_monarch_nessy.lat is not None
            and global_monarch_nessy.lon is not None
            and global_monarch_nessy.time is not None
            and global_monarch_nessy.lev is not None
        ), "NES object is missing one or more of the required attributes: lat, lon, time, lev."

        assert np.shape(global_monarch_nessy.lat["data"]) == (np.shape(SampleData.GLOBAL_MONARCH["data"])[2],), (
            "Latitude data shape does not match expected shape."
        )
        assert np.shape(global_monarch_nessy.lon["data"]) == (np.shape(SampleData.GLOBAL_MONARCH["data"])[3],), (
            "Longitude data shape does not match expected shape."
        )
        assert len(global_monarch_nessy.time) == (len(self.p["times"])), (
            "Time dimension length does not match expected length."
        )
        assert np.shape(global_monarch_nessy.lev["data"]) == (len(self.p["levels"]),), (
            "Level data shape does not match expected shape."
        )
        assert global_monarch_nessy.variables["var1"]["data"].shape == (
            len(self.p["times"]),
            len(self.p["levels"]),
            np.shape(SampleData.GLOBAL_MONARCH["data"])[2],
            np.shape(SampleData.GLOBAL_MONARCH["data"])[3],
        ), "Variable 'var1' data shape does not match expected shape."

    def test_variable_data(self):
        """Test that the variable data in the global monarch grid NES object matches expected values."""

        global_monarch_nessy = create_global_monarch_grid()
        var1 = global_monarch_nessy.variables["var1"]

        assert var1 is not None, "Variable 'var1' should be present in the NES object."
        assert np.array_equal(var1["data"], SampleData.GLOBAL_MONARCH["data"]), (
            "Variable 'var1' data does not match expected values."
        )
        assert var1["units"] == SampleData.GLOBAL_MONARCH["units"], (
            "Variable 'var1' units do not match expected values."
        )
        assert var1["dimensions"] == SampleData.GLOBAL_MONARCH["dimensions"], (
            "Variable 'var1' dimensions do not match expected values."
        )

    def test_odd_number_of_points(self):
        """Test that the global_monarch grid always has an odd number of latitude and longitude points."""

        global_monarch_nessy = create_global_monarch_grid()

        projection_data = global_monarch_nessy.projection_data
        assert projection_data["n_lat"] == int(2 * (90 / self.p["inc_lat"])) + 1, (
            "n_lat does not follow the 2 * (90 / inc_lat) + 1 rule."
        )
        assert projection_data["n_lon"] == int(2 * (180 / self.p["inc_lon"])) + 1, (
            "n_lon does not follow the 2 * (180 / inc_lon) + 1 rule."
        )
        assert projection_data["n_lat"] % 2 == 1, "Number of latitude points should be odd."
        assert projection_data["n_lon"] % 2 == 1, "Number of longitude points should be odd."
        assert global_monarch_nessy.lat is not None and global_monarch_nessy.lon is not None, (
            "NES object is missing one or more of the required attributes: lat, lon."
        )
        assert len(global_monarch_nessy.lat["data"]) % 2 == 1, "Latitude coordinate array length should be odd."
        assert len(global_monarch_nessy.lon["data"]) % 2 == 1, "Longitude coordinate array length should be odd."

    def test_central_cell_centred_at_origin(self):
        """Test that the central grid cell of the global_monarch grid is always centred at (0, 0)."""

        global_monarch_nessy = create_global_monarch_grid()

        assert global_monarch_nessy.lat is not None and global_monarch_nessy.lon is not None, (
            "NES object is missing one or more of the required attributes: lat, lon."
        )
        lat = global_monarch_nessy.lat["data"]
        lon = global_monarch_nessy.lon["data"]

        assert lat[len(lat) // 2] == pytest.approx(0.0), "Central latitude centroid should be exactly 0 degrees."
        assert lon[len(lon) // 2] == pytest.approx(0.0), "Central longitude centroid should be exactly 0 degrees."

    def test_border_centroids_quarter_cell_shift(self):
        """Test that the border cells are half cells: their centroids are shifted one quarter of a grid cell
        inwards from the domain boundaries, while interior cells keep the regular grid spacing.
        """

        global_monarch_nessy = create_global_monarch_grid()

        inc_lat = self.p["inc_lat"]
        inc_lon = self.p["inc_lon"]
        assert global_monarch_nessy.lat is not None and global_monarch_nessy.lon is not None, (
            "NES object is missing one or more of the required attributes: lat, lon."
        )
        lat = global_monarch_nessy.lat["data"]
        lon = global_monarch_nessy.lon["data"]

        # Border centroids are shifted one quarter of a grid cell inwards
        assert lat[0] == pytest.approx(-90 + inc_lat / 4), (
            "First latitude centroid should be shifted a quarter of a cell from the South Pole."
        )
        assert lat[-1] == pytest.approx(90 - inc_lat / 4), (
            "Last latitude centroid should be shifted a quarter of a cell from the North Pole."
        )
        assert lon[0] == pytest.approx(-180 + inc_lon / 4), (
            "First longitude centroid should be shifted a quarter of a cell from the antimeridian."
        )
        assert lon[-1] == pytest.approx(180 - inc_lon / 4), (
            "Last longitude centroid should be shifted a quarter of a cell from the antimeridian."
        )

        # Interior centroids keep the regular grid spacing
        assert np.allclose(np.diff(lat[1:-1]), inc_lat), "Interior latitude centroids should be regularly spaced."
        assert np.allclose(np.diff(lon[1:-1]), inc_lon), "Interior longitude centroids should be regularly spaced."

    def test_domain_boundaries(self):
        """Test that the domain boundaries are exactly [-90, 90] in latitude and [-180, 180] in longitude."""

        global_monarch_nessy = create_global_monarch_grid()

        try:
            global_monarch_nessy.create_spatial_bounds()
        except Exception as e:
            pytest.fail(f"Creation of spatial bounds failed with exception: {e}")

        assert global_monarch_nessy.lat_bnds is not None and global_monarch_nessy.lon_bnds is not None, (
            "Latitude and longitude boundaries should be present after creating spatial bounds."
        )

        lat_bnds = global_monarch_nessy.lat_bnds["data"]
        lon_bnds = global_monarch_nessy.lon_bnds["data"]

        # The domain edges are exactly the poles and the antimeridian
        assert lat_bnds[0, 0] == pytest.approx(-90), "First latitude boundary should be exactly -90 degrees."
        assert lat_bnds[-1, -1] == pytest.approx(90), "Last latitude boundary should be exactly 90 degrees."
        assert lon_bnds[0, 0] == pytest.approx(-180), "First longitude boundary should be exactly -180 degrees."
        assert lon_bnds[-1, -1] == pytest.approx(180), "Last longitude boundary should be exactly 180 degrees."

        # No boundary exceeds the domain limits
        assert np.all(lat_bnds >= -90) and np.all(lat_bnds <= 90), (
            "Latitude boundaries should be within [-90, 90] degrees."
        )
        assert np.all(lon_bnds >= -180) and np.all(lon_bnds <= 180), (
            "Longitude boundaries should be within [-180, 180] degrees."
        )

        # Border cells are half cells, so they are narrower than a full grid cell
        assert lat_bnds[0, 1] - lat_bnds[0, 0] < self.p["inc_lat"], (
            "First latitude cell should be narrower than a full grid cell."
        )
        assert lat_bnds[-1, 1] - lat_bnds[-1, 0] < self.p["inc_lat"], (
            "Last latitude cell should be narrower than a full grid cell."
        )
        assert lon_bnds[0, 1] - lon_bnds[0, 0] < self.p["inc_lon"], (
            "First longitude cell should be narrower than a full grid cell."
        )
        assert lon_bnds[-1, 1] - lon_bnds[-1, 0] < self.p["inc_lon"], (
            "Last longitude cell should be narrower than a full grid cell."
        )

    def test_invalid_increment(self):
        """Test that increments that do not divide the domain exactly raise a ValueError."""

        with pytest.raises(ValueError, match="inc_lat must divide 90 exactly"):
            nes.create_nes(comm=None, info=False, projection="global_monarch", inc_lat=0.7, inc_lon=1)

        with pytest.raises(ValueError, match="inc_lon must divide 180 exactly"):
            nes.create_nes(comm=None, info=False, projection="global_monarch", inc_lat=1, inc_lon=0.7)

    def test_create_shapefile(self, filepath):
        """
        Test that the create_shapefile method creates shapefile
        and geojson without errors and that the files are created.
        """

        global_monarch_nessy = create_global_monarch_grid()

        # Create shapefile and geojson, check for exceptions
        try:
            global_monarch_nessy.create_shapefile()
        except Exception as e:
            pytest.fail(f"Creation of shapefile failed with exception: {e}")

        try:
            global_monarch_nessy.to_shapefile(f"{filepath}.shp", time=self.p["times"][0], lev=0)
        except Exception as e:
            pytest.fail(f"Saving of shapefile (.shp) failed with exception: {e}")

        try:
            global_monarch_nessy.to_shapefile(f"{filepath}.geojson", time=self.p["times"][0], lev=0)
        except Exception as e:
            pytest.fail(f"Saving of shapefile (.geojson) failed with exception: {e}")

        for ext in EXTENSIONS:
            assert os.path.exists(f"{filepath}{ext}"), f"Expected file {filepath}{ext} was not created."

    def test_write_and_read(self, filepath):
        """Tests that writing the global monarch grid NES to NetCDF
        and reading it back produces an equivalent NES object.
        """

        # Creation is already tested in first test
        global_monarch_nessy = create_global_monarch_grid()

        # Writing
        try:
            global_monarch_nessy.to_netcdf(f"{filepath}.nc")
            assert os.path.exists(f"{filepath}.nc")
        except Exception as e:
            pytest.fail(f"Writing to NetCDF failed with exception: {e}")

        # Reading
        try:
            global_monarch_nessy_read = nes.open_netcdf(f"{filepath}.nc")
        except Exception as e:
            pytest.fail(f"Reading from NetCDF failed with exception: {e}")

        # Loading
        try:
            global_monarch_nessy_read.load()
        except Exception as e:
            pytest.fail(f"Loading NES data failed with exception: {e}")

        assert global_monarch_nessy_read is not None, "Global Nes object should not be None after reading from NetCDF."
        assert isinstance(global_monarch_nessy, nes.LatLonNes), "Created object should be an instance of nes.LatLonNes."

        # Check that all necessary attributes are present
        assert global_monarch_nessy.projection_data and global_monarch_nessy_read.projection_data is not None, (
            "Projection data should be present in both original and read NES objects."
        )
        assert global_monarch_nessy.lat and global_monarch_nessy_read.lat is not None, (
            "Latitude data should be present in both original and read NES objects."
        )
        assert global_monarch_nessy.lon and global_monarch_nessy_read.lon is not None, (
            "Longitude data should be present in both original and read NES objects."
        )
        assert global_monarch_nessy.time and global_monarch_nessy_read.time is not None, (
            "Time data should be present in both original and read NES objects."
        )
        assert global_monarch_nessy.lev and global_monarch_nessy_read.lev is not None, (
            "Level data should be present in both original and read NES objects."
        )
        assert global_monarch_nessy.variables and global_monarch_nessy_read.variables is not None, (
            "Variables should be present in both original and read NES objects."
        )

        # Check that the grid mapping name matches after read
        projection_data = global_monarch_nessy.projection_data
        projection_data_read = global_monarch_nessy_read.projection_data

        # Equivalence checks
        assert projection_data["grid_mapping_name"] == projection_data_read["grid_mapping_name"], (
            "Grid mapping name does not match after reading from NetCDF."
        )
        assert projection_data_read.get("nes_projection") == self.p["projection"], (
            "nes_projection attribute does not match after reading from NetCDF."
        )
        assert projection_data["semi_major_axis"] == projection_data_read["semi_major_axis"], (
            "Semi-major axis does not match after reading from NetCDF."
        )
        assert projection_data["inverse_flattening"] == projection_data_read["inverse_flattening"], (
            "Inverse flattening does not match after reading from NetCDF."
        )
        assert np.array_equal(global_monarch_nessy.lat["data"], global_monarch_nessy_read.lat["data"]), (
            "Latitude data does not match after reading from NetCDF."
        )
        assert np.array_equal(global_monarch_nessy.lon["data"], global_monarch_nessy_read.lon["data"]), (
            "Longitude data does not match after reading from NetCDF."
        )
        assert global_monarch_nessy.time == global_monarch_nessy_read.time, (
            "Time data does not match after reading from NetCDF."
        )
        assert np.array_equal(global_monarch_nessy.lev["data"], global_monarch_nessy_read.lev["data"]), (
            "Level data does not match after reading from NetCDF."
        )
        assert np.array_equal(
            global_monarch_nessy.variables["var1"]["data"],
            global_monarch_nessy_read.variables["var1"]["data"],
        ), "Variable 'var1' data does not match after reading from NetCDF."
        assert (
            global_monarch_nessy.variables["var1"]["dimensions"]
            == global_monarch_nessy_read.variables["var1"]["dimensions"]
        ), "Variable 'var1' dimensions do not match after reading from NetCDF."
        assert (
            global_monarch_nessy.variables["var1"]["units"] == global_monarch_nessy_read.variables["var1"]["units"]
        ), "Variable 'var1' units do not match after reading from NetCDF."
