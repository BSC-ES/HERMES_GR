# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import numpy as np
import pytest
from expected_ISO_grids import EXPECTED_ISO_GRID

import nes
from nes.tests.utils import (
    DummyData,
    create_synthetic_spatial_files,
    load_fresh_nes,
)

FILENAME = "test_spatial_join.nc"
SHP_FILE = "test_spatial_join_regions.shp"
GEOJSON_FILE = "test_spatial_join_regions.geojson"
SpatialJoinData = DummyData.SpatialJoin


@pytest.fixture(scope="module")
def tmp_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("spatial_join")


@pytest.fixture(scope="module")
def filepath(tmp_dir):
    return str(tmp_dir / FILENAME)


@pytest.fixture(scope="module")
def filepath_shp(tmp_dir):
    return str(tmp_dir / SHP_FILE)


@pytest.fixture(scope="module")
def filepath_geojson(tmp_dir):
    return str(tmp_dir / GEOJSON_FILE)


@pytest.fixture(scope="module", autouse=True)
def setup_and_cleanup(filepath, filepath_shp, filepath_geojson):
    """Create test NetCDF and synthetic shapefile/GeoJSON before tests, clean up after."""

    nessy = nes.create_nes(
        comm=None,
        info=False,
        projection=SpatialJoinData["projection"],
        lat_orig=SpatialJoinData["lat_orig"],
        lon_orig=SpatialJoinData["lon_orig"],
        inc_lat=SpatialJoinData["inc_lat"],
        inc_lon=SpatialJoinData["inc_lon"],
        n_lat=SpatialJoinData["n_lat"],
        n_lon=SpatialJoinData["n_lon"],
        times=SpatialJoinData["times"],
    )
    nessy.variables = {
        "var1": {
            "data": SpatialJoinData["var1"]["data"],
            "dimensions": SpatialJoinData["var1"]["dimensions"],
            "units": SpatialJoinData["var1"]["units"],
        }
    }
    nessy.to_netcdf(filepath)

    # Create synthetic external shapefiles
    create_synthetic_spatial_files(filepath_shp, filepath_geojson)


def _get_iso_grid(nessy):
    """Extract the ISO column from the shapefile as a 2D grid (nlat x nlon)."""

    iso_values = nessy.shapefile["ISO"].values
    return iso_values.reshape(DummyData._nlat, DummyData._nlon)


class TestSpatialJoinCentroid:
    """Tests for spatial_join with method='centroid'."""

    def test_centroid_shp(self, filepath, filepath_shp):
        """Test that spatial_join with method='centroid' correctly assigns ISO values from a shapefile."""

        nessy = load_fresh_nes(filepath)
        nessy.create_shapefile()
        nessy.spatial_join(filepath_shp, method="centroid", var_list="ISO")

        assert nessy.shapefile is not None, "Shapefile should be created after spatial_join."
        assert "ISO" in nessy.shapefile.columns, "ISO column should be added to shapefile after spatial_join."
        iso_grid = _get_iso_grid(nessy)
        assert np.array_equal(iso_grid, EXPECTED_ISO_GRID), (
            "The ISO grid from centroid spatial join does not match the expected grid."
        )

    def test_centroid_geojson(self, filepath, filepath_geojson):
        """Test that spatial_join with method='centroid' correctly assigns ISO values from a GeoJSON file."""

        nessy = load_fresh_nes(filepath)
        nessy.create_shapefile()
        nessy.spatial_join(filepath_geojson, method="centroid", var_list="ISO")

        assert nessy.shapefile is not None, "Shapefile should be created after spatial_join."
        assert "ISO" in nessy.shapefile.columns, "ISO column should be added to shapefile after spatial_join."
        iso_grid = _get_iso_grid(nessy)
        assert np.array_equal(iso_grid, EXPECTED_ISO_GRID), (
            "The ISO grid from centroid spatial join with GeoJSON does not match the expected grid."
        )


class TestSpatialJoinNearest:
    """Tests for spatial_join with method='nearest'."""

    def test_nearest_shp(self, filepath, filepath_shp):
        """Test that spatial_join with method='nearest' correctly assigns ISO values from a shapefile."""

        nessy = load_fresh_nes(filepath)
        nessy.create_shapefile()
        nessy.spatial_join(filepath_shp, method="nearest", var_list="ISO")

        assert nessy.shapefile is not None, "Shapefile should be created after spatial_join."
        assert "ISO" in nessy.shapefile.columns, "ISO column should be added to shapefile after spatial_join."
        assert nessy.shapefile["ISO"].notna().all(), "All ISO values should be non-NA after spatial_join."

    def test_nearest_geojson(self, filepath, filepath_geojson):
        """Test that spatial_join with method='nearest' produces expected results with GeoJSON input."""

        nessy = load_fresh_nes(filepath)
        nessy.create_shapefile()
        nessy.spatial_join(filepath_geojson, method="nearest", var_list="ISO")

        assert nessy.shapefile is not None, "Shapefile should be created after spatial_join."
        assert "ISO" in nessy.shapefile.columns, "ISO column should be added to shapefile after spatial_join."
        assert nessy.shapefile["ISO"].notna().all(), "All ISO values should be non-NA after spatial_join with GeoJSON."

    def test_nearest_matches_expected_grid(self, filepath, filepath_shp):
        """Nearest with full coverage should match centroid assignments."""

        nessy = load_fresh_nes(filepath)
        nessy.create_shapefile()
        nessy.spatial_join(filepath_shp, method="nearest", var_list="ISO")

        iso_grid = _get_iso_grid(nessy)
        assert np.array_equal(iso_grid, EXPECTED_ISO_GRID), (
            "The ISO grid from nearest spatial join does not match the expected grid."
        )


class TestSpatialJoinIntersection:
    """Tests for spatial_join with method='intersection'."""

    def test_intersection_shp(self, filepath, filepath_shp):
        """Test that spatial_join with method='intersection' correctly assigns ISO values from a shapefile."""

        nessy = load_fresh_nes(filepath)
        nessy.create_shapefile()
        nessy.spatial_join(filepath_shp, method="intersection", var_list="ISO")

        assert nessy.shapefile is not None, "Shapefile should be created after spatial_join."
        assert "ISO" in nessy.shapefile.columns, "ISO column should be added to shapefile after spatial_join."
        iso_grid = _get_iso_grid(nessy)
        assert np.array_equal(iso_grid, EXPECTED_ISO_GRID), (
            "The ISO grid from intersection spatial join does not match the expected grid."
        )

    def test_intersection_geojson(self, filepath, filepath_geojson):
        """Test that spatial_join with method='intersection' produces expected results with GeoJSON input."""

        nessy = load_fresh_nes(filepath)
        nessy.create_shapefile()
        nessy.spatial_join(filepath_geojson, method="intersection", var_list="ISO")

        assert nessy.shapefile is not None, "Shapefile should be created after spatial_join."
        assert "ISO" in nessy.shapefile.columns, "ISO column should be added to shapefile after spatial_join."
        iso_grid = _get_iso_grid(nessy)
        assert np.array_equal(iso_grid, EXPECTED_ISO_GRID), (
            "The ISO grid from intersection spatial join with GeoJSON does not match the expected grid."
        )


class TestSpatialJoinEdgeCases:
    """Tests for edge cases and error handling."""

    def test_invalid_method_raises(self, filepath, filepath_shp):
        nessy = load_fresh_nes(filepath)
        nessy.create_shapefile()
        with pytest.raises(NotImplementedError):
            nessy.spatial_join(filepath_shp, method="invalid_method", var_list="ISO")

    def test_var_list_as_list(self, filepath, filepath_shp):
        """Passing var_list as a list (instead of string) should also work."""

        nessy = load_fresh_nes(filepath)
        nessy.create_shapefile()
        nessy.spatial_join(filepath_shp, method="centroid", var_list=["ISO"])

        assert nessy.shapefile is not None, "Shapefile should be created after spatial_join."
        assert "ISO" in nessy.shapefile.columns, "ISO column should be added to shapefile after spatial_join."

    def test_shapefile_auto_created(self, filepath, filepath_shp):
        """spatial_join should auto-create the internal shapefile if missing."""

        nessy = load_fresh_nes(filepath)
        nessy.spatial_join(filepath_shp, method="centroid", var_list="ISO")  # ? handles shapefile creation

        assert nessy.shapefile is not None, (
            "Shapefile should be created after spatial_join even if it was not created beforehand."
        )
        assert "ISO" in nessy.shapefile.columns, (
            "ISO column should be added to shapefile after spatial_join even if shapefile was not created beforehand."
        )
