# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import os

import numpy as np
import pytest

import nes
from nes.tests.utils import DummyData, ExpectedData, load_fresh_nes

# pyright: reportAttributeAccessIssue=false

FILENAME_BASE = "test_horizontal_interp_base.nc"
FILENAME_NO_OVERLAP = "test_horizontal_interp_no_overlap.nc"
FILENAME_OVERLAP = "test_horizontal_interp_overlap.nc"
FILENAME_NN_WM = "test_horizontal_interp_nn_wm.nc"
FILENAME_CONSERVATIVE_WM = "test_horizontal_interp_conservative_wm.nc"

HorizontalInterpData = DummyData.HorizontalInterp
HorizontalExpectedData = ExpectedData.HorizontalInterp


@pytest.fixture(scope="module")
def tmp_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("horizontal_interpolation")


@pytest.fixture(scope="module")
def filepath_base(tmp_dir):
    return str(tmp_dir / FILENAME_BASE)


@pytest.fixture(scope="module")
def filepath_no_overlap(tmp_dir):
    return str(tmp_dir / FILENAME_NO_OVERLAP)


@pytest.fixture(scope="module")
def filepath_overlap(tmp_dir):
    return str(tmp_dir / FILENAME_OVERLAP)


@pytest.fixture(scope="module")
def filepath_nn_wm(tmp_dir):
    return str(tmp_dir / FILENAME_NN_WM)


@pytest.fixture(scope="module")
def filepath_conservative_wm(tmp_dir):
    return str(tmp_dir / FILENAME_CONSERVATIVE_WM)


@pytest.fixture(scope="module", autouse=True)
def setup_test_files(filepath_base, filepath_no_overlap, filepath_overlap):
    """Create the test netCDF file before any tests run."""

    nessy_base = nes.create_nes(
        comm=None,
        info=False,
        projection=HorizontalInterpData["NetCDF_base"]["projection"],
        lat_orig=HorizontalInterpData["NetCDF_base"]["lat_orig"],
        lon_orig=HorizontalInterpData["NetCDF_base"]["lon_orig"],
        inc_lat=HorizontalInterpData["NetCDF_base"]["inc_lat"],
        inc_lon=HorizontalInterpData["NetCDF_base"]["inc_lon"],
        n_lat=HorizontalInterpData["NetCDF_base"]["n_lat"],
        n_lon=HorizontalInterpData["NetCDF_base"]["n_lon"],
        times=HorizontalInterpData["NetCDF_base"]["times"],
    )

    # Add dummy variable data with deterministic values
    nessy_base.variables = {
        "var1": {
            "data": HorizontalInterpData["NetCDF_base"]["var1"]["data"],
            "dimensions": HorizontalInterpData["NetCDF_base"]["var1"]["dimensions"],
            "units": HorizontalInterpData["NetCDF_base"]["var1"]["units"],
        }
    }
    nessy_base.to_netcdf(filepath_base)

    nessy_no_overlap = nes.create_nes(
        comm=None,
        info=False,
        projection=HorizontalInterpData["NetCDF_no_overlap"]["projection"],
        lat_orig=HorizontalInterpData["NetCDF_no_overlap"]["lat_orig"],
        lon_orig=HorizontalInterpData["NetCDF_no_overlap"]["lon_orig"],
        inc_lat=HorizontalInterpData["NetCDF_no_overlap"]["inc_lat"],
        inc_lon=HorizontalInterpData["NetCDF_no_overlap"]["inc_lon"],
        n_lat=HorizontalInterpData["NetCDF_no_overlap"]["n_lat"],
        n_lon=HorizontalInterpData["NetCDF_no_overlap"]["n_lon"],
        times=HorizontalInterpData["NetCDF_no_overlap"]["times"],
    )

    nessy_no_overlap.to_netcdf(filepath_no_overlap)

    nessy_overlap = nes.create_nes(
        comm=None,
        info=False,
        projection=HorizontalInterpData["NetCDF_overlap"]["projection"],
        lat_orig=HorizontalInterpData["NetCDF_overlap"]["lat_orig"],
        lon_orig=HorizontalInterpData["NetCDF_overlap"]["lon_orig"],
        inc_lat=HorizontalInterpData["NetCDF_overlap"]["inc_lat"],
        inc_lon=HorizontalInterpData["NetCDF_overlap"]["inc_lon"],
        n_lat=HorizontalInterpData["NetCDF_overlap"]["n_lat"],
        n_lon=HorizontalInterpData["NetCDF_overlap"]["n_lon"],
        times=HorizontalInterpData["NetCDF_overlap"]["times"],
    )

    nessy_overlap.to_netcdf(filepath_overlap)


class TestHorizontalInterpolationNN:
    """Test the nearest neighbor interpolation method with and without overlap between source and destination grids."""

    def test_plain_interpolation(self, filepath_base, filepath_no_overlap):
        """Test that the nearest neighbor interpolation method produces
        expected results when there is no overlap between source and destination grids.
        """

        nessy_base = load_fresh_nes(filepath_base)
        nessy_no_overlap = load_fresh_nes(filepath_no_overlap)

        # Plain nn interpolation
        try:
            nessy_interp = nessy_base.interpolate_horizontal(dst_grid=nessy_no_overlap, kind="NN")
        except Exception as e:
            pytest.fail(f"Error during interpolation with method 'NN' and no overlap file: {e}")

        # Checks
        assert nessy_interp is not None, "Interpolation result should not be None."
        assert "var1" in nessy_interp.variables, "Interpolated NES should contain variable 'var1'."
        assert np.allclose(
            nessy_interp.variables["var1"]["data"].squeeze(),
            HorizontalExpectedData["NN"],
        ), "The interpolated values do not match the expected values for method 'NN' with no overlap file."

    def test_create_weight_matrix(self, filepath_base, filepath_no_overlap, filepath_nn_wm):
        """
        Test that the weight matrix for nearest neighbor interpolation
        can be created without errors and saved to a file.
        """

        nessy_base = load_fresh_nes(filepath_base)
        nessy_no_overlap = load_fresh_nes(filepath_no_overlap)

        # Create weight matrix
        try:
            _ = nessy_base.interpolate_horizontal(
                dst_grid=nessy_no_overlap,
                kind="NN",
                weight_matrix_path=filepath_nn_wm,
                only_create_wm=True,
            )
        except Exception as e:
            pytest.fail(f"Error during weight matrix creation: {e}")

        # Save weight matrix to file
        assert os.path.exists(filepath_nn_wm), "Weight matrix file should exist after creation."

    def test_use_weight_matrix(self, filepath_base, filepath_no_overlap, filepath_nn_wm):
        """
        Test that the weight matrix for nearest neighbor interpolation
        can be used without errors and produces expected results.
        """

        nessy_base = load_fresh_nes(filepath_base)
        nessy_no_overlap = load_fresh_nes(filepath_no_overlap)
        nessy_wm = load_fresh_nes(filepath_nn_wm)

        assert nessy_wm is not None, "Weight matrix NES should not be None."
        assert "weight" in nessy_wm.variables, "Weight matrix NES should contain 'weight' variable."
        assert "idx" in nessy_wm.variables, "Weight matrix NES should contain 'weight' and 'idx' variables."
        assert np.allclose(
            nessy_wm.variables["weight"]["data"].squeeze(),
            HorizontalExpectedData["NN_wm"]["weight"],
        ), "Weight matrix values do not match expected values."
        assert np.array_equal(
            nessy_wm.variables["idx"]["data"].squeeze(),
            HorizontalExpectedData["NN_wm"]["idx"],
        ), "Weight matrix indices do not match expected values."

        # Use weight matrix
        try:
            interp_nes = nessy_base.interpolate_horizontal(dst_grid=nessy_no_overlap, kind="NN", wm=nessy_wm)
        except Exception as e:
            pytest.fail(f"Error during weight matrix usage: {e}")

        assert interp_nes is not None, "Interpolated NES using weight matrix should not be None."
        assert "var1" in interp_nes.variables, "Interpolated NES using weight matrix should contain variable 'var1'."
        assert np.allclose(
            interp_nes.variables["var1"]["data"].squeeze(),
            HorizontalExpectedData["NN"],
        ), (
            "The interpolated values using the weight matrix do not match "
            "the expected values for method 'NN' with no overlap file."
        )


class TestHorizontalInterpolationConservative:
    """Test the conservative interpolation method with and without overlap between source and destination grids."""

    def test_plain_interpolation(self, filepath_base, filepath_overlap):
        """
        Test that the conservative interpolation method produces expected
        results when there is overlap between source and destination grids.
        """

        nessy_base = load_fresh_nes(filepath_base)
        nessy_overlap = load_fresh_nes(filepath_overlap)

        # Plain conservative interpolation
        try:
            nessy_interp = nessy_base.interpolate_horizontal(dst_grid=nessy_overlap, kind="Conservative")
        except Exception as e:
            pytest.fail(f"Error during interpolation with method 'Conservative' and overlap: {e}")

        # Checks
        assert nessy_interp is not None, "Interpolation result should not be None."
        assert "var1" in nessy_interp.variables, "Interpolated NES should contain variable 'var1'."
        assert np.allclose(
            nessy_interp.variables["var1"]["data"].squeeze(),
            HorizontalExpectedData["Conservative"],
        ), "The interpolated values do not match the expected values for method 'Conservative' with overlap file."

    def test_create_weight_matrix(self, filepath_base, filepath_overlap, filepath_conservative_wm):
        """
        Test that the weight matrix for conservative interpolation
        can be created without errors and saved to a file.
        """

        nessy_base = load_fresh_nes(filepath_base)
        nessy_overlap = load_fresh_nes(filepath_overlap)
        # Create weight matrix
        try:
            _ = nessy_base.interpolate_horizontal(
                dst_grid=nessy_overlap,
                kind="Conservative",
                weight_matrix_path=filepath_conservative_wm,
                only_create_wm=True,
            )
        except Exception as e:
            pytest.fail(f"Error during weight matrix creation: {e}")

        assert os.path.exists(filepath_conservative_wm), "Weight matrix file should exist after creation."

    def test_use_weight_matrix(self, filepath_base, filepath_overlap, filepath_conservative_wm):
        """
        Test that the weight matrix for conservative interpolation
        can be used without errors and produces expected results.
        """

        nessy_base = load_fresh_nes(filepath_base)
        nessy_overlap = load_fresh_nes(filepath_overlap)
        nessy_wm = load_fresh_nes(filepath_conservative_wm)

        assert nessy_wm is not None, "Weight matrix NES should not be None."
        assert "weight" in nessy_wm.variables, "Weight matrix NES should contain 'weight' variable."
        assert "idx" in nessy_wm.variables, "Weight matrix NES should contain 'idx' variable."

        # Use weight matrix
        try:
            interp_nes = nessy_base.interpolate_horizontal(
                dst_grid=nessy_overlap,
                kind="Conservative",
                wm=nessy_wm,
            )
        except Exception as e:
            pytest.fail(f"Error during weight matrix usage: {e}")

        assert interp_nes is not None, "Interpolated NES using weight matrix should not be None."
        assert "var1" in interp_nes.variables, "Interpolated NES using weight matrix should contain variable 'var1'."


# ! HYBRID OVERLAPPING ERRORS
# ! CHECK METHOD WITH AND WITHOUT FLUX
# ! TEST REAL CASE WITH REAL DATA
