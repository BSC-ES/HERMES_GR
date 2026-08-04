# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import numpy as np
import pytest

import nes
from nes.tests.utils import DummyData, load_fresh_nes

N_CELLS = 2
FILENAME = "test_expand_contract.nc"
ExpandContractData = DummyData.ExpandContract


@pytest.fixture(scope="module")
def tmp_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("expand_contract")


@pytest.fixture(scope="module")
def filepath(tmp_dir):
    return str(tmp_dir / FILENAME)


@pytest.fixture(scope="module", autouse=True)
def setup_test_file(filepath):
    """Create the test netCDF file before any tests run."""

    nessy = nes.create_nes(
        comm=None,
        info=False,
        projection=ExpandContractData["projection"],
        lat_orig=ExpandContractData["lat_orig"],
        lon_orig=ExpandContractData["lon_orig"],
        inc_lat=ExpandContractData["inc_lat"],
        inc_lon=ExpandContractData["inc_lon"],
        n_lat=ExpandContractData["n_lat"],
        n_lon=ExpandContractData["n_lon"],
        times=ExpandContractData["times"],
    )

    # Add dummy variable data with deterministic values
    nessy.variables = {
        "var1": {
            "data": ExpandContractData["var1"]["data"],
            "dimensions": ExpandContractData["var1"]["dimensions"],
            "units": ExpandContractData["var1"]["units"],
        }
    }
    nessy.to_netcdf(filepath)


class TestExpandContract:
    """Tests for the expand and contract methods of the NES class."""

    def test_expand(self, filepath):
        """Test expanding method."""

        nessy = load_fresh_nes(filepath)

        # Gather original data for comparison
        original_data = nessy.variables["var1"]["data"].copy()

        # Expand
        nessy.expand(n_cells=N_CELLS)
        post_expand_data = nessy.variables["var1"]["data"].copy()

        # Check for shape
        assert (
            post_expand_data.shape[2] == original_data.shape[2] + 2 * N_CELLS
            and post_expand_data.shape[3] == original_data.shape[3] + 2 * N_CELLS
        ), "Expanded data shape does not match expected shape "
        assert post_expand_data.shape[3] == original_data.shape[3] + 2 * N_CELLS, (
            "Expanded data shape does not match expected shape"
        )

        # Check that data in the center of post expand matches original data
        assert np.allclose(post_expand_data[:, :, N_CELLS:-N_CELLS, N_CELLS:-N_CELLS], original_data), (
            "Expanded data in the center does not match original data"
        )

        # now the opposite that is all zeros
        assert (
            np.allclose(post_expand_data[:, :, :N_CELLS, :], 0)
            and np.allclose(post_expand_data[:, :, -N_CELLS:, :], 0)
            and np.allclose(post_expand_data[:, :, :, :N_CELLS], 0)
            and np.allclose(post_expand_data[:, :, :, -N_CELLS:], 0)
        ), "Expanded data in the new areas is not all zeros"

    def test_contract(self, filepath):
        """Test contracting method."""

        nessy = load_fresh_nes(filepath)

        # Gather original data for comparison
        original_data = nessy.variables["var1"]["data"].copy()

        # Contract
        nessy.contract(n_cells=N_CELLS)
        post_contract_data = nessy.variables["var1"]["data"].copy()

        # Check for shape
        assert (
            post_contract_data.shape[2] == original_data.shape[2] - 2 * N_CELLS
            and post_contract_data.shape[3] == original_data.shape[3] - 2 * N_CELLS
        ), "Contracted data shape does not match expected shape"

        # Check that data in the center of post contract matches original data
        assert np.allclose(post_contract_data, original_data[:, :, N_CELLS:-N_CELLS, N_CELLS:-N_CELLS]), (
            "Contracted data in the center does not match original data"
        )

    def test_expand_contract(self, filepath):
        """Test that expanding and then contracting returns to original state."""

        nessy = load_fresh_nes(filepath)

        # Gather original data for comparison
        original_data = nessy.variables["var1"]["data"].copy()

        # Expand and then contract
        nessy.expand(n_cells=N_CELLS)
        nessy.contract(n_cells=N_CELLS)
        post_expand_contract_data = nessy.variables["var1"]["data"].copy()

        # Check that we return to the original data
        assert np.allclose(post_expand_contract_data, original_data), (
            "Data after expand and contract does not match original data"
        )

    def test_contract_expand(self, filepath):
        """Test that contracting and then expanding returns to original state."""

        nessy = load_fresh_nes(filepath)

        # Gather original data for comparison
        original_data = nessy.variables["var1"]["data"].copy()

        # Contract and then expand
        nessy.contract(n_cells=N_CELLS)
        nessy.expand(n_cells=N_CELLS)
        post_contract_expand_data = nessy.variables["var1"]["data"].copy()

        # Check that expanded data in the center matches original data
        assert np.allclose(
            post_contract_expand_data[:, :, N_CELLS:-N_CELLS, N_CELLS:-N_CELLS],
            original_data[:, :, N_CELLS:-N_CELLS, N_CELLS:-N_CELLS],
        ), "Expanded data in the center does not match original data"

        # Check that the new expanded areas are zeros
        assert (
            np.allclose(post_contract_expand_data[:, :, :N_CELLS, :], 0)
            and np.allclose(post_contract_expand_data[:, :, -N_CELLS:, :], 0)
            and np.allclose(post_contract_expand_data[:, :, :, :N_CELLS], 0)
            and np.allclose(post_contract_expand_data[:, :, :, -N_CELLS:], 0)
        ), "Expanded data in the new areas is not all zeros"
