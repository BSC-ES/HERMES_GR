# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import pytest

import nes
from nes.tests.utils import DummyData, load_fresh_nes

FILENAME = "test_vertical_interp.nc"
VertInterpData = DummyData.VerticalInterp


@pytest.fixture(scope="module")
def tmp_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("vertical_interp")


@pytest.fixture(scope="module")
def filepath(tmp_dir):
    return str(tmp_dir / FILENAME)


@pytest.fixture(scope="module", autouse=True)
def setup_test_file(filepath):
    """Create the test netCDF file before any tests run."""

    nessy = nes.create_nes(
        comm=None,
        info=False,
        projection=VertInterpData["projection"],
        lat_orig=VertInterpData["lat_orig"],
        lon_orig=VertInterpData["lon_orig"],
        inc_lat=VertInterpData["inc_lat"],
        inc_lon=VertInterpData["inc_lon"],
        n_lat=VertInterpData["n_lat"],
        n_lon=VertInterpData["n_lon"],
        times=VertInterpData["times"],
    )
    nessy.set_levels({"data": VertInterpData["levels"], "units": "", "positive": "up"})

    # Add dummy variable data with deterministic values
    nessy.variables = {
        "var1": {
            "data": VertInterpData["var1"]["data"],
            "dimensions": VertInterpData["var1"]["dimensions"],
            "units": VertInterpData["var1"]["units"],
        }
    }
    nessy.to_netcdf(filepath)


def test_vertical_interpolation(filepath):
    """
    Test that interpolate_vertical correctly interpolates variable data
    to new vertical levels using different methods and extrapolation options.
    """

    nessy = load_fresh_nes(filepath)

    for lev in [
        VertInterpData["interpolation_shorter"],
        VertInterpData["interpolation_longer"],
    ]:
        for interp_method in VertInterpData["interpolation_methods"]:
            for extrapolate_option in VertInterpData["extrapolate_options"]:
                try:
                    interp_nes = nessy.interpolate_vertical(
                        lev,
                        info=False,
                        kind=interp_method,
                        extrapolate=extrapolate_option,
                    )
                except Exception as e:
                    pytest.fail(
                        f"Vertical interpolation with method '{interp_method}' "
                        f"and extrapolate option '{extrapolate_option}' failed with error: {e}"
                    )
                assert interp_nes is not None, "interpolate_vertical should return a new NES object, but returned None."
                assert interp_nes.variables["var1"]["data"].shape[0] == len(VertInterpData["times"]), (
                    "The time dimension should remain unchanged after vertical interpolation."
                )
                assert interp_nes.variables["var1"]["data"].shape[1] == len(lev), (
                    "The vertical dimension should match the length of the new levels after interpolation."
                )
                assert interp_nes.variables["var1"]["data"].shape[2] == VertInterpData["n_lat"], (
                    "The latitude dimension should remain unchanged after vertical interpolation."
                )
                assert interp_nes.variables["var1"]["data"].shape[3] == VertInterpData["n_lon"], (
                    "The longitude dimension should remain unchanged after vertical interpolation."
                )


# ! CHECK VALUES ONLY FOR LINEAR INTERPOLATION WITH ONE TEST CASE
