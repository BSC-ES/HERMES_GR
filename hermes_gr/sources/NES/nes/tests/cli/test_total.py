# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

"""Integration tests for variable totals using a real NetCDF file."""

import sys

import numpy as np
import pytest
from netCDF4 import Dataset

from nes.cli.cli import main
from nes.cli.total import total


@pytest.fixture
def input_file(tmp_path):
    path = tmp_path / "totals.nc"
    with Dataset(path, "w") as nc:
        for name, size in [("time", 2), ("lev", 1), ("lat", 2), ("lon", 2)]:
            nc.createDimension(name, size)
            var = nc.createVariable(name, "f8", (name,))
            var[:] = np.arange(size)
        nc.variables["time"].units = "hours since 2026-10-19 00:00:00"
        nc.variables["lat"].units = "degrees_north"
        nc.variables["lon"].units = "degrees_east"
        nox = nc.createVariable("NOx", "f8", ("time", "lev", "lat", "lon"), fill_value=-999)
        nox.units = "kg"
        nox[:] = np.array([1, 2, 3, 4, 5, np.nan, -999, 8]).reshape(2, 1, 2, 2)
        co = nc.createVariable("CO", "f8", ("time", "lev", "lat", "lon"))
        co[:] = 2
    return str(path)


def test_all_variables(input_file, capsys):
    total(input_file)
    output = capsys.readouterr().out
    assert "NOx: 23.0 kg" in output
    assert "CO: 16.0 unknown units" in output
    assert "lat:" not in output


@pytest.mark.parametrize("flag", ["-l", "--var_list"])
def test_cli_selection(input_file, capsys, monkeypatch, flag):
    monkeypatch.setattr(sys, "argv", ["nes", "total", "-i", input_file, flag, "NOx"])
    main()
    output = capsys.readouterr().out
    assert "NOx: 23.0 kg" in output
    assert "CO:" not in output


def test_missing_variable(input_file):
    with pytest.raises(ValueError, match="missing"):
        total(input_file, ["missing"])
