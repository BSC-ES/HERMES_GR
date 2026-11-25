# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

from datetime import datetime
from unittest.mock import Mock

import numpy as np
import pytest

try:
    from nes import LCCNes
    from nes.formats.cmaq_format import set_global_attributes
    from nes.formats.cmaq_format import create_dimensions as cmaq_dimensions
    from nes.formats.wrf_chem_format import set_global_attributes as wrf_attributes
    from nes.formats.wrf_chem_format import create_dimensions as wrf_dimensions
except ImportError as exc:
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

pytestmark = pytest.mark.skipif(IMPORT_ERROR is not None, reason=str(IMPORT_ERROR))


@pytest.mark.parametrize("master", [True, False])
@pytest.mark.parametrize("output_format", ["CMAQ", "WRF_CHEM"])
def test_attributes_use_full_coordinates_on_every_rank(master, output_format):
    grid = LCCNes.__new__(LCCNes)
    x = {"data": np.array([1000.0, 3000.0, 5000.0])}
    y = {"data": np.array([2000.0, 6000.0])}
    grid._full_x = x if master else None
    grid._full_y = y if master else None
    grid.get_full_x = Mock(return_value=x)
    grid.get_full_y = Mock(return_value=y)
    grid.global_attrs = {}
    grid.time = [datetime(2026, 10, 7)]
    grid.lev = {"data": np.array([1.0])}
    grid.variables = {"NO2": {}}
    grid.projection_data = {
        "standard_parallel": [30.0, 60.0],
        "longitude_of_central_meridian": 10.0,
        "latitude_of_projection_origin": 45.0,
    }

    if output_format == "CMAQ":
        set_global_attributes(grid)
        assert grid.global_attrs["NCOLS"] == 3
        assert grid.global_attrs["NROWS"] == 2
        assert grid.global_attrs["XORIG"] == 0.0
        assert grid.global_attrs["YORIG"] == 0.0
        assert grid.global_attrs["XCELL"] == 2000.0
        assert grid.global_attrs["YCELL"] == 4000.0
    else:
        wrf_attributes(grid)
        assert grid.global_attrs["WEST-EAST_GRID_DIMENSION"] == 4
        assert grid.global_attrs["SOUTH-NORTH_GRID_DIMENSION"] == 3
        assert grid.global_attrs["WEST-EAST_PATCH_END_UNSTAG"] == 3
        assert grid.global_attrs["SOUTH-NORTH_PATCH_END_UNSTAG"] == 2
        assert grid.global_attrs["DX"] == 2000.0
        assert grid.global_attrs["DY"] == 4000.0

    grid.get_full_x.assert_called_once_with()
    grid.get_full_y.assert_called_once_with()


@pytest.mark.parametrize("output_format", ["CMAQ", "WRF_CHEM"])
def test_dimensions_use_full_coordinates_on_non_master_rank(output_format):
    grid = LCCNes.__new__(LCCNes)
    grid._full_x = None
    grid._full_y = None
    grid.get_full_x = Mock(return_value={"data": np.arange(3)})
    grid.get_full_y = Mock(return_value={"data": np.arange(2)})
    grid.get_full_times = Mock(return_value=[datetime(2026, 10, 7)])
    grid.get_full_levels = Mock(return_value={"data": np.array([1.0])})
    grid.variables = {"NO2": {}}
    dataset = Mock()

    if output_format == "CMAQ":
        cmaq_dimensions(grid, dataset)
        expected_dimensions = {"COL": 3, "ROW": 2}
    else:
        wrf_dimensions(grid, dataset)
        expected_dimensions = {"west_east": 3, "south_north": 2}

    dimensions = dict(call.args for call in dataset.createDimension.call_args_list)
    for name, size in expected_dimensions.items():
        assert dimensions[name] == size
