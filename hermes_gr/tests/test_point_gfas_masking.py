# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

"""Mass-balance regressions for the real GFAS point allocation methods.

Execute source class definitions without importing the optional MPI/NES stack.
NetCDF I/O, the country lookup and MPI communication are controlled doubles;
mask combination, vertical weights and emission allocation are real methods.
"""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
ROOT = Path(__file__).resolve().parents[2]
INVENTORIES = ROOT / "hermes_gr/modules/emision_inventories"


def load_class(path, name, namespace):
    tree = ast.parse(path.read_text(), filename=str(path))
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class Dataset:
    """In-memory fields with only the NES operations used by these methods."""

    def __init__(self, variables, count=1, area=2):
        self.variables = variables
        self.lat = {"data": np.zeros(count)}
        self.lon = {"data": np.zeros(1)}
        self.comm = Mock()
        self.comm.bcast.side_effect = lambda value, root: value
        self.write_axis_limits = dict(y_min=0, y_max=count, x_min=0, x_max=1)
        self.cell_measures = {"cell_area": {"data": np.full((count, 1), area, dtype=float)}}
        self.sel = Mock()
        self.keep_vars = Mock()
        self.load = Mock()
        self.calculate_grid_area = Mock()
        self.to_dtype = Mock()

    def free_vars(self, name):
        del self.variables[name]

    def get_fids(self):
        return np.arange(len(self.lat["data"])).reshape(-1, 1)

    def copy(self, **kwargs):
        return Dataset({}, len(self.lat["data"]), area=4)


@pytest.mark.filterwarnings("error::DeprecationWarning")
@pytest.mark.parametrize("class_name", ["PointGfasEmissionInventory", "PointGfasHourlyEmissionInventory"])
@pytest.mark.parametrize("geographic,expected_scale", [
    (None, 1), (1, 1), (1.5, 1.5), (0.5, 0.5), (0, 0),
])
@pytest.mark.parametrize("heights,injection,vertical", [
    ([100., 200.], 200., [.5, .5]),
    ([100., 400.], 400., [.25, .75]),
    ([100., 200., 300.], 200., [.5, .5, 0.]),
])
def test_point_gfas_geographic_and_vertical_mass_balance(
        class_name, geographic, expected_scale, heights, injection, vertical):
    """A factor is applied once, exclusions stay zero and empty levels stay empty.

    None represents both mask options omitted. Numeric values represent the
    already combined factor_mask/regrid_mask returned by the shared Masking
    implementation. That combination is tested independently below.
    """
    namespace = dict(
        np=np, pd=pd, DataFrame=pd.DataFrame, os=os, precision=np.float64,
        EmissionInventory=object, Nes=Dataset, MPI=SimpleNamespace(COMM_SELF=None),
        log_message=Mock(), get_time_stamp=lambda: 0, GFAS_BUFFER=.1, GRID_BUFFER=.1,
    )
    load_class(INVENTORIES / "gfas_emission_inventory.py", "GfasEmissionInventory", namespace)
    point = load_class(INVENTORIES / "point_gfas_emission_inventory.py", "PointGfasEmissionInventory", namespace)
    load_class(INVENTORIES / "point_gfas_hourly_emission_inventory.py", "PointGfasHourlyEmissionInventory", namespace)
    shape = (1, 1, 1, 1)
    source = Dataset({"co": {"data": np.full(shape, 50.)}, "apt": {"data": np.full(shape, injection)}})
    cached_indices = Dataset({"idx": {"data": np.zeros(shape)}})
    namespace["open_netcdf"] = lambda *args, **kwargs: (
        cached_indices if kwargs.get("path") == __file__ else source
    )
    mask_array = None if geographic is None else np.full(shape, geographic, dtype=float)
    namespace["Masking"] = Mock(return_value=SimpleNamespace(mask_factors=mask_array))
    grid = Dataset({})
    grid.lev = {"data": np.array(heights)}
    grid.master, grid.balanced, grid.parallel_method = True, False, "Y"
    grid.comm.Get_size.return_value = 1
    grid.get_full_latitudes_boundaries = lambda: {"data": np.array([0., 1.])}
    grid.get_full_longitudes_boundaries = grid.get_full_latitudes_boundaries
    inventory = namespace[class_name].__new__(namespace[class_name])
    inventory.grid = grid
    inventory.layer_widths = np.diff([0.] + heights)
    inventory.method, inventory.approach, inventory.altitude = "sovief", "uniform", "apt"
    inventory.input_pollutants, inventory.src_emis_file_list = ["co"], ["source"]
    inventory.auxiliary_files_path, inventory.inventory_name = "/unused", "GFASv12"
    inventory.factors_mask, inventory.regrid_mask, inventory.countries_shapefile = None, None, "controlled"
    inventory._record_initialization_time = Mock()
    inventory.temporal_factors = None
    # Reuse controlled destination indices, exercising actual factor defaults.
    inventory.weight_matrix = inventory.init_regrid(__file__)
    inventory.read_input_emissions()
    inventory.do_regrid()
    # Bypass the hourly reload hook: both variants' shared allocation is under test.
    point.prepare_time_step_emissions(inventory)
    actual_flux = inventory.get_4d_emission("co")[0, :, 0, 0]
    actual_mass = actual_flux * 4  # Destination cell area; source mass rate is 50 * 2.
    np.testing.assert_allclose(actual_mass, 100 * expected_scale * np.array(vertical))
    np.testing.assert_allclose(actual_mass.sum(), 100 * expected_scale)
    np.testing.assert_allclose(inventory.weight_matrix["factor"], expected_scale)


@pytest.mark.parametrize("scaling,inclusion,expected", [
    (None, None, None), (1., 1., 1.), (1.5, 1., 1.5), (.5, 1., .5),
    (0., 1., 0.), (1.5, 0., 0.), (None, 0., 0.),
    ([1.5, .5, 1.], [1., 1., 0.], [1.5, .5, 0.]),
])
def test_shared_masks_combine_multiplicatively(scaling, inclusion, expected):
    """Verify the actual shared combination using controlled per-country arrays."""
    namespace = {}
    masking = load_class(ROOT / "hermes_gr/modules/masking/masking.py", "Masking", namespace)
    mask = masking.__new__(masking)
    mask.factors_mask_values, mask.regrid_mask_values = scaling, inclusion
    mask.get_countries_data = Mock(return_value=object())
    mask.calculate_mask_array = lambda values, *args, **kwargs: (
        None if values is None else np.atleast_1d(values).astype(float)
    )
    actual = mask.get_mask_factors(None, None, None, None)
    if expected is None:
        assert actual is None
    else:
        np.testing.assert_allclose(actual, expected)
