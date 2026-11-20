# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

"""GFAS initialization contracts without emission datasets or an MPI runtime."""

import ast
from datetime import datetime
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

try:
    import pandas as pd
except ImportError:
    pd = None


SOURCE_ROOT = Path(__file__).resolve().parents[2]
INVENTORIES = SOURCE_ROOT / "hermes_gr" / "modules" / "emision_inventories"
ERROR_MESSAGE = (
    "GFAS area emission inventories are not currently supported "
    "by HERMES_GR. Use GFAS point emissions instead."
)


def load_class(path, class_name, namespace, methods=None):
    """Execute actual class definitions, excluding imports and optional unused methods.

    Only the dataset-loading base constructor, logger and grid arrays are stand-ins.
    Keeping the GFAS inheritance and constructors intact exercises the shared guard
    and its effect on point subclasses without importing NES or initializing MPI.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    if methods is not None:
        definition.body = [
            node for node in definition.body if isinstance(node, ast.FunctionDef) and node.name in methods
        ]
    module = ast.Module(body=[definition], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[class_name]


class TestGfasAreaUnsupported(unittest.TestCase):
    def setUp(self):
        self.base_init = Mock(return_value=None)
        base = type("EmissionInventory", (), {"__init__": self.base_init})
        self.namespace = {
            "__name__": __name__,
            "EmissionInventory": base,
            "re": re,
            "log_message": Mock(),
        }
        for filename, name in [
            ("gfas_emission_inventory.py", "GfasEmissionInventory"),
            ("gfas_hourly_emission_inventory.py", "GfasHourlyEmissionInventory"),
            ("point_gfas_emission_inventory.py", "PointGfasEmissionInventory"),
            ("point_gfas_hourly_emission_inventory.py", "PointGfasHourlyEmissionInventory"),
        ]:
            load_class(INVENTORIES / filename, name, self.namespace)

    def arguments(self, source_type, frequency, grid=None):
        return {
            "options": None,
            "grid": grid,
            "current_date": datetime(2026, 1, 1),
            "inventory_name": "GFASv12",
            "source_type": source_type,
            "sector": None,
            "pollutants": ["co"],
            "inputs_path": "/unused",
            "frequency": frequency,
            "vertical_output_profile": [100, 200],
            "p_vertical": "method=sovief,approach=uniform",
        }

    def assert_area_rejected(self, class_name, frequency):
        inventory_class = self.namespace[class_name]
        arguments = self.arguments("area", frequency)
        arguments["p_vertical"] = None
        with patch.object(inventory_class, "parse_profile_gfas", side_effect=AssertionError("Profile parsing reached")):
            with self.assertRaises(NotImplementedError) as raised:
                inventory_class(**arguments)
        self.assertEqual(str(raised.exception), ERROR_MESSAGE)
        self.base_init.assert_not_called()

    def test_daily_area_rejected_before_profile_grid_or_data_access(self):
        self.assert_area_rejected("GfasEmissionInventory", "daily")

    def test_hourly_area_rejected_before_profile_grid_or_data_access(self):
        self.assert_area_rejected("GfasHourlyEmissionInventory", "hourly")

    def assert_point_initializes(self, class_name, frequency):
        heights = MagicMock()
        grid = SimpleNamespace(lev={"data": heights})
        inventory = self.namespace[class_name](**self.arguments("point", frequency, grid=grid))
        self.base_init.assert_called_once()
        args, kwargs = self.base_init.call_args
        self.assertIs(args[1], grid)
        self.assertEqual(args[4], "point")
        self.assertEqual(args[8], frequency)
        self.assertEqual(inventory.method, "sovief")
        self.assertEqual(inventory.approach, "uniform")
        self.assertEqual(inventory.altitude, "apt")
        self.assertIsNone(inventory.dst_fid)
        self.assertIsNone(inventory.dst_level)
        self.assertIsNone(inventory.vertical)
        self.assertIsNone(kwargs["p_vertical"])
        heights.copy.assert_called_once_with()

    def test_daily_point_still_reaches_base_initialization(self):
        self.assert_point_initializes("PointGfasEmissionInventory", "daily")

    def test_hourly_point_still_reaches_base_initialization(self):
        self.assert_point_initializes("PointGfasHourlyEmissionInventory", "hourly")

    def assert_factory_rejects_area(self, frequency):
        from hermes_gr.config.proxy_catalogue import load_proxy_catalogue, select_proxies

        self.namespace.update(
            pd=pd, load_proxy_catalogue=load_proxy_catalogue,
            select_proxies=select_proxies,
        )
        driver_class = load_class(
            SOURCE_ROOT / "hermes_gr" / "hermes.py", "HermesGr", self.namespace, methods={"make_emission_list"}
        )
        driver = driver_class.__new__(driver_class)
        driver.grid = None
        driver.levels = [100, 200]
        driver.comm = Mock()
        driver.rescaling = None
        row = {
            "ei": "GFASv12", "sector": "", "ref_year": 2026, "active": 1,
            "factor_mask": "", "regrid_mask": "", "pollutants": "co", "path": "/unused",
            "frequency": frequency, "source_type": "area", "coverage": "global",
            "p_vertical": "", "p_month": "", "p_week": "", "p_day": "", "p_hour": "", "p_speciation": "",
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            cross_table = Path(temporary_directory) / "inventories.csv"
            pd.DataFrame([row]).to_csv(cross_table, sep=";", index=False)
            driver.options = SimpleNamespace(
                cross_table=str(cross_table), data_path="/unused", input_dir="/unused",
                start_date=datetime(2026, 1, 1), countries_shapefile=None,
            )
            with self.assertRaises(NotImplementedError) as raised:
                driver.make_emission_list()
        self.assertEqual(str(raised.exception), ERROR_MESSAGE)
        self.base_init.assert_not_called()
        driver.comm.Barrier.assert_not_called()

    @unittest.skipIf(pd is None, "pandas is required to exercise the cross-table factory")
    def test_factory_rejects_daily_gfas_area(self):
        self.assert_factory_rejects_area("daily")

    @unittest.skipIf(pd is None, "pandas is required to exercise the cross-table factory")
    def test_factory_rejects_hourly_gfas_area(self):
        self.assert_factory_rejects_area("hourly")


if __name__ == "__main__":
    unittest.main()
