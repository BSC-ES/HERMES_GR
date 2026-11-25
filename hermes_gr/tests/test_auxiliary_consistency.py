# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import unittest
from types import SimpleNamespace

import numpy as np

from hermes_gr.tools.auxiliary_consistency import check_aux_file
from hermes_gr.tools.auxiliary_consistency import check_grid
from hermes_gr.tools.auxiliary_consistency import get_expected_grid_shape


class FakeNes:
    write_axis_limits = {
        "t_min": None,
        "t_max": None,
        "z_min": None,
        "z_max": None,
        "y_min": 0,
        "y_max": 12,
        "x_min": None,
        "x_max": None,
    }
    lev = {"data": np.zeros((1,))}
    lat = {"data": np.zeros((12, 13))}
    lon = {"data": np.zeros((12, 13))}
    lat_bnds = {"data": np.zeros((12, 13, 4))}
    lon_bnds = {"data": np.zeros((12, 13, 4))}
    variables = {
        "NOx": {
            "units": "kg.h-1",
            "long_name": "Nitrogen oxides",
            "data": np.zeros((1, 12, 13)),
        },
        "CO": {
            "units": "kg.h-1",
            "data": np.zeros((1, 12, 13)),
        },
    }
    global_attrs = {
        "Conventions": "CF-1.7",
    }

    @staticmethod
    def get_full_shape():
        return 12, 13


class TestAuxiliaryConsistency(unittest.TestCase):
    def test_check_grid_passes_expected_shape(self):
        self.assertTrue(check_grid(FakeNes(), expected_shape=(12, 13)))

    def test_check_aux_file_passes_requested_checks(self):
        self.assertTrue(
            check_aux_file(
                FakeNes(),
                nes_object_to_compare=FakeNes(),
                var_list=["NOx", "CO"],
                global_attributes=["Conventions"],
                variable_attributes={
                    "NOx": ["units", "long_name"],
                },
            )
        )

    def test_check_aux_file_skips_not_requested_checks(self):
        self.assertTrue(check_aux_file(FakeNes(), var_list=["NOx"]))

    def test_check_aux_file_fails_on_missing_variable_attribute(self):
        variable_attributes = {
            "CO": ["long_name"],
        }

        self.assertFalse(check_aux_file(FakeNes(), variable_attributes=variable_attributes))

    def test_check_aux_file_checks_all_var_list_variable_attributes(self):
        self.assertTrue(check_aux_file(FakeNes(), var_list=["NOx"], variable_attributes=["units"]))

    def test_check_aux_file_fails_on_local_shape(self):
        class BadAuxiliary(FakeNes):
            lat = {"data": np.zeros((11, 13))}

        self.assertFalse(check_aux_file(BadAuxiliary(), nes_object_to_compare=FakeNes()))

    def test_get_expected_grid_shape_global(self):
        options = SimpleNamespace(domain_type="global", inc_lat=1.0, inc_lon=1.40625)

        self.assertEqual(get_expected_grid_shape(options), (180, 256))

    def test_get_expected_grid_shape_lcc(self):
        options = SimpleNamespace(domain_type="lcc", ny=100, nx=200)

        self.assertEqual(get_expected_grid_shape(options), (100, 200))


if __name__ == "__main__":
    unittest.main()
