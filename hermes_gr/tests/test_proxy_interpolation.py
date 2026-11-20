# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from hermes_gr.config.proxy_catalogue import to_nes_proxies
from hermes_gr.modules.emision_inventories.emission_inventory import EmissionInventory


class TestProxyInterpolation(unittest.TestCase):

    def test_catalogue_types_and_priority(self):
        """
        This test ensures that the `to_nes_proxies` function correctly converts various proxy types
        to the expected NES proxy format and maintains the correct priority order.
        """
        # Input
        proxies = [
            {"type": "vector_line", "path": "roads.geojson", "weight_column": "traffic"},
            {"type": "raster_categorical", "path": "land.tif", "categories": [122]},
            {"type": "raster_numerical_categorized", "path_categorical": "mask.tif",
             "path_numerical": "values.tif", "categories": [30]},
            {"type": "vector_point", "path": "plants.geojson"},
            {"type": "vector_polygon", "path": "areas.geojson"},
            {"type": "raster_numerical", "path": "population.tif"},
        ]
        # Check that the conversion to NES proxies is correct
        self.assertEqual(to_nes_proxies(proxies), [
            ("lines", "roads.geojson", None, "traffic"),
            ("raster_categorical", "land.tif", [122]),
            ("raster_categorical_numerical", ("mask.tif", "values.tif"), [30]),
            ("points", "plants.geojson", None, "weight"),
            ("polygons", "areas.geojson", None, "weight"),
            ("raster", "population.tif", None),
        ])

    def test_proxy_matrix_does_not_reuse_plain_or_other_proxy_matrix(self):
        """
        Test that the proxy matrix does not reuse the plain weight matrix or other proxy matrices.
        """
        # Create a new EmissionInventory instance without calling __init__ to set up the test scenario.
        inventory = EmissionInventory.__new__(EmissionInventory)
        inventory.auxiliary_files_path = "/aux"
        inventory.inventory_name = "CAMS"
        inventory.sector = "F1"
        inventory.reference_year = "2024"
        inventory.input_frequency = "yearly"
        inventory.proxies = []

        # Get the path to the plain weight matrix before any proxies are set.
        plain_path = inventory.get_weight_matrix_path()

        # Assert that the plain weight matrix path is as expected and the auxiliary emissions file is correct.
        self.assertEqual(plain_path, "/aux/Weight_Matrix_CAMS.nc")
        self.assertEqual(inventory.get_aux_emis_file_name(), "/aux/CAMS_F1_2024.nc")

        # Set the first proxy and get the corresponding weight matrix and emissions paths.
        inventory.proxies = [{"type": "vector_point", "path": "a.geojson"}]
        first_path = inventory.get_weight_matrix_path()
        first_emissions_path = inventory.get_aux_emis_file_name()
        inventory.proxies = [{"type": "vector_point", "path": "b.geojson"}]

        # Assert that changing the proxy results in a different weight matrix and emissions path.
        self.assertNotEqual(first_path, inventory.get_weight_matrix_path())
        self.assertNotEqual(first_path, plain_path)
        self.assertNotEqual(first_emissions_path, inventory.get_aux_emis_file_name())
        self.assertIn("_proxy_", first_emissions_path)

        # Assert that the emissions path for the first proxy contains the "_proxy_" substring.
        inventory.proxies = [{"type": "vector_point", "path": "a.geojson"}]
        inventory.sector = "F2"

        # Assert that reverting to the first proxy still results in a different weight matrix due to the sector change.
        self.assertNotEqual(first_path, inventory.get_weight_matrix_path())

    def test_init_regrid_passes_mixed_proxies_to_nes(self):
        """
        This test ensures that when the inventory has a mix of vector line and raster categorical proxies,
        the init_regrid method converts them to the expected NES proxy format and passes them to the
        interpolate_horizontal function along with the flux argument.
        """
        # Test that the init_regrid method correctly passes mixed proxy types to the NES
        # interpolate_horizontal function.
        inventory = EmissionInventory.__new__(EmissionInventory)
        inventory.grid = SimpleNamespace(comm=None, balanced=False, parallel_method="T")
        inventory.src_emis_file_list = ["input.nc"]
        inventory.inventory_name = "CAMS"
        inventory.sector = "F1"
        inventory.proxies = [
            {"type": "vector_line", "path": "roads.geojson", "weight_column": "traffic"},
            {"type": "raster_categorical", "path": "land.tif", "categories": [122]},
        ]
        emissions = SimpleNamespace()
        emissions.calculate_grid_area = unittest.mock.Mock()
        emissions.interpolate_horizontal = unittest.mock.Mock(return_value="matrix")

        # Patch the necessary functions and methods to simulate the NES environment and test the init_regrid method.
        # This helps isolate the behavior of init_regrid
        with patch("hermes_gr.modules.emision_inventories.emission_inventory.open_netcdf",
                   return_value=emissions), patch.object(inventory, "_record_initialization_time"), \
                patch("hermes_gr.modules.emision_inventories.emission_inventory.log_message"), \
                patch("hermes_gr.modules.emision_inventories.emission_inventory.os.path.exists",
                      return_value=False):
            result = inventory.init_regrid("/aux/matrix.nc")

        # The result of init_regrid should be the matrix returned by the mocked interpolate_horizontal function.
        self.assertEqual(result, "matrix")
        self.assertEqual(emissions.interpolate_horizontal.call_args.kwargs["proxy"], [
            ("lines", "roads.geojson", None, "traffic"),
            ("raster_categorical", "land.tif", [122]),
        ])
        self.assertTrue(emissions.interpolate_horizontal.call_args.kwargs["flux"])
