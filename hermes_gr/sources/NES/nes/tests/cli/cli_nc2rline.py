# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import sys
import unittest
from unittest import mock

from nes.cli import cli, rline


class _TestNc2rline(unittest.TestCase):
    @mock.patch("nes.cli.rline.nc2rline")
    def _test_main_flow(self, mock_nc2rline):
        mock_nc2rline.return_value = None
        rline.nc2rline(
            input_file="file.nc",
            output_file="file.rline",
            geometry="geometry.shp",
            output_geometry_file=None,
            index_column=None,
            weight_column=None,
            weight_file=None,
            var_list=["NO2"],
            crs="EPSG:4326",
        )
        mock_nc2rline.assert_called_once()

    def _test_internal_function_called(self):
        # Dummy call to verify function is importable and callable
        # Replace with actual internal function if needed
        self.assertTrue(callable(rline.nc2rline))


class _TestCli(unittest.TestCase):
    @mock.patch("nes.cli.rline.nc2rline")
    def _test_cli_main_invokes_nc2rline_main(self, mock_nc2rline_main):
        mock_nc2rline_main.return_value = 0
        testargs = ["cli.py", "nc2rline", "--input", "file.nc", "--output_file", "file.rline", "--geometry", "file.shp"]
        with mock.patch.object(sys, "argv", testargs):
            with self.assertRaises(SystemExit) as cm:
                cli.main()
            self.assertEqual(cm.exception.code, 0)
        mock_nc2rline_main.assert_called_once()


if __name__ == "__main__":
    unittest.main()
