#!/usr/bin/env python

# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import os
import tempfile
import unittest

import numpy as np
from configargparse import ArgumentTypeError

from hermes_gr.config import parse_vertical_description, precision


class TestVerticalDescription(unittest.TestCase):
    def test_parse_vertical_description_from_list(self):
        result = parse_vertical_description("48, 104, 168")

        np.testing.assert_array_equal(result, np.array([48, 104, 168], dtype=precision))
        self.assertEqual(result.dtype, np.dtype(precision))

    def test_parse_vertical_description_from_single_value(self):
        result = parse_vertical_description("20000")

        np.testing.assert_array_equal(result, np.array([20000], dtype=precision))
        self.assertEqual(result.dtype, np.dtype(precision))

    def test_parse_vertical_description_from_historical_csv(self):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w")
        try:
            tmp.write("Ilayer,height_magl\n1,39\n2,78\n3,119\n")
            tmp.close()

            result = parse_vertical_description(tmp.name)

            np.testing.assert_array_equal(result, np.array([39, 78, 119], dtype=precision))
            self.assertEqual(result.dtype, np.dtype(precision))
        finally:
            if not tmp.closed:
                tmp.close()
            os.remove(tmp.name)

    def test_parse_vertical_description_csv_requires_height_magl(self):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w")
        try:
            tmp.write("Ilayer,height\n1,39\n")
            tmp.close()

            with self.assertRaises(ArgumentTypeError):
                parse_vertical_description(tmp.name)
        finally:
            if not tmp.closed:
                tmp.close()
            os.remove(tmp.name)

    def test_parse_vertical_description_csv_requires_numeric_height_magl(self):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w")
        try:
            tmp.write("Ilayer,height_magl\n1,not_a_height\n")
            tmp.close()

            with self.assertRaises(ArgumentTypeError):
                parse_vertical_description(tmp.name)
        finally:
            if not tmp.closed:
                tmp.close()
            os.remove(tmp.name)


if __name__ == "__main__":
    unittest.main()
