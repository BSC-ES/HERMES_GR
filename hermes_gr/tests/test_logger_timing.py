# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import unittest
import importlib.util
from pathlib import Path

import pandas as pd

import hermes_gr


HAS_MPI4PY = importlib.util.find_spec("mpi4py") is not None
logger_config = None

if HAS_MPI4PY:
    LOGGER_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "logger_config.py"
    LOGGER_CONFIG_SPEC = importlib.util.spec_from_file_location("logger_config_for_tests", LOGGER_CONFIG_PATH)
    logger_config = importlib.util.module_from_spec(LOGGER_CONFIG_SPEC)
    LOGGER_CONFIG_SPEC.loader.exec_module(logger_config)


@unittest.skipUnless(HAS_MPI4PY, "mpi4py is required to test logger timing records")
class TestLoggerTiming(unittest.TestCase):

    def tearDown(self):
        hermes_gr.DEBUG = False
        hermes_gr.WRITE_TIME_LOG = False
        logger_config.time_records = pd.Series(
            dtype=float,
            index=pd.MultiIndex.from_tuples([], names=["section", "subsection"]),
            name=logger_config.my_rank,
        )

    def test_record_time_without_debug_mode(self):
        hermes_gr.DEBUG = False
        hermes_gr.WRITE_TIME_LOG = True
        logger_config.time_records = pd.Series(
            dtype=float,
            index=pd.MultiIndex.from_tuples([], names=["section", "subsection"]),
            name=logger_config.my_rank,
        )

        logger_config.record_time("HERMES", "TOTAL", 1.5)

        self.assertIn(("HERMES", "TOTAL"), logger_config.time_records.index)
        self.assertEqual(logger_config.time_records.loc[("HERMES", "TOTAL")], 1.5)


if __name__ == "__main__":
    unittest.main()
