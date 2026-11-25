# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import numpy as np
import pytest

try:
    from mpi4py import MPI
    from nes.default_nes import Nes
except ImportError as exc:
    MPI = None
    Nes = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

pytestmark = pytest.mark.skipif(IMPORT_ERROR is not None, reason=str(IMPORT_ERROR))


def _make_minimal_nes(parallel_method="Y"):
    nessy = Nes.__new__(Nes)
    nessy.comm = MPI.COMM_SELF
    nessy.rank = 0
    nessy.master = True
    nessy.size = 1
    nessy.info = False
    nessy.parallel_method = parallel_method
    nessy.balanced = False
    nessy.dataset = None
    nessy.variables = {}
    nessy.cell_measures = {}
    nessy._full_time = [0]
    nessy._full_lev = {"data": np.array([0, 1])}
    nessy._full_lat = {"data": np.arange(2)}
    nessy._full_lon = {"data": np.arange(3)}
    nessy._full_time_bnds = None
    nessy._full_lat_bnds = None
    nessy._full_lon_bnds = None
    return nessy


def test_gather_data_keeps_optimized_path_for_different_4d_shapes():
    """4D variables with different local shapes should share the optimized gather path."""

    nessy = _make_minimal_nes()
    variables = {
        "surface": {
            "data": np.arange(6, dtype=np.float64).reshape(1, 1, 2, 3),
            "units": "kg m-2 s-1",
        },
        "profile": {
            "data": np.arange(12, dtype=np.float64).reshape(1, 2, 2, 3),
            "units": "kg m-2 s-1",
        },
    }

    gathered = nessy._gather_data(variables)

    assert np.array_equal(gathered["surface"]["data"], variables["surface"]["data"])
    assert np.array_equal(gathered["profile"]["data"], variables["profile"]["data"])
    assert set(nessy._gather_data_new_cache.keys()) == {
        ("4d", "Y", variables["surface"]["data"].shape),
        ("4d", "Y", variables["profile"]["data"].shape),
    }
