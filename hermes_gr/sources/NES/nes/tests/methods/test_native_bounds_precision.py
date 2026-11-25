# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

"""Precision-aware native interval inference without MPI or NetCDF imports."""

import ast
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def native_bounds():
    """Load the actual interval helper independently of the native runtime."""
    path = Path(__file__).resolve().parents[2] / 'projections/_coordinates.py'
    tree = ast.parse(path.read_text(), filename=str(path))
    definition = next(node for node in tree.body if getattr(node, 'name', None) == 'native_bounds')
    namespace = dict(np=np)
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['native_bounds']


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('descending', [False, True])
@pytest.mark.parametrize('origin,step,count', [
    (-34.95, .15, 467), (-27., .15, 361), (40., .1, 41), (1000000., 1000.1, 8),
])
def test_regular_encoded_axes_keep_centres(native_bounds, dtype, descending, origin, step, count):
    """Accept uniform axes after storage rounding without changing their centres."""
    values = (origin + np.arange(count) * step).astype(dtype)
    if descending:
        values = values[::-1].copy()
    original = values.copy()
    result = native_bounds({'data': values}, -step if descending else step)
    np.testing.assert_array_equal(values, original)
    np.testing.assert_allclose(result.mean(axis=1), values, rtol=0, atol=1e-9)
    inferred = (float(values[-1]) - float(values[0])) / (count - 1)
    np.testing.assert_allclose(result[:, 1] - result[:, 0], inferred, rtol=1e-10, atol=1e-9)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('values', [[0, 1, 2.2], [-34.95, -34.8, -34.5], [3, 2, .5]])
def test_genuinely_nonuniform_axes_still_rejected(native_bounds, dtype, values):
    """Creation increments must not override incompatible coordinate values."""
    with pytest.raises(ValueError, match='Nonuniform native axes require explicit bounds'):
        native_bounds({'data': np.array(values, dtype=dtype)}, .15)


def test_small_individual_drifts_cannot_accumulate(native_bounds):
    """Reject curvature even if each adjacent difference is near the mean step."""
    values = np.arange(1000, dtype=float)
    values += .0001 * np.sin(np.linspace(0, np.pi, values.size))
    with pytest.raises(ValueError, match='Nonuniform'):
        native_bounds({'data': values})


def test_explicit_nonuniform_bounds_unchanged(native_bounds):
    """Preserve provided intervals rather than infer or regularize them."""
    intervals = np.array([[-1., 1.], [1., 3.], [3., 7.]])
    np.testing.assert_array_equal(native_bounds({'data': np.array([0., 2., 5.]), 'bounds_data': intervals}), intervals)


def test_singleton_keeps_signed_increment(native_bounds):
    """A singleton still requires its actual increment or explicit bounds."""
    np.testing.assert_allclose(native_bounds({'data': np.array([2.], dtype=np.float32)}, -.2), [[2.1, 1.9]])
    with pytest.raises(ValueError, match='Singleton'):
        native_bounds({'data': np.array([2.])})
