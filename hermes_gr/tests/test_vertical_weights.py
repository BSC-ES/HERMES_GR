# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

"""Run real vertical allocation methods without loading MPI or emission data."""

import ast
from pathlib import Path
import re
from unittest.mock import Mock

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")


@pytest.fixture
def vertical_class():
    path = Path(__file__).resolve().parents[1] / "modules/vertical/vertical.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    logger = Mock()
    namespace = dict(np=np, pd=pd, re=re, precision=np.float64, log_message=logger)
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["VerticalDistribution"], logger


@pytest.mark.parametrize("bottom,top,weight,expected", [
    (0, 300, 1, [(0, 1 / 3), (1, 1 / 3)]),
    (200, 300, 1, []),
    (250, 400, 1, []),
    (150, 250, .4, [(1, .2)]),
    (100, 200, 1, [(1, 1)]),
    (0, 200, 1, [(0, .5), (1, .5)]),
    (0, 100, 1, [(0, 1)]),
])
def test_interval_weights_only_reference_valid_levels(vertical_class, bottom, top, weight, expected):
    cls, _ = vertical_class
    actual = cls.get_weights(bottom, top, weight, [100, 200])
    assert [entry["index"] for entry in actual] == [index for index, value in expected]
    np.testing.assert_allclose([entry["weight"] for entry in actual], [value for index, value in expected])
    assert all(0 <= entry["index"] < 2 for entry in actual)


@pytest.mark.parametrize("profile,expected,warn", [
    ([(300, 1)], [1 / 3, 1 / 3], True),
    ([(200, 0), (300, 1)], [0, 0], True),
    ([(250, 0), (400, 1)], [0, 0], True),
    ([(100, .2), (300, .8)], [.2, .4], True),
    ([(100, .2), (200, .3), (300, .2), (400, .3)], [.2, .3], True),
    ([(100, .4), (200, .6)], [.4, .6], False),
    ([(200, 1)], [.5, .5], False),
    ([(100, 1)], [1, 0], False),
    ([(200, 1), (400, 0)], [.5, .5], False),
    ([(100, 0), (300, 0)], [0, 0], False),
    ([(100, 40), (200, 60)], [.4, .6], False),
    ([(100, 20), (300, 80)], [.2, .4], True),
])
def test_profile_discards_overflow_without_redistribution(vertical_class, profile, expected, warn):
    cls, logger = vertical_class
    instance = cls.__new__(cls)
    instance.id, instance.output_heights, instance.vertical_profile = "V_TEST", [100, 200], profile
    actual = instance.calculate_weights()
    np.testing.assert_allclose(actual, expected)
    assert np.isfinite(actual).all()
    warnings = [call for call in logger.call_args_list if call.kwargs.get("level") == 7]
    assert len(warnings) == int(warn)
    if warn:
        message = warnings[0].args[0]
        assert "V_TEST" in message
        assert "200" in message
        assert "discard" in message.lower()
        assert "renormal" in message.lower()


def test_csv_loading_and_partial_discard(vertical_class, tmp_path):
    cls, logger = vertical_class
    path = tmp_path / "vertical.csv"
    path.write_text('ID;layers;weights\nV001;"100,300";"0.2,0.8"\n')
    instance = cls("V001", str(path), [100, 200])
    np.testing.assert_allclose(instance.calculate_weights(), [.2, .4])
    assert sum(call.kwargs.get("level") == 7 for call in logger.call_args_list) == 1
