# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

"""Public imports and serialization must use the canonical implementations."""

from importlib import import_module
import pickle

import pytest

import nes


@pytest.mark.parametrize("name", [
    "Nes", "LatLonNes", "LCCNes", "MercatorNes", "RotatedNes", "RotatedNestedNes",
    "PointsNes", "PointsNesGHOST", "PointsNesProvidentia", "LAEANes",
])
def test_public_classes_and_pickle_lookups(name):
    """Keep public class identities and resolve their canonical serialized paths."""
    cls = getattr(nes, name)
    assert cls is getattr(import_module("nes.projections"), name)
    assert cls is getattr(import_module(cls.__module__), name)
    assert pickle.loads(pickle.dumps(cls)) is cls


def test_base_class_has_a_general_module():
    """Expose the base class outside the projection implementations."""
    current = import_module("nes.default_nes")
    assert nes.Nes is current.Nes
    assert nes.Nes.__module__ == "nes.default_nes"
