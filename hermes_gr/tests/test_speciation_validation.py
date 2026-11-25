# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

"""Exercise real speciation CSV loading without the optional MPI/NES runtime."""

import ast
from pathlib import Path
from unittest.mock import Mock

import pytest

pd = pytest.importorskip("pandas")
SOURCE = Path(__file__).resolve().parents[1] / "modules/speciation/speciation.py"


@pytest.fixture
def speciation_class():
    tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Speciation")
    namespace = {"pd": pd, "log_message": Mock()}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["Speciation"]


def write_profile(tmp_path, rows):
    path = tmp_path / "speciation.csv"
    path.write_text(
        "ID;NO2;PM10\nunits;mol.s-1;kg.s-1\nshort_description;nitrogen_dioxide;particles\n" + rows
    )
    return path


@pytest.mark.parametrize("missing", ["", "NaN", "nan", " NAN ", "NA", "null", "   "])
def test_missing_formula_rejected_before_molecular_weights(tmp_path, speciation_class, missing):
    path = write_profile(tmp_path, f"E001;{missing};pm10\n")
    speciation_class.extract_molecular_weights = Mock()
    with pytest.raises(ValueError) as raised:
        speciation_class("E001", str(path), "/unused/MolecularWeights.csv")
    message = str(raised.value)
    assert str(path) in message
    assert "Row 4, profile E001, species NO2" in message
    assert "NaN" in message
    assert "0" in message
    speciation_class.extract_molecular_weights.assert_not_called()


def test_all_invalid_cells_reported_including_unselected_profiles(tmp_path, speciation_class):
    path = write_profile(tmp_path, "E001;;NaN\nE002;no2;\n")
    inventory = speciation_class.__new__(speciation_class)
    inventory.id = "E001"
    with pytest.raises(ValueError) as raised:
        inventory.get_speciation_profile(str(path))
    message = str(raised.value)
    for location in ["Row 4, profile E001, species NO2", "Row 4, profile E001, species PM10",
                     "Row 5, profile E002, species PM10"]:
        assert location in message


def test_valid_formulas_and_explicit_zero_unchanged(tmp_path, speciation_class):
    path = write_profile(tmp_path, "E001;0.9*nox_no2;0\nE002;no2;pm10\n")
    inventory = speciation_class.__new__(speciation_class)
    inventory.id = "E001"
    assert inventory.get_speciation_profile(str(path)) == [
        {"name": "NO2", "formula": "0.9*nox_no2", "units": "mol.s-1", "long_name": "nitrogen_dioxide"},
        {"name": "PM10", "formula": "0", "units": "kg.s-1", "long_name": "particles"},
    ]


def test_metadata_rows_are_not_speciation_expressions(tmp_path, speciation_class):
    path = tmp_path / "metadata.csv"
    path.write_text("ID;NO2\nunits;\nshort_description;\nE001;no2\n")
    inventory = speciation_class.__new__(speciation_class)
    inventory.id = "E001"
    assert inventory.get_speciation_profile(str(path))[0]["formula"] == "no2"


def test_missing_profile_keeps_existing_error(tmp_path, speciation_class):
    path = write_profile(tmp_path, "E001;no2;pm10\n")
    inventory = speciation_class.__new__(speciation_class)
    inventory.id = "absent"
    with pytest.raises(AttributeError, match="Speciation profile ID absent"):
        inventory.get_speciation_profile(str(path))


def test_blank_lines_preserve_reported_line_numbers(tmp_path, speciation_class):
    path = write_profile(tmp_path, "\nE001;no2;\n")
    inventory = speciation_class.__new__(speciation_class)
    inventory.id = "E001"
    with pytest.raises(ValueError, match="Row 5, profile E001, species PM10"):
        inventory.get_speciation_profile(str(path))
