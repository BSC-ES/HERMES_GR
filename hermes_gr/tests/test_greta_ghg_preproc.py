# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

"""Unit tests for GRETA greenhouse-gas preprocessing."""

import importlib

import pandas as pd
import pytest


greta = importlib.import_module("preproc.greta_preproc")


def _cams_table():
    """Build a compact CAMS table with known DEU F-subsector totals."""
    return pd.DataFrame(
        [
            ("DEU", 2022, "F1", 1.0, 1.0, 9.0),
            ("DEU", 2022, "F2", 2.0, 2.0, 0.0),
            ("DEU", 2022, "F3", 3.0, 0.0, 3.0),
            ("DEU", 2022, "F4", 4.0, 5.0, 0.0),
            ("FRA", 2022, "F1", 100.0, 100.0, 0.0),
            ("DEU", 2021, "F1", 100.0, 100.0, 0.0),
        ],
        columns=["ISO3", "Year", "GNFR_Category", "CH4", "CO2_ff", "CO2_bf"],
    )


def test_ghg_fractions_use_total_co2_and_ch4_by_sector():
    """CO2 shares combine fossil/biofuel columns; CH4 uses its own column."""
    table = _cams_table()

    co2 = greta.get_gnfr_f_fractions("unused.csv", "CO2", table=table)
    ch4 = greta.get_gnfr_f_fractions("unused.csv", "CH4", table=table)

    assert co2 == pytest.approx({"F1": 0.5, "F2": 0.1, "F3": 0.15, "F4": 0.25})
    assert ch4 == pytest.approx({"F1": 0.1, "F2": 0.2, "F3": 0.3, "F4": 0.4})


def test_greta_ghg_preproc_only_writes_co2_and_ch4_and_splits_f(tmp_path, monkeypatch):
    """The GHG driver skips N2O and creates the expected F1--F4 outputs."""
    cams_path = tmp_path / "cams.csv"
    _cams_table().to_csv(cams_path, sep=";", index=False)
    input_path = tmp_path / "2022_Sub2026_GHG_GNFR_Greta1.7.1.1.nc"
    input_path.touch()
    source_names = (
        "E_GNFR_A_CO2",
        "E_GNFR_F_CO2",
        "E_GNFR_A_CH4",
        "E_GNFR_F_CH4",
        "E_GNFR_A_N2O",
        "E_SUM_CO2",
    )

    class FakeInventory:
        variables = {name: {} for name in source_names}

        def close(self):
            pass

    fake_inventory = FakeInventory()
    monkeypatch.setattr(greta.nes, "open_netcdf", lambda *args, **kwargs: fake_inventory)
    writes = []
    monkeypatch.setattr(
        greta,
        "_write_variable",
        lambda *args, **kwargs: writes.append((args, kwargs)),
    )

    result = greta.greta_ghg_preproc(
        original_file=input_path,
        output_path=tmp_path / "out",
        year=2022,
        cams_reg_path=cams_path,
    )

    assert result is True
    assert len(writes) == 12
    written = {
        (args[3], args[4], args[5]): (args[6], kwargs)
        for args, kwargs in writes
    }
    assert not any("N2O" in source for source, _, _ in written)
    assert not any(source.startswith("E_SUM_") for source, _, _ in written)

    for pollutant_code, output_name, fractions in (
        ("CO2", "co2", {"F1": 0.5, "F2": 0.1, "F3": 0.15, "F4": 0.25}),
        ("CH4", "ch4", {"F1": 0.1, "F2": 0.2, "F3": 0.3, "F4": 0.4}),
    ):
        assert (f"E_GNFR_A_{pollutant_code}", output_name, "A") in written
        assert (f"E_GNFR_F_{pollutant_code}", output_name, "F") in written
        for sector, fraction in fractions.items():
            entry = written[(f"E_GNFR_F_{pollutant_code}", output_name, sector)]
            assert entry[0] == pytest.approx(fraction)
            assert entry[1]["inventory_source"] == "GRETA v1.7.1.1"
            assert entry[1]["references"] is None


def test_greta_ghg_preproc_rejects_other_years(tmp_path):
    """The only configured GRETA GHG input represents 2022."""
    with pytest.raises(ValueError, match="available for 2022 only"):
        greta.greta_ghg_preproc(year=2030)
