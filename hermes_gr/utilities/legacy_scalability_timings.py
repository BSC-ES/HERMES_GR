#!/usr/bin/env python

# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

"""Convert older HERMES_GR timing logs to the workflow timing schema."""

import pandas as pd


STAT_COLUMNS = ["min", "mean", "max", "median"]
WORKFLOW_TIMING_ORDER = [
    ("Initialization", "Grid"),
    ("Initialization", "init_vertical"),
    ("Initialization", "init_speciation"),
    ("Initialization", "init_temporal"),
    ("Initialization", "load_emissions"),
    ("Initialization", "init_horizontal_get_weight_matrix"),
    ("Initialization", "init_horizontal_regrid"),
    ("Initialization", "rescaling"),
    ("Initialization", "TOTAL"),
    ("Calculation", "do_vertical"),
    ("Calculation", "do_speciation"),
    ("Calculation", "do_temporal"),
    ("Calculation", "get_4d_emissions"),
    ("Calculation", "combine_inventories"),
    ("Calculation", "TOTAL"),
    ("Writing", "prepare_output"),
    ("Writing", "write_time_step_emissions"),
    ("Writing", "unit_conversion"),
    ("Writing", "communication"),
    ("Writing", "netcdf_write"),
    ("Writing", "netcdf_write_unaccounted"),
    ("Writing", "close_output"),
    ("Writing", "TOTAL"),
    ("HERMES", "TOTAL"),
]
ROW_MAP = {
    ("Initialization", "Grid"): [
        ("Grid", "select_grid"),
        ("Grid", "to_shapefile"),
    ],
    ("Initialization", "init_vertical"): [
        ("GfasVerticalDistribution", "Init"),
        ("GfasHourlyVerticalDistribution", "Init"),
        ("VerticalDistribution", "Init"),
    ],
    ("Initialization", "init_speciation"): [
        ("Speciation", "Init"),
    ],
    ("Initialization", "init_temporal"): [
        ("TemporalDistribution", "Init"),
    ],
    ("Initialization", "load_emissions"): [
    ],
    ("Initialization", "init_horizontal_get_weight_matrix"): [
        ("ConservativeRegrid", "create_weight_matrix"),
    ],
    ("Initialization", "init_horizontal_regrid"): [
        ("EmissionInventory", "do_regrid"),
        ("GfasEmissionInventory", "do_regrid"),
        ("PointGfasEmissionInventory", "do_regrid"),
        ("PointSourceEmissionInventory", "do_regrid"),
    ],
    ("Initialization", "rescaling"): [],
    ("Calculation", "do_vertical"): [
        ("VerticalDistribution", "calculate_weights"),
    ],
    ("Calculation", "do_speciation"): [
        ("Speciation", "do_speciation"),
    ],
    ("Calculation", "do_temporal"): [
        ("TemporalDistribution", "calculate_3d_temporal_factors"),
    ],
    ("Calculation", "get_4d_emissions"): [
        ("Writer", "calculate_data_by_var"),
    ],
    ("Calculation", "combine_inventories"): [],
    ("Writing", "prepare_output"): [
        ("Writer", "set_variable_attributes"),
        ("Writer", "calculate_displacements"),
        ("Writer", "tuple_to_index"),
    ],
    ("Writing", "write_time_step_emissions"): [
    ],
    ("Writing", "unit_conversion"): [],
    ("Writing", "communication"): [],
    ("Writing", "netcdf_write"): [],
    ("Writing", "netcdf_write_unaccounted"): [],
    ("Writing", "close_output"): [],
}
UNIT_CHANGE_ROWS = [
    ("WriterMonarch", "unit_change"),
    ("WriterMocage", "unit_change"),
]


def _read_legacy_log(path):
    """Read a v2.1.6 timing log and normalize metadata column names."""
    skiprows = 0
    with open(path) as file_handler:
        for line_number, line in enumerate(file_handler):
            if "Class" in line and "Function" in line:
                skiprows = line_number
                break
        else:
            raise ValueError("{} is not a HERMES_GR v2.1.6 timing log.".format(path))

    log = pd.read_csv(path, skiprows=skiprows)
    unnamed_columns = [column for column in log.columns if str(column).startswith("Unnamed:")]
    if unnamed_columns:
        log = log.drop(columns=unnamed_columns)

    if "Class" not in log.columns or "Function" not in log.columns:
        raise ValueError("{} is not a HERMES_GR v2.1.6 timing log.".format(path))

    log = log.rename(columns={"Class": "class", "Function": "function"})
    return log.set_index(["class", "function"], drop=False)


def _numeric_columns(log):
    metadata_columns = {"class", "function"}
    return [
        column for column in log.columns
        if column not in metadata_columns and pd.api.types.is_numeric_dtype(log[column])
    ]


def _rank_columns(log):
    return [
        column for column in _numeric_columns(log)
        if column not in STAT_COLUMNS
    ]


def _zero_series(log):
    return pd.Series(0.0, index=_numeric_columns(log), dtype=float)


def _sum_rows(log, legacy_rows):
    selected_rows = [
        legacy_row for legacy_row in legacy_rows
        if legacy_row in log.index
    ]
    if not selected_rows:
        return _zero_series(log)
    selected = log.loc[selected_rows, _numeric_columns(log)]
    if isinstance(selected, pd.Series):
        return selected.astype(float)
    return selected.astype(float).sum(axis=0)


def _single_row(log, legacy_row):
    if legacy_row not in log.index:
        return _zero_series(log)
    selected = log.loc[legacy_row, _numeric_columns(log)]
    if isinstance(selected, pd.DataFrame):
        selected = selected.iloc[0]
    return selected.astype(float)


def _clip_nonnegative(row):
    """Clip negative timing residuals introduced by non-nested legacy timers."""
    return row.clip(lower=0)


def _recompute_stats(row, ranks):
    if ranks:
        rank_values = row[ranks].astype(float)
        row["min"] = rank_values.min()
        row["mean"] = rank_values.mean()
        row["max"] = rank_values.max()
        row["median"] = rank_values.median()
    else:
        for stat in STAT_COLUMNS:
            if stat not in row.index:
                row[stat] = row["mean"] if "mean" in row.index else 0.0
    return row


def convert_legacy_log(path):
    """
    Convert one v2.1.6 timing log to the v3.1.0 workflow timing schema.

    Returns
    -------
    pandas.DataFrame
        CSV-ready table with ``section`` and ``subsection`` metadata columns.
    """
    log = _read_legacy_log(path)
    ranks = _rank_columns(log)
    converted_rows = {}

    for workflow_row, legacy_rows in ROW_MAP.items():
        converted_rows[workflow_row] = _sum_rows(log, legacy_rows)

    grid_row = ("Initialization", "Grid")
    if (converted_rows[grid_row] == 0).all():
        converted_rows[grid_row] = _sum_rows(log, [("Grid", "Init")])

    weight_matrix_row = ("Initialization", "init_horizontal_get_weight_matrix")
    if (converted_rows[weight_matrix_row] == 0).all():
        converted_rows[weight_matrix_row] = _sum_rows(log, [("ConservativeRegrid", "Init")])

    temporal_row = ("Calculation", "do_temporal")
    if (converted_rows[temporal_row] == 0).all():
        converted_rows[temporal_row] = _sum_rows(log, [("TemporalDistribution", "calculate_2d_temporal_factors")])

    get_4d_row = ("Calculation", "get_4d_emissions")
    unit_change = _sum_rows(log, UNIT_CHANGE_ROWS)
    converted_rows[get_4d_row] = _clip_nonnegative(converted_rows[get_4d_row] - unit_change)

    load_emissions_row = ("Initialization", "load_emissions")
    converted_rows[load_emissions_row] = _clip_nonnegative(
        _sum_rows(log, [("EmissionInventory", "make_emission_list")])
        - converted_rows[("Initialization", "init_vertical")]
        - converted_rows[("Initialization", "init_speciation")]
        - converted_rows[("Initialization", "init_temporal")]
        - converted_rows[weight_matrix_row]
    )

    prepare_output_row = ("Writing", "prepare_output")
    write_time_step_row = ("Writing", "write_time_step_emissions")
    converted_rows[write_time_step_row] = _clip_nonnegative(
        _sum_rows(log, [("Writer", "write")])
        - converted_rows[("Calculation", "get_4d_emissions")]
        - converted_rows[prepare_output_row]
    )

    hermes_total = _single_row(log, ("HERMES", "TOTAL"))

    init_partial_rows = [
        ("Initialization", "Grid"),
        ("Initialization", "init_vertical"),
        ("Initialization", "init_speciation"),
        ("Initialization", "init_temporal"),
        ("Initialization", "load_emissions"),
        ("Initialization", "init_horizontal_get_weight_matrix"),
        ("Initialization", "init_horizontal_regrid"),
    ]
    converted_rows[("Initialization", "TOTAL")] = sum(
        (converted_rows[row] for row in init_partial_rows),
        _zero_series(log),
    )
    converted_rows[("Writing", "TOTAL")] = (
        converted_rows[prepare_output_row]
        + converted_rows[write_time_step_row]
        + converted_rows[("Writing", "close_output")]
    )
    converted_rows[("HERMES", "TOTAL")] = hermes_total.copy()

    calculation_partial_rows = [
        ("Calculation", "do_vertical"),
        ("Calculation", "do_speciation"),
        ("Calculation", "do_temporal"),
        ("Calculation", "get_4d_emissions"),
        ("Calculation", "combine_inventories"),
    ]
    calculation_partial_total = sum(
        (converted_rows[row] for row in calculation_partial_rows),
        _zero_series(log),
    )
    converted_rows[("Calculation", "TOTAL")] = calculation_partial_total

    output_rows = []
    for section, subsection in WORKFLOW_TIMING_ORDER:
        row = converted_rows[(section, subsection)].copy()
        row = _recompute_stats(row, ranks)
        row["section"] = section
        row["subsection"] = subsection
        output_rows.append(row)

    columns = ["section", "subsection"] + ranks + STAT_COLUMNS
    converted = pd.DataFrame(output_rows)
    return converted.reindex(columns=columns)


def is_legacy_log(path):
    """Return whether a timing log looks like the HERMES_GR v2.1.6 schema."""
    try:
        with open(path) as file_handler:
            for line in file_handler:
                if "Class" in line and "Function" in line:
                    return True
                if "section" in line and "subsection" in line:
                    return False
    except OSError:
        return False
    return False


def is_early_v3_log(log):
    """Identify v3.0.0 logs by their phase timers, irrespective of filename."""
    if not {"section", "subsection"}.issubset(log.columns):
        return False
    rows = set(zip(log["section"], log["subsection"]))
    return {("Init", "TOTAL"), ("HERMES", "Main"), ("HERMES", "Write")}.issubset(rows)


def convert_early_v3_log(log):
    """Map v3.0.0 phase totals and directly comparable timers to workflow rows."""
    source = log.set_index(["section", "subsection"])
    numeric = [column for column in source.columns if pd.api.types.is_numeric_dtype(source[column])]
    ranks = [column for column in numeric if column not in STAT_COLUMNS]
    zero = pd.Series(0.0, index=numeric)

    def timer(section, subsection):
        key = (section, subsection)
        if key not in source.index:
            return zero.copy()
        return source.loc[key, numeric].astype(float)

    mapped = {
        ("Initialization", "Grid"): timer("Grid", "select_grid"),
        ("Initialization", "init_vertical"): timer("EmissionInventory", "init_vertical"),
        ("Initialization", "init_speciation"): timer("EmissionInventory", "init_speciation"),
        ("Initialization", "init_temporal"): timer("EmissionInventory", "init_temporal"),
        ("Initialization", "init_horizontal_regrid"): timer("PointGfasEmissionInventory", "do_regrid"),
        ("Calculation", "do_vertical"): timer("VerticalDistribution", "calculate_weights"),
        ("Calculation", "do_speciation"): timer("Speciation", "do_speciation"),
        ("Calculation", "do_temporal"): timer("TemporalDistribution", "calculate_2d_temporal_factors"),
        ("Writing", "prepare_output"): timer("Write", "Creation"),
        ("Initialization", "TOTAL"): timer("Init", "TOTAL"),
        ("Writing", "TOTAL"): timer("HERMES", "Write"),
        ("HERMES", "TOTAL"): timer("HERMES", "TOTAL"),
    }
    mapped[("Calculation", "TOTAL")] = _clip_nonnegative(
        timer("HERMES", "Main") - mapped[("Writing", "TOTAL")]
    )
    mapped[("Writing", "write_time_step_emissions")] = _clip_nonnegative(
        mapped[("Writing", "TOTAL")] - mapped[("Writing", "prepare_output")]
    )

    output_rows = []
    for section, subsection in WORKFLOW_TIMING_ORDER:
        if (section, subsection) not in mapped:
            continue
        row = _recompute_stats(mapped[(section, subsection)].copy(), ranks)
        row["section"] = section
        row["subsection"] = subsection
        output_rows.append(row)
    return pd.DataFrame(output_rows).reindex(columns=["section", "subsection"] + ranks + STAT_COLUMNS)
