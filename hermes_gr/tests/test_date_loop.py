# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

"""BUG-004 regressions: require normal termination after the last valid period.

Execute real constructor/main/run and date helpers via AST to avoid native MPI
imports. Only costly grid, inventory, logger and writer services are doubles.
Require inclusive period-start boundaries and a fresh instance per valid period.
MPI abort is intercepted only to save diagnostics, then reported as a failure.
"""

import ast
from calendar import isleap, monthrange
from datetime import datetime, timedelta
import json
from pathlib import Path
import traceback
from types import SimpleNamespace
from typing import Optional
from unittest.mock import Mock

import pytest
from dateutil.relativedelta import relativedelta

ROOT = Path(__file__).resolve().parents[2]


def load_definition(relative_path, name, namespace, methods=None):
    """Compile actual definitions, preserving the tested statements unchanged."""
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(), filename=str(path))
    definition = next(node for node in tree.body if getattr(node, "name", None) == name)
    if methods is not None:
        definition.body = [node for node in definition.body if getattr(node, "name", None) in methods]
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def date(value):
    return datetime.fromisoformat(value)


class RecordedAbort(RuntimeError):
    """Stand in for the real MPI abort so pytest can record the failure."""


@pytest.fixture
def runtime():
    import os

    comm = Mock()
    comm.Get_rank.return_value, comm.Get_size.return_value = 0, 1
    comm.Abort.side_effect = RecordedAbort("MPI Abort(1) requested")
    logger = Mock()
    namespace = dict(
        os=os, traceback=traceback, datetime=datetime, timedelta=timedelta, Optional=Optional,
        relativedelta=relativedelta, monthrange=monthrange, isleap=isleap, ArgParser=object,
        ConfigGr=object, MPI=SimpleNamespace(COMM_WORLD=comm, Comm=object),
        hermes_gr=SimpleNamespace(DEBUG=False, __version__="test"),
        get_time_stamp=lambda: 0., configure_logger=Mock(), log_message=logger,
        record_time=Mock(), finalize_time_log=Mock(), select_grid=Mock(return_value=object()),
    )
    config_class = load_definition("hermes_gr/config/config.py", "ConfigGr", namespace, {"_parse_end_date"})
    load_definition("hermes_gr/config/config.py", "parse_start_date", namespace)
    load_definition("hermes_gr/modules/temporal/temporal.py", "TemporalDistribution", namespace,
                    {"calculate_date_array"})
    model = load_definition("hermes_gr/hermes.py", "HermesGr", namespace)
    run = load_definition("hermes_gr/hermes.py", "run", namespace)
    return SimpleNamespace(namespace=namespace, model=model, run=run, comm=comm, logger=logger,
                           config_class=config_class)


SCENARIOS = [
    ("T1", "monthly", "2020-01-01", "2020-01-15", ["2020-01-01"], "2020-02-01"),
    ("T2", "monthly", "2020-01-01", "2020-01-31", ["2020-01-01"], "2020-02-01"),
    ("T3", "monthly", "2020-01-01", "2020-02-15", ["2020-01-01", "2020-02-01"], "2020-03-01"),
    ("T4", "monthly", "2020-01-01", "2020-03-01", ["2020-01-01", "2020-02-01", "2020-03-01"], None),
    ("T5", "monthly", "2020-12-01", "2021-01-15", ["2020-12-01", "2021-01-01"], "2021-02-01"),
    ("T6", "hourly", "2020-01-01", "2020-01-02", ["2020-01-01", "2020-01-02"], None),
    ("T7", "daily", "2020-01-01", "2020-01-02T12:00", ["2020-01-01", "2020-01-02"], "2020-01-03"),
    ("T8", "yearly", "2020-01-01", "2021-01-15", ["2020-01-01", "2021-01-01"], "2022-01-01"),
    ("T9", "yearly", "2020-01-01", "2021-01-01", ["2020-01-01", "2021-01-01"], None),
    ("T10", "monthly", "2020-01-31", "2020-03-29", ["2020-01-31", "2020-02-29", "2020-03-29"], None),
    ("T11", "hourly", "2020-01-01", "2020-01-01T12:00", ["2020-01-01"], "2020-01-02"),
    ("T12", "daily", "2020-01-01", "2020-01-02", ["2020-01-01", "2020-01-02"], None),
    ("T13", "monthly", "2020-01-01", None, ["2020-01-01"], None),
    ("T14", "daily", "2020-02-28", "2020-03-01", ["2020-02-28", "2020-02-29", "2020-03-01"], None),
    ("T15", "daily", "2019-02-28", "2019-03-01", ["2019-02-28", "2019-03-01"], None),
    ("T16", "hourly", "2020-12-31", "2021-01-01", ["2020-12-31", "2021-01-01"], None),
    ("T17", "monthly", "2019-01-31", "2019-03-28", ["2019-01-31", "2019-02-28", "2019-03-28"], None),
    ("T18", "monthly", "2020-01-01", "2020-01-31T23:59:59", ["2020-01-01"], "2020-02-01"),
    ("T19", "monthly", "2020-01-01", "2020-02-01T00:00:01", ["2020-01-01", "2020-02-01"], "2020-03-01"),
    ("T20", "hourly", "2020-01-01", "2020-01-01T23:59:59", ["2020-01-01"], "2020-01-02"),
    ("T21", "daily", "2020-01-01", "2020-01-02T00:00:01", ["2020-01-01", "2020-01-02"], "2020-01-03"),
    ("T22", "yearly", "2019-01-01", "2019-12-31", ["2019-01-01"], "2020-01-01"),
    ("T23", "yearly", "2020-02-29", "2021-02-28", ["2020-02-29", "2021-02-28"], None),
    ("T24", "hourly", "2020-01-01T06:00", "2020-01-02T06:00",
     ["2020-01-01T06:00", "2020-01-02T06:00"], None),
    ("T25", "daily", "2020-01-01", "2020-01-01", ["2020-01-01"], None),
    ("T26", "hourly", "2020-01-01", "2020-01-01", ["2020-01-01"], None),
    ("T27", "yearly", "2020-01-01", "2020-01-01", ["2020-01-01"], None),
    ("T28", "hourly", "2020-01-01", None, ["2020-01-01"], None),
    ("T29", "daily", "2020-01-01", None, ["2020-01-01"], None),
    ("T30", "yearly", "2020-01-01", None, ["2020-01-01"], None),
]


@pytest.mark.parametrize("first_time", [False, True], ids=["calculation", "first_time"])
@pytest.mark.parametrize("label,frequency,start,end,valid,overshoot", SCENARIOS, ids=[row[0] for row in SCENARIOS])
def test_date_loop(
        runtime, tmp_path, label, frequency, start, end, valid, overshoot, first_time):
    """Require reinitialization for valid periods and no out-of-range attempt."""
    options = SimpleNamespace(
        start_date=runtime.namespace["parse_start_date"](date(start).strftime("%Y/%m/%d_%H:%M:%S")),
        end_date=runtime.config_class._parse_end_date(
            None if end is None else date(end).strftime("%Y/%m/%d_%H:%M:%S"), date(start)),
        output_dir=str(tmp_path), log_level=3, first_time=first_time, output_model="MONARCH",
        domain_type="rotated", vertical_description=[100.], output_timestep_type=frequency,
        output_timestep_num=3 if frequency == "hourly" else 1, output_timestep_freq=2,
    )
    config = SimpleNamespace(options=options, get_output_name=lambda value: "synthetic.nc")
    attempted, completed, processed, instances = [], [], [], []
    writer = Mock()
    writer.append_time_step_data.return_value = dict(unit_conversion=0, communication=0, netcdf_write=0)
    original = runtime.model

    class ObservedModel(original):
        def __init__(self, config, new_date=None, comm=None):
            attempted.append(new_date if new_date is not None else config.options.start_date)
            instances.append(self)
            super().__init__(config, new_date=new_date, comm=comm)
            if hasattr(self, "grid"):
                completed.append(self.options.start_date)

        def init_rescaling(self):
            return None

        def make_emission_list(self):
            return []

        def prepare_inventories(self):
            processed.append(self.options.start_date)

        def prepare_output(self):
            return writer, "synthetic.nc"

        def get_time_step_emissions(self, *args):
            return {}

    runtime.namespace["HermesGr"] = ObservedModel
    runtime.namespace["ConfigGr"] = lambda: config
    aborted = False
    try:
        runtime.run()
    except RecordedAbort:
        # Preserve real run() exception diagnostics without accepting the abort.
        aborted = True
    captured = "\n".join(str(call.args[0]) for call in runtime.logger.call_args_list)
    observation = dict(
        test=label, frequency=frequency, first_time=first_time, start=start, end=str(options.end_date),
        attempted=list(map(str, attempted)), completed=list(map(str, completed)),
        processed=list(map(str, processed)), initialization_attempts=len(attempted),
        completed_initializations=len(completed), valid_periods_processed=len(processed),
        termination="MPI Abort(1) requested" if aborted else "main returned None",
        traceback=captured[captured.index("Traceback"):] if "Traceback" in captured else None,
    )
    (tmp_path / "observation.json").write_text(json.dumps(observation, indent=2))
    print(json.dumps({key: value for key, value in observation.items() if key != "traceback"}))
    if aborted:
        pytest.fail("Date loop must terminate normally after its last valid period. "
                    "Actual execution requested MPI Abort(1):\n" + captured)
    expected_valid = list(map(date, valid))
    assert attempted == expected_valid, "No initialization may be attempted beyond end_date"
    assert completed == expected_valid
    assert processed == ([] if first_time else expected_valid)
    assert len({id(instance) for instance in instances}) == len(expected_valid)
    assert runtime.namespace["select_grid"].call_count == len(expected_valid)
    runtime.comm.Abort.assert_not_called()
    assert all(hasattr(instance, "first_time") and hasattr(instance, "grid") for instance in instances)
    if frequency == "hourly":
        assert instances[0].time_step_list == [date(start) + timedelta(hours=i) for i in (0, 2, 4)]
