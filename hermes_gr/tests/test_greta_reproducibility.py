# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

"""Compare one full-grid GRETA VOC output with one and two MPI ranks.

Run directly with --input-dir and --output-dir for the real inventory check.
Pytest runs it only when HERMES_GRETA_TEST_INPUT_DIR is explicitly provided.
The source inventories and definitive outputs are never modified.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import numpy as np
import pytest
from netCDF4 import Dataset


# Use the production writer, including NES decomposition and parallel NetCDF I/O.
_WORKER = """
import json
import sys
from mpi4py import MPI
import nes
from preproc.greta_preproc import _write_variable
case = json.loads(sys.argv[1])
comm = MPI.COMM_WORLD
assert comm.size == int(sys.argv[3])
inventory = nes.open_netcdf(
    case['source'], crs='EPSG:3035',
    coordinate_names={'x': 'LON', 'y': 'LAT'},
    coordinate_units={'x': 'm', 'y': 'm'},
    time_exception=True, parallel_method='Y',
)
try:
    inventory.load('E_GNFR_F_NMVOC')
    print('rank', comm.rank, 'source shape',
          inventory.variables['E_GNFR_F_NMVOC']['data'].shape, flush=True)
    _write_variable(case['source'], sys.argv[2], 2030,
                    'E_GNFR_F_NMVOC', case['species'], case['sector'],
                    case['fraction'], inventory=inventory)
finally:
    inventory.close()
comm.Barrier()
print('completed rank', comm.rank, flush=True)
"""


def _select_case(input_dir):
    """Select the largest nonzero combined F-subsector/VOC fraction.

    Parameters
    ----------
    input_dir : pathlib.Path
        GRETA root containing original_files and the reference CSV tables.

    Returns
    -------
    dict
        Source path and the production sector/species factors for 2030.
    """
    from preproc.greta_preproc import (
        _read_csv, _split_factor, _voc_output_name,
        get_gnfr_f_fractions, ORIGINAL_FILES,
    )

    cams = _read_csv(input_dir / 'CAMS-REG-v8_1_emissions_year2022.csv', sep=';')
    shares = get_gnfr_f_fractions('', 'NMVOC', table=cams)
    table = _read_csv(input_dir / 'NMVOC_split_for_CAMS-REG-v81_KO_melt_20251020_v2.csv', sep=',')
    table = table[(table.ISO3 == 'DEU') & (table.year == 2022)]
    candidates = []
    for sector, share in shares.items():
        for code in table.loc[table.gnfr == sector, 'vcode'].unique():
            factor = _split_factor('', 'vcode', code, sector, table=table)
            if 0 < factor < 1 and 0 < share < 1:
                candidates.append((share * factor, sector, code, share, factor))
    fraction, sector, code, share, factor = max(candidates)
    return dict(source=str(input_dir / 'original_files' / ORIGINAL_FILES[2030]),
                species=_voc_output_name(code), sector=sector,
                fraction=fraction, sector_fraction=share, species_fraction=factor)


def _run_worker(command, log_path, timeout):
    """Run a bounded MPI check and terminate its process group on timeout.

    Parameters
    ----------
    command : list of str
        Worker invocation using the selected Python environment.
    log_path : pathlib.Path
        Destination for combined worker output and diagnostics.
    timeout : int
        Maximum runtime in seconds.

    Raises
    ------
    RuntimeError
        If the worker times out or exits unsuccessfully.
    """
    with log_path.open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            status = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise RuntimeError(f'Worker timed out; see {log_path}') from error
    if status:
        raise RuntimeError(f'Worker exited with status {status}; see {log_path}')


def run_check(input_dir, output_dir, timeout=180):
    """Write one serial/MPI pair and compare all NetCDF variable values.

    Parameters
    ----------
    input_dir : pathlib.Path
        GRETA input root. Original data are read-only.
    output_dir : pathlib.Path
        New directory reserved for test outputs and logs.
    timeout : int, optional
        Per-worker timeout in seconds, by default 180.

    Returns
    -------
    dict
        Selected case, comparison counts and maximum absolute differences.

    Raises
    ------
    AssertionError
        If dimensions, masks, scientific metadata or variable values differ.
    FileExistsError
        If the requested test directory already exists.
    RuntimeError
        If a worker fails or times out.
    """
    output_dir.mkdir(parents=True, exist_ok=False)
    case = _select_case(input_dir)
    (output_dir / 'case.json').write_text(json.dumps(case, indent=2) + '\n')
    print('Selected GRETA reproducibility case:', case, flush=True)
    serial = output_dir / 'serial.nc'
    parallel = output_dir / 'parallel.nc'
    # A fresh singleton run avoids comparing against an older cached output.
    _run_worker([sys.executable, '-u', '-c', _WORKER, json.dumps(case), str(serial), '1'],
                output_dir / 'serial.log', timeout)
    print('Serial output completed; starting two MPI ranks.', flush=True)
    _run_worker(['mpiexec', '-n', '2', sys.executable, '-u', '-c', _WORKER,
                 json.dumps(case), str(parallel), '2'], output_dir / 'parallel.log', timeout)
    differences = {}
    with Dataset(serial) as left, Dataset(parallel) as right:
        assert {k: len(v) for k, v in left.dimensions.items()} == {
            k: len(v) for k, v in right.dimensions.items()}
        assert set(left.variables) == set(right.variables)
        for name, variable in left.variables.items():
            other = right[name]
            assert variable.dimensions == other.dimensions, name
            assert variable.dtype == other.dtype, name
            for attribute in ('units', 'calendar', 'grid_mapping', 'standard_name'):
                assert getattr(variable, attribute, None) == getattr(other, attribute, None), (name, attribute)
            a, b = np.ma.asarray(variable[:]), np.ma.asarray(other[:])
            np.testing.assert_array_equal(np.ma.getmaskarray(a), np.ma.getmaskarray(b), err_msg=name)
            # Require exact values, including partition borders, not just totals.
            np.testing.assert_array_equal(a.compressed(), b.compressed(), err_msg=name)
            differences[name] = 0.0
    result = dict(case=case, ranks=[1, 2], variables=len(differences),
                  exact_values=True, maximum_absolute_differences=differences)
    (output_dir / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print('PASS: all NetCDF variable arrays are exactly equal.', flush=True)
    return result


def test_greta_serial_parallel(tmp_path):
    """Check real-data reproducibility only when explicitly requested."""
    input_dir = os.environ.get('HERMES_GRETA_TEST_INPUT_DIR')
    if not input_dir:
        pytest.skip('Set HERMES_GRETA_TEST_INPUT_DIR for the real GRETA MPI check')
    run_check(Path(input_dir), tmp_path / 'greta-reproducibility')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--timeout', type=int, default=180)
    arguments = parser.parse_args()
    run_check(arguments.input_dir, arguments.output_dir, arguments.timeout)
