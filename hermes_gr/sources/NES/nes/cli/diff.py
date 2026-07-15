# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

"""
NES CLI utility: absolute difference between two NetCDF files.

This module exposes `diff()` to be used by the CLI subcommand
`nes diff -1 file1.nc -2 file2.nc -o output.nc`.

Formula (per variable, per cell):

    diff = file2 - file1
"""
from .cli_logger import get_cli_logger
from nes.load_nes import open_netcdf
from numpy import isinf, isnan
from typing import Optional, List


logger = get_cli_logger("diff")


def diff(file1: str, file2: str, output_file: str,
         var_list: list[str] | None = None) -> None:
    """
    Compute the **absolute difference** between two NetCDF files and write the
    result to a new NetCDF file.

    The computation is performed **variable-wise** and **cell-wise** using the
    following formula:

        diff = file2 - file1

    Notes
    -----
    * `file1` is treated as the baseline.
    * Only variables present in **both** files and selected in `var_list`
      (if provided) are processed.

    Parameters
    ----------
    file1 : str
        Path to the baseline NetCDF file.
    file2 : str
        Path to the comparison NetCDF file.
    output_file : str
        Path to the output NetCDF file that will store the absolute differences.
    var_list : list[str] | None, optional
        Subset of variable names to process. If None, the intersection of
        data variables between `file1` and `file2` will be used.

    Raises
    ------
    KeyError
        If any variable in `var_list` is not present in *both* datasets.
    ValueError
        If a processed variable has incompatible shapes between the two files.
    Exception
        If the output cannot be written.
    """
    # --- Open input files ---
    logger.info(f"Opening baseline NetCDF (file1): {file1}")
    nessy_1 = open_netcdf(file1)

    logger.info(f"Opening comparison NetCDF (file2): {file2}")
    nessy_2 = open_netcdf(file2)

    # Determine variables to process
    shared_vars = set(nessy_1.variables.keys()).intersection(
        set(nessy_2.variables.keys())
    )
    if var_list is None:
        target_vars = sorted(shared_vars)
    else:
        missing = [v for v in var_list if v not in shared_vars]
        if missing:
            raise KeyError(
                "Variables not present in both files: " + ", ".join(missing)
            )
        target_vars = var_list

    # Restrict to the chosen variables and load
    if target_vars:
        nessy_1.keep_vars(target_vars)
        nessy_2.keep_vars(target_vars)
    nessy_1.load()
    logger.info(f"Baseline NetCDF loaded (file1): {file1}")
    nessy_2.load()
    logger.info(f"Baseline NetCDF loaded (file2): {file2}")

    logger.info("Computing absolute differences for %d variables", len(target_vars))

    # --- Compute absolute difference per variable ---
    for var_name in target_vars:
        logger.debug("Computing variable: %s", var_name)

        data1 = nessy_1.variables[var_name]["data"]
        data2 = nessy_2.variables[var_name]["data"]

        # Check shape compatibility
        if getattr(data1, "shape", None) != getattr(data2, "shape", None):
            raise ValueError(
                f"Variable '{var_name}' has incompatible shapes: "
                f"{getattr(data1, 'shape', None)} vs {getattr(data2, 'shape', None)}"
            )

        diff_data = data2 - data1
        nessy_1.variables[var_name]["data"] = diff_data

        # Update attributes
        nessy_1.variables[var_name]["long_name"] = f"{var_name} (absolute difference)"
        nessy_1.variables[var_name]["_min"] = f"{nessy_1.get_min(var_name):.3e}"
        nessy_1.variables[var_name]["_max"] = f"{nessy_1.get_max(var_name):.3e}"
        nessy_1.variables[var_name]["_nes_operation"] = "diff"
        nessy_1.variables[var_name]["_nes_baseline_file"] = file1
        nessy_1.variables[var_name]["_nes_comparison_file"] = file2

    # --- Write output ---
    logger.info("Writing output NetCDF: %s", output_file)
    wrote = False
    for writer_name in ("to_netcdf", "write", "save", "to_nc"):
        writer = getattr(nessy_1, writer_name, None)
        if callable(writer):
            writer(output_file)
            wrote = True
            break
    if not wrote:
        logger.error(
            "Cannot write output: no suitable writer method found on NES object."
        )
        raise Exception(
            "No writer found (tried: to_netcdf, write, save, to_nc). Please implement or choose the correct method."
        )

    logger.info("Done. Absolute differences saved to: %s", output_file)


__all__ = ["diff"]
