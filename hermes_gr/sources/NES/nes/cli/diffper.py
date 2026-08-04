# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

"""
NES CLI utility: percentage difference between two NetCDF files.

This module exposes `diffper()` to be used by the CLI subcommand
`nes diffper -1 file1.nc -2 file2.nc -o output.nc`.

Formula (per variable, per cell):

    pct = 100 * (file2 - file1) / (file1 + eps)

Where `eps` is a small constant to avoid division-by-zero.
"""
from .cli_logger import get_cli_logger
from nes.load_nes import open_netcdf
from numpy import isinf, isnan
from typing import Optional, List


logger = get_cli_logger("diffper")


def diffper(file1: str, file2: str, output_file: str, var_list: list[str] | None = None, eps: float = 0.0) -> None:
    """
    Compute the **percentage difference** between two NetCDF files and write the result to a new NetCDF file.

    The computation is performed **variable-wise** and **cell-wise** using the following formula:

        pct = 100 * (file2 - file1) / (file1 + eps)

    Notes
    -----
    * `file1` is treated as the baseline (denominator). `file2` is the comparison dataset (numerator difference).
    * A small `eps` can be provided to prevent division by zero.
    * Only variables present in **both** files and selected in `var_list` if provided) are processed.

    Parameters
    ----------
    file1 : str
        Path to the baseline NetCDF file.
    file2 : str
        Path to the comparison NetCDF file.
    output_file : str
        Path to the output NetCDF file that will store the percentage differences.
    var_list : list[str] | None, optional
        Subset of variable names to process. If None, the intersection of
        data variables between `file1` and `file2` will be used.
    eps : float, optional
        Small constant added to the denominator to avoid division by zero.
        Default is 0.0.

    Raises
    ------
    KeyError
        If any variable in `var_list` is not present in *both* datasets.
    ValueError
        If a processed variable has incompatible shapes between the two files.
    Exception
        If the output cannot be written.
    """
    # --- Open input files (metadata first, then load selected variables) ---
    logger.info(f"Opening baseline NetCDF (file1): {file1}")
    nessy_1 = open_netcdf(file1)

    logger.info(f"Opening comparison NetCDF (file2): {file2}")
    nessy_2 = open_netcdf(file2)

    # Determine variables to process: selection or intersection of shared vars
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

    # Restrict to the chosen variables (memory friendly) and load data
    if target_vars:
        nessy_1.keep_vars(target_vars)
        nessy_2.keep_vars(target_vars)
    nessy_1.load()
    logger.info(f"Baseline NetCDF loaded (file1): {file1}")
    nessy_2.load()
    logger.info(f"Baseline NetCDF loaded (file2): {file2}")

    logger.info(
        "Computing percentage differences for %d variables", len(target_vars)
    )

    # --- Compute percentage difference per variable ---
    for var_name in target_vars:
        logger.debug("Computing variable: %s", var_name)

        data1 = nessy_1.variables[var_name]["data"]
        data2 = nessy_2.variables[var_name]["data"]

        # Basic shape validation to avoid broadcasting surprises
        if getattr(data1, "shape", None) != getattr(data2, "shape", None):
            raise ValueError(
                f"Variable '{var_name}' has incompatible shapes: "
                f"{getattr(data1, 'shape', None)} vs {getattr(data2, 'shape', None)}"
            )

        # pct = 100 * (file2 - file1) / (file1 + eps)
        denom = data1 + eps
        pct = 100.0 * (data2 - data1) / denom

        # Optionally, NaN/Inf handling can be logged (not altering data here)
        n_nan = int(isnan(pct).sum()) if hasattr(pct, 'sum') else 0
        n_inf = int(isinf(pct).sum()) if hasattr(pct, 'sum') else 0
        if n_nan or n_inf:
            logger.warning(
                "Variable '%s' produced %d NaN and %d Inf values. Consider increasing --eps.",
                var_name, n_nan, n_inf
            )

        # Store back into file1's in-memory structure, replacing the data array
        nessy_1.variables[var_name]["data"] = pct

        # Add/augment variable attributes to reflect the new meaning
        nessy_1.variables[var_name]["long_name"] = f"{var_name} (percent change)"
        nessy_1.variables[var_name]["units"] = "%"
        nessy_1.variables[var_name]["_min"] = f"{nessy_1.get_min(var_name):.3e}"
        nessy_1.variables[var_name]["_max"] = f"{nessy_1.get_max(var_name):.3e}"
        nessy_1.variables[var_name]["_nes_operation"] = "diffper"
        nessy_1.variables[var_name]["_nes_denominator_file"] = file1
        nessy_1.variables[var_name]["_nes_numerator_file"] = file2
        nessy_1.variables[var_name]["_nes_eps"] = eps

    # --- Write output ---
    logger.info("Writing output NetCDF: %s", output_file)

    nessy_1.to_netcdf(output_file)
    logger.info("Done. Percentage differences saved to: %s", output_file)


__all__ = ["diffper"]
