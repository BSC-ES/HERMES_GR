# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

"""
NES CLI utility: apply mask to a NetCDF file with same grid.

This module exposes `mask()` to be used by the CLI subcommand
`nes mask -f file.nc -m mask.nc -o output.nc`.

Formula (per variable, per cell):

    output = file * mask
"""

from .cli_logger import get_cli_logger
from nes.load_nes import open_netcdf
from numpy import isinf, isnan
from typing import Optional, List
import warnings
import numpy as np


logger = get_cli_logger("mask")


def mask(file: str, mask: str, output_file: str,
         var_list: list[str] | None = None, var_mask: str | None = None,
         inverse_mask: bool = False) -> None:
    """
    Compute the **applied mask** to a NetCDF file and write the
    result to a new NetCDF file.

    The computation is performed **variable-wise** and **cell-wise** using the
    following formula:

        output = file * mask

    If `inverse_mask=True`, the mask is inverted before applying:

        output = file * (1 - mask)

    Notes
    -----
    * `file` is treated as the baseline.
    * Only variables present in file and selected in `var_list`
      (if provided) are processed.

    Parameters
    ----------
    file : str
        Path to the baseline NetCDF file.
    mask : str
        Path to the mask NetCDF file.
    output_file : str
        Path to the output NetCDF file that will store the baseline file with the mask applied.
    var_list : list[str] | None, optional
        Subset of variable names to process. If None, all variables in `file` will be used.
    var_mask : str | None, optional
        Name of the variable of the mask file. If None, the name of the variable
        will be assumed to be `mask`.
    inverse_mask : bool, optional
        If True, invert the mask values before applying (`1 - mask`). A warning is
        emitted if mask values are outside the [0, 1] range.

    Raises
    ------
    KeyError
        If any variable in `var_list` is not present in file.
        If `var_mask` is not present in the mask file.
    ValueError
        If a processed variable has incompatible shapes between the two files.
    Exception
        If the output cannot be written.
    """
    # --- Open input files ---
    logger.info(f"Opening baseline NetCDF (file): {file}")
    nessy = open_netcdf(file)
    logger.info(f"Opening mask NetCDF (mask): {mask}")
    masky = open_netcdf(mask)

    file_vars = nessy.variables.keys()
    if var_list is None:
        target_vars = sorted(file_vars)
    else:
        missing = [v for v in var_list if v not in file_vars]
        if missing:
            raise KeyError(
                "Variables not present in both files: " + ", ".join(missing)
            )
        target_vars = var_list

    # Restrict to the chosen variables and load
    if target_vars:
        nessy.keep_vars(target_vars)

    if var_mask is None:
        var_mask = 'mask'
    masky.keep_vars(var_mask)

    nessy.load()
    logger.info(f"Baseline NetCDF loaded (file): {file}")
    masky.load()
    data_mask = masky.variables[var_mask]["data"]
    logger.info(f"Mask NetCDF loaded (mask): {mask}")

    if inverse_mask:
        # Invert mask values; warn if the mask is not normalized to [0, 1].
        if not ((data_mask.min() >= 0) and (data_mask.max() <= 1)):
            warnings.warn(
                "Mask values out of [0,1] range",
                UserWarning
            )
        data_mask = 1 - data_mask

    logger.info("Applying mask for %d variables", len(target_vars))

    # --- Apply mask per variable ---
    for var_name in target_vars:
        logger.debug("Computing variable: %s", var_name)

        data = nessy.variables[var_name]["data"]
        # Check shape compatibility
        if getattr(data, "shape", None) != getattr(data_mask, "shape", None):
            raise ValueError(
                f"Variable '{var_name}' has incompatible shapes: "
                f"{getattr(data, 'shape', None)} vs {getattr(data_mask, 'shape', None)}"
            )

        masked_data = np.array(data * data_mask, dtype=data.dtype)
        nessy.variables[var_name]["data"] = masked_data

    # --- Write output ---
    logger.info("Writing output NetCDF: %s", output_file)
    wrote = False
    for writer_name in ("to_netcdf", "write", "save", "to_nc"):
        writer = getattr(nessy, writer_name, None)
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

    logger.info("Done. File with applied mask saved to: %s", output_file)


__all__ = ["mask"]
