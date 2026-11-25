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


logger = get_cli_logger("gridarea")


def gridarea(filepath: str, output_file: str = None) -> None:
    """
    Compute the **grid cell area** for a given NetCDF file and write the result
    to disk.

    This function loads a NetCDF dataset using the NES interface and computes
    the surface area of each grid cell based on its spatial definition
    (latitude/longitude bounds). The computed area is stored internally in the
    NES object and then written to an output NetCDF file.

    Notes
    -----
    * The computation relies on the spatial bounds (`lat_bnds`, `lon_bnds`).
      If they are not available, they will be generated automatically.
    * The area is typically expressed in square meters (m²), depending on the
      implementation of `calculate_grid_area()`.
    * The output overwrites the input file if `output_file` is not provided.

    Parameters
    ----------
    filepath : str
        Path to the input NetCDF file.
    output_file : str, optional
        Path to the output NetCDF file where the grid cell area will be stored.
        If None, the input file will be overwritten.

    Raises
    ------
    Exception
        If no valid method is available to write the NetCDF output.

    Workflow
    --------
    1. Open NetCDF file using NES.
    2. Load data into memory.
    3. Compute grid cell area using spatial bounds.
    4. Write updated dataset to disk.

    Examples
    --------
    >>> gridarea("input.nc", "output.nc")

    This will compute the grid cell area of `input.nc` and save the result
    in `output.nc`.
    """
    # If no output is provided, overwrite input file
    if output_file is None:
        logger.info(f"No output file provided, overwriting input with new cell area: {filepath}")
        output_file = filepath

    # --- Open input file ---
    logger.info(f"Opening NetCDF file: {filepath}")
    nessy = open_netcdf(filepath)

    # Load dataset into memory (required before computations)
    nessy.load()

    # Compute grid cell area based on spatial bounds
    logger.info("Computing grid cell area...")
    nessy.calculate_grid_area(overwrite=True)

    # --- Write output ---
    logger.info("Writing output NetCDF: %s", output_file)

    # Try different writer methods depending on NES implementation
    wrote = False
    for writer_name in ("to_netcdf", "write", "save", "to_nc"):
        writer = getattr(nessy, writer_name, None)
        if callable(writer):
            writer(output_file)
            wrote = True
            break

    # Raise error if no writer is available
    if not wrote:
        logger.error("Cannot write output: no suitable writer method found on NES object.")
        raise Exception(
            "No writer found (tried: to_netcdf, write, save, to_nc). "
            "Please implement or choose the correct method."
        )

    logger.info("Done. Cell area saved to: %s", output_file)


__all__ = ["gridarea"]
