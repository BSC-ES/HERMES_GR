# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

from nes.load_nes import open_netcdf
from mpi4py import MPI
from .cli_logger import get_cli_logger

logger = get_cli_logger("reorder_longitudes")


def reorder_longitudes(input_file: str, output_file: str):
    """
    Convert longitudes in a NetCDF file to the [-180, 180] range and save the modified file.

    Parameters
    ----------
    input_file : str
        Path to the input NetCDF file.
    output_file : str
        Path where the reordered NetCDF file will be saved.

    Raises
    ------
    ValueError
        If the script is run using more than one MPI process.

    Notes
    -----
    This function must be executed in serial mode only.
    It uses `convert_longitudes()` from the NES API, which updates coordinate values
    and ensures variables depending on longitude are shifted accordingly.
    """
    if MPI.COMM_WORLD.Get_size() > 1:
        logger.error("This script must be run with a single process (serial mode only).")
        raise ValueError("This script must be run with a single process (serial mode only).")

    logger.info(f"Opening NetCDF file: {input_file}")
    # Open the input NetCDF file
    nc = open_netcdf(input_file)

    # Load data into memory
    nc.load()
    logger.info("NetCDF file loaded into memory.")

    # Reorder longitudes from [0, 360] to [-180, 180]
    nc.convert_longitudes()
    logger.info("Converted longitudes from [0, 360] to [-180, 180].")

    # Save the result to the output path
    nc.to_netcdf(output_file)
    logger.info(f"Saved reordered NetCDF to: {output_file}")

    return True
