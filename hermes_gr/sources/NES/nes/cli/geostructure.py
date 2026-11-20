# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

"""
NES CLI Utility: Convert NetCDF to geospatial vector format (Shapefile, GeoJSON)

This script defines a function `nc2geostructure` that extracts a selected time step and level
from a NetCDF file and writes the corresponding geospatial structure as a shapefile,
optionally filtering specific variables.

Intended to be used as part of a CLI interface (e.g. via `nes geostructure`).
"""

from typing import Optional, List

from nes import open_netcdf
from .cli_logger import get_cli_logger

logger = get_cli_logger("geostructure")


def nc2geostructure(
    input_file: str,
    output_file: str,
    var_list: Optional[List[str]] = None,
    time_step: int = 0,
    level: int = 0,
) -> None:
    """
    Convert a selected time step and level from a NetCDF file into a geospatial vector format.

    This function extracts emissions or concentration data from a NetCDF file and exports it
    to a shapefile or GeoJSON format, optionally filtering specific variables.

    Parameters
    ----------
    input_file : str
        Path to the source NetCDF file.
    output_file : str
        Path where the output file will be written. Can be a Shapefile (.shp) or GeoJSON (.geojson).
    var_list : list of str, optional
        List of variable names to include in the output. If None, all available variables are used.
    time_step : int, default=0
        Index of the time step to extract.
    level : int, default=0
        Index of the vertical level to extract.

    Returns
    -------
    None
    """
    logger.info(f"Opening NetCDF file: {input_file}")
    nessy = open_netcdf(input_file)

    logger.info("Determining variable list...")
    if var_list is None:
        var_list = nessy.variables.keys()
    logger.info(f"Variables selected: {list(var_list)}")

    if level == -1:
        logger.info(f"Summing all vertical levels.")
        nessy.sum_axis('Z')
        level = 0

    if time_step == -1:
        logger.info(f"Summing all time-steps.")
        nessy.sum_axis('T')
        time_step = 0

    logger.info(f"Selecting time step {time_step} and level {level}")
    nessy.sel(
        time_min=nessy.time[time_step],  # Minimum time index to extract
        time_max=nessy.time[time_step],  # Maximum time index to extract (same as min for single step)
        lev_min=level,                 # Minimum vertical level to extract
        lev_max=level                  # Maximum vertical level to extract (same as min for single level)
    )

    logger.info("Filtering and loading selected variables into memory...")
    nessy.keep_vars(var_list)
    nessy.load()

    logger.info("Creating geospatial structure...")
    nessy.create_shapefile()
    nessy.add_variables_to_shapefile(idx_time=0, idx_lev=0, var_list=var_list)

    logger.info(f"Writing shapefile to: {output_file}")
    nessy.write_shapefile(output_file)

    return None


def nc2mbtiles(
    input_file: str, output_path: str, mbtiles_file: str, geostructure: str = None
):
    """Converts a monthly NetCDF into daily mbtiles

    Parameters
    ----------
    input_file : str
        Input NetCDF
    output_path : str
        Path where the mbtiles should be saved
    mbtiles_file: str
        Name of the mbtiles to be saved, this will be used as the prefix for the date portion and file extension
    geostructure: str
        Optional geostructure file to be provided, allowing to skip the manual computation, by default None
    """
    nessy = open_netcdf(input_file, parallel_method="T")
    nessy.load()
    nessy.to_mbtile(output_path, mbtiles_file, geostructure)
