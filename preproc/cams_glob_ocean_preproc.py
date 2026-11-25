# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import os
import numpy as np
from datetime import datetime
from nes import Nes, open_netcdf, create_nes

from .cli_logger import get_cli_logger

logger = get_cli_logger("cams_glob_ocean_v31")

# ===================== Example execution parameters ====================
# These constants provide a simple standalone configuration for running
# the preprocessing manually. The main preprocessing function remains
# generic and can be reused with different input paths, years, and
# pollutant selections.
ORIGINAL_FILE = ('/esarchive/recon/ecmwf/cams_glob_oceanv31/original_files/'
                 'CAMS-GLOB-OCE_Glb_0.5x0.5_oce_<pollutant>_v3.1_monthly_<year>.nc')
OUTPUT_PATH = '/esarchive/recon/ecmwf/cams_glob_oceanv31'
YEAR = 2020
# ==============================================================


def obtain_lats_lons(nessy):
    """
    Extract latitude and longitude coordinates from the input dataset.

    This helper also computes the minimum latitude and longitude spacing,
    which are later used to define the regular output grid in NES.

    Parameters
    ----------
    nessy : Nes
        NES object containing the ``lat`` and ``lon`` coordinate variables.

    Returns
    -------
    tuple
        A tuple containing:

        - latitude array,
        - longitude array,
        - minimum latitude increment,
        - minimum longitude increment.
    """
    lats = nessy.lat["data"]
    lons = nessy.lon["data"]

    lats_interval = lats[1:] - lats[:-1]
    min_lats_interval = lats_interval.min()

    lons_interval = lons[1:] - lons[:-1]
    min_lons_interval = lons_interval.min()

    return lats, lons, min_lats_interval, min_lons_interval


def obtain_data(nessy, year):
    """
    Extract the 12 monthly ocean-emission fields for a given year.

    The input files may contain multiple time steps along the time dimension.
    This function locates the first month of the requested year, loads the
    ocean-emission variable, and returns the 12 monthly slices for that year.

    Parameters
    ----------
    nessy : Nes
        NES object containing the time coordinate and ocean-emission data.
    year : int
        Year to extract from the input dataset.

    Returns
    -------
    numpy.ndarray
        Three-dimensional array with shape ``(time, lat, lon)`` containing
        the 12 monthly fields for the selected year.
    """
    # Normalize the input timestamps to the first day of each month so the
    # requested year can be located reliably.
    time = np.array([datetime(year=x.year, month=x.month, day=1) for x in nessy.time])
    i_time = np.where(time == datetime(year=year, month=1, day=1))[0][0]
    # Load the variable containing oceanic emissions.
    nessy.load("emi_oceans")
    data = nessy.variables["emi_oceans"]["data"][i_time:i_time + 12, :, :]
    return data


# The input CAMS-GLOB-OCE files contain monthly oceanic emissions by
# pollutant. This preprocessing reorganizes the data into one output file
# per month, preserving the original grid definition and metadata in NES
# format.
def cams_glob_ocean_preproc(original_file, output_path, year, pollutant_list=None, overwrite=False):
    """
    Convert CAMS-GLOB-OCEAN monthly files into HERMES-ready monthly outputs.

    For each requested pollutant, the function reads the original
    CAMS-GLOB-OCEAN file, builds an NES regular grid using the source spatial
    coordinates, and writes one output NetCDF file per month under the
    ``monthly_mean`` directory structure.

    Parameters
    ----------
    original_file : str
        Input file template containing the ``<pollutant>`` and ``<year>`` placeholders.
    output_path : str
        Root directory where the preprocessed monthly files will be written.
    year : int
        Year to process.
    pollutant_list : list[str], optional
        Pollutants to process. If ``None``, the default CAMS-GLOB-OCEAN list used by this script is processed.
    overwrite : bool, optional
        Whether to overwrite existing output files. If ``False``, existing output
        files are skipped.
    """
    # Use the default CAMS-GLOB-OCEAN pollutant list when none is provided.
    if pollutant_list is None:
        pollutant_list = ['DMS']

    for pollutant in pollutant_list:
        in_file = original_file.replace('<pollutant>', pollutant).replace('<year>', str(year))
        if os.path.exists(in_file):
            logger.info(f"Processing pollutant {pollutant} using input file \n {in_file}")

            # Open the source CAMS NetCDF file with NES.
            nessy_in = open_netcdf(in_file)

            # Read the source grid definition to reproduce the same
            # regular grid in the output dataset.
            lats, lons, min_lats_interval, min_lons_interval = obtain_lats_lons(nessy_in)

            # Preserve the original global metadata in the output files.
            global_attributes = nessy_in.global_attrs

            # Extract the 12 monthly fields for the selected year.
            data_year = obtain_data(nessy_in, year)

            # Create an NES object describing the output regular grid.
            # A single time step is used because one file is written per month.
            nessy = create_nes(
                    comm=None,
                    info=False,
                    projection="regular",
                    lat_orig=lats[0],
                    lon_orig=lons[0],
                    inc_lat=min_lats_interval,
                    inc_lon=min_lons_interval,
                    n_lat=len(lats),
                    n_lon=len(lons),
                    times=[datetime(year, month=1, day=1)],
                )

            # Adapt pollutant names to the target naming convention.
            # For example, ``DMS`` is exported using the species name ``c2h6s``.
            if pollutant == 'DMS':
                name = 'c2h6s'
            else:
                name = pollutant.lower()

            # Add spatial bounds required by the output grid definition.
            nessy.create_spatial_bounds()

            # Compute grid-cell area so the output file includes cell
            # measures explicitly.
            logger.info("Calculating cell area...")
            nessy.calculate_grid_area()
            logger.info("Cell area calculated.")
            # Explicitly set the area units in the output metadata.
            nessy.cell_measures["cell_area"]["units"] = "m2"

            cell_area_shape = nessy.cell_measures["cell_area"]["data"].shape

            # Copy the original global attributes to the output NES object.
            nessy.global_attrs = global_attributes

            # Create one output directory per pollutant.
            complete_output_path = os.path.join(output_path, 'monthly_mean', f'{name}')
            os.makedirs(complete_output_path, exist_ok=True)

            for month in range(1, 12+1, 1):
                # Store the monthly field as a NES variable using the
                # expected output shape: (time, level, lat, lon).
                nessy.variables[name] = {'data': data_year[month-1, :, :].reshape((1, 1,) + cell_area_shape),
                                         'units': 'kg.m-2.s-1',
                                         'description': f'CAMS-GLOB-OCE oceanic emissions of {pollutant} for the '
                                                        f'month {month} and year {year}',
                                         'short_name': name}
                # Build the output filename for the current month.
                complete_output_file = os.path.join(complete_output_path, f'{name}_{year}{str(month).zfill(2)}.nc')
                if os.path.exists(complete_output_file) and not overwrite:
                    logger.info(f"Skipping existing file: {complete_output_file}")
                    continue
                # Write the monthly NES file to disk.
                nessy.to_netcdf(complete_output_file)

        else:
            raise FileNotFoundError(f"Input file not found: {in_file}")


# Example standalone execution using the module-level testing parameters
# defined at the top of this script.
if __name__ == '__main__':
    cams_glob_ocean_preproc(original_file=ORIGINAL_FILE, output_path=OUTPUT_PATH, year=YEAR, overwrite=False)
