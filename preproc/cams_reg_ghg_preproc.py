# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import os
from nes import create_nes
from datetime import datetime
import pandas as pd
import numpy as np

from .cli_logger import get_cli_logger

logger = get_cli_logger("cams_reg_ghg")

# ============================ Dataset notes ============================
"""
Dataset access:
    Contact hugo.deniervandergon@tno.nl or jeroen.kuenen@tno.nl.

Reference:
    Kuenen, J. J. P., Visschedijk, A. J. H., Jozwicka, M., and Denier van der Gon, H. A. C.
    TNO-MACC_II emission inventory: a multi-year (2003-2009) consistent high-resolution
    European emission inventory for air quality modelling.
    Atmospheric Chemistry and Physics, 14, 10963-10976, 2014.

Besides citing HERMES_GR, users should also acknowledge the corresponding emission
inventory in their works.
"""

# ===================== Example execution parameters ====================
# These constants provide a simple standalone configuration for running
# the preprocessing manually. The main preprocessing function remains
# generic and can be reused with different input paths and years.
ORIGINAL_FILE = '/esarchive/recon/ecmwf/cams_reg_ghgv51/original_files/CSV/CAMS-REG-GHG_v5_1_emissions_year<year>.csv'
OUTPUT_PATH = '/esarchive/recon/ecmwf/cams_reg_ghgv221/yearly_mean'
# FROM 2000 to 2015
YEAR = 2015


def calculate_grid_definition(in_path):
    """
    Derive the regular grid definition from the input emissions table.

    The input CSV provides rounded latitude and longitude coordinates for the
    cell centres. This helper reconstructs the complete regular global grid
    using the minimum coordinate spacing found in the table.

    Parameters
    ----------
    in_path : str
        Path to the input emissions table.

    Returns
    -------
    tuple
        Tuple containing:

        - latitude array,
        - longitude array,
        - minimum latitude increment,
        - minimum longitude increment.
    """
    dataframe = pd.read_table(in_path, sep=';')
    # Point sources are not filtered out because they are already reported at
    # the centre of the corresponding grid cell in the original table.
    # dataframe = dataframe[dataframe.SourceType != 'P']

    # Reconstruct the longitude coordinate from the rounded cell centres.
    lons = np.sort(np.unique(dataframe.Lon_rounded))
    lons_interval = lons[1:] - lons[:-1]

    # Reconstruct the latitude coordinate from the rounded cell centres.
    lats = np.sort(np.unique(dataframe.Lat_rounded))
    lats_interval = lats[1:] - lats[:-1]

    lats = np.arange(-90 + lats_interval.min() / 2, 90, lats_interval.min(), dtype=np.float64)
    lons = np.arange(-180 + lons_interval.min() / 2, 180, lons_interval.min(), dtype=np.float64)

    return lats, lons, lats_interval.min(), lons_interval.min()


def get_pollutants(in_path):
    """
    Extract the list of pollutant columns from the input emissions table.

    Parameters
    ----------
    in_path : str
        Path to the ASCII/CSV emissions table.

    Returns
    -------
    list[str]
        Pollutant column names present in the input file.
    """
    columns = list(pd.read_table(in_path, sep=';', nrows=1).columns)
    return columns[6:]


def create_pollutant_empty_list(in_path, len_c_lats, len_c_lons):
    """
    Create the pollutant metadata and empty data arrays for one sector.

    The returned structure stores the original pollutant name used in the
    source table, the target output name, the output units, and an empty
    two-dimensional array that will later be filled with yearly totals.

    Parameters
    ----------
    in_path : str
        Path to the input emissions table.
    len_c_lats : int
        Number of latitude points in the reconstructed grid.
    len_c_lons : int
        Number of longitude points in the reconstructed grid.

    Returns
    -------
    list[dict]
        List of pollutant dictionaries ready to be filled with data.
    """
    pollutant_list = []
    for pollutant in get_pollutants(in_path):
        aux_dict = {}
        if pollutant == 'CO2_ff':
            aux_dict['name'] = 'co2_fossil'
        elif pollutant == 'CO2_bf':
            aux_dict['name'] = 'co2_bio'
        else:
            aux_dict['name'] = pollutant.lower()
        aux_dict['TNO_name'] = pollutant
        aux_dict['units'] = 'kg.m-2.s-1'
        # aux_dict['units'] = 'Mg.km-2.year-1'
        aux_dict['data'] = np.zeros((len_c_lats, len_c_lons))
        pollutant_list.append(aux_dict)
    return pollutant_list


# The input CAMS-REG-GHG table contains yearly greenhouse-gas emissions by
# GNFR sector and grid cell. This preprocessing reconstructs the regular
# grid, aggregates emissions by sector and cell, and writes one yearly NES
# file per pollutant and GNFR sector.
def cams_reg_ghg_preproc(original_file, output_path, year, overwrite=False):
    """
    Convert the CAMS-REG-GHG yearly table into HERMES-ready yearly outputs.

    The function reconstructs the source regular grid from the rounded cell
    centres in the CSV file, aggregates yearly emissions by GNFR sector and
    grid cell, converts them to ``kg.m-2.s-1``, and writes one output NetCDF
    file per pollutant and sector.

    Parameters
    ----------
    original_file : str
        Input file template containing the ``<year>`` placeholder.
    output_path : str
        Root directory where the preprocessed yearly files will be written.
    year : int
        Year to process.
    overwrite : bool, optional
        Whether to overwrite existing output files. If ``False``, existing output
        files are skipped.

    Returns
    -------
    bool
        ``True`` when the preprocessing finishes successfully.
    """
    in_file = original_file.replace('<year>', str(year))

    if os.path.exists(in_file):
        # Convert yearly totals to per-second fluxes before normalizing by
        # grid-cell area.
        unit_factor = 1. / (365. * 24. * 3600.)  # To pass from kg/year to Kg/s
        # Alternative conversion kept here for reference during debugging or
        # validation of the original inventory units.
        # unit_factor = 1000000  # To pass from Mg/m2.year to Mg/Km2.year

        lats, lons, min_lats_interval, min_lons_interval = calculate_grid_definition(in_file)

        # Read the yearly emissions table.
        dataframe = pd.read_table(in_file, sep=';')

        # Map each rounded latitude and longitude to the corresponding row and
        # column position in the reconstructed regular grid.
        dataframe.loc[:, 'row_lat'] = np.array(
            (dataframe.Lat_rounded - (-90 + min_lats_interval / 2)) / min_lats_interval, dtype=np.int64)
        dataframe.loc[:, 'col_lon'] = np.array(
            (dataframe.Lon_rounded - (-180 + min_lons_interval / 2)) / min_lons_interval, dtype=np.int64)

        # Process each GNFR sector independently and generate one output file
        # per pollutant.
        for name, group in dataframe.groupby('GNFR_Sector'):
            pollutant_list = create_pollutant_empty_list(in_file, len(lats), len(lons))
            # Aggregate all emissions falling in the same grid cell.
            group = group.groupby(['row_lat', 'col_lon']).sum().reset_index()

            for i in range(len(pollutant_list)):
                # Fill the 2D grid with the yearly total of the current
                # pollutant for this GNFR sector.
                pollutant_list[i]['data'][group.row_lat, group.col_lon] += group[pollutant_list[i]['TNO_name']]
                # Expand the array to the NES convention: (time, level, lat, lon).
                pollutant_list[i]['data'] = pollutant_list[i]['data'].reshape((1, 1,) + pollutant_list[i]['data'].shape)

                # Create one output directory per pollutant and GNFR sector.
                aux_output_dir = os.path.join(output_path, f'{pollutant_list[i]["name"]}_gnfr_{name}')
                aux_output_path = os.path.join(aux_output_dir, f'{pollutant_list[i]["name"]}_{year}.nc')
                if os.path.exists(aux_output_path) and not overwrite:
                    logger.info(f"Skipping existing file: {aux_output_path}")
                    continue
                elif os.path.exists(aux_output_path) and overwrite:
                    logger.info(f"Overwriting existing file: {aux_output_path}")
                    # remove existing file before writing the new one to avoid issues with the NES writer
                    os.remove(aux_output_path)

                os.makedirs(aux_output_dir, exist_ok=True)

                # Create an NES object describing the reconstructed regular
                # grid. A single time step is used because the output is yearly.
                nessy = create_nes(
                    comm=None,
                    info=False,
                    projection="regular",
                    lat_orig=lats[0] - min_lats_interval / 2,
                    lon_orig=lons[0] - min_lons_interval / 2,
                    inc_lat=min_lats_interval,
                    inc_lon=min_lons_interval,
                    n_lat=len(lats),
                    n_lon=len(lons),
                    times=[datetime(year, month=1, day=1)],
                )

                # Add spatial bounds required by the output grid definition.
                nessy.create_spatial_bounds()
                logger.info("Calculating cell area...")
                cell_area = nessy.calculate_grid_area()
                logger.info("Cell area calculated.")

                # Explicitly set the area units in the output metadata.
                nessy.cell_measures["cell_area"] = {"data": cell_area,
                                                    "units": "m2"}

                # Convert yearly mass totals to fluxes and store the result as
                # a NES variable.
                nessy.variables[pollutant_list[i]['name']] = {
                    "data": pollutant_list[i]['data'] * unit_factor/cell_area,
                    "units": pollutant_list[i]['units'],
                    "description": f"CAMS regional GHG emissions of {pollutant_list[i]['name']} for GNFR sector {name}",
                    "short_name": pollutant_list[i]['name'],
                }

                # Add dataset provenance and rewriting metadata to the output.
                nessy.global_attrs = {
                            'references': 'J. J. P. Kuenen, A. J. H. Visschedijk, M. Jozwicka, and '
                                          'H. A. C. Denier van der Gon TNO-MACC_II emission inventory; a multi-year '
                                          '(2003-2009) consistent high-resolution European emission inventory '
                                          'for air quality modelling Atmospheric Chemistry and Physics 14 '
                                          '10963-10976 2014',
                            'comment': 'Re-writing done by Carles Tena (carles.tena@bsc.es) from the BSC-CNS '
                                       '(Barcelona Supercomputing Center)'
                }

                # Write the yearly NES file to disk.
                nessy.to_netcdf(aux_output_path)

    else:
        raise FileNotFoundError(f'File {in_file} does not exist.')
    return True


# Example standalone execution using the module-level testing parameters
# defined at the top of this script.
if __name__ == '__main__':
    cams_reg_ghg_preproc(original_file=ORIGINAL_FILE, output_path=OUTPUT_PATH, year=YEAR, overwrite=False)
