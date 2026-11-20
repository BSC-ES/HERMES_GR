# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

"""Preprocess CAMS global shipping emissions into monthly mean NetCDF files.

This script reads annual CAMS global shipping emission files, applies a correction to
latitude and longitude centroid coordinates, computes grid-cell areas, aggregates the
original time series into one monthly field per pollutant, converts the accumulated
mass into emission flux units, and writes one NetCDF file per output pollutant and
month.

Notes
-----
- The input files are expected to contain annual data for a single pollutant.
- The output files contain monthly mean emissions expressed in ``kg.m-2.s-1``.
- ``VOC_SUM`` is split into multiple output VOC species using predefined scaling
  factors.
"""
from .cli_logger import get_cli_logger
import os
from nes import open_netcdf
import numpy as np
from calendar import monthrange
from copy import deepcopy

logger = get_cli_logger("cams_glob_ship")


# ============== TESTING CONFIGURATION PARAMETERS ======================
# Reference year to preprocess.
YEAR = 2022
fmi = False

# Template path to the original annual CAMS global shipping files.
# `<in_pollutant>` and `<year>` are replaced at runtime.

# ORIGINAL PATH up until 2022
if YEAR <= 2022:
    CAMS_GLOB_SHIP_ORIGINAL_PATH = ("/esarchive/recon/ecmwf/cams_glob_shipv32/original_files/"
                                    "CAMS-GLOB-SHIP_Glb_0.1x0.1_anthro_<in_pollutant>_v3.2_daily_<year>.nc")
    OUTPUT_PATH = "/esarchive/recon/ecmwf/cams_glob_shipv32/monthly_mean/<pollutant>/<pollutant>_<YYYYMM>.nc"
# ORIGINAL PATH from 2023 onwards
else:
    CAMS_GLOB_SHIP_ORIGINAL_PATH = ("/esarchive/recon/ecmwf/cams_glob_shipv33/original_files/"
                                    "CAMS-GLOB-SHIP_Glb_0.1x0.1_anthro_<in_pollutant>_v3.3_daily_<year>.nc")
    OUTPUT_PATH = "/esarchive/recon/ecmwf/cams_glob_shipv33/monthly_mean/<pollutant>/<pollutant>_<YYYYMM>.nc"

if fmi:
    CAMS_GLOB_SHIP_ORIGINAL_PATH = ("/esarchive/recon/ecmwf/cams_glob_shipv32/original_files/"
                                    "<in_pollutant>_allHeights_<year>-01-01T00_<year>-12-31T00.nc")


def read_input_data(filepath, coordinate_correction=False):
    """Read an original CAMS shipping file and prepare its spatial metadata.

    Parameters
    ----------
    filepath : str
        Path to the input NetCDF file.

    Returns
    -------
    NES
        Loaded NES object with corrected coordinates, spatial bounds, and grid-cell
        areas.

    Notes
    -----
    The input coordinate centroids are shifted by 0.05 degrees in both latitude and
    longitude because the original files place the first centroid at ``(-180, 90)``
    instead of ``(-179.95, 89.95)``.
    """
    nessy = open_netcdf(filepath)

    if coordinate_correction:
        # Correct centroid coordinates stored in the original files.
        nessy.lat['data'] = np.round(np.array(nessy.lat['data'], dtype=np.float64) - 0.05, decimals=4)
        nessy.lon['data'] = np.round(np.array(nessy.lon['data'], dtype=np.float64) + 0.05, decimals=4)
        # Correct the full coordinate arrays too when they are available.
        if nessy._full_lat is not None:
            nessy._full_lat['data'] = np.round(np.array(nessy._full_lat['data'], dtype=np.float64) - 0.05, decimals=4)
            nessy._full_lon['data'] = np.round(np.array(nessy._full_lon['data'], dtype=np.float64) + 0.05, decimals=4)

    # Load variable data and derive the spatial metadata required for the unit
    # conversion from mass to flux.
    nessy.load()
    nessy.create_spatial_bounds()
    logger.info("Calculating grid area...")
    nessy.calculate_grid_area()
    logger.info("Calculated grid area")

    return nessy


def get_monthly_data(src_nessy, year, month,
                     out_pollutant, in_pollutant, factor=None, fmi=False):
    """Aggregate one month of data and convert it to emission flux units.

    Parameters
    ----------
    src_nessy : NES
        Source NES object containing the original annual time series.
    year : int
        Reference year of the processed dataset.
    month : int
        Month to extract, from 1 to 12.
    # in_pollutant : str
    #     Pollutant name in the source dataset.
    out_pollutant : str
        Pollutant name to use in the output dataset.
    in_pollutant : str
        Pollutant name in the source dataset. If Eccad format, this is always "shipping".
    factor : float, optional
        Multiplicative factor applied to the aggregated monthly data. This is mainly
        used to split ``VOC_SUM`` into multiple output VOC species.
    fmi : bool, optional
        Whether the input data came from FMI. If ``True``,
        unit conversions must be applied. If False, the input data is assumed
        to be from ECCAD and units are in ``kg.m-2.s-1`` so no unit conversion is applied.

    Returns
    -------
    NES
        NES object containing a single monthly field for the requested output
        pollutant.
    """
    dst_nessy = src_nessy.copy(copy_vars=False)
    dst_nessy.set_communicator(src_nessy.comm)
    dst_nessy.cell_measures['cell_area'] = deepcopy(src_nessy.cell_measures['cell_area'])

    # Identify all time steps belonging to the requested month.
    month_indexes = []
    for i_time, time in enumerate(dst_nessy.time):
        if time.month == month:
            month_indexes.append(i_time)
    month_indexes = np.array(month_indexes)

    # Store a single representative time stamp and its time bounds for the month.
    dst_nessy.set_time([src_nessy.time[month_indexes.min()]])
    dst_nessy.set_time_bnds([[src_nessy.time[month_indexes.min()], src_nessy.time[month_indexes.max()]]])

    # Accumulate all time steps within the month.
    if fmi:
        aux_data = src_nessy.variables[in_pollutant]['data'][month_indexes.min():month_indexes.max() + 1, :].sum(
            axis=0, keepdims=True)
        # Convert accumulated monthly mass [kg] into mean flux [kg.m-2.s-1].
        sec_per_month = monthrange(year, month)[1] * 24 * 60 * 60
        aux_data = aux_data / (dst_nessy.cell_measures['cell_area']['data'] * sec_per_month)

    else:
        aux_data = src_nessy.variables["shipping"]['data'][month_indexes.min():month_indexes.max() + 1, :].sum(
            axis=0, keepdims=True)
        aux_data = aux_data / dst_nessy.cell_measures['cell_area']['data']

    if factor is not None:
        aux_data *= factor

    dst_nessy.variables[out_pollutant] = {'data': aux_data, 'units': 'kg.m-2.s-1'}

    return dst_nessy


def cams_glob_ship_preproc(original_file, output_path, year, pollutant_info=None, overwrite=False, fmi=False):
    """Generate monthly mean NetCDF files for all configured pollutants.

    Parameters
    ----------
    original_file : str
        Template path to the original annual input files.
    output_path : str
        Template path for the generated monthly mean output files.
    pollutant_info : dict
        Mapping between input pollutant names and output pollutant names or VOC split
        factors.
    overwrite : bool, optional
        Whether to overwrite existing output files. If ``False``, existing output
        files are skipped.
    year : int
        Reference year to preprocess.
    fmi : bool, optional
        Whether the input data came from FMI. If ``True``, unit conversions must be applied.
        If False, the input data is assumed to be from ECCAD and units are in
        ``kg.m-2.s-1`` so no unit conversion is applied. Default is ``False``.

    Returns
    -------
    None
        This function writes the generated NetCDF files to disk.
    """
    if pollutant_info is None:
        # Mapping between pollutant names in the original CAMS files and output pollutant names.
        #
        # Most entries map one input pollutant to one output pollutant. The `VOC_SUM` entry is
        # distributed into several VOC species using fixed fractions.

        pollutant_info = {
                          'NOx': 'nox_no',
                          'SOx': 'sox',
                          'SO4': 'so4',
                          'Ash': 'ash',
                          'EC': 'ec',
                          'OC': 'oc',
                          'CH4': 'ch4',
                          'CO2': 'co2',
                          'CO': 'co',
                          'VOC_SUM': {'voc01': 0, 'voc02': 0.01, 'voc03': 0.01, 'voc04': 0.02, 'voc05': 0.02,
                                      'voc06': 0.3, 'voc07': 0.12, 'voc08': 0.03, 'voc09': 0.04, 'voc10': 0,
                                      'voc11': 0, 'voc12': 0.05, 'voc13': 0.02, 'voc14': 0.015, 'voc15': 0.02,
                                      'voc16': 0, 'voc17': 0.205, 'voc18': 0, 'voc19': 0, 'voc20': 0,
                                      'voc21': 0.06, 'voc22': 0.065, 'voc23': 0.015, 'voc24': 0, 'voc25': 0}
                          }

    if fmi:
        # map NOx to nox_no2 if the source is FMI
        pollutant_info["NOx"] = "nox_no2"

    for in_pollutant, out_pollutant in pollutant_info.items():
        logger.info("Loading {0}".format(in_pollutant))

        input_file_path = original_file.replace('<in_pollutant>', in_pollutant).replace('<year>', str(year))
        if os.path.exists(input_file_path):
            logger.info(f"Processing pollutant {in_pollutant} using input file \n {input_file_path}")

            src_nessy = read_input_data(input_file_path, coordinate_correction=True)

            for month in range(1, 13):
                logger.info("\tCalculating month {0} data".format(month))
                if isinstance(out_pollutant, dict):
                    # Split one input pollutant into several output pollutants.
                    for aux_pollutant, aux_factor in out_pollutant.items():
                        if aux_factor > 0:
                            logger.info("\t\t{0}".format(aux_pollutant))
                            dst_nessy = get_monthly_data(src_nessy, year, month,
                                                         aux_pollutant,
                                                         in_pollutant=in_pollutant,
                                                         factor=aux_factor,
                                                         fmi=fmi)

                            aux_out_path = output_path.replace('<pollutant>', aux_pollutant).replace(
                                "<YYYYMM>", "{0}{1}".format(year, str(month).zfill(2)))
                            if os.path.exists(aux_out_path) and not overwrite:
                                logger.info(f"\t\tSkipping existing file: {aux_out_path}")
                                continue
                            os.makedirs(os.path.dirname(aux_out_path), exist_ok=True)
                            dst_nessy.to_netcdf(aux_out_path, serial=True)
                else:
                    dst_nessy = get_monthly_data(src_nessy, year, month, out_pollutant,
                                                 in_pollutant=in_pollutant, fmi=fmi)

                    aux_out_path = (
                        output_path
                        .replace('<pollutant>', out_pollutant)
                        .replace(
                            "<YYYYMM>",
                            "{0}{1}".format(year, str(month).zfill(2))
                        )
                    )
                    if os.path.exists(aux_out_path) and not overwrite:
                        logger.info(f"\tSkipping existing file: {aux_out_path}")
                        continue
                    os.makedirs(os.path. dirname(aux_out_path), exist_ok=True)
                    dst_nessy.to_netcdf(aux_out_path, serial=True)
        else:
            logger.warning(f"Input file not found for pollutant {in_pollutant}: \n {input_file_path}")
    return None


# Run the full preprocessing workflow for the configured year.
if __name__ == '__main__':
    cams_glob_ship_preproc(original_file=CAMS_GLOB_SHIP_ORIGINAL_PATH, output_path=OUTPUT_PATH, year=YEAR,
                           overwrite=False, coordinate_correction=False, fmi=fmi)
