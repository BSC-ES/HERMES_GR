#!/usr/bin/env python

# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import os
from datetime import datetime
from warnings import warn as warning

import nes
import numpy as np
from .cli_logger import get_cli_logger

logger = get_cli_logger("edgar")


# ============== README ======================
"""
Download access:
- AP: http://edgar.jrc.ec.europa.eu/overview.php?v=432_AP
- VOC: http://edgar.jrc.ec.europa.eu/overview.php?v=432_VOC_spec

References:
- AP: https://doi.org/10.5194/essd-10-1987-2018
- VOC: https://www.atmos-chem-phys.net/17/7683/2017/

Besides citing HERMES_GR, users must also acknowledge the use of the
corresponding emission inventories in their works.
"""

# ============== TEST CONFIGURATION PARAMETERS ======================
ORIGINAL_FILE = ("/esarchive/recon/jrc/edgarv432_ap/original_files/yearly/"
                 "v432_<pollutant>_<year>_<ipcc>.0.1x0.1.nc")
# ORIGINAL_FILE = ("/esarchive/recon/jrc/edgarv432_ap/original_files/yearly/"
#                  "v432_VOC_spec_<pollutant>_<year>_<ipcc>.0.1x0.1.nc")
# ORIGINAL_FILE = ("/esarchive/recon/jrc/edgarv432_ap/original_files/monthly/"
#                  "v432_<pollutant>_2010_<month>_<ipcc>.0.1x0.1.nc")
# ORIGINAL_FILE = ("/esarchive/recon/jrc/edgarv432_ap/original_files/monthly/"
#                  "v432_VOC_spec_<pollutant>_2010_<month>_<ipcc>.0.1x0.1.nc")
MONTHLY_TEMPORAL_PROFILES = ("/esarchive/recon/jrc/edgarv432_ap/original_files/temporal_profiles/"
                             "v432_FM_<sector>.0.1x0.1.nc")
OUTPUT_PATH = "/home/jgehlen/Documents/HERMES_GR/hermesv3_gr/preproc/tests/output_edgar"
YEAR = 2010
YEARLY = False
MONTHLY = True
# ==================================================================


def get_default_ap_pollutant_list():
    """
    Get the default EDGARv4.3.2 AP pollutant list.

    Returns
    -------
    list[str]
        Default AP pollutant names.
    """
    return ['BC', 'CO', 'NH3', 'NOx', 'OC', 'PM10', 'PM2.5_bio', 'PM2.5_fossil', 'SO2', 'NMVOC']


def get_default_voc_pollutant_list():
    """
    Get the default EDGARv4.3.2 VOC pollutant list.

    Returns
    -------
    list[str]
        Default VOC pollutant names.
    """
    return [f'voc{x}' for x in range(1, 26)]


# Unified AP + VOC mapping function
def get_default_ipcc_to_sector_mapping():
    """
    Get the default combined IPCC to sector mapping for EDGAR (AP + VOC).

    Notes
    -----
    This mapping merges AP and VOC sector aggregations. Keys are mostly
    disjoint, so they can coexist safely. Sections are kept separated by
    comments for clarity.

    Returns
    -------
    dict
        Mapping from EDGAR IPCC categories to HERMES_GR sector names.
    """
    return {
        # ---------------- AP ----------------
        "IPCC_1A1a": "ENE",
        "IPCC_1A1b_1A1c_1A5b1_1B1b_1B2a5_1B2a6_1B2b5_2C1b": "REF_TRF",
        "IPCC_1A2": "IND",
        "IPCC_1A3a_CDS": "TNR_Aviation_CDS",
        "IPCC_1A3a_CRS": "TNR_Aviation_CRS",
        "IPCC_1A3a_LTO": "TNR_Aviation_LTO",
        "IPCC_1A3b": "TRO",
        "IPCC_1A3c_1A3e": "TNR_Other",
        "IPCC_1A3d_1C2": "TNR_Ship",
        "IPCC_1A4": "RCO",
        "IPCC_1B1a_1B2a1_1B2a2_1B2a3_1B2a4_1B2c": "PRO",
        "IPCC_2A": "NMM",
        "IPCC_2B": "CHE",
        "IPCC_2C1a_2C1c_2C1d_2C1e_2C1f_2C2": "IRO",
        "IPCC_2C3_2C4_2C5": "NFE",
        "IPCC_2D": "FOO_PAP",
        "IPCC_2G": "NEU",
        "IPCC_3": "PRU_SOL",
        "IPCC_4B": "MNM",
        "IPCC_4C_4D1_4D2_4D4": "AGS",
        "IPCC_4F": "AWB",
        "IPCC_6A_6D": "SWD_LDF",
        "IPCC_6B": "WWT",
        "IPCC_6C": "SWD_INC",
        "IPCC_7A": "FFF",

        # ---------------- VOC ----------------
        "IPCC_1A1": "ENE",
        "IPCC_2_3": "PPA",
        "IPCC_1A1b_1B2a5": "REF",
        "IPCC_6": "SWD",
        "IPCC_1A1c_1A5b1_1B1b_1B2a6_1B2b5_2C1b": "TRF",
    }


def obtain_lats_lons(nc):
    """
    Extract latitude and longitude centers and increments from an NES object.

    Parameters
    ----------
    nc : nes.Nes
        NES object containing latitude and longitude coordinates.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, float, float]
        Latitude centers, longitude centers, latitude increment, and longitude
        increment.
    """
    lats = nc.lat["data"]
    lons = nc.lon["data"]

    lats_interval = lats[1:] - lats[:-1]
    min_lats_interval = lats_interval.min()

    lons_interval = lons[1:] - lons[:-1]
    min_lons_interval = lons_interval.min()

    return lats, lons, min_lats_interval, min_lons_interval


def obtain_data(nc, pollutant):
    """
    Load and retrieve the emission field for one pollutant.

    Parameters
    ----------
    nc : nes.Nes
        Input EDGAR NetCDF opened as NES object.
    pollutant : str
        Pollutant name in the EDGAR naming convention.

    Returns
    -------
    numpy.ndarray
        Emission field for the selected pollutant.
    """
    print("variables", nc.variables)
    nc.load(f"emi_{pollutant.lower()}")
    data = nc.variables[f"emi_{pollutant.lower()}"]["data"]
    return np.array(data)


def normalize_pollutant_name(pollutant):
    """
    Convert an EDGAR pollutant name to the HERMES_GR output naming convention.

    The normalization is inferred directly from the pollutant name, without
    requiring an explicit inventory type.

    Parameters
    ----------
    pollutant : str
        Pollutant name in the EDGAR naming convention.

    Returns
    -------
    str
        Pollutant name in the HERMES_GR output naming convention.
    """
    if pollutant == 'NOx':
        return 'nox_no2'
    elif pollutant == 'PM2.5_bio':
        return 'pm25_bio'
    elif pollutant == 'PM2.5_fossil':
        return 'pm25_fossil'
    elif pollutant in [f'voc{x}' for x in range(1, 10)]:
        return pollutant.replace('voc', 'voc0').lower()
    return pollutant.lower()


def normalize_sector_list(sector_list):
    """
    Normalize sector names to lowercase for filtering.

    Parameters
    ----------
    sector_list : list[str] | None
        Sector names provided by the user.

    Returns
    -------
    list[str] | None
        Normalized sector names, or ``None`` if no filtering is requested.
    """
    if sector_list is None:
        return None
    return [sector.lower() for sector in sector_list]


def infer_default_pollutant_list(original_file):
    """
    Infer the default EDGAR pollutant list from the input file template.

    Parameters
    ----------
    original_file : str
        Path template to the original EDGAR files.

    Returns
    -------
    list[str]
        Default pollutant list inferred from the input template.
    """
    if 'VOC_spec' in original_file:
        return get_default_voc_pollutant_list()
    return get_default_ap_pollutant_list()


def write_output_file(
    data,
    lats,
    lons,
    min_lats_interval,
    min_lons_interval,
    output_file,
    output_pollutant,
    description,
    global_attributes,
    time_date,
    overwrite=False,
):
    """
    Write one EDGAR output file in HERMES_GR format.

    Existing files are skipped unless ``overwrite=True``.

    Parameters
    ----------
    data : numpy.ndarray
        Emission field to write.
    lats : numpy.ndarray
        Latitude centers.
    lons : numpy.ndarray
        Longitude centers.
    min_lats_interval : float
        Latitude increment.
    min_lons_interval : float
        Longitude increment.
    output_file : str
        Output NetCDF file path.
    output_pollutant : str
        Pollutant name in the HERMES_GR output naming convention.
    description : str
        Variable description for the output file.
    global_attributes : dict
        Global attributes copied from the source file.
    time_date : datetime.datetime
        Timestamp to assign to the output file.
    overwrite : bool, optional
        Whether to overwrite existing output files.

    Returns
    -------
    None
    """
    if os.path.exists(output_file) and not overwrite:
        logger.info(f"Skipping existing file: {output_file}")
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    nessy = nes.create_nes(
        comm=None,
        info=False,
        projection="regular",
        lat_orig=lats[0],
        lon_orig=lons[0],
        inc_lat=min_lats_interval,
        inc_lon=min_lons_interval,
        n_lat=len(lats),
        n_lon=len(lons),
        times=[time_date],
    )
    nessy.global_attrs = global_attributes
    print("calculating grid area")
    nessy.calculate_grid_area()
    print("Grid area calculated")
    nessy.cell_measures["cell_area"]["units"] = "m2"
    nessy.variables[output_pollutant] = {
        'data': data,
        'units': 'kg.m-2.s-1',
        'description': description,
        'short_name': output_pollutant,
    }
    nessy.to_netcdf(output_file)


def do_yearly_transformation(
    original_file,
    output_path,
    year,
    pollutant_list,
    ipcc_to_sector_mapping,
    description_prefix,
    overwrite=False,
    sector_list=None,
):
    """
    Generate yearly EDGAR outputs.

    Parameters
    ----------
    original_file : str
        Path template to the original EDGAR files.
    output_path : str
        Root directory where the preprocessed files will be written.
    year : int
        Year to process.
    pollutant_list : list[str]
        Pollutants to process.
    ipcc_to_sector_mapping : dict
        Mapping from IPCC categories to output sector names.
    description_prefix : str
        Prefix used in the output variable descriptions.
    overwrite : bool, optional
        Whether to overwrite existing output files.
    sector_list : list[str], optional
        Sector filter provided by the user.

    Returns
    -------
    None
    """
    selected_sectors = normalize_sector_list(sector_list)

    for pollutant in pollutant_list:
        for ipcc, sector in ipcc_to_sector_mapping.items():
            if selected_sectors is not None and sector.lower() not in selected_sectors:
                continue

            input_file = (
                original_file
                .replace('<pollutant>', pollutant)
                .replace('<year>', str(year))
                .replace('<ipcc>', ipcc)
            )

            if not os.path.exists(input_file):
                warning(
                    f"The pollutant {pollutant} for the IPCC sector {ipcc} does not exist.\n"
                    f"File not found: {input_file}"
                )
                continue

            nessy_in = nes.open_netcdf(input_file, time_exception=True)
            lats, lons, min_lats_interval, min_lons_interval = obtain_lats_lons(nessy_in)

            if pollutant in ['PM2.5_bio', 'PM2.5_fossil']:
                pollutant_in = 'PM2.5'
            else:
                pollutant_in = pollutant

            data = obtain_data(nessy_in, pollutant_in)
            global_attributes = nessy_in.global_attrs
            output_pollutant = normalize_pollutant_name(pollutant)
            output_file = os.path.join(
                output_path,
                'yearly_mean',
                f'{output_pollutant}_{sector.lower()}',
                f'{output_pollutant}_{year}.nc',
            )
            description = (
                f'{description_prefix} annual emissions for {output_pollutant} '
                f'sector {sector} for year {year}'
            )
            write_output_file(
                data=data,
                lats=lats,
                lons=lons,
                min_lats_interval=min_lats_interval,
                min_lons_interval=min_lons_interval,
                output_file=output_file,
                output_pollutant=output_pollutant,
                description=description,
                global_attributes=global_attributes,
                time_date=datetime(year, month=1, day=1),
                overwrite=overwrite,
            )


def do_monthly_transformation(
    original_file,
    monthly_temporal_profiles,
    output_path,
    year,
    pollutant_list,
    ipcc_to_sector_mapping,
    description_prefix,
    overwrite=False,
    sector_list=None,
):
    """
    Generate monthly EDGAR outputs for years other than 2010 using monthly factors.

    Parameters
    ----------
    original_file : str
        Path template to the original yearly EDGAR files.
    monthly_temporal_profiles : str
        Path template to the monthly temporal profile files.
    output_path : str
        Root directory where the preprocessed files will be written.
    year : int
        Year to process.
    pollutant_list : list[str]
        Pollutants to process.
    ipcc_to_sector_mapping : dict
        Mapping from IPCC categories to output sector names.
    description_prefix : str
        Prefix used in the output variable descriptions.
    overwrite : bool, optional
        Whether to overwrite existing output files.
    sector_list : list[str], optional
        Sector filter provided by the user.

    Returns
    -------
    None
    """
    selected_sectors = normalize_sector_list(sector_list)

    for pollutant in pollutant_list:
        for ipcc, sector in ipcc_to_sector_mapping.items():
            if selected_sectors is not None and sector.lower() not in selected_sectors:
                continue

            input_file = (
                original_file
                .replace('<pollutant>', pollutant)
                .replace('<year>', str(year))
                .replace('<ipcc>', ipcc)
            )

            if not os.path.exists(input_file):
                warning(
                    f"The pollutant {pollutant} for the IPCC sector {ipcc} does not exist.\n"
                    f"File not found: {input_file}"
                )
                continue

            nessy_in = nes.open_netcdf(input_file, time_exception=True)
            lats, lons, min_lats_interval, min_lons_interval = obtain_lats_lons(nessy_in)

            if pollutant in ['PM2.5_bio', 'PM2.5_fossil']:
                pollutant_in = 'PM2.5'
            else:
                pollutant_in = pollutant

            data = obtain_data(nessy_in, pollutant_in)
            global_attributes = nessy_in.global_attrs
            output_pollutant = normalize_pollutant_name(pollutant)

            month_factor_file = monthly_temporal_profiles.replace('<sector>', sector)
            nc_month_factors = nes.open_netcdf(month_factor_file, time_exception=True)
            nc_month_factors.set_time([datetime(year, month=i, day=1) for i in range(1, 13)])
            nc_month_factors.read_axis_limits = nc_month_factors._get_read_axis_limits()
            nc_month_factors.write_axis_limits = nc_month_factors._get_write_axis_limits()
            nc_month_factors.load(sector)
            month_factors = nc_month_factors.variables[sector]["data"]

            for month in range(1, 13):
                data_aux = data * month_factors[month - 1, :, :]
                output_file = os.path.join(
                    output_path,
                    'monthly_mean',
                    f'{output_pollutant}_{sector.lower()}',
                    f'{output_pollutant}_{year}{month:02d}.nc',
                )
                description = (
                    f'{description_prefix} monthly emissions for {output_pollutant} '
                    f'sector {sector} for year {year} and month {month}'
                )
                write_output_file(
                    data=data_aux,
                    lats=lats,
                    lons=lons,
                    min_lats_interval=min_lats_interval,
                    min_lons_interval=min_lons_interval,
                    output_file=output_file,
                    output_pollutant=output_pollutant,
                    description=description,
                    global_attributes=global_attributes,
                    time_date=datetime(year, month=month, day=1),
                    overwrite=overwrite,
                )


def do_2010_monthly_transformation(
    original_file,
    output_path,
    year,
    pollutant_list,
    ipcc_to_sector_mapping,
    description_prefix,
    overwrite=False,
    sector_list=None,
):
    """
    Generate monthly EDGAR outputs for 2010 directly from the monthly input files.

    Parameters
    ----------
    original_file : str
        Path template to the original monthly EDGAR files.
    output_path : str
        Root directory where the preprocessed files will be written.
    year : int
        Year to process.
    pollutant_list : list[str]
        Pollutants to process.
    ipcc_to_sector_mapping : dict
        Mapping from IPCC categories to output sector names.
    description_prefix : str
        Prefix used in the output variable descriptions.
    overwrite : bool, optional
        Whether to overwrite existing output files.
    sector_list : list[str], optional
        Sector filter provided by the user.

    Returns
    -------
    None
    """
    selected_sectors = normalize_sector_list(sector_list)

    for pollutant in pollutant_list:
        for ipcc, sector in ipcc_to_sector_mapping.items():
            if selected_sectors is not None and sector.lower() not in selected_sectors:
                continue

            for month in range(1, 13):
                input_file = (
                    original_file
                    .replace('<pollutant>', pollutant)
                    .replace('<month>', str(month))
                    .replace('<ipcc>', ipcc)
                )

                if not os.path.exists(input_file):
                    warning(
                        f"The pollutant {pollutant} for the IPCC sector {ipcc} does not exist.\n"
                        f"File not found: {input_file}"
                    )
                    continue

                nessy_in = nes.open_netcdf(input_file, time_exception=True)
                lats, lons, min_lats_interval, min_lons_interval = obtain_lats_lons(nessy_in)

                if pollutant in ['PM2.5_bio', 'PM2.5_fossil']:
                    pollutant_in = 'PM2.5'
                else:
                    pollutant_in = pollutant

                data = obtain_data(nessy_in, pollutant_in)
                global_attributes = nessy_in.global_attrs
                output_pollutant = normalize_pollutant_name(pollutant)
                output_file = os.path.join(
                    output_path,
                    'monthly_mean',
                    f'{output_pollutant}_{sector.lower()}',
                    f'{output_pollutant}_{year}{month:02d}.nc',
                )
                description = (
                    f'{description_prefix} monthly emissions for {output_pollutant} '
                    f'sector {sector} for year {year} and month {month}'
                )
                write_output_file(
                    data=data,
                    lats=lats,
                    lons=lons,
                    min_lats_interval=min_lats_interval,
                    min_lons_interval=min_lons_interval,
                    output_file=output_file,
                    output_pollutant=output_pollutant,
                    description=description,
                    global_attributes=global_attributes,
                    time_date=datetime(year, month=month, day=1),
                    overwrite=overwrite,
                )


def edgar_preproc(
    original_file,
    output_path,
    year,
    pollutant_list=None,
    sector_list=None,
    ipcc_to_sector_mapping=None,
    monthly_temporal_profiles=None,
    yearly=False,
    monthly=False,
    overwrite=False,
):
    """
    Convert EDGARv4.3.2 annual or monthly emissions into HERMES-ready outputs.

    Exactly one processing mode must be selected: yearly or monthly. When
    ``monthly=True`` and ``year != 2010``, monthly outputs are derived from the
    yearly emissions using monthly temporal profiles.

    Parameters
    ----------
    original_file : str
        Path template to the original EDGAR files.
    output_path : str
        Root directory where the preprocessed files will be written.
    year : int
        Year to process.
    pollutant_list : list[str], optional
        Pollutants to process. If ``None``, the default list is used.
    sector_list : list[str], optional
        Output sectors to process. If ``None``, all sectors are processed.
    ipcc_to_sector_mapping : dict, optional
        Mapping from IPCC categories to output sector names. If ``None``, the
        default mapping is used.
    monthly_temporal_profiles : str, optional
        Path template to the monthly temporal profile files. This is only used
        when ``monthly=True`` and ``year != 2010``.
    yearly : bool, optional
        Whether to process yearly outputs. Exactly one of ``yearly`` or
        ``monthly`` must be ``True``.
    monthly : bool, optional
        Whether to process monthly outputs. Exactly one of ``yearly`` or
        ``monthly`` must be ``True``.
    overwrite : bool, optional
        Whether to overwrite existing output files. If ``False``, existing
        output files are skipped.

    Returns
    -------
    None
    """
    if yearly == monthly:
        raise ValueError(
            'Exactly one processing mode must be selected: '
            'yearly=True, monthly=False or yearly=False, monthly=True.'
        )

    if pollutant_list is None:
        if 'VOC_spec' in original_file:
            pollutant_list = get_default_voc_pollutant_list()
        else:
            pollutant_list = get_default_ap_pollutant_list()
    if ipcc_to_sector_mapping is None:
        ipcc_to_sector_mapping = get_default_ipcc_to_sector_mapping()

    description_prefix = 'EDGARv4.3.2'

    if monthly and year != 2010 and monthly_temporal_profiles is None:
        raise ValueError(
            'monthly_temporal_profiles must be provided when '
            'monthly=True and year != 2010.'
        )

    if yearly:
        do_yearly_transformation(
            original_file=original_file,
            output_path=output_path,
            year=year,
            pollutant_list=pollutant_list,
            ipcc_to_sector_mapping=ipcc_to_sector_mapping,
            description_prefix=description_prefix,
            overwrite=overwrite,
            sector_list=sector_list,
        )
    elif year == 2010:
        do_2010_monthly_transformation(
            original_file=original_file,
            output_path=output_path,
            year=year,
            pollutant_list=pollutant_list,
            ipcc_to_sector_mapping=ipcc_to_sector_mapping,
            description_prefix=description_prefix,
            overwrite=overwrite,
            sector_list=sector_list,
        )
    else:
        do_monthly_transformation(
            original_file=original_file,
            monthly_temporal_profiles=monthly_temporal_profiles,
            output_path=output_path,
            year=year,
            pollutant_list=pollutant_list,
            ipcc_to_sector_mapping=ipcc_to_sector_mapping,
            description_prefix=description_prefix,
            overwrite=overwrite,
            sector_list=sector_list,
        )

    return None


# Example standalone execution using the module-level testing parameters
# defined at the top of this script.
if __name__ == '__main__':
    edgar_preproc(
        original_file=ORIGINAL_FILE,
        output_path=OUTPUT_PATH,
        year=YEAR,
        pollutant_list=None,
        sector_list=None,
        ipcc_to_sector_mapping=None,
        monthly_temporal_profiles=MONTHLY_TEMPORAL_PROFILES,
        yearly=YEARLY,
        monthly=MONTHLY,
        overwrite=False,
    )
