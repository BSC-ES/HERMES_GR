#!/usr/bin/env python

# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.


import os
import nes
import pandas as pd
import numpy as np
from datetime import datetime
from .cli_logger import get_cli_logger

logger = get_cli_logger("cams_reg_ap")

# ============== README ======================
"""
Download access: contact hugo.deniervandergon@tno.nl or jeroen.kuenen@tno.nl.
Reference: https://www.atmos-chem-phys.net/14/10963/2014/
Besides citing HERMES_GR, users must also acknowledge the use of the
corresponding emission inventories in their works.
"""

# ============== CONFIGURATION PARAMETERS ======================
ORIGINAL_FILE = "/esarchive/recon/ecmwf/cams_reg_apv81/original_files/csv/CAMS-REG-v8_1_emissions_<year>.csv"

OUTPUT_PATH = "/esarchive/recon/ecmwf/cams_reg_apv81/yearly_mean"

YEAR = 2024

# Additional test configuration constants
WORLD_INFO_PATH = ('/gpfs/projects/bsc32/models/HERMES/HERMES_data/Benchmarks/HERMES_GR_Benchmark/preproc/'
                   'tz_world_country_iso3166.csv')
WORLD_MASK_PATH = ("/gpfs/projects/bsc32/models/HERMES/HERMES_data/Benchmarks/HERMES_GR_Benchmark/preproc/"
                   "CAMS-REG-AP_v2_WorldMask.nc")
VOC_RATIO_CSV_PATH = ("/gpfs/projects/bsc32/models/HERMES/HERMES_data/Benchmarks/HERMES_GR_Benchmark/preproc/"
                      "NMVOC_split_for_CAMS-REG-v81_melt.csv")
PM_RATIO_CSV_PATH = ("/gpfs/projects/bsc32/models/HERMES/HERMES_data/Benchmarks/HERMES_GR_Benchmark/preproc/"
                     "PM_split_for_CAMS-REG-v81_2023_2024.csv")

# ==============================================================


def calculate_grid_definition(in_path):
    """
    Calculate the regular grid definition from the CAMS-REG-AP input table.

    Parameters
    ----------
    in_path : str
        Path to the CAMS-REG-AP yearly CSV file.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, float, float]
        Latitude centers, longitude centers, latitude increment, and longitude
        increment.
    """
    dataframe = pd.read_table(in_path, sep=';')

    # Longitudes
    lons = np.sort(np.unique(dataframe.Lon))
    lons_interval = lons[1:] - lons[:-1]

    # Latitudes
    lats = np.sort(np.unique(dataframe.Lat))
    lats_interval = lats[1:] - lats[:-1]

    lats = np.arange(-90 + lats_interval.min() / 2, 90, lats_interval.min(), dtype=np.float64)
    lons = np.arange(-180 + lons_interval.min() / 2, 180, lons_interval.min(), dtype=np.float64)

    return lats, lons, lats_interval.min(), lons_interval.min()


def get_pollutants(in_path):
    """
    Get the pollutant columns available in the CAMS-REG-AP input table.

    Parameters
    ----------
    in_path : str
        Path to the CAMS-REG-AP yearly CSV file.

    Returns
    -------
    list[str]
        Pollutant column names present in the input table.
    """
    columns = list(pd.read_table(in_path, sep=';', nrows=1).columns)

    return columns[8:]


def create_pollutant_empty_list(in_path, len_c_lats, len_c_lons):
    """
    Create an empty pollutant definition list for the target grid.

    Parameters
    ----------
    in_path : str
        Path to the CAMS-REG-AP yearly CSV file.
    len_c_lats : int
        Number of latitude cells.
    len_c_lons : int
        Number of longitude cells.

    Returns
    -------
    list[dict]
        Pollutant definitions including output name, input column name, units,
        and an empty data array.
    """
    pollutant_list = []
    for pollutant in get_pollutants(in_path):
        aux_dict = {}
        if pollutant == 'PM2_5':
            aux_dict['name'] = 'pm25'
        elif pollutant == 'NOX':
            aux_dict['name'] = 'nox_no2'
        else:
            aux_dict['name'] = pollutant.lower()
        aux_dict['TNO_name'] = pollutant
        aux_dict['units'] = 'kg.m-2.s-1'
        # aux_dict['units'] = 'Mg.km-2.year-1'
        aux_dict['data'] = np.zeros((len_c_lats, len_c_lons))
        pollutant_list.append(aux_dict)
    return pollutant_list


def get_sector_list():
    """
    Get the list of supported GNFR sectors.

    Returns
    -------
    list[str]
        GNFR sector names in the output naming convention.
    """
    return ['gnfr_{0}'.format(x) for x in
            ['A', 'B', 'C', 'D', 'E', 'F1', 'F2', 'F3', 'F4', 'G', 'H', 'I', 'J', 'K', 'L']]


def get_voc_list(voc_ratio_csv):
    """
    Get the list of VOC split species from the ratio CSV file.

    Parameters
    ----------
    voc_ratio_csv : str
        Path to the VOC ratio CSV file.

    Returns
    -------
    list[str]
        VOC species names in the HERMES_GR naming convention.
    """
    df = pd.read_csv(voc_ratio_csv, sep=',')
    del df['year'], df['ISO3'], df['gnfr'], df['fr']
    df = df.drop_duplicates().dropna()
    voc_list = df.vcode.values
    for index, voc in enumerate(voc_list):
        voc_list[index] = voc.replace('v', 'voc')
    return list(voc_list)


def get_pm_list(pm_ratio_csv):
    """
    Get the list of PM split species from the ratio CSV file.

    Parameters
    ----------
    pm_ratio_csv : str
        Path to the PM ratio CSV file.

    Returns
    -------
    list[str]
        PM split species names.
    """
    df = pd.read_csv(pm_ratio_csv, sep=',')
    del df['ISO3'], df['gnfr'], df['fr'], df['year']
    df = df.drop_duplicates().dropna()
    return df.pmcode.to_list()


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
    lat_data = nc.lat["data"]
    lon_data = nc.lon["data"]

    if len(lat_data.shape) == 2:
        lats = lat_data[:, 0]
    else:
        lats = lat_data

    if len(lon_data.shape) == 2:
        lons = lon_data[0, :]
    else:
        lons = lon_data

    lats_interval = lats[1:] - lats[:-1]
    min_lats_interval = lats_interval.min()

    lons_interval = lons[1:] - lons[:-1]
    min_lons_interval = lons_interval.min()

    return lats, lons, min_lats_interval, min_lons_interval


# === Helper functions for ratio, sector, and country handling ===

def normalize_sector_list(sector_list):
    """
    Normalize sector names to the ``gnfr_X`` output naming convention.

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

    normalized_sector_list = []
    for sector in sector_list:
        if sector.lower().startswith('gnfr_'):
            normalized_sector_list.append(sector.lower())
        else:
            normalized_sector_list.append(f'gnfr_{sector}')

    return normalized_sector_list


def get_iso_codes(world_info_path):
    """
    Read the mapping between ISO alpha codes and world mask numeric codes.

    Parameters
    ----------
    world_info_path : str
        Path to the CSV file containing country code mappings.

    Returns
    -------
    dict
        Mapping from ISO alpha codes to world mask numeric identifiers.
    """
    df = pd.read_csv(world_info_path, sep=';')
    del df['time_zone'], df['time_zone_code']
    df = df.drop_duplicates().dropna()
    df = df.set_index('country_code_alpha')
    codes_dict = df.to_dict()
    return codes_dict['country_code']


def get_available_ratio_year(year, csv_path):
    """
    Get the ratio year to use for the requested processing year.

    If the requested year is not available in the ratio CSV file, the latest
    available year is used instead.

    Parameters
    ----------
    year : int
        Requested processing year.
    csv_path : str
        Path to the ratio CSV file.

    Returns
    -------
    int
        Year available in the ratio CSV file.
    """
    df = pd.read_csv(csv_path, sep=',')
    if year in df.year.values:
        return year

    available_year = df.year.values.max()
    logger.info(f'Year {year} not available in the dataframe at {csv_path}, '
                f'using the latest year available: {available_year}')
    return available_year


def get_ratio_sector_list(csv_path, species_column, species_name):
    """
    Get the GNFR sectors available for one split species.

    Parameters
    ----------
    csv_path : str
        Path to the ratio CSV file.
    species_column : str
        Column containing the split species code.
    species_name : str
        Split species identifier to filter.

    Returns
    -------
    list[str]
        GNFR sector names available for the selected species.
    """
    df = pd.read_csv(csv_path, sep=',')
    # df = df[df.gnfr != 'G']
    df = df[df[species_column] == species_name]
    del df['ISO3'], df[species_column], df['year'], df['fr']
    df = df.drop_duplicates().dropna()
    return ['gnfr_{0}'.format(sector) for sector in df.gnfr.values]


def get_country_code_and_factor(csv_path, species_column, species_name, gnfr, year):
    """
    Get country-specific split factors for one species and one GNFR sector.

    Parameters
    ----------
    csv_path : str
        Path to the ratio CSV file.
    species_column : str
        Column containing the split species code.
    species_name : str
        Split species identifier to filter.
    gnfr : str
        GNFR sector name in the output naming convention.
    year : int
        Requested processing year.

    Returns
    -------
    dict
        Mapping from ISO alpha codes to split factors.
    """
    df = pd.read_csv(csv_path, sep=',')
    available_year = get_available_ratio_year(year, csv_path)
    df = df[df.year == available_year]
    df = df[df[species_column] == species_name]
    df = df[df.gnfr == gnfr.replace('gnfr_', '').upper()]
    del df['gnfr'], df[species_column], df['year']
    df = df.drop_duplicates().dropna()
    df = df.set_index('ISO3')
    country_dict = df.to_dict()
    return country_dict['fr']


def get_default_ratio(csv_path, species_column, species_name, gnfr, year):
    """
    Get the default EUR split factor for one species and one GNFR sector.

    Parameters
    ----------
    csv_path : str
        Path to the ratio CSV file.
    species_column : str
        Column containing the split species code.
    species_name : str
        Split species identifier to filter.
    gnfr : str
        GNFR sector name in the output naming convention.
    year : int
        Requested processing year.

    Returns
    -------
    float
        Default EUR split factor.
    """
    df = pd.read_csv(csv_path, sep=',')
    available_year = get_available_ratio_year(year, csv_path)
    df = df[df.year == available_year]
    df = df.loc[df[species_column] == species_name, :]
    df = df.loc[df['gnfr'] == gnfr.replace('gnfr_', '').upper(), :]
    return df.loc[df['ISO3'] == 'EUR', 'fr'].item()


def build_ratio_mask(world_mask_path, world_info_path, csv_path, species_column, species_name, gnfr, year):
    """
    Build the spatial split mask for one species and one GNFR sector.

    Parameters
    ----------
    world_mask_path : str
        Path to the CAMS world mask NetCDF file.
    world_info_path : str
        Path to the CSV file containing country code mappings.
    csv_path : str
        Path to the ratio CSV file.
    species_column : str
        Column containing the split species code.
    species_name : str
        Split species identifier to filter.
    gnfr : str
        GNFR sector name in the output naming convention.
    year : int
        Requested processing year.

    Returns
    -------
    numpy.ndarray
        Two-dimensional spatial mask with the split factor per grid cell.
    """
    nc_world_mask = nes.open_netcdf(world_mask_path)
    nc_world_mask.load("timezone_id")
    country_values = nc_world_mask.variables["timezone_id"]["data"]
    country_values = country_values.reshape((country_values.shape[2], country_values.shape[3]))

    mask_factor = np.zeros(country_values.shape)
    iso_codes = get_iso_codes(world_info_path)
    for country_code, factor in get_country_code_and_factor(csv_path, species_column, species_name, gnfr, year).items():
        try:
            mask_factor[country_values == iso_codes[country_code]] = factor
        except Exception:
            pass

    mask_factor[mask_factor <= 0] = get_default_ratio(csv_path, species_column, species_name, gnfr, year)
    return mask_factor


def filter_base_pollutants(in_file, pollutant_list, voc_ratio_csv, pm_ratio_csv):
    """
    Determine which base pollutants must be generated from the yearly table.

    If a split VOC or PM species is requested, the corresponding base pollutant
    (``nmvoc`` or ``pm25``) is automatically included.

    Parameters
    ----------
    in_file : str
        Path to the CAMS-REG-AP yearly CSV file.
    pollutant_list : list[str] | None
        Pollutant filter provided by the user.
    voc_ratio_csv : str | None
        Path to the VOC ratio CSV file.
    pm_ratio_csv : str | None
        Path to the PM ratio CSV file.

    Returns
    -------
    list[str]
        Base pollutant names that must be generated.
    """
    pollutant_definitions = create_pollutant_empty_list(in_file, 1, 1)
    available_base_pollutants = [pollutant['name'] for pollutant in pollutant_definitions]

    if pollutant_list is None:
        return available_base_pollutants

    selected_pollutants = set(pollutant_list)
    selected_base_pollutants = [
        pollutant for pollutant in available_base_pollutants if pollutant in selected_pollutants
    ]

    if voc_ratio_csv is not None:
        voc_list = set(get_voc_list(voc_ratio_csv))
        if 'nmvoc' in selected_pollutants or len(selected_pollutants.intersection(voc_list)) > 0:
            selected_base_pollutants.append('nmvoc')

    if pm_ratio_csv is not None:
        pm_list = set(get_pm_list(pm_ratio_csv))
        if 'pm25' in selected_pollutants or len(selected_pollutants.intersection(pm_list)) > 0:
            selected_base_pollutants.append('pm25')

    return sorted(set(selected_base_pollutants))


def do_transformation(original_file, output_path, year, pollutant_list=None, sector_list=None, overwrite=False,
                      voc_ratio_csv=None, pm_ratio_csv=None):
    """
    Generate the base yearly pollutant files from the CAMS-REG-AP CSV table.

    Parameters
    ----------
    original_file : str
        Path to the CAMS-REG-AP yearly CSV file.
    output_path : str
        Root directory where the preprocessed yearly files will be written.
    year : int
        Year to process.
    pollutant_list : list[str], optional
        Pollutants to process. If ``None``, all available base pollutants are
        processed. If a VOC or PM split species is requested, the corresponding
        base pollutant (``nmvoc`` or ``pm25``) is also generated internally.
    sector_list : list[str], optional
        GNFR sectors to process. If ``None``, all supported sectors are
        processed.
    overwrite : bool, optional
        Whether to overwrite existing output files. If ``False``, existing
        output files are skipped.
    voc_ratio_csv : str, optional
        Path to the VOC ratio CSV file. Needed only to infer whether ``nmvoc``
        must be generated when split VOC species are requested.
    pm_ratio_csv : str, optional
        Path to the PM ratio CSV file. Needed only to infer whether ``pm25``
        must be generated when split PM species are requested.

    Returns
    -------
    bool
        ``True`` when the transformation finishes.
    """
    unit_factor = 1.0 / (365.0 * 24.0 * 3600.0)

    if not os.path.exists(original_file):
        logger.info(f"File not found: {original_file}")
        return True

    logger.info(f"Processing year {year} using input file: {original_file}")
    lats, lons, min_lats_interval, min_lons_interval = calculate_grid_definition(original_file)

    nessy_default = nes.create_nes(
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
    nessy_default.create_spatial_bounds()
    logger.info("Calculating cell area...")
    cell_area = nessy_default.calculate_grid_area()
    logger.info("Cell area calculated.")

    dataframe = pd.read_table(original_file, sep=';')
    dataframe.loc[:, 'row_lat'] = np.array(
        (dataframe.Lat - (-90 + min_lats_interval / 2)) / min_lats_interval,
        dtype=np.int32,
    )
    dataframe.loc[:, 'col_lon'] = np.array(
        (dataframe.Lon - (-180 + min_lons_interval / 2)) / min_lons_interval,
        dtype=np.int32,
    )

    selected_sectors = normalize_sector_list(sector_list)
    selected_base_pollutants = filter_base_pollutants(original_file, pollutant_list, voc_ratio_csv, pm_ratio_csv)

    for name, group in dataframe.groupby('GNFR_Category'):
        output_sector_name = f'gnfr_{name}'
        if selected_sectors is not None and output_sector_name.lower() not in selected_sectors:
            continue

        pollutant_list_year = create_pollutant_empty_list(original_file, len(lats), len(lons))
        pollutant_list_year = [
            pollutant for pollutant in pollutant_list_year if pollutant['name'] in selected_base_pollutants
        ]
        if len(pollutant_list_year) == 0:
            continue

        group = group.groupby(['row_lat', 'col_lon']).sum().reset_index()

        for pollutant_dict in pollutant_list_year:
            aux_output_dir = os.path.join(output_path, f"{pollutant_dict['name']}_{output_sector_name}")
            aux_output_file = os.path.join(aux_output_dir, f"{pollutant_dict['name']}_{year}.nc")
            if os.path.exists(aux_output_file) and not overwrite:
                logger.info(f"Skipping existing file: {aux_output_file}")
                continue

            pollutant_data = np.zeros((len(lats), len(lons)))
            pollutant_data[group.row_lat, group.col_lon] += group[pollutant_dict['TNO_name']]
            pollutant_data = pollutant_data.reshape((1,) + pollutant_data.shape)
            pollutant_data = pollutant_data * unit_factor / cell_area

            nessy = nes.create_nes(
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
            nessy.create_spatial_bounds()
            nessy.cell_measures["cell_area"] = {"data": cell_area, "units": "m2"}
            nessy.variables[pollutant_dict["name"]] = {
                "data": pollutant_data.reshape((1, 1) + cell_area.shape),
                "units": pollutant_dict["units"],
                "short_name": pollutant_dict["name"],
            }
            nessy.global_attrs = {
                'references': 'J. J. P. Kuenen, A. J. H. Visschedijk, M. Jozwicka, and H. A. C. '
                              'Denier van der Gon TNO-MACC_II emission inventory; a multi-year '
                              '(2003-2009) consistent high-resolution European emission inventory '
                              'for air quality modelling Atmospheric Chemistry and Physics 14 '
                              '10963-10976 2014',
                'comment': 'Re-writing done by Carles Tena (carles.tena@bsc.es) from the BSC-CNS '
                           '(Barcelona Supercomputing Center)',
            }

            os.makedirs(aux_output_dir, exist_ok=True)
            nessy.to_netcdf(aux_output_file)

    return True


def do_voc_transformation(output_path, year, world_info_path, world_mask_path, voc_ratio_csv,
                          pollutant_list=None, sector_list=None, overwrite=False):
    """
    Generate the VOC split pollutant files from the base ``nmvoc`` outputs.

    Parameters
    ----------
    output_path : str
        Root directory containing the base and speciated yearly outputs.
    year : int
        Year to process.
    world_info_path : str
        Path to the CSV file containing the mapping between ISO alpha codes and
        the world mask codes.
    world_mask_path : str
        Path to the CAMS world mask NetCDF file.
    voc_ratio_csv : str
        Path to the CSV file containing the VOC split factors.
    pollutant_list : list[str], optional
        Pollutants to process. If ``None``, all VOC split species available in
        the ratio CSV are processed.
    sector_list : list[str], optional
        GNFR sectors to process. If ``None``, all supported sectors are
        processed.
    overwrite : bool, optional
        Whether to overwrite existing output files. If ``False``, existing
        output files are skipped.

    Returns
    -------
    bool
        ``True`` when the transformation finishes.
    """
    available_voc_list = get_voc_list(voc_ratio_csv)
    selected_voc_list = available_voc_list if pollutant_list is None else [
        voc for voc in available_voc_list if voc in pollutant_list
    ]
    selected_sectors = normalize_sector_list(sector_list)

    for gnfr in get_sector_list():
        if selected_sectors is not None and gnfr.lower() not in selected_sectors:
            continue

        in_path_voc = os.path.join(output_path, f'nmvoc_{gnfr}', f'nmvoc_{year}.nc')
        if not os.path.exists(in_path_voc):
            continue

        nessy_in_voc = nes.open_netcdf(in_path_voc)
        lats, lons, min_lats_interval, min_lons_interval = obtain_lats_lons(nessy_in_voc)
        nessy_in_voc.load("nmvoc")
        nmvoc_data = nessy_in_voc.variables["nmvoc"]["data"]

        for voc in selected_voc_list:
            if gnfr not in get_ratio_sector_list(voc_ratio_csv, 'vcode', voc.replace('voc', 'v')):
                continue

            out_dir_aux = os.path.join(output_path, f'{voc}_{gnfr}')
            out_file_aux = os.path.join(out_dir_aux, f'{voc}_{year}.nc')
            if os.path.exists(out_file_aux) and not overwrite:
                logger.info(f"Skipping existing file: {out_file_aux}")
                continue

            mask = build_ratio_mask(
                world_mask_path=world_mask_path,
                world_info_path=world_info_path,
                csv_path=voc_ratio_csv,
                species_column='vcode',
                species_name=voc.replace('voc', 'v'),
                gnfr=gnfr,
                year=year,
            )
            new_voc_data = nmvoc_data * mask

            nessy = nes.create_nes(
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
            nessy.create_spatial_bounds()
            nessy.lat_bnds["data"] = nessy_in_voc.lat_bnds["data"]
            nessy.lon_bnds["data"] = nessy_in_voc.lon_bnds["data"]
            nessy.cell_measures["cell_area"] = {
                "data": nessy_in_voc.cell_measures["cell_area"]["data"],
                "units": "m2",
            }
            nessy.variables[voc] = {
                "data": new_voc_data,
                "units": 'kg.m-2.s-1',
                "description": f"Emissions of pollutant VOC {voc} of sector {gnfr}",
                "short_name": voc,
            }
            nessy.global_attrs = {
                'references': 'J. J. P. Kuenen, A. J. H. Visschedijk, M. Jozwicka, and H. A. C. '
                              'Denier van der Gon TNO-MACC_II emission inventory; a multi-year '
                              '(2003-2009) consistent high-resolution European emission inventory '
                              'for air quality modelling Atmospheric Chemistry and Physics 14 '
                              '10963-10976 2014',
                'comment': 'Re-writing done by Carles Tena (carles.tena@bsc.es) from the BSC-CNS '
                           '(Barcelona Supercomputing Center)',
            }

            os.makedirs(out_dir_aux, exist_ok=True)
            nessy.to_netcdf(out_file_aux)

    return True


def check_vocs(output_path, year, voc_ratio_csv, sector_list=None):
    """
    Check the consistency between the base ``nmvoc`` totals and the VOC split totals.

    Parameters
    ----------
    output_path : str
        Root directory containing the base and speciated yearly outputs.
    year : int
        Year to evaluate.
    voc_ratio_csv : str
        Path to the VOC ratio CSV file.
    sector_list : list[str], optional
        GNFR sectors to evaluate. If ``None``, all supported sectors are
        checked.

    Returns
    -------
    bool
        ``True`` when the check finishes.
    """
    selected_sectors = normalize_sector_list(sector_list)

    for gnfr in get_sector_list():
        if selected_sectors is not None and gnfr.lower() not in selected_sectors:
            continue

        nmvoc_path = os.path.join(output_path, 'nmvoc_{0}'.format(gnfr), 'nmvoc_{0}.nc'.format(year))
        if not os.path.exists(nmvoc_path):
            continue

        nessy_nmvoc = nes.open_netcdf(nmvoc_path)
        nessy_nmvoc.load("nmvoc")
        new_voc = {"name": "nmvoc", "data": nessy_nmvoc.variables["nmvoc"]["data"]}
        nmvoc_sum = new_voc['data'].sum()

        voc_sum = 0
        for voc in get_voc_list(voc_ratio_csv):
            voc_path = os.path.join(output_path, f'{voc}_{gnfr}', f'{voc}_{year}.nc')
            if os.path.exists(voc_path):
                nessy_voc = nes.open_netcdf(voc_path)
                nessy_voc.load(voc)
                new_voc = {"name": voc,
                           "data": nessy_voc.variables[voc]["data"]}
                voc_sum += new_voc['data'].sum()

        logger.info(f'{gnfr} NMVOC sum: {nmvoc_sum}; VOCs sum: {voc_sum}; '
                    f'%diff: {100*(nmvoc_sum - voc_sum) / nmvoc_sum}')

    return True


def do_pm_transformation(output_path, year, world_info_path, world_mask_path, pm_ratio_csv,
                         pollutant_list=None, sector_list=None, overwrite=False):
    """
    Generate the PM split pollutant files from the base ``pm25`` outputs.

    Parameters
    ----------
    output_path : str
        Root directory containing the base and speciated yearly outputs.
    year : int
        Year to process.
    world_info_path : str
        Path to the CSV file containing the mapping between ISO alpha codes and
        the world mask codes.
    world_mask_path : str
        Path to the CAMS world mask NetCDF file.
    pm_ratio_csv : str
        Path to the CSV file containing the PM split factors.
    pollutant_list : list[str], optional
        Pollutants to process. If ``None``, all PM split species available in
        the ratio CSV are processed.
    sector_list : list[str], optional
        GNFR sectors to process. If ``None``, all supported sectors are
        processed.
    overwrite : bool, optional
        Whether to overwrite existing output files. If ``False``, existing
        output files are skipped.

    Returns
    -------
    bool
        ``True`` when the transformation finishes.
    """
    available_pm_list = get_pm_list(pm_ratio_csv)
    selected_pm_list = available_pm_list if pollutant_list is None else [
        pm for pm in available_pm_list if pm in pollutant_list
    ]
    selected_sectors = normalize_sector_list(sector_list)

    for gnfr in get_sector_list():
        if selected_sectors is not None and gnfr not in selected_sectors:
            continue

        in_path_pm = os.path.join(output_path, f'pm25_{gnfr}', f'pm25_{year}.nc')
        if not os.path.exists(in_path_pm):
            continue

        nessy_in_pm = nes.open_netcdf(in_path_pm)
        lats, lons, min_lats_interval, min_lons_interval = obtain_lats_lons(nessy_in_pm)
        nessy_in_pm.load("pm25")
        pm25_data = nessy_in_pm.variables["pm25"]["data"]

        for pm in selected_pm_list:
            if gnfr not in get_ratio_sector_list(pm_ratio_csv, 'pmcode', pm):
                continue

            out_dir_aux = os.path.join(output_path, f'{pm}_{gnfr}')
            out_file_aux = os.path.join(out_dir_aux, f'{pm}_{year}.nc')
            if os.path.exists(out_file_aux) and not overwrite:
                logger.info(f"Skipping existing file: {out_file_aux}")
                continue

            mask = build_ratio_mask(
                world_mask_path=world_mask_path,
                world_info_path=world_info_path,
                csv_path=pm_ratio_csv,
                species_column='pmcode',
                species_name=pm,
                gnfr=gnfr,
                year=year,
            )
            new_pm_data = pm25_data * mask

            nessy = nes.create_nes(
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
            nessy.create_spatial_bounds()
            nessy.lat_bnds["data"] = nessy_in_pm.lat_bnds["data"]
            nessy.lon_bnds["data"] = nessy_in_pm.lon_bnds["data"]
            nessy.cell_measures["cell_area"] = {
                "data": nessy_in_pm.cell_measures["cell_area"]["data"],
                "units": "m2",
            }
            nessy.variables[pm] = {
                "data": new_pm_data,
                "units": 'kg.m-2.s-1',
                "description": f"Emissions of pollutant PM {pm} of sector {gnfr}",
                "short_name": pm,
            }
            nessy.global_attrs = {
                'references': 'J. J. P. Kuenen, A. J. H. Visschedijk, M. Jozwicka, and H. A. C. '
                              'Denier van der Gon TNO-MACC_II emission inventory; a multi-year '
                              '(2003-2009) consistent high-resolution European emission inventory '
                              'for air quality modelling Atmospheric Chemistry and Physics 14 '
                              '10963-10976 2014',
                'comment': 'Re-writing done by Carles Tena (carles.tena@bsc.es) from the BSC-CNS '
                           '(Barcelona Supercomputing Center)',
            }

            os.makedirs(out_dir_aux, exist_ok=True)
            nessy.to_netcdf(out_file_aux)

    return True


def check_pm(output_path, year, pm_ratio_csv, sector_list=None):
    """
    Check the consistency between the base ``pm25`` totals and the PM split totals.

    Parameters
    ----------
    output_path : str
        Root directory containing the base and speciated yearly outputs.
    year : int
        Year to evaluate.
    pm_ratio_csv : str
        Path to the PM ratio CSV file.
    sector_list : list[str], optional
        GNFR sectors to evaluate. If ``None``, all supported sectors are
        checked.

    Returns
    -------
    bool
        ``True`` when the check finishes.
    """
    selected_sectors = normalize_sector_list(sector_list)

    for gnfr in get_sector_list():
        if selected_sectors is not None and gnfr not in selected_sectors:
            continue

        pm25_path = os.path.join(output_path, 'pm25_{0}'.format(gnfr), 'pm25_{0}.nc'.format(year))
        if not os.path.exists(pm25_path):
            continue

        nessy_in_pm25 = nes.open_netcdf(pm25_path)
        nessy_in_pm25.load("pm25")
        new_pm25 = {"name": "pm25",
                    "data": nessy_in_pm25.variables["pm25"]["data"]}
        pm25_sum = new_pm25['data'].sum()

        pm_sum = 0
        for pm in get_pm_list(pm_ratio_csv):
            pm_path = os.path.join(output_path, '{0}_{1}'.format(pm, gnfr), '{0}_{1}.nc'.format(pm, year))
            if os.path.exists(pm_path):
                nessy_pm = nes.open_netcdf(pm_path)
                nessy_pm.load(pm)
                new_pm = {"name": pm, "data": nessy_pm.variables[pm]["data"]}
                pm_sum += new_pm['data'].sum()

        logger.info(f'{gnfr} PM2.5 sum: {pm25_sum}; PM sum: {pm_sum}; %diff: {100*(pm25_sum - pm_sum) / pm25_sum}')
    return True


def cams_reg_ap_preproc(original_file, output_path, year, world_info_path, world_mask_path, voc_ratio_csv,
                        pm_ratio_csv, pollutant_list=None, sector_list=None, overwrite=False,
                        check_totals=False):
    """
    Convert CAMS-REG-AP yearly emissions into HERMES-ready yearly outputs.

    This function writes the base yearly pollutants from the CSV input and then
    applies the VOC and PM split factors to generate the speciated yearly outputs.

    Parameters
    ----------
    original_file : str
        Path to the CAMS-REG-AP yearly CSV file.
    output_path : str
        Root directory where the preprocessed yearly files will be written.
    year : int
        Year to process.
    world_info_path : str
        Path to the CSV file containing the mapping between ISO alpha codes and
        the world mask codes.
    world_mask_path : str
        Path to the CAMS world mask NetCDF file.
    voc_ratio_csv : str
        Path to the CSV file containing the VOC split factors.
    pm_ratio_csv : str
        Path to the CSV file containing the PM split factors.
    pollutant_list : list[str], optional
        Pollutants to process. If ``None``, all available base and split
        pollutants are processed. If only split species are requested, the
        required base pollutants are generated internally as needed.
    sector_list : list[str], optional
        GNFR sectors to process. If ``None``, all supported sectors are
        processed.
    overwrite : bool, optional
        Whether to overwrite existing output files. If ``False``, existing
        output files are skipped.
    check_totals : bool, optional
        Whether to print the VOC and PM consistency checks after processing.

    Returns
    -------
    None
    """

    # Resolve the year placeholder in the input path template.
    original_file = original_file.replace('<year>', str(year))

    do_transformation(
        original_file=original_file,
        output_path=output_path,
        year=year,
        pollutant_list=pollutant_list,
        sector_list=sector_list,
        overwrite=overwrite,
        voc_ratio_csv=voc_ratio_csv,
        pm_ratio_csv=pm_ratio_csv
    )
    do_voc_transformation(
        output_path=output_path,
        year=year,
        world_info_path=world_info_path,
        world_mask_path=world_mask_path,
        voc_ratio_csv=voc_ratio_csv,
        pollutant_list=pollutant_list,
        sector_list=sector_list,
        overwrite=overwrite
    )
    do_pm_transformation(
        output_path=output_path,
        year=year,
        world_info_path=world_info_path,
        world_mask_path=world_mask_path,
        pm_ratio_csv=pm_ratio_csv,
        pollutant_list=pollutant_list,
        sector_list=sector_list,
        overwrite=overwrite
    )

    if check_totals:
        check_vocs(output_path=output_path, year=year, voc_ratio_csv=voc_ratio_csv, sector_list=sector_list)
        check_pm(output_path=output_path, year=year, pm_ratio_csv=pm_ratio_csv, sector_list=sector_list)

    return None


# Example standalone execution using the module-level testing parameters
# defined at the top of this script.
if __name__ == '__main__':
    cams_reg_ap_preproc(
        original_file=ORIGINAL_FILE,
        output_path=OUTPUT_PATH,
        year=YEAR,
        world_info_path=WORLD_INFO_PATH,
        world_mask_path=WORLD_MASK_PATH,
        voc_ratio_csv=VOC_RATIO_CSV_PATH,
        pm_ratio_csv=PM_RATIO_CSV_PATH,
        pollutant_list=None,
        sector_list=None,
        overwrite=False,
        check_totals=True,
    )
