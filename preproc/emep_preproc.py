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
import pandas as pd
from .cli_logger import get_cli_logger

logger = get_cli_logger("emep")


# ============== CONFIGURATION PARAMETERS ======================
ORIGINAL_FILE_NC = "/esarchive/recon/ceip/emepv25/original_files/<pollutant>_2025_GRID_1990_to_2023.nc"
ORIGINAL_FILE_TXT = "/esarchive/recon/ceip/emepv19/original_files/<pollutant>_<sector>_2019_GRID_<year>.txt"
OUTPUT_PATH = "/esarchive/recon/ceip/emepv25/yearly_mean"
YEAR = 2015

LIST_POLLUTANTS = ["NOx", "NMVOC", "SOx", "NH3", "PM2_5", "PM10", "CO"]
LIST_SECTORS = [
    "A_PublicPower",
    "B_Industry",
    "C_OtherStationaryComb",
    "D_Fugitive",
    "E_Solvents",
    "F_RoadTransport",
    "G_Shipping",
    "H_Aviation",
    "I_Offroad",
    "J_Waste",
    "K_AgriLivestock",
    "L_AgriOther",
]


def get_default_pollutant_mapping():
    """
    Get the default mapping between EMEP pollutant names and HERMES_GR output names.

    Returns
    -------
    dict
        Default pollutant name mapping.
    """
    return {"nox": "nox_no2", "pm2_5": "pm25", "sox": "so2", "voc": "nmvoc"}


def correct_input_error(dataframe):
    """
    Correct specific known coordinate errors present in some EMEP text files.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input EMEP text table.

    Returns
    -------
    pandas.DataFrame
        Corrected dataframe.
    """
    dataframe.loc[dataframe["LATITUDE"] == 36.14, "LATITUDE"] = 36.15
    dataframe.loc[dataframe["LONGITUDE"] == 29.58, "LONGITUDE"] = 29.55
    return dataframe


def obtain_lats_lons_from_nc(nessy_in):
    """
    Extract latitude and longitude centers and increments from an NES object.

    Parameters
    ----------
    nessy_in : nes.Nes
        NES object containing latitude and longitude coordinates.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, float, float]
        Latitude centers, longitude centers, latitude increment, and longitude
        increment.
    """
    lats = nessy_in.lat["data"]
    lons = nessy_in.lon["data"]

    lats_interval = lats[1:] - lats[:-1]
    min_lats_interval = lats_interval.min()

    lons_interval = lons[1:] - lons[:-1]
    min_lons_interval = lons_interval.min()

    return lats, lons, min_lats_interval, min_lons_interval


def obtain_lats_lons_from_txt(dataframe):
    """
    Calculate the regular grid definition from an EMEP text table.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input EMEP text table.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, float, float]
        Latitude centers, longitude centers, latitude increment, and longitude
        increment.
    """
    lons = np.sort(np.unique(dataframe.LONGITUDE))
    lons_interval = lons[1:] - lons[:-1]
    min_lons_interval = lons_interval.min()

    lats = np.sort(np.unique(dataframe.LATITUDE))
    lats_interval = lats[1:] - lats[:-1]
    min_lats_interval = lats_interval.min()

    lats = np.arange(-90 + min_lats_interval / 2, 90, min_lats_interval, dtype=np.float64)
    lons = np.arange(-180 + min_lons_interval / 2, 180, min_lons_interval, dtype=np.float64)

    return lats, lons, min_lats_interval, min_lons_interval


def obtain_data_from_nc(nessy_in, sector, year):
    """
    Get EMEP emissions for one sector and year from the NetCDF input.
    The input time coordinate is expected to contain yearly timestamps.

    Parameters
    ----------
    nessy_in : nes.Nes
        Input EMEP NetCDF opened as NES object.
    sector : str
        GNFR sector name in the input naming convention.
    year : int
        Year to process.

    Returns
    -------
    numpy.ndarray
        Emission data for the selected sector and year.
    """
    time_array = np.array([datetime(year=date.year, month=1, day=1) for date in nessy_in.time])
    time_index = np.where(time_array == datetime(year=year, month=1, day=1))[0][0]
    nessy_in.load(sector_to_index(sector))
    return nessy_in.variables[sector_to_index(sector)]["data"][time_index, :, :]


def obtain_data_from_txt(dataframe, lats, lons, min_lats_interval, min_lons_interval):
    """
    Build the data array from the EMEP text table using the ``EMISSION`` column.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input EMEP text table.
    lats : numpy.ndarray
        Latitude centers.
    lons : numpy.ndarray
        Longitude centers.
    min_lats_interval : float
        Latitude increment.
    min_lons_interval : float
        Longitude increment.

    Returns
    -------
    numpy.ndarray
        Emission data array for the selected pollutant, sector, and year.
    """
    dataframe["row_lat"] = np.array(
        (dataframe.LATITUDE - (-90 + min_lats_interval / 2)) / min_lats_interval,
        dtype=np.int32,
    )
    dataframe["col_lon"] = np.array(
        (dataframe.LONGITUDE - (-180 + min_lons_interval / 2)) / min_lons_interval,
        dtype=np.int32,
    )
    dataframe = dataframe.groupby(["row_lat", "col_lon"]).sum().reset_index()

    data = np.zeros((len(lats), len(lons)))
    data[dataframe.row_lat, dataframe.col_lon] += dataframe["EMISSION"]
    return data


def sector_to_index(sector):
    """
    Convert a GNFR sector name into the EMEP sector variable name.

    Parameters
    ----------
    sector : str
        GNFR sector name.

    Returns
    -------
    str
        Sector variable name used in the EMEP NetCDF files.
    """
    sector_dict = {
        "A_PublicPower": "publicpower",
        "B_Industry": "industry",
        "C_OtherStationaryComb": "otherstationarycomb",
        "D_Fugitive": "fugitive",
        "E_Solvents": "solvents",
        "F_RoadTransport": "roadtransport",
        "G_Shipping": "shipping",
        "H_Aviation": "aviation",
        "I_Offroad": "offroad",
        "J_Waste": "waste",
        "K_AgriLivestock": "agrilivestock",
        "L_AgriOther": "agriother",
    }
    return sector_dict[sector]


def map_pollutant_name(pollutant, pollutant_mapping):
    """
    Map the EMEP pollutant name to the HERMES_GR output name.

    Parameters
    ----------
    pollutant : str
        Pollutant name in the EMEP naming convention.
    pollutant_mapping : dict
        Mapping between EMEP pollutant names and HERMES_GR output names.

    Returns
    -------
    str
        Pollutant name in the HERMES_GR naming convention.
    """
    return pollutant_mapping.get(pollutant.lower(), pollutant.lower())


def read_txt_input(input_file):
    """
    Read one EMEP text input file.
    The first four lines of the file are skipped during parsing.

    Parameters
    ----------
    input_file : str
        Path to the EMEP text input file.

    Returns
    -------
    pandas.DataFrame
        Parsed EMEP text table.
    """
    dataframe = pd.read_table(
        input_file,
        sep=";",
        skiprows=[0, 1, 2, 3],
        names=[
            "ISO2",
            "YEAR",
            "SECTOR",
            "POLLUTANT",
            "LONGITUDE",
            "LATITUDE",
            "UNIT",
            "EMISSION",
        ],
        header=0,
    )
    return correct_input_error(dataframe)


def emep_preproc(original_file, output_path, year, pollutant_list=None, sector_list=None,
                 pollutant_mapping=None, nc=True, txt=False, overwrite=False):
    """
    Convert EMEP annual emissions into HERMES-ready yearly outputs.

    Depending on the selected mode, the function reads either NetCDF EMEP inputs
    or text EMEP inputs and writes one yearly output file per pollutant and sector.
    Exactly one input mode must be selected.

    Parameters
    ----------
    original_file : str
        Path template to the original EMEP files. It must contain the required
        placeholders for the selected input mode, such as ``<pollutant>``,
        ``<sector>`` and ``<year>``.
    output_path : str
        Root directory where the preprocessed yearly files will be written.
    year : int
        Year to process.
    pollutant_list : list[str], optional
        Pollutants to process. If ``None``, all configured pollutants are used.
    sector_list : list[str], optional
        Sectors to process. If ``None``, all configured sectors are used.
    pollutant_mapping : dict, optional
        Mapping between EMEP pollutant names and HERMES_GR output names. If
        ``None``, the default mapping used by this script is applied.
    nc : bool, optional
        Whether to process the NetCDF input format. Exactly one of ``nc`` or
        ``txt`` must be ``True``.
    txt : bool, optional
        Whether to process the text input format. Exactly one of ``nc`` or
        ``txt`` must be ``True``.
    overwrite : bool, optional
        Whether to overwrite existing output files. If ``False``, existing
        output files are skipped.

    Returns
    -------
    None
    """
    default_pollutant_list = ["NOx", "NMVOC", "SOx", "NH3", "PM2_5", "PM10", "CO"]
    default_sector_list = [
        "A_PublicPower",
        "B_Industry",
        "C_OtherStationaryComb",
        "D_Fugitive",
        "E_Solvents",
        "F_RoadTransport",
        "G_Shipping",
        "H_Aviation",
        "I_Offroad",
        "J_Waste",
        "K_AgriLivestock",
        "L_AgriOther",
    ]
    if nc == txt:
        raise ValueError("Exactly one input mode must be selected: nc=True, txt=False or nc=False, txt=True.")

    if pollutant_mapping is None:
        pollutant_mapping = get_default_pollutant_mapping()

    selected_pollutants = default_pollutant_list if pollutant_list is None else pollutant_list
    selected_sectors = default_sector_list if sector_list is None else sector_list
    unit_factor_txt = 1000.0 / (365.0 * 24.0 * 3600.0)  # Convert annual emissions from Mg/year to kg/s. Use on txt
    unit_factor_nc = 1000000000 / (365.0 * 24.0 * 3600.0)  # Convert annual emissions from Tg/year to kg/s. Use on nc

    for pollutant in selected_pollutants:
        for sector in selected_sectors:
            if nc:
                input_file = original_file.replace("<pollutant>", pollutant)
            else:
                input_file = (
                    original_file
                    .replace("<year>", str(year))
                    .replace("<sector>", sector)
                    .replace("<pollutant>", pollutant)
                )

            if not os.path.exists(input_file):
                warning(
                    f"The pollutant {pollutant} for the GNFR14 sector {sector} for year {year} does not exist.\n"
                    f"File not found: {input_file}"
                )
                continue

            logger.info(f"Processing pollutant {pollutant} for the sector {sector} using input file\n{input_file}")

            if nc:
                nessy_in = nes.open_netcdf(input_file)
                lats, lons, min_lats_interval, min_lons_interval = obtain_lats_lons_from_nc(nessy_in)
                data = obtain_data_from_nc(nessy_in, sector, year)
                # Keep the NetCDF sector naming convention in the output path.
                sector_name = sector_to_index(sector)
            else:
                dataframe = read_txt_input(input_file)
                lats, lons, min_lats_interval, min_lons_interval = obtain_lats_lons_from_txt(dataframe)
                data = obtain_data_from_txt(dataframe, lats, lons, min_lats_interval, min_lons_interval)
                # Use the lowercase text sector name in the output path.
                sector_name = sector.lower()

            pollutant_name = map_pollutant_name(pollutant, pollutant_mapping)
            output_dir = os.path.join(output_path, f"{pollutant_name}_{sector_name}")
            output_file = os.path.join(output_dir, f"{pollutant_name}_{year}.nc")

            if os.path.exists(output_file) and not overwrite:
                logger.info(f"Skipping existing file: {output_file}")
                continue

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
            logger.info("Calculating cell area...")
            nessy.calculate_grid_area()
            logger.info("Cell area calculated.")

            nessy.cell_measures["cell_area"]["units"] = "m2"

            # make data 4D.
            if data.ndim == 2:
                data = data.reshape((1, 1,) + data.shape).astype(np.float32)
            elif data.ndim == 3:
                data = data.reshape((1, ) + data.shape).astype(np.float32)
            elif data.ndim == 4:
                print("Data is already 4D, no reshaping needed.")
            else:
                raise ValueError(f"Unexpected data dimensions: {data.ndim}. Expected 2, 3, or 4 dimensions.")

            if nc:
                unit_factor = unit_factor_nc
            else:
                unit_factor = unit_factor_txt
            data = data * unit_factor / nessy.cell_measures["cell_area"]["data"]

            nessy.variables[pollutant_name] = {
                "data": data,
                "units": "kg.m-2.s-1",
                "description": f"EMEP annual emissions of {pollutant_name} for GNFR14 sector {sector_name}",
                "short_name": pollutant_name,
            }
            nessy.global_attrs = {
                "references": "web: http://www.ceip.at/ms/ceip_home1/ceip_home/webdab_emepdatabase/"
                              "emissions_emepmodels/",
                "comment": "Re-writing done by Carles Tena (carles.tena@bsc.es) from the BSC-CNS "
                           "(Barcelona Supercomputing Center)",
            }

            os.makedirs(output_dir, exist_ok=True)
            nessy.to_netcdf(output_file)

    return None


# Example standalone execution using the module-level testing parameters
# defined at the top of this script.
# For TXT inputs, use ORIGINAL_FILE_TXT and set nc=False, txt=True.
if __name__ == "__main__":
    emep_preproc(
        original_file=ORIGINAL_FILE_NC,
        output_path=OUTPUT_PATH,
        year=YEAR,
        pollutant_list=LIST_POLLUTANTS,
        sector_list=LIST_SECTORS,
        nc=True,
        txt=False,
        overwrite=False,
    )
