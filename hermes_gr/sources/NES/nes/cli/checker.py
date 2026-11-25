# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml
from numpy import isinf, isnan

from nes.load_nes import open_netcdf

from .cli_logger import get_cli_logger

logger = get_cli_logger("checker")


# Class to hold min/max parameters boundaries from YAML
@dataclass
class Bounds:
    min_abs: float
    min_lower: float
    min_upper: float
    max_lower: float
    max_upper: float
    max_abs: float


@dataclass
class NcBounds:
    min: float
    max: float


def run_checks(
    input_file: str,
    check_nan: Optional[bool] = True,
    check_inf: Optional[bool] = True,
    var_list: Optional[List[str]] = None,
    avoid_first_hours: int = 0,
) -> bool:
    """
    Check for NaN and/or Inf values in selected variables of a NetCDF file.

    This function uses the NES `open_netcdf()` interface to load a NetCDF dataset,
    optionally restricts the check to a subset of variables (`var_list`),
    reads all selected variables into memory, and checks for the presence of NaN or Inf values.

    Parameters
    ----------
    input_file : str
        Path to the input NetCDF file.
    check_nan : Optional[bool]
        If True, check for NaN (Not a Number) values in the dataset. Default is True.
    check_inf : Optional[bool]
        If True, check for Inf (infinite) values in the dataset. Default is True.
    var_list : Optional[List[str]]
        List of variable names to check. If None, all variables will be checked.
    avoid_first_hours : Optional[int]
        Skips the first "N" hours. Default is 0 (checks all times).
    Returns
    -------
    bool
        True if no NaN or Inf values are found in any selected variable.

    Raises
    ------
    ValueError
        If any selected variable contains NaN or Inf values, a ValueError is raised
        indicating the variable name.

    Notes
    -----
    This function loads selected variable data into memory. Use with caution on large datasets.
    """

    # Open the NetCDF file with NES (metadata only)
    logger.info(f"Opening NetCDF file: {input_file}")
    dataset = open_netcdf(input_file, avoid_first_hours=avoid_first_hours)
    if var_list is not None:
        dataset.keep_vars(var_list)

    logger.info("Loading variable data into memory...")
    dataset.load()

    # Loop over all variables in the dataset
    for var_name, var_info in dataset.variables.items():
        logger.debug(f"Checking variable: {var_name}")
        # Check for Inf values, if requested
        has_inf = isinf(var_info["data"]).any() if check_inf else False

        # Check for NaN values, if requested
        has_nan = isnan(var_info["data"]).any() if check_nan else False

        # Raise an error if problematic values are found
        if has_nan or has_inf:
            logger.warning(f"Variable '{var_name}' contains NaN or Inf values.")
            raise ValueError(f"Variable '{var_name}' contains NaN or Inf values.")

    logger.info("All variables passed NaN/Inf checks.")
    return True


def min_max_check(
    input_path: str,
    config_path: str,
    avoid_first_hours: int = 0,
):
    """
    Check min and max values of variables against expected ranges from a YAML config file.

    Parameters
    ----------
    input_path : str
        Path to the input NetCDF file.
    config_path : str
        Path to the YAML configuration file containing expected min and max values for variables.
    avoid_first_hours : int
        Number of initial hours to skip for checking. Default is 0 (check all hours).
    Raises
    ----------
    ValueError
        If config_path is None or empty.
    FileNotFoundError
        If the config file does not exist or is not a file.
    """

    # Check that config path was provided
    if config_path is None or config_path == "":
        raise ValueError("YAML configuration file path must be provided.")

    # Convert to path object
    config_file = Path(config_path)

    # Check that config file is a file and it exists
    if not config_file.is_file():
        raise IsADirectoryError(
            f"Provided YAML configuration file {config_file} is not a file"
        )
    if not config_file.exists():
        raise FileExistsError(
            f"Provided YAML configuration file {config_file} does not exist."
        )

    # Open YAML file
    with open(config_file, "r") as config_file:
        logger.info(f"Opening YAML configuration file: {config_file}")
        try:
            config = yaml.safe_load(config_file)
        except Exception as e:
            raise yaml.YAMLError(
                f"Something went wrong during the opening of {config_file}: {e}"
            )

    # Assign config to Params class
    # Form of dict = {pollutant: {min_lower: value, min_upper: value, max_lower: value, max_upper: value}}
    try:
        params = {
            pollutant: Bounds(
                min_abs=value["min_abs"],
                min_lower=value["min_lower"],
                min_upper=value["min_upper"],
                max_lower=value["max_lower"],
                max_upper=value["max_upper"],
                max_abs=value["max_abs"],
            )
            for pollutant, value in config.items()
        }
    except KeyError as e:
        raise KeyError(
            f"YAML configuration file is missing expected key: {e}. Please check the format."
        )

    try:
        logger.info(f"Opening NetCDF file: {input_path}")
        nc = open_netcdf(input_path, avoid_first_hours=avoid_first_hours)
    except Exception as e:
        raise RuntimeError(f"Could not open NetCDF file {input_path}: {e}")

    # Get NetCDF variables
    nc_variables = nc.variables.keys()
    # Get conf file variables
    conf_vars = params.keys()
    # Get missing vars from NetCDF and present vars in NetCDF
    missing_vars = [var for var in conf_vars if var not in nc_variables]
    present_vars = [var for var in conf_vars if var in nc_variables]

    if missing_vars:
        logger.warning(
            f"Variable(s) {missing_vars} provided in configuration file were not found in NetCDF - "
            f"available variable(s) is/are {list(nc_variables)}"
        )
    if len(missing_vars) == len(conf_vars):
        raise ValueError(
            "None of the variables provided in the configuration file were found in the NetCDF. "
            "Please check that variable names match and are present in the NetCDF."
        )

    try:
        nc.keep_vars(present_vars)
        nc.load()
    except Exception as e:
        raise RuntimeError(f"Could not load NetCDF file {input_path}: {e}")

    # Extract variables from NetCDF
    nc_variables = nc.variables.keys()

    # Iterate through variables provided in YAML configuration file
    error_messages = []
    for var in present_vars:
        logger.debug(f"Checking min/max for variable '{var}'")
        bounds: Bounds = params[var]

        # Extract minimums and maximums for the current variable from the NetCDF
        nc_bounds = _find_variable_min_max(nc, var)

        # Check that the minimum of the NetCDF is above the expected minimi
        if check_min(bounds=bounds, nc_bounds=nc_bounds):
            logger.info(f"Min check passed for variable '{var}'")
        else:
            message = (
                f"Variable '{var}' min value {nc_bounds.min} is not in the expected range "
                f"[{bounds.min_lower}, {bounds.min_upper}] from YAML."
            )
            logger.warning(message)

        # Check that the maximum of the NetCDF is below the expected maximum
        if check_max(bounds=bounds, nc_bounds=nc_bounds):
            logger.info(f"Max check passed for '{var}'")
        else:
            message = (
                f"Variable '{var}' max value {nc_bounds.max} is not in the expected range "
                f"[{bounds.max_lower}, {bounds.max_upper}] from YAML."
            )
            logger.warning(message)

        if check_min_abs(bounds=bounds, nc_bounds=nc_bounds):
            logger.info(f"Min absolute check passed for variable '{var}'")
        else:
            message = (
                f"Variable '{var}' min value {nc_bounds.min} is below the expected "
                f"absolute minimum {bounds.min_abs} from YAML."
            )
            logger.warning(message)
            error_messages.append(message)

        # Check that the maximum of the NetCDF is below the expected absolute maximum
        if check_max_abs(bounds=bounds, nc_bounds=nc_bounds):
            logger.info(f"Max absolute check passed for variable '{var}'")
        else:
            message = (
                f"Variable '{var}' max value {nc_bounds.max} is above the expected "
                f"absolute maximum {bounds.max_abs} from YAML."
            )
            logger.warning(message)
            error_messages.append(message)

        logger.info(f"Finished min/max checks for variable '{var}'\n")
    # check if any errors were found
    if error_messages:
        raise ValueError("Min/Max checks failed:\n" + "\n".join(error_messages))


def _find_variable_min_max(nc, var: str) -> NcBounds:
    """Find the minimum and maximum for a specific variable and construct an NcBounds object holding its values.

    Parameters
    ----------
    nc :
        NES NetCDF dataset object.

    var : str
        Name of variable to find min and max for.

    Returns
    -------
    NcBounds
        A "NcBounds" object holding the min and max for the NetCDF.
    """

    return NcBounds(
        min=np.min(nc.variables[var]["data"]), max=np.max(nc.variables[var]["data"])
    )


def check_min(bounds: Bounds, nc_bounds: NcBounds) -> bool:
    """Check if the minimum value from the NetCDF is within the expected bounds provided in the config.

    Parameters
    ----------
    bounds : Bounds
        Configured lower/upper bounds from YAML.
    nc_bounds : NcBounds
        NetCDF min/max bounds.

    Returns
    -------
    bool
        Whether the minimum value is within the expected bounds.
    """

    if nc_bounds.min >= bounds.min_lower and nc_bounds.min <= bounds.min_upper:
        return True
    else:
        return False


def check_max(bounds: Bounds, nc_bounds: NcBounds) -> bool:
    """Check if the maximum value from the NetCDF is within the expected bounds provided in the config.

    Parameters
    ----------
    bounds : Bounds
        Configured lower/upper max bounds from YAML.
    nc_bounds : NcBounds
        NetCDF min/max bounds.

    Returns
    -------
    bool
        Whether the maximum value is within the expected bounds.
    """

    if nc_bounds.max >= bounds.max_lower and nc_bounds.max <= bounds.max_upper:
        return True
    else:
        return False


def check_min_abs(bounds: Bounds, nc_bounds: NcBounds) -> bool:
    """Check if the minimum value from the NetCDF is within the expected absolute bounds provided in the config.

    Parameters
    ----------
    bounds : Bounds
        Configured absolute bounds from YAML.
    nc_bounds : NcBounds
        NetCDF min/max bounds.
    Returns
    -------
    bool
        Whether the minimum value is within the expected absolute bounds.
    """

    if nc_bounds.min < bounds.min_abs:
        return False
    else:
        return True


def check_max_abs(bounds: Bounds, nc_bounds: NcBounds) -> bool:
    """Check if the maximum value from the NetCDF is within the expected absolute bounds provided in the config.

    Parameters
    ----------
    bounds : Bounds
        Configured absolute bounds from YAML.
    nc_bounds : NcBounds
        NetCDF min/max bounds.
    Returns
    -------
    bool
        Whether the maximum value is within the expected absolute bounds.
    """
    if nc_bounds.max > bounds.max_abs:
        return False
    else:
        return True
