# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import traceback
from mpi4py import MPI
from configargparse import ArgParser
import argcomplete
from .cli_logger import get_cli_logger
import logging


def _add_cams_glob_ship_subparser(subparsers):
    """
    Add the CAMS GLOB SHIP preprocessing subcommand to the CLI.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        The subparsers object returned by `add_subparsers()` on the main parser.
    """
    from .cams_glob_ship_preproc import cams_glob_ship_preproc

    aux_parser = subparsers.add_parser("CAMS_GLOB_SHIP", help="HERMES_GR CAMS GLOB SHIP Preprocess")
    aux_parser.add_argument(
        "-i", "--original_file", required=True, help="Path to the CAMS GLOB SHIP original file."
    )
    aux_parser.add_argument(
        "-o", "--output_path", required=True, help="Output path template for the generated preprocessed files."
    )
    aux_parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing output files. If not set, existing outputs are skipped.",
    )
    aux_parser.add_argument(
        "-y",
        "--year",
        required=True,
        type=int,
        help="Year to process",
    )

    aux_parser.add_argument(
        "--fmi",
        action="store_true",
        default=False,
        help="Whether the input data came from FMI. If False, it came from ECCAD.",
    )

    aux_parser.add_argument(
        "--coordinate_correction",
        action="store_true",
        default=False,
        help="Correct centroid coordinates stored in the original files. If not set, original coordinates will be used "
             "which have a shift of 0.05 degrees in both latitude and longitude.",
    )
    aux_parser.set_defaults(func=cams_glob_ship_preproc)


def _add_cams_glob_ant_subparser(subparsers):
    """
    Add the CAMS GLOB ANT preprocessing subcommand to the CLI.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        The subparsers object returned by `add_subparsers()` on the main parser.
    """
    from .cams_glob_ant_preproc import cams_glob_ant_preproc

    aux_parser = subparsers.add_parser("CAMS_GLOB_ANT", help="HERMES_GR CAMS GLOB ANT Preprocess")
    aux_parser.add_argument(
        "-i", "--original_file", required=True, help="Path to the CAMS GLOB ANT original file."
    )
    aux_parser.add_argument(
        "-o", "--output_path", required=True, help="Output path template for the generated preprocessed files."
    )
    aux_parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing output files. If not set, existing outputs are skipped.",
    )
    aux_parser.add_argument(
        "-y",
        "--year",
        required=True,
        type=int,
        help="Year to process",
    )
    aux_parser.set_defaults(func=cams_glob_ant_preproc)


def _add_cams_glob_ocean_subparser(subparsers):
    """
    Add the CAMS GLOB OCEAN v3.1 preprocessing subcommand to the CLI.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        The subparsers object returned by `add_subparsers()` on the main parser.
    """
    from .cams_glob_ocean_preproc import cams_glob_ocean_preproc

    aux_parser = subparsers.add_parser("CAMS_GLOB_OCEAN", help="HERMES_GR CAMS GLOB OCEAN v3.1 Preprocess")
    aux_parser.add_argument(
        "-i", "--original_file", required=True, help="Path to the CAMS GLOB OCEAN original file."
    )
    aux_parser.add_argument(
        "-o", "--output_path", required=True, help="Output path template for the generated preprocessed files."
    )
    aux_parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing output files. If not set, existing outputs are skipped.",
    )
    aux_parser.add_argument(
        "-y",
        "--year",
        required=True,
        type=int,
        help="Year to process",
    )
    aux_parser.set_defaults(func=cams_glob_ocean_preproc)


def _add_cams_reg_ghg_subparser(subparsers):
    """
    Add the CAMS REG GHG preprocessing subcommand to the CLI.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        The subparsers object returned by `add_subparsers()` on the main parser.
    """
    from .cams_reg_ghg_preproc import cams_reg_ghg_preproc

    aux_parser = subparsers.add_parser("CAMS_REG_GHG", help="HERMES_GR CAMS REG GHG Preprocess")
    aux_parser.add_argument(
        "-i", "--original_file", required=True, help="Path to the CAMS REG GHG original file."
    )
    aux_parser.add_argument(
        "-o", "--output_path", required=True, help="Output path template for the generated preprocessed files."
    )
    aux_parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing output files. If not set, existing outputs are skipped.",
    )
    aux_parser.add_argument(
        "-y",
        "--year",
        required=True,
        type=int,
        help="Year to process",
    )
    aux_parser.set_defaults(func=cams_reg_ghg_preproc)


# 1) Add CAMS REG AP subparser function after _add_cams_reg_ghg_subparser
def _add_cams_reg_ap_subparser(subparsers):
    """
    Add the CAMS REG AP preprocessing subcommand to the CLI.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        The subparsers object returned by `add_subparsers()` on the main parser.
    """
    from .cams_reg_ap_preproc import cams_reg_ap_preproc

    aux_parser = subparsers.add_parser("CAMS_REG_AP", help="HERMES_GR CAMS REG AP Preprocess")
    aux_parser.add_argument(
        "-i", "--original_file", required=True,
        help="Path template to the CAMS REG AP original file (use <year> placeholder)."
    )
    aux_parser.add_argument(
        "-o", "--output_path", required=True,
        help="Output path for the generated preprocessed files."
    )
    aux_parser.add_argument(
        "-y", "--year", required=True, type=int,
        help="Year to process."
    )
    aux_parser.add_argument(
        "--world_info_path", required=True,
        help="Path to the world ISO mapping CSV file."
    )
    aux_parser.add_argument(
        "--world_mask_path", required=True,
        help="Path to the world mask NetCDF file."
    )
    aux_parser.add_argument(
        "--voc_ratio_csv", required=True,
        help="Path to the VOC ratio CSV file."
    )
    aux_parser.add_argument(
        "--pm_ratio_csv", required=True,
        help="Path to the PM ratio CSV file."
    )
    aux_parser.add_argument(
        "--pollutant_list", nargs="+", default=None,
        help="Optional list of pollutants to process."
    )
    aux_parser.add_argument(
        "--sector_list", nargs="+", default=None,
        help="Optional list of GNFR sectors to process."
    )
    aux_parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing output files. If not set, existing outputs are skipped.",
    )
    aux_parser.add_argument(
        "--check_totals",
        action="store_true",
        default=False,
        help="Check VOC and PM totals consistency after processing.",
    )

    aux_parser.set_defaults(func=cams_reg_ap_preproc)


# EMEP preprocessing subparser
def _add_emep_subparser(subparsers):
    """
    Add the EMEP preprocessing subcommand to the CLI.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        The subparsers object returned by `add_subparsers()` on the main parser.
    """
    from .emep_preproc import emep_preproc

    aux_parser = subparsers.add_parser("EMEP", help="HERMES_GR EMEP Preprocess")
    aux_parser.add_argument(
        "-i", "--original_file", required=True,
        help="Path template to the original EMEP files."
    )
    aux_parser.add_argument(
        "-o", "--output_path", required=True,
        help="Output path for the generated preprocessed files."
    )
    aux_parser.add_argument(
        "-y", "--year", required=True, type=int,
        help="Year to process."
    )
    aux_parser.add_argument(
        "--pollutant_list", nargs="+", default=None,
        help="Optional list of pollutants to process."
    )
    aux_parser.add_argument(
        "--sector_list", nargs="+", default=None,
        help="Optional list of sectors to process."
    )
    input_group = aux_parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--txt",
        action="store_true",
        default=False,
        help="Process the TXT input format. By default, NetCDF input is used.",
    )
    aux_parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing output files. If not set, existing outputs are skipped.",
    )

    aux_parser.set_defaults(func=emep_preproc, nc=True)


# EDGAR preprocessing subparser
def _add_edgar_subparser(subparsers):
    """
    Add the EDGAR preprocessing subcommand to the CLI.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        The subparsers object returned by `add_subparsers()` on the main parser.
    """
    from .edgar_preproc import edgar_preproc

    aux_parser = subparsers.add_parser("EDGAR", help="HERMES_GR EDGAR Preprocess")
    aux_parser.add_argument(
        "-i", "--original_file", required=True,
        help="Path template to the original EDGAR files."
    )
    aux_parser.add_argument(
        "-o", "--output_path", required=True,
        help="Output path for the generated preprocessed files."
    )
    aux_parser.add_argument(
        "-y", "--year", required=True, type=int,
        help="Year to process."
    )
    aux_parser.add_argument(
        "--pollutant_list", nargs="+", default=None,
        help="Optional list of pollutants to process."
    )
    aux_parser.add_argument(
        "--sector_list", nargs="+", default=None,
        help="Optional list of sectors to process."
    )
    aux_parser.add_argument(
        "--monthly_temporal_profiles", default=None,
        help="Path template to monthly temporal profiles (required if monthly and year != 2010)."
    )
    processing_group = aux_parser.add_mutually_exclusive_group(required=True)
    processing_group.add_argument(
        "--yearly",
        action="store_true",
        default=False,
        help="Process yearly emissions."
    )
    processing_group.add_argument(
        "--monthly",
        action="store_true",
        default=False,
        help="Process monthly emissions."
    )
    aux_parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing output files. If not set, existing outputs are skipped.",
    )

    aux_parser.set_defaults(func=edgar_preproc)


def _add_greta_subparser(subparsers):
    """
    Add the GRETA preprocessing subcommand to the CLI.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        The subparsers object returned by ``add_subparsers()`` on the main parser.
    """
    from .greta_preproc import (
        CAMS_REG_PATH,
        NMVOC_SPLIT_PATH,
        ORIGINAL_DIR,
        ORIGINAL_FILES,
        OUTPUT_PATH,
        PM_SPLIT_PATH,
        greta_preproc,
    )

    aux_parser = subparsers.add_parser("GRETA", help="HERMES_GR GRETA Preprocess")
    aux_parser.add_argument(
        "-i", "--original_dir", default=ORIGINAL_DIR,
        help="Directory containing GRETA NetCDF files, or path to one GRETA NetCDF file.",
    )
    # Match the archive layout: output_path is the inventory root, not a year.
    aux_parser.add_argument(
        "-o", "--output_path", default=OUTPUT_PATH,
        help="Inventory root directory; outputs are grouped under yearly_mean by pollutant and sector.",
    )
    aux_parser.add_argument(
        "-y", "--year", "--years", dest="years", nargs="+", type=int,
        choices=sorted(ORIGINAL_FILES),
        default=[2022],
        help="Inventory year(s) to process (2022 and/or 2030; default: 2022).",
    )
    aux_parser.add_argument(
        "--cams_reg_path", default=CAMS_REG_PATH,
        help="Path to the CAMS-REG 2022 inventory CSV used for the GNFR F split.",
    )
    aux_parser.add_argument(
        "--nmvoc_split_path", default=NMVOC_SPLIT_PATH,
        help="Path to the NMVOC speciation-factor CSV.",
    )
    aux_parser.add_argument(
        "--pm_split_path", default=PM_SPLIT_PATH,
        help="Path to the PM2.5 component-factor CSV.",
    )
    aux_parser.add_argument(
        "--overwrite", action="store_true", default=False,
        help="Overwrite existing output files. If not set, existing outputs are skipped.",
    )
    aux_parser.set_defaults(func=greta_preproc)


def _add_greta_ghg_subparser(subparsers):
    """Add the GRETA greenhouse-gas preprocessing subcommand.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        The subparsers object returned by ``add_subparsers()`` on the main parser.
    """
    from .greta_preproc import (
        CAMS_REG_PATH,
        GHG_ORIGINAL_FILE,
        OUTPUT_PATH,
        greta_ghg_preproc,
    )

    aux_parser = subparsers.add_parser(
        "GRETA_GHG", help="HERMES_GR GRETA greenhouse-gas preprocess"
    )
    aux_parser.add_argument(
        "-i", "--original_file", default=GHG_ORIGINAL_FILE,
        help="Path to the GRETA 2022 GHG NetCDF file.",
    )
    aux_parser.add_argument(
        "-o", "--output_path", default=OUTPUT_PATH,
        help="GRETA inventory root for the generated files.",
    )
    aux_parser.add_argument(
        "-y", "--year", type=int, choices=(2022,), default=2022,
        help="Inventory year (the configured GHG file is for 2022).",
    )
    aux_parser.add_argument(
        "--cams_reg_path", default=CAMS_REG_PATH,
        help="Path to the CAMS-REG 2022 inventory CSV used for the GNFR F split.",
    )
    aux_parser.add_argument(
        "--overwrite", action="store_true", default=False,
        help="Overwrite existing output files. If not set, existing outputs are skipped.",
    )
    aux_parser.set_defaults(func=greta_ghg_preproc)


def _filter_args(func, args_namespace):
    """
    Filter parsed CLI arguments to only include those accepted by the target function.

    Parameters
    ----------
    func : Callable
        The function to match arguments against.
    args_namespace : argparse.Namespace
        The full set of parsed CLI arguments.

    This allows unrelated CLI arguments to be ignored safely when dispatching
    execution to a specific preprocessing function.
    """
    import inspect

    sig = inspect.signature(func)
    arg_keys = set(sig.parameters.keys())
    args_dict = vars(args_namespace)
    filtered = {k: args_dict[k] for k in arg_keys if k in args_dict}

    return filtered


def main():
    """
    Main entry point for the HERMES_GR preprocessing command-line interface.

    Sets up the available subcommands, parses user input from the CLI,
    and dispatches execution to the appropriate preprocessing function.
    """
    parser = ArgParser(description="HERMES_GR preprocessing utilities")
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Suppress informational logging messages and keep only critical errors.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Add subcommands
    _add_cams_glob_ship_subparser(subparsers)
    _add_cams_glob_ant_subparser(subparsers)
    _add_cams_glob_ocean_subparser(subparsers)
    _add_cams_reg_ghg_subparser(subparsers)
    _add_cams_reg_ap_subparser(subparsers)
    _add_emep_subparser(subparsers)
    _add_edgar_subparser(subparsers)
    _add_greta_subparser(subparsers)
    _add_greta_ghg_subparser(subparsers)

    # Enable autocomplete
    argcomplete.autocomplete(parser)

    args = parser.parse_args()

    if args.command == "EMEP" and args.txt:
        args.nc = False

    log_level = logging.ERROR if args.no_log else logging.INFO
    logger = get_cli_logger(f"HERMES_GR preproc::{args.command}", level=log_level)

    try:
        filtered_args = _filter_args(args.func, args)
        args.func(**filtered_args)
    except Exception as e:
        rank = MPI.COMM_WORLD.Get_rank()
        logger.error(f"[{args.command}] Rank {rank}: HERMES_GR preproc critical error: {e}")
        logger.error(f"[{args.command}] Rank {rank}: Traceback:\n{traceback.format_exc()}")
        MPI.COMM_WORLD.Abort(1)


if __name__ == "__main__":
    main()
