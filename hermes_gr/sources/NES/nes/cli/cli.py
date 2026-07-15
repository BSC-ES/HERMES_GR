# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import traceback
from mpi4py import MPI
from configargparse import ArgParser
import argcomplete
from .cli_logger import get_cli_logger
import logging


# def _add_nc2mbtiles_subparser(subparsers):
#     """
#     Add the 'mbtiles' subcommand to the NES CLI.

#     This command converts a monthly NetCDF into daily mbtiles.

#     Parameters
#     ----------
#     subparsers : argparse._SubParsersAction
#         The subparsers object returned by `add_subparsers()` on the main parser.
#     """
#     from nes.cli import nc2mbtiles

#     mbtiles_parser = subparsers.add_parser(
#         "nc2mbtiles", help="Convert a monthly NetCDF into daily mbtiles"
#     )
#     mbtiles_parser.add_argument(
#         "-i", "--input_file", required=True, help="Path to input NetCDF file"
#     )
#     mbtiles_parser.add_argument(
#         "-o",
#         "--output_path",
#         required=True,
#         help="Path to the path where the mbtiles will be saved",
#     )
#     mbtiles_parser.add_argument(
#         "-f",
#         "--mbtiles_file",
#         required=True,
#         help="Name for the mbtile files, will be used as a prefix for the daily mbtiles",
#     )
#     mbtiles_parser.add_argument(
#         "-gs",
#         "--geostructure",
#         required=False,
#         default=None,
#         help="Optional path to geostructure file, allowing to skip the manual computation.",
#     )

#     mbtiles_parser.set_defaults(func=nc2mbtiles)


def _add_plot_subparser(subparsers):
    """
    Add the 'plot' subcommand to the NES CLI.

    Plot data from NetCDF files.
    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        The subparsers object returned by `add_subparsers()` on the main parser.
    """
    from nes.cli import plot_cli

    plot_parser = subparsers.add_parser("plot", help="Plot data from NetCDF files")
    plot_parser.add_argument(
        "-i", "--input_file", required=True, help="Path to input NetCDF file"
    )
    plot_parser.add_argument(
        "-v", "--variable", required=True, help="Variable name to plot"
    )
    plot_parser.add_argument(
        "-l",
        "--level",
        required=False,
        type=int,
        default=0,
        help="Vertical level index to plot (default: 0)",
    )
    plot_parser.add_argument(
        "-t",
        "--time_step",
        required=False,
        type=int,
        default=0,
        help="Time step index to plot (default: 0)",
    )
    plot_parser.add_argument(
        "-m",
        "--method",
        required=False,
        type=str,
        default="colormesh",
        help="Plotting method (default: colormesh)",
        choices=["colormesh", "surface", "contour", "histogram", "animation"],
    )

    plot_parser.add_argument(
        "--animation_method",
        required=False,
        type=str,
        default="colormesh",
        help="Animation method (default: colormesh)",
        choices=["colormesh", "contour"],
    )

    plot_parser.add_argument(
        "-c", "--cmap", default="viridis", help="Colormap to use for plotting"
    )
    plot_parser.add_argument(
        "--save",
        default=False,
        action="store_true",
        help='Save the plot instead of displaying it - ignored for method "animation"',
    )
    plot_parser.add_argument(
        "--scale",
        type=str,
        default="linear",
        choices=["linear", "log", "power"],
        help="Type of scale to use for the plot (default: linear)",
    )
    plot_parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Threshold value for filtering data (default: 0.0)",
    )
    plot_parser.add_argument(
        "--percentile_clip",
        nargs=2,
        type=float,
        default=(),
        metavar=("LOWER", "UPPER"),
        help="Clip data to the specified percentiles (e.g., 5 95)",
    )

    plot_parser.set_defaults(func=plot_cli)


def _add_check_min_max_subparser(subparsers):
    """
    Add the 'check_min_max' subcommand to the NES CLI.

    Check min and max values of variables in a NetCDF file against a YAML configuration.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        The subparsers object returned by `add_subparsers()` on the main parser.
    """
    from nes.cli import min_max_check

    check_parser = subparsers.add_parser(
        "check_min_max",
        help="Check min and max values of variables in a NetCDF file against a YAML configuration",
    )
    check_parser.add_argument(
        "-i", "--input_path", required=True, help="Path to input NetCDF file"
    )
    check_parser.add_argument(
        "-c", "--config_path", required=True, help="Path to YAML configuration file"
    )
    check_parser.add_argument(
        "--avoid_first_hours",
        dest="avoid_first_hours",
        type=int,
        default=0,
        help="Number of initial hours to skip for checking. Default is 0 (check all hours).",
    )

    check_parser.set_defaults(func=min_max_check)


def _add_nc2geostructure_subparser(subparsers):
    """
    Add the 'nc2geostructure' subcommand to the NES CLI.

    Convert NetCDF to geospatial structure (GeoJSON, shapefile) (TESTING PHASE).

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        The subparsers object returned by `add_subparsers()` on the main parser.
    """
    from nes.cli import nc2geostructure

    # TODO: TEST
    geo_parser = subparsers.add_parser(
        "nc2geostructure",
        help="Convert NetCDF to geospatial structure (GeoJSON, shapefile) (TESTING PHASE)",
    )
    geo_parser.add_argument(
        "-i", "--input_file", required=True, help="Path to input NetCDF file"
    )
    geo_parser.add_argument(
        "-o", "--output_file", required=True, help="Path to output geostructure"
    )
    geo_parser.add_argument(
        "-l",
        "--var_list",
        nargs="+",
        help="List of variable names to include in the geostructure. If omitted, all variables will be included.",
    )
    geo_parser.add_argument(
        "-z",
        "--level",
        type=int,
        default=0,
        help="Vertical level index to extract (default: 0); -1 to sum all vertical levels",
    )
    geo_parser.add_argument(
        "-t",
        "--time_step",
        type=int,
        default=0,
        help="Time step index to extract (default: 0); -1 to sum all time steps",
    )
    geo_parser.set_defaults(func=nc2geostructure)


def _add_interpolate_subparser(subparsers):
    """
    Add the 'interpolate' subcommand to the NES CLI.

    Interpolate data onto a different grid (TESTING PHASE).

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        The subparsers object returned by `add_subparsers()` on the main parser.
    """
    from nes.cli import interpolate

    # TODO: TEST
    interp_parser = subparsers.add_parser(
        "interpolate", help="Interpolate data onto a different grid (TESTING PHASE)"
    )

    # Main input/output
    general = interp_parser.add_argument_group("General options")
    general.add_argument(
        "-i", "--input_file", required=True, help="Path to source NetCDF file"
    )
    general.add_argument(
        "-o", "--output_file", required=True, help="Path to output NetCDF file"
    )
    general.add_argument(
        "--axis",
        choices=["horizontal", "vertical"],
        default="horizontal",
        help="Interpolation axis (default: horizontal)",
    )

    dst_group = general.add_mutually_exclusive_group(required=True)
    dst_group.add_argument(
        "-d", "--destination", help="Path to destination grid NetCDF file"
    )
    dst_group.add_argument(
        "--projection",
        help="Projection type to generate destination grid (e.g. regular, rotated, lcc)",
    )

    # Horizontal interpolation options
    horizontal = interp_parser.add_argument_group("Horizontal interpolation options")
    horizontal.add_argument(
        "--kind",
        choices=["NearestNeighbour", "Conservative"],
        help="Interpolation method for horizontal axis",
    )
    horizontal.add_argument(
        "--n-neighbours", type=int, help="Number of neighbors (NearestNeighbour)"
    )
    horizontal.add_argument(
        "--flux",
        action="store_true",
        help="Treat variables as fluxes (Conservative only)",
    )
    horizontal.add_argument(
        "--keep-nan", action="store_true", help="Keep NaN values after interpolation"
    )
    horizontal.add_argument(
        "--fix-border",
        action="store_true",
        help="Fix border effects (NearestNeighbour only)",
    )
    horizontal.add_argument("--weight-matrix-path", help="Path to weight matrix file")
    horizontal.add_argument(
        "--only-create-wm", action="store_true", help="Only generate weight matrix"
    )
    horizontal.add_argument(
        "--to-providentia", action="store_true", help="Format output for Providentia"
    )

    # Vertical interpolation options
    vertical = interp_parser.add_argument_group("Vertical interpolation options")
    vertical.add_argument(
        "--method", help="Interpolation method for vertical axis (e.g. linear)"
    )
    vertical.add_argument(
        "--extrapolate",
        action="store_true",
        help="Allow extrapolation in vertical interpolation",
    )

    # Grid creation arguments
    grid = interp_parser.add_argument_group("Grid creation options (for --projection)")
    grid.add_argument("--lat_orig", type=float, help="Latitude origin (regular/global)")
    grid.add_argument(
        "--lon_orig", type=float, help="Longitude origin (regular/global)"
    )
    grid.add_argument(
        "--inc_lat", type=float, help="Latitude increment (regular/global)"
    )
    grid.add_argument(
        "--inc_lon", type=float, help="Longitude increment (regular/global)"
    )
    grid.add_argument(
        "--n_lat", type=int, help="Number of latitude points (regular/global)"
    )
    grid.add_argument(
        "--n_lon", type=int, help="Number of longitude points (regular/global)"
    )

    grid.add_argument(
        "--centre_lat", type=float, help="Rotated pole latitude (rotated)"
    )
    grid.add_argument(
        "--centre_lon", type=float, help="Rotated pole longitude (rotated)"
    )
    grid.add_argument("--west_boundary", type=float, help="Western boundary (rotated)")
    grid.add_argument(
        "--south_boundary", type=float, help="Southern boundary (rotated)"
    )
    grid.add_argument("--inc_rlat", type=float, help="Latitude increment (rotated)")
    grid.add_argument("--inc_rlon", type=float, help="Longitude increment (rotated)")

    grid.add_argument(
        "--parent_grid_path", help="Path to parent grid NetCDF (rotated_nested)"
    )
    grid.add_argument("--parent_ratio", type=int, help="Parent ratio (rotated_nested)")
    grid.add_argument(
        "--i_parent_start", type=int, help="Parent grid i index start (rotated_nested)"
    )
    grid.add_argument(
        "--j_parent_start", type=int, help="Parent grid j index start (rotated_nested)"
    )
    grid.add_argument(
        "--n_rlat", type=int, help="Number of lat points (rotated_nested)"
    )
    grid.add_argument(
        "--n_rlon", type=int, help="Number of lon points (rotated_nested)"
    )

    grid.add_argument(
        "--lat_1", type=float, help="First standard parallel (LCC projection)"
    )
    grid.add_argument(
        "--lat_2", type=float, help="Second standard parallel (LCC projection)"
    )
    grid.add_argument("--lon_0", type=float, help="Central meridian (LCC projection)")
    grid.add_argument("--x_0", type=float, help="False easting (LCC projection)")
    grid.add_argument("--y_0", type=float, help="False northing (LCC projection)")
    grid.add_argument(
        "--dx", type=float, help="Grid spacing in x direction (LCC projection)"
    )
    grid.add_argument(
        "--dy", type=float, help="Grid spacing in y direction (LCC projection)"
    )
    grid.add_argument(
        "--nx", type=int, help="Number of grid points in x (LCC projection)"
    )
    grid.add_argument(
        "--ny", type=int, help="Number of grid points in y (LCC projection)"
    )

    grid.add_argument(
        "--lat_ts", type=float, help="Latitude of true scale (Mercator projection)"
    )

    interp_parser.set_defaults(func=interpolate)


def _add_check_subparser(subparsers):
    """
    Add the 'check' subcommand to the NES CLI.

    Run checks on a NetCDF file.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        The subparsers object returned by `add_subparsers()` on the main parser.
    """
    from nes.cli import run_checks

    check_parser = subparsers.add_parser("check", help="Run checks on a NetCDF file")
    check_parser.add_argument(
        "-i", "--input_file", required=True, help="Input NetCDF file path"
    )
    check_parser.add_argument(
        "-l",
        "--var_list",
        nargs="+",
        help="List of NetCDF variables to include. If not provided, all variables will be used.",
        default=None,
    )
    check_parser.add_argument(
        "--nan", dest="check_nan", action="store_true", help="Check for NaN values"
    )
    check_parser.add_argument(
        "--no-nan",
        dest="check_nan",
        action="store_false",
        help="Do not check NaN values",
    )
    check_parser.add_argument(
        "--inf", dest="check_inf", action="store_true", help="Check for Inf values"
    )
    check_parser.add_argument(
        "--no-inf",
        dest="check_inf",
        action="store_false",
        help="Do not check Inf values",
    )
    check_parser.add_argument(
        "--avoid_first_hours",
        dest="avoid_first_hours",
        type=int,
        default=0,
        help="Number of initial hours to skip for checking. Default is 0 (check all hours).",
    )
    check_parser.set_defaults(check_nan=True, check_inf=True)
    check_parser.set_defaults(func=run_checks)


def _add_reorder_subparser(subparsers):
    """
    Add the 'reorder' subcommand to the NES CLI.

    Reorder longitudes in a NetCDF file (ONLY SERIAL).

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        The subparsers object returned by `add_subparsers()` on the main parser.
    """
    from nes.cli import reorder_longitudes

    # TODO: Add support for parallel version
    reorder_parser = subparsers.add_parser(
        "reorder", help="Reorder longitudes in a NetCDF file (ONLY SERIAL)"
    )
    reorder_parser.add_argument(
        "-i", "--input_file", required=True, help="Input NetCDF file path"
    )
    reorder_parser.add_argument(
        "-o", "--output_file", required=True, help="Output NetCDF file path"
    )
    reorder_parser.set_defaults(func=reorder_longitudes)


def _add_nc2rline_subparser(subparsers):
    """
    Add the 'nc2rline' subcommand to the NES CLI.

    Convert NetCDF to RLINE format using geometry lines.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        The subparsers object returned by `add_subparsers()` on the main parser.
    """
    from nes.cli import nc2rline

    rline_parser = subparsers.add_parser(
        "nc2rline", help="Convert NetCDF to RLINE format using geometry lines"
    )
    rline_parser.add_argument(
        "-i",
        "--input_file",
        required=True,
        help="NetCDF file to transform to RLINE format.",
    )
    rline_parser.add_argument(
        "-o",
        "--output_file",
        required=True,
        help=(
            "Output CSV file path for RLINE emissions. "
            "If the filename includes the placeholder '<pollutant>', it will be replaced with each pollutant name, "
            "producing one CSV per variable. "
            "Otherwise, the variable name will be appended before the .csv extension."
        ),
    )
    rline_parser.add_argument(
        "-g",
        "--geometry",
        required=True,
        help="Geometry file (e.g., GeoJSON or Shapefile) with roads.",
    )
    rline_parser.add_argument(
        "--crs",
        help="Coordinate Reference System to use for reprojection (e.g., 'EPSG:4326' or a PROJ string). "
        "If not provided, the CRS from the geometry file will be used.",
    )
    rline_parser.add_argument(
        "--output_geometry_file",
        help="Output CSV file path for RLINE-compatible geometry.",
        default=None,
    )
    rline_parser.add_argument(
        "--index_column",
        help="Column name in the geometry to use as index.",
        default=None,
    )
    rline_parser.add_argument(
        "--weight_column",
        help="Column name in the geometry to use as weight.",
        default=None,
    )
    rline_parser.add_argument(
        "--weight_file",
        help="Auxiliary weight file (read if exists, write otherwise).",
        default=None,
    )
    rline_parser.add_argument(
        "-l",
        "--var_list",
        nargs="+",
        help="List of NetCDF variables to include. If not provided, all variables will be used.",
        default=None,
    )
    rline_parser.set_defaults(func=nc2rline)


def _add_diffper_subparser(subparsers):
    """
    Add the 'diffper' subcommand to the NES CLI.

    Compute percentage difference between two NetCDF files.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        The subparsers object returned by `add_subparsers()` on the main parser.
    """
    from nes.cli import diffper

    diffper_parser = subparsers.add_parser(
        "diffper",
        help="Compute percentage difference between two NetCDF files",
    )

    # Inputs/outputs
    diffper_parser.add_argument(
        "-1",
        dest="file1",
        required=True,
        help="Path to first NetCDF file (baseline)",
    )
    diffper_parser.add_argument(
        "-2",
        dest="file2",
        required=True,
        help="Path to second NetCDF file (comparison)",
    )
    diffper_parser.add_argument(
        "-o",
        "--output_file",
        required=True,
        help="Path to output NetCDF file",
    )

    # Optional controls
    diffper_parser.add_argument(
        "-l",
        "--var_list",
        nargs="+",
        default=None,
        help="List of variables to process; defaults to all data variables.",
    )
    diffper_parser.add_argument(
        "--eps",
        type=float,
        default=0.0,
        help="Small value added to the denominator to avoid division by zero (default: 0.0)",
    )

    diffper_parser.set_defaults(func=diffper)


def _add_diff_subparser(subparsers):
    """
    Add the 'diff' subcommand to the NES CLI.

    Compute percentage difference between two NetCDF files.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        The subparsers object returned by `add_subparsers()` on the main parser.
    """
    from nes.cli import diff

    diff_parser = subparsers.add_parser(
        "diff",
        help="Compute percentage difference between two NetCDF files",
    )

    # Inputs/outputs
    diff_parser.add_argument(
        "-1",
        dest="file1",
        required=True,
        help="Path to first NetCDF file (baseline)",
    )
    diff_parser.add_argument(
        "-2",
        dest="file2",
        required=True,
        help="Path to second NetCDF file (comparison)",
    )
    diff_parser.add_argument(
        "-o",
        "--output_file",
        required=True,
        help="Path to output NetCDF file",
    )

    # Optional controls
    diff_parser.add_argument(
        "-l",
        "--var_list",
        nargs="+",
        default=None,
        help="List of variables to process; defaults to all data variables.",
    )

    diff_parser.set_defaults(func=diff)


def _add_mask_subparser(subparsers):
    """
    Add the 'mask' subcommand to the NES CLI.

    Apply mask to a NetCDF file.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        The subparsers object returned by `add_subparsers()` on the main parser.
    """
    from nes.cli import mask

    diff_parser = subparsers.add_parser(
        "mask",
        help="Apply mask to a NetCDF file",
    )

    # Inputs/outputs
    diff_parser.add_argument(
        "-f",
        "--file",
        required=True,
        help="Path to the baseline NetCDF file",
    )
    diff_parser.add_argument(
        "-m",
        "--mask",
        required=True,
        help="Path to the mask NetCDF file",
    )
    diff_parser.add_argument(
        "-o",
        "--output_file",
        required=True,
        help="Path to output NetCDF file",
    )

    # Optional controls
    diff_parser.add_argument(
        "-l",
        "--var_list",
        nargs="+",
        default=None,
        help="List of variables to process; defaults to all data variables.",
    )
    diff_parser.add_argument(
        "-v",
        "--var_mask",
        default=None,
        help="Name of the variable of the mask file; defaults to 'mask'.",
    )
    diff_parser.add_argument(
        "--inverse_mask",
        action="store_true",
        help="Invert mask values before applying (1 - mask).",
    )

    diff_parser.set_defaults(func=mask)


def _add_gridarea_subparser(subparsers):
    """
    Add the 'gridarea' subcommand to the NES CLI.

    Apply grid area to a NetCDF file.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        The subparsers object returned by `add_subparsers()` on the main parser.
    """
    from nes.cli import gridarea

    diff_parser = subparsers.add_parser(
        "gridarea",
        help="Apply gridarea to a NetCDF file",
    )

    # Inputs/outputs
    diff_parser.add_argument(
        "-f",
        "--filepath",
        required=True,
        help="Path to the baseline NetCDF file",
    )

    diff_parser.add_argument(
        "-o",
        "--output_file",
        required=False,
        help="Path to output NetCDF file",
    )

    diff_parser.set_defaults(func=gridarea)


def _filter_args(func, args_namespace):
    """
    Filters arguments from argparse to only include those relevant to the target function.

    Parameters
    ----------
    func : Callable
        The function to match arguments against.
    args_namespace : argparse.Namespace
        The full set of parsed CLI arguments.

    Returns
    -------
    dict
        A dictionary containing only the arguments accepted by the function.
    """
    import inspect

    sig = inspect.signature(func)
    arg_keys = set(sig.parameters.keys())
    args_dict = vars(args_namespace)
    filtered = {k: args_dict[k] for k in arg_keys if k in args_dict}

    return filtered


def main():
    """
    Main entry point for the NES command-line interface.

    Sets up the available subcommands, parses user input from the CLI,
    and dispatches execution to the appropriate subcommand handler.
    """
    parser = ArgParser(description="NES - NetCDF for Earth Science utilities")
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Suppress logging messages (only errors will be shown).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Add subcommands
    # _add_nc2mbtiles_subparser(subparsers)
    _add_nc2geostructure_subparser(subparsers)
    _add_check_subparser(subparsers)
    _add_reorder_subparser(subparsers)
    _add_nc2rline_subparser(subparsers)
    _add_interpolate_subparser(subparsers)
    _add_diffper_subparser(subparsers)
    _add_diff_subparser(subparsers)
    _add_mask_subparser(subparsers)
    _add_plot_subparser(subparsers)
    _add_check_min_max_subparser(subparsers)
    _add_gridarea_subparser(subparsers)

    # Enable autocomplete
    argcomplete.autocomplete(parser)

    args = parser.parse_args()

    log_level = logging.ERROR if args.no_log else logging.INFO
    logger = get_cli_logger(f"NES::{args.command}", level=log_level)

    try:
        filtered_args = _filter_args(args.func, args)
        args.func(**filtered_args)
    except Exception as e:
        logger.error(
            f"[{args.command}] Rank {MPI.COMM_WORLD.Get_rank()}: NES critical error: {e}"
        )
        logger.error(
            f"[{args.command}] Rank {MPI.COMM_WORLD.Get_rank()}: Traceback:\n{traceback.format_exc()}"
        )
        MPI.COMM_WORLD.Abort(1)
    return


if __name__ == "__main__":
    main()
