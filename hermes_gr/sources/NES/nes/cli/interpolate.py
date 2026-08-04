# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

from nes import open_netcdf, create_nes
from .cli_logger import get_cli_logger
logger = get_cli_logger()


def validate_axis_options(args):
    """
    Validates that interpolation axis and related options are consistent.

    This function ensures that horizontal and vertical interpolation options are not mixed.
    It raises an error if options intended for one axis are used with the other, or if the axis is not supported.

    Parameters
    ----------
    args : Namespace
        Parsed arguments from the CLI, including interpolation axis and all related options.

    Raises
    ------
    ValueError
        If vertical options are used with horizontal axis or vice versa, or if an invalid axis is specified.
    """
    if args.axis == "horizontal":
        if args.method or args.extrapolate:
            logger.error("Vertical options (--method, --extrapolate) cannot be used with horizontal interpolation.")
            raise ValueError("Vertical options (--method, --extrapolate) cannot be used with horizontal interpolation.")
    elif args.axis == "vertical":
        if args.kind or args.n_neighbours or args.flux or args.weight_matrix_path or args.fix_border:
            logger.error("Horizontal interpolation options cannot be used with vertical interpolation.")
            raise ValueError("Horizontal interpolation options cannot be used with vertical interpolation.")
    else:
        logger.error(f"Unsupported interpolation axis: {args.axis}")
        raise ValueError(f"Unsupported interpolation axis: {args.axis}")


def get_destination(args):
    """
    Determines the destination grid for interpolation.

    This function either loads an existing destination file (if --destination is provided),
    or creates a new grid using the specified projection parameters (if --projection is provided).
    Only the relevant parameters for each projection type are passed to `create_nes`.

    Parameters
    ----------
    args : Namespace
        Parsed arguments from the CLI, containing options such as destination file path,
        projection type, and all necessary projection parameters.

    Returns
    -------
    nes.NES
        A loaded or newly created NES object representing the destination grid.

    Raises
    ------
    ValueError
        If neither --destination nor --projection is provided,
        or if the projection type is unsupported.
    """
    if args.destination:
        destination = open_netcdf(args.destination)
        nessy = destination

    if not args.projection:
        logger.error("Either --destination or --projection must be specified.")
        raise ValueError("Either --destination or --projection must be specified.")

    valid_projections = ["regular", "global", "rotated", "rotated_nested", "lcc", "mercator"]
    proj = args.projection.lower()
    if proj not in valid_projections:
        logger.error(f"Unsupported projection type: {args.projection}. "
                     f"Valid options are: {', '.join(valid_projections)}")
        raise ValueError(f"Unsupported projection type: {args.projection}. "
                         f"Valid options are: {', '.join(valid_projections)}")

    if proj == "regular":
        nessy = create_nes(
            projection=proj,
            lat_orig=args.lat_orig,
            lon_orig=args.lon_orig,
            inc_lat=args.inc_lat,
            inc_lon=args.inc_lon,
            n_lat=args.n_lat,
            n_lon=args.n_lon,
        )
    elif proj == "global":
        nessy = create_nes(
            projection=proj,
            inc_lat=args.inc_lat,
            inc_lon=args.inc_lon,
        )
    elif proj == "rotated":
        nessy = create_nes(
            projection=proj,
            centre_lat=args.centre_lat,
            centre_lon=args.centre_lon,
            west_boundary=args.west_boundary,
            south_boundary=args.south_boundary,
            inc_rlat=args.inc_rlat,
            inc_rlon=args.inc_rlon,
        )
    elif proj == "rotated_nested":
        nessy = create_nes(
            projection=proj,
            parent_grid_path=args.parent_grid_path,
            parent_ratio=args.parent_ratio,
            i_parent_start=args.i_parent_start,
            j_parent_start=args.j_parent_start,
            n_rlat=args.n_rlat,
            n_rlon=args.n_rlon,
        )
    elif proj == "lcc":
        nessy = create_nes(
            projection=proj,
            lat_1=args.lat_1,
            lat_2=args.lat_2,
            lon_0=args.lon_0,
            lat_0=args.lat_0,
            nx=args.nx,
            ny=args.ny,
            inc_x=args.inc_x,
            inc_y=args.inc_y,
            x_0=args.x_0,
            y_0=args.y_0,
        )
    elif proj == "mercator":
        nessy = create_nes(
            projection=proj,
            lat_ts=args.lat_ts,
            lon_0=args.lon_0,
            nx=args.nx,
            ny=args.ny,
            inc_x=args.inc_x,
            inc_y=args.inc_y,
            x_0=args.x_0,
            y_0=args.y_0,
        )
    else:
        logger.error(f"Unsupported projection type: {args.projection}")
        raise ValueError(f"Unsupported projection type: {args.projection}")

    return nessy


def interpolate(args):
    """
    Main entry point for the NES interpolation CLI command.

    Depending on the selected interpolation axis (horizontal or vertical), this function:
    - Loads the source NetCDF dataset.
    - Loads or creates the destination grid based on CLI options.
    - Calls the appropriate interpolation method.

    Parameters
    ----------
    args : Namespace
        Parsed arguments from the CLI, including interpolation settings and input/output file paths.

    Raises
    ------
    ValueError
        If axis-specific options are misused or required arguments are missing.
    """
    logger.info("Starting interpolation process.")
    validate_axis_options(args)

    source = open_netcdf(args.source)
    source.load()
    logger.info("Source NetCDF file loaded.")

    destination = get_destination(args)
    logger.info("Destination grid prepared.")

    if args.axis == "horizontal":
        logger.info("Performing horizontal interpolation.")
        source.interpolate_horizontal(
            source,
            destination,
            kind=args.kind,
            n_neighbours=args.n_neighbours,
            flux=args.flux,
            keep_nan=args.keep_nan,
            fix_border=args.fix_border,
            weight_matrix_path=args.weight_matrix_path,
            only_create_wm=args.only_create_wm,
            to_providentia=args.to_providentia,
            output=args.output
        )
    elif args.axis == "vertical":
        logger.info("Performing vertical interpolation.")
        source.interpolate_vertical(
            source,
            destination,
            method=args.method,
            extrapolate=args.extrapolate,
            output=args.output
        )
    logger.info("Interpolation completed successfully.")
