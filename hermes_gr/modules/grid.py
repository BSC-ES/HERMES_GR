#!/usr/bin/env python

# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

from nes import create_nes, open_netcdf
from hermes_gr.config import log_message
from hermes_gr.config import get_time_stamp, record_time
import os
import pytz
from datetime import timedelta
from datetime import timezone as dtz
import pandas as pd

BALANCED = True


def select_grid(comm, options, levels, time_step_list=None):
    """
    Select & create the destination grid.

    1. If grid auxiliary file exists will be read, otherwise will be created


    Parameters
    ----------
    comm : MPI.COMMUNICATOR
        MPI communicator.
    options : NameSpace
        Configuration options.
    levels: List[float]
        List of vertical levels.
    time_step_list : List[datetime]
        List of date times

    Returns
    -------
    Nes
        A NES grid object with the shapefile & cell_area calculated.
    """
    st_time = get_time_stamp()
    # start_increment("Grid", "select_grid")
    # change_labels("Grid", "select_grid")

    grid_path = os.path.join(options.auxiliary_files_path, "grid.nc")
    grid_geo_path = os.path.join(options.auxiliary_files_path, "grid.geojson")

    if os.path.exists(grid_path):
        log_message(f"Reading existing grid from {grid_path}", level=1)
        grid = open_netcdf(grid_path, comm=comm, balanced=BALANCED)
        grid.load()
        write_grid = False
    else:
        log_message("Creating grid", level=1)
        write_grid = True

        # Creating a different object depending on the grid type
        if options.domain_type == "global_monarch":
            msg = f"Global grid with: \n\t\tinc_lat: {options.inc_lat}\n\t\tinc_lon: {options.inc_lon}"
            log_message(msg, level=1)

            grid = create_nes(
                comm=comm,
                info=False,
                times=time_step_list,
                projection="global_monarch",
                inc_lat=options.inc_lat,
                inc_lon=options.inc_lon,
                balanced=BALANCED,
            )
        elif options.domain_type == "global":
            msg = f"Global grid with: \n\t\tinc_lat: {options.inc_lat}\n\t\tinc_lon: {options.inc_lon}"
            log_message(msg, level=1)

            grid = create_nes(
                comm=comm,
                info=False,
                times=time_step_list,
                projection="global",
                inc_lat=options.inc_lat,
                inc_lon=options.inc_lon,
                balanced=BALANCED,
            )
        elif options.domain_type == "regular":
            log_message(
                f"Creating Regular Lat-Lon grid with:"
                + f"\n\t\tlat_orig: {options.lat_orig}\n\t\tlon_orig: {options.lon_orig}"
                + f"\n\t\tinc_lat: {options.inc_lat}\n\t\tinc_lon: {options.inc_lon}"
                + f"\n\t\tn_lat: {options.n_lat}\n\t\tn_lon: {options.n_lon}",
                level=1)
            grid = create_nes(
                comm=comm,
                info=False,
                times=time_step_list,
                projection="regular",
                lat_orig=options.lat_orig,
                lon_orig=options.lon_orig,
                inc_lat=options.inc_lat,
                inc_lon=options.inc_lon,
                n_lat=options.n_lat,
                n_lon=options.n_lon,
                balanced=BALANCED,
            )
            log_message("Regular grid created", level=2)

        elif options.domain_type == "rotated":
            log_message(
                f"Creating Rotated grid selected with:"
                + f"\n\t\tcentre_lat: {options.centre_lat}\n\t\tcentre_lon: {options.centre_lon}"
                + f"\n\t\twest_boundary: {options.west_boundary}\n\t\tsouth_boundary: {options.south_boundary}"
                + f"\n\t\tinc_rlat: {options.inc_rlat}\n\t\tinc_rlon: {options.inc_rlon}",
                level=1)

            grid = create_nes(
                comm=comm,
                info=False,
                times=time_step_list,
                projection="rotated",
                centre_lat=options.centre_lat,
                centre_lon=options.centre_lon,
                west_boundary=options.west_boundary,
                south_boundary=options.south_boundary,
                inc_rlat=options.inc_rlat,
                inc_rlon=options.inc_rlon,
                balanced=BALANCED,
            )

        elif options.domain_type == "rotated_nested":
            log_message(
                f"Creating Rotated-Nested grid with:"
                + f"\n\t\tparent_grid_path: {options.parent_grid_path}"
                + f"\n\t\tparent_ratio: {options.parent_ratio}"
                + f"\n\t\ti_parent_start: {options.i_parent_start}"
                + f"\n\t\tj_parent_start: {options.j_parent_start}"
                + f"\n\t\tn_rlat: {options.n_rlat}\n\t\tn_rlat: {options.n_lon}",
                level=1)

            grid = create_nes(
                comm=comm,
                info=False,
                times=time_step_list,
                projection="rotated-nested",
                parent_grid_path=options.parent_grid_path,
                parent_ratio=options.parent_ratio,
                i_parent_start=options.i_parent_start,
                j_parent_start=options.j_parent_start,
                n_rlat=options.n_rlat,
                n_rlon=options.n_rlon,
                balanced=BALANCED,
            )

        elif options.domain_type == "lcc":
            log_message(
                f"Creating Lambert Conformal Conic grid with:"
                + f"\n\t\tlat_1: {options.lat_1}\n\t\tlat_2: {options.lat_2}"
                + f"\n\t\tlon_0: {options.lon_0}\n\t\tlat_0: {options.lat_0}"
                + f"\n\t\tnx: {options.nx}\n\t\tny: {options.ny}"
                + f"\n\t\tinc_x: {options.inc_x}\n\t\tinc_y: {options.inc_y}"
                + f"\n\t\tx_0: {options.x_0}\n\t\ty_0: {options.y_0}",
                level=1)

            grid = create_nes(
                comm=comm,
                info=False,
                times=time_step_list,
                projection="lcc",
                lat_1=options.lat_1,
                lat_2=options.lat_2,
                lon_0=options.lon_0,
                lat_0=options.lat_0,
                nx=options.nx,
                ny=options.ny,
                inc_x=options.inc_x,
                inc_y=options.inc_y,
                x_0=options.x_0,
                y_0=options.y_0,
                balanced=BALANCED,
            )

        elif options.domain_type == "mercator":
            log_message(
                f"Creating Mercator grid with: "
                f"\n\t\tlat_ts: {options.lat_ts}\n\t\tlon_0: {options.lon_0}"
                f"\n\t\tnx: {options.nx}\n\t\tny: {options.ny}"
                f"\n\t\tinc_x: {options.inc_x}\n\t\tinc_y: {options.inc_y}"
                f"\n\t\tx_0: {options.x_0}\n\t\ty_0: {options.y_0}",
                level=1)

            grid = create_nes(
                comm=comm,
                info=False,
                times=time_step_list,
                projection="mercator",
                lat_ts=options.lat_ts,
                lon_0=options.lon_0,
                nx=options.nx,
                ny=options.ny,
                inc_x=options.inc_x,
                inc_y=options.inc_y,
                x_0=options.x_0,
                y_0=options.y_0,
                balanced=BALANCED,
            )
        else:
            log_message("ERROR: Check the .err file to get more info.")
            msg = (f"The grid type {options.domain_type} is not implemented. "
                   f"Use 'global', 'regular, 'rotated', 'rotated_nested', 'lcc' or 'mercator.")
            raise NotImplementedError(msg)

    # Cell area
    if "cell_area" not in grid.cell_measures.keys():
        write_grid = True
        grid.calculate_grid_area()
        log_message("Grid: cell area created", level=3)

    # Shapefile
    grid.create_shapefile()
    log_message("Grid: shapefile created", level=3)

    # Timezones
    if options.timezones_shapefile is None:
        raise FileNotFoundError("No timezones shapefile was defined. Fill the '--timezones_shapefile' option.")

    if "tzid" not in grid.variables.keys():
        write_grid = True
        log_message("Grid: Timezones calculation starting", level=3)
        grid.spatial_join(options.timezones_shapefile, method="nearest")
        grid.variables["tzid"] = {
            "data": grid.shapefile["tzid"].values.reshape((grid.lat["data"].shape[0], grid.lon["data"].shape[-1])),
            "dtype": str, }
        grid.set_strlen(32)
        log_message("Grid: Timezones created", level=3)
    else:
        grid.shapefile["tzid"] = grid.variables["tzid"]["data"].flatten()
        log_message("Grid: Timezones read from grid.nc", level=3)

    # Write
    if write_grid:
        grid.to_netcdf(grid_path, serial=True)
        log_message(f"Grid: file created: {grid_path}", level=2)
    grid.comm.Barrier()

    if not os.path.exists(grid_geo_path):
        grid.write_shapefile(grid_geo_path)
        log_message(f"Grid: geostructure file created: {grid_geo_path}", level=2)

    log_message("Destination grid done!", level=2)

    grid.set_levels({"data": levels, "units": "m", "positive": "up"})

    grid = add_time_info_to_grid(grid=grid, time_step_list=time_step_list)

    # stop_increment("Grid", "select_grid")
    grid.close()
    record_time("Grid", "select_grid", get_time_stamp() - st_time)
    return grid


def add_local_dates(grid, date_time, pandas=False):
    def parse_tz(timezone):
        """
        Parse the timezone (string format).

        It is needed because some libraries have more timezones than others, and it
        tries to simplify setting the strange ones into the nearest common one.
        Examples:
            "America/Punta_Arenas": "America/Santiago",
            "Europe/Astrakhan": "Europe/Moscow",
            "Asia/Atyrau": "Asia/Aqtau",
            "Asia/Barnaul": "Asia/Almaty",
            "Europe/Saratov": "Europe/Moscow",
            "Europe/Ulyanovsk": "Europe/Moscow",
            "Europe/Kirov": "Europe/Moscow",
            "Asia/Tomsk": "Asia/Novokuznetsk",
            "America/Fort_Nelson": "America/Vancouver"

        Parameters
        ----------
        timezone : str
            Not parsed timezone.

        Returns
        -------
        str
            Parsed timezone
        """

        tz_dict = {
            "America/Punta_Arenas": "America/Santiago",
            "Europe/Astrakhan": "Europe/Moscow",
            "Asia/Atyrau": "Asia/Aqtau",
            "Asia/Barnaul": "Asia/Almaty",
            "Europe/Saratov": "Europe/Moscow",
            "Europe/Ulyanovsk": "Europe/Moscow",
            "Europe/Kirov": "Europe/Moscow",
            "Asia/Tomsk": "Asia/Novokuznetsk",
            "America/Fort_Nelson": "America/Vancouver",
            "Asia/Famagusta": "Asia/Nicosia",
            "America/Nuuk": dtz(timedelta(hours=-3)),
        }

        if timezone in iter(tz_dict.keys()):
            timezone = tz_dict[timezone]

        return timezone

    grid.shapefile["local"] = pd.to_datetime(date_time, utc=True)
    try:
        # print(f"{grid.rank}, real local?: {grid.shapefile.groupby("tzid")["local"].apply(
        #     lambda x: x.dt.tz_convert(x.name).dt.tz_localize(None)).reset_index(level="tzid", drop=True)}")
        grid.shapefile["local"] = grid.shapefile.groupby("tzid")["local"].apply(
            lambda x: x.dt.tz_convert(x.name).dt.tz_localize(None)).reset_index(level="tzid", drop=True)

    # except TypeError as e:
    #     print(f"{grid.rank}, date_time: {date_time}")
    #     print(f"{grid.rank}, grid.shapefile["local"]: {grid.shapefile["local"]}")
    #     print(f"{grid.rank}, grid.shapefile["tzid"]: {grid.shapefile["tzid"]}")
    #     raise e
    except pytz.exceptions.UnknownTimeZoneError:
        grid.shapefile["local"] = grid.shapefile.groupby("tzid")["local"].apply(
            lambda x: x.dt.tz_convert(parse_tz(x.name)).dt.tz_localize(None))
    if pandas:
        time_stamp = grid.shapefile["local"].copy()
    else:
        time_stamp = (grid.shapefile["local"].to_numpy().reshape(
            (grid.lat["data"].shape[0], grid.lon["data"].shape[-1])))

    del grid.shapefile["local"]

    return time_stamp


def add_time_info_to_grid(grid, time_step_list):
    import numpy as np

    st_time = get_time_stamp()
    # start_increment("HERMES", "add_time_info_to_grid")
    # change_labels("HERMES", "add_time_info_to_grid")

    shape_3d = (
        len(time_step_list),
        grid.lat["data"].shape[0],
        grid.lon["data"].shape[-1],
    )
    # Empty 2D (time, lat*lon)
    grid.variables["month"] = {"data": np.empty((shape_3d[0], shape_3d[1] * shape_3d[2]), dtype=np.uint8)}
    grid.variables["weekday"] = {"data": np.empty((shape_3d[0], shape_3d[1] * shape_3d[2]), dtype=np.uint8)}
    grid.variables["hour"] = {"data": np.empty((shape_3d[0], shape_3d[1] * shape_3d[2]), dtype=np.uint8)}
    grid.variables["julian"] = {"data": np.empty((shape_3d[0], shape_3d[1] * shape_3d[2]), dtype=np.uint16)}

    for i_time, aux_date in enumerate(time_step_list):
        pandas_dates = add_local_dates(grid, aux_date, pandas=True)
        # Filling
        grid.variables["month"]["data"][i_time] = pandas_dates.dt.month
        grid.variables["weekday"]["data"][i_time] = pandas_dates.dt.dayofweek
        grid.variables["hour"]["data"][i_time] = pandas_dates.dt.hour
        grid.variables["julian"]["data"][i_time] = pandas_dates.dt.dayofyear
    # 2D (time, lat*lon) to 3D (time, lat, lon)
    grid.variables["month"]["data"] = grid.variables["month"]["data"].reshape(shape_3d)
    grid.variables["weekday"]["data"] = grid.variables["weekday"]["data"].reshape(shape_3d)
    grid.variables["hour"]["data"] = grid.variables["hour"]["data"].reshape(shape_3d)
    grid.variables["julian"]["data"] = grid.variables["julian"]["data"].reshape(shape_3d)

    # stop_increment("HERMES", "add_time_info_to_grid")
    record_time("HERMES", "add_time_info_to_grid", get_time_stamp() - st_time)

    return grid
