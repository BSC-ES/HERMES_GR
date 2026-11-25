#!/usr/bin/env python

# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

from nes import create_nes, open_netcdf
from hermes_gr.config import log_message
from hermes_gr.config import get_time_stamp, record_time
from hermes_gr.tools.country_codes import get_country_holiday_calendar, iso3_to_iso2, missing_iso3_codes
import hermes_gr
import os
import pytz
from datetime import timedelta
from datetime import timezone as dtz
import pandas as pd

BALANCED = True
COUNTRY_ISO3_VAR = "ISO"
COUNTRY_ISO2_VAR = "country_iso2"
UNKNOWN_COUNTRY_ISO2 = "UNK"


def _grid_configuration_arguments(options):
    """Select geographical arguments shared by NES creation and validation.

    Parameters
    ----------
    options : argparse.Namespace
        GR domain type and geographical configuration.

    Returns
    -------
    tuple[str, dict]
        NES projection name and unchanged geographical creation arguments.

    Raises
    ------
    NotImplementedError
        The requested GR domain type is unsupported.

    Notes
    -----
    No grid, coordinates or geometries are constructed. Time, vertical levels
    and MPI settings are excluded from the horizontal configuration comparison.
    """
    fields = {
        "global_monarch": ("inc_lat", "inc_lon"),
        "global": ("inc_lat", "inc_lon"),
        "regular": ("lat_orig", "lon_orig", "inc_lat", "inc_lon", "n_lat", "n_lon"),
        "rotated": ("centre_lat", "centre_lon", "west_boundary", "south_boundary", "inc_rlat", "inc_rlon"),
        "rotated_nested": ("parent_grid_path", "parent_ratio", "i_parent_start", "j_parent_start", "n_rlat", "n_rlon"),
        "lcc": ("lat_1", "lat_2", "lon_0", "lat_0", "nx", "ny", "inc_x", "inc_y", "x_0", "y_0"),
        "mercator": ("lat_ts", "lon_0", "nx", "ny", "inc_x", "inc_y", "x_0", "y_0"),
    }
    if options.domain_type not in fields:
        raise NotImplementedError(f"Unsupported grid type {options.domain_type!r}.")
    projection = "rotated-nested" if options.domain_type == "rotated_nested" else options.domain_type
    return projection, {name: getattr(options, name) for name in fields[options.domain_type]}


def _validate_grid_auxiliary(grid, projection, parameters, grid_path, grid_geo_path):
    """Collectively validate cached horizontal metadata and the paired file's presence.

    Parameters
    ----------
    grid : Nes
        Open cached grid on the current communicator.
    projection : str
        Requested NES projection.
    parameters : dict
        Same geographical arguments as new-grid creation.
    grid_path, grid_geo_path : str
        Paired auxiliary paths, included in actionable error messages.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        Installed NES lacks the canonical configuration API.
    ValueError
        NES detects a mismatch or the master cannot find the paired GeoJSON.

    Notes
    -----
    Every rank must participate. NES owns canonicalization, tolerances, legacy
    recovery and collective metadata comparison. Only the master's GeoJSON
    availability verdict is broadcast here. No reference grid is created and
    incompatible auxiliaries are never regenerated or overwritten.
    """
    if not callable(getattr(grid, "validate_grid_configuration", None)):
        raise RuntimeError("Installed NES lacks validate_grid_configuration(); update NES from the bundled source.")
    try:
        grid.validate_grid_configuration(projection=projection, **parameters)
    except (ValueError, NotImplementedError, AttributeError) as error:
        raise ValueError(
            f"Incompatible grid auxiliary {grid_path}: {error}. Files were not modified. "
            f"Use a matching auxiliary_files_path or explicitly replace both {grid_path} and {grid_geo_path}."
        ) from error
    error = None
    if grid.master and not os.path.isfile(grid_geo_path):
        error = (f"Missing paired grid GeoJSON: {grid_geo_path}. Files were not modified. "
                 f"Use a matching auxiliary_files_path or explicitly regenerate both {grid_path} and {grid_geo_path}.")
    error = grid.comm.bcast(error, root=0)
    if error is not None:
        raise ValueError(error)


def _validate_grid_geojson_debug(grid, grid_geo_path):
    """Compare the complete paired geometry with the accepted grid in DEBUG only.

    Parameters
    ----------
    grid : Nes
        Accepted grid with its normal in-memory shapefile already created.
    grid_geo_path : str
        Existing paired GeoJSON path.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        The paired file is unreadable or its FIDs, CRS or cell geometries differ.

    Notes
    -----
    Spatial partitions are gathered only in DEBUG; replicated time partitions
    use the master copy. All ranks receive the same error. Geometry tolerance
    is 1e-6 in the accepted shapefile CRS (NES geographic coordinates in degrees).
    GeoJSON may encode the geographic CRS differently (for example EPSG:4326
    instead of the NES sphere). Reproject it in memory to the accepted grid CRS
    before comparing cell coordinates. No auxiliary files are changed.
    """
    if not hermes_gr.DEBUG:
        return
    expected = grid.shapefile
    if grid.parallel_method in ("X", "Y") and grid.size > 1:
        parts = grid.comm.gather(expected, root=0)
        expected = pd.concat(parts) if grid.master else None
    error = None
    if grid.master:
        try:
            from geopandas import read_file
            actual = read_file(grid_geo_path)
            if "FID" in actual.columns:
                actual = actual.set_index("FID")
            if actual.crs is None or expected.crs is None:
                raise ValueError("GeoJSON or accepted grid has no CRS")
            if actual.crs != expected.crs:
                actual = actual.to_crs(expected.crs)
            if (actual.index.has_duplicates or len(actual) != len(expected)
                    or set(actual.index) != set(expected.index)):
                raise ValueError("GeoJSON cell count or FIDs differ from the accepted grid")
            actual = actual.reindex(expected.index)
            for fid, found, wanted in zip(expected.index, actual.geometry, expected.geometry):
                if (found is None or wanted is None
                        or not (found.equals(wanted) or found.equals_exact(wanted, tolerance=1e-6))):
                    raise ValueError(f"GeoJSON geometry differs for FID {fid}")
        except Exception as exception:
            error = (f"Invalid paired grid GeoJSON {grid_geo_path}: {exception}. Files were not modified. "
                     "Use a matching auxiliary_files_path or explicitly regenerate both grid.nc and grid.geojson.")
    error = grid.comm.bcast(error, root=0)
    if error is not None:
        raise ValueError(error)


def select_grid(comm, options, levels, time_step_list=None):
    """
    Select & create the destination grid.

    Reuse existing paired auxiliaries only after collective NES configuration
    validation. Create the grid normally when grid.nc is absent. Incompatible
    pairs are rejected without regeneration; geometry comparison is DEBUG-only.


    Parameters
    ----------
    comm : MPI.COMMUNICATOR
        MPI communicator.
    options : NameSpace
        Configuration options.
    levels : list[float]
        List of vertical levels.
    time_step_list : list[datetime], optional
        List of date times

    Returns
    -------
    Nes
        A NES grid object with the shapefile & cell_area calculated.

    Raises
    ------
    ValueError
        Cached grid metadata or the paired GeoJSON is incompatible or absent.
    RuntimeError
        The installed NES lacks the required canonical validation API.

    Notes
    -----
    Creation and validation share geographical arguments for every GR grid type.
    NES owns global axes, projection tolerances and legacy metadata recovery.
    The in-memory geometry required by normal GR processing is created only after
    metadata validation; no second/reference grid is created for comparison.
    Paired geometry is read and compared only when ``hermes_gr.DEBUG`` is enabled.
    """
    st_time = get_time_stamp()

    grid_path = os.path.join(options.auxiliary_files_path, "grid.nc")
    grid_geo_path = os.path.join(options.auxiliary_files_path, "grid.geojson")

    projection, parameters = _grid_configuration_arguments(options)
    # One shared branch decision avoids divergent collectives on different ranks.
    reuse_grid = comm.bcast(os.path.exists(grid_path) if comm.Get_rank() == 0 else None, root=0)
    if reuse_grid:
        log_message(f"Reading existing grid from {grid_path}", level=1)
        grid = open_netcdf(grid_path, comm=comm, balanced=BALANCED)
        try:
            _validate_grid_auxiliary(grid, projection, parameters, grid_path, grid_geo_path)
        except Exception:
            grid.close()
            raise
        grid.load()
        write_grid = False
    else:
        log_message(f"Creating {projection} grid with {parameters}", level=1)
        write_grid = True
        grid = create_nes(
            comm=comm, info=False, times=time_step_list,
            projection=projection, balanced=BALANCED, **parameters,
        )

    # Cell area
    if "cell_area" not in grid.cell_measures.keys():
        write_grid = True
        grid.calculate_grid_area()
        log_message("Grid: cell area created", level=3)

    # Reuse the normal processing geometry; DEBUG checks it before any writes.
    grid.create_shapefile()
    if reuse_grid:
        try:
            _validate_grid_geojson_debug(grid, grid_geo_path)
        except Exception:
            grid.close()
            raise

    # Shapefile
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

    write_grid = add_country_iso2_to_grid(grid, getattr(options, "countries_shapefile", None), write_grid=write_grid)

    # Write
    if write_grid:
        grid.to_netcdf(grid_path, serial=True)
        log_message(f"Grid: file created: {grid_path}", level=2)
    grid.comm.Barrier()

    if not reuse_grid and not os.path.exists(grid_geo_path):
        grid.write_shapefile(grid_geo_path)
        log_message(f"Grid: geostructure file created: {grid_geo_path}", level=2)

    log_message("Destination grid done!", level=2)

    grid.set_levels({"data": levels, "units": "m", "positive": "up"})

    grid = add_time_info_to_grid(
        grid=grid,
        time_step_list=time_step_list,
        holidays=getattr(options, "holidays", False),
    )

    grid.close()
    record_time("Initialization", "Grid", get_time_stamp() - st_time)
    return grid


def add_country_iso2_to_grid(grid, countries_shapefile, write_grid=False):
    """
    Add ISO 3166-1 alpha-2 country codes to the grid auxiliary file.

    HERMES_GR country masks are based on the ISO3 ``ISO`` column. Holiday
    calendars need ISO2 codes, so this derives ``country_iso2`` from the same
    country shapefile without requiring the deprecated ``world_info`` file.
    """
    if COUNTRY_ISO2_VAR in grid.variables.keys():
        grid.shapefile[COUNTRY_ISO2_VAR] = grid.variables[COUNTRY_ISO2_VAR]["data"].flatten()
        log_message("Grid: Country ISO2 codes read from grid.nc", level=3)
        return write_grid

    if countries_shapefile is None:
        log_message(
            "WARNING: No countries shapefile was defined. Grid country_iso2 variable will not be created.",
            level=7,
        )
        return write_grid

    log_message("Grid: Country ISO2 calculation starting", level=3)
    grid.spatial_join(countries_shapefile, method="intersection", var_list=[COUNTRY_ISO3_VAR], info=True)

    missing_codes = missing_iso3_codes(grid.shapefile[COUNTRY_ISO3_VAR].dropna().unique())
    if len(missing_codes) > 0:
        log_message(
            "WARNING: Country ISO3 codes without ISO2 mapping: {0}. These cells will use {1}.".format(
                ", ".join(sorted(missing_codes)),
                UNKNOWN_COUNTRY_ISO2,
            ),
            level=7,
        )

    country_iso2 = grid.shapefile[COUNTRY_ISO3_VAR].apply(
        lambda country_code: iso3_to_iso2(country_code) or UNKNOWN_COUNTRY_ISO2
    )
    grid.shapefile[COUNTRY_ISO2_VAR] = country_iso2
    grid.variables[COUNTRY_ISO2_VAR] = {
        "data": country_iso2.values.reshape((grid.lat["data"].shape[0], grid.lon["data"].shape[-1])),
        "dtype": str,
    }
    grid.set_strlen(32)
    log_message("Grid: Country ISO2 codes created", level=3)

    return True


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


def apply_national_holidays_to_weekday(pandas_dates, country_iso2, weekdays, holiday_cache, unsupported_countries):
    """
    Treat national holidays as Sundays in a per-cell weekday series.

    Parameters
    ----------
    pandas_dates : pandas.Series
        Local dates for the current timestep, one value per grid cell.
    country_iso2 : pandas.Series
        ISO2 country code for each grid cell.
    weekdays : pandas.Series
        Weekday values to update, where Monday=0 and Sunday=6.
    holiday_cache : dict
        Cache of holidays calendars.
    unsupported_countries : set
        Countries unsupported by the holidays package.

    Returns
    -------
    pandas.Series
        Weekday values where national holidays are set to 6.
    """
    weekdays = weekdays.copy()
    for country_code in country_iso2.dropna().unique():
        if country_code == UNKNOWN_COUNTRY_ISO2:
            continue

        country_mask = country_iso2 == country_code
        country_dates = pandas_dates.loc[country_mask]
        holiday_calendar = get_country_holiday_calendar(
            country_iso2=country_code,
            years=country_dates.dt.year.unique(),
            holiday_cache=holiday_cache,
            unsupported_countries=unsupported_countries,
        )
        if holiday_calendar is None:
            continue

        holiday_mask = country_dates.dt.date.isin(set(holiday_calendar.keys()))
        holiday_indexes = country_dates.index[holiday_mask]
        weekdays.loc[holiday_indexes] = 6

    return weekdays


def add_time_info_to_grid(grid, time_step_list, holidays=False):
    import numpy as np

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

    holiday_cache = {}
    unsupported_countries = set()
    if holidays:
        if COUNTRY_ISO2_VAR not in grid.variables.keys():
            raise ValueError(
                f"Cannot apply holidays without grid variable '{COUNTRY_ISO2_VAR}'. "
                "Check the 'countries_shapefile' option and regenerate grid.nc."
            )
        country_iso2 = pd.Series(grid.variables[COUNTRY_ISO2_VAR]["data"].flatten(), index=grid.shapefile.index)
    else:
        country_iso2 = None

    for i_time, aux_date in enumerate(time_step_list):
        pandas_dates = add_local_dates(grid, aux_date, pandas=True)
        weekdays = pandas_dates.dt.dayofweek
        if holidays:
            weekdays = apply_national_holidays_to_weekday(
                pandas_dates=pandas_dates,
                country_iso2=country_iso2,
                weekdays=weekdays,
                holiday_cache=holiday_cache,
                unsupported_countries=unsupported_countries,
            )
        # Filling
        grid.variables["month"]["data"][i_time] = pandas_dates.dt.month
        grid.variables["weekday"]["data"][i_time] = weekdays
        grid.variables["hour"]["data"][i_time] = pandas_dates.dt.hour
        grid.variables["julian"]["data"][i_time] = pandas_dates.dt.dayofyear

    if holidays and len(unsupported_countries) > 0:
        log_message(
            "WARNING: Holidays package does not support national calendars for ISO2 countries: {0}. "
            "Those cells kept their calendar weekday.".format(", ".join(sorted(unsupported_countries))),
            level=7,
        )

    # 2D (time, lat*lon) to 3D (time, lat, lon)
    grid.variables["month"]["data"] = grid.variables["month"]["data"].reshape(shape_3d)
    grid.variables["weekday"]["data"] = grid.variables["weekday"]["data"].reshape(shape_3d)
    grid.variables["hour"]["data"] = grid.variables["hour"]["data"].reshape(shape_3d)
    grid.variables["julian"]["data"] = grid.variables["julian"]["data"].reshape(shape_3d)

    return grid
