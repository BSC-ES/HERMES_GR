# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

from dataclasses import dataclass
from datetime import datetime

import geopandas as gpd
import numpy as np
from shapely.geometry import box

import nes

EXTENSIONS = [".cpg", ".dbf", ".prj", ".shx", ".shp", ".geojson"]


def _make_nan_data(data, nan_fraction=0.05, seed=42):
    """Return a copy of *data* with *nan_fraction* of values set to NaN.

    Uses a fixed *seed* so results are deterministic across runs.
    """
    rng = np.random.default_rng(seed)
    out = data.copy()
    mask = rng.random(out.shape) < nan_fraction
    out[mask] = np.nan
    return out


@dataclass
class ProjectionParameters:
    # Grid details for CAMS-REG-v5.1
    REGULAR = {
        "grid_mapping_name": "latitude_longitude",
        "projection": "regular",
        "lat_orig": 10,
        "lon_orig": 10,
        "inc_lat": 0.1,
        "inc_lon": 0.1,
        "n_lat": 100,
        "n_lon": 100,
        "times": [datetime(2020, 1, 1)],
        "levels": [0],
    }
    # Grid details for CAMS-REG-v5.1
    ROTATED = {
        "grid_mapping_name": "rotated_latitude_longitude",
        "projection": "rotated",
        "centre_lat": 10,
        "centre_lon": 10,
        "west_boundary": -10,
        "south_boundary": -10,
        "inc_rlat": 0.1,
        "inc_rlon": 0.1,
        "times": [datetime(2020, 1, 1)],
        "levels": [0],
    }

    LCC = {
        "grid_mapping_name": "lambert_conformal_conic",
        "lat_1": 37,
        "lat_2": 43,
        "lon_0": -3,
        "lat_0": 40,
        "nx": 100,
        "ny": 100,
        "inc_x": 4000,
        "inc_y": 4000,
        "x_0": -807847.688,
        "y_0": -797137.125,
        "times": [datetime(2020, 1, 1)],
        "levels": [0],
    }

    MERCATOR = {
        "grid_mapping_name": "mercator",
        "lat_ts": -1.5,
        "lon_0": -18,
        "nx": 210,
        "ny": 236,
        "inc_x": 50000,
        "inc_y": 50000,
        "x_0": -126017.5,
        "y_0": -5407460.0,
        "times": [datetime(2020, 1, 1)],
        "levels": [0],
    }

    GLOBAL = {
        "projection": "global",
        "grid_mapping_name": "latitude_longitude",
        "inc_lat": 1,
        "inc_lon": 1,
        "lat_orig": -90,
        "lon_orig": -180,
        "times": [datetime(2020, 1, 1)],
        "levels": [0],
    }

    _n_stations = 5
    POINTS = {
        "projection": None,
        "n_stations": _n_stations,
        "lat": np.random.rand(_n_stations) * 180 - 90,
        "lon": np.random.rand(_n_stations) * 360 - 180,
        "times": [datetime(2020, 1, 1), datetime(2020, 2, 1), datetime(2020, 3, 1)],
    }


@dataclass
class SampleData:
    REGULAR = {
        "data": np.random.rand(
            len(ProjectionParameters.REGULAR["times"]),
            len(ProjectionParameters.REGULAR["levels"]),
            ProjectionParameters.REGULAR["n_lat"],
            ProjectionParameters.REGULAR["n_lon"],
        ),
        "dimensions": ("time", "lev", "lat", "lon"),
        "units": "kg/m2/s",
    }

    # Calculate rotated grid dimensions from projection parameters
    _n_rlat = (
        int(
            (-ProjectionParameters.ROTATED["south_boundary"] - ProjectionParameters.ROTATED["south_boundary"])
            / ProjectionParameters.ROTATED["inc_rlat"]
        )
        + 1
    )
    _n_rlon = (
        int(
            (-ProjectionParameters.ROTATED["west_boundary"] - ProjectionParameters.ROTATED["west_boundary"])
            / ProjectionParameters.ROTATED["inc_rlon"]
        )
        + 1
    )

    ROTATED = {
        "data": np.random.rand(
            len(ProjectionParameters.ROTATED["times"]),
            len(ProjectionParameters.ROTATED["levels"]),
            _n_rlat,
            _n_rlon,
        ),
        "dimensions": ("time", "lev", "rlat", "rlon"),
        "units": "kg/m2/s",
    }

    LCC = {
        "data": np.random.rand(
            len(ProjectionParameters.LCC["times"]),
            len(ProjectionParameters.LCC["levels"]),
            ProjectionParameters.LCC["ny"],
            ProjectionParameters.LCC["nx"],
        ),
        "dimensions": ("time", "lev", "y", "x"),
        "units": "kg/m2/s",
    }

    MERCATOR = {
        "data": np.random.rand(
            len(ProjectionParameters.MERCATOR["times"]),
            len(ProjectionParameters.MERCATOR["levels"]),
            ProjectionParameters.MERCATOR["ny"],
            ProjectionParameters.MERCATOR["nx"],
        ),
        "dimensions": ("time", "lev", "y", "x"),
        "units": "kg/m2/s",
    }

    GLOBAL = {
        "data": np.random.rand(
            len(ProjectionParameters.GLOBAL["times"]),
            len(ProjectionParameters.GLOBAL["levels"]),
            int(180 // np.float64(ProjectionParameters.GLOBAL["inc_lat"])),
            int(360 // np.float64(ProjectionParameters.GLOBAL["inc_lon"])),
        ),
        "dimensions": ("time", "lev", "lat", "lon"),
        "units": "kg/m2/s",
    }

    POINTS = {
        "var1": {
            "data": np.random.rand(
                len(ProjectionParameters.POINTS["times"]),
                ProjectionParameters.POINTS["n_stations"],
            ),
            "dimensions": ("time", "station"),
            "units": "kg/m2/s",
        },
        "stations": {
            "data": np.array(["Station_A", "Station_B", "Station_C", "Station_D", "Station_E"]),
            "dimensions": ("station",),
        },
    }


@dataclass
class DummyData:
    DIMS = {"lon": 10, "lat": 10, "lev": 1, "time": 1}
    TYPES = {
        "lon": "f4",
        "lat": "f4",
        "lev": "i4",
        "time": "i4",
        "variable": "f4",
    }
    UNITS = {
        "lon": "degrees_east",
        "lat": "degrees_north",
        "lev": "lev_units",
        "time": "hours since 1970-01-01 00:00:00",
        "variable": "variable_units",
    }
    DATA_SHAPE = (DIMS["time"], DIMS["lev"], DIMS["lat"], DIMS["lon"])

    # Fix this
    _ntimes = 1
    _nlevels = 1
    _nlat = 10
    _nlon = 10
    ExpandContract = {
        "projection": "regular",
        "lat_orig": 10,
        "lon_orig": 10,
        "inc_lat": 0.1,
        "inc_lon": 0.1,
        "n_lat": _nlat,
        "n_lon": _nlon,
        # Generate times as datetime objects for 6 consecutive days starting from 2020-01-01
        "times": [datetime(2020, 1, 1) + np.timedelta64(i, "D") for i in range(_ntimes)],
        "var1": {
            "data": np.arange(_nlevels, _ntimes * (_nlat * _nlon) + 1, dtype=np.float64).reshape(
                _ntimes, _nlevels, _nlat, _nlon
            ),
            "dimensions": ("time", "lev", "lat", "lon"),
            "units": "kg/m2/s",
        },
    }
    Selecting = {
        "projection": "regular",
        "lat_orig": 10,
        "lon_orig": 10,
        "inc_lat": 0.1,
        "inc_lon": 0.1,
        "n_lat": _nlat,
        "n_lon": _nlon,
        "times": [datetime(2020, 1, 1) + np.timedelta64(i, "D") for i in range(_ntimes)],
        "var1": {
            "data": np.arange(_nlevels, _ntimes * (_nlat * _nlon) + 1, dtype=np.float64).reshape(
                _ntimes, _nlevels, _nlat, _nlon
            ),
            "dimensions": ("time", "lev", "lat", "lon"),
            "units": "kg/m2/s",
        },
    }
    SpatialJoin = {
        "projection": "regular",
        "lat_orig": 10,
        "lon_orig": 10,
        "inc_lat": 0.1,
        "inc_lon": 0.1,
        "n_lat": _nlat,
        "n_lon": _nlon,
        "times": [datetime(2020, 1, 1) + np.timedelta64(i, "D") for i in range(_ntimes)],
        "var1": {
            "data": np.arange(_nlevels, _ntimes * (_nlat * _nlon) + 1, dtype=np.float64).reshape(
                _ntimes, _nlevels, _nlat, _nlon
            ),
            "dimensions": ("time", "lev", "lat", "lon"),
            "units": "kg/m2/s",
        },
    }
    GridArea = {
        "projection": "regular",
        "lat_orig": 10,
        "lon_orig": 10,
        "inc_lat": 0.1,
        "inc_lon": 0.1,
        "n_lat": 2,
        "n_lon": 2,
        "times": [datetime(2020, 1, 1) + np.timedelta64(i, "D") for i in range(1)],
        "var1": {
            "data": np.arange(_nlevels, 1 * (2 * 2) + 1, dtype=np.float64).reshape(1, _nlevels, 2, 2),
            "dimensions": ("time", "lev", "lat", "lon"),
            "units": "kg/m2/s",
        },
    }
    LongitudeConversion = {
        "projection": "regular",
        "lat_orig": -90,
        "lon_orig": 0,
        "inc_lat": 18,
        "inc_lon": 36,
        "n_lat": _nlat,
        "n_lon": _nlon,
        "times": [datetime(2020, 1, 1) + np.timedelta64(i, "D") for i in range(_ntimes)],
        "var1": {
            "data": np.arange(_nlevels, _ntimes * (_nlat * _nlon) + 1, dtype=np.float64).reshape(
                _ntimes, _nlevels, _nlat, _nlon
            ),
            "dimensions": ("time", "lev", "lat", "lon"),
            "units": "kg/m2/s",
        },
    }
    _nlevels_vert = 6
    VerticalInterp = {
        "projection": "regular",
        "lat_orig": 10,
        "lon_orig": 10,
        "inc_lat": 0.1,
        "inc_lon": 0.1,
        "n_lat": _nlat,
        "n_lon": _nlon,
        "times": [datetime(2020, 1, 1) + np.timedelta64(i, "D") for i in range(_ntimes)],
        "var1": {
            "data": np.arange(
                _nlevels_vert,
                _ntimes * (_nlevels_vert * _nlat * _nlon) + _nlevels_vert,
                dtype=np.float64,
            ).reshape(_ntimes, _nlevels_vert, _nlat, _nlon),
            "dimensions": ("time", "lev", "lat", "lon"),
            "units": "kg/m2/s",
        },
        "levels": np.arange(1, _nlevels_vert + 1),
        "interpolation_shorter": np.array([1, 2.5, 5], dtype=np.float64),
        "interpolation_longer": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float64),
        "interpolation_methods": [
            "linear",
            "nearest",
            "nearest-up",
            "zero",
            "slinear",
            "quadratic",
            "cubic",
            "previous",
            "next",
        ],
        "extrapolate_options": [
            # Bool options==
            True,  # Both bounds extrapolated
            False,  # Bottom/top fill with nearest value
            None,  # Same as False: ("bottom", "top")
            # Numeric options==
            0,  # Fill both bounds with 0
            np.nan,  # Fill both bounds with NaN
            -999,  # Fill both bounds with -999
            # Tuple with bool options==
            (True, True),  # Both bounds extrapolated
            (True, False),  # Extrapolate below, top fill with nearest
            (False, True),  # Bottom fill with nearest, extrapolate above
            (False, False),  # Both fill with nearest value
            # Tuple with None==
            (None, None),  # Both bounds fill with NaN
            (True, None),  # Extrapolate below, NaN above
            (None, True),  # NaN below, extrapolate above
            # Tuple with numeric values==
            (0, 0),  # Fill both bounds with 0
            (np.nan, np.nan),  # Fill both bounds with NaN
            (-999, -999),  # Fill both bounds with -999
            # Tuple with mixed types==
            (True, 0),  # Extrapolate below, fill 0 above
            (0, True),  # Fill 0 below, extrapolate above
            (True, np.nan),  # Extrapolate below, NaN above
            (np.nan, True),  # NaN below, extrapolate above
            (False, np.nan),  # Bottom nearest, NaN above
            (np.nan, False),  # NaN below, top nearest
            (False, 0),  # Bottom nearest, fill 0 above
            (0, False),  # Fill 0 below, top nearest
        ],
    }
    _ntimes_horiz = 1
    _nlat_horiz = 2
    _nlon_horiz = 2
    HorizontalInterp = {
        "NetCDF_base": {
            "projection": "regular",
            "lat_orig": 10,
            "lon_orig": 10,
            "inc_lat": 0.1,
            "inc_lon": 0.1,
            "n_lat": _nlat_horiz,
            "n_lon": _nlon_horiz,
            "times": [datetime(2020, 1, 1) + np.timedelta64(i, "D") for i in range(_ntimes_horiz)],
            "var1": {
                "data": np.arange(
                    _nlevels,
                    _ntimes_horiz * (_nlat_horiz * _nlon_horiz) + 1,
                    dtype=np.float64,
                ).reshape(_ntimes_horiz, _nlevels, _nlat_horiz, _nlon_horiz),
                "dimensions": ("time", "lev", "lat", "lon"),
                "units": "kg/m2/s",
            },
        },
        "NetCDF_no_overlap": {
            "projection": "regular",
            "lat_orig": 0,
            "lon_orig": 0,
            "inc_lat": 0.08,
            "inc_lon": 0.08,
            "n_lat": _nlat_horiz,
            "n_lon": _nlon_horiz,
            "times": [datetime(2020, 1, 1) + np.timedelta64(i, "D") for i in range(_ntimes_horiz)],
            "var1": {
                "data": np.arange(
                    _nlevels,
                    _ntimes_horiz * (_nlat_horiz * _nlon_horiz) + 1,
                    dtype=np.float64,
                ).reshape(_ntimes_horiz, _nlevels, _nlat_horiz, _nlon_horiz)
                * 2,
                "dimensions": ("time", "lev", "lat", "lon"),
                "units": "kg/m2/s",
            },
        },
        "NetCDF_overlap": {
            "projection": "regular",
            "lat_orig": 10,
            "lon_orig": 10,
            "inc_lat": 1,
            "inc_lon": 1,
            "n_lat": _nlat_horiz,
            "n_lon": _nlon_horiz,
            "times": [datetime(2020, 1, 1) + np.timedelta64(i, "D") for i in range(_ntimes_horiz)],
            "var1": {
                "data": np.arange(
                    _nlevels,
                    _ntimes_horiz * (_nlat_horiz * _nlon_horiz) + 1,
                    dtype=np.float64,
                ).reshape(_ntimes_horiz, _nlevels, _nlat_horiz, _nlon_horiz)
                * 200,
                "dimensions": ("time", "lev", "lat", "lon"),
                "units": "kg/m2/s",
            },
        },
        "interpolation_methods": ["NN", "Conservative"],
    }
    _ndays_mean = 3
    _ntimes_mean = _ndays_mean * 24
    # plus-one dataset: 25 hourly steps (one full day plus one additional hour)
    _ntimes_25 = 25
    _ntimes_rolling = 10
    DailyStatistics = {
        "no_nan": {
            "projection": "regular",
            "lat_orig": 10,
            "lon_orig": 10,
            "inc_lat": 0.1,
            "inc_lon": 0.1,
            "n_lat": _nlat,
            "n_lon": _nlon,
            # Generate hourly times for 3 consecutive days starting from 2020-01-01
            "times": [datetime(2020, 1, 1) + np.timedelta64(i, "h") for i in range(_ntimes_mean)],
            "var1": {
                "data": np.arange(_nlevels, _ntimes_mean * (_nlat * _nlon) + 1, dtype=np.float64).reshape(
                    _ntimes_mean, _nlevels, _nlat, _nlon
                ),
                "dimensions": ("time", "lev", "lat", "lon"),
                "units": "kg/m2/s",
            },
        },
        "nan": {
            "projection": "regular",
            "lat_orig": 10,
            "lon_orig": 10,
            "inc_lat": 0.1,
            "inc_lon": 0.1,
            "n_lat": _nlat,
            "n_lon": _nlon,
            "times": [datetime(2020, 1, 1) + np.timedelta64(i, "h") for i in range(_ntimes_mean)],
            "var1": {
                "data": _make_nan_data(
                    np.arange(_nlevels, _ntimes_mean * (_nlat * _nlon) + 1, dtype=np.float64).reshape(
                        _ntimes_mean, _nlevels, _nlat, _nlon
                    ),
                    nan_fraction=0.05,
                    seed=42,
                ),
                "dimensions": ("time", "lev", "lat", "lon"),
                "units": "kg/m2/s",
            },
        },
        "no_nan_25": {
            "projection": "regular",
            "lat_orig": 10,
            "lon_orig": 10,
            "inc_lat": 0.1,
            "inc_lon": 0.1,
            "n_lat": _nlat,
            "n_lon": _nlon,
            # Generate 25 hourly times starting from 2020-01-01
            "times": [datetime(2020, 1, 1) + np.timedelta64(i, "h") for i in range(_ntimes_25)],
            "var1": {
                "data": np.arange(_nlevels, _ntimes_25 * (_nlat * _nlon) + 1, dtype=np.float64).reshape(
                    _ntimes_25, _nlevels, _nlat, _nlon
                ),
                "dimensions": ("time", "lev", "lat", "lon"),
                "units": "kg/m2/s",
            },
        },
        "nan_25": {
            "projection": "regular",
            "lat_orig": 10,
            "lon_orig": 10,
            "inc_lat": 0.1,
            "inc_lon": 0.1,
            "n_lat": _nlat,
            "n_lon": _nlon,
            "times": [datetime(2020, 1, 1) + np.timedelta64(i, "h") for i in range(_ntimes_25)],
            "var1": {
                "data": _make_nan_data(
                    np.arange(_nlevels, _ntimes_25 * (_nlat * _nlon) + 1, dtype=np.float64).reshape(
                        _ntimes_25, _nlevels, _nlat, _nlon
                    ),
                    nan_fraction=0.05,
                    seed=42,
                ),
                "dimensions": ("time", "lev", "lat", "lon"),
                "units": "kg/m2/s",
            },
        },
    }
    _nhours_rm = 12
    RollingMean = {
        "projection": "regular",
        "lat_orig": 10,
        "lon_orig": 10,
        "inc_lat": 0.1,
        "inc_lon": 0.1,
        "n_lat": _nlat,
        "n_lon": _nlon,
        # Generate hourly times for 12 hours starting from 2020-01-01
        "times": [datetime(2020, 1, 1) + np.timedelta64(i, "h") for i in range(_nhours_rm)],
        "var1": {
            "data": np.arange(_nlevels, _nhours_rm * (_nlat * _nlon) + 1, dtype=np.float64).reshape(
                _nhours_rm, _nlevels, _nlat, _nlon
            ),
            "dimensions": ("time", "lev", "lat", "lon"),
            "units": "kg/m2/s",
        },
    }
    Sum = {
        "projection": "regular",
        "lat_orig": 10,
        "lon_orig": 10,
        "inc_lat": 0.1,
        "inc_lon": 0.1,
        "n_lat": _nlat,
        "n_lon": _nlon,
        "times": [datetime(2020, 1, 1)],
        "var1": {
            "data": np.ones((1, 1, _nlat, _nlon), dtype=np.float64),
            "dimensions": ("time", "lev", "lat", "lon"),
            "units": "kg/m2/s",
        },
        "var2": {
            "data": np.ones((1, 1, _nlat, _nlon), dtype=np.float64) * 2,
            "dimensions": ("time", "lev", "lat", "lon"),
            "units": "kg/m2/s",
        },
    }
    n_hours_timestep = 24
    WriteTimestep = {
        "projection": "regular",
        "lat_orig": 10,
        "lon_orig": 10,
        "inc_lat": 0.1,
        "inc_lon": 0.1,
        "n_lat": _nlat,
        "n_lon": _nlon,
        "var1": {
            "data": None,  # To be filled in setup fixture
            "units": "kg/m2/s",
            "dtype": np.float32,
        },
    }


@dataclass
class ExpectedData:
    # earth_radius_dict = {"WGS84": [6356752.3142, 6378137.0]}
    # GridArea = {"cell_area": np.array([[1.21628552e08, 1.21628552e08], [1.21590936e08, 1.21590936e08]])}
    # earth_radius_dict = {"WGS84": [6371000.0, 6371000.0]}
    GridArea = {"cell_area": np.array([[1.21745930e+08, 1.21745930e+08], [1.21708086e+08, 1.21708086e+08]])}
    Selecting = {
        "lat": {
            "data": np.array([10.05, 10.15, 10.25, 10.35, 10.45, 10.55, 10.65, 10.75, 10.85, 10.95]),
            "lower_bound": DummyData.ExpandContract["lat_orig"],
            "upper_bound": DummyData.ExpandContract["lat_orig"] + 1,  # 1 is offset
        },
        "lon": {
            "data": np.array([10.05, 10.15, 10.25, 10.35, 10.45, 10.55, 10.65, 10.75, 10.85, 10.95]),
            "lower_bound": DummyData.ExpandContract["lon_orig"],
            "upper_bound": DummyData.ExpandContract["lon_orig"] + 1,
        },
    }
    LongitudeConversion = {"lon": np.array([-162.0, -126.0, -90.0, -54.0, -18.0, 18.0, 54.0, 90.0, 126.0, 162.0])}
    HorizontalInterp = {
        "NN": np.array([[2.49631853, 2.49629846], [2.49630888, 2.49628856]]),
        "NN_wm": {
            "weight": np.array(
                [
                    [
                        [6.39935399e-07, 6.42488474e-07],
                        [6.42494008e-07, 6.45077948e-07],
                    ],
                    [
                        [6.36799779e-07, 6.39314579e-07],
                        [6.39345991e-07, 6.41891194e-07],
                    ],
                    [
                        [6.36758541e-07, 6.39298750e-07],
                        [6.39279089e-07, 6.41849700e-07],
                    ],
                    [
                        [6.33670123e-07, 6.36172626e-07],
                        [6.36178837e-07, 6.38711295e-07],
                    ],
                ]
            ),
            "idx": np.array([[[0, 0], [0, 0]], [[2, 2], [2, 2]], [[1, 1], [1, 1]], [[3, 3], [3, 3]]]),
        },
        "Conservative": np.array([[10.0, 0.0], [0.0, 0.0]]),
    }


def create_synthetic_spatial_files(shp_path, geojson_path):
    """
    Create synthetic shapefile (.shp) and GeoJSON (.geojson) files with polygon
    geometries covering the SpatialJoin grid area (lat 10-11, lon 10-11).

    The grid is split into 4 quadrant polygons, each labelled with a synthetic
    ISO code, so that spatial join methods (centroid, nearest, intersection) can
    be tested deterministically.

    Quadrant layout (looking at the grid from above):
        +---------+---------+
        |  REG_C  |  REG_D  |   lat 10.5 – 11.0
        +---------+---------+
        |  REG_A  |  REG_B  |   lat 10.0 – 10.5
        +---------+---------+
        lon 10.0  10.5  11.0

    Parameters
    ----------
    shp_path : str
        Output path for the ESRI Shapefile (.shp).
    geojson_path : str
        Output path for the GeoJSON file (.geojson).
    """
    polygons = [
        box(10.0, 10.0, 10.5, 10.5),  # REG_A – bottom-left
        box(10.5, 10.0, 11.0, 10.5),  # REG_B – bottom-right
        box(10.0, 10.5, 10.5, 11.0),  # REG_C – top-left
        box(10.5, 10.5, 11.0, 11.0),  # REG_D – top-right
    ]
    iso_codes = ["REG_A", "REG_B", "REG_C", "REG_D"]

    gdf = gpd.GeoDataFrame(
        {"ISO": iso_codes},
        geometry=polygons,
        crs="EPSG:4326",
    )

    # Write both formats
    gdf.to_file(shp_path)
    gdf.to_file(geojson_path, driver="GeoJSON")

    return gdf


def load_fresh_nes(filename):
    """Load and return a fresh NES object from the test file."""
    nessy = nes.open_netcdf(filename)
    nessy.load()
    return nessy


def compute_expected_rolling_mean(data, hours):
    """Mirror the rolling_mean implementation to produce expected values.

    For each timestep *t* the implementation computes:
        prev_t = t - (hours - 1)
        result[t] = mean(data[prev_t:t])   if prev_t >= 0
                   = NaN                    otherwise
    """
    n_times = data.shape[0]
    expected = np.full_like(data, np.nan, dtype=np.float64)
    for t in range(n_times):
        prev_t = t - (hours - 1)
        if prev_t >= 0:
            expected[t] = data[prev_t:t].mean(axis=0)
    return expected
