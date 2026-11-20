# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

"""Shared helpers for the synthetic conservative interpolation scripts (test_*_synthetic.py) of this folder.

Every script builds small regular lat-lon grids and proxy files in a temporary folder, runs the interpolation
with any number of MPI processes, gathers the result on rank 0 through a netCDF file and checks it there.

check_flux_and_mass() is the common conservation check. A source field v is interpolated twice:
  - flux=True with v as a flux (per unit area): its emissions per destination cell are F * A_dst.
  - flux=False with the source mass v * A_src.
Both must add up to the source mass sum(v * A_src) for every time step and level, and they must agree cell by
cell (the flux weights are the mass weights times A_src / A_dst, so F * A_dst == M).
"""

import json
import os
import tempfile
from datetime import datetime
from itertools import count

import numpy as np
import rasterio
from mpi4py import MPI
from netCDF4 import Dataset

import nes

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()
CRS84 = {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}}
# Tolerance of the hand-computed shares (fractions of a source cell sent to a destination cell)
SHARE_RTOL = 1e-8
# Tolerance of the mass and emission totals (float64 data)
TOTAL_RTOL = 1e-10
# Earth radius
EARTH_RADIUS = 6371000.0

_TMP = None
_RUN_ID = count()


def tmp_dir():
    """Temporary folder shared by all the ranks (created once per run)."""
    global _TMP
    if _TMP is None:
        _TMP = COMM.bcast(tempfile.mkdtemp(prefix="nes_synthetic_") if RANK == 0 else None, root=0)
    return _TMP


def tmp_path(name):
    return os.path.join(tmp_dir(), name)


def feature(weight, geom_type, coords, column="weight"):
    return {"type": "Feature", "properties": {column: weight}, "geometry": {"type": geom_type, "coordinates": coords}}


def rect(x0, y0, x1, y1):
    return [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]


def write_geojson(name, features):
    """Write a CRS84 GeoJSON on rank 0 and return its path (collective)."""
    path = tmp_path(name)
    if RANK == 0:
        with open(path, "w") as f:
            json.dump({"type": "FeatureCollection", "crs": CRS84, "features": features}, f)
    COMM.Barrier()
    return path


def write_raster(name, data, transform, crs="EPSG:4326", nodata=None):
    """Write a single band GeoTIFF on rank 0 and return its path (collective)."""
    path = tmp_path(name)
    if RANK == 0:
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype.name,
            crs=crs,
            transform=transform,
            nodata=nodata,
        ) as dst_r:
            dst_r.write(data, 1)
    COMM.Barrier()
    return path


def make_grid(
    inc, n, lat0=0.0, lon0=0.0, comm=None, parallel_method="Y", n_times=1, n_levels=1, coord_dtype=np.float64
):
    """Regular n x n lat-lon grid of inc degree cells with its shapefile, its coordinates stored as coord_dtype."""
    times = [datetime(2000, 1, 1, hour) for hour in range(n_times)]
    grid = nes.create_nes(
        comm=comm,
        info=False,
        projection="regular",
        parallel_method=parallel_method,
        times=times,
        lat_orig=lat0,
        lon_orig=lon0,
        inc_lat=inc,
        inc_lon=inc,
        n_lat=n,
        n_lon=n,
    )
    if n_levels > 1:
        grid.set_levels({"data": np.arange(n_levels), "units": "", "dimensions": ("lev",), "positive": "up"})
    if np.dtype(coord_dtype) != np.float64:
        # Mimic a grid read from a file with lower precision coordinates (e.g. float32 CAMS-REG bounds)
        grid.create_spatial_bounds()
        for attr in (
            "lat",
            "lon",
            "lat_bnds",
            "lon_bnds",
            "_full_lat",
            "_full_lon",
            "_full_lat_bnds",
            "_full_lon_bnds",
        ):
            coordinate = getattr(grid, attr)
            if coordinate is not None and coordinate.get("data") is not None:
                coordinate["data"] = np.asarray(coordinate["data"], dtype=coord_dtype)
    grid.create_shapefile()
    return grid


def cell_areas(inc, n, lat0=0.0, lon0=0.0):
    """Cell areas (as used by the flux conversion) of the complete grid, as (lat, lon)."""
    grid = make_grid(inc, n, lat0=lat0, lon0=lon0, comm=MPI.COMM_SELF)
    lat_edges = np.radians(lat0 + np.arange(n + 1) * inc)
    areas = EARTH_RADIUS**2 * np.radians(inc) * np.diff(np.sin(lat_edges))
    areas = np.repeat(areas[:, None], n, axis=1)
    # The shapefile is ordered by lat (outer) and lon (inner): check it with the cell bounds
    bounds = grid.shapefile.geometry.bounds
    cent_x = ((bounds["minx"] + bounds["maxx"]) / 2).values.reshape(n, n)
    cent_y = ((bounds["miny"] + bounds["maxy"]) / 2).values.reshape(n, n)
    assert np.allclose(cent_y[:, 0], grid.lat["data"]) and np.allclose(cent_x[0, :], grid.lon["data"])
    return areas.reshape(n, n)


def plain_area_shares(src=(1.0, 4), dst=(0.5, 8), lat0=0.0):
    """Analytic spherical-area fractions for regular destination cells nested in source cells."""
    ratio = round(src[0] / dst[0])
    return cell_areas(*dst, lat0=lat0) / parent_field(cell_areas(*src, lat0=lat0), ratio)


def as_4d(values):
    values = np.asarray(values, dtype=np.float64)
    return values[None, None] if values.ndim == 2 else values


def local_part(grid, values):
    """Rank-local (time, lev, lat, lon) part of a complete source field."""
    lim = grid.read_axis_limits
    return np.ascontiguousarray(
        values[lim["t_min"]:lim["t_max"], :, lim["y_min"]:lim["y_max"], lim["x_min"]:lim["x_max"]]
    )


def interpolate(
    values,
    src=(1.0, 4),
    dst=(0.5, 8),
    lat0=0.0,
    parallel_method="Y",
    data_dtype=np.float64,
    src_coord_dtype=np.float64,
    dst_coord_dtype=np.float64,
    **kwargs,
):
    """
    Interpolate a complete source field with every rank and gather the complete result on rank 0.

    Parameters
    ----------
    values : numpy.ndarray
        Complete source field, (lat, lon) or (time, lev, lat, lon).
    src, dst : tuple
        (increment, number of cells per side) of the source and destination grids.
    data_dtype : numpy.dtype
        Dtype of the source variable.
    src_coord_dtype, dst_coord_dtype : numpy.dtype
        Dtype of the coordinates of the source and destination grids.
    kwargs
        Passed to interpolate_horizontal (kind defaults to "Conservative").

    Returns
    -------
    numpy.ndarray, None
        Complete (time, lev, lat, lon) result on rank 0, None on the other ranks.
    """
    values = as_4d(values)
    n_times, n_levels = values.shape[:2]
    src_grid = make_grid(
        *src,
        lat0=lat0,
        parallel_method=parallel_method,
        n_times=n_times,
        n_levels=n_levels,
        coord_dtype=src_coord_dtype,
    )
    dst_grid = make_grid(
        *dst,
        lat0=lat0,
        parallel_method=parallel_method,
        n_times=n_times,
        n_levels=n_levels,
        coord_dtype=dst_coord_dtype,
    )
    src_grid.variables["var"] = {
        "data": local_part(src_grid, values).astype(data_dtype),
        "dtype": np.dtype(data_dtype),
        "units": "-",
    }
    kwargs.setdefault("kind", "Conservative")
    # Keep a reference to the output: Nes.__del__ empties the variables of a garbage collected object
    out = src_grid.interpolate_horizontal(dst_grid=dst_grid, info=False, **kwargs)
    out_nc = tmp_path(f"out_{next(_RUN_ID)}.nc")
    out.to_netcdf(out_nc)
    COMM.Barrier()
    result = None
    if RANK == 0:
        with Dataset(out_nc) as ds:
            ds.set_auto_mask(False)
            result = np.array(ds.variables["var"][:])
        assert result.shape[:2] == (n_times, n_levels), f"unexpected output shape {result.shape}"
    return result


def parent_field(values, ratio):
    """Repeat every source cell value on the ratio x ratio destination cells nested in it."""
    return np.repeat(np.repeat(values, ratio, axis=-2), ratio, axis=-1)


def check_flux_and_mass(values, label, src=(1.0, 4), dst=(0.5, 8), lat0=0.0, total_rtol=TOTAL_RTOL, **kwargs):
    """
    Interpolate values as a flux and its mass as a non-flux field, and check the conservation of both.

    Returns
    -------
    tuple, None
        On rank 0, (mass, flux, src_mass): the no-flux output, the flux output and the source mass, all
        (time, lev, lat, lon). None on the other ranks.
    """
    values = as_4d(values)
    a_src = cell_areas(*src, lat0=lat0)
    a_dst = cell_areas(*dst, lat0=lat0)
    src_mass = values * a_src
    flux_out = interpolate(values, src=src, dst=dst, lat0=lat0, flux=True, **kwargs)
    mass_out = interpolate(src_mass, src=src, dst=dst, lat0=lat0, flux=False, **kwargs)
    if RANK != 0:
        return None
    expected = np.nansum(src_mass, axis=(2, 3))
    mass_total = np.nansum(mass_out, axis=(2, 3))
    emissions = flux_out * a_dst
    emissions_total = np.nansum(emissions, axis=(2, 3))
    assert np.allclose(mass_total, expected, rtol=total_rtol, atol=0.0), (
        f"[{label}] no-flux mass not conserved: {mass_total} != {expected}"
    )
    assert np.allclose(emissions_total, expected, rtol=total_rtol, atol=0.0), (
        f"[{label}] flux emissions not conserved: {emissions_total} != {expected}"
    )
    atol = total_rtol * np.nanmax(np.abs(mass_out))
    assert np.allclose(emissions, mass_out, rtol=total_rtol, atol=atol, equal_nan=True), (
        f"[{label}] flux * A_dst differs from the mass interpolation:\n{emissions - mass_out}"
    )
    print(
        f"[{label}] conserved: no-flux mass {mass_total.ravel()}, flux emissions {emissions_total.ravel()}", flush=True
    )
    return mass_out, flux_out, src_mass


def shares(mass_out, src_mass, ratio=2):
    """Fraction of the mass of the parent source cell received by every destination cell (first time and level)."""
    return mass_out[0, 0] / parent_field(src_mass[0, 0], ratio)


def assert_shares(share, expected, label):
    """expected: list of ((lat_idx, lon_idx) or numpy.s_[lat_slice, lon_slice], fraction) pairs."""
    for key, value in expected:
        assert np.allclose(share[key], value, rtol=SHARE_RTOL, atol=1e-12), (
            f"[{label}] share{key} = {share[key]} != {value}"
        )


def default_values(n=4):
    """Distinct positive source values, so that a wrongly mapped source cell is detected."""
    return 1.0 + np.arange(n * n, dtype=np.float64).reshape(n, n)


def shifted_pixels(n_pixels):
    """
    Values and transform of a raster of n_pixels x n_pixels 1 deg pixels with origin (-0.5, n_pixels - 0.5).

    Over the default grids (src 1 deg, dst 0.5 deg from (0, 0)) every dst cell lies in a single pixel (or outside
    the raster), and the 4 dst cells of a src cell lie in 4 different pixels with equal overlaps. With
    n_pixels < 5 the raster ends inside the domain (north and east).
    """
    pixels = np.arange(1, n_pixels * n_pixels + 1, dtype=np.float32).reshape(n_pixels, n_pixels)
    return pixels, rasterio.transform.from_origin(-0.5, n_pixels - 0.5, 1.0, 1.0)


def shifted_pixels_shares(pixels):
    """Expected shares with a shifted_pixels() proxy: each src cell is split proportionally to its pixel values."""
    n_pixels = pixels.shape[0]
    # Value of the pixel holding every dst cell (raster row 0 is the northern one), 0 outside the raster
    idx = (np.arange(8) + 1) // 2
    inside = idx < n_pixels
    dst_pixel = np.zeros((8, 8))
    dst_pixel[np.ix_(inside, inside)] = pixels[n_pixels - 1 - idx[inside]][:, idx[inside]]
    src_total = parent_field(dst_pixel.reshape(4, 2, 4, 2).sum(axis=(1, 3)), 2)
    # Src cells without any pixel fall back to plain area weights
    return np.where(src_total > 0, dst_pixel / np.where(src_total > 0, src_total, 1.0), plain_area_shares())
