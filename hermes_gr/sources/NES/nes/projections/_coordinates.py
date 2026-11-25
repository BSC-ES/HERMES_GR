# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
# Licensed under the Apache License, Version 2.0. See LICENSE for details.
"""Shared CF metadata and native geometry helpers for structured projections."""

from copy import deepcopy
import re

import numpy as np
from pyproj import CRS, Proj, Transformer


EARTH_FIELDS = ('earth_radius', 'semi_major_axis', 'semi_minor_axis', 'inverse_flattening',
                'longitude_of_prime_meridian')
CF_FIELDS = EARTH_FIELDS + (
    'grid_mapping_name', 'standard_parallel', 'longitude_of_central_meridian',
    'longitude_of_projection_origin', 'latitude_of_projection_origin',
    'false_easting', 'false_northing', 'scale_factor_at_projection_origin',
    'grid_north_pole_latitude', 'grid_north_pole_longitude', 'north_pole_grid_longitude',
    'crs_wkt', 'reference_ellipsoid_name', 'prime_meridian_name', 'horizontal_datum_name',
    'geographic_crs_name', 'projected_crs_name',
)


def mapping_references(text):
    """Return mapping variable names from simple or expanded CF syntax.

    Parameters
    ----------
    text : str
        Value of a scientific variable's grid_mapping attribute.

    Returns
    -------
    list of str
        Referenced mapping names, excluding expanded coordinate lists.
    """
    return re.findall(r'(?:^|\s)([^\s:]+)\s*:', text) if ':' in text else text.split()


def mapping_name(dataset):
    """Find the single referenced CF structured-grid mapping name.

    Parameters
    ----------
    dataset : netCDF4.Dataset
        Open input file; only variable metadata are inspected.

    Returns
    -------
    str or None
        CF grid_mapping_name, or None for legacy files without mapping references.

    Raises
    ------
    ValueError
        A mapping reference is absent from the file or references multiple grids.
    """
    names = {name for variable in dataset.variables.values()
             for name in mapping_references(getattr(variable, 'grid_mapping', ''))}
    if len(names) > 1:
        raise ValueError('NES requires one referenced structured grid mapping per dataset')
    if not names:
        return None
    name = names.pop()
    if name not in dataset.variables:
        raise ValueError(f'CF grid_mapping references missing variable {name!r}')
    legacy = {'rotated_pole': 'rotated_latitude_longitude', 'Lambert_Conformal': 'lambert_conformal_conic',
              'Lambert_conformal': 'lambert_conformal_conic', 'mercator': 'mercator', 'crs': 'latitude_longitude'}
    result = getattr(dataset.variables[name], 'grid_mapping_name', legacy.get(name))
    if result is None:
        raise ValueError(f'CF mapping variable {name!r} is missing grid_mapping_name')
    return result


def read_mapping(grid, mapping, legacy_names=(), required=True):
    """Read a CF mapping while retaining legacy NES variable-name compatibility.

    Parameters
    ----------
    grid : Nes
        Object with lazy variable metadata and an open input dataset.
    mapping : str
        Expected CF grid_mapping_name.
    legacy_names : tuple of str, optional
        Historical mapping variable names, used only without a CF reference.
    required : bool, optional
        Raise when no mapping can be found. Default is True.

    Returns
    -------
    dict or None
        Independent mapping attributes, with storage/data entries removed.

    Raises
    ------
    ValueError
        A required mapping is absent or its required projection fields are missing.

    Notes
    -----
    Cache attributes because geographical coordinate recovery can require the CRS
    before the base initializer's final projection setup. No MPI collectives run.
    """
    cache = getattr(grid, '_input_mapping_cache', {})
    if mapping in cache:
        return deepcopy(cache[mapping])
    references = {name for variable in grid.dataset.variables.values()
                  for name in mapping_references(getattr(variable, 'grid_mapping', ''))}
    candidates = [name for name, metadata in grid.variables.items()
                  if metadata.get('grid_mapping_name', mapping if name in legacy_names else None) == mapping
                  and (not references or name in references)]
    if not candidates and not references:
        candidates = [name for name in legacy_names if name in grid.variables]
    if len(candidates) != 1:
        if not candidates and not required:
            return None
        raise ValueError(f'Expected one CF {mapping!r} mapping, found {len(candidates)}')
    metadata = {key: value for key, value in grid.variables[candidates[0]].items()
                if key not in ('data', 'dimensions', 'dtype', '_FillValue', 'missing_value')}
    metadata.setdefault('grid_mapping_name', mapping)
    fields = {
        'lambert_conformal_conic': ('standard_parallel', 'longitude_of_central_meridian',
                                    'latitude_of_projection_origin'),
        'mercator': ('longitude_of_projection_origin',),
        'rotated_latitude_longitude': ('grid_north_pole_latitude', 'grid_north_pole_longitude'),
    }.get(mapping, ())
    for field in fields:
        if field not in metadata:
            raise ValueError(f'Missing mandatory CF {field} for {mapping}')
    if mapping == 'mercator' and not any(key in metadata for key in
                                         ('standard_parallel', 'scale_factor_at_projection_origin')):
        raise ValueError('Mercator requires standard_parallel or scale_factor_at_projection_origin')
    if isinstance(metadata.get('standard_parallel'), str):
        values = [float(value) for value in metadata['standard_parallel'].replace(',', ' ').split()]
        metadata['standard_parallel'] = values[0] if len(values) == 1 else values
    if mapping == 'lambert_conformal_conic':
        parallels = np.asarray(metadata['standard_parallel'], dtype=float).reshape(-1)
        if parallels.size not in (1, 2) or not np.isfinite(parallels).all():
            raise ValueError('Lambert Conformal Conic requires one or two finite standard parallels')
        if parallels.size == 1 and not np.isclose(parallels[0], metadata['latitude_of_projection_origin']):
            raise ValueError('A single standard_parallel must equal latitude_of_projection_origin for CF LCC')
    cache[mapping] = deepcopy(metadata)
    grid._input_mapping_cache = cache
    grid.free_vars(candidates[0])
    return metadata


def length_factor(units):
    """Convert supported CF projected coordinate units to metres.

    Parameters
    ----------
    units : str
        Case-sensitive m, km, cm or mm, or a supported metre-name spelling.

    Returns
    -------
    float
        Multiplier converting native projected coordinates to metres.

    Raises
    ------
    ValueError
        Units are missing or unsupported.
    """
    factors = {'m': 1., 'km': 1000., 'cm': .01, 'mm': .001}
    for prefix, scale in (('', 1.), ('kilo', 1000.), ('centi', .01), ('milli', .001)):
        for suffix in ('metre', 'metres', 'meter', 'meters'):
            factors[prefix + suffix] = scale
    try:
        return factors[units.strip()]
    except (KeyError, AttributeError):
        raise ValueError('Projected coordinates require length units m, km, cm or mm') from None


def build_projection(grid):
    """Construct a structured CRS and geographical transformer from CF metadata.

    Parameters
    ----------
    grid : Nes
        Grid with projection_data and the legacy default earth_radius axes.

    Returns
    -------
    pyproj.Proj
        Projection using metre-valued inputs for linear axes.

    Raises
    ------
    ValueError
        An explicitly supplied Earth shape is incomplete or invalid.

    Notes
    -----
    Legacy grids without Earth attributes keep the established NES radius. Their
    native output now records that radius explicitly. A supplied Earth definition
    is never replaced with the fallback. WKT is retained for output, while numeric
    CF attributes control coordinate calculations and offset-unit conversion.
    """
    metadata = grid.projection_data
    reference = CRS.from_wkt(metadata["crs_wkt"]) if "crs_wkt" in metadata else None
    cf = {key: value for key, value in metadata.items() if key in CF_FIELDS and key != "crs_wkt"}
    if not any(
        key in cf for key in ("earth_radius", "semi_major_axis", "semi_minor_axis", "inverse_flattening")
    ):
        if reference is not None:
            cf["semi_major_axis"] = reference.ellipsoid.semi_major_metre
            cf["semi_minor_axis"] = reference.ellipsoid.semi_minor_metre
            metadata.update(semi_major_axis=cf["semi_major_axis"], semi_minor_axis=cf["semi_minor_axis"])
        else:
            cf["earth_radius"] = float(grid.earth_radius[1])
            metadata["earth_radius"] = cf["earth_radius"]
    if "earth_radius" not in cf:
        if "semi_major_axis" not in cf or not any(
            key in cf for key in ("semi_minor_axis", "inverse_flattening")
        ):
            raise ValueError(
                "A CF ellipsoid requires semi_major_axis and semi_minor_axis or inverse_flattening"
            )
    for field in ("earth_radius", "semi_major_axis", "semi_minor_axis"):
        if field in cf and (not np.isfinite(float(cf[field])) or float(cf[field]) <= 0):
            raise ValueError(f"CF {field} must be positive and finite")
    for axis, field in (("x", "false_easting"), ("y", "false_northing")):
        coordinate = getattr(grid, "_full_" + axis, None)
        units = (
            coordinate.get("units", "m")
            if coordinate is not None
            else getattr(grid, "_laea_units", {}).get(axis, "m")
        )
        if coordinate is None and getattr(grid, "dataset", None) is not None:
            variable = next(
                (
                    v
                    for v in grid.dataset.variables.values()
                    if getattr(v, "standard_name", "") == f"projection_{axis}_coordinate"
                ),
                None,
            )
            if variable is None:
                variable = grid.dataset.variables.get(axis)
            if variable is not None:
                units = getattr(
                    variable,
                    "units",
                    None if getattr(variable, "standard_name", "").startswith("projection_") else "m",
                )
        if field in cf:
            cf[field] = float(cf[field]) * length_factor(units)
    if "earth_radius" in cf and any(
        key in cf for key in ("semi_major_axis", "semi_minor_axis", "inverse_flattening")
    ):
        raise ValueError("CF earth_radius cannot be combined with ellipsoid axes or inverse_flattening")
    if "inverse_flattening" in cf:
        rf = float(cf["inverse_flattening"])
        if not np.isfinite(rf) or rf < 0 or (0 < rf <= 1):
            raise ValueError("CF inverse_flattening must be zero or greater than one and finite")
    if "semi_minor_axis" in cf:
        a, b = float(cf["semi_major_axis"]), float(cf["semi_minor_axis"])
        if b > a:
            raise ValueError("CF semi_minor_axis cannot exceed semi_major_axis")
        if "inverse_flattening" in cf:
            rf = float(cf["inverse_flattening"])
            if not np.isclose(b, a if rf == 0 else a * (1 - 1 / rf), rtol=1e-10):
                raise ValueError("CF ellipsoid axes and inverse_flattening disagree")
    for key in (
        "standard_parallel",
        "latitude_of_projection_origin",
        "longitude_of_projection_origin",
        "longitude_of_central_meridian",
        "grid_north_pole_latitude",
        "grid_north_pole_longitude",
        "north_pole_grid_longitude",
        "longitude_of_prime_meridian",
        "false_easting",
        "false_northing",
        "scale_factor_at_projection_origin",
    ):
        if key in cf:
            value = np.asarray(cf[key], dtype=float)
            if not np.isfinite(value).all() or (key != "standard_parallel" and value.size != 1):
                raise ValueError(f"Invalid finite scalar CF projection attribute {key}")
            if key != "standard_parallel":
                cf[key] = value.item()
    if cf["grid_mapping_name"] == "mercator" and "standard_parallel" in cf:
        if np.asarray(cf["standard_parallel"]).size != 1 or abs(float(cf["standard_parallel"])) >= 90:
            raise ValueError("Mercator requires one standard_parallel strictly between -90 and 90")
    if "scale_factor_at_projection_origin" in cf and cf["scale_factor_at_projection_origin"] <= 0:
        raise ValueError("CF scale_factor_at_projection_origin must be positive")
    for key in ("standard_parallel", "latitude_of_projection_origin", "grid_north_pole_latitude"):
        if key in cf and np.any(np.abs(np.asarray(cf[key], dtype=float)) > 90):
            raise ValueError(f"CF {key} must lie between -90 and 90 degrees")
    crs = CRS.from_cf(cf)
    if reference is not None:
        reference_cf = reference.to_cf()
        for key, abbreviations in (("false_easting", ("E", "X")), ("false_northing", ("N", "Y"))):
            if key in reference_cf:
                axis = next(
                    (axis for axis in reference.axis_info if axis.abbrev.upper() in abbreviations),
                    reference.axis_info[0],
                )
                reference_cf[key] = float(reference_cf[key]) * axis.unit_conversion_factor
        numeric = (
            "standard_parallel",
            "longitude_of_central_meridian",
            "longitude_of_projection_origin",
            "latitude_of_projection_origin",
            "false_easting",
            "false_northing",
            "scale_factor_at_projection_origin",
            "grid_north_pole_latitude",
            "grid_north_pole_longitude",
            "north_pole_grid_longitude",
            "longitude_of_prime_meridian",
        )
        valid = reference_cf.get("grid_mapping_name") == cf["grid_mapping_name"]
        for key in numeric:
            if key in cf or key in reference_cf:
                left, right = np.asarray(cf.get(key, 0.0)), np.asarray(reference_cf.get(key, 0.0))
                valid &= left.shape == right.shape and np.allclose(left, right, rtol=1e-10, atol=1e-8)
        valid &= np.isclose(
            crs.ellipsoid.semi_major_metre, reference.ellipsoid.semi_major_metre, rtol=1e-10, atol=1e-6
        )
        valid &= np.isclose(
            crs.ellipsoid.semi_minor_metre, reference.ellipsoid.semi_minor_metre, rtol=1e-10, atol=1e-6
        )
        if not valid:
            raise ValueError("CF crs_wkt disagrees with numerical projection attributes")
        if all(np.isclose(axis.unit_conversion_factor, 1) for axis in reference.axis_info):
            crs = reference
    geographic = {key: value for key, value in cf.items() if key in EARTH_FIELDS}
    geographic["longitude_of_prime_meridian"] = 0.0
    geographic["grid_mapping_name"] = "latitude_longitude"
    grid._geographic_crs = crs if cf["grid_mapping_name"] == "latitude_longitude" else CRS.from_cf(geographic)
    grid._to_geographical = Transformer.from_crs(crs, grid._geographic_crs, always_xy=True)
    grid.earth_radius = [crs.ellipsoid.semi_minor_metre, crs.ellipsoid.semi_major_metre]
    return Proj(crs)


def native_bounds(coordinate, increment=None):
    """Obtain explicit axis intervals or infer uniform projected/angular edges.

    Parameters
    ----------
    coordinate : dict
        One-dimensional data with optional bounds_data of shape (n, 2).
    increment : float or None, optional
        Signed creation increment for singleton axes, in native units.

    Returns
    -------
    numpy.ndarray
        Native two-vertex intervals, preserving coordinate order.

    Raises
    ------
    ValueError
        Axis values/bounds are invalid or edges cannot be recovered unambiguously.

    Notes
    -----
    Uniformity checks account for the original floating-point encoding. In
    particular, decoded float32 centres need not have identical differences.
    Estimate the step from the complete span and check both adjacent differences
    and the affine axis within the encoding tolerance. Genuine nonuniform axes
    still require explicit bounds. Centre values are never replaced or rounded.
    """
    stored_values = np.asarray(coordinate["data"])
    values = np.asarray(stored_values, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("Native grid axes must be nonempty finite one-dimensional arrays")
    delta = np.diff(values)
    if delta.size and not (np.all(delta > 0) or np.all(delta < 0)):
        raise ValueError("Native grid axes must be strictly monotonic")
    if "bounds_data" in coordinate:
        bounds = np.asarray(coordinate["bounds_data"], dtype=float)
    else:
        step = (values[-1] - values[0]) / (values.size - 1) if delta.size else increment
        if step is None or not np.isfinite(step) or step == 0:
            raise ValueError("Singleton axes require explicit bounds or a known nonzero increment")
        if delta.size:
            tolerance = max(1e-9, abs(step) * 1e-6)
            if np.issubdtype(stored_values.dtype, np.floating):
                tolerance = max(tolerance, 4 * np.finfo(stored_values.dtype).eps * float(np.abs(values).max()))
            affine_values = values[0] + np.arange(values.size) * step
            if (not np.allclose(delta, step, rtol=0, atol=tolerance)
                    or not np.allclose(values, affine_values, rtol=0, atol=tolerance)):
                raise ValueError("Nonuniform native axes require explicit bounds")
        bounds = np.column_stack((values - step / 2, values + step / 2))
    if bounds.shape != (values.size, 2) or not np.isfinite(bounds).all():
        raise ValueError("Native axis bounds must be finite with shape (n, 2)")
    if (
        np.any(bounds[:, 0] == bounds[:, 1])
        or np.any(values < bounds.min(axis=1))
        or np.any(values > bounds.max(axis=1))
    ):
        raise ValueError("Native axis bounds must have nonzero width and contain their centres")
    return bounds


def structured_bounds(grid, y_name, x_name):
    """Set geographical corners from actual native intervals on a structured grid.

    Parameters
    ----------
    grid : Nes
        Structured grid with full native axes and a geographical transformer.
    y_name, x_name : str
        Native axis names, such as y/x or rlat/rlon.

    Notes
    -----
    Collective on the grid communicator. Spatial cells use counterclockwise native
    corners, independent of axis direction, and retain rank-local NES slicing.
    """
    x = getattr(grid, 'get_full_' + x_name)()
    y = getattr(grid, 'get_full_' + y_name)()
    xb = native_bounds(x, grid.projection_data.get('inc_' + x_name))
    yb = native_bounds(y, grid.projection_data.get('inc_' + y_name))
    if x_name == 'x':
        xb = xb * length_factor(x.get('units', 'm'))
        yb = yb * length_factor(y.get('units', 'm'))
    else:
        xb = xb * angular_factor(x.get('units', 'degrees'))
        yb = yb * angular_factor(y.get('units', 'degrees'))
    xl, xr = xb.min(axis=1), xb.max(axis=1)
    yl, yr = yb.min(axis=1), yb.max(axis=1)
    xx = np.stack(np.broadcast_arrays(xl[None, :], xr[None, :], xr[None, :], xl[None, :]))
    yy = np.stack(np.broadcast_arrays(yl[:, None], yl[:, None], yr[:, None], yr[:, None]))
    xx, yy = np.broadcast_arrays(xx, yy)
    lon, lat = grid._to_geographical.transform(np.moveaxis(xx, 0, -1), np.moveaxis(yy, 0, -1))
    if not np.isfinite([lon, lat]).all():
        raise ValueError('Native cell corners fall outside the projection domain')
    grid.set_full_latitudes_boundaries({'data': lat})
    grid.set_full_longitudes_boundaries({'data': lon})
    grid.lat_bnds = grid._get_coordinate_values({'data': lat}, 'Y', bounds=True)
    grid.lon_bnds = grid._get_coordinate_values({'data': lon}, 'X', bounds=True)


def select_native_axes(grid, y_name, x_name):
    """Filter native centres, bounds and metadata without slicing local arrays twice.

    Parameters
    ----------
    grid : Nes
        Structured grid with the original file read indices and current selection.
    y_name, x_name : str
        Native coordinate names in Y/X order.

    Notes
    -----
    Collective on the grid communicator. The caller subsequently filters the base
    geographical/time/level coordinates through its superclass implementation.
    """
    idx = grid._get_idx_intervals()
    for name, axis in ((y_name, 'Y'), (x_name, 'X')):
        info = getattr(grid, 'get_full_' + name)()
        local = grid._get_coordinate_values(info, axis)
        section = slice(idx[f'idx_{axis.lower()}_min'], idx[f'idx_{axis.lower()}_max'])
        for key in ('data', 'bounds_data'):
            if key in info:
                info[key] = info[key][section].copy()
        getattr(grid, 'set_full_' + name)(info)
        setattr(grid, name, local)


def write_native_bounds(grid, netcdf, names):
    """Write CF bounds for native structured coordinates already created by a writer.

    Parameters
    ----------
    grid : Nes
        Structured object retaining full native coordinate dictionaries.
    netcdf : netCDF4.Dataset
        Open output dataset; every communicator rank must participate.
    names : tuple of str
        Native axes for which two-vertex bounds are written.
    """
    for name in names:
        info = getattr(grid, 'get_full_' + name)()
        bounds = native_bounds(info, grid.projection_data.get('inc_' + name))
        coordinate = netcdf.variables[name]
        coordinate.bounds = name + '_bnds'
        coordinate.axis = 'X' if name in ('x', 'rlon') else 'Y'
        variable = netcdf.createVariable(coordinate.bounds, 'f8', (name, 'native_nv'))
        if grid.size > 1:
            variable.set_collective(True)
        variable[:] = bounds


def resize_native_axes(grid, n_cells, expand, full=False, is_first=True, is_last=True):
    """Resize native intervals and regenerate geographically transformed geometry.

    Parameters
    ----------
    grid : Nes
        Grid exposing native axis names, full getters and a geographical transformer.
    n_cells : int
        Number of cells to add/remove at each outer edge.
    expand : bool
        Add cells when True, otherwise remove them.
    full : bool, optional
        Update full arrays on rank zero rather than local arrays. Default is False.
    is_first, is_last : bool, optional
        Whether this rank owns the first/last distributed partition. Defaults are True.

    Raises
    ------
    ValueError
        Expansion is requested for nonuniform axes or transformed corners are invalid.

    Notes
    -----
    Local updates collect full-axis spacing first, so even single-cell partitions
    retain the direction of the original global grid. Full updates do not communicate.
    """
    names = grid._native_axis_names
    if full and not grid.master:
        return
    steps = {}
    if expand:
        for name in names:
            info = getattr(grid, '_full_' + name) if full else getattr(grid, 'get_full_' + name)()
            delta = np.diff(info['data'])
            if delta.size and not np.allclose(delta, delta[0], rtol=1e-6, atol=1e-9):
                raise ValueError('Grid expansion requires uniformly spaced native axes')
            steps[name] = delta[0] if delta.size else float(np.diff(native_bounds(
                info, grid.projection_data.get('inc_' + name))[0])[0])
    prefix = '_full_' if full else ''
    for name, axis in zip(names, ('Y', 'X')):
        info = getattr(grid, prefix + name)
        bounds = native_bounds(info, grid.projection_data.get('inc_' + name))
        distributed = not full and grid.parallel_method == axis
        left, right = (is_first, is_last) if distributed else (True, True)
        info['bounds_data'] = bounds
        for key in ('data', 'bounds_data'):
            values = info[key]
            if expand:
                before = np.arange(n_cells, 0, -1) * steps[name]
                after = np.arange(1, n_cells + 1) * steps[name]
                if key == 'bounds_data':
                    before, after = before[:, None], after[:, None]
                parts = ([values[0] - before] if left else []) + [values]
                parts += [values[-1] + after] if right else []
                info[key] = np.concatenate(parts)
            else:
                info[key] = values[slice(n_cells if left else None, -n_cells if right and n_cells else None)]
    y, x = (getattr(grid, prefix + name) for name in names)
    linear = names[1] == 'x'
    xf = length_factor(x.get('units', 'm')) if linear else angular_factor(x.get('units', 'degrees'))
    yf = length_factor(y.get('units', 'm')) if linear else angular_factor(y.get('units', 'degrees'))
    lon, lat = grid._to_geographical.transform(*np.meshgrid(x['data'] * xf, y['data'] * yf))
    xb = native_bounds(x, grid.projection_data.get('inc_' + names[1])) * xf
    yb = native_bounds(y, grid.projection_data.get('inc_' + names[0])) * yf
    xl, xr = xb.min(axis=1), xb.max(axis=1)
    yl, yr = yb.min(axis=1), yb.max(axis=1)
    xx = np.stack(np.broadcast_arrays(xl[None, :], xr[None, :], xr[None, :], xl[None, :]))
    yy = np.stack(np.broadcast_arrays(yl[:, None], yl[:, None], yr[:, None], yr[:, None]))
    xx, yy = np.broadcast_arrays(xx, yy)
    lon_bounds, lat_bounds = grid._to_geographical.transform(np.moveaxis(xx, 0, -1), np.moveaxis(yy, 0, -1))
    if not np.isfinite([lon, lat]).all() or not np.isfinite([lon_bounds, lat_bounds]).all():
        raise ValueError('Expanded/contracted native geometry falls outside the projection domain')
    for name, values in (('lat', lat), ('lon', lon), ('lat_bnds', lat_bounds), ('lon_bnds', lon_bounds)):
        setattr(grid, prefix + name, {'data': values})


def angular_factor(units):
    """Convert supported angular coordinate units to degrees.

    Parameters
    ----------
    units : str
        Degree/radian units, including conventional geographical degree suffixes.

    Returns
    -------
    float
        Native angular coordinate to degree multiplier.

    Raises
    ------
    ValueError
        Units are unsupported.
    """
    if units in ('radian', 'radians', 'rad'):
        return 180. / np.pi
    degrees = {'degree', 'degrees', 'deg'}
    for stem in ('degree', 'degrees'):
        for suffix in ('north', 'east', 'N', 'E', 'n', 'e'):
            degrees.add(stem + '_' + suffix)
    if units not in degrees:
        raise ValueError(f'Unsupported angular coordinate units: {units!r}')
    return 1.


def grid_perimeter(grid, y_name, x_name):
    """Transform a closed clockwise perimeter from contiguous native intervals.

    Parameters
    ----------
    grid : Nes
        Structured object with local native axes and a geographical transformer.
    y_name, x_name : str
        Native coordinate names in Y/X order.

    Returns
    -------
    tuple of dict
        Latitude/longitude perimeter arrays in degrees, starting at the native
        lower-left corner and repeating that vertex at the end.

    Raises
    ------
    ValueError
        Native intervals are not contiguous or a transformed vertex is nonfinite.
    """
    edges = []
    for name in (x_name, y_name):
        info = getattr(grid, name)
        bounds = np.sort(native_bounds(info, grid.projection_data.get('inc_' + name)), axis=1)
        bounds = bounds[np.argsort(bounds[:, 0])]
        if not np.allclose(bounds[:-1, 1], bounds[1:, 0], rtol=1e-9, atol=1e-9):
            raise ValueError('A rectangular grid perimeter requires contiguous native bounds')
        factor = length_factor(info.get('units', 'm')) if x_name == 'x' else angular_factor(
            info.get('units', 'degrees'))
        edges.append(np.r_[bounds[:, 0], bounds[-1, 1]] * factor)
    x, y = edges
    xx = np.r_[np.full(len(y), x[0]), x[1:], np.full(len(y) - 1, x[-1]), x[-2::-1]]
    yy = np.r_[y, np.full(len(x) - 1, y[-1]), y[-2::-1], np.full(len(x) - 1, y[0])]
    lon, lat = grid._to_geographical.transform(xx, yy)
    if not np.isfinite([lon, lat]).all():
        raise ValueError('Grid perimeter falls outside the projection domain')
    return {'data': lat}, {'data': lon}


def validate_creation(kind, parameters):
    """Validate native creation geometry before entering distributed constructors.

    Parameters
    ----------
    kind : str
        regular, global, global_monarch, rotated, rotated-nested, lcc, mercator or points.
    parameters : dict
        Creation arguments in native angular or metre units.

    Raises
    ------
    ValueError
        Counts, increments, origins or station coordinates are invalid.

    Notes
    -----
    The same scalar/axis checks run on every rank. This prevents a master-only
    coordinate allocation failure from leaving other ranks inside collectives.
    """
    if kind == 'points':
        latitude, longitude = np.asarray(parameters['lat'], dtype=float), np.asarray(parameters['lon'], dtype=float)
        if latitude.ndim != 1 or longitude.shape != latitude.shape or not latitude.size:
            raise ValueError('Point coordinates must be nonempty corresponding one-dimensional arrays')
        if not np.isfinite(latitude).all() or not np.isfinite(longitude).all() or np.any(np.abs(latitude) > 90):
            raise ValueError('Point coordinates must be finite and latitudes lie between -90 and 90')
        return
    counts, increments, origins = (), (), ()
    if kind in ('lcc', 'mercator'):
        counts, increments, origins = ('nx', 'ny'), ('inc_x', 'inc_y'), (
            'x_0', 'y_0', 'lon_0', 'lat_0', 'lat_1', 'lat_2') if kind == 'lcc' else (
            'x_0', 'y_0', 'lon_0', 'lat_ts')
    elif kind == 'regular':
        counts, increments, origins = ('n_lat', 'n_lon'), ('inc_lat', 'inc_lon'), ('lat_orig', 'lon_orig')
    elif kind in ('global', 'global_monarch'):
        increments = ('inc_lat', 'inc_lon')
    elif kind == 'rotated':
        increments, origins = ('inc_rlat', 'inc_rlon'), ('south_boundary', 'west_boundary', 'centre_lat', 'centre_lon')
    elif kind == 'rotated-nested':
        counts = ('n_rlat', 'n_rlon', 'parent_ratio', 'i_parent_start', 'j_parent_start')
    for key in counts + increments + origins:
        if key not in parameters:
            raise ValueError(f'Missing grid creation parameter {key}')
        value = parameters[key]
        if np.ndim(value) != 0 or not np.isfinite(float(value)):
            raise ValueError(f'Grid creation {key} must be a finite scalar')
        if key in counts and (float(value) != int(value) or int(value) <= 0):
            raise ValueError(f'Grid creation {key} must be a positive integer')
        if key in increments and float(value) == 0:
            raise ValueError(f'Grid creation {key} must be nonzero')
        if kind in ('global', 'global_monarch', 'rotated') and key in increments and float(value) < 0:
            raise ValueError(f'{kind} creation requires positive {key}')
    if kind == 'global' and (parameters['inc_lat'] > 180 or parameters['inc_lon'] > 360):
        raise ValueError('Global increments must retain at least one latitude and longitude cell')
