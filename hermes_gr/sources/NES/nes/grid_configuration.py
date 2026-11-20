"""Canonical structured-grid definitions, independent of MPI layout and time.

These helpers only inspect metadata and one-dimensional native axes. They do
not create a NES object, geographic coordinate mesh, cell areas or geometries.
"""
import json

import numpy as np


ATTRIBUTE = 'nes_grid_configuration'
SCHEMA_VERSION = 1
PROJECTION_FIELDS = {
    'latitude_longitude': ['semi_major_axis', 'inverse_flattening'],
    'rotated_latitude_longitude': ['grid_north_pole_latitude', 'grid_north_pole_longitude'],
    'lambert_conformal_conic': [
        'standard_parallel', 'longitude_of_central_meridian', 'latitude_of_projection_origin',
    ],
    'mercator': ['longitude_of_projection_origin'],
    'lambert_azimuthal_equal_area': ['latitude_of_projection_origin', 'longitude_of_projection_origin'],
}
AXES = {'latitude_longitude': ('lat', 'lon'), 'rotated_latitude_longitude': ('rlat', 'rlon'),
        'lambert_conformal_conic': ('y', 'x'), 'mercator': ('y', 'x'),
        'lambert_azimuthal_equal_area': ('y', 'x')}


def _number(value):
    """
    Convert finite numeric values to JSON-compatible scalars or lists.

    Parameters
    ----------
    value : array_like
        Numeric scalar or array convertible to floating-point values.

    Returns
    -------
    float or list
        A float for one value, otherwise a nested list preserving array shape.

    Raises
    ------
    ValueError
        If conversion fails or any value is nonfinite.
    TypeError
        If the input does not support numeric conversion.
    """
    values = np.asarray(value, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f'Grid configuration contains nonfinite values: {value}')
    return float(values.reshape(-1)[0]) if values.size == 1 else values.tolist()


def _axis(first, step, size, layout='uniform'):
    """
    Describe a native grid axis without constructing its coordinates.

    Parameters
    ----------
    first : float
        First nominal cell centre, before any Global Monarch edge adjustment.
    step : float
        Nonzero signed spacing in native axis units.
    size : int
        Positive number of cells along the axis.
    layout : {'uniform', 'global_monarch'}, optional
        Uniform spacing or the NES Global Monarch polar/end-cell convention.

    Returns
    -------
    dict
        JSON-compatible first/last centres, spacing, cell count and layout.

    Raises
    ------
    ValueError
        If the axis is empty, spacing is zero or a numeric value is nonfinite.

    Notes
    -----
    Global Monarch moves the first/last nominal centres inward by one quarter
    of the spacing. No coordinate array is allocated.
    """
    size = int(size)
    if size <= 0:
        raise ValueError(f'Grid axis size must be positive, found {size}')
    first, step = _number(first), _number(step)
    if step == 0:
        raise ValueError(f'Grid resolution must be nonzero, found {step}')
    last = first + step * (size - 1)
    if layout == 'global_monarch':
        first += step / 4
        last -= step / 4
    return dict(first=first, last=last, step=step, size=size, layout=layout)


def _earth_definition(parameters, fallback=6371000.):
    """Normalize a spherical or ellipsoidal Earth without constructing a CRS.

    Parameters
    ----------
    parameters : dict
        CF Earth radius or ellipsoid axes/inverse flattening in metres.
    fallback : float, optional
        Established NES spherical radius when no Earth shape is supplied.

    Returns
    -------
    dict
        Canonical semi_major_axis and inverse_flattening values.

    Raises
    ------
    ValueError
        An explicit Earth definition is incomplete, nonfinite or invalid.
    """
    if 'earth_radius' in parameters:
        a, rf = _number(parameters['earth_radius']), 0.
    elif 'semi_major_axis' in parameters:
        a = _number(parameters['semi_major_axis'])
        if 'inverse_flattening' in parameters:
            rf = _number(parameters['inverse_flattening'])
        elif 'semi_minor_axis' in parameters:
            b = _number(parameters['semi_minor_axis'])
            if b <= 0 or b > a:
                raise ValueError('Invalid CF semi_minor_axis')
            rf = 0. if a == b else a / (a - b)
        else:
            raise ValueError('A CF ellipsoid requires semi_minor_axis or inverse_flattening')
    elif any(key in parameters for key in ('semi_minor_axis', 'inverse_flattening')):
        raise ValueError('A CF ellipsoid requires semi_major_axis')
    else:
        a, rf = _number(fallback), 0.
    if a <= 0 or rf < 0 or (0 < rf <= 1):
        raise ValueError('Invalid CF Earth shape')
    return dict(semi_major_axis=a, inverse_flattening=rf)


def _linear_factor(units):
    """Return a supported native coordinate unit's metre conversion factor.

    Parameters
    ----------
    units : str
        Case-sensitive CF length symbol or metre-name spelling.

    Returns
    -------
    float
        Native coordinate to metre multiplier.

    Raises
    ------
    ValueError
        Units are unsupported.
    """
    factors = {'m': 1., 'km': 1000., 'cm': .01, 'mm': .001}
    for prefix, scale in (('', 1.), ('kilo', 1000.), ('centi', .01), ('milli', .001)):
        for suffix in ('metre', 'metres', 'meter', 'meters'):
            factors[prefix + suffix] = scale
    if units not in factors:
        raise ValueError(f'Unsupported projected coordinate units: {units!r}')
    return factors[units]


def configuration_from_parameters(projection, parameters, earth_radius=6371000., parent=None):
    """
    Normalize geographical creation arguments without constructing a grid.

    Parameters
    ----------
    projection : str
        Projection accepted by ``create_nes``: regular, global, global_monarch,
        rotated, rotated-nested, lcc or mercator. Underscore aliases are accepted.
    parameters : dict
        Geographical keyword arguments for the selected projection. Origins,
        increments and counts use the conventions of ``create_nes``.
    earth_radius : float, optional
        Spherical Earth radius in metres used by latitude/longitude CF metadata.
    parent : dict, optional
        Resolved rotated-parent metadata returned by
        ``RotatedNestedNes._get_parent_attributes``. Required for a nested grid;
        supplies the child increments, first coordinates and rotated pole.

    Returns
    -------
    dict
        Schema version, CF projection parameters, global ``[ny, nx]`` shape and
        canonical native-axis descriptors. Paths, time and MPI options are absent.

    Raises
    ------
    KeyError
        If a required geographical parameter or resolved parent field is missing.
    ValueError
        If axis descriptors are invalid, Global Monarch increments are incompatible
        or a nested request lacks resolved parent metadata.
    NotImplementedError
        If the projection has no supported canonical representation.

    Notes
    -----
    The result describes physical grid configuration rather than reproducing all
    original input arguments. Nested grids retain resolved axes and pole, not
    parent paths. Work and storage depend only on the number of parameters.
    """
    p = parameters
    projection = projection.replace('_', '-')
    if projection in ['regular', 'global', 'global-monarch']:
        dy, dx = float(p['inc_lat']), float(p['inc_lon'])
        if dy == 0 or dx == 0 or (projection != 'regular' and (dy < 0 or dx < 0)):
            raise ValueError('Invalid grid angular increments')
        layout = 'global_monarch' if projection == 'global-monarch' else 'uniform'
        if projection == 'regular':
            y0, x0, ny, nx = p['lat_orig'], p['lon_orig'], p['n_lat'], p['n_lon']
        elif projection == 'global':
            y0, x0, ny, nx = -90., -180., int(180 // dy), int(360 // dx)
        else:
            if not np.isclose(90 / dy, round(90 / dy), rtol=0, atol=1e-9) or not np.isclose(
                    180 / dx, round(180 / dx), rtol=0, atol=1e-9):
                raise ValueError('global_monarch increments must divide 90/180 exactly')
            y0, x0 = -90 - dy / 2, -180 - dx / 2
            ny, nx = 2 * round(90 / dy) + 1, 2 * round(180 / dx) + 1
        axes = {'lat': _axis(y0 + dy / 2, dy, ny, layout), 'lon': _axis(x0 + dx / 2, dx, nx, layout)}
        mapping = 'latitude_longitude'
        cf = dict(semi_major_axis=_number(earth_radius), inverse_flattening=0.)
    elif projection in ['rotated', 'rotated-nested']:
        if projection == 'rotated':
            dy, dx = p['inc_rlat'], p['inc_rlon']
            y0, x0 = p['south_boundary'], p['west_boundary']
            ny, nx = int(abs(y0) / dy * 2 + 1), int(abs(x0) / dx * 2 + 1)
            cf = dict(grid_north_pole_latitude=90 - p['centre_lat'],
                      grid_north_pole_longitude=-180 + p['centre_lon'])
        else:
            if parent is None:
                raise ValueError('Resolved parent grid metadata is required for rotated-nested configuration')
            dy, dx = parent['inc_rlat'], parent['inc_rlon']
            y0, x0, ny, nx = parent['1st_rlat'], parent['1st_rlon'], p['n_rlat'], p['n_rlon']
            cf = {key: parent[key] for key in PROJECTION_FIELDS['rotated_latitude_longitude']}
        axes = {'rlat': _axis(y0, dy, ny), 'rlon': _axis(x0, dx, nx)}
        mapping = 'rotated_latitude_longitude'
    elif projection in ['lcc', 'mercator', 'laea']:
        axes = {'y': _axis(p['y_0'] + p['inc_y'] / 2, p['inc_y'], p['ny']),
                'x': _axis(p['x_0'] + p['inc_x'] / 2, p['inc_x'], p['nx'])}
        if projection == 'lcc':
            mapping = 'lambert_conformal_conic'
            cf = dict(standard_parallel=[p['lat_1'], p['lat_2']], longitude_of_central_meridian=p['lon_0'],
                      latitude_of_projection_origin=p['lat_0'])
        elif projection == 'mercator':
            mapping = 'mercator'
            cf = dict(standard_parallel=p['lat_ts'], longitude_of_projection_origin=p['lon_0'])
        else:
            mapping = 'lambert_azimuthal_equal_area'
            latitude = p.get('latitude_of_projection_origin', p.get('lat_0'))
            longitude = p.get('longitude_of_projection_origin', p.get('lon_0'))
            if latitude is None or longitude is None or not any(k in p for k in ('earth_radius', 'semi_major_axis')):
                raise ValueError('LAEA configuration requires explicit CF origins and Earth shape')
            cf = dict(latitude_of_projection_origin=latitude, longitude_of_projection_origin=longitude)
            for axis in axes.values():
                axis['bounds_offsets'] = [-abs(axis['step']) / 2, abs(axis['step']) / 2]
        cf.update(false_easting=p.get('false_easting', 0.), false_northing=p.get('false_northing', 0.))
    else:
        raise NotImplementedError(f'Canonical grid configuration is not supported for {projection!r}')
    earth_parameters = dict(parent or {})
    earth_parameters.update(p)
    if mapping != 'latitude_longitude':
        for axis in axes.values():
            axis['bounds_offsets'] = [-abs(axis['step']) / 2, abs(axis['step']) / 2]
    cf.update(_earth_definition(earth_parameters, earth_radius))
    cf['longitude_of_prime_meridian'] = earth_parameters.get('longitude_of_prime_meridian', 0.)
    if mapping == 'rotated_latitude_longitude':
        cf['north_pole_grid_longitude'] = p.get('north_pole_grid_longitude',
                                                (parent or {}).get('north_pole_grid_longitude', 0.))
    return dict(schema_version=SCHEMA_VERSION, projection=dict(grid_mapping_name=mapping,
                **{key: _number(value) for key, value in cf.items()}), axes=axes,
                shape=[axis['size'] for axis in axes.values()])


def describe_grid_configuration(grid):
    """
    Recover canonical configuration from the master's existing full native axes.

    Parameters
    ----------
    grid : Nes
        Master-side NES object with CF projection metadata and complete native
        coordinate arrays. Must not be a rank-local view of those arrays.

    Returns
    -------
    dict
        JSON-compatible schema, projection, global shape and axis descriptors.

    Raises
    ------
    ValueError
        If required coordinates are missing, masked, empty or nonfinite; a saved
        schema is unsupported; singleton spacing cannot be recovered; or native
        axes and geographical coordinate shapes disagree.
    KeyError
        If required CF projection parameters are absent.
    NotImplementedError
        If the grid is unstructured, curvilinear or has unsupported nonuniform axes.

    Notes
    -----
    This helper is noncollective and performs no I/O. The public NES method calls
    it only on the communicator master. Work is O(nx + ny), without constructing
    2D coordinates or geometry. Float32 spacing checks account for storage precision;
    configuration comparisons still use the caller's explicit tolerances.

    Legacy files recover their definition from native axes and CF metadata.
    Singleton spacing additionally needs creation metadata, saved configuration or
    recoverable geographic bounds. Saved origins, dimensions and CF values never
    override current coordinates/metadata.
    """
    data = grid.projection_data or {}
    mapping = data.get('grid_mapping_name')
    if mapping not in AXES:
        raise NotImplementedError(f'Canonical grid configuration unavailable for {mapping!r}')
    saved = grid.global_attrs.get(ATTRIBUTE) if grid.global_attrs else None
    saved = json.loads(saved) if saved is not None else {}
    if saved and saved.get('schema_version') != SCHEMA_VERSION:
        raise ValueError(f"Unsupported saved grid configuration schema: {saved.get('schema_version')}")
    axes = {}
    layout = 'global_monarch' if data.get('nes_projection') == 'global_monarch' else 'uniform'
    for name in AXES[mapping]:
        coordinate = getattr(grid, '_full_' + name, None)
        if coordinate is None:
            raise ValueError(f'Missing full native grid axis {name}')
        values = coordinate['data']
        if np.ma.is_masked(values):
            raise ValueError(f'Masked native grid axis {name}')
        # Preserve encoding precision before conversion; float32 axes need not
        # have bitwise-identical adjacent differences after decoding.
        stored_dtype = np.asarray(values).dtype
        factor = _linear_factor(coordinate.get('units', 'm')) if name in ('x', 'y') else (
            180. / np.pi if coordinate.get('units') in ('rad', 'radian', 'radians') else 1.)
        values = np.asarray(values, dtype=float) * factor
        if values.ndim != 1:
            raise NotImplementedError(f'Canonical grid configuration needs 1D native axes, found {name}')
        if not values.size or not np.isfinite(values).all():
            raise ValueError(f'Invalid full native grid axis {name}')
        tolerance = 1e-3 if name in ['x', 'y'] else 1e-6
        if np.issubdtype(stored_dtype, np.floating):
            tolerance = max(tolerance, 4 * np.finfo(stored_dtype).eps * float(np.abs(values).max()))
        if layout == 'global_monarch':
            lower = -90 if name == 'lat' else -180
            step = 4 * (values[0] - lower)
            axis = _axis(lower, step, values.size, layout)
            expected_diff = np.full(max(values.size - 1, 0), step)
            if expected_diff.size:
                expected_diff[[0, -1]] = step * .75
        else:
            if values.size > 1:
                # Use the whole span, avoiding first-interval rounding drift.
                step = (values[-1] - values[0]) / (values.size - 1)
            else:
                # A single centre cannot determine resolution on its own.
                step = data.get('inc_' + name)
                if step is not None:
                    step = _number(step) * factor
                if step is None and 'bounds_data' in coordinate:
                    step = float(np.diff(coordinate['bounds_data'][0])[0]) * factor
                if step is None:
                    step = saved.get('axes', {}).get(name, {}).get('step')
                if step is None and name in ['lat', 'lon']:
                    bounds = getattr(grid, '_full_' + name + '_bnds', None)
                    if bounds is not None:
                        step = abs(float(np.diff(bounds['data'][0])[0]))
                if step is None:
                    raise ValueError(f'Cannot recover singleton resolution for {name}; missing grid metadata')
            axis = _axis(values[0], step, values.size)
            expected_diff = np.full(max(values.size - 1, 0), step)
        if not np.allclose(np.diff(values), expected_diff, rtol=0, atol=tolerance) or not np.isclose(
                values[-1], axis['last'], rtol=0, atol=tolerance):
            raise NotImplementedError(f'Native grid axis {name} is not a supported regular spacing/layout')
        if mapping != 'latitude_longitude':
            if 'bounds_data' in coordinate:
                bounds = np.sort(np.asarray(coordinate['bounds_data'], dtype=float), axis=1) * factor
                offsets = bounds - values[:, None]
                if not np.allclose(offsets, offsets[0], rtol=0, atol=tolerance):
                    raise NotImplementedError(
                        'Canonical structured configuration requires constant native cell-bound offsets')
                axis['bounds_offsets'] = offsets[0].tolist()
            else:
                axis['bounds_offsets'] = [-abs(axis['step']) / 2, abs(axis['step']) / 2]
        axes[name] = axis
    shape = tuple(axis['size'] for axis in axes.values())
    for name, expected_shape in [('lat', (shape[0],) if mapping == 'latitude_longitude' else shape),
                                 ('lon', (shape[1],) if mapping == 'latitude_longitude' else shape)]:
        if getattr(grid, '_full_' + name)['data'].shape != expected_shape:
            raise ValueError(f'Full geographic {name} shape does not match native axes {expected_shape}')
    cf = {key: _number(data[key]) for key in PROJECTION_FIELDS[mapping]}
    cf.update(_earth_definition(data))
    cf['longitude_of_prime_meridian'] = _number(data.get('longitude_of_prime_meridian', 0.))
    if mapping == 'mercator':
        for key in ('standard_parallel', 'scale_factor_at_projection_origin'):
            if key in data:
                cf[key] = _number(data[key])
        if not any(key in cf for key in ('standard_parallel', 'scale_factor_at_projection_origin')):
            raise ValueError('Mercator requires a CF standard parallel or scale factor')
    if mapping in ('lambert_conformal_conic', 'mercator', 'lambert_azimuthal_equal_area'):
        for axis, key in (('x', 'false_easting'), ('y', 'false_northing')):
            unit = getattr(grid, '_full_' + axis).get('units', 'm')
            cf[key] = _number(data.get(key, 0.)) * _linear_factor(unit)
    if mapping == 'rotated_latitude_longitude':
        cf['north_pole_grid_longitude'] = _number(data.get('north_pole_grid_longitude', 0.))
    return dict(schema_version=SCHEMA_VERSION, projection=dict(grid_mapping_name=mapping, **cf),
                axes=axes, shape=[axis['size'] for axis in axes.values()])


def compare_grid_configurations(actual, expected, angular_tolerance=1e-6, projected_tolerance=1e-3):
    """
    Compare two canonical global geographical configurations.

    Parameters
    ----------
    actual : dict
        Configuration recovered from the NES object's current axes and metadata.
    expected : dict
        Requested canonical definition using the same schema.
    angular_tolerance : float, optional
        Absolute angular tolerance in degrees; defaults to 1e-6. Must be nonnegative.
    projected_tolerance : float, optional
        Absolute linear tolerance in metres; defaults to 1e-3. Must be nonnegative.

    Returns
    -------
    bool
        True when shape, schema, projection parameters and axis descriptors match.

    Raises
    ------
    ValueError
        If a compared property differs. The message identifies expected/found values.
    KeyError
        If a required canonical configuration field is missing.

    Notes
    -----
    Relative tolerance is zero. Counts, layouts and textual projection names compare
    exactly. Time, levels, file paths, writer settings and MPI layout are excluded.
    This is configuration compatibility, not exhaustive 2D data integrity validation.
    The public validation method checks tolerance signs before calling this helper.
    """
    for key in ["schema_version", "shape"]:
        if actual.get(key) != expected.get(key):
            raise ValueError(f"Grid {key}: expected {expected.get(key)!r}, found {actual.get(key)!r}")
    if set(actual["projection"]) != set(expected["projection"]) or set(actual["axes"]) != set(expected["axes"]):
        raise ValueError(
            f"Grid projection/axes: expected {expected['projection']}, found {actual['projection']}"
        )
    for name, wanted in expected["projection"].items():
        found = actual["projection"][name]
        if isinstance(wanted, str):
            matches = wanted == found
        else:
            tolerance = (
                projected_tolerance
                if name
                in ["semi_major_axis", "semi_minor_axis", "earth_radius", "false_easting", "false_northing"]
                else angular_tolerance
            )
            matches = np.asarray(found).shape == np.asarray(wanted).shape and np.allclose(
                found, wanted, rtol=0, atol=tolerance
            )
        if not matches:
            raise ValueError(f"Grid projection {name}: expected {wanted!r}, found {found!r}")
    for name, axis in expected["axes"].items():
        tolerance = projected_tolerance if name in ["x", "y"] else angular_tolerance
        for key, wanted in axis.items():
            found = actual["axes"][name].get(key)
            matches = (
                found == wanted
                if key in ["size", "layout"]
                else (found is not None and np.allclose(found, wanted, rtol=0, atol=tolerance))
            )
            if not matches:
                raise ValueError(f"Grid axis {name}.{key}: expected {wanted!r}, found {found!r}")
    return True
