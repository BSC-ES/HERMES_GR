#!/usr/bin/env python

# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

"""
Auxiliary file consistency checks.

This module provides small, side-effect-free checks for already opened NES
objects. It does not open, delete or regenerate auxiliary files; callers decide
what to do when a check returns False.
"""


def check_grid(grid_nessy, expected_shape):
    """
    Check that a grid NES object has the expected full 2D domain shape.

    This check is meant for the first validation step of a run: once the
    auxiliary grid file is opened as a NES object, compare its global horizontal
    shape against the shape derived from the current configuration. It uses
    ``Nes.get_full_shape()``, so the comparison is independent of the current
    rank's domain slice.

    Parameters
    ----------
    grid_nessy : Nes
        Opened NES object representing the grid auxiliary file.
    expected_shape : tuple, list or None
        Expected global horizontal shape as ``(n_y, n_x)``. If None, the check is
        skipped.

    Returns
    -------
    bool
        True when the check passes or is skipped.

    Examples
    --------
    >>> check_grid(grid_nessy, expected_shape=(400, 700))
    True
    """
    if expected_shape is None:
        return True

    expected_shape = tuple(expected_shape)
    actual_shape = grid_nessy.get_full_shape()
    if actual_shape != expected_shape:
        return False

    return True


def get_expected_grid_shape(options):
    """
    Get the expected global 2D grid shape from HERMES grid configuration.

    The returned tuple follows NES' convention for ``get_full_shape()``:
    ``(n_y, n_x)``. The formulas mirror the arguments passed to
    ``nes.create_nes`` in ``hermes_gr.modules.grid.select_grid``.

    Parameters
    ----------
    options : Namespace
        HERMES configuration options containing ``domain_type`` and the grid
        parameters required by that domain type.

    Returns
    -------
    tuple
        Expected NES full horizontal shape as ``(n_y, n_x)``.
    """
    if options.domain_type == "global_monarch":
        lat_cells_each_side = round(90 / options.inc_lat)
        lon_cells_each_side = round(180 / options.inc_lon)
        return int((lat_cells_each_side * 2) + 1), int((lon_cells_each_side * 2) + 1)
    if options.domain_type == "global":
        return int(180 // options.inc_lat), int(360 // options.inc_lon)
    if options.domain_type == "regular":
        return options.n_lat, options.n_lon
    if options.domain_type == "rotated":
        n_lat = int((abs(options.south_boundary) / options.inc_rlat) * 2 + 1)
        n_lon = int((abs(options.west_boundary) / options.inc_rlon) * 2 + 1)
        return n_lat, n_lon
    if options.domain_type == "rotated_nested":
        return options.n_rlat, options.n_rlon
    if options.domain_type in ["lcc", "mercator"]:
        return options.ny, options.nx

    raise NotImplementedError(
        f"The grid type {options.domain_type} is not implemented. "
        "Use 'global_monarch', 'global', 'regular', 'rotated', 'rotated_nested', 'lcc' or 'mercator'."
    )


def check_aux_file(
        aux_nessy_to_validate,
        nes_object_to_compare=None,
        var_list=None,
        variable_attributes=None,
        global_attributes=None):
    """
    Check a NES NetCDF auxiliary object.

    The first argument is always the auxiliary NES object to validate. All checks
    are opt-in: only the optional arguments provided by the caller are evaluated.

    When ``nes_object_to_compare`` is provided, the check validates that both NES
    objects use the same domain decomposition in the current MPI rank. This is
    stricter than comparing only the global grid shape: it also checks
    ``write_axis_limits`` and local coordinate array shapes. That matters for
    parallel runs, where two files can describe the same full domain but still be
    decomposed differently across ranks.

    ``var_list``, ``variable_attributes`` and ``global_attributes`` check
    existence only. They intentionally do not compare attribute values or data
    values.

    Parameters
    ----------
    aux_nessy_to_validate : Nes
        NES object loaded from the auxiliary NetCDF file to validate.
    nes_object_to_compare : Nes, optional
        Base NES object used to check that both objects share the same domain decomposition.
    var_list : list, optional
        Variable names that must exist in the auxiliary file.
    variable_attributes : list or dict, optional
        Attribute names that must exist in variables.

        If a list/tuple/set is provided, those attributes are required for every
        variable in ``var_list``. If ``var_list`` is None, they are required for
        all variables found in the auxiliary object.

        If a dict is provided, it must map variable names to the expected
        attribute names for each variable.
    global_attributes : list, optional
        Global attribute names that must exist in the auxiliary file.

    Returns
    -------
    bool
        True when all requested checks pass.

    Examples
    --------
    Validate only domain decomposition against a previously accepted grid:

    >>> check_aux_file(aux_nessy, nes_object_to_compare=grid_nessy)
    True

    Validate variable and attribute existence:

    >>> check_aux_file(
    ...     aux_nessy,
    ...     var_list=["NOx", "CO"],
    ...     variable_attributes=["units", "long_name"],
    ...     global_attributes=["Conventions"],
    ... )
    True

    Validate variable-specific attributes:

    >>> check_aux_file(
    ...     aux_nessy,
    ...     variable_attributes={"NOx": ["units"], "CO": ["units"]},
    ... )
    True
    """
    try:
        if nes_object_to_compare is not None:
            _check_same_domain_decomposition(nes_object_to_compare, aux_nessy_to_validate)
        if var_list is not None:
            _check_variables(aux_nessy_to_validate, var_list)
        if variable_attributes is not None:
            _check_variable_attributes(aux_nessy_to_validate, variable_attributes, var_list=var_list)
        if global_attributes is not None:
            _check_global_attributes(aux_nessy_to_validate, global_attributes)
    except ValueError:
        return False

    return True


def _check_same_domain_decomposition(base_nessy, aux_nessy_to_validate, data_names=None):
    """
    Check that two NES objects have the same domain decomposition in this rank.

    The full shape only proves that both objects describe the same global grid.
    For parallel runs, the local data shape and write axis limits must also match
    so both objects use the same domain decomposition in each rank.
    """
    _check_same_grid_shape(base_nessy, aux_nessy_to_validate)
    _check_same_write_axis_limits(base_nessy, aux_nessy_to_validate)
    _check_same_local_shapes(base_nessy, aux_nessy_to_validate, data_names=data_names)

    return True


def _check_same_grid_shape(base_nessy, aux_nessy_to_validate):
    """
    Check that two NES objects have the same full 2D grid shape.
    """
    base_shape = base_nessy.get_full_shape()
    auxiliary_shape = aux_nessy_to_validate.get_full_shape()
    if base_shape != auxiliary_shape:
        raise ValueError(
            f"Auxiliary NetCDF full grid shape mismatch. Base has {base_shape}, "
            f"auxiliary has {auxiliary_shape}."
        )

    return True


def _check_same_write_axis_limits(base_nessy, aux_nessy_to_validate):
    """
    Check that two NES objects have the same write axis limits in this rank.
    """
    base_limits = getattr(base_nessy, "write_axis_limits", None)
    auxiliary_limits = getattr(aux_nessy_to_validate, "write_axis_limits", None)
    if base_limits != auxiliary_limits:
        raise ValueError(
            f"Auxiliary NetCDF write axis limits mismatch. Base has {base_limits}, "
            f"auxiliary has {auxiliary_limits}."
        )

    return True


def _check_same_local_shapes(base_nessy, aux_nessy_to_validate, data_names=None):
    """
    Check that selected local data arrays have the same shape in two NES objects.

    By default this checks local coordinate arrays common to both objects:
    ``lat``, ``lon``, ``lat_bnds`` and ``lon_bnds``. These arrays are already
    sliced by NES according to the current rank's read/write limits, so matching
    shapes are a practical proxy for matching domain decomposition.
    """
    if data_names is None:
        data_names = _get_common_local_data_names(base_nessy, aux_nessy_to_validate)

    for data_name in data_names:
        base_shape = _get_local_data_shape(base_nessy, data_name)
        auxiliary_shape = _get_local_data_shape(aux_nessy_to_validate, data_name)
        if base_shape != auxiliary_shape:
            raise ValueError(
                f"Auxiliary NetCDF local data shape mismatch for '{data_name}'. "
                f"Base has {base_shape}, auxiliary has {auxiliary_shape}."
            )

    return True


def _check_variables(nessy, expected_variables):
    """
    Check that expected variables exist.
    """
    if expected_variables is None:
        return True

    variables = _get_variables(nessy)
    for var_name in expected_variables:
        if var_name not in variables:
            raise ValueError(f"Auxiliary NetCDF missing variable '{var_name}'.")

    return True


def _check_global_attributes(nessy, expected_attributes):
    """
    Check that expected global attributes exist.
    """
    if expected_attributes is None:
        return True

    global_attrs = getattr(nessy, "global_attrs", None)
    if global_attrs is None:
        global_attrs = getattr(nessy, "attrs", {})

    for attr_name in expected_attributes:
        if attr_name not in global_attrs:
            raise ValueError(f"Auxiliary NetCDF missing global attribute '{attr_name}'.")

    return True


def _check_variable_attributes(nessy, expected_variable_attributes, var_list=None):
    """
    Check that expected variable attributes exist.

    Attribute values are intentionally not checked here. This only validates that
    the metadata keys expected by the caller are present.
    """
    if expected_variable_attributes is None:
        return True

    variables = _get_variables(nessy)
    if isinstance(expected_variable_attributes, (list, tuple, set)):
        if var_list is None:
            var_list = variables.keys()
        expected_variable_attributes = {
            var_name: expected_variable_attributes for var_name in var_list
        }

    for var_name, expected_attributes in expected_variable_attributes.items():
        if var_name not in variables:
            raise ValueError(
                f"Auxiliary NetCDF missing variable '{var_name}' required for "
                "variable attribute checks."
            )

        var_attrs = _get_variable_attributes(variables[var_name])
        for attr_name in expected_attributes:
            if attr_name not in var_attrs:
                raise ValueError(
                    f"Auxiliary NetCDF variable '{var_name}' missing attribute '{attr_name}'."
                )

    return True


def _get_variables(nessy):
    return getattr(nessy, "variables", {})


def _get_variable_attributes(variable_info):
    if hasattr(variable_info, "ncattrs"):
        return variable_info.ncattrs()
    if isinstance(variable_info, dict):
        return variable_info.keys()
    return getattr(variable_info, "attrs", {}).keys()


def _get_common_local_data_names(grid, auxiliary):
    names = ["lat", "lon", "lat_bnds", "lon_bnds"]
    return [
        name for name in names
        if _get_local_data_shape(grid, name) is not None and _get_local_data_shape(auxiliary, name) is not None
    ]


def _get_local_data_shape(nessy, data_name):
    data_info = getattr(nessy, data_name, None)
    if data_info is None:
        data_info = _get_variables(nessy).get(data_name)
    if isinstance(data_info, dict):
        data_info = data_info.get("data")
    if data_info is None:
        return None
    if hasattr(data_info, "shape"):
        return tuple(data_info.shape)
    return (len(data_info),)
