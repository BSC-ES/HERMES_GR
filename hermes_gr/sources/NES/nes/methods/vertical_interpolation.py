#!/usr/bin/env python
# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import sys
from numpy import nan, flip, cumsum, nanmean, empty, ndarray, ma, float32, float64, array, interp, where
from scipy.interpolate import interp1d
from copy import copy


def add_4d_vertical_info(self, info_to_add):
    """
    To add the vertical information from other source.

    Parameters
    ----------
    self : nes.Nes
        Source Nes object.
    info_to_add : nes.Nes, str
        Nes object with the vertical information as variable or str with the path to the NetCDF file that contains
        the vertical data.
    """

    vertical_var = list(self.concatenate(info_to_add))
    try:
        self.vertical_var_name = vertical_var[0]
    except IndexError:
        raise ValueError(f"{info_to_add} variable set as vertical information does not exist.")

    return None


def __parse_extrapolate(extrapolate) -> tuple:
    """
    Parses the "extrapolate" parameter and returns a tuple representing the extrapolation options.

    Parameters
    ----------
    extrapolate : bool or tuple or None or number or NaN
        If bool:
            - If True, both extrapolation options are set to "extrapolate".
            - If False, extrapolation options are set to ("bottom", "top").
        If tuple:
            - The first element represents the extrapolation option for the lower bound.
            - The second element represents the extrapolation option for the upper bound.
            - If any element is bool:
                - If True, it represents "extrapolate".
                - If False:
                    - If it"s the first element, it represents "bottom".
                    - If it"s the second element, it represents "top".
            - If any element is None, it is replaced with numpy.nan.
            - Other numeric values are kept as they are.
            - If any element is NaN, it is kept as NaN.
        If None:
            - Both extrapolation options are set to (NaN, NaN).
        If number:
            - Both extrapolation options are set to the provided number.
        If NaN:
            - Both extrapolation options are set to NaN.

    Returns
    -------
    tuple
        A tuple representing the extrapolation options. If the input is invalid, it returns
        ("extrapolate", "extrapolate").
    """
    if isinstance(extrapolate, bool):
        if extrapolate:
            extrapolate_options = ("extrapolate", "extrapolate")
        else:
            extrapolate_options = ("bottom", "top")
    elif isinstance(extrapolate, tuple):
        extrapolate_options = [None, None]
        for i in range(len(extrapolate)):
            if isinstance(extrapolate[i], bool):
                if extrapolate[i]:
                    extrapolate_options[i] = "extrapolate"
                else:
                    if i == 0:
                        extrapolate_options[i] = "bottom"
                    else:
                        extrapolate_options[i] = "top"
            elif extrapolate[i] is None:
                extrapolate_options[i] = nan
            else:
                extrapolate_options[i] = extrapolate[i]
        extrapolate_options = tuple(extrapolate_options)
    elif extrapolate is None:
        extrapolate_options = ("bottom", "top")
    else:
        extrapolate_options = (extrapolate, extrapolate)

    return extrapolate_options


def interpolate_vertical(self, new_levels, new_src_vertical=None, kind="linear", extrapolate_options=False, info=None,
                         overwrite=False):
    """
    Vertical interpolation.

    Parameters
    ----------
    self : Nes
        Source Nes object.
    new_levels : List
        A List of new vertical levels.
    new_src_vertical : nes.Nes, str
        Nes object with the vertical information as variable or str with the path to the NetCDF file that contains
        the vertical data.
    kind : str
        Vertical methods type.
    extrapolate_options : bool or tuple or None or number or NaN
        If bool:
            - If True, both extrapolation options are set to "extrapolate".
            - If False, extrapolation options are set to ("bottom", "top").
        If tuple:
            - The first element represents the extrapolation option for the lower bound.
            - The second element represents the extrapolation option for the upper bound.
            - If any element is bool:
                - If True, it represents "extrapolate".
                - If False:
                    - If it"s the first element, it represents "bottom".
                    - If it"s the second element, it represents "top".
            - If any element is None, it is replaced with numpy.nan.
            - Other numeric values are kept as they are.
            - If any element is NaN, it is kept as NaN.
        If None:
            - Both extrapolation options are set to (NaN, NaN).
        If number:
            - Both extrapolation options are set to the provided number.
        If NaN:
            - Both extrapolation options are set to NaN.
    info: None, bool
        Indicates if you want to print extra information.
    overwrite: bool
        Indicates if you want to compute the vertical interpolation in the same object or not.
    """
    src_levels_aux = None
    fill_value = None

    extrapolate_options = __parse_extrapolate(extrapolate_options)
    do_extrapolation = "extrapolate" in extrapolate_options

    if len(self.lev) == 1:
        raise RuntimeError("1D data cannot be vertically interpolated.")
    if not overwrite:
        self = self.copy(copy_vars=True)
    if info is None:
        info = self.info

    if new_src_vertical is not None:
        self.add_4d_vertical_info(new_src_vertical)
    if new_levels[0] > new_levels[-1]:
        ascendant = False
    else:
        ascendant = True

    nz_new = len(new_levels)

    if self.vertical_var_name is None:
        # To use current level data
        current_level = True
        # Checking old order
        src_levels = self.lev["data"]
        if src_levels[0] > src_levels[-1]:
            if not ascendant:
                do_flip = False
            else:
                do_flip = True
                src_levels = flip(src_levels)
        else:
            if ascendant:
                do_flip = False
            else:
                do_flip = True
                src_levels = flip(src_levels)
    else:
        current_level = False
        src_levels = self.variables[self.vertical_var_name]["data"]
        if self.vertical_var_name == "layer_thickness":
            src_levels = flip(cumsum(flip(src_levels, axis=1), axis=1))
        else:
            # src_levels = flip(src_levels, axis=1)
            pass
        # Checking old order
        if nanmean(src_levels[:, 0, :, :]) > nanmean(src_levels[:, -1, :, :]):
            if not ascendant:
                do_flip = False
            else:
                do_flip = True
                src_levels = flip(src_levels, axis=1)
        else:
            if ascendant:
                do_flip = False
            else:
                do_flip = True
                src_levels = flip(src_levels, axis=1)

    # Loop over variables
    for var_name in self.variables.keys():
        if self.variables[var_name]["data"] is None:
            # Load data if it is not loaded yet
            self.load(var_name)

        if var_name != self.vertical_var_name:
            if do_flip:
                self.variables[var_name]["data"] = flip(self.variables[var_name]["data"], axis=1)
            if info and self.master:
                print("\t{var} vertical methods".format(var=var_name))
                sys.stdout.flush()
            nt, nz, ny, nx = self.variables[var_name]["data"].shape

            if self.variables[var_name]["data"].dtype != float64 and self.variables[var_name]["data"].dtype != float32:
                dst_data = empty((nt, nz_new, ny, nx), dtype=float64)
            else:
                dst_data = empty((nt, nz_new, ny, nx), dtype=self.variables[var_name]["data"].dtype)
            for t in range(nt):
                # if info and self.rank == self.size - 1:
                if self.info and self.master:
                    print("\t\t{3} time step {0} ({1}/{2}))".format(self.time[t], t + 1, nt, var_name))
                    sys.stdout.flush()
                for j in range(ny):
                    for i in range(nx):
                        if len(src_levels.shape) == 1:
                            # To use 1D level information
                            curr_level_values = src_levels
                        else:
                            # To use 4D level data
                            curr_level_values = src_levels[t, :, j, i]
                        try:
                            # Check if all values are identical or masked
                            if ((isinstance(curr_level_values, ndarray) and
                                 (curr_level_values == curr_level_values[0]).all()) or
                                    (isinstance(curr_level_values, ma.core.MaskedArray) and
                                     curr_level_values.mask.all())):
                                kind = "slinear"
                            else:
                                kind = kind  # "cubic"

                            # Filtering filling values to extrapolation
                            fill_value = [nan, nan]
                            if "bottom" in extrapolate_options:
                                if ascendant:
                                    fill_value[0] = float64(self.variables[var_name]["data"][t, 0, j, i])
                                else:
                                    fill_value[0] = float64(self.variables[var_name]["data"][t, -1, j, i])
                            else:
                                fill_value[0] = extrapolate_options[0]
                            if "top" in extrapolate_options:
                                if ascendant:
                                    fill_value[1] = float64(self.variables[var_name]["data"][t, -1, j, i])
                                else:
                                    fill_value[1] = float64(self.variables[var_name]["data"][t, 0, j, i])
                            else:
                                fill_value[1] = extrapolate_options[1]
                            fill_value = tuple(fill_value)

                            # We force the methods with float64 to avoid negative values
                            # We don"t know why the negatives appears with float34
                            if current_level:
                                # 1D vertical component
                                src_levels_aux = src_levels
                            else:
                                # 4D vertical component
                                src_levels_aux = src_levels[t, :, j, i]

                            if kind == "linear" and ascendant and not do_extrapolation:
                                dst_data[t, :, j, i] = array(
                                    interp(new_levels,
                                           array(src_levels_aux, dtype=float64),
                                           array(self.variables[var_name]["data"][t, :, j, i], dtype=float64),
                                           left=fill_value[0], right=fill_value[1]),
                                    dtype=self.variables[var_name]["data"].dtype)
                            else:
                                if not do_extrapolation:
                                    dst_data[t, :, j, i] = array(
                                        interp1d(array(src_levels_aux, dtype=float64),
                                                 array(self.variables[var_name]["data"][t, :, j, i], dtype=float64),
                                                 kind=kind,
                                                 bounds_error=False,
                                                 fill_value=fill_value)(new_levels),
                                        dtype=self.variables[var_name]["data"].dtype)
                                else:
                                    # If extrapolation first we need to extrapolate all (below & above)
                                    dst_data[t, :, j, i] = array(
                                        interp1d(array(src_levels_aux, dtype=float64),
                                                 array(self.variables[var_name]["data"][t, :, j, i],
                                                       dtype=float64),
                                                 kind=kind,
                                                 bounds_error=False,
                                                 fill_value="extrapolate")(new_levels),
                                        dtype=self.variables[var_name]["data"].dtype)
                                    # Check values below the lower vertical level
                                    if fill_value[0] != "extrapolate":
                                        if ascendant:
                                            idx_bellow = where(new_levels < src_levels_aux[0])
                                        else:
                                            idx_bellow = where(new_levels > src_levels_aux[0])
                                        dst_data[t, idx_bellow, j, i] = fill_value[0]
                                    # Check values above the upper vertical level
                                    if fill_value[1] != "extrapolate":
                                        if ascendant:
                                            idx_above = where(new_levels > src_levels_aux[-1])
                                        else:
                                            idx_above = where(new_levels < src_levels_aux[-1])
                                        dst_data[t, idx_above, j, i] = fill_value[1]
                        # catch interp1d unique values error
                        except ValueError as e:
                            if str(e) == "Expect x to not have duplicates":
                                dst_data[t, :, j, i] = empty(len(new_levels), dtype=float64)
                            else:
                                print("time lat lon", t, j, i)
                                print("***********************")
                                print("LEVELS", src_levels_aux)
                                print("DATA", array(self.variables[var_name]['data'][t, :, j, i], dtype=float64))
                                print("METHOD", kind)
                                print("FILL_VALUE", fill_value)
                                print("+++++++++++++++++++++++")
                                raise Exception(str(e))

                        except Exception as e:
                            print("time lat lon", t, j, i)
                            print("***********************")
                            print("LEVELS", src_levels_aux)
                            print("DATA", array(self.variables[var_name]["data"][t, :, j, i], dtype=float64))
                            print("METHOD", kind)
                            print("FILL_VALUE", fill_value)
                            print("+++++++++++++++++++++++")
                            raise Exception(str(e))
                        # if level_array is not None:
                        #     dst_data[t, :, j, i] = array(f(level_array), dtype=float32)

            self.variables[var_name]["data"] = copy(dst_data)
            # print(self.variables[var_name]["data"])

    # Update level information
    new_lev_info = {"data": array(new_levels)}
    if "positive" in self.lev.keys():
        # Vertical level direction
        if flip:
            self.reverse_level_direction()
        new_lev_info["positive"] = self.lev["positive"]

    if self.vertical_var_name is not None:
        for var_attr, attr_info in self.variables[self.vertical_var_name].items():
            if var_attr not in ["data", "dimensions", "crs", "grid_mapping"]:
                new_lev_info[var_attr] = copy(attr_info)
        self.free_vars(self.vertical_var_name)
        self.vertical_var_name = None

    self.set_levels(new_lev_info)

    # Remove original file information
    self.__ini_path = None
    self.dataset = None
    self.dataset = None

    return self
