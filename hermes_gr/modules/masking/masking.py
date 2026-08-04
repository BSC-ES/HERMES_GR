#!/usr/bin/env python

# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.


import os
import re
import numpy as np
from hermes_gr.config import precision, log_message
from nes import open_netcdf, Nes

COUNTRY_SHP_VAR = "ISO"
NAN_MASKING_VALUE = "UNK"  # Value used in the country mask to indicate no data (probably sea)


class Masking(object):
    """
    Masking class instance

    Attributes
    ----------
    factors_mask_values : dict
        Dictionary with the country name as key and the scaling factor as value. It also contains the default one.
    regrid_mask_values : dict
        Dictionary with the country name as key and the  factor (1 or 0) as value. It also contains the default one.
    """

    def __init__(
        self,
        dst_grid,
        src_grid_path,
        factors_mask_values=None,
        regrid_mask_values=None,
        world_mask_file=None,
        countries_shapefile=None,
    ):
        """
        Masking class initialization

        Parameters
        ----------
        dst_grid : Nes
            Destination grid
        src_grid_path : str , Nes
            Path to a single file with the source grid definition
        factors_mask_values : str
            Scaling factors
        regrid_mask_values : str
            Mask factors
        world_mask_file : str
            Path to the CountryMask file that will be used as auxiliary file
        countries_shapefile : str
            Path to the shapefile that contains the 'ISO' colum with the country codes.
        """
        log_message("Creating mask.", level=2)
        if countries_shapefile is None:
            raise ValueError(f"Countries shapefile cannot be None. Check 'countries_shapefile' argument.")
        self.factors_mask_values = self.parse_factor_values(factors_mask_values)
        self.regrid_mask_values = self.parse_masking_values(regrid_mask_values)

        self.mask_factors = self.get_mask_factors(dst_grid, src_grid_path, countries_shapefile, world_mask_file)

    @staticmethod
    def parse_factor_values(values):
        """
        Obtain the factor mask values

        Parameters
        ----------
        values : str
            Factor mask to parse

        Returns
        -------
        dict
            Factor masks where country is the key and the factor the value
        """

        if not isinstance(values, str):
            return None
        values = list(map(str, re.split(" , |, | ,|,", values)))
        scale_dict = {"default": 1}
        if len(values) == 1:
            try:
                scale_dict["default"] = precision(values[0])
            except ValueError:
                res = list(map(str, re.split("{0}{0}|{0}".format(" "), values[0])))
                scale_dict[res[0]] = precision(res[1])
        else:
            for element in values:
                res = list(map(str, re.split("{0}{0}|{0}".format(" "), element)))
                scale_dict[res[0]] = precision(res[1])

        log_message(f"Creating factor mask: {scale_dict}.", level=3)

        return scale_dict

    @staticmethod
    def parse_masking_values(values):
        """
        Obtain the list of countries to take into account.

        Parameters
        ----------
        values : str
            String values to parse

        Returns
        -------
        dict
            Factor masks where country is the key and the factor the value
        """
        if not isinstance(values, str):
            return None
        values = list(map(str, re.split(" , |, | ,|,| ", values)))
        if values[0] == "+":
            adding = True
        elif values[0] == "-":
            adding = False
        else:
            if len(values) > 0:
                log_message("WARNING: The list of masking does not start with '+' or '-'. Ignoring mask.", level=7)
            return None

        if adding:
            mask_factors = {"default": 0}
            for country in values[1:]:
                mask_factors[country] = 1
        else:
            mask_factors = {"default": 1}
            for country in values[1:]:
                mask_factors[country] = 0

        log_message(f"Creating country mask: {mask_factors}.", level=3)
        return mask_factors

    def get_countries_data(
        self,
        dst_grid,
        src_grid_path,
        countries_shapefile,
        world_mask_file,
        do_write=True,
    ):
        """

        Parameters
        ----------
        dst_grid : Nes
            Destination grid Nes object
        src_grid_path : str
            Path to the grid of the source emission inventory
        countries_shapefile : str
            Path to the shapefile that contains the country information
        world_mask_file : str
            Auxiliary file path to the emission inventory country mask
        do_write : bool
            Indicates if you want to write the country mask or not.

        Returns
        -------
        Nes
            Nes object with the country information
        """

        if self.factors_mask_values is None and self.regrid_mask_values is None:
            return None

        if os.path.exists(world_mask_file):
            nessy = open_netcdf(
                path=world_mask_file,
                comm=dst_grid.comm,
                parallel_method=dst_grid.parallel_method,
                balanced=dst_grid.balanced,
            )
            nessy.load()
        else:
            if isinstance(src_grid_path, Nes):
                nessy = src_grid_path.copy(copy_vars=False)
            else:
                nessy = open_netcdf(
                    path=src_grid_path,
                    comm=dst_grid.comm,
                    parallel_method=dst_grid.parallel_method,
                    balanced=dst_grid.balanced,
                )
            nessy.variables = {}  # Erase lazy variables
            nessy.cell_measures = {}
            nessy.create_shapefile()

            log_message("Stating spatial joint", level=3)

            nessy.spatial_join(
                countries_shapefile,
                method="intersection",
                var_list=[COUNTRY_SHP_VAR],
                info=True,
            )
            # nessy.spatial_join(countries_shapefile, method='centroid', var_list=[COUNTRY_SHP_VAR], info=True)
            nessy.shapefile[COUNTRY_SHP_VAR] = nessy.shapefile[COUNTRY_SHP_VAR].fillna(NAN_MASKING_VALUE)
            nessy.variables[COUNTRY_SHP_VAR] = {
                "data": nessy.shapefile[COUNTRY_SHP_VAR].values.reshape(
                    (1, 1, nessy.lat["data"].shape[0], nessy.lon["data"].shape[-1])),
                "dtype": str,
            }

            nessy.set_strlen(3)
            log_message("Spatial join done", level=3)

            if do_write:
                # Transform to object
                # nessy.variables[COUNTRY_SHP_VAR]["data"] = nessy.variables[COUNTRY_SHP_VAR]["data"].astype(object)

                nessy.to_netcdf(world_mask_file, serial=True)
                nessy.comm.Barrier()
                log_message(f"Write done at {world_mask_file}", level=3)

            # Back to nan
            mask = nessy.variables[COUNTRY_SHP_VAR]["data"] == NAN_MASKING_VALUE
            nessy.variables[COUNTRY_SHP_VAR]["data"][mask] = np.nan

        return nessy

    @staticmethod
    def get_next_mask_varname(nessy):
        """
        Find the available variable name to be used.

        Parameters
        ----------
        nessy : Nes
            Object with the masking properties
        Returns
        -------
        str
            Available variable name
        """
        var_pattern = "mask_<count>"
        count = 0
        while True:
            var_name = var_pattern.replace("<count>", f"{count}")
            if var_name not in nessy.variables.keys():
                return var_name
            else:
                count += 1

    def calculate_mask_array(self, mask_info, countries):
        """
        Calculate 4D (1, 1, lat, lon) float array with the mask values

        Parameters
        ----------
        mask_info : dict
            Information of the masking with the country as key and the factor as value
        countries : Nes
            Countries information.

        Returns
        -------
        np.ndarray
            4D float array with the mask values (1, 1, lat, lon)
        """
        if mask_info is None:
            return None

        mask_array = None
        found = False
        for var_name, var_info in countries.variables.items():
            if "mask_info" in var_info.keys() and var_info["mask_info"] == str(mask_info):
                countries.load(var_name)
                mask_array = countries.variables[var_name]["data"]
                found = True
                break

        shape_4d = (1, 1, countries.lat["data"].shape[0], countries.lon["data"].shape[-1], )

        if not found:
            if countries.variables[COUNTRY_SHP_VAR]["data"] is None:
                countries.load(COUNTRY_SHP_VAR)

            mask_array = np.ones(shape_4d, dtype=precision) * mask_info["default"]
            for country, value in mask_info.items():
                try:
                    mask_array[countries.variables[COUNTRY_SHP_VAR]["data"] == country] = value
                except IndexError as e:
                    raise e

            var_name = self.get_next_mask_varname(countries)
            countries.variables[var_name] = {
                "data": mask_array,
                "mask_info": str(mask_info),
                "dtype": precision,
            }

        return mask_array

    def get_mask_factors(
        self, dst_grid, src_grid_path, countries_shapefile, world_mask_file
    ):
        """
        Calculate the 2D mask to apply

        Parameters
        ----------
        dst_grid : Nes
            Destination grid Nes object
        src_grid_path : str
            Path to the grid of the source emission inventory
        countries_shapefile : str
            Path to the shapefile that contains the country information
        world_mask_file : str
            Auxiliary file path to the emission inventory country mask

        Returns
        -------
        np.array
            2D array with the mask to apply
        """
        if self.factors_mask_values is None and self.regrid_mask_values is None:
            return None
        else:
            countries = self.get_countries_data(
                dst_grid,
                src_grid_path,
                countries_shapefile,
                world_mask_file,
                do_write=True,
            )
            # mask_vars = set(list(countries.variables.keys()))
        result = self.calculate_mask_array(self.factors_mask_values, countries)
        r_mask = self.calculate_mask_array(self.regrid_mask_values, countries)
        if result is None:
            result = r_mask
        else:
            if r_mask is not None:
                result *= r_mask
                del r_mask

        # if len(set(list(countries.variables.keys())).difference(mask_vars)) > 0:
        #     countries.to_netcdf(world_mask_file, serial=True)
        #     log_message(f"Write done at {world_mask_file}", level=3)

        return result
