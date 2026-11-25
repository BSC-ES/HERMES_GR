#!/usr/bin/env python

# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

"""Country-based masking utilities."""

import os
import re
import numpy as np
from hermes_gr.config import precision, log_message
from hermes_gr.tools.country_codes import (
    iso3_to_iso2,
    normalize_iso2,
    normalize_iso3,
    normalize_iso3_mapping,
    validate_iso3_codes,
)
from hermes_gr.tools.auxiliary_consistency import check_aux_file
from nes import open_netcdf, Nes

COUNTRY_SHP_VAR = "ISO"
COUNTRY_ISO2_VAR = "country_iso2"
NAN_MASKING_VALUE = "UNK"  # Value used in the country mask to indicate no data (probably sea)


class Masking(object):
    """
    Masking class instance.

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
        Masking class initialization.

        Parameters
        ----------
        dst_grid : Nes
            Destination grid
        src_grid_path : str or Nes
            Path to a single file with the source grid definition or an already opened Nes object.
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
        self.warning_messages = []

        self.mask_factors = self.get_mask_factors(dst_grid, src_grid_path, countries_shapefile, world_mask_file)

    @staticmethod
    def _get_nes_object_to_compare(dst_grid, src_grid_path):
        """
        Get the NES object used to validate an existing country mask auxiliary file.

        Parameters
        ----------
        dst_grid : Nes
            Destination grid Nes object.
        src_grid_path : str or Nes
            Path to the grid of the source emission inventory or an already opened Nes object.

        Returns
        -------
        Nes
            Nes object used as reference for auxiliary file consistency checks.
        """
        if isinstance(src_grid_path, Nes):
            return src_grid_path

        return open_netcdf(
            path=src_grid_path,
            comm=dst_grid.comm,
            parallel_method=dst_grid.parallel_method,
            balanced=dst_grid.balanced,
        )

    @staticmethod
    def parse_factor_values(values):
        """
        Obtain the factor mask values.

        Parameters
        ----------
        values : str
            Factor mask to parse

        Returns
        -------
        dict
            Factor masks where country is the key and the factor the value
        """

        def parse_factor(value, raw_value):
            try:
                return precision(value)
            except ValueError:
                raise ValueError(
                    "Invalid factor_mask value '{value}' in '{raw_value}'. "
                    "The factor_mask column must contain numeric scaling factors, "
                    "for example '0.5' or 'ESP 0.5, FRA 1.2'. When using country "
                    "codes, each code must be followed by a numeric factor. If the "
                    "value was intended as a list of countries without scaling "
                    "factors, check whether it should be placed in the regrid_mask "
                    "column instead."
                    .format(value=value, raw_value=raw_value)
                )

        if not isinstance(values, str):
            return None
        raw_values = values
        values = list(map(str, re.split(" , |, | ,|,", values)))
        scale_dict = {"default": 1}
        if len(values) == 1:
            try:
                scale_dict["default"] = parse_factor(values[0], raw_values)
            except ValueError:
                res = list(map(str, re.split("{0}{0}|{0}".format(" "), values[0])))
                if len(res) < 2:
                    raise
                scale_dict[res[0]] = parse_factor(res[1], raw_values)
        else:
            for element in values:
                res = list(map(str, re.split("{0}{0}|{0}".format(" "), element)))
                if len(res) < 2:
                    raise ValueError(
                        "Invalid factor_mask element '{element}' in '{raw_value}'. "
                        "Expected a numeric factor or a '<region> <factor>' pair."
                        .format(element=element, raw_value=raw_values)
                    )
                scale_dict[res[0]] = parse_factor(res[1], raw_values)

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
        Get the country data used to calculate mask factors.

        Parameters
        ----------
        dst_grid : Nes
            Destination grid Nes object
        src_grid_path : str or Nes
            Path to the grid of the source emission inventory or an already opened Nes object.
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
            nes_object_to_compare = self._get_nes_object_to_compare(dst_grid, src_grid_path)
            if not check_aux_file(
                nessy,
                nes_object_to_compare=nes_object_to_compare,
                var_list=[COUNTRY_SHP_VAR],
            ):
                raise ValueError(
                    f"Existing world mask auxiliary file is not consistent with the expected country mask structure "
                    f"or dimensions: {world_mask_file}. Expected variable '{COUNTRY_SHP_VAR}' and the same domain "
                    "shape/decomposition as the source inventory grid. Remove this auxiliary file or "
                    "regenerate the auxiliary directory."
                )
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

    @staticmethod
    def _normalize_country_code(country_code):
        """
        Normalize a country mask value before comparing it with user-provided ISO3 codes.

        Parameters
        ----------
        country_code : str, bytes, float, list, tuple or np.ndarray
            Country code value read from the country mask or provided by the user.

        Returns
        -------
        str or None
            Normalized country code, or None for missing/no-data values.
        """
        if country_code is None:
            return None
        if isinstance(country_code, (list, tuple, np.ndarray)):
            country_code = "".join(
                Masking._country_code_to_text(country_code_part)
                for country_code_part in np.asarray(country_code).reshape(-1)
            )

        return normalize_iso3(country_code)

    @classmethod
    def _country_code_to_text(cls, country_code):
        """
        Convert a raw country code value to text without filtering no-data codes.

        Parameters
        ----------
        country_code : str, bytes, float, list, tuple or np.ndarray
            Country code value read from the country mask.

        Returns
        -------
        str
            Text representation of the country code value.
        """
        country_code = cls._normalize_country_code(country_code)
        if country_code is None:
            return ""
        return country_code

    @classmethod
    def _is_char_encoded_country_data(cls, country_data):
        """
        Check whether country data is stored as characters in its last dimension.

        Parameters
        ----------
        country_data : np.ndarray
            Country mask data array.

        Returns
        -------
        bool
            True when the last dimension stores one character per element.
        """
        if country_data.ndim < 1 or country_data.shape[-1] <= 1:
            return False

        sample_values = country_data.reshape(-1)[:100]
        sample_lengths = [
            len(cls._country_code_to_text(country_code))
            for country_code in sample_values
        ]
        sample_lengths = [length for length in sample_lengths if length > 0]

        return len(sample_lengths) > 0 and max(sample_lengths) == 1

    @classmethod
    def _get_country_code_array(cls, country_data):
        """
        Get country mask data as an array of normalized country code strings.

        Parameters
        ----------
        country_data : np.ndarray
            Country mask data array.

        Returns
        -------
        np.ndarray
            Country mask data array with complete country codes as values.
        """
        country_data = np.asarray(country_data)
        if not cls._is_char_encoded_country_data(country_data):
            normalize = np.vectorize(cls._normalize_country_code, otypes=[object])
            return normalize(country_data)

        country_codes = np.empty(country_data.shape[:-1], dtype=object)
        for index in np.ndindex(country_data.shape[:-1]):
            country_codes[index] = cls._normalize_country_code(
                "".join(cls._country_code_to_text(country_code) for country_code in country_data[index])
            )
        return country_codes

    @classmethod
    def _get_domain_country_codes(cls, country_data, comm=None):
        """
        Get the country codes represented in the country mask.

        Parameters
        ----------
        country_data : np.ndarray
            Country mask data array.
        comm : mpi4py.MPI.Comm, optional
            MPI communicator used to gather country codes from all ranks.

        Returns
        -------
        set
            Country codes present in the global country mask domain.
        """
        country_codes = set()
        for country_code in cls._get_country_code_array(country_data).reshape(-1):
            if country_code is not None:
                country_codes.add(country_code)

        if comm is not None and hasattr(comm, "allgather"):
            # Each rank only sees a domain slice, so warnings must use the global set of ISO3 codes.
            all_country_codes = comm.allgather(country_codes)
            country_codes = set()
            for rank_country_codes in all_country_codes:
                country_codes.update(rank_country_codes)
        return country_codes

    @classmethod
    def _get_destination_country_iso2_codes(cls, grid, comm=None):
        """
        Get ISO2 country codes represented in the destination grid.

        Parameters
        ----------
        grid : Nes
            Destination grid.
        comm : mpi4py.MPI.Comm, optional
            MPI communicator used to gather country codes from all ranks.

        Returns
        -------
        set or None
            ISO2 country codes present in the global destination domain, or None
            when the destination grid has no country information.
        """
        if grid is None:
            return None

        if COUNTRY_ISO2_VAR in getattr(grid, "variables", {}):
            if grid.variables[COUNTRY_ISO2_VAR]["data"] is None and hasattr(grid, "load"):
                grid.load(COUNTRY_ISO2_VAR)
            country_data = grid.variables[COUNTRY_ISO2_VAR]["data"]
        elif hasattr(grid, "shapefile") and COUNTRY_ISO2_VAR in grid.shapefile:
            country_data = grid.shapefile[COUNTRY_ISO2_VAR].values
        else:
            return None
        if country_data is None:
            return None

        country_codes = set()
        normalize = np.vectorize(normalize_iso2, otypes=[object])
        for country_code in normalize(country_data).reshape(-1):
            if country_code is not None:
                country_codes.add(country_code)

        if comm is not None and hasattr(comm, "allgather"):
            all_country_codes = comm.allgather(country_codes)
            country_codes = set()
            for rank_country_codes in all_country_codes:
                country_codes.update(rank_country_codes)
        return country_codes

    @classmethod
    def _warn_missing_country_codes(cls, mask_info, country_data, mask_name, comm=None, master=True, dst_grid=None):
        """
        Warn when the mask contains country codes that are absent from the country mask domain.

        Parameters
        ----------
        mask_info : dict or None
            Mask information with country codes as keys and mask factors as values.
        country_data : np.ndarray
            Country mask data array.
        mask_name : str
            Name of the mask being validated.
        comm : mpi4py.MPI.Comm, optional
            MPI communicator used to gather country codes from all ranks.
        master : bool
            Indicates if the current rank should emit the warning.
        dst_grid : Nes, optional
            Destination grid used to warn based on the output domain when it has
            country information.

        Returns
        -------
        None
        """
        if mask_info is None:
            return

        requested_codes = {
            cls._normalize_country_code(country_code)
            for country_code in mask_info
            if country_code != "default"
        }
        requested_codes.discard(None)

        destination_iso2_codes = cls._get_destination_country_iso2_codes(dst_grid, comm=comm)
        if destination_iso2_codes is None:
            missing_codes = sorted(requested_codes.difference(cls._get_domain_country_codes(country_data, comm=comm)))
            domain_name = "country mask/domain"
        else:
            missing_codes = sorted(
                country_code
                for country_code in requested_codes
                if iso3_to_iso2(country_code) not in destination_iso2_codes
            )
            domain_name = "output domain"
        if len(missing_codes) > 0:
            message = (
                "WARNING: {mask_name} contains ISO3 codes that are not present in the {domain_name}: "
                "{missing_codes}. These codes are valid ISO3 codes, but they are outside that domain and "
                "will not affect the mask.".format(
                    mask_name=mask_name,
                    domain_name=domain_name,
                    missing_codes=", ".join(missing_codes),
                )
            )
            if master:
                log_message(message, level=7)
            return message
        return None

    def calculate_mask_array(self, mask_info, countries, mask_name, dst_grid=None):
        """
        Calculate 4D (1, 1, lat, lon) float array with the mask values.

        Parameters
        ----------
        mask_info : dict
            Information of the masking with the country as key and the factor as value
        countries : Nes
            Countries information.
        mask_name : str
            Name of the mask being calculated, used in warning messages.
        dst_grid : Nes, optional
            Destination grid used to detect mask codes outside the output domain.

        Returns
        -------
        np.ndarray
            4D float array with the mask values (1, 1, lat, lon)
        """
        if mask_info is None:
            return None

        if not hasattr(self, "warning_messages"):
            self.warning_messages = []

        mask_info = normalize_iso3_mapping(mask_info)
        validate_iso3_codes(
            (country_code for country_code in mask_info if country_code != "default"),
            context=mask_name,
        )

        if countries.variables[COUNTRY_SHP_VAR]["data"] is None:
            countries.load(COUNTRY_SHP_VAR)
        country_data = self._get_country_code_array(countries.variables[COUNTRY_SHP_VAR]["data"])
        warning_message = self._warn_missing_country_codes(
            mask_info,
            country_data,
            mask_name,
            comm=getattr(countries, "comm", None),
            master=getattr(countries, "master", True),
            dst_grid=dst_grid,
        )
        if warning_message is not None:
            self.warning_messages.append(warning_message)

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
            mask_array = np.ones(shape_4d, dtype=precision) * mask_info["default"]
            for country, value in mask_info.items():
                country = self._normalize_country_code(country)
                if country is None:
                    continue
                try:
                    mask_array[country_data == country] = value
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
        Calculate the 2D mask to apply.

        Parameters
        ----------
        dst_grid : Nes
            Destination grid Nes object
        src_grid_path : str or Nes
            Path to the grid of the source emission inventory or an already opened Nes object.
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
        result = self.calculate_mask_array(self.factors_mask_values, countries, "factor_mask", dst_grid=dst_grid)
        r_mask = self.calculate_mask_array(self.regrid_mask_values, countries, "regrid_mask", dst_grid=dst_grid)
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
