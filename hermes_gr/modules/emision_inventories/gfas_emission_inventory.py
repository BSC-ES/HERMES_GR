#!/usr/bin/env python

# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import os
import re
import numpy as np
from nes import open_netcdf
from hermes_gr.modules.masking.masking import Masking
from hermes_gr.config import precision, log_message
from .emission_inventory import EmissionInventory


class GfasEmissionInventory(EmissionInventory):

    def __init__(self, options, grid, current_date, inventory_name, source_type, sector, pollutants, inputs_path,
                 frequency, vertical_output_profile,
                 reference_year=2010, coverage="global", factors=None, regrid_mask=None, p_vertical=None, p_month=None,
                 p_week=None, p_day=None, p_hour=None, p_speciation=None, countries_shapefile=None):

        self.approach = self.parse_profile_gfas(p_vertical, "approach")
        self.method = self.parse_profile_gfas(p_vertical, "method")
        self.altitude = self.get_altitude()
        self.layer_widths = grid.lev["data"].copy()
        self.layer_widths[1:] -= self.layer_widths[:-1]

        super(GfasEmissionInventory, self).__init__(
            options, grid, current_date, inventory_name, source_type, sector, pollutants, inputs_path, frequency,
            vertical_output_profile,
            reference_year=reference_year, coverage=coverage, factors=factors, regrid_mask=regrid_mask,
            p_vertical=None, p_month=p_month, p_week=p_week, p_day=p_day, p_hour=p_hour, p_speciation=p_speciation,
            countries_shapefile=countries_shapefile)

        self.vertical = None

    @staticmethod
    def print_ini_message():
        """
        Print initial message
        """
        log_message("Creating GFAS area source emission inventory.", level=3)
        return None

    def prepare_input_emissions(self, write_aux_file=True):
        """
        Prepare the emissions but without writing the auxiliary file.

        1. Combine the pollutants of the source
        2. Apply the masks
        2. Perform the Regrid
        3. DO NOT save the regridded file as auxiliary files

        Parameters
        ----------
        write_aux_file : bool
            Indicates if you want to save the regridded emissions as auxiliary file.
        """

        super(GfasEmissionInventory, self).prepare_input_emissions(write_aux_file=False)

        return self.emissions

    def read_input_emissions(self):
        """
        Prepare tje emissions for the regridding

        1. Read the emissions
        2. Apply mask
        3. Distribute vertically

        The resultant emissions are stored in `self.emissions`
        """

        # In the initialization only the first pollutant is read but not loaded.
        self.emissions = open_netcdf(self.src_emis_file_list[0], comm=self.grid.comm, balanced=self.grid.balanced,
                                     parallel_method=self.grid.parallel_method)
        self.emissions.keep_vars(self.input_pollutants + [self.altitude])

        # self.emissions.sel(lat_min=self.grid.get_full_latitudes_boundaries()["data"].min(),
        #                    lat_max=self.grid.get_full_latitudes_boundaries()["data"].max(),
        #                    lon_min=self.grid.get_full_longitudes_boundaries()["data"].min(),
        #                    lon_max=self.grid.get_full_longitudes_boundaries()["data"].max())
        self.emissions.load()

        # Casting emissions to the selected precision
        self.emissions.to_dtype(precision)

        # Masking
        mask_path = os.path.join(os.path.dirname(self.auxiliary_files_path), f"{self.inventory_name}_WorldMask.nc")
        mask_factors = Masking(
            dst_grid=self.grid, src_grid_path=self.emissions, factors_mask_values=self.factors_mask,
            regrid_mask_values=self.regrid_mask, world_mask_file=mask_path,
            countries_shapefile=self.countries_shapefile).mask_factors
        if mask_factors is not None:
            for var_name in self.emissions.variables.keys():
                self.emissions.variables[var_name]["data"] *= mask_factors

        # Vertical distribution
        vertical_factor = self.get_vertical_weights(self.emissions.variables[self.altitude]["data"])
        self.emissions.free_vars(self.altitude)

        for var_name in self.emissions.variables.keys():
            self.emissions.variables[var_name]["data"] = np.array(
                self.emissions.variables[var_name]["data"] * vertical_factor,
                dtype=precision)

        return None

    def get_src_file_list(self, pollutant_list):
        """
        Generates the input file list

        Parameters
        ----------
        pollutant_list : list
            Pollutant list

        Returns
        -------
        List[str]
            List of source file paths
        """
        file_list = [self.get_input_path(pollutant=None)]
        return file_list

    def get_input_path(self, pollutant=None, extension="nc"):
        """
        Completes the path of the NetCDF that contains the needed information of the given pollutant.

        Parameters
        ----------
        pollutant : str , None
            Name of the pollutant of the NetCDF.
        extension : str
            Extension of the input file.

        Returns
        -------
        str
            Full path of the needed NetCDF.
        """
        netcdf_path = os.path.join(self.inputs_path, "multivar", f"ga_{self.date.strftime('%Y%m%d')}.{extension}")
        return netcdf_path

    def get_altitude(self):
        """
        Obtain the altitude variable name depending on the selected method

        Returns
        -------
        str
            Altitude variable name
        """

        if self.method == "sovief":
            alt_var = "apt"
        elif self.method == "prm":
            alt_var = "mami"
        else:
            raise ValueError("ERROR: Only 'sovief' and 'prm' methods are accepted.")

        return alt_var

    @staticmethod
    def parse_profile_gfas(p_vertical, value):
        """
        Parses the vertical profile to obtain the corresponding value

        Parameters
        ----------
        p_vertical : str
            Profile to parse
        value : str
            Value to extract
            Options: "approach" or "method"
        Returns
        -------
        str
            Extracted value
        """
        return_value = None
        aux_list = re.split(", |,| , | ,", p_vertical)
        for element in aux_list:
            aux_value = re.split("=| =|= | = ", element)
            if aux_value[0] == value:
                return_value = aux_value[1]

        return return_value

    def get_vertical_weights(self, altitude_data):
        """
        Compute the input vertical weights

        Parameters
        ----------
        altitude_data : np.ndarray
            2D (4D with 1st and 2nd dim as 1) with the altitude of each emission

        Returns
        -------
        np.array
            3D (4D with first dim as 1) with the vertical distribution.
        """
        weights_3d = np.zeros((1, len(self.grid.lev["data"]), altitude_data.shape[-2], altitude_data.shape[-1]),
                              dtype=precision)

        top_layer = self.get_top_layer_data(altitude_data)

        for y in range(altitude_data.shape[-2]):
            for x in range(altitude_data.shape[-1]):
                weights_3d[0, :, y, x] = self.get_cell_weight_list(altitude_data[0, 0, y, x], top_layer[y, x])

        return weights_3d

    def get_top_layer_data(self, altitude_data):
        """
        Obtain a 2D array with the top vertical layer to allocate the emissions

        Parameters
        ----------
        altitude_data : np.ndarray
            Array with the altitude of the emission

        Returns
        -------
        np.ndarray
            2D array with the top vertical layer where allocate the emissions.
        """
        aux_altitude = altitude_data.copy()
        top_data = np.zeros(aux_altitude.shape, dtype=int)

        for level_i, level_m in enumerate(self.grid.lev["data"]):
            top_data[aux_altitude <= level_m] = level_i
            aux_altitude[aux_altitude <= level_m] = None
        del aux_altitude
        top_data = top_data.reshape((altitude_data.shape[-2], altitude_data.shape[-1]))

        return top_data

    def get_cell_weight_list(self, altitude, top_layer):
        """
        Compute the vertical distribution weights for emissions in a single grid cell.

        This function distributes the emission mass vertically across model layers
        according to the selected approach:

        - "uniform": distributes all emissions proportionally across layers below `top_layer`
        - "50_top": assigns 50% of emissions to `top_layer` and distributes the remaining
          50% proportionally across lower layers

        Parameters
        ----------
        altitude : float
            Effective emission height (not strictly used for normalization; kept for logic consistency).
        top_layer : int
            Index of the highest vertical layer affected by the emission.

        Returns
        -------
        np.ndarray
            1D array of length equal to the number of vertical levels, containing
            normalized weights (summing to 1.0) for vertical emission distribution.

        Notes
        -----
        - The weights are normalized using the sum of layer thicknesses (`layer_widths`)
          rather than `altitude`, to ensure numerical consistency.
        - The function guarantees that the sum of weights is 1.0 (within floating point tolerance).
        """

        # Number of vertical levels
        nlev = len(self.grid.lev["data"])

        # Initialize weights array with zeros
        vertical_weights = np.zeros(nlev, dtype=precision)

        # Safety check: ensure top_layer is within bounds
        if top_layer < 0 or top_layer >= nlev:
            raise IndexError(f"top_layer {top_layer} out of bounds for {nlev} levels")

        # Case 1: emission is entirely within the first layer
        if altitude <= self.grid.lev["data"][0]:
            vertical_weights[0] = 1.0
            return vertical_weights

        # Case 2: distribute emissions depending on selected approach
        if self.approach == "50_top":
            # Assign 50% of emissions to the top layer
            vertical_weights[top_layer] = 0.5
            to_distribute = 0.5

            # Edge case: if top_layer is the first layer
            if top_layer == 0:
                vertical_weights[0] += 0.5
                return vertical_weights

            # Distribute remaining emissions below top_layer
            top_layer -= 1

        elif self.approach == "uniform":
            # Distribute 100% of emissions below top_layer
            to_distribute = 1.0

        else:
            raise ValueError(f"Unknown {self.approach} GFAS vertical approach")

        # Compute total vertical extent (sum of layer thicknesses)
        total_depth = np.sum(self.layer_widths[:top_layer + 1])

        if total_depth <= 0:
            raise ValueError("Total depth must be positive")

        # Distribute emissions proportionally to layer thickness
        for level_i in range(top_layer + 1):
            vertical_weights[level_i] = (
                    to_distribute * self.layer_widths[level_i] / total_depth
            )

        return vertical_weights
