#!/usr/bin/env python

# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import os
from hermes_gr.config import log_message
from .gfas_emission_inventory import GfasEmissionInventory


class GfasHourlyEmissionInventory(GfasEmissionInventory):

    def __init__(self, options, grid, current_date, inventory_name, source_type, sector, pollutants, inputs_path,
                 frequency, vertical_output_profile,
                 reference_year=2010, coverage="global", factors=None, regrid_mask=None, p_vertical=None, p_month=None,
                 p_week=None, p_day=None, p_hour=None, p_speciation=None, countries_shapefile=None):

        super(GfasHourlyEmissionInventory, self).__init__(
            options, grid, current_date, inventory_name, source_type, sector, pollutants, inputs_path, frequency,
            vertical_output_profile,
            reference_year=reference_year, coverage=coverage, factors=factors, regrid_mask=regrid_mask,
            p_vertical=p_vertical, p_month=None, p_week=None, p_day=None, p_hour=None,
            p_speciation=p_speciation, countries_shapefile=countries_shapefile)

    @staticmethod
    def print_ini_message():
        """
        Print initial message
        """
        log_message("Creating GFAS hourly area source emission inventory.", level=3)
        return None

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
        netcdf_path = os.path.join(self.inputs_path, "multivar", f"ga_{self.date.strftime('%Y%m%d_%H')}.{extension}")
        return netcdf_path

    def get_4d_emissions(self):
        self.src_emis_file_list = self.get_src_file_list(None)
        self.prepare_input_emissions(write_aux_file=False)

        if self.speciation is not None:
            self.emissions = self.speciation.do_speciation(self.emissions)

        result = super(GfasHourlyEmissionInventory, self).get_4d_emissions()
        return result
