#!/usr/bin/env python

# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import os
import pandas as pd
import numpy as np
from copy import deepcopy
from hermes_gr.config import precision, log_message
from hermes_gr.config import get_time_stamp, record_time
from hermes_gr.modules.vertical import VerticalDistribution
from hermes_gr.modules.temporal import TemporalDistribution
from hermes_gr.modules.speciation import Speciation
from hermes_gr.modules.masking import Masking
from nes import open_netcdf, concatenate_netcdfs, Nes
from datetime import datetime


class EmissionInventory(object):
    """
    Class that defines the content and the methodology for the area emission inventories

    Attributes
    ----------
    source_type : str

    coverage : str
    grid : Nes
        Destination grid.
    auxiliary_files_path : str
        Path to the auxiliary directory

    date : datetime
        Date to simulate.
    inventory_name : str
        Name of the inventory to use.
    sector : str
        Name of the sector of the inventory to use.

    inputs_path : str
        Path where are stored all the datasets to use. eg: /esarchive/recon/jrc/htapv2/monthly_mean
    input_frequency : str
        Frequency of the inputs. [yearly, monthly, daily]
    reference_year : int
        Year of reference of the information of the dataset.

    emissions : Nes
        Emissions

    """

    def __init__(
        self,
        options,
        grid,
        current_date,
        inventory_name,
        source_type,
        sector,
        pollutants,
        inputs_path,
        input_frequency,
        vertical_output_profile,
        reference_year=2010,
        coverage="global",
        factors=None,
        regrid_mask=None,
        p_vertical=None,
        p_month=None,
        p_week=None,
        p_day=None,
        p_hour=None,
        p_speciation=None,
        countries_shapefile=None,
    ):
        """

        Parameters
        ----------
        options : NameSpace
            Configuration options
        grid : Nes
            Destination grid
        current_date : datetime
            Date to simulate
        inventory_name : str
            Inventory name
        source_type : str
            Source type
        sector : str
            Sector name
        pollutants : List[str]
            List of the pollutant name to take into account.
        inputs_path : str
            Path to the input files
        input_frequency : str
            Frequency of the input emission inventory
        vertical_output_profile : List[float]
            Vertical levels
        reference_year : int
            Emission inventory reference year
        coverage : str
            Global or regional emission inventory coverage
        factors : str
            Description of the scale factors per country. (e.g. SPN 1.5, CHN 3.)
        regrid_mask : str
            Description of the masking countries (adding e.g. + SPN AND) (subtracting e.g. - SPN)
        p_vertical : str or None
            ID of the vertical profile to use.
        p_month : str
            ID of the temporal monthly profile to use.
        p_week : str
            ID of the temporal daily profile to use.
        p_hour : str
            ID of the temporal hourly profile to use.
        p_speciation : str
            ID of the speciation profile to use.
        p_day : str
            Path to the day profile to use.
        countries_shapefile : str
            Path to the shapefile that contains the countries.
        """

        self.print_ini_message()

        # Emission Inventory parameters
        self.source_type = source_type
        self.date = current_date
        self.inventory_name = inventory_name
        self.sector = sector
        self.reference_year = reference_year
        self.coverage = coverage
        self.inputs_path = inputs_path
        self.input_frequency = input_frequency
        self.grid = grid
        self.auxiliary_files_path = options.auxiliary_files_path

        # Profiles
        p_vertical = self.parse_profile(p_vertical)
        p_month = self.parse_profile(p_month)
        p_week = self.parse_profile(p_week)
        p_hour = self.parse_profile(p_hour)
        p_speciation = self.parse_profile(p_speciation)

        self.input_pollutants = pollutants
        self.pollutant_dicts = self.create_pollutants_dicts(pollutants)
        self.src_emis_file_list = self.get_src_file_list(pollutants)

        self.regrid_mask = regrid_mask
        self.factors_mask = factors
        self.countries_shapefile = countries_shapefile

        # Vertical initialization
        log_message("Initializing Vertical Distribution.", level=2)
        self.vertical_weights = None  # Filled during calculation step
        self.vertical_factors = None  # Filled during calculation step
        self.vertical_output_profile = vertical_output_profile
        self.vertical = self.init_vertical(
            profile_id=self.parse_profile(p_vertical),
            profiles_path=options.p_vertical,
            vertical_output_profile=vertical_output_profile,
            only_init=options.first_time,
        )
        log_message("Vertical Distribution initialization done.", level=3)

        # Temporal initialization
        log_message("Initializing Temporal Distribution.", level=2)
        self.temporal_factors = None  # Filled during calculation step
        self.temporal = self.init_temporal(
            timestep_type=options.output_timestep_type,
            timestep_num=options.output_timestep_num,
            timestep_freq=options.output_timestep_freq,
            profile_info={
                "Month": {"profile_id": p_month, "path": options.p_month},
                "Week": {"profile_id": p_week, "path": options.p_week},
                "Day": {"profile_id": p_day, "path": options.p_day},
                "Hour": {"profile_id": p_hour, "path": options.p_hour},
            },
            only_init=options.first_time,
        )
        log_message("Temporal Distribution initialization done.", level=3)

        # Speciation initialization
        log_message("Initializing Speciation.", level=2)
        self.speciation = self.init_speciation(
            profile_id=p_speciation,
            profiles_path=options.p_speciation,
            molecular_weights_path=options.molecular_weights,
            only_init=options.first_time,
        )
        self.weight_matrix = None

        self.emissions = self.prepare_input_emissions()

        log_message("Speciation initialization done.", level=3)

        log_message("Emission inventory initialization done.", level=2)

    def get_weight_matrix_path(self):
        return os.path.join(self.auxiliary_files_path, f"Weight_Matrix_{self.inventory_name}.nc")

    @staticmethod
    def print_ini_message():
        log_message("Creating area source emission inventory.", level=3)
        return None

    def get_aux_emis_file_name(self):
        """
        Obtain the Regridded auxiliary filename.

        Returns
        -------
        str
            Path to the auxiliary file that contains the regridded data
        """
        extension = "nc"
        # Finding pollutant folder and filename.
        if self.input_frequency == "yearly":
            file_name = f"{self.inventory_name}_{self.sector}_{self.reference_year}.{extension}"
        elif self.input_frequency == "monthly":
            file_name = (f"{self.inventory_name}_{self.sector}_{self.reference_year}"
                         f"{self.date.strftime('%m')}.{extension}")
        elif self.input_frequency == "daily":
            file_name = (f"{self.inventory_name}_{self.sector}_{self.reference_year}"
                         f"{self.date.strftime('%m')}{self.date.strftime('%d')}.{extension}")
        elif self.input_frequency == "hourly":
            file_name = (f"{self.inventory_name}_{self.sector}_{self.reference_year}"
                         f"{self.date.strftime('%m')}{ self.date.strftime('%d')}{self.date.strftime('%H')}.{extension}")
        else:
            log_message("ERROR: Check the .err file to get more info.")
            raise ValueError(f"ERROR: frequency {self.input_frequency} not implemented. Use yearly, monthly or daily.")
        return os.path.join(self.auxiliary_files_path, file_name)

    def prepare_input_emissions(self, write_aux_file=True):
        """
        Prepare the emissions.

        1. Combine the pollutants of the source
        2. Apply the masks
        2. Perform the Regrid
        3. Save the regridded file as auxiliary files

        Parameters
        ----------
        write_aux_file : bool
            Indicates if you want to save the regridded emissions as auxiliary file.
        """
        st_time = get_time_stamp()
        # start_increment("EmissionInventory", "Prepare Input Emissions")
        # change_labels("EmissionInventory", "Prepare Input Emissions")

        emis_aux_path = self.get_aux_emis_file_name()
        if os.path.exists(emis_aux_path):
            # Loading already existent emissions file
            aux_emis = open_netcdf(
                comm=self.grid.comm,
                path=emis_aux_path,
                balanced=self.grid.balanced,
                parallel_method=self.grid.parallel_method,
            )
            loaded_vars = list(aux_emis.variables.keys())
            if len(set(self.input_pollutants).difference(set(loaded_vars))) > 0:
                # Some pollutants are missing in the emission file
                aux_emis.load()
                # Creating/Reading weight matrix
                self.weight_matrix = self.init_regrid(
                    weight_matrix_path=self.get_weight_matrix_path()
                )
                # Preparing input emissions
                self.read_input_emissions()
                # Applying regrid
                self.do_regrid()
                # Adding new pollutants into the loaded ones
                self.emissions = concatenate_netcdfs(
                    [self.emissions, aux_emis],
                    comm=self.grid.comm,
                    balanced=self.grid.balanced,
                    parallel_method=self.grid.parallel_method,
                )
                # Creating new emissions file
                if write_aux_file:
                    self.emissions.to_netcdf(emis_aux_path, serial=False, info=False)
                # Removing not needed pollutants
                self.emissions.keep_vars(self.input_pollutants)
            else:
                # All emissions are loaded and there is no pollutant missing
                self.emissions = aux_emis
                self.emissions.keep_vars(self.input_pollutants)
                self.emissions.load()
        else:
            # Creating new regridded emissions file
            log_message("Creating weight matrix.", level=2)
            # Creating/Reading weight matrix
            self.weight_matrix = self.init_regrid(
                weight_matrix_path=self.get_weight_matrix_path()
            )
            log_message(f"Weight Matrix created in {self.get_weight_matrix_path()}", level=3)
            # Preparing input emissions
            self.read_input_emissions()
            # Applying regrid
            self.do_regrid()
            if write_aux_file:
                self.emissions.to_netcdf(emis_aux_path, serial=False)
        record_time("EmissionInventory", "prepare_input_emissions", get_time_stamp() - st_time)
        # stop_increment("EmissionInventory", "Prepare Input Emissions")
        return self.emissions

    def prepare_lazy_output_emissions(self):
        """
        Prepare the output file with the metadata (and no data) to create the empty file

        Returns
        -------
        dict
            Final output variables in NES format without data
        """
        lazy_vars = {}
        for var_info in self.speciation.speciation_profile:
            lazy_vars[var_info["name"]] = {
                "data": None,
                "units": var_info["units"],
                "long_name": var_info["long_name"],
                "dtype": precision,
            }
        return lazy_vars

    def init_regrid(self, weight_matrix_path):
        """
        Initialise the Weight matrix

        Parameters
        ----------
        weight_matrix_path : str
            Weight matrix auxiliary file path

        Returns
        -------
        Nes
            Weight matrix
        """
        emissions = open_netcdf(
            comm=self.grid.comm,
            path=self.src_emis_file_list[0],
            balanced=self.grid.balanced,
            parallel_method=self.grid.parallel_method,
        )
        emissions.calculate_grid_area(overwrite=True)
        weight_matrix = emissions.interpolate_horizontal(
            self.grid,
            weight_matrix_path=weight_matrix_path,
            info=False,
            kind="Conservative",
            only_create_wm=True,
            flux=True,
        )

        return weight_matrix

    @staticmethod
    def init_vertical(
        profile_id, profiles_path, vertical_output_profile, only_init=False
    ):
        """
        Initialise the Vertical class

        Parameters
        ----------
        profile_id : str
            Vertical profile ID
        profiles_path : str
            Path to the file that contains all the vertical profiles.
        vertical_output_profile : List[float]
            Vertical levels
        only_init : bool
            Indicates if the run is for only initialization that vertical distribution is not needed

        Returns
        -------
        VerticalDistribution
        """
        st_time = get_time_stamp()
        # start_increment("EmissionInventory", "init_vertical")
        # change_labels("EmissionInventory", "init_vertical")

        if only_init:
            return None

        if profile_id is None:
            log_message("None vertical profile set.", level=2)
            return None
        vertical = VerticalDistribution(
            profile_id, profiles_path, vertical_output_profile
        )
        record_time("EmissionInventory", "init_vertical", get_time_stamp() - st_time)
        # stop_increment("EmissionInventory", "init_vertical")
        return vertical

    def init_temporal(
        self, timestep_type, timestep_num, timestep_freq, profile_info, only_init=False
    ):
        """
        Initialize the temporal disaggregation

        Parameters
        ----------
        timestep_type : str
            Relation between time-steps. It can be hourly, monthly or yearly.
        timestep_num : int
            Quantity of time-steps.
        timestep_freq : int
            Quantity of timestep_type between time-steps.
            eg: If timestep_type = hourly; timestep_freq = 2; The difference between time of each timestep is 2 hours.
        profile_info : dict
            Profile information.
            profiles = {
                            "Month": {"profile_id": p_month, "path": options.p_month},
                            "Week": {"profile_id": p_week, "path": options.p_week},
                            "Day": {"profile_id": p_day, "path": options.p_day},
                            "Hour": {"profile_id": p_hour, "path": options.p_hour}
                       }
        only_init : bool
            Indicates if the simulation is only to do the initialization and do not need the temporal distribution

        Returns
        -------
        TemporalDistribution
            Object to do the temporal disaggregation
        """
        st_time = get_time_stamp()
        # start_increment("EmissionInventory", "init_temporal")
        # change_labels("EmissionInventory", "init_temporal")
        if only_init:
            return None

        if (
            (profile_info["Month"]["profile_id"] is None)
            and (profile_info["Week"]["profile_id"] is None)
            and (profile_info["Day"]["profile_id"] is None)
            and (profile_info["Hour"]["profile_id"] is None)
        ):
            log_message("None temporal profile set.", level=2)
            return None

        temporal = TemporalDistribution(
            self.grid,
            self.date,
            timestep_type,
            timestep_num,
            timestep_freq,
            profile_info,
        )

        record_time("EmissionInventory", "init_temporal", get_time_stamp() - st_time)
        # stop_increment("EmissionInventory", "init_temporal")
        return temporal

    @staticmethod
    def init_speciation(
        profile_id, profiles_path, molecular_weights_path, only_init=False
    ):
        """
        Initialise the Speciation class

        Parameters
        ----------
        profile_id : str
            Speciation profile ID
        profiles_path : str
            Path to the file that contains all the speciation profiles.
        molecular_weights_path : str
            Path to the file that contains the molecular weights.
        only_init : bool
            Indicates if the run is for only initialization that speciation is not needed

        Returns
        -------
        Speciation
        """
        st_time = get_time_stamp()
        # start_increment("EmissionInventory", "init_speciation")
        # change_labels("EmissionInventory", "init_speciation")

        if only_init:
            return None

        if profile_id is None:
            log_message("None speciation profile set.", level=2)
            return None
        speciation = Speciation(profile_id, profiles_path, molecular_weights_path)

        record_time("EmissionInventory", "init_speciation", get_time_stamp() - st_time)
        # stop_increment("EmissionInventory", "init_speciation")
        return speciation

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
        file_list = []
        for pollutant_name in pollutant_list:
            file_list.append(self.get_input_path(pollutant=pollutant_name))
        return file_list

    def create_pollutants_dicts(self, pollutants):
        """
        Create a list of dictionaries with the information of the name, path and Dataset of each pollutant

        Parameters
        ----------
        pollutants : list
            Pollutant name list

        Returns
        -------
        List[dict]
            List of dictionaries with "name", "path" and "Dataset" information
        """

        pollutant_list = []

        for pollutant_name in pollutants:
            pollutant_list.append(
                {
                    "name": pollutant_name,
                    "path": self.get_input_path(pollutant=pollutant_name),
                    "Dataset": f"{self.inventory_name}_{self.sector}",
                }
            )
        return pollutant_list

    @staticmethod
    def parse_profile(id_aux):
        """
        Parse the id of the profiles.

        Parameters
        ----------
        id_aux : str
            ID of the profile.

        Returns
        -------
        str
            ID of the profile parsed.
        """
        if pd.isnull(id_aux):
            return None
        else:
            return id_aux

    def get_input_path(self, pollutant=None):
        """
        Completes the path of the input file that contains the needed information of the given pollutant.

        Parameters
        ----------
        pollutant : str
            Pollutant name.

        Returns
        -------
        str
            Absolute path to the needed file
        """
        if self.source_type == "area":
            extension = "nc"

        elif self.source_type == "point":
            if self.inventory_name[:4] == "GFAS":
                extension = "nc"
            else:
                extension = "csv"
        else:
            log_message("ERROR: Check the .err file to get more info.")
            raise AttributeError(f"ERROR: Unknown source type {self.source_type}")

        # Finding upper folder
        if pd.isnull(self.sector):
            upper_folder = f"{pollutant}"
        else:
            upper_folder = f"{pollutant}_{self.sector}"

        # Finding pollutant folder and filename.
        if self.input_frequency == "yearly":
            file_name = f"{pollutant}_{self.reference_year}.{extension}"
        elif self.input_frequency == "monthly":
            file_name = f"{pollutant}_{self.reference_year}{self.date.strftime('%m')}.{extension}"
        elif self.input_frequency == "daily":
            file_name = (f"{pollutant}_{self.reference_year}{self.date.strftime('%m')}"
                         f"{self.date.strftime('%d')}.{extension}")
        else:
            log_message("ERROR: Check the .err file to get more info.")
            raise ValueError(f"ERROR: frequency {self.input_frequency} not implemented. Use yearly, monthly or daily.")

        # Filename
        file_path = os.path.join(self.inputs_path, upper_folder, file_name)

        # Checking input file
        if not os.path.exists(file_path):
            log_message("ERROR: Check the .err file to get more info.")
            raise IOError(f"ERROR: File {file_path} not found.")

        return file_path

    def read_input_emissions(self):
        """
        Read all the input emissions

        During the initialization only the first pollutant is read but not loaded.
        Read means to have only the metadata and location of the emission inventory to calculate the weight matrix.
        This function should be called before the regridding in order to load in memory all the pollutants before the
        regridding function.
        """
        msg = "\tLoading emissions from:"
        for aux_file in self.src_emis_file_list:
            msg += f"\n\t\t{os.path.basename(aux_file)}"
        log_message(msg, level=3)
        # In the initialization only the first pollutant is read but not loaded.
        self.emissions = concatenate_netcdfs(
            self.src_emis_file_list,
            comm=self.grid.comm,
            balanced=self.grid.balanced,
            parallel_method=self.grid.parallel_method,
        )
        # Casting emissions to the selected precision
        self.emissions.to_dtype(precision)
        mask_factors = Masking(
            dst_grid=self.grid,
            src_grid_path=self.src_emis_file_list[0],
            factors_mask_values=self.factors_mask,
            regrid_mask_values=self.regrid_mask,
            world_mask_file=os.path.join(os.path.dirname(self.auxiliary_files_path),
                                         f"{self.inventory_name}_WorldMask.nc"),
            countries_shapefile=self.countries_shapefile,
        ).mask_factors
        if mask_factors is not None:
            for var_name in self.emissions.variables.keys():
                self.emissions.variables[var_name]["data"] *= mask_factors

        return None

    def do_regrid(self):
        """
        Regridding of the emissions
        """

        st_time = get_time_stamp()
        # start_increment("EmissionInventory", "do_regrid")
        # change_labels("EmissionInventory", "do_regrid")

        log_message("Regridding", level=2)

        # self.emissions.calculate_grid_area(overwrite=True)
        log_message("Start conservative interpolation", level=3)
        self.emissions = self.emissions.interpolate_horizontal(
            self.grid, kind="Conservative", wm=self.weight_matrix, flux=True)
        log_message("End conservative interpolation", level=3)
        self.emissions.to_dtype(precision)

        record_time("EmissionInventory", "do_regrid", get_time_stamp() - st_time)
        # stop_increment("EmissionInventory", "do_regrid")
        return None

    def get_4d_emissions(self):
        """
        Obtain the Emission inventory in 4D format

        """
        emis = deepcopy(self.emissions.variables)

        for var_name in self.emissions.variables.keys():
            if (self.emissions.variables[var_name]["data"] is None) or isinstance(
                self.emissions.variables[var_name]["data"], int
            ):
                # emis.variables[var_name]["data"] = np.empty(data_shape)
                emis[var_name]["data"] = 0
            else:
                # Multiply surface by each vertical level
                emis[var_name]["data"] = np.array(
                    emis[var_name]["data"]
                    * self.vertical_factors[np.newaxis, :, np.newaxis, np.newaxis],
                    dtype=precision,
                )
                # Multiply surface by each vertical level
                if self.temporal_factors is not None:
                    emis[var_name]["data"] = np.array(
                        emis[var_name]["data"]
                        * self.temporal_factors[:, np.newaxis, :, :],
                        dtype=precision,
                    )

        return emis
