#!/usr/bin/env python

# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import os
import hashlib
import json
import pandas as pd
import numpy as np
from hermes_gr.config import precision, log_message
from hermes_gr.config.proxy_catalogue import to_nes_proxies
from hermes_gr.config import get_time_stamp, record_time
from hermes_gr.modules.vertical import VerticalDistribution
from hermes_gr.modules.temporal import TemporalDistribution
from hermes_gr.modules.speciation import Speciation
from hermes_gr.modules.masking import Masking
from hermes_gr.tools.auxiliary_consistency import check_aux_file
from hermes_gr.tools.auxiliary_consistency import check_grid
from nes import open_netcdf, concatenate_netcdfs, Nes
from datetime import datetime

AUXILIARY_WARNINGS_ATTR = "HERMES_GR_warnings"


class EmissionInventory(object):
    """
    Define the content and processing workflow for area emission inventories.

    Attributes
    ----------
    source_type : str
        Type of emission source.
    coverage : str
        Spatial coverage of the emission inventory.
    grid : Nes
        Destination grid.
    auxiliary_files_path : str
        Path to the auxiliary directory.
    date : datetime
        Date to simulate.
    inventory_name : str
        Name of the inventory to use.
    sector : str
        Name of the sector of the inventory to use.
    inputs_path : str
        Path where all datasets to use are stored.
    input_frequency : str
        Frequency of the input emissions. Supported values depend on the source type.
    reference_year : int
        Year of reference of the information of the dataset.
    input_pollutants : list[str]
        Input pollutants processed by the inventory.
    pollutant_dicts : list[dict]
        Pollutant metadata used by the processing workflow.
    src_emis_file_list : list[str]
        Source emission file paths for the selected pollutants.
    regrid_mask : str or None
        Country mask expression to apply before regridding.
    factors_mask : str or None
        Country scaling-factor expression to apply before regridding.
    countries_shapefile : str or None
        Path to the country shapefile used by masking.
    vertical : VerticalDistribution or None
        Vertical distribution handler.
    vertical_weights : object or None
        Vertical weights filled during the calculation step.
    vertical_factors : object or None
        Vertical factors filled during the calculation step.
    vertical_output_profile : list[float]
        Output vertical profile definition.
    temporal : TemporalDistribution or None
        Temporal distribution handler.
    temporal_factors : object or None
        Temporal factors filled during the calculation step.
    speciation : Speciation or None
        Speciation handler.
    rescaling : ReScaling or None
        Regional emission rescaling object when this inventory and sector must be rescaled.
    rescaling_factors : dict or None
        Regional rescaling factors by region and pollutant.
    weight_matrix : Nes or None
        Horizontal interpolation weight matrix.
    emissions : Nes
        Emissions on the destination grid.
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
        rescaling=None,
        proxies=None,
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
            Description of the scale factors per country (e.g. SPN 1.5, CHN 3.).
        regrid_mask : str
            Description of the masking countries (adding e.g. + SPN AND; subtracting e.g. - SPN).
        p_vertical : str or None
            ID of the vertical profile to use.
        p_month : str or None
            ID of the temporal monthly profile to use.
        p_week : str or None
            ID of the temporal weekly profile to use.
        p_day : str or None
            ID of the temporal daily profile to use.
        p_hour : str or None
            ID of the temporal hourly profile to use.
        p_speciation : str or None
            ID of the speciation profile to use.
        countries_shapefile : str or None
            Path to the shapefile that contains the countries.
        rescaling : ReScaling or None
            Regional emission rescaling object when this inventory and sector must be rescaled.
        proxies : list[dict] or None
            Ordered proxy definitions for horizontal interpolation.
        """

        self.print_ini_message()
        self._timing_phase = "Initialization"

        # Emission Inventory parameters
        self.proxies = proxies or []
        self.source_type = source_type
        self.date = current_date
        self.inventory_name = inventory_name
        self.sector = sector
        self.reference_year = self.parse_reference_year(reference_year)
        self.coverage = coverage
        self.inputs_path = inputs_path
        self.input_frequency = input_frequency
        self.grid = grid
        self.auxiliary_files_path = options.auxiliary_files_path
        self.output_dir = options.output_dir

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
        self.rescaling = rescaling
        self.rescaling_factors = None

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
            holidays=options.holidays,
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
        log_message("Speciation initialization done.", level=3)

        log_message("Initializing Horizontal Distribution.", level=2)

        self.weight_matrix = None

        self.emissions = self.prepare_input_emissions()
        aux_time = get_time_stamp()
        self.emissions = self.rescale_emissions(self.emissions)
        self._record_initialization_time("rescaling", aux_time)

        log_message("Horizontal Distribution initialization done.", level=3)

        self._timing_phase = "Calculation"

        log_message("Emission inventory initialization done.", level=2)

    def _record_initialization_time(self, subsection, start_time):
        if getattr(self, "_timing_phase", None) == "Initialization":
            record_time("Initialization", subsection, get_time_stamp() - start_time)

    def _proxy_cache_suffix(self):
        if not self.proxies:
            return ""
        proxy_spec = to_nes_proxies(self.proxies)
        fingerprint = hashlib.sha256(
            json.dumps([self.sector, proxy_spec], sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        return f"_proxy_{fingerprint}"

    def get_weight_matrix_path(self):
        """
        Get the path to the auxiliary weight matrix file.

        Returns
        -------
        str
            Path to the auxiliary weight matrix file.
        """
        filename = f"Weight_Matrix_{self.inventory_name}{self._proxy_cache_suffix()}.nc"
        return os.path.join(self.auxiliary_files_path, filename)

    @staticmethod
    def print_ini_message():
        log_message("Creating area source emission inventory.", level=3)
        return None

    @staticmethod
    def parse_reference_year(reference_year):
        """
        Parse the inventory reference year preserving integer-like values.

        Pandas may read the cross-table ``ref_year`` column as float when the
        column contains empty values. File names must still use ``2015`` rather
        than ``2015.0``.
        """
        if pd.isnull(reference_year):
            raise ValueError("ERROR: Emission inventory reference year cannot be empty.")
        try:
            year = float(reference_year)
        except ValueError:
            return str(reference_year)
        if not year.is_integer():
            raise ValueError(
                f"ERROR: Emission inventory reference year '{reference_year}' is not an integer year."
            )
        return str(int(year))

    def get_aux_emis_file_name(self):
        """
        Obtain the Regridded auxiliary filename.

        Returns
        -------
        str
            Path to the auxiliary file that contains the regridded data
        """
        extension = "nc"
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
            raise ValueError(
                f"ERROR: frequency {self.input_frequency} not implemented. Use yearly, monthly, daily or hourly.")
        file_name = file_name[:-len(extension) - 1] + self._proxy_cache_suffix() + f".{extension}"
        return os.path.join(self.auxiliary_files_path, file_name)

    @staticmethod
    def _get_auxiliary_warning_messages(nessy):
        """
        Get warning messages stored in a NES auxiliary file.

        Parameters
        ----------
        nessy : Nes
            Auxiliary NES object.

        Returns
        -------
        list
            Stored warning messages.
        """
        global_attrs = getattr(nessy, "global_attrs", None) or {}
        warnings = global_attrs.get(AUXILIARY_WARNINGS_ATTR, "")
        if warnings is None:
            return []
        return [warning for warning in str(warnings).splitlines() if warning]

    def _emit_auxiliary_warning_messages(self, nessy):
        """
        Emit warnings stored in a NES auxiliary file.

        Parameters
        ----------
        nessy : Nes
            Auxiliary NES object.

        Returns
        -------
        None
        """
        if not getattr(self.grid, "master", True):
            return

        for warning in self._get_auxiliary_warning_messages(nessy):
            log_message(warning, level=7)

    @classmethod
    def _store_auxiliary_warning_messages(cls, nessy, warning_messages):
        """
        Store warning messages in a NES object so they are replayed when reused.

        Parameters
        ----------
        nessy : Nes
            NES object to write as auxiliary file.
        warning_messages : list
            Warning messages to store.

        Returns
        -------
        None
        """
        if not warning_messages:
            return

        global_attrs = getattr(nessy, "global_attrs", None)
        if global_attrs is None:
            nessy.global_attrs = {}
            global_attrs = nessy.global_attrs

        stored_warnings = cls._get_auxiliary_warning_messages(nessy)
        for warning in warning_messages:
            if warning not in stored_warnings:
                stored_warnings.append(warning)
        global_attrs[AUXILIARY_WARNINGS_ATTR] = "\n".join(stored_warnings)

    def prepare_input_emissions(self, write_aux_file=True):
        """
        Prepare the emissions.

        1. Combine the pollutants of the source
        2. Apply the masks
        3. Perform the regrid
        4. Save the regridded file as auxiliary files

        Parameters
        ----------
        write_aux_file : bool
            Indicates if you want to save the regridded emissions as auxiliary file.

        Returns
        -------
        Nes
            Prepared emissions on the destination grid.
        """
        emis_aux_path = self.get_aux_emis_file_name()
        if os.path.exists(emis_aux_path):
            aux_load_time = get_time_stamp()
            aux_emis = open_netcdf(
                comm=self.grid.comm,
                path=emis_aux_path,
                balanced=self.grid.balanced,
                parallel_method=self.grid.parallel_method,
            )
            if not check_aux_file(aux_emis, nes_object_to_compare=self.grid):
                raise ValueError(
                    f"Existing regridded emissions auxiliary file is not consistent with the current grid: "
                    f"{emis_aux_path}. Remove this auxiliary file or regenerate the auxiliary directory."
                )
            loaded_vars = list(aux_emis.variables.keys())
            if len(set(self.input_pollutants).difference(set(loaded_vars))) > 0:
                aux_emis.load()
                self._record_initialization_time("load_emissions", aux_load_time)
                self.weight_matrix = self.init_regrid(
                    weight_matrix_path=self.get_weight_matrix_path()
                )
                self.read_input_emissions()
                self.do_regrid()
                # Merge newly prepared pollutants with the pollutants already stored in the auxiliary file.
                self.emissions = concatenate_netcdfs(
                    [self.emissions, aux_emis],
                    comm=self.grid.comm,
                    balanced=self.grid.balanced,
                    parallel_method=self.grid.parallel_method,
                )
                if write_aux_file:
                    self._store_auxiliary_warning_messages(
                        self.emissions,
                        self._get_auxiliary_warning_messages(aux_emis) + getattr(self, "mask_warning_messages", []),
                    )
                    self.emissions.to_netcdf(emis_aux_path, serial=False, info=False)
                self.emissions.keep_vars(self.input_pollutants)
            else:
                self._emit_auxiliary_warning_messages(aux_emis)
                self.emissions = aux_emis
                self.emissions.keep_vars(self.input_pollutants)
                self.emissions.load()
                self._record_initialization_time("load_emissions", aux_load_time)
        else:
            log_message("Creating weight matrix.", level=2)
            self.weight_matrix = self.init_regrid(
                weight_matrix_path=self.get_weight_matrix_path()
            )
            log_message(f"Weight Matrix created in {self.get_weight_matrix_path()}", level=3)
            self.read_input_emissions()
            self.do_regrid()
            if write_aux_file:
                self._store_auxiliary_warning_messages(
                    self.emissions,
                    getattr(self, "mask_warning_messages", []),
                )
                self.emissions.to_netcdf(emis_aux_path, serial=False)

        self.emissions.close()

        return self.emissions

    def rescale_emissions(self, emissions):
        """
        Apply regional emission rescaling to prepared emissions.

        Parameters
        ----------
        emissions : Nes
            Prepared emissions on the destination grid.

        Returns
        -------
        Nes
            Emissions after regional rescaling.
        """
        if self.rescaling is None:
            return emissions

        self.rescaling_factors = self.rescaling.calculate_rescaling_factors(
            emissions,
            self.inventory_name,
            self.sector,
            self.reference_year,
            self.input_frequency,
            self.date,
            self.auxiliary_files_path,
            self.output_dir,
        )

        return self.rescaling.apply_factors(emissions, self.rescaling_factors)

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
        Initialise the weight matrix.

        Parameters
        ----------
        weight_matrix_path : str
            Weight matrix auxiliary file path

        Returns
        -------
        Nes
            Weight matrix
        """
        st_time = get_time_stamp()
        weight_matrix_exists = os.path.exists(weight_matrix_path)
        emissions = open_netcdf(
            comm=self.grid.comm,
            path=self.src_emis_file_list[0],
            balanced=self.grid.balanced,
            parallel_method=self.grid.parallel_method,
        )
        emissions.calculate_grid_area(overwrite=True)
        proxies = to_nes_proxies(self.proxies) if self.proxies else None
        if proxies:
            log_message(f"Downscaling proxies for {self.inventory_name}/{self.sector}: "
                        f"{self.proxies}", level=2)
        log_message(f"Type of data: {type(self.proxies)}", level=2)
        weight_matrix = emissions.interpolate_horizontal(
            self.grid,
            weight_matrix_path=weight_matrix_path,
            info=False,
            kind="Conservative",
            only_create_wm=True,
            flux=True,
            proxy=proxies,
        )
        if weight_matrix_exists:
            self.check_existing_weight_matrix(weight_matrix, weight_matrix_path)

        self._record_initialization_time("init_horizontal_get_weight_matrix", st_time)
        return weight_matrix

    def check_existing_weight_matrix(self, weight_matrix, weight_matrix_path):
        """
        Check an existing weight matrix auxiliary file before reusing it.

        NES may return the weight matrix object only on the master rank and a
        placeholder on the rest. For that reason the actual checks are performed
        where the NES object exists, and the result is broadcast to all ranks so
        every process fails with the same contextual error.
        """
        expected_shape = self.grid.get_full_shape()
        valid = True
        message = None

        if isinstance(weight_matrix, Nes):
            valid = (
                check_grid(weight_matrix, expected_shape=expected_shape)
                and check_aux_file(weight_matrix, var_list=["idx", "weight"])
            )
            if not valid:
                message = (
                    f"Existing weight matrix auxiliary file is not consistent with the current grid: "
                    f"{weight_matrix_path}. Expected full grid shape {expected_shape}, found "
                    f"{weight_matrix.get_full_shape()}, and variables 'idx' and 'weight'. Remove this auxiliary file "
                    "or regenerate the auxiliary directory."
                )

        valid = self.grid.comm.bcast(valid, root=0)
        message = self.grid.comm.bcast(message, root=0)
        if not valid:
            raise ValueError(message)

        return True

    def init_vertical(
        self, profile_id, profiles_path, vertical_output_profile, only_init=False
    ):
        """
        Initialise the vertical distribution.

        Parameters
        ----------
        profile_id : str
            Vertical profile ID
        profiles_path : str
            Path to the file that contains all the vertical profiles.
        vertical_output_profile : List[float]
            Vertical levels
        only_init : bool
            Indicates if the run is only for initialization and vertical distribution is not needed.

        Returns
        -------
        VerticalDistribution or None
            Vertical distribution handler.
        """
        st_time = get_time_stamp()

        if only_init:
            self._record_initialization_time("init_vertical", st_time)
            return None

        if profile_id is None:
            log_message("None vertical profile set.", level=2)
            self._record_initialization_time("init_vertical", st_time)
            return None
        vertical = VerticalDistribution(
            profile_id, profiles_path, vertical_output_profile
        )
        self._record_initialization_time("init_vertical", st_time)
        return vertical

    def init_temporal(
        self, timestep_type, timestep_num, timestep_freq, profile_info, only_init=False, holidays=False
    ):
        """
        Initialize the temporal disaggregation.

        Parameters
        ----------
        timestep_type : str
            Relation between time-steps. It can be hourly, daily, monthly or yearly.
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
            Indicates if the run is only for initialization and temporal distribution is not needed.
        holidays : bool
            Indicates if national holidays must be treated as Sundays during temporal allocation.

        Returns
        -------
        TemporalDistribution or None
            Object to do the temporal disaggregation.
        """
        st_time = get_time_stamp()
        if only_init:
            self._record_initialization_time("init_temporal", st_time)
            return None

        if (
            (profile_info["Month"]["profile_id"] is None)
            and (profile_info["Week"]["profile_id"] is None)
            and (profile_info["Day"]["profile_id"] is None)
            and (profile_info["Hour"]["profile_id"] is None)
        ):
            log_message("None temporal profile set.", level=2)
            self._record_initialization_time("init_temporal", st_time)
            return None

        temporal = TemporalDistribution(
            self.grid,
            self.date,
            timestep_type,
            timestep_num,
            timestep_freq,
            profile_info,
            holidays=holidays,
        )

        self._record_initialization_time("init_temporal", st_time)
        return temporal

    def init_speciation(
        self, profile_id, profiles_path, molecular_weights_path, only_init=False
    ):
        """
        Initialise the speciation.

        Parameters
        ----------
        profile_id : str
            Speciation profile ID
        profiles_path : str
            Path to the file that contains all the speciation profiles.
        molecular_weights_path : str
            Path to the file that contains the molecular weights.
        only_init : bool
            Indicates if the run is only for initialization and speciation is not needed.

        Returns
        -------
        Speciation or None
            Speciation handler.
        """
        st_time = get_time_stamp()

        if only_init:
            self._record_initialization_time("init_speciation", st_time)
            return None

        if profile_id is None:
            log_message("None speciation profile set.", level=2)
            self._record_initialization_time("init_speciation", st_time)
            return None
        speciation = Speciation(profile_id, profiles_path, molecular_weights_path)

        self._record_initialization_time("init_speciation", st_time)
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

        if pd.isnull(self.sector):
            upper_folder = f"{pollutant}"
        else:
            upper_folder = f"{pollutant}_{self.sector}"

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

        file_path = os.path.join(self.inputs_path, upper_folder, file_name)

        if not os.path.exists(file_path):
            log_message("ERROR: Check the .err file to get more info.")
            raise IOError(f"ERROR: File {file_path} not found.")

        return file_path

    def read_input_emissions(self):
        """
        Read all the input emissions.

        During the initialization only the first pollutant is read but not loaded.
        Read means to have only the metadata and location of the emission inventory to calculate the weight matrix.
        This function should be called before the regridding in order to load in memory all the pollutants before the
        regridding function.

        Returns
        -------
        None
        """
        st_time = get_time_stamp()
        msg = "\tLoading emissions from:"
        for aux_file in self.src_emis_file_list:
            msg += f"\n\t\t{os.path.basename(aux_file)}"
        log_message(msg, level=3)
        self.emissions = concatenate_netcdfs(
            self.src_emis_file_list,
            comm=self.grid.comm,
            balanced=self.grid.balanced,
            parallel_method=self.grid.parallel_method,
        )
        self.emissions.to_dtype(precision)
        masking = Masking(
            dst_grid=self.grid,
            src_grid_path=self.src_emis_file_list[0],
            factors_mask_values=self.factors_mask,
            regrid_mask_values=self.regrid_mask,
            world_mask_file=os.path.join(os.path.dirname(self.auxiliary_files_path),
                                         f"{self.inventory_name}_WorldMask.nc"),
            countries_shapefile=self.countries_shapefile,
        )
        self.mask_warning_messages = masking.warning_messages
        mask_factors = masking.mask_factors
        if mask_factors is not None:
            for var_name in self.emissions.variables.keys():
                self.emissions.variables[var_name]["data"] *= mask_factors

        self._record_initialization_time("load_emissions", st_time)
        return None

    def do_regrid(self):
        """
        Regrid the emissions onto the destination grid.

        Returns
        -------
        None
        """

        st_time = get_time_stamp()

        log_message("Regridding", level=2)

        log_message("Start conservative interpolation", level=3)
        self.emissions = self.emissions.interpolate_horizontal(
            self.grid, kind="Conservative", wm=self.weight_matrix, flux=True)
        log_message("End conservative interpolation", level=3)
        self.emissions.to_dtype(precision)

        self._record_initialization_time("init_horizontal_regrid", st_time)
        return None

    def get_4d_emissions(self):
        """
        Obtain the emission inventory in 4D format.

        Returns
        -------
        dict
            Emission variables after vertical and temporal distribution.
        """
        emis = {
            var_name: {"units": var_info["units"]}
            for var_name, var_info in self.emissions.variables.items()
        }

        for var_name, var_info in self.emissions.variables.items():
            emis[var_name]["data"] = self.get_4d_emission(var_name)

        return emis

    def prepare_time_step_emissions(self):
        """
        Prepare inventory data before calculating one output time step.

        Regular inventories keep their already prepared emissions. Hourly inventories
        override this hook to load the current hour data.
        """
        return None

    def get_4d_emission(self, var_name):
        """
        Obtain one emission variable after vertical and temporal distribution.

        Parameters
        ----------
        var_name : str
            Emission variable name.

        Returns
        -------
        numpy.ndarray or int
            Variable data in 4D format, or 0 when no emissions are available.
        """
        var_info = self.emissions.variables[var_name]
        if (var_info["data"] is None) or isinstance(var_info["data"], int):
            return 0

        # Multiply surface by each vertical level
        vertical_shape = np.broadcast_shapes(
            var_info["data"].shape,
            self.vertical_factors[np.newaxis, :, np.newaxis, np.newaxis].shape,
        )
        data = np.empty(vertical_shape, dtype=precision)
        np.multiply(
            var_info["data"],
            self.vertical_factors[np.newaxis, :, np.newaxis, np.newaxis],
            out=data,
            casting="unsafe",
        )
        if self.temporal_factors is not None:
            # Apply time-dependent factors to each vertical level.
            temporal_shape = np.broadcast_shapes(
                data.shape,
                self.temporal_factors[:, np.newaxis, :, :].shape,
            )
            temporal_data = np.empty(temporal_shape, dtype=precision)
            np.multiply(
                data,
                self.temporal_factors[:, np.newaxis, :, :],
                out=temporal_data,
                casting="unsafe",
            )
            data = temporal_data
        return data
