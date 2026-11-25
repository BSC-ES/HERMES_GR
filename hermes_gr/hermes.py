#!/usr/bin/env python

# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import os
import traceback
import numpy as np
import pandas as pd
import re
import hermes_gr
from hermes_gr.config import ConfigGr
from hermes_gr.config.proxy_catalogue import load_proxy_catalogue, select_proxies
from hermes_gr.config import configure_logger, log_message, precision
from hermes_gr.config import get_time_stamp, record_time, finalize_time_log

# from hermes_gr.config import get_memory_stamp, record_memory, finalize_mem_log
from hermes_gr.modules.grid import select_grid, add_time_info_to_grid
from hermes_gr.modules.emision_inventories import (
    EmissionInventory,
    GfasEmissionInventory,
    GfasHourlyEmissionInventory,
    PointGfasEmissionInventory,
    PointGfasHourlyEmissionInventory,
    PointSourceEmissionInventory,
)
from hermes_gr.modules.rescaling import ReScaling
from hermes_gr.modules.vertical.vertical import VerticalDistribution
from hermes_gr.modules.temporal.temporal import TemporalDistribution
from mpi4py import MPI
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Optional


class HermesGr(object):
    """
    Interface class for HERMES_GR.

    Attributes
    ----------
    comm : MPI.COMMUNICATOR
    rank : int
        MPI rank
    master : bool
        Indicates if the process is the master one.
    size : int
        Total amount of processes involved.
    options : NameSpace
        Configuration options
    levels : List[float]
        Vertical level description
    grid : Nes
        Destination grid
    rescaling : ReScaling or None
        Regional emission rescaling configuration.
    emission_list : List[EmissionInventory]
        Emission inventory list
    time_step_list : List[int]
        List of hours between time-steps. e.g. [0, 1, 2, ..., 24]
    """

    def __init__(
        self,
        config: ConfigGr,
        new_date: Optional[datetime] = None,
        comm: Optional[MPI.Comm] = None,
    ):
        """

        Parameters
        ----------
        config : ConfigGr
            Configuration instance
        new_date : datetime or None
            New date to simulate
        comm : MPI.Communicator
        """
        global full_time
        st_time = full_time = get_time_stamp()

        # MPI Initialization
        if comm is None:
            comm = MPI.COMM_WORLD
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.master = self.rank == 0

        self.config = config
        self.options = config.options

        # Updating starting date
        if new_date is not None:
            if new_date > self.options.end_date:
                return
            else:
                self.options.start_date = new_date

        # Logger
        configure_logger(log_path=str(os.path.join(
            self.options.output_dir,
            "logs",
            os.path.basename(self.config.get_output_name(self.options.start_date)))).replace(".nc", "_<rank>.log"),
                         log_detail=self.options.log_level)

        self.first_time = self.config.options.first_time

        log_message(f"===== Starting HERMES_GR {hermes_gr.__version__} initialization =====")

        if (
            self.options.output_model in ["CMAQ", "WRF_CHEM"]
            and self.options.domain_type == "global"
        ):
            log_message("ERROR: Check the .err file to get more info.")
            raise AttributeError(f"ERROR: Global domain is not available for {self.options.output_model} output model.")

        self.vertical_description = self.config.options.vertical_description
        self.levels = self.vertical_description
        log_message(f"Levels: {self.levels}", level=3)
        self.time_step_list = TemporalDistribution.calculate_date_array(
                                    self.options.start_date,
                                    self.options.output_timestep_type,
                                    self.options.output_timestep_num,
                                    self.options.output_timestep_freq,
                                )

        self.grid = select_grid(self.comm, self.options, levels=self.levels, time_step_list=self.time_step_list)

        self.rescaling = self.init_rescaling()

        self.emission_list = self.make_emission_list()

        self.out_format = self.options.output_model

        log_message("===== End of HERMES_GR initialization =====\n")
        record_time("Initialization", "TOTAL", get_time_stamp() - st_time)

    def init_rescaling(self):
        """
        Initialise regional emission rescaling if configured.

        Returns
        -------
        ReScaling or None
            Regional emission rescaling configuration.
        """
        rescaling_inputs = {
            "rescaling_geometries": self.options.rescaling_geometries,
            "rescaling_geometries_attribute": self.options.rescaling_geometries_attribute,
            "rescaling_local_total_emissions": self.options.rescaling_local_total_emissions,
        }

        if all(value is None for value in rescaling_inputs.values()):
            return None
        if any(value is None for value in rescaling_inputs.values()):
            missing_arguments = [name for name, value in rescaling_inputs.items() if value is None]
            raise ValueError(
                "ERROR: Incomplete regional emission rescaling configuration. Missing arguments: "
                f"{', '.join(missing_arguments)}"
            )

        log_message("Initializing regional emission rescaling.", level=2)
        rescaling = ReScaling(
            self.grid,
            self.options.rescaling_geometries,
            self.options.rescaling_geometries_attribute,
            self.options.rescaling_local_total_emissions,
            self.options.auxiliary_files_path,
            self.options.rescaling_everything,
            rescaling_regions=self.options.rescaling_regions,
            rescaling_allow_partial_regions=self.options.rescaling_allow_partial_regions,
        )
        log_message("Regional emission rescaling initialization done.", level=3)

        return rescaling

    def make_emission_list(self):
        """
        Extract the information of the cross table to read all the needed emissions.

        Returns
        -------
        List[EmissionInventory]
            Emission inventory list
        """
        path = self.options.cross_table
        df = pd.read_csv(path, sep=";", index_col=False)
        # Accept the plural spelling without silently ignoring configured proxies.
        if "proxies" in df.columns:
            if "proxy" in df.columns:
                raise ValueError("Use only one proxy selection column: 'proxy' or 'proxies', not both")
            df = df.rename(columns={"proxies": "proxy"})
        proxy_catalogue = load_proxy_catalogue(
            getattr(self.options, "proxy_catalogue", None),
            input_dir=self.options.input_dir, data_path=self.options.data_path,
        )
        for column in [
            "ei",
            "sector",
            "ref_year",
            "active",
            "factor_mask",
            "regrid_mask",
            "pollutants",
            "path",
            "frequency",
            "source_type",
            "coverage",
            "p_vertical",
            "p_month",
            "p_week",
            "p_day",
            "p_hour",
            "p_speciation",
        ]:
            df_cols = list(df.columns.values)
            if column not in df_cols:
                raise AttributeError(f"ERROR: Column {column} is not in the {path} file.")

        df["active"] = pd.to_numeric(df["active"], errors="coerce").fillna(0).astype(int)
        df = df[df["active"] == 1]
        active_inventory_count = len(df)
        if active_inventory_count == 0:
            raise ValueError(
                "ERROR: No active emission inventories were found in the cross table "
                f"'{path}'. Check that the 'active' column contains at least one value equal to 1."
            )
        num = 1
        emission_inventory_list = []
        for i, src_emission_inventory in df.iterrows():
            log_message(f"Loading emission {num}/{len(df)} (Inventory: {src_emission_inventory.ei}; "
                        f"Sector: {src_emission_inventory.sector})", level=1)
            num += 1
            proxies = select_proxies(src_emission_inventory.get("proxy"), proxy_catalogue)
            pollutants = list(map(str, re.split(", |,|; |;| ", src_emission_inventory.pollutants)))

            # Attach the shared ReScaling object only to configured area inventories.
            emission_rescaling = None
            if self.rescaling is not None:
                has_rescaling = self.rescaling.do_i_have_to_rescale(
                    src_emission_inventory.ei,
                    src_emission_inventory.sector,
                    src_emission_inventory.frequency,
                    src_emission_inventory.ref_year,
                    self.options.start_date,
                )
                if has_rescaling and src_emission_inventory.source_type != "area":
                    raise ValueError(
                        "ERROR: Regional emission rescaling is only available for area emission inventories. "
                        f"Inventory '{src_emission_inventory.ei}' and sector '{src_emission_inventory.sector}' are "
                        f"configured for rescaling but source_type is '{src_emission_inventory.source_type}'."
                    )
                if has_rescaling:
                    emission_rescaling = self.rescaling

            # MONTHLY
            try:
                # gridded temporal profile
                p_month = src_emission_inventory.p_month.replace("<data_path>", self.options.data_path)
                p_month = p_month.replace("<input_dir>", self.options.input_dir)
            except AttributeError:
                p_month = src_emission_inventory.p_month
            # WEEKLY
            try:
                # gridded temporal profile
                p_week = src_emission_inventory.p_week.replace("<data_path>", self.options.data_path)
                p_week = p_week.replace("<input_dir>", self.options.input_dir)
            except AttributeError:
                p_week = src_emission_inventory.p_week
            # DAILY
            try:
                # gridded temporal profile
                p_day = src_emission_inventory.p_day.replace("<data_path>", self.options.data_path)
                p_day = p_day.replace("<input_dir>", self.options.input_dir)
            except AttributeError:
                p_day = src_emission_inventory.p_day
            # HOURLY
            try:
                # gridded temporal profile
                p_hour = src_emission_inventory.p_hour.replace("<data_path>", self.options.data_path)
                p_hour = p_hour.replace("<input_dir>", self.options.input_dir)
            except AttributeError:
                p_hour = src_emission_inventory.p_hour

            emission_inventory_path = src_emission_inventory.path.replace("<data_path>", self.options.data_path)
            emission_inventory_path = emission_inventory_path.replace("<input_dir>", self.options.input_dir)

            if src_emission_inventory.source_type == "area":
                if src_emission_inventory.ei[:4] == "GFAS":
                    if src_emission_inventory.frequency == "daily":
                        emission_inventory_list.append(
                            GfasEmissionInventory(
                                self.options,
                                self.grid,
                                self.options.start_date,
                                src_emission_inventory.ei,
                                src_emission_inventory.source_type,
                                src_emission_inventory.sector,
                                pollutants,
                                emission_inventory_path,
                                src_emission_inventory.frequency,
                                self.levels,
                                reference_year=src_emission_inventory.ref_year,
                                coverage=src_emission_inventory.coverage,
                                factors=src_emission_inventory.factor_mask,
                                regrid_mask=src_emission_inventory.regrid_mask,
                                p_vertical=src_emission_inventory.p_vertical,
                                p_month=p_month,
                                p_week=p_week,
                                p_day=p_day,
                                p_hour=p_hour,
                                p_speciation=src_emission_inventory.p_speciation,
                                countries_shapefile=self.options.countries_shapefile,
                                rescaling=emission_rescaling,
                                proxies=proxies,
                            )
                        )
                    elif src_emission_inventory.frequency == "hourly":
                        emission_inventory_list.append(
                            GfasHourlyEmissionInventory(
                                self.options,
                                self.grid,
                                self.options.start_date,
                                src_emission_inventory.ei,
                                src_emission_inventory.source_type,
                                src_emission_inventory.sector,
                                pollutants,
                                emission_inventory_path,
                                src_emission_inventory.frequency,
                                self.levels,
                                reference_year=src_emission_inventory.ref_year,
                                coverage=src_emission_inventory.coverage,
                                factors=src_emission_inventory.factor_mask,
                                regrid_mask=src_emission_inventory.regrid_mask,
                                p_vertical=src_emission_inventory.p_vertical,
                                p_month=p_month,
                                p_week=p_week,
                                p_day=p_day,
                                p_hour=p_hour,
                                p_speciation=src_emission_inventory.p_speciation,
                                countries_shapefile=self.options.countries_shapefile,
                                rescaling=emission_rescaling,
                                proxies=proxies,
                            )
                        )
                    else:
                        raise NotImplementedError(
                            f"ERROR: {src_emission_inventory.frequency} frequency is not implemented for "
                            f"GFAS area emission inventories. Use 'daily' or 'hourly'."
                        )
                else:
                    emission_inventory_list.append(
                        EmissionInventory(
                            self.options,
                            self.grid,
                            self.options.start_date,
                            src_emission_inventory.ei,
                            src_emission_inventory.source_type,
                            src_emission_inventory.sector,
                            pollutants,
                            emission_inventory_path,
                            src_emission_inventory.frequency,
                            self.levels,
                            reference_year=src_emission_inventory.ref_year,
                            coverage=src_emission_inventory.coverage,
                            factors=src_emission_inventory.factor_mask,
                            regrid_mask=src_emission_inventory.regrid_mask,
                            p_vertical=src_emission_inventory.p_vertical,
                            p_month=p_month,
                            p_week=p_week,
                            p_day=p_day,
                            p_hour=p_hour,
                            p_speciation=src_emission_inventory.p_speciation,
                            countries_shapefile=self.options.countries_shapefile,
                            rescaling=emission_rescaling,
                            proxies=proxies,
                        )
                    )
            elif src_emission_inventory.source_type == "point":
                if proxies:
                    raise ValueError("Proxy-weighted interpolation is supported for area inventories only")
                if src_emission_inventory.ei[:4] == "GFAS":
                    expanded_grid = select_grid(self.comm, self.options, levels=self.levels,
                                                time_step_list=self.time_step_list)
                    if src_emission_inventory.frequency == "daily":
                        emission_inventory_list.append(
                            PointGfasEmissionInventory(
                                self.options,
                                expanded_grid,
                                self.options.start_date,
                                src_emission_inventory.ei,
                                src_emission_inventory.source_type,
                                src_emission_inventory.sector,
                                pollutants,
                                emission_inventory_path,
                                src_emission_inventory.frequency,
                                self.levels,
                                reference_year=src_emission_inventory.ref_year,
                                factors=src_emission_inventory.factor_mask,
                                regrid_mask=src_emission_inventory.regrid_mask,
                                p_vertical=src_emission_inventory.p_vertical,
                                p_month=p_month,
                                p_week=p_week,
                                p_day=p_day,
                                p_hour=p_hour,
                                p_speciation=src_emission_inventory.p_speciation,
                                countries_shapefile=self.options.countries_shapefile,
                            )
                        )
                    elif src_emission_inventory.frequency == "hourly":
                        emission_inventory_list.append(
                            PointGfasHourlyEmissionInventory(
                                self.options,
                                expanded_grid,
                                self.options.start_date,
                                src_emission_inventory.ei,
                                src_emission_inventory.source_type,
                                src_emission_inventory.sector,
                                pollutants,
                                emission_inventory_path,
                                src_emission_inventory.frequency,
                                self.levels,
                                reference_year=src_emission_inventory.ref_year,
                                factors=src_emission_inventory.factor_mask,
                                regrid_mask=src_emission_inventory.regrid_mask,
                                p_vertical=src_emission_inventory.p_vertical,
                                p_speciation=src_emission_inventory.p_speciation,
                                countries_shapefile=self.options.countries_shapefile,
                            )
                        )
                    else:
                        raise NotImplementedError(f"ERROR: {src_emission_inventory.frequency} frequency are not "
                                                  f"implemented, use 'daily' or 'hourly.")
                else:
                    emission_inventory_list.append(
                        PointSourceEmissionInventory(
                            self.options,
                            self.grid,
                            self.options.start_date,
                            src_emission_inventory.ei,
                            src_emission_inventory.source_type,
                            src_emission_inventory.sector,
                            pollutants,
                            emission_inventory_path,
                            src_emission_inventory.frequency,
                            self.levels,
                            reference_year=src_emission_inventory.ref_year,
                            factors=src_emission_inventory.factor_mask,
                            regrid_mask=src_emission_inventory.regrid_mask,
                            p_vertical=src_emission_inventory.p_vertical,
                            p_month=p_month,
                            p_week=p_week,
                            p_day=p_day,
                            p_hour=p_hour,
                            p_speciation=src_emission_inventory.p_speciation,
                            countries_shapefile=self.options.countries_shapefile,
                        )
                    )
            else:
                raise ValueError(f"ERROR: The emission inventory source type '{src_emission_inventory.source_type}'"
                                 f" is not implemented. Use 'area' or 'point'")
            self.comm.Barrier()

        if not emission_inventory_list:
            raise ValueError(
                "ERROR: Active emission inventory rows were found in the cross table "
                f"'{path}', but no emission inventory object was created. Check the 'ei', "
                "'source_type' and 'frequency' columns for unsupported combinations. "
                f"Active rows: {active_inventory_count}."
            )

        return emission_inventory_list

    def prepare_inventories(self):
        """
        Proces all the emission inventories to calculate the vertical distribution and speciation
        """
        vertical_time = 0
        speciation_time = 0

        for num, ei in enumerate(self.emission_list):
            log_message(f"Processing emission inventory {ei.inventory_name} for the sector {ei.sector} "
                        f"({num + 1}/{len(self.emission_list)}):")

            # Vertical distribution
            aux_time = get_time_stamp()
            if ei.vertical is not None:
                log_message("Vertical distribution.", level=2)
                if ei.source_type == "area":
                    ei.vertical_factors = ei.vertical.calculate_weights()
                    del ei.vertical

                elif ei.source_type == "point":
                    # To avoid use point source as area source when is going to apply vertical factors while writing
                    ei.vertical = None
                else:
                    raise AttributeError(f"Unrecognized emission source type {ei.source_type}")
            else:
                if ei.source_type == "area":
                    ei.vertical_factors = np.zeros(len(self.levels), dtype=precision)
                    ei.vertical_factors[0] = 1
                elif ei.source_type == "point":
                    ei.vertical_factors = np.ones(len(self.levels), dtype=precision)
                else:
                    raise AttributeError(f"Unrecognized emission source type {ei.source_type}")
            vertical_time += get_time_stamp() - aux_time

            # Speciation
            aux_time = get_time_stamp()
            if ei.speciation is not None:
                ei.emissions = ei.speciation.do_speciation(ei.emissions)
            speciation_time += get_time_stamp() - aux_time

        record_time("Calculation", "do_vertical", vertical_time)
        record_time("Calculation", "do_speciation", speciation_time)
        return None

    def get_time_step_emissions(self, time_step, i_time):
        """
        Obtain the emissions of the selected time step

        Parameters
        ----------
        time_step : datetime
            Time stamp to simulate
        i_time : int
            Index of the time-step

        Returns
        -------
        dict
            Time stamp emissions in Nes format
        """
        temporal_time = 0
        emissions_4d_time = 0
        combine_time = 0
        result = self.emission_list[0].prepare_lazy_output_emissions()

        for num, ei in enumerate(self.emission_list):
            # Temporal distribution
            aux_time = get_time_stamp()
            if ei.temporal is not None:
                ei.temporal_factors = np.array([ei.temporal.calculate_2d_temporal_factors(i_time)], dtype=precision)
            else:
                ei.temporal_factors = None

            if isinstance(ei, GfasHourlyEmissionInventory) or isinstance(ei, PointGfasHourlyEmissionInventory):
                # Needed to load hourly emissions
                ei.date = time_step
            temporal_time += get_time_stamp() - aux_time

            aux_time = get_time_stamp()
            ei.prepare_time_step_emissions()
            emissions_4d_time += get_time_stamp() - aux_time

            for var_name, var_info in ei.emissions.variables.items():
                # 4D emissions
                aux_time = get_time_stamp()
                emission_data = ei.get_4d_emission(var_name)
                emissions_4d_time += get_time_stamp() - aux_time

                # Combine inventories
                aux_time = get_time_stamp()
                if var_name not in result:
                    result[var_name] = {
                        attr_name: attr_value
                        for attr_name, attr_value in var_info.items()
                        if attr_name != "data"
                    }
                    result[var_name]["data"] = None
                    result[var_name]["dtype"] = precision
                if isinstance(emission_data, int):
                    if result[var_name]["data"] is None:
                        result[var_name]["data"] = 0
                elif result[var_name]["data"] is None or isinstance(result[var_name]["data"], int):
                    result[var_name]["data"] = emission_data
                else:
                    result[var_name]["data"] += emission_data
                combine_time += get_time_stamp() - aux_time

        record_time("Calculation", "do_temporal", temporal_time)
        record_time("Calculation", "get_4d_emissions", emissions_4d_time)
        record_time("Calculation", "combine_inventories", combine_time)
        return result

    def prepare_output(self):
        """
        Prepare the output object and create the empty output file.

        Returns
        -------
        tuple
            Output Nes object with lazy variables and the output file path.
        """
        result = self.grid.copy(copy_vars=False)
        result.cell_measures = self.grid.cell_measures
        result.set_strlen(None)
        result.set_communicator(self.comm)
        result.set_levels(
            {
                "data": np.array(self.levels, dtype=np.float32),
                "units": "m",
                "positive": "up",
            }
        )
        result.set_time(self.time_step_list)
        result.variables = self.emission_list[0].prepare_lazy_output_emissions()

        # Global attributes
        result.global_attrs = {}
        if self.out_format == "CMAQ":
            if self.options.output_attributes is not None:
                global_attributes = pd.read_csv(self.options.output_attributes, sep=",", index_col="attribute")
                for attr_name in [
                    "EXEC_ID",
                    "FTYPE",
                    "NTHIK",
                    "VGTYP",
                    "VGTOP",
                    "VGLVLS",
                    "GDNAM",
                ]:
                    if attr_name in global_attributes.index:
                        result.global_attrs[attr_name] = global_attributes.loc[attr_name, "value"]

            result.global_attrs["FILEDESC"] = "Emissions generated by HERMES_GR."
        elif self.out_format == "WRF_CHEM":
            if self.options.output_attributes is not None:
                global_attributes = pd.read_csv(self.options.output_attributes, sep=",", index_col="attribute")
                for attr_name in [
                    "DAMPCOEF",
                    "KHDIF",
                    "KVDIF",
                    "CEN_LAT",
                    "CEN_LON",
                    "DT",
                    "BOTTOM-TOP_GRID_DIMENSION",
                    "DIFF_OPT",
                    "KM_OPT",
                    "DAMP_OPT",
                    "MP_PHYSICS",
                    "RA_LW_PHYSICS",
                    "RA_SW_PHYSICS",
                    "SF_SFCLAY_PHYSICS",
                    "SF_SURFACE_PHYSICS",
                    "BL_PBL_PHYSICS",
                    "CU_PHYSICS",
                    "SF_LAKE_PHYSICS",
                    "SURFACE_INPUT_SOURCE",
                    "SST_UPDATE",
                    "GRID_FDDA",
                    "GFDDA_INTERVAL_M",
                    "GFDDA_END_H",
                    "GRID_SFDDA",
                    "SGFDDA_INTERVAL_M",
                    "SGFDDA_END_H",
                    "BOTTOM-TOP_PATCH_START_UNSTAG",
                    "BOTTOM-TOP_PATCH_END_UNSTAG",
                    "BOTTOM-TOP_PATCH_START_STAG",
                    "BOTTOM-TOP_PATCH_END_STAG",
                    "GRID_ID",
                    "PARENT_ID",
                    "I_PARENT_START",
                    "J_PARENT_START",
                    "PARENT_GRID_RATIO",
                    "NUM_LAND_CAT",
                    "ISWATER",
                    "ISLAKE",
                    "ISICE",
                    "ISURBAN",
                    "ISOILWATER",
                    "GRIDTYPE",
                    "MMINLU",
                ]:
                    if attr_name in global_attributes.index:
                        result.global_attrs[attr_name] = global_attributes.loc[
                            attr_name, "value"
                        ]
        result.global_attrs["HISTORY"] = (
            "Code developed by Barcelona Supercomputing Center (BSC, https://www.bsc.es/). "
            + "Developer: Carles Tena Medina (carles.tena@bsc.es) "
            + "Software reference: Tena et al., 2026, HERMES_GR: High-Elective Resolution "
            + "Modelling Emission System - Global-Regional "
            + f"(version v{hermes_gr.__version__}). Zenodo. "
            + "https://doi.org/10.5281/zenodo.24530149. "
            + "Methodology reference: Guevara et al., 2019, GMD. "
            + "https://doi.org/10.5194/gmd-12-1885-2019"
        )
        out_path = self.config.get_output_name(self.options.start_date)
        result.to_netcdf(
            out_path,
            serial=self.options.serial_write,
            keep_open=True,
            nc_type=self.out_format,
            parallel_io_mode=self.options.parallel_io_mode,
        )
        return result, out_path

    def main(self):
        """
        Main functionality of the model.

        Returns
        -------
        datetime
            New datetime if other simulation is requested, otherwise None
        """
        if self.first_time:
            # Stop run
            log_message("===== HERMES_GR First Time finished successfully =====")

        else:
            writing_total = 0
            st_time = get_time_stamp()

            log_message("")
            log_message("===== Starting HERMES_GR Calculation =====")

            self.prepare_inventories()  # To obtain speciated emissions and vertical factors

            aux_time = get_time_stamp()
            result, out_path = self.prepare_output()
            elapsed_time = get_time_stamp() - aux_time
            writing_total += elapsed_time
            record_time("Writing", "prepare_output", elapsed_time)

            log_message(f"Created empty output file {out_path}", level=1)

            for i_time, time_step in enumerate(self.time_step_list):
                log_message(f"Calculating {time_step} data ({i_time + 1}/{len(self.time_step_list)})", level=2)
                result.variables = self.get_time_step_emissions(time_step, i_time)
                log_message("Writing data", level=3)

                aux_time = get_time_stamp()
                timing = result.append_time_step_data(i_time, out_format=self.out_format, return_timing=True)
                elapsed_time = get_time_stamp() - aux_time
                writing_total += elapsed_time
                record_time("Writing", "write_time_step_emissions", elapsed_time)
                record_time("Writing", "unit_conversion", timing["unit_conversion"])
                record_time("Writing", "communication", timing["communication"])
                record_time("Writing", "netcdf_write", timing["netcdf_write"])
                record_time("Writing", "netcdf_write_unaccounted", timing.get("netcdf_write_unaccounted", 0))

            log_message(f"***** Output file done: {out_path}")

            aux_time = get_time_stamp()
            result.close()
            elapsed_time = get_time_stamp() - aux_time
            writing_total += elapsed_time
            record_time("Writing", "close_output", elapsed_time)

            log_message("===== HERMES_GR simulation finished successfully =====")
            record_time("Calculation", "TOTAL", get_time_stamp() - st_time - writing_total)
            record_time("Writing", "TOTAL", writing_total)
        record_time("HERMES", "TOTAL", get_time_stamp() - full_time)
        finalize_time_log(str(os.path.join(self.options.output_dir, "logs", os.path.basename(
            self.config.get_output_name(self.options.start_date)))).replace(
            ".nc", f"_Times_{str(self.size).zfill(4)}.csv"))

        # Updating the date for a date-loop HERMES simulation
        if self.options.start_date < self.options.end_date:
            if self.options.output_timestep_type in ["hourly", "daily"]:
                new_date = self.options.start_date + timedelta(days=1)
            elif self.options.output_timestep_type in ["monthly"]:
                new_date = self.options.start_date + relativedelta(months=1)
            elif self.options.output_timestep_type in ["yearly"]:
                new_date = self.options.start_date + relativedelta(years=1)
            else:
                raise RuntimeError("Unknown timestep_type")
            # Only return valid period starts: run() reinitializes each one.
            # An out-of-range start would leave the next instance incomplete.
            if new_date <= self.options.end_date:
                return new_date

        return None


def run():
    try:
        config = ConfigGr()
        if hermes_gr.DEBUG:
            from pandas import set_option
            set_option("display.max_columns", None)

        model = HermesGr(config)

        date = model.main()
        while date is not None:
            date = HermesGr(config, new_date=date).main()
    except Exception as e:
        log_message(f"Process {MPI.COMM_WORLD.Get_rank()}: Critical error detected {e}, aborting MPI.", level=9)
        log_message(f"Process {MPI.COMM_WORLD.Get_rank()}: Traceback:\n{traceback.format_exc()}", level=9,)

        MPI.COMM_WORLD.Abort(1)

    return


if __name__ == "__main__":
    run()
