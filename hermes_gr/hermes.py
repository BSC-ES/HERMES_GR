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

        # # Memory Profiler
        # configure_profiler(
        #     comm=self.comm,
        #     profile=False,  # TODO add flag in hermes for this
        #     base_output_path=os.path.join(
        #         self.options.output_dir, "logs", "memory_profiler"
        #     ),
        # )
        # start_increment("HERMES", "TOTAL")
        # start_increment("Initialization", "TOTAL")
        # start_memory(interval=0.1, section="Initialization", subsection="TOTAL")

        self.first_time = self.config.options.first_time

        log_message(f"===== Starting HERMES_GR {hermes_gr.__version__} initialization =====")

        if (
            self.options.output_model in ["CMAQ", "WRF_CHEM"]
            and self.options.domain_type == "global"
        ):
            log_message("ERROR: Check the .err file to get more info.")
            raise AttributeError(f"ERROR: Global domain is not available for {self.options.output_model} output model.")

        self.vertical_description = self.config.options.vertical_description
        self.levels = VerticalDistribution.get_vertical_output_profile(
            self.vertical_description
        )
        log_message(f"Levels: {self.levels}", level=3)
        self.time_step_list = TemporalDistribution.calculate_date_array(
                                    self.options.start_date,
                                    self.options.output_timestep_type,
                                    self.options.output_timestep_num,
                                    self.options.output_timestep_freq,
                                )

        self.grid = select_grid(self.comm, self.options, levels=self.levels, time_step_list=self.time_step_list)

        self.emission_list = self.make_emission_list()

        self.out_format = self.options.output_model

        log_message("===== End of HERMES_GR initialization =====\n")
        record_time("Init", "TOTAL", get_time_stamp() - st_time)
        # stop_increment("Initialization", "TOTAL")
        # stop_section_labels()

    def make_emission_list(self):
        """
        Extract the information of the cross table to read all the needed emissions.

        Returns
        -------
        List[EmissionInventory]
            Emission inventory list
        """
        st_time = get_time_stamp()
        # start_increment("EmissionInventory", "make_emission_list")
        # change_labels("EmissionInventory", "make_emission_list")

        path = self.options.cross_table
        df = pd.read_csv(path, sep=";", index_col=False)
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

        df = df[df["active"] == 1]
        num = 1
        emission_inventory_list = []
        for i, src_emission_inventory in df.iterrows():
            log_message(f"Loading emission {num}/{len(df)} (Inventory: {src_emission_inventory.ei}; "
                        f"Sector: {src_emission_inventory.sector})", level=1)
            num += 1
            pollutants = list(map(str, re.split(", |,|; |;| ", src_emission_inventory.pollutants)))

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
                            )
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
                        )
                    )
            elif src_emission_inventory.source_type == "point":
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
        record_time("EmissionInventory", "make_emission_list", get_time_stamp() - st_time)
        # stop_increment("EmissionInventory", "make_emission_list")

        return emission_inventory_list

    def prepare_inventories(self):
        """
        Proces all the emission inventories to calculate the vertical distribution and speciation
        """
        st_time = get_time_stamp()
        # start_increment("EmissionInventory", "prepare_inventories")
        # change_labels("EmissionInventory", "prepare_inventories")

        for num, ei in enumerate(self.emission_list):
            log_message(f"Processing emission inventory {ei.inventory_name} for the sector {ei.sector} "
                        f"({num + 1}/{len(self.emission_list)}):")

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
            if ei.speciation is not None:
                ei.emissions = ei.speciation.do_speciation(ei.emissions)
        record_time("EmissionInventory", "prepare_inventories", get_time_stamp() - st_time)
        # stop_increment("EmissionInventory", "prepare_inventories")
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
        st_time = get_time_stamp()
        # start_increment("EmissionInventory", "get_time_step_emissions")
        # change_labels("EmissionInventory", "get_time_step_emissions")

        result = None
        for num, ei in enumerate(self.emission_list):
            if ei.temporal is not None:
                ei.temporal_factors = np.array([ei.temporal.calculate_2d_temporal_factors(i_time)], dtype=precision)
            else:
                ei.temporal_factors = None

            if isinstance(ei, GfasHourlyEmissionInventory) or isinstance(ei, PointGfasHourlyEmissionInventory):
                # Needed to load hourly emissions
                ei.date = time_step

            if result is None:
                result = ei.get_4d_emissions()
            else:
                for var_name, var_info in ei.get_4d_emissions().items():
                    result[var_name]["data"] += var_info["data"]

        record_time("EmissionInventory", "get_time_step_emissions", get_time_stamp() - st_time)
        # stop_increment("EmissionInventory", "get_time_step_emissions")
        return result

    def prepare_output(self):
        """
        Prepares the output to be written

        Returns
        -------
        Nes
            Output Nes object with lazy variables
        """
        st_time = get_time_stamp()
        # start_increment("Write", "Creation")
        # change_labels("Write", "Creation")

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
            + "Modelling Emission System - Global-Regional (version v3.0.0). "
            + "https://doi.org/10.82201/TQF44A. "
            + "Methodology reference: Guevara et al., 2019, GMD. "
            + "https://doi.org/10.5194/gmd-12-1885-2019"
        )
        record_time("Write", "Creation", get_time_stamp() - st_time)
        # stop_increment("Write", "Creation")
        # stop_section_labels()
        return result

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
            write_time = 0
            st_time = get_time_stamp()
            # start_increment("Calculation", "TOTAL")
            # change_labels("Calculation", "TOTAL")

            log_message("")
            log_message("===== Starting HERMES_GR Calculation =====")

            self.prepare_inventories()  # To obtain speciated emissions and vertical factors

            # Preparing output file
            aux_time = get_time_stamp()

            # stop_increment("Calculation", "TOTAL")
            # start_increment("Write", "TOTAL")
            # change_labels("Write", "TOTAL")

            result = self.prepare_output()
            out_path = self.config.get_output_name(self.options.start_date)
            result.to_netcdf(out_path, serial=self.options.serial_write, keep_open=True, nc_type=self.out_format)
            write_time += get_time_stamp() - aux_time
            # stop_increment("Write", "TOTAL")
            # start_increment("Calculation", "TOTAL")
            # change_labels("Calculation", "TOTAL")

            log_message(f"Created empty output file {out_path}", level=1)

            for i_time, time_step in enumerate(self.time_step_list):
                log_message(f"Calculating {time_step} data ({i_time + 1}/{len(self.time_step_list)})", level=2)
                result.variables = self.get_time_step_emissions(time_step, i_time)
                log_message("Writing data", level=3)

                # stop_increment("Calculation", "TOTAL")
                # start_increment("Write", "TOTAL")
                # change_labels("Write", "TOTAL")
                aux_time = get_time_stamp()
                result.append_time_step_data(i_time, out_format=self.out_format)
                write_time += get_time_stamp() - aux_time
                # stop_increment("Write", "TOTAL")
                # start_increment("Calculation", "TOTAL")
                # change_labels("Calculation", "TOTAL")

            log_message(f"***** Output file done: {out_path}")

            # Closing output file
            # stop_increment("Calculation", "TOTAL")
            # start_increment("Write", "TOTAL")
            # change_labels("Write", "TOTAL")
            # aux_time = get_time_stamp()
            result.close()

            # stop_increment("Write", "TOTAL")
            # stop_memory()
            # write_time += get_time_stamp() - aux_time

            log_message("===== HERMES_GR simulation finished successfully =====")
            record_time("HERMES", "Main", get_time_stamp() - st_time)
            record_time("HERMES", "Write", write_time)
        record_time("HERMES", "TOTAL", get_time_stamp() - full_time)
        # stop_increment("HERMES", "TOTAL")
        # finalize_increments()
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
