#!/usr/bin/env python

# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import numpy as np
from pandas import DataFrame, read_csv
from geopandas import GeoDataFrame
from shapely.geometry import Point
from hermes_gr.config import log_message, precision
from .emission_inventory import EmissionInventory


class PointSourceEmissionInventory(EmissionInventory):

    def __init__(self, options, grid, current_date, inventory_name, source_type, sector, pollutants, inputs_path,
                 frequency, vertical_output_profile, reference_year=2010, factors=None, regrid_mask=None,
                 p_vertical=None, p_month=None, p_week=None, p_day=None, p_hour=None, p_speciation=None,
                 countries_shapefile=None):

        self.crs = 4326
        self.vertical = None

        super(PointSourceEmissionInventory, self).__init__(
            options, grid, current_date, inventory_name, source_type, sector, pollutants, inputs_path, frequency,
            vertical_output_profile, reference_year=reference_year, factors=factors, regrid_mask=regrid_mask,
            p_vertical=p_vertical, p_month=p_month, p_week=p_week, p_day=p_day, p_hour=p_hour,
            p_speciation=p_speciation, countries_shapefile=countries_shapefile)

        self.weight_matrix = self.init_regrid(weight_matrix_path=self.get_weight_matrix_path())

    @staticmethod
    def print_ini_message():
        """
        Print initial message
        """
        log_message("Creating point source emission inventory.", level=3)
        return None

    def prepare_input_emissions(self, write_aux_file=False):
        self.read_input_emissions()
        return self.emissions

    def read_input_emissions(self):
        """
        Fills the emissions class attribute with the Point source emissions distributed vertically
        """
        emissions = self.grid.copy(copy_vars=False, new_comm=self.grid.comm)
        grid_shape = self.grid.create_shapefile()
        grid_shape.reset_index(inplace=True, drop=True)

        grid_shape["area"] = self.grid.cell_measures["cell_area"]["data"].flatten()

        log_message("Allocating point sources on grid:", level=2)

        num = 1
        for pollutant in self.pollutant_dicts:
            log_message(f"Pollutant {pollutant['name']} ({num}/{len(self.pollutant_dicts)})", level=3)
            num += 1

            # Read complete CSV
            point_emis = read_csv(pollutant["path"], usecols=["Lat", "Lon", "Alt_Injection", "Emis"], dtype={
                "Lat": precision, "Lon": precision,
                "Alt_Injection": precision, "Emis": precision})

            # Transform it to Shapefile
            point_emis = GeoDataFrame(point_emis.loc[:, ["Emis", "Alt_Injection"]], crs=self.crs,
                                      geometry=[Point(xy) for xy in zip(point_emis.Lon, point_emis.Lat)])
            # Spatial Join
            point_emis = point_emis.to_crs(grid_shape.crs)
            point_emis = point_emis.sjoin(grid_shape, how="inner", predicate="intersects")
            point_emis.rename(columns={"index_right": "FID"}, inplace=True)
            # Drops duplicates when the point source is on the boundary of the cell
            point_emis = point_emis[~point_emis.index.duplicated(keep="first")]
            point_emis = DataFrame(point_emis[["FID", "Emis", "area", "Alt_Injection"]])

            emissions.variables[pollutant["name"]] = {
                "data": np.zeros((1, len(self.grid.lev["data"]), self.grid.lat["data"].shape[0],
                                  self.grid.lon["data"].shape[-1]), dtype=precision)}
            if len(point_emis) > 0:
                # kg to kg/m2
                point_emis["Emis"] /= point_emis["area"]
                del point_emis["area"]
                # Alt_Injection to level
                point_emis = self.alt_injection_to_level(point_emis)
                # Sum all emissions of same cell (FID) and level
                point_emis = point_emis.groupby(["FID", "level"]).sum()

                for level_i, level_emis in point_emis.groupby(level="level"):
                    aux_data = np.zeros((self.grid.lat["data"].shape[0], self.grid.lon["data"].shape[-1]),
                                        dtype=precision).flatten()

                    aux_data[level_emis.index.get_level_values("FID")] = level_emis["Emis"].values
                    aux_data = aux_data.reshape((self.grid.lat["data"].shape[0], self.grid.lon["data"].shape[-1]))
                    emissions.variables[pollutant["name"]]["data"][0, level_i, :] = aux_data

        self.emissions = emissions
        return None

    def alt_injection_to_level(self, emis_df):
        """
        Transform the "Alt_Injection" column to "level"

        Parameters
        ----------
        emis_df : DataFrame
            Table with the emissions containing the "Alt_injection" column

        Returns
        -------
        DataFrame
            Table with the emissions and the corresponding level
        """
        emis_df["level"] = 0
        for level_i, level_m in enumerate(self.grid.lev["data"]):
            emis_df.loc[emis_df["Alt_Injection"] <= level_m, "level"] = level_i
            emis_df.loc[emis_df["Alt_Injection"] <= level_m, "Alt_Injection"] = None
        del emis_df["Alt_Injection"]

        return emis_df

    def init_regrid(self, weight_matrix_path):
        """
        Initialise the Weight matrix to None

        Parameters
        ----------
        weight_matrix_path : str
            Weight matrix auxiliary file path

        Returns
        -------
        Nes
            Weight matrix
        """

        return None

    def do_regrid(self):
        """
        Regridding of the emissions
        """

        return None
