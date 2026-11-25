#!/usr/bin/env python
# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

"""Allocate gridded GFAS fire emissions as points in destination cells and levels.

Source-cell fluxes are converted to mass rates using source-cell areas. Each
source is mapped to one destination cell and distributed using GFAS injection
heights. Geographic scaling and inclusion masks form one multiplicative factor,
independent of the normalized vertical fractions. Contributions are summed by
destination cell and level, then divided by destination-cell area to recover flux.

Daily and hourly GFAS point inventories share this allocation implementation.
Only destination indices are reused from the weight-matrix auxiliary; geographic
factors are obtained for the current configuration during initialization.
"""

import os
import pandas as pd
from nes import open_netcdf, Nes
from hermes_gr.modules.masking.masking import Masking
from pandas import DataFrame
import numpy as np
from hermes_gr.config import precision, log_message
from hermes_gr.config import get_time_stamp
from .gfas_emission_inventory import GfasEmissionInventory
from mpi4py import MPI

GFAS_BUFFER = 0.1
GRID_BUFFER = 0.1


class PointGfasEmissionInventory(GfasEmissionInventory):
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
        frequency,
        vertical_output_profile,
        reference_year=2010,
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
        self.dst_fid = None
        self.dst_level = None

        super(PointGfasEmissionInventory, self).__init__(
            options,
            grid,
            current_date,
            inventory_name,
            source_type,
            sector,
            pollutants,
            inputs_path,
            frequency,
            vertical_output_profile,
            reference_year=reference_year,
            factors=factors,
            regrid_mask=regrid_mask,
            p_vertical=p_vertical,
            p_month=p_month,
            p_week=p_week,
            p_day=p_day,
            p_hour=p_hour,
            p_speciation=p_speciation,
            countries_shapefile=countries_shapefile,
        )

    def get_weight_matrix_path(self):
        return os.path.join(self.auxiliary_files_path, f"Weight_Matrix_{self.inventory_name}_point.nc")

    def get_grid_geo_path(self):
        return os.path.join(self.auxiliary_files_path, "grid.geojson")

    @staticmethod
    def print_ini_message():
        log_message("Creating GFAS point source emission inventory.", level=3)

        return None

    def read_input_emissions(self):
        """Load source fluxes and accumulate geographically scaled mass by level.

        Read the selected pollutants and injection altitude within the buffered
        destination extent. Multiplication by source-cell area converts fluxes to
        mass rates. The injection method and approach supply vertical fractions
        through ``get_vertical_weights``; they are not geographic mask values.

        ``self.weight_matrix`` must map source FIDs to destination FIDs and contain
        the combined geographic multiplier in ``factor`` (one when masks are
        omitted). For source i and level k, the contribution is source mass rate
        times geographic factor times vertical fraction. Zero contributions are
        discarded; the reusable geographic factors are left unchanged.

        Source locations are selected from positive values of the first input
        pollutant, assuming it represents the other pollutants' active locations.
        Store a DataFrame indexed by destination FID and level in ``self.emissions``.
        Values remain mass rates until ``do_regrid`` divides by destination area.

        Returns
        -------
        None
            Updates ``self.emissions`` in place; no generic mask is applied again.
        """
        st_time = get_time_stamp()
        log_message("Loading emissions")
        # In the initialization only the first pollutant is read but not loaded.
        emissions = open_netcdf(
            self.src_emis_file_list[0],
            comm=self.grid.comm,
            balanced=self.grid.balanced,
            parallel_method=self.grid.parallel_method,
        )
        emissions.sel(
            lat_min=self.grid.get_full_latitudes_boundaries()["data"].min() - GFAS_BUFFER,
            lon_min=self.grid.get_full_longitudes_boundaries()["data"].min() - GFAS_BUFFER,
            lat_max=self.grid.get_full_latitudes_boundaries()["data"].max() + GFAS_BUFFER,
            lon_max=self.grid.get_full_longitudes_boundaries()["data"].max() + GFAS_BUFFER,
        )

        emissions.keep_vars(self.input_pollutants + [self.altitude])
        emissions.load()

        # Flux to mass
        emissions.calculate_grid_area(overwrite=True)
        for var_name in emissions.variables.keys():
            if var_name != self.get_altitude():
                emissions.variables[var_name]["data"] *= emissions.cell_measures["cell_area"]["data"]

        emissions.to_dtype(data_type=precision)

        # Vertical distribution
        vertical_factor = self.get_vertical_weights(emissions.variables[self.altitude]["data"])
        vertical_factor = vertical_factor.reshape(
            (vertical_factor.shape[1], vertical_factor.shape[2] * vertical_factor.shape[3], ))
        emissions.free_vars(self.altitude)

        # Obtaining the location of the emissions
        # We assume that the first pollutant is representative of the location of the GFAS data.
        # If the first pollutant has data the others has data too. The same in the other side.
        representative_var = self.input_pollutants[0]
        src_index = np.where(emissions.variables[representative_var]["data"].flatten() > 0)[0]

        src_fids = emissions.get_fids().flatten()
        if len(src_index) > 0:
            src_fids = src_fids[src_index]
            result = self.weight_matrix.loc[src_fids].copy()
            for var_name in self.input_pollutants:
                result[var_name] = emissions.variables[var_name]["data"].flatten()[src_index]

            result_by_level = []
            for i_level in range(vertical_factor.shape[0]):
                aux_result = result.reset_index().copy()
                aux_result["level"] = i_level
                # Independent geographic scaling and vertical allocation compose by product.
                aux_result["factor"] *= vertical_factor[i_level, src_index]
                aux_result = aux_result.loc[aux_result["factor"] > 0]
                aux_result[self.input_pollutants] = aux_result[self.input_pollutants].multiply(
                    aux_result["factor"], axis="index")
                del aux_result["factor"], aux_result["src_fid"]

                result_by_level.append(aux_result)
            result = pd.concat(result_by_level)
            result = result.groupby(["dst_fid", "level"]).sum()

        else:
            result = DataFrame(
                index=pd.MultiIndex(levels=[[], []], codes=[[], []], names=["dst_fid", "level"]),
                columns=self.input_pollutants,
            )

        self.emissions = result

        self._record_initialization_time("load_emissions", st_time)
        return None

    def init_regrid(self, weight_matrix_path):
        """
        Map source cells to destination cells and attach geographic multipliers.

        Build destination indices by spatially joining source-cell points with
        output cells, or load existing indices from the auxiliary NetCDF file.
        Unmatched sources retain an absent destination and are filtered during
        output-grid selection. Duplicate spatial matches keep the first cell.

        Independently obtain geographic multipliers from ``Masking`` on the source
        grid. That component already multiplies factor_mask scaling by regrid_mask
        inclusion/exclusion. When neither is configured it returns None, whose
        multiplicative identity here is one. Vertical fractions are applied later
        by ``read_input_emissions`` and must not be added to these factors.

        Parameters
        ----------
        weight_matrix_path : str
            Weight matrix auxiliary file path

        Returns
        -------
        DataFrame
            Source FID index with destination FID and geographic ``factor`` columns.
            Factors are recomputed for the current configuration even when the
            destination-index auxiliary is reused.
        """
        # To consider save the Dataframe
        st_time = get_time_stamp()
        emissions = open_netcdf(
            comm=self.grid.comm,
            path=self.src_emis_file_list[0],
            balanced=self.grid.balanced,
            parallel_method=self.grid.parallel_method,
        )
        emissions.sel(
            lat_min=self.grid.get_full_latitudes_boundaries()["data"].min() - GFAS_BUFFER,
            lon_min=self.grid.get_full_longitudes_boundaries()["data"].min() - GFAS_BUFFER,
            lat_max=self.grid.get_full_latitudes_boundaries()["data"].max() + GFAS_BUFFER,
            lon_max=self.grid.get_full_longitudes_boundaries()["data"].max() + GFAS_BUFFER,
        )

        if not os.path.exists(weight_matrix_path):
            from geopandas import read_file, sjoin
            # Creating Inverse weight matrix to obtain the nearest dst_cell to each grid_cell
            log_message(f"Creating weight matrix at {weight_matrix_path}", level=3)
            self.grid.comm.Barrier()

            gfas_points = emissions.create_shapefile(points=True)
            gfas_points.index.name = "index"  # erasing index name FID to do not mix it with grid FID
            gfas_points_index = gfas_points.index.copy()

            # Re-projecting gfas_points using Grid Geostructure CSR
            grid_geo_path = self.get_grid_geo_path()
            tmp = read_file(grid_geo_path, rows=1)
            # TODO: Revisit whether the paired grid.geojson auxiliary needs this validation.
            # self.check_grid_geojson(tmp, grid_geo_path)
            if tmp.crs != gfas_points.crs:
                points_bbox_gdf = gfas_points.to_crs(tmp.crs)
                minx, miny, maxx, maxy = points_bbox_gdf.total_bounds
                bbox = (minx - GRID_BUFFER, miny - GRID_BUFFER, maxx + GRID_BUFFER, maxy + GRID_BUFFER)
                # bbox = points_bbox_gdf.total_bounds

            else:
                minx, miny, maxx, maxy = gfas_points.total_bounds
                bbox = (minx - GRID_BUFFER, miny - GRID_BUFFER, maxx + GRID_BUFFER, maxy + GRID_BUFFER)
                # bbox = gfas_points.total_bounds

            grid_gdf = read_file(grid_geo_path, bbox=bbox)
            # TODO: Revisit whether the paired grid.geojson auxiliary needs this validation.
            # self.check_grid_geojson(grid_gdf, grid_geo_path, bbox=bbox)

            # Reproject the join operands, not only the bbox helper above.
            gfas_points = gfas_points.to_crs(grid_gdf.crs)

            weight_matrix = emissions.copy(copy_vars=False)
            self.grid.comm.Barrier()
            gfas_points = sjoin(
                gfas_points,
                grid_gdf[["geometry", "FID"]],
                how="left",
                predicate="within"
            )
            self.grid.comm.Barrier()
            aux_shape = (1, 1, emissions.lat["data"].shape[0], emissions.lon["data"].shape[-1])
            expected_points = int(np.prod(aux_shape))
            if len(gfas_points_index) != expected_points:
                raise ValueError(
                    f"GFAS point grid has {len(gfas_points_index)} points, but expected {expected_points} "
                    f"from shape {aux_shape}."
                )
            if gfas_points.index.has_duplicates:
                duplicated_points = int(gfas_points.index.duplicated(keep="first").sum())
                log_message(
                    f"GFAS spatial join returned {duplicated_points} duplicated point matches. "
                    "Keeping the first destination cell for each source point.",
                    level=4,
                )
                gfas_points = gfas_points[~gfas_points.index.duplicated(keep="first")]
            dst_fids = gfas_points["FID"].reindex(gfas_points_index).to_numpy().reshape(aux_shape)
            # gfas_points["FID"] = gfas_points["FID"].fillna(-999)  # Filling NaN
            weight_matrix.variables = {
                "idx": {'data': dst_fids, 'units': '-'},
                "weight": {'data': np.ones(aux_shape), 'units': '-'}
            }
            weight_matrix.to_netcdf(weight_matrix_path)

            # weight_matrix = self.grid.interpolate_horizontal(
            #     emissions,
            #     weight_matrix_path=weight_matrix_path,
            #     info=False,
            #     kind="NearestNeighbour",
            #     n_neighbours=1,
            #     only_create_wm=True,
            # )
            idx = weight_matrix.variables["idx"]["data"]

        else:
            if self.grid.master:
                weight_matrix = open_netcdf(
                    comm=MPI.COMM_SELF,
                    path=weight_matrix_path,
                    balanced=self.grid.balanced,
                    parallel_method=self.grid.parallel_method,
                )
                weight_matrix.keep_vars("idx")
                weight_matrix.load()

                idx = weight_matrix.variables["idx"]["data"]
            else:
                idx = None
            self.grid.comm.Barrier()
            idx = emissions.comm.bcast(idx, root=0)

            idx = idx[:,
                      :,
                      emissions.write_axis_limits["y_min"]:emissions.write_axis_limits["y_max"],
                      emissions.write_axis_limits["x_min"]:emissions.write_axis_limits["x_max"], ]

        self.grid.comm.Barrier()

        weight_matrix = DataFrame(
            index=emissions.get_fids().flatten(),
            data=idx.flatten(),
            columns=["dst_fid"],
        )
        weight_matrix.index.name = "src_fid"

        # Masking
        weight_matrix["factor"] = 1
        weight_matrix["factor"] = np.array(weight_matrix["factor"], dtype=precision)
        mask_path = os.path.join(os.path.dirname(self.auxiliary_files_path), f"{self.inventory_name}_WorldMask.nc")
        mask_factors = Masking(
            dst_grid=self.grid,
            src_grid_path=emissions,
            factors_mask_values=self.factors_mask,
            regrid_mask_values=self.regrid_mask,
            world_mask_file=mask_path,
            countries_shapefile=self.countries_shapefile,
        ).mask_factors

        if mask_factors is not None:
            weight_matrix["factor"] = mask_factors.flatten()

        self._record_initialization_time("init_horizontal_get_weight_matrix", st_time)
        return weight_matrix

    @staticmethod
    def check_grid_geojson(grid_gdf, grid_geo_path, bbox=None):
        """
        Check the grid GeoJSON auxiliary file used by Point GFAS.
        """
        if grid_gdf.empty:
            if bbox is None:
                raise ValueError(
                    f"Existing grid GeoJSON auxiliary file is empty or unreadable: {grid_geo_path}. "
                    "Remove grid.nc and grid.geojson or regenerate the auxiliary directory."
                )
            raise ValueError(
                f"Existing grid GeoJSON auxiliary file has no cells in the GFAS points bounding box: {grid_geo_path}. "
                f"Bounding box: {bbox}. Remove grid.nc and grid.geojson or regenerate the auxiliary directory."
            )
        if grid_gdf.crs is None:
            raise ValueError(
                f"Existing grid GeoJSON auxiliary file has no CRS: {grid_geo_path}. "
                "Remove grid.nc and grid.geojson or regenerate the auxiliary directory."
            )
        if "FID" not in grid_gdf.columns:
            raise ValueError(
                f"Existing grid GeoJSON auxiliary file is missing required column 'FID': {grid_geo_path}. "
                "Remove grid.nc and grid.geojson or regenerate the auxiliary directory."
            )

        return True

    def do_regrid(self):
        """Assemble destination-cell GFAS point fluxes from allocated mass rates.

        Returns
        -------
        None
            Replace ``self.emissions`` with a copy of the destination NES grid
            containing one flux value per active destination-cell/level pair.
            Set ``self.dst_fid`` and ``self.dst_level`` to the corresponding IDs.

        Raises
        ------
        ValueError
            A retained destination FID does not identify exactly one cell area.

        Notes
        -----
        The input ``self.emissions`` is a DataFrame of mass rates indexed by
        destination FID and vertical level. Gather and sum contributions across
        MPI ranks, broadcast them, and retain only FIDs belonging to this rank's
        output grid using ``numpy.isin``. Serial runs use the local table directly.

        Divide each mass rate by its destination-cell area to recover flux in
        the source pollutant's mass-per-area-per-time units. Explicit scalar
        extraction with ``item()`` requires a unique cell match and avoids NumPy's
        deprecated implicit conversion of a one-element array to a scalar.
        Geographic and vertical factors have already been applied during input
        loading; this step does not apply them again or change their allocation.
        """
        st_time = get_time_stamp()

        log_message("Regridding", level=2)

        # Gathering into 0 and broadcasting emissions
        if self.grid.comm.Get_size() == 1:
            emissions = self.emissions
        else:
            emissions = self.grid.comm.gather(self.emissions, root=0)
            if self.grid.comm.Get_rank() == 0:
                emissions = pd.concat(emissions)
                emissions = emissions.groupby(level=("dst_fid", "level")).sum()
            else:
                emissions = None
            emissions = self.grid.comm.bcast(emissions, root=0)

        # Selecting only involved emissions
        fid_list = self.grid.get_fids().flatten()
        emissions = emissions[np.isin(emissions.index.get_level_values("dst_fid"), fid_list)]

        # To Nes mode
        self.dst_fid = emissions.index.get_level_values("dst_fid").to_numpy()
        self.dst_level = emissions.index.get_level_values("level").to_numpy()

        self.emissions = self.grid.copy(copy_vars=False)
        self.emissions.shapefile = None
        self.emissions.variables = {}
        for poll_name in self.input_pollutants:
            self.emissions.variables[poll_name] = {"data": emissions[poll_name].to_numpy(dtype=precision)}

        # Mass to flux
        fids = self.grid.get_fids()

        self.emissions.calculate_grid_area(overwrite=True)
        for var_name in self.emissions.variables.keys():
            if var_name != self.get_altitude():
                for i_data, cell_emission in enumerate(self.emissions.variables[var_name]["data"]):
                    y_pos, x_pos = np.where(fids == self.dst_fid[i_data])
                    self.emissions.variables[var_name]["data"][i_data] /= (
                        self.emissions.cell_measures)["cell_area"]["data"][y_pos, x_pos].item()

                # self.emissions.variables[var_name]['data'] /= self.emissions.cell_measures["cell_area"]['data']
        self._record_initialization_time("init_horizontal_regrid", st_time)
        return None

    def get_4d_emissions(self):
        if not isinstance(self.emissions, Nes):
            log_message(f"Emissions are not a NES object. It is a {type(self.emissions)}", level=9)
        emis = self.emissions.copy(copy_vars=True)
        # emis.comm = self.emissions.comm

        # Point emissions to 3D grid format
        self.prepare_time_step_emissions()
        for var_name in self.emissions.variables.keys():
            emis.variables[var_name]["data"] = self.get_4d_emission(var_name)
        return emis.variables

    def prepare_time_step_emissions(self):
        """Cache point-to-grid lookup data needed by each variable."""
        self._time_step_fids = self.emissions.get_fids()
        return None

    def get_4d_emission(self, var_name):
        if (self.emissions.variables[var_name]["data"] is None or
                isinstance(self.emissions.variables[var_name]["data"], int)):
            return 0

        # Point to 3D
        data = np.zeros(
            (1, len(self.grid.lev["data"]), self.grid.lat["data"].shape[0], self.grid.lon["data"].shape[-1], ),
            dtype=precision)
        for i_data, cell_emission in enumerate(self.emissions.variables[var_name]["data"]):
            y_pos, x_pos = np.where(self._time_step_fids == self.dst_fid[i_data])
            data[0, self.dst_level[i_data], y_pos[0], x_pos[0]] = cell_emission

        # Temporal disaggregation
        if self.temporal_factors is not None:
            data = np.array(data * self.temporal_factors[:, np.newaxis, :, :], dtype=precision)
        return data
