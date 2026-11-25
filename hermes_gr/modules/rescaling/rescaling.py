#!/usr/bin/env python

# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

"""
Regional emission rescaling.

The module adjusts gridded area-source inventories so their regional totals match independent local/reference totals.
Rescaling factors are calculated by region, inventory, sector and pollutant, then applied either to the whole inventory
or only to cells belonging to each configured region.

Notes
-----
Local totals are read from CSV files with ``Region``, ``ei``, ``sector`` and ``units`` columns. Supported temporal
conventions are:

- yearly inventories with ``kg/year`` or ``t/year`` totals;
- monthly inventories with ``kg/month`` or ``t/month`` totals.

When ``rescaling_local_total_emissions`` contains ``<date>``, the placeholder is resolved from the simulation date:
yearly inventories use ``YYYY`` and monthly inventories use ``YYYYMM``. If the resolved path does not exist,
``<date>`` is replaced by ``default`` and that fallback file is used with a warning. A path without ``<date>`` is
treated as a fixed file for all periods and also emits a warning.

The regional mask is stored as one NetCDF variable with regions represented as vertical levels. Region names are stored
in attributes so names with spaces or special characters are preserved. The synthetic ``OTHERS`` region is only a mask
helper and is never used as a rescaling target.
"""

import os
from calendar import monthrange
import geopandas as gpd
import hermes_gr
import numpy as np
import pandas as pd
from hermes_gr.config import log_message
from hermes_gr.tools.auxiliary_consistency import check_aux_file
from mpi4py import MPI
from nes import open_netcdf


LOCAL_TOTALS_REGION_COLUMN = "Region"
REGION_ID_COLUMN = "region_id"
EI_COLUMN = "ei"
SECTOR_COLUMN = "sector"
UNITS_COLUMN = "units"
REGION_TOTALS_INDEX_COLUMNS = [REGION_ID_COLUMN, EI_COLUMN, SECTOR_COLUMN]
DATE_PLACEHOLDER = "<date>"
SUPPORTED_UNITS = ["kg/year", "t/year", "kg/month", "t/month"]
PERIOD_UNITS = {
    "yearly": ["kg/year", "t/year"],
    "monthly": ["kg/month", "t/month"],
}
UNIT_FACTORS_TO_KG = {
    "kg/year": 1.0,
    "t/year": 1000.0,
    "kg/month": 1.0,
    "t/month": 1000.0,
}
OTHERS_REGION_ID = "OTHERS"
REGION_MASK_FILE = "ReScaling_RegionsMask.nc"
REGION_MASK_GEOJSON_FILE = "ReScaling_RegionsMask_spatial_join.geojson"
INVENTORY_TOTALS_FILE_TEMPLATE = "ReScaling_InventoryTotals_{date_suffix}.csv"
FACTORS_FILE_TEMPLATE = "ReScaling_Factors_{date_suffix}.csv"
REGION_MASK_VAR = "rescaling_mask"
REGION_IDS_SEPARATOR = ";"
SECONDS_PER_YEAR = 365 * 24 * 60 * 60
SECONDS_PER_DAY = 24 * 60 * 60


def _union_geometries(geometries):
    """Combine geometries using the unary union algorithm.

    Parameters
    ----------
    geometries : geopandas.GeoSeries
        Geometries in a common coordinate reference system.

    Returns
    -------
    shapely.geometry.base.BaseGeometry
        Union of the input geometries in their original coordinate system.

    Notes
    -----
    Prefer ``union_all()`` on current GeoPandas releases to avoid the deprecated
    ``unary_union`` property. Fall back to that property only on older releases
    without ``union_all``. Both paths retain the default unary union algorithm;
    no coverage approximation, reprojection or precision snapping is requested.
    """
    union_all = getattr(geometries, "union_all", None)
    if union_all is not None:
        return union_all()
    return geometries.unary_union


class ReScaling(object):
    """
    Store the configuration needed to apply regional emission rescaling.

    Factors are calculated after inventory regridding, using local totals that match the inventory frequency. The
    inventory total for region ``r`` and pollutant ``p`` is::

        I[r, p] = sum(E[p, y, x] * A[y, x] * seconds_per_period * M[r, y, x])

    where ``E`` is the regridded emission flux in ``kg.m-2.s-1``, ``A`` is the destination cell area and ``M`` is the
    regional mask. The factor is ``local_total / inventory_total``.

    Attributes
    ----------
    master : bool
        Indicates if the process is the master one.
    comm : MPI.Comm
        MPI communicator.
    grid : Nes
        Destination grid where the rescaling will be applied.
    apply_to_all_emissions : bool
        Indicates if derived factors apply to the complete inventory instead of only the selected regions.
    region_totals : pandas.DataFrame
        Local totals loaded during initialisation. For date-aware configurations this is the default file and is used
        to discover initial regions and inventory/sector pairs.
    region_totals_by_path : dict
        Cache of inventory-period local totals keyed by path, frequency, inventory and sector.
    region_mask : Nes or None
        Regional mask over the destination grid. It is created lazily from the selected local totals file.
    """

    def __init__(
        self,
        grid,
        rescaling_geometries,
        rescaling_geometries_attribute,
        rescaling_local_total_emissions,
        auxiliary_files_path,
        rescaling_everything,
        rescaling_regions=None,
        rescaling_allow_partial_regions=False,
    ):
        """
        Initialise the regional emission rescaling configuration.

        Parameters
        ----------
        grid : Nes
            Destination grid where the rescaling will be applied.
        rescaling_geometries : str
            Path to the shapefile or GeoJSON containing the rescaling regions. The file must contain the column given
            by ``rescaling_geometries_attribute`` and a valid ``geometry`` column. Example header when
            ``rescaling_geometries_attribute`` is ``NAME_4``::

                NAME_4,geometry
                Barcelona,<Polygon>
                A Coruña,<Polygon>

        rescaling_geometries_attribute : str
            Geometry attribute containing the region identifier. Values are matched with the ``Region`` values in the
            local totals CSV. Regions present in the geometry file but not present in the CSV are merged into a common
            ``OTHERS`` region before the spatial join.
        rescaling_local_total_emissions : str
            Path or path template to the CSV file containing reference regional emission totals. If it contains
            ``<date>``, yearly inventories use simulation-date ``YYYY`` and monthly inventories use simulation-date
            ``YYYYMM``. The fallback path uses ``default`` in place of ``<date>``.
        auxiliary_files_path : str
            Path to the auxiliary files directory.
        rescaling_everything : bool
            Indicates if derived factors apply to the complete inventory instead of only the selected regions.
        rescaling_regions : str, optional
            Comma-separated region names. Empty, omitted or ALL selects every region in the local totals CSV.
        rescaling_allow_partial_regions : bool, optional
            Continue with a warning instead of failing when selected regions are not fully inside the domain.

        Raises
        ------
        ValueError
            If the initial local totals file is malformed or if ``rescaling_everything`` is enabled with more than one
            configured region.
        FileNotFoundError
            If the initial local totals file cannot be read.
        """
        self.master = grid.master
        self.comm = grid.comm
        self.grid = grid
        self.apply_to_all_emissions = rescaling_everything
        self.selected_regions = self._parse_regions(rescaling_regions)
        self.allow_partial_regions = rescaling_allow_partial_regions
        self.local_total_emissions_path_template = rescaling_local_total_emissions
        self.auxiliary_files_path = auxiliary_files_path
        self.rescaling_geometries = rescaling_geometries
        self.rescaling_geometries_attribute = rescaling_geometries_attribute
        self.region_totals_by_path = {}
        self.region_mask = None
        self.validated_domain_regions = set()
        self.domain_geometry = None
        self.domain_geometry_initialized = False
        self.domain_region_geometries = None
        self.debug_inventory_totals_files = set()
        self.debug_factors_files = set()

        # Phase 1: read the reference totals and derive the target regions from the configured/default CSV.
        self.region_totals = self._read_initial_region_totals(rescaling_local_total_emissions)
        region_ids = self._get_region_ids()
        self._check_rescaling_everything(self.selected_regions if self.selected_regions is not None else region_ids)

        # Phase 2 is deferred until the inventory-specific local totals are known. This allows date-aware monthly
        # local-total files to define the active regions.

        if hermes_gr.DEBUG:
            log_message("DEBUG ReScaling initialization")
            log_message(f"grid: {self.grid.get_full_bbox()}", level=5)
            log_message(f"rescaling_geometries: {rescaling_geometries}", level=5)
            log_message(f"rescaling_geometries_attribute: {rescaling_geometries_attribute}", level=5)
            log_message(f"rescaling_local_total_emissions: {rescaling_local_total_emissions}", level=5)
            log_message(f"auxiliary_files_path: {auxiliary_files_path}", level=5)
            log_message(f"rescaling_everything: {self.apply_to_all_emissions}", level=5)
            log_message(f"regions: {region_ids}", level=5)
            log_message(f"selected_regions: {self.selected_regions}", level=5)
            log_message(f"rescaling_allow_partial_regions: {self.allow_partial_regions}", level=5)
            log_message(f"region_totals: {self.region_totals}", level=5)
            log_message(f"region_mask: {self.region_mask}", level=5)

    @staticmethod
    def _parse_regions(value):
        """Parse a comma-separated selection while reserving ALL for the complete CSV."""
        if value is None or not value.strip():
            return None
        regions = list(dict.fromkeys(region.strip() for region in value.split(",")))
        if any(region.upper() == "ALL" for region in regions):
            if len(regions) == 1:
                return None
            raise ValueError("ERROR: rescaling_regions must use ALL alone, not combined with region names.")
        if "" in regions or OTHERS_REGION_ID in regions:
            raise ValueError("ERROR: rescaling_regions cannot contain empty names or the reserved OTHERS region.")
        return regions

    def _select_region_totals(self, region_totals):
        """Validate the selection against the complete CSV, then retain matching regional rows."""
        selected_regions = getattr(self, "selected_regions", None)
        if selected_regions is None:
            return region_totals
        missing_regions = sorted(set(selected_regions).difference(self._get_region_ids(region_totals)))
        if missing_regions:
            raise ValueError(
                "ERROR: rescaling_regions contains regions absent from the selected local totals CSV: "
                f"{', '.join(missing_regions)}. Region names must match the Region column exactly."
            )
        return region_totals.loc[region_totals.index.get_level_values(REGION_ID_COLUMN).isin(selected_regions)]

    def _check_rescaling_everything(self, region_ids):
        """
        Check if applying the rescaling factors to all emissions is compatible with the configured regions.

        ``apply_to_all_emissions`` can only be used when a single target region is configured. With more than one
        region, a single inventory-wide factor would be ambiguous because each region can have different reference
        totals and therefore different rescaling factors.

        Parameters
        ----------
        region_ids : list[str]
            Region identifiers found in the local total emissions file.

        Raises
        ------
        ValueError
            If more than one region is configured and ``apply_to_all_emissions`` is enabled.
        """
        if self.apply_to_all_emissions and len(region_ids) > 1:
            raise ValueError(
                "ERROR: 'rescaling_everything' can only be True when a single rescaling region is configured. "
                f"Configured regions: {', '.join(region_ids)}."
            )

        return None

    @staticmethod
    def _get_region_mask_path(auxiliary_files_path):
        """
        Get the path to the auxiliary regional mask file.

        Parameters
        ----------
        auxiliary_files_path : str
            Path to the auxiliary files directory.

        Returns
        -------
        str
            Path to the auxiliary regional mask file.
        """
        return os.path.join(auxiliary_files_path, REGION_MASK_FILE)

    @staticmethod
    def _get_region_mask_geojson_path(auxiliary_files_path):
        """
        Get the path to the debug regional mask GeoJSON file.

        Parameters
        ----------
        auxiliary_files_path : str
            Path to the auxiliary files directory.

        Returns
        -------
        str
            Path to the debug regional mask GeoJSON file.
        """
        return os.path.join(auxiliary_files_path, REGION_MASK_GEOJSON_FILE)

    @staticmethod
    def _get_inventory_totals_path(
        output_path,
        reference_year,
        input_frequency,
        date,
    ):
        """
        Get the path to the CSV file with regional inventory totals.

        Parameters
        ----------
        output_path : str
            Path to the output directory.
        reference_year : int
            Emission inventory reference year.
        input_frequency : str
            Frequency of the input emission inventory.
        date : datetime
            Date to simulate.

        Returns
        -------
        str
            Path to the regional inventory totals CSV file.
        """
        date_suffix = ReScaling._get_inventory_totals_date_suffix(reference_year, input_frequency, date)
        file_name = INVENTORY_TOTALS_FILE_TEMPLATE.format(date_suffix=date_suffix)

        return os.path.join(output_path, file_name)

    def _write_debug_csv_rows(self, output_path, rows, tracker_attribute):
        """
        Write or update a debug CSV for the current execution.

        The first write to a path in the current execution overwrites any previous run output. Later writes merge rows
        with the existing in-run content so columns remain aligned even when inventories expose different pollutants.

        Parameters
        ----------
        output_path : str
            Path to the CSV file to write.
        rows : pandas.DataFrame
            Rows to write.
        tracker_attribute : str
            Name of the instance set tracking paths already written during the current execution.
        """
        written_files = getattr(self, tracker_attribute, set())
        if output_path in written_files:
            rows = pd.concat([pd.read_csv(output_path), rows], ignore_index=True, sort=False)

        rows.to_csv(output_path, index=False)
        written_files.add(output_path)
        setattr(self, tracker_attribute, written_files)

    @staticmethod
    def _get_factors_path(
        output_path,
        reference_year,
        input_frequency,
        date,
    ):
        """
        Get the path to the CSV file with regional rescaling factors.

        Parameters
        ----------
        output_path : str
            Path to the output directory.
        reference_year : int
            Emission inventory reference year.
        input_frequency : str
            Frequency of the input emission inventory.
        date : datetime
            Date to simulate.

        Returns
        -------
        str
            Path to the regional rescaling factors CSV file.
        """
        date_suffix = ReScaling._get_factors_date_suffix(reference_year, input_frequency, date)
        file_name = FACTORS_FILE_TEMPLATE.format(date_suffix=date_suffix)

        return os.path.join(output_path, file_name)

    @staticmethod
    def _get_factors_date_suffix(reference_year, input_frequency, date):
        """
        Get the date suffix for the regional rescaling factors CSV.

        The factors compare the reference inventory period with the simulation period:

        - yearly: ``<reference_year>_to_<YYYY>``
        - monthly: ``<reference_year><MM>_to_<YYYYMM>``

        Parameters
        ----------
        reference_year : int
            Emission inventory reference year.
        input_frequency : str
            Frequency of the input emission inventory.
        date : datetime
            Date to simulate.

        Returns
        -------
        str
            Date suffix.
        """
        reference_year = ReScaling._format_reference_year(reference_year)
        if input_frequency == "yearly":
            return f"{reference_year}_to_{date.strftime('%Y')}"
        if input_frequency == "monthly":
            return f"{reference_year}{date.strftime('%m')}_to_{date.strftime('%Y%m')}"
        raise ValueError(
            "ERROR: Regional emission rescaling factors are only supported for yearly and monthly "
            f"emission inventories. Found frequency '{input_frequency}'."
        )

    @staticmethod
    def _get_inventory_totals_date_suffix(reference_year, input_frequency, date):
        """
        Get the date suffix for the regional inventory totals CSV.

        The suffix follows the same date convention as the regridded emission auxiliary file name:

        - yearly: ``<reference_year>``
        - monthly: ``<reference_year><MM>``
        - daily: ``<reference_year><MM><DD>``
        - hourly: ``<reference_year><MM><DD><HH>``

        Parameters
        ----------
        reference_year : int
            Emission inventory reference year.
        input_frequency : str
            Frequency of the input emission inventory.
        date : datetime
            Date to simulate.

        Returns
        -------
        str
            Date suffix.
        """
        if input_frequency == "yearly":
            date_suffix = f"{reference_year}"
        elif input_frequency == "monthly":
            date_suffix = f"{reference_year}{date.strftime('%m')}"
        elif input_frequency == "daily":
            date_suffix = f"{reference_year}{date.strftime('%m')}{date.strftime('%d')}"
        elif input_frequency == "hourly":
            date_suffix = f"{reference_year}{date.strftime('%m')}{date.strftime('%d')}{date.strftime('%H')}"
        else:
            raise ValueError(
                f"ERROR: frequency {input_frequency} not implemented. Use yearly, monthly, daily or hourly."
            )

        return date_suffix

    @staticmethod
    def _get_local_totals_date_suffix(reference_year, input_frequency, date):
        """
        Get the date suffix used to resolve date-aware local-total paths.

        Parameters
        ----------
        reference_year : int or str
            Inventory reference year. Kept for API consistency; the local-total path uses ``date``.
        input_frequency : str
            Inventory frequency. Supported values are ``yearly`` and ``monthly``.
        date : datetime.datetime
            Simulation date used to select the month for monthly inventories.

        Returns
        -------
        str
            Date suffix from the simulation date. Yearly inventories use ``YYYY`` and monthly inventories use
            ``YYYYMM``.

        Raises
        ------
        ValueError
            If ``input_frequency`` is not supported by regional rescaling local totals.
        """
        if input_frequency == "yearly":
            return date.strftime("%Y")
        if input_frequency == "monthly":
            return date.strftime("%Y%m")
        raise ValueError(
            "ERROR: Regional emission rescaling local totals are only supported for yearly and monthly "
            f"emission inventories. Found frequency '{input_frequency}'."
        )

    @staticmethod
    def _format_reference_year(reference_year):
        """
        Format integer-like reference years without pandas float artifacts.

        Parameters
        ----------
        reference_year : int, float or str
            Reference year read from the inventory configuration.

        Returns
        -------
        str
            Reference year formatted without a trailing decimal part when it is integer-like.
        """
        try:
            year = float(reference_year)
        except ValueError:
            return str(reference_year)
        if year.is_integer():
            return str(int(year))
        return str(reference_year)

    @staticmethod
    def _get_default_local_totals_path(local_total_emissions_path):
        """
        Get the default fallback path for a date-aware local-total path.

        Parameters
        ----------
        local_total_emissions_path : str
            Configured local totals path template.

        Returns
        -------
        str
            Path obtained by replacing ``<date>`` with ``default``.
        """
        return local_total_emissions_path.replace(DATE_PLACEHOLDER, "default")

    def _resolve_local_totals_path(self, local_total_emissions_path, reference_year, input_frequency, date):
        """
        Resolve the configured local-total path for the current inventory period.

        Parameters
        ----------
        local_total_emissions_path : str
            Configured CSV path or path template. Templates can contain ``<date>``.
        reference_year : int or str
            Inventory reference year.
        input_frequency : str
            Inventory frequency. Supported date-aware values are ``yearly`` and ``monthly``.
        date : datetime.datetime
            Simulation date.

        Returns
        -------
        str
            Existing local totals path to read.

        Raises
        ------
        FileNotFoundError
            If neither the resolved date-aware path nor its default fallback exists.
        ValueError
            If the path is date-aware and ``input_frequency`` is not supported.

        Notes
        -----
        Fixed paths without ``<date>`` are returned unchanged and emit a warning because the same totals are used for
        all periods.
        """
        if DATE_PLACEHOLDER not in local_total_emissions_path:
            resolved_path = local_total_emissions_path
            if self.master:
                log_message(
                    "WARNING: Regional rescaling local totals path does not contain "
                    f"{DATE_PLACEHOLDER}. Using configured file for all periods: {resolved_path}",
                    level=7,
                )
            return resolved_path

        date_suffix = self._get_local_totals_date_suffix(reference_year, input_frequency, date)
        resolved_path = local_total_emissions_path.replace(DATE_PLACEHOLDER, date_suffix)
        if os.path.exists(resolved_path):
            if self.master:
                log_message(f"Using regional rescaling local totals file: {resolved_path}", level=2)
            return resolved_path

        default_path = self._get_default_local_totals_path(local_total_emissions_path)
        if os.path.exists(default_path):
            if self.master:
                log_message(
                    "WARNING: Regional rescaling local totals file "
                    f"{resolved_path} not found. Falling back to default file: {default_path}",
                    level=7,
                )
            return default_path

        raise FileNotFoundError(
            "ERROR: Regional rescaling local totals file not found. "
            f"Expected '{resolved_path}' or fallback default file '{default_path}'."
        )

    @staticmethod
    def _clean_file_name(value):
        """
        Clean a value to be safely used inside an auxiliary file name.

        Parameters
        ----------
        value : object
            Value to clean.

        Returns
        -------
        str
            Clean file-name fragment.
        """
        value = str(value)

        return "".join(character if character.isalnum() or character in ["-", "_", "."] else "_" for character in value)

    def _read_initial_region_totals(self, local_total_emissions_path):
        """
        Read the configured/default regional totals during initialisation.

        Parameters
        ----------
        local_total_emissions_path : str
            Configured CSV path or path template.

        Returns
        -------
        pandas.DataFrame
            Regional totals read without inventory-frequency validation.
        """
        if DATE_PLACEHOLDER in local_total_emissions_path:
            local_total_emissions_path = self._get_default_local_totals_path(local_total_emissions_path)
        return self._read_region_totals(local_total_emissions_path)

    def _get_region_totals_for_period(self, reference_year, input_frequency, date, inventory_name=None, sector=None):
        """
        Read and cache local totals for the current inventory period.

        Parameters
        ----------
        reference_year : int or str
            Inventory reference year.
        input_frequency : str
            Inventory frequency.
        date : datetime.datetime
            Simulation date.
        inventory_name : str, optional
            Inventory name used to validate only the selected inventory/sector rows.
        sector : str, optional
            Sector name used to validate only the selected inventory/sector rows.

        Returns
        -------
        pandas.DataFrame
            Local totals loaded from the resolved path and normalized to kg for the selected period.
        """
        local_totals_path = self._resolve_local_totals_path(
            self.local_total_emissions_path_template,
            reference_year,
            input_frequency,
            date,
        )
        cache_key = (local_totals_path, input_frequency, inventory_name, sector)
        if cache_key not in self.region_totals_by_path:
            self.region_totals_by_path[cache_key] = self._read_region_totals(
                local_totals_path,
                input_frequency=input_frequency,
                inventory_name=inventory_name,
                sector=sector,
            )
        return self.region_totals_by_path[cache_key]

    def _read_region_totals(self, local_total_emissions_path, input_frequency=None, inventory_name=None, sector=None):
        """
        Read the reference regional emission totals and convert them to kg for their period.

        Parameters
        ----------
        local_total_emissions_path : str
            Path to the CSV file containing reference regional emission totals.
        input_frequency : str, optional
            Inventory frequency used to validate temporal units. When omitted, only unit support is checked.
        inventory_name : str, optional
            Inventory name used to restrict frequency validation to one inventory/sector pair.
        sector : str, optional
            Sector name used to restrict frequency validation to one inventory/sector pair.

        Returns
        -------
        pandas.DataFrame
            Regional totals indexed by ``region_id``, ``ei`` and ``sector``. The ``units`` column is removed and
            pollutant totals are expressed in kg for their configured period.

        Raises
        ------
        ValueError
            If any mandatory column is missing.
        ValueError
            If the ``units`` column contains unsupported units or units incompatible with ``input_frequency``.

        Notes
        -----
        The CSV must include ``Region``, ``ei``, ``sector`` and ``units``. Remaining columns are pollutant totals.
        Supported units are ``kg/year``, ``t/year``, ``kg/month`` and ``t/month``. Tonnes are converted to kg.
        """
        if self.master:
            # Read and validate the CSV only in master; the checked dataframe is broadcast afterwards.
            region_totals = pd.read_csv(local_total_emissions_path)
            if LOCAL_TOTALS_REGION_COLUMN not in region_totals.columns:
                raise ValueError(
                    f"ERROR: Column '{LOCAL_TOTALS_REGION_COLUMN}' is not in {local_total_emissions_path} file."
                )
            region_totals = region_totals.rename(columns={LOCAL_TOTALS_REGION_COLUMN: REGION_ID_COLUMN})
            for column in REGION_TOTALS_INDEX_COLUMNS + [UNITS_COLUMN]:
                if column not in region_totals.columns:
                    raise ValueError(f"ERROR: Column '{column}' is not in {local_total_emissions_path} file.")

            unsupported_units = sorted(set(region_totals[UNITS_COLUMN].dropna()).difference(SUPPORTED_UNITS))
            if len(unsupported_units) > 0:
                raise ValueError(
                    "ERROR: Unsupported regional emission rescaling units: "
                    f"{', '.join(unsupported_units)}. Supported units are: {', '.join(SUPPORTED_UNITS)}."
                )
            if input_frequency is not None:
                self._check_units_match_frequency(
                    region_totals,
                    local_total_emissions_path,
                    input_frequency,
                    inventory_name=inventory_name,
                    sector=sector,
                )

            value_columns = [column for column in region_totals.columns
                             if column not in REGION_TOTALS_INDEX_COLUMNS + [UNITS_COLUMN]]
            region_totals[value_columns] = region_totals[value_columns].apply(pd.to_numeric, errors="coerce")

            # Normalize all supported units to kg for their configured period so factor calculations use one mass unit.
            for unit, factor in UNIT_FACTORS_TO_KG.items():
                unit_mask = region_totals[UNITS_COLUMN] == unit
                region_totals.loc[unit_mask, value_columns] = region_totals.loc[unit_mask, value_columns] * factor

            # Store the identifying columns as an index; pollutant columns remain as numeric data.
            region_totals = region_totals.drop(columns=[UNITS_COLUMN])
            region_totals = region_totals.set_index(REGION_TOTALS_INDEX_COLUMNS)
        else:
            region_totals = None

        self.comm.Barrier()
        return self.comm.bcast(region_totals, root=0)

    @staticmethod
    def _check_units_match_frequency(
        region_totals,
        local_total_emissions_path,
        input_frequency,
        inventory_name=None,
        sector=None,
    ):
        """
        Check that local-total units are compatible with the emission inventory frequency.

        Parameters
        ----------
        region_totals : pandas.DataFrame
            Raw local totals table before dropping the ``units`` column.
        local_total_emissions_path : str
            Path used in error messages.
        input_frequency : str
            Inventory frequency.
        inventory_name : str, optional
            Inventory name used to validate only matching rows.
        sector : str, optional
            Sector name used to validate only matching rows.

        Raises
        ------
        ValueError
            If the inventory frequency is unsupported or if matching rows use incompatible units.
        """
        if input_frequency not in PERIOD_UNITS:
            raise ValueError(
                "ERROR: Regional emission rescaling local totals are only supported for yearly and monthly "
                f"emission inventories. Found frequency '{input_frequency}' while reading "
                f"{local_total_emissions_path}."
            )

        if inventory_name is not None and sector is not None:
            inventory_mask = (region_totals[EI_COLUMN] == inventory_name) & (region_totals[SECTOR_COLUMN] == sector)
            region_totals = region_totals.loc[inventory_mask]

        supported_units = PERIOD_UNITS[input_frequency]
        incompatible_units = sorted(set(region_totals[UNITS_COLUMN].dropna()).difference(supported_units))
        if len(incompatible_units) > 0:
            raise ValueError(
                "ERROR: Incompatible regional emission rescaling units for "
                f"{input_frequency} emission inventory in {local_total_emissions_path}: "
                f"{', '.join(incompatible_units)}. Supported units for {input_frequency} inventories are: "
                f"{', '.join(supported_units)}."
            )

    def _get_region_ids(self, region_totals=None):
        """
        Get the region identifiers configured in the reference regional emission totals.

        Region identifiers are read from the ``region_id`` level of the supplied totals, or ``self.region_totals``.
        The special ``OTHERS`` ID
        is explicitly excluded even if it is accidentally present in the local totals CSV. ``OTHERS`` is an internal
        mask level only and is added only when building the regional mask.

        Returns
        -------
        list[str]
            Sorted unique region identifiers.
        """
        if region_totals is None:
            region_totals = self.region_totals
        region_ids = region_totals.index.get_level_values(REGION_ID_COLUMN).dropna().astype(str)
        region_ids = region_ids[region_ids != OTHERS_REGION_ID].unique()

        return sorted(region_ids)

    def do_i_have_to_rescale(self, inventory_name, sector, input_frequency=None, reference_year=None, date=None):
        """
        Check if an emission inventory and sector are configured for regional rescaling.

        Parameters
        ----------
        inventory_name : str
            Emission inventory name, matching the ``ei`` level of ``region_totals``.
        sector : str
            Sector name, matching the ``sector`` level of ``region_totals``.
        input_frequency : str, optional
            Inventory frequency used to resolve date-aware local totals.
        reference_year : int or str, optional
            Inventory reference year used to resolve date-aware local totals.
        date : datetime.datetime, optional
            Simulation date used to resolve date-aware local totals.

        Returns
        -------
        bool
            True if the selected local totals file contains at least one matching inventory/sector row.

        Raises
        ------
        ValueError
            If a configured inventory/sector pair uses a frequency unsupported by regional rescaling.
        """
        region_totals = self.region_totals
        if input_frequency is not None and reference_year is not None and date is not None:
            if input_frequency not in PERIOD_UNITS:
                has_inventory_sector = self._has_inventory_sector(inventory_name, sector, region_totals=region_totals)
                if has_inventory_sector:
                    raise ValueError(
                        "ERROR: Regional emission rescaling local totals are only supported for yearly and monthly "
                        f"emission inventories. Inventory '{inventory_name}' and sector '{sector}' are configured "
                        f"for rescaling but frequency is '{input_frequency}'."
                    )
                return False
            region_totals = self._get_region_totals_for_period(
                reference_year,
                input_frequency,
                date,
                inventory_name=inventory_name,
                sector=sector,
            )

        return self._has_inventory_sector(inventory_name, sector, region_totals=region_totals)

    def _has_inventory_sector(self, inventory_name, sector, region_totals=None):
        """
        Check if regional totals are available for an inventory and sector pair.

        Parameters
        ----------
        inventory_name : str
            Emission inventory name, matching the ``ei`` level of ``region_totals``.
        sector : str
            Sector name, matching the ``sector`` level of ``region_totals``.

        Returns
        -------
        bool
            True if the pair is present in ``region_totals``.
        """
        if region_totals is None:
            region_totals = self.region_totals
        region_totals = self._select_region_totals(region_totals)
        inventory_sector_index = region_totals.index.droplevel(REGION_ID_COLUMN).unique()

        return (inventory_name, sector) in inventory_sector_index

    def _get_inventory_sector_totals(self, inventory_name, sector, region_totals=None):
        """
        Get regional totals for an inventory and sector pair.

        ``OTHERS`` is always removed from the returned table. It can exist in the mask, but it is not a target region
        for factor calculation.

        Parameters
        ----------
        inventory_name : str
            Emission inventory name, matching the ``ei`` level of ``region_totals``.
        sector : str
            Sector name, matching the ``sector`` level of ``region_totals``.

        Returns
        -------
        pandas.DataFrame
            Regional totals indexed by ``region_id`` for the requested inventory and sector.

        Raises
        ------
        KeyError
            If the requested ``inventory_name`` and ``sector`` pair is not present in ``region_totals``.
        """
        if region_totals is None:
            region_totals = self.region_totals
        region_totals = self._select_region_totals(region_totals)
        local_totals = region_totals.xs((inventory_name, sector), level=[EI_COLUMN, SECTOR_COLUMN])

        return local_totals.loc[local_totals.index != OTHERS_REGION_ID]

    def _get_region_mask_ids(self):
        """
        Get the region identifiers stored as vertical levels in the regional mask.

        Returns
        -------
        list[str]
            Region identifiers ordered as the mask vertical levels.

        Notes
        -----
        This list can contain ``OTHERS`` because it describes the auxiliary mask layout, not the configured target
        regions.
        """
        return self.region_mask.variables[REGION_MASK_VAR]["region_ids"].split(REGION_IDS_SEPARATOR)

    def calculate_rescaling_factors(
        self,
        emissions,
        inventory_name,
        sector,
        reference_year,
        input_frequency,
        date,
        auxiliary_files_path,
        output_path,
    ):
        """
        Calculate inventory totals and compare them with the configured local totals.

        Parameters
        ----------
        emissions : Nes
            Emissions after ``do_regrid``. Variables are expected in ``kg.m-2.s-1``.
        inventory_name : str
            Emission inventory name.
        sector : str
            Sector name.
        reference_year : int or str
            Emission inventory reference year.
        input_frequency : str
            Input inventory frequency. Regional rescaling supports ``yearly`` and ``monthly`` totals.
        date : datetime.datetime
            Date to simulate.
        auxiliary_files_path : str
            Path to the auxiliary files directory.
        output_path : str
            Path to the output directory where debug CSV files are written.

        Returns
        -------
        dict
            Nested dictionary with regions as first-level keys and pollutants as second-level keys. Values are:
            ``None`` when the local total is missing, ``0`` when the inventory total is zero, or the
            ``local_total / inventory_total`` fraction otherwise.

        Raises
        ------
        ValueError
            If the resolved local totals use units incompatible with ``input_frequency`` or if the regional mask cannot
            represent the selected local-total regions, or a selected region is not completely inside the domain
            and ``rescaling_allow_partial_regions`` is disabled.

        Warnings
        --------
        A warning is emitted through ``log_message(..., level=7)`` when a positive local total is configured but the
        original inventory total is zero or ``NaN`` for that region and pollutant. In that case, the factor is set to
        ``0`` because there are no original emissions to rescale.

        Notes
        -----
        This method runs after ``prepare_input_emissions`` so it works with either newly regridded emissions or
        emissions loaded from a regridded auxiliary file.

        In ``DEBUG`` mode, the inventory totals used to derive the factors and the final rescaling factors are written
        as CSV files in the output directory. In normal runs these CSVs are not written. ``DEBUG`` mode also logs the
        calculated factors.
        """
        log_message("Calculating regional inventory totals for rescaling.", level=2)

        region_totals = self._get_region_totals_for_period(
            reference_year,
            input_frequency,
            date,
            inventory_name=inventory_name,
            sector=sector,
        )
        local_totals = self._get_inventory_sector_totals(inventory_name, sector, region_totals=region_totals)
        selected_region_ids = self._get_region_ids(self._select_region_totals(region_totals))
        self._validate_regions_in_domain(selected_region_ids)
        self._check_rescaling_everything(local_totals.index.astype(str).tolist())
        # Sectors may cover different regions; the shared mask must cover the complete selected totals file.
        self._ensure_region_mask(self._get_region_ids(region_totals))
        seconds_per_period = self._get_inventory_total_seconds(reference_year, input_frequency, date)

        # Calculate inventory-derived totals first, then restrict them to configured local-total regions.
        inventory_totals = self._calculate_inventory_totals(emissions, seconds_per_period=seconds_per_period)
        inventory_totals = inventory_totals.reindex(index=local_totals.index)

        # OTHERS is a spatial attribution helper in the mask, not a rescaling target.
        inventory_totals = inventory_totals.drop(index=OTHERS_REGION_ID, errors="ignore")
        inventory_totals_path = self._get_inventory_totals_path(
            output_path,
            reference_year,
            input_frequency,
            date,
        )

        # Inventory totals are diagnostic output only; keep normal runs from writing extra output files.
        if hermes_gr.DEBUG and self.master:
            inventory_totals_to_write = inventory_totals.copy()
            inventory_totals_to_write.insert(0, UNITS_COLUMN, self._get_inventory_total_units(input_frequency))
            inventory_totals_to_write.index.name = REGION_ID_COLUMN
            inventory_totals_to_write = inventory_totals_to_write.reset_index()
            inventory_totals_to_write.insert(1, EI_COLUMN, inventory_name)
            inventory_totals_to_write.insert(2, SECTOR_COLUMN, sector)
            self._write_debug_csv_rows(
                inventory_totals_path,
                inventory_totals_to_write,
                "debug_inventory_totals_files",
            )
            log_message(f"Regional inventory totals written at {inventory_totals_path}", level=3)

        # Derive final multiplicative factors from reference totals and inventory-derived totals.
        factors = self._calculate_factors_from_totals(
            inventory_totals,
            local_totals,
            inventory_name=inventory_name,
            sector=sector,
        )
        factors.pop(OTHERS_REGION_ID, None)

        if hermes_gr.DEBUG and self.master:
            factors_path = self._get_factors_path(
                output_path,
                reference_year,
                input_frequency,
                date,
            )
            factors_to_write = pd.DataFrame.from_dict(factors, orient="index")
            factors_to_write.index.name = REGION_ID_COLUMN
            factors_to_write = factors_to_write.reset_index()
            factors_to_write.insert(1, EI_COLUMN, inventory_name)
            factors_to_write.insert(2, SECTOR_COLUMN, sector)
            self._write_debug_csv_rows(
                factors_path,
                factors_to_write,
                "debug_factors_files",
            )
            log_message(f"Regional rescaling factors written at {factors_path}", level=3)

        if hermes_gr.DEBUG:
            log_message("DEBUG ReScaling factors.", level=5)
            log_message(f"inventory_name: {inventory_name}", level=5)
            log_message(f"sector: {sector}", level=5)
            log_message(f"rescaling_factors: {factors}", level=5)

        return factors

    def _validate_regions_in_domain(self, region_ids):
        """Reject selected regions whose complete geometry is not covered by the global grid-cell footprint.

        Parameters
        ----------
        region_ids : list[str]
            Selected region identifiers requiring complete domain coverage.

        Returns
        -------
        None
            Cache successful checks in ``validated_domain_regions`` and retain
            the global footprint and region geometries for subsequent calls.

        Raises
        ------
        ValueError
            A selected region is missing, has invalid geometry or extends outside
            the domain, or the master rank cannot complete geometry validation.

        Notes
        -----
        All MPI ranks participate in gathering the domain and receive the same validation error. Geometry inputs
        are read without bbox filtering so detached parts and regions outside the domain cannot be overlooked.
        Successful checks are cached for the execution, independently of auxiliary mask reuse.
        With ``allow_partial_regions``, incomplete coverage emits a warning and the usual factor calculation continues.
        Missing or invalid geometries remain errors.
        """
        if set(region_ids).issubset(self.validated_domain_regions):
            return None

        domain_parts = None
        if not self.domain_geometry_initialized:
            self.grid.create_shapefile()
            local_domain = _union_geometries(self.grid.shapefile.to_crs("EPSG:4326").geometry)
            domain_parts = self.comm.gather(local_domain, root=0)

        error = None
        if self.master:
            try:
                if self.domain_geometry is None:
                    self.domain_geometry = _union_geometries(gpd.GeoSeries(domain_parts, crs="EPSG:4326"))
                if self.domain_region_geometries is None:
                    geometries = gpd.read_file(self.rescaling_geometries)
                    attribute = self.rescaling_geometries_attribute
                    if attribute not in geometries.columns:
                        raise ValueError(f"Column '{attribute}' is not in {self.rescaling_geometries} file.")
                    geometries = geometries.to_crs("EPSG:4326")
                    geometries[attribute] = geometries[attribute].astype(str)
                    self.domain_region_geometries = geometries

                geometries = self.domain_region_geometries
                attribute = self.rescaling_geometries_attribute
                missing = sorted(set(region_ids).difference(geometries[attribute]))
                if missing:
                    raise ValueError(f"Selected regions missing from the geometry file: {', '.join(missing)}.")
                selected = geometries.loc[geometries[attribute].isin(region_ids)]
                if (selected.geometry.isna().any() or selected.geometry.is_empty.any()
                        or not selected.geometry.is_valid.all()):
                    raise ValueError("Selected rescaling regions contain missing, empty or invalid geometries.")
                regions = selected.dissolve(by=attribute)
                outside = sorted(region_id for region_id, geometry in regions.geometry.items()
                                 if not self.domain_geometry.covers(geometry))
                if outside:
                    message = (
                        f"Of the {len(region_ids)} selected rescaling regions, {len(outside)} are not completely "
                        f"inside the simulation domain: [{', '.join(outside)}]."
                    )
                    if self.allow_partial_regions:
                        log_message(
                            f"WARNING: {message} Continuing because rescaling_allow_partial_regions=True. "
                            "The in-domain emissions will be scaled to the full regional local totals, which may "
                            "overestimate emissions within the domain. Regions with zero inventory totals retain "
                            "the existing zero-factor behavior.",
                            level=7,
                        )
                    else:
                        raise ValueError(
                            f"{message} Regional rescaling requires complete regions. Expand the simulation domain, "
                            "exclude these regions using rescaling_regions, or explicitly set "
                            "rescaling_allow_partial_regions=True to continue with a warning."
                        )
            except Exception as exc:
                error = f"ERROR: Regional rescaling domain validation failed. {exc}"

        error = self.comm.bcast(error, root=0)
        if error is not None:
            raise ValueError(error)
        self.domain_geometry_initialized = True
        self.validated_domain_regions.update(region_ids)
        return None

    def _ensure_region_mask(self, region_ids):
        """
        Ensure the reusable region mask exists and contains the requested regions.

        Parameters
        ----------
        region_ids : list[str]
            Region identifiers required by the selected local totals file.

        Raises
        ------
        ValueError
            If an existing mask does not contain all requested regions.
        """
        if self.region_mask is None:
            self.region_mask = self._obtain_region_mask(
                self.auxiliary_files_path,
                self.rescaling_geometries,
                self.rescaling_geometries_attribute,
                region_ids,
            )

        mask_region_ids = set(self._get_region_mask_ids())
        missing_region_ids = sorted(set(region_ids).difference(mask_region_ids))
        if len(missing_region_ids) > 0:
            raise ValueError(
                "ERROR: Existing regional rescaling mask does not contain all regions required by the selected "
                "local totals file. Missing regions: "
                f"{', '.join(missing_region_ids)}. Delete only the regional rescaling mask file "
                f"'{self._get_region_mask_path(self.auxiliary_files_path)}' and rerun with local totals containing "
                "all required regions to regenerate the mask. The other auxiliary files can be reused."
            )

        return None

    @staticmethod
    def _get_inventory_total_seconds(reference_year, input_frequency, date):
        """
        Get the integration period used to convert flux emissions into inventory totals.

        Parameters
        ----------
        reference_year : int or str
            Inventory reference year.
        input_frequency : str
            Inventory frequency. Supported values are ``yearly`` and ``monthly``.
        date : datetime.datetime
            Simulation date used to select the month length for monthly inventories.

        Returns
        -------
        int
            Number of seconds in the inventory period.

        Raises
        ------
        ValueError
            If ``input_frequency`` is unsupported by regional rescaling.
        """
        if input_frequency == "yearly":
            return SECONDS_PER_YEAR
        if input_frequency == "monthly":
            return monthrange(int(float(reference_year)), date.month)[1] * SECONDS_PER_DAY
        raise ValueError(
            "ERROR: Regional emission rescaling inventory totals are only supported for yearly and monthly "
            f"emission inventories. Found frequency '{input_frequency}'."
        )

    @staticmethod
    def _get_inventory_total_units(input_frequency):
        """
        Get the diagnostic inventory-total units for the inventory frequency.

        Parameters
        ----------
        input_frequency : str
            Inventory frequency.

        Returns
        -------
        str
            ``kg/year`` for yearly inventories and ``kg/month`` for monthly inventories.

        Raises
        ------
        ValueError
            If ``input_frequency`` is unsupported by regional rescaling.
        """
        if input_frequency == "yearly":
            return "kg/year"
        if input_frequency == "monthly":
            return "kg/month"
        raise ValueError(
            "ERROR: Regional emission rescaling inventory totals are only supported for yearly and monthly "
            f"emission inventories. Found frequency '{input_frequency}'."
        )

    def apply_factors(self, emissions, factors):
        """
        Apply regional rescaling factors to emissions.

        This is the operational step that modifies the prepared emission fields after the factors have been calculated.
        Two strategies are available:

        - ``rescaling_everything=True`` applies the single configured region factor to the complete inventory field.
        - ``rescaling_everything=False`` applies region-specific factors only where the regional mask is active.

        The regional strategy preserves emissions outside configured regions because the spatial factor field starts at
        ``1`` everywhere. Only cells belonging to a configured region and pollutant with a non-``None`` factor are
        modified.

        Parameters
        ----------
        emissions : Nes
            Prepared emissions on the destination grid.
        factors : dict
            Regional rescaling factors by region and pollutant.

        Returns
        -------
        Nes
            Emissions after applying regional rescaling factors.

        Notes
        -----
        When ``rescaling_everything`` is enabled, each pollutant is multiplied by the factor of the single configured
        region. Otherwise, one spatial factor field is created per pollutant using the regional mask. ``None`` factors
        leave the original emissions unchanged for that pollutant and region.
        """
        log_message("Applying regional emission rescaling factors.", level=2)
        if self.apply_to_all_emissions:
            return self._apply_factors_to_all_emissions(emissions, factors)

        return self._apply_factors_by_region(emissions, factors)

    @staticmethod
    def _apply_factors_to_all_emissions(emissions, factors):
        """
        Apply the single-region rescaling factor to the complete emission inventory.

        This mode is intended for configurations where the local/reference total represents the complete inventory
        domain, or where the user explicitly wants to apply the factor derived for one region to all cells. It is only
        valid when there is exactly one configured region, which is also checked during initialisation.

        Parameters
        ----------
        emissions : Nes
            Prepared emissions on the destination grid.
        factors : dict
            Regional rescaling factors by region and pollutant. It must contain exactly one region.

        Returns
        -------
        Nes
            Emissions after applying the inventory-wide factors.

        Raises
        ------
        ValueError
            If more than one region factor is provided.

        Notes
        -----
        ``None`` factors are skipped, leaving the corresponding pollutant unchanged. Factor ``0`` sets the complete
        pollutant field to zero.
        """
        if len(factors) != 1:
            raise ValueError(
                "ERROR: Applying regional rescaling to all emissions requires exactly one configured region. "
                f"Found {len(factors)} regions."
            )

        region_factors = next(iter(factors.values()))
        for pollutant_name in emissions.variables.keys():
            factor = region_factors.get(pollutant_name)
            if factor is not None:
                emissions.variables[pollutant_name]["data"] = emissions.variables[pollutant_name]["data"] * factor

        return emissions

    def _apply_factors_by_region(self, emissions, factors):
        """
        Apply regional rescaling factors using the regional mask levels.

        For each pollutant, a two-dimensional factor field is built on the local MPI domain slice. It is initialised to
        one, and cells belonging to each configured region are overwritten with the corresponding pollutant factor. The
        field is then broadcast over any non-spatial dimensions of the emission variable, such as time or vertical
        level.

        ``OTHERS`` may be present in the mask, but it is not expected in ``factors``. Therefore non-target land cells,
        sea cells and regions with ``None`` factors remain unchanged.

        Parameters
        ----------
        emissions : Nes
            Prepared emissions on the destination grid.
        factors : dict
            Regional rescaling factors by region and pollutant.

        Returns
        -------
        Nes
            Emissions after applying regional factors on the configured mask levels.

        Raises
        ------
        ValueError
            If an emission variable has fewer than two spatial dimensions.
        ValueError
            If the regional mask spatial shape does not match an emission variable spatial shape.
        """
        region_ids = self._get_region_mask_ids()
        mask_data = np.asarray(self.region_mask.variables[REGION_MASK_VAR]["data"])[0, :, :, :]

        for pollutant_name in emissions.variables.keys():
            data = emissions.variables[pollutant_name]["data"]
            if data.ndim < 2:
                raise ValueError("ERROR: Emission data must have at least two spatial dimensions.")
            if data.shape[-2:] != mask_data.shape[-2:]:
                raise ValueError(
                    "ERROR: Emission data spatial shape does not match the regional rescaling mask. "
                    f"pollutant={pollutant_name}, emission_shape={data.shape[-2:]}, mask_shape={mask_data.shape[-2:]}."
                )

            # Start with factor 1 everywhere, then overwrite configured region cells with their pollutant factor.
            spatial_factors = np.ones(data.shape[-2:], dtype=np.float64)
            for region_index, region_id in enumerate(region_ids):
                region_factors = factors.get(region_id)
                if region_factors is None:
                    continue
                factor = region_factors.get(pollutant_name)
                if factor is None:
                    continue
                spatial_factors[mask_data[region_index, :, :] == 1] = factor

            broadcast_shape = (1,) * (data.ndim - 2) + spatial_factors.shape
            emissions.variables[pollutant_name]["data"] = data * spatial_factors.reshape(broadcast_shape)

        return emissions

    def _calculate_inventory_totals(self, emissions, seconds_per_period=SECONDS_PER_YEAR):
        """
        Calculate regional inventory totals from emissions on the destination grid.

        The input emission variables can have two or more dimensions. The last two dimensions are interpreted as the
        horizontal grid, and all leading dimensions are summed before regional aggregation. This means that hourly,
        daily, vertical-level or other leading dimensions contribute to the inventory total.

        The unit conversion is::

            kg/period = kg.m-2.s-1 * cell_area_m2 * seconds_per_period

        Parameters
        ----------
        emissions : Nes
            Emissions regridded to the destination grid. Variables are expected in ``kg.m-2.s-1``.

        Returns
        -------
        pandas.DataFrame
            Regional inventory totals in ``kg`` for the selected period with regions as rows and pollutants as columns.

        Raises
        ------
        ValueError
            Raised by ``_sum_emission_data_to_2d`` if an emission variable has fewer than two dimensions.
        """
        region_ids = self._get_region_mask_ids()
        mask_data = np.asarray(self.region_mask.variables[REGION_MASK_VAR]["data"])[0, :, :, :]
        cell_area = np.asarray(self.grid.cell_measures["cell_area"]["data"])
        pollutant_names = list(emissions.variables.keys())
        inventory_totals_data = np.zeros((len(region_ids), len(pollutant_names)), dtype=np.float64)

        # Convert kg.m-2.s-1 emissions to kg/period and sum them inside each regional mask level.
        for pollutant_index, pollutant_name in enumerate(pollutant_names):
            pollutant_data = self._sum_emission_data_to_2d(emissions.variables[pollutant_name]["data"])
            pollutant_data = pollutant_data * cell_area * seconds_per_period
            for region_index in range(len(region_ids)):
                inventory_totals_data[region_index, pollutant_index] = np.nansum(
                    mask_data[region_index, :, :] * pollutant_data
                )

        # Each rank owns a domain slice; reduce the partial regional totals into global inventory totals.
        global_totals = np.zeros_like(inventory_totals_data)
        self.comm.Allreduce(inventory_totals_data, global_totals, op=MPI.SUM)

        return pd.DataFrame(global_totals, index=region_ids, columns=pollutant_names)

    @staticmethod
    def _sum_emission_data_to_2d(data):
        """
        Sum emission data over all non-spatial dimensions.

        Parameters
        ----------
        data : numpy.ndarray
            Emission variable data. The last two dimensions must be the spatial dimensions.

        Returns
        -------
        numpy.ndarray
            Two-dimensional emission data.

        Raises
        ------
        ValueError
            If ``data`` has fewer than two dimensions. The last two dimensions are required to be spatial.
        """
        data = np.ma.filled(data, np.nan)
        data = np.asarray(data)
        if data.ndim == 2:
            return data
        if data.ndim < 2:
            raise ValueError("ERROR: Emission data must have at least two spatial dimensions.")

        return np.nansum(data, axis=tuple(range(data.ndim - 2)))

    @staticmethod
    def _calculate_factors_from_totals(inventory_totals, local_totals, inventory_name=None, sector=None):
        """
        Calculate regional rescaling factors from inventory and local totals.

        For each configured region and pollutant, the factor is calculated as::

            factor = local_total / inventory_total

        Both totals must be expressed in ``kg`` for the same period. The function preserves the distinction between
        missing reference data and a valid zero total:

        - missing local totals produce ``None`` and are not applied;
        - local totals equal to zero produce factor ``0`` when the inventory total is non-zero;
        - inventory totals equal to zero produce factor ``0`` because there are no emissions to rescale.

        Parameters
        ----------
        inventory_totals : pandas.DataFrame
            Inventory totals in ``kg`` for the selected period with regions as rows and pollutants as columns.
        local_totals : pandas.DataFrame
            Reference local totals in ``kg`` for the selected period with regions as rows and pollutants as columns.
        inventory_name : str or None
            Emission inventory name, used only to make warnings more explicit.
        sector : str or None
            Sector name, used only to make warnings more explicit.

        Returns
        -------
        dict
            Nested dictionary with ``region_id`` -> ``pollutant`` -> factor.

        Checks
        ------
        - ``OTHERS`` is removed from ``local_totals`` before factor calculation.
        - Regions are aligned using ``local_totals.index``.
        - Pollutants present in the inventory but missing from ``local_totals`` receive ``None``.
        - Missing local totals receive ``None``.
        - Zero local totals naturally produce factor ``0`` when the inventory total is non-zero.
        - Zero or missing inventory totals produce factor ``0``.

        Warnings
        --------
        Emits ``log_message(..., level=7)`` when the local total is positive but the inventory total is zero or ``NaN``.
        The warning means that rescaling was requested but cannot add emissions where the original inventory has none.
        """
        factors = {}

        # Only configured regions can receive factors. OTHERS is intentionally discarded here.
        local_totals = local_totals.drop(index=OTHERS_REGION_ID, errors="ignore")
        inventory_totals = inventory_totals.reindex(index=local_totals.index)

        # Evaluate each region/pollutant pair independently to preserve missing-value semantics.
        for region_id in local_totals.index:
            factors[region_id] = {}
            for pollutant_name in inventory_totals.columns:
                inventory_total = inventory_totals.loc[region_id, pollutant_name]
                if pollutant_name not in local_totals.columns:
                    factors[region_id][pollutant_name] = None
                    continue
                local_total = local_totals.loc[region_id, pollutant_name]
                if pd.isna(local_total):
                    factors[region_id][pollutant_name] = None
                elif pd.isna(inventory_total) or inventory_total == 0:
                    if not pd.isna(local_total) and local_total != 0:
                        # A positive local total cannot be introduced if the source inventory has no emissions.
                        inventory_label = inventory_name if inventory_name is not None else "unknown"
                        sector_label = sector if sector is not None else "unknown"
                        log_message(
                            "WARNING: Regional rescaling cannot be applied because the original inventory total is 0. "
                            f"inventory={inventory_label}, sector={sector_label}, region_id={region_id}, "
                            f"pollutant={pollutant_name}, local_total={local_total} kg. "
                            "Setting rescaling factor to 0.",
                            level=7,
                        )
                    factors[region_id][pollutant_name] = 0
                else:
                    factors[region_id][pollutant_name] = float(local_total / inventory_total)

        return factors

    def _read_geometries(self, geometries_path, geometries_region_id_attribute, region_ids):
        """
        Read the rescaling geometries intersecting the domain and configured regions.

        The input file must be readable by GeoPandas and must contain:

        - the configured region identifier column, e.g. ``NAME_4``;
        - a valid ``geometry`` column.

        Example input GeoDataFrame when ``geometries_region_id_attribute`` is ``NAME_4``::

            NAME_4      geometry
            Barcelona   POLYGON (...)
            A Coruña    POLYGON (...)
            Girona      POLYGON (...)

        The returned GeoDataFrame keeps only ``region_id`` and ``geometry``. Region names not present in
        ``region_ids`` are recoded as ``OTHERS`` and dissolved into a single geometry. This keeps neighbouring land
        regions in the spatial join without adding one mask level per unused region. Cells with no intersecting
        geometry, such as sea cells, are not converted to ``OTHERS`` later and remain zero in all mask levels.

        Example returned GeoDataFrame::

            region_id   geometry
            A Coruña    MULTIPOLYGON (...)
            Barcelona   MULTIPOLYGON (...)
            OTHERS      MULTIPOLYGON (...)

        Parameters
        ----------
        geometries_path : str
            Path to the shapefile or GeoJSON containing the rescaling regions.
        geometries_region_id_attribute : str
            Geometry attribute containing the region identifier.
        region_ids : list[str]
            Region identifiers found in the local total emissions file.

        Returns
        -------
        geopandas.GeoDataFrame
            Geometries clipped by the domain bounding box. Regions not configured in the local totals file are merged
            into a common ``OTHERS`` region.

        Raises
        ------
        ValueError
            If ``geometries_region_id_attribute`` is not present in the geometry file.
        """
        domain_bbox = self.grid.get_full_bbox()

        # Read only geometries intersecting the destination grid bbox; this avoids loading a full country/world layer.
        geometries = gpd.read_file(geometries_path, bbox=domain_bbox)
        if geometries_region_id_attribute not in geometries.columns:
            raise ValueError(f"ERROR: Column '{geometries_region_id_attribute}' is not in {geometries_path} file.")

        # Keep the configured region column plus geometry, and normalize the column name used internally.
        geometries = geometries.loc[:, [geometries_region_id_attribute, "geometry"]]
        geometries = geometries.rename(columns={geometries_region_id_attribute: REGION_ID_COLUMN})
        geometries[REGION_ID_COLUMN] = geometries[REGION_ID_COLUMN].astype(str)

        # Merge all non-target land regions into OTHERS so neighbouring land is not mistaken for sea.
        geometries.loc[~geometries[REGION_ID_COLUMN].isin(region_ids), REGION_ID_COLUMN] = OTHERS_REGION_ID
        geometries = geometries.dissolve(by=REGION_ID_COLUMN).reset_index()
        geometries = geometries.reset_index(drop=True)

        # TODO: Explore writing these filtered geometries to an auxiliary file and passing that path to NES
        # spatial_join. NES can then read only the local rank bbox, but the auxiliary invalidation rules need to be
        # defined first when the configured regions or geometry inputs change.
        return geometries

    def _obtain_region_mask(self, auxiliary_files_path, geometries_path, geometries_region_id_attribute, region_ids):
        """
        Read or create the auxiliary regional mask.

        The mask is stored in ``ReScaling_RegionsMask.nc`` under ``auxiliary_files_path``. If the file already exists,
        it is opened with NES and loaded. If it does not exist, it is created from the configured geometries and written
        as an auxiliary file.

        Checks performed:

        - if ``ReScaling_RegionsMask.nc`` exists, it is loaded with NES;
        - if it does not exist, geometries are read and a new mask is created;
        - existing masks are checked against the current grid and required rescaling metadata before reuse.

        Parameters
        ----------
        auxiliary_files_path : str
            Path to the auxiliary files directory.
        geometries_path : str
            Path to the shapefile or GeoJSON containing the rescaling regions.
        geometries_region_id_attribute : str
            Geometry attribute containing the region identifier.
        region_ids : list[str]
            Region identifiers found in the local total emissions file.

        Returns
        -------
        Nes
            Regional mask over the destination grid. It contains the variable described in ``_create_region_mask``.
        """
        region_mask_path = self._get_region_mask_path(auxiliary_files_path)

        # Reuse an existing auxiliary mask only when it is consistent with the current grid and metadata.
        if os.path.exists(region_mask_path):
            log_message(f"Reading existing regional rescaling mask from {region_mask_path}", level=2)
            region_mask = open_netcdf(
                path=region_mask_path,
                comm=self.comm,
                parallel_method=self.grid.parallel_method,
                balanced=self.grid.balanced,
            )
            region_mask.load()
            if not check_aux_file(
                region_mask,
                nes_object_to_compare=self.grid,
                var_list=[REGION_MASK_VAR],
                variable_attributes={
                    REGION_MASK_VAR: [
                        "region_ids",
                        "region_ids_separator",
                        "region_id_column",
                        "long_name",
                    ],
                },
            ):
                raise ValueError(
                    f"Existing regional rescaling mask auxiliary file is not consistent with the current grid or "
                    f"rescaling metadata: {region_mask_path}. Expected variable '{REGION_MASK_VAR}' with region "
                    "metadata attributes. Remove this auxiliary file or regenerate the auxiliary directory."
                )
        else:
            # Create the auxiliary mask once and write it for the following executions.
            log_message("Creating regional rescaling mask.", level=2)
            region_mask = self._create_region_mask(
                auxiliary_files_path,
                geometries_path,
                geometries_region_id_attribute,
                region_ids,
            )
            region_mask.to_netcdf(region_mask_path, serial=True)
            region_mask.comm.Barrier()
            log_message(f"Regional rescaling mask written at {region_mask_path}", level=3)

        return region_mask

    def _create_region_mask(self, auxiliary_files_path, geometries_path, geometries_region_id_attribute, region_ids):
        """
        Create the regional mask over the destination grid.

        The output NES object contains one variable named ``rescaling_mask`` with shape::

            time, lev, y, x

        where ``time`` has one time step and ``lev`` has one level per selected region plus ``OTHERS``. The data are
        integer 0/1 values. The mapping between vertical levels and original region names is stored in the variable
        attributes:

        - ``region_ids``: ``;``-separated list of region names, e.g. ``A Coruña;Barcelona;OTHERS``.
        - ``region_ids_separator``: separator used in ``region_ids``.
        - ``region_id_column``: internal region column name, currently ``region_id``.

        This design allows region names with spaces or special characters because they are stored as attribute content,
        not as NetCDF variable names.

        Checks performed:

        - input geometries are filtered to the model domain bounding box;
        - geometry regions not present in ``region_ids`` are merged into ``OTHERS``;
        - cells without intersecting land geometry remain zero in every mask level;
        - mask values are written as ``np.int8`` 0/1 values.

        The grid shapefile after the spatial join can be written as GeoJSON for deep spatial diagnostics by re-enabling
        the commented block below the ``spatial_join`` call::

            ReScaling_RegionsMask_spatial_join.geojson

        Parameters
        ----------
        auxiliary_files_path : str
            Path to the auxiliary files directory.
        geometries_path : str
            Path to the shapefile or GeoJSON containing the rescaling regions.
        geometries_region_id_attribute : str
            Geometry attribute containing the region identifier.
        region_ids : list[str]
            Region identifiers found in the local total emissions file.

        Returns
        -------
        Nes
            Regional mask over the destination grid.
        """
        geometries = self._read_geometries(geometries_path, geometries_region_id_attribute, region_ids)

        # Start from a copy of the destination grid so coordinates, projection and MPI decomposition are preserved.
        region_mask = self.grid.copy(copy_vars=False)
        region_mask.variables = {}
        region_mask.cell_measures = {}

        # Build a grid-cell shapefile and spatially assign each cell to one region.
        region_mask.create_shapefile()
        region_mask.spatial_join(geometries, method="intersection", var_list=[REGION_ID_COLUMN], info=True)
        # Re-enable this block if the regional mask spatial join needs to be inspected manually.
        # geojson_path = self._get_region_mask_geojson_path(auxiliary_files_path)
        # region_mask.write_shapefile(geojson_path)
        # log_message(f"Regional rescaling spatial join GeoJSON written at {geojson_path}", level=5)

        region_mask_ids = region_ids.copy()
        if OTHERS_REGION_ID not in region_mask_ids:
            region_mask_ids.append(OTHERS_REGION_ID)

        # Convert the joined region labels into one 0/1 vertical level per region.
        region_data = region_mask.shapefile[REGION_ID_COLUMN].values.reshape(
            (region_mask.lat["data"].shape[0], region_mask.lon["data"].shape[-1])
        )
        mask_data = np.zeros(
            (1, len(region_mask_ids), region_mask.lat["data"].shape[0], region_mask.lon["data"].shape[-1]),
            dtype=np.int8,
        )
        for region_level, region_id in enumerate(region_mask_ids):
            mask_data[0, region_level, :, :] = region_data == region_id

        if region_mask.time is not None:
            region_mask.set_time([region_mask.time[0]])
        region_mask.set_levels({"data": np.arange(len(region_mask_ids), dtype=np.int32), "units": "region"})

        # Store original region names in attributes, which supports names that are invalid NetCDF variable names.
        region_mask.variables[REGION_MASK_VAR] = {
            "data": mask_data,
            "dtype": np.int8,
            "region_ids": REGION_IDS_SEPARATOR.join(region_mask_ids),
            "region_ids_separator": REGION_IDS_SEPARATOR,
            "region_id_column": REGION_ID_COLUMN,
            "long_name": "Regional emission rescaling mask",
        }

        return region_mask
