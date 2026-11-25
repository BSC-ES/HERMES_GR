#!/usr/bin/env python

# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

"""Pre-process annual GRETA Germany emissions on their native LAEA grid.

The workflow writes one HERMES_GR NetCDF file per pollutant and GNFR sector.
It uses CAMS-REG DEU totals to distribute aggregate GNFR F across F1--F4 and
applies the available DEU 2022 NMVOC and PM2.5 factors by GNFR sector.
The related GRETA GHG workflow writes CO2 and CH4 and distributes aggregate
GNFR F using CAMS-REG DEU totals.
"""

import os
from datetime import datetime

import nes
import numpy as np
import pandas as pd

ORIGINAL_DIR = os.path.join(os.path.dirname(__file__), "greta_preproc_original_files")
ORIGINAL_FILES = {
    2022: "2022_Sub2024_Condensables_Inland_GNFR_Greta1.6.0.1.nc",
    2030: "2030_Sub2024_WM_20250306_Condensables_Inland_GNFR_Greta1.6.0.1.nc",
}
GHG_ORIGINAL_FILE = os.path.join(
    ORIGINAL_DIR, "2022_Sub2026_GHG_GNFR_Greta1.7.1.1.nc"
)
# Use the inventory root to match the shared archive directory convention.
OUTPUT_PATH = "/esarchive/recon/uba_germany/gretav1.6.0.1"
# OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "greta_preproc_output_files")
# Keep input tables alongside the GRETA files for a portable workflow.
_INPUT_DIR = ORIGINAL_DIR
CAMS_REG_PATH = os.path.join(_INPUT_DIR, "CAMS-REG-v8_1_emissions_year2022.csv")
NMVOC_SPLIT_PATH = os.path.join(
    _INPUT_DIR, "NMVOC_split_for_CAMS-REG-v81_KO_melt_20251020_v2.csv"
)
PM_SPLIT_PATH = os.path.join(_INPUT_DIR, "PM_split_for_CAMS-REG-v81.csv")

YEAR_FOR_SPLITS = 2022
GNFR_F_SUBSECTORS = ("F1", "F2", "F3", "F4")
IGNORED_SECTORS = {"M", "SUM"}
# Both supported inventory years are non-leap years.
SECONDS_PER_YEAR = 365.0 * 24 * 60 * 60
AIR_POLLUTANT_REFERENCE = (
    "Feigenspan, S. (2026). Gridded emissions of air pollutants for Germany "
    "[Dataset]. Zenodo. https://doi.org/10.5281/zenodo.23467497"
)


def _read_csv(path, sep):
    """Read a delimited text file after checking that it exists.

    Parameters
    ----------
    path : str or os.PathLike
        Path to the CSV or semicolon-separated input file.
    sep : str
        Fixed delimiter: ``;`` for CAMS-REG and ``,`` for split tables.

    Returns
    -------
    pandas.DataFrame
        Parsed table using the explicitly specified delimiter.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not point to a file.
    """
    # Delimiters are fixed by the known input formats, not inferred.
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep=sep)


def _column(df, name):
    """Find a DataFrame column by case-insensitive, whitespace-trimmed name.

    Parameters
    ----------
    df : pandas.DataFrame
        Table whose columns are searched.
    name : str
        Required column name.

    Returns
    -------
    str
        The original column label matching ``name``.

    Raises
    ------
    ValueError
        If no matching column is present.
    """
    lookup = {str(col).strip().lower(): col for col in df.columns}
    if name.lower() not in lookup:
        raise ValueError(f"Required column {name!r} missing; found {list(df.columns)!r}")
    return lookup[name.lower()]


def get_gnfr_f_fractions(cams_reg_path, pollutant, table=None):
    """Calculate the DEU CAMS-REG shares for aggregate GNFR F.

    The pollutant totals are summed over all grid cells in categories F1--F4
    for the configured reference year, then normalized to sum to one.

    Parameters
    ----------
    cams_reg_path : str or os.PathLike
        Path to the CAMS-REG inventory CSV.
    pollutant : str
        CAMS-REG pollutant column used to estimate the subsector shares.
        ``CO2`` is calculated by adding ``CO2_ff`` and ``CO2_bf`` before
        summing over grid cells.
    table : pandas.DataFrame, optional
        Previously loaded CAMS-REG table, avoiding repeated file reads.

    Returns
    -------
    dict[str, float]
        Fractions keyed by ``F1``, ``F2``, ``F3``, and ``F4``.

    Raises
    ------
    ValueError
        If the requested pollutant has no positive DEU total across F1--F4.
    """
    # Reuse the large CAMS table across pollutants and inventory years.
    df = _read_csv(cams_reg_path, sep=";") if table is None else table
    iso_col = _column(df, "ISO3")
    sector_col = _column(df, "GNFR_Category")
    year_col = _column(df, "Year")
    # These national totals provide one uniform split ratio per pollutant.
    # Keep only Germany, the reference year, and the four detailed F
    # categories.
    selected = df[(df[iso_col].astype(str).str.upper() == "DEU") &
                  (pd.to_numeric(df[year_col], errors="coerce") == YEAR_FOR_SPLITS) &
                  (df[sector_col].astype(str).str.upper().isin(GNFR_F_SUBSECTORS))]
    # CAMS stores fossil and biofuel CO2 separately; combine them per row
    # before summing grid cells so the F-subsector shares reflect total CO2.
    if pollutant.upper() == "CO2":
        co2_ff_col = _column(selected, "CO2_ff")
        co2_bf_col = _column(selected, "CO2_bf")
        values = (
            pd.to_numeric(selected[co2_ff_col], errors="coerce").fillna(0)
            + pd.to_numeric(selected[co2_bf_col], errors="coerce").fillna(0)
        )
    else:
        pollutant_col = _column(selected, pollutant)
        values = pd.to_numeric(selected[pollutant_col], errors="coerce").fillna(0)
    totals = values.groupby(selected[sector_col].astype(str).str.upper()).sum()
    totals = totals.reindex(GNFR_F_SUBSECTORS, fill_value=0).astype(float)
    total = totals.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"No positive CAMS-REG DEU split total for {pollutant}")
    # Normalize the four totals so their shares add up to one.
    return (totals / total).to_dict()


def _split_factor(path, species_column, species_code, sector, table=None):
    """Look up one country-, year-, and sector-specific species fraction.

    Parameters
    ----------
    path : str or os.PathLike
        Path to an NMVOC or PM split-factor table.
    species_column : str
        Column containing the split species code, such as ``vcode`` or
        ``pmcode``.
    species_code : str
        Species identifier to look up.
    sector : str
        GNFR category to look up.
    table : pandas.DataFrame, optional
        Previously loaded split-factor table.

    Returns
    -------
    float or None
        The unique DEU factor for ``YEAR_FOR_SPLITS``, or ``None`` if no row
        matches the requested species and sector.

    Raises
    ------
    ValueError
        If multiple distinct factors match the lookup.
    """
    # Reuse the split table instead of reopening it for every species.
    df = _read_csv(path, sep=",") if table is None else table
    iso_col, year_col = _column(df, "ISO3"), _column(df, "year")
    sector_col, species_col = _column(df, "gnfr"), _column(df, species_column)
    factor_col = _column(df, "fr")
    # The same 2022 reference factors are used for each GRETA inventory year.
    # The sector and species filters select one number for this specific split.
    selected = df[
        (df[iso_col].astype(str).str.upper() == "DEU") &
        (pd.to_numeric(df[year_col], errors="coerce") == YEAR_FOR_SPLITS) &
        (df[sector_col].astype(str).str.upper() == sector.upper()) &
        (df[species_col].astype(str).str.lower() == species_code.lower())
    ]
    if selected.empty:
        return None
    factors = pd.to_numeric(selected[factor_col], errors="coerce").dropna().unique()
    if len(factors) != 1:
        raise ValueError(
            f"Expected one DEU {YEAR_FOR_SPLITS} factor for {species_code}/{sector}; "
            f"found {factors.tolist()}"
        )
    return float(factors[0])


def _output_pollutant(code):
    """Map a GRETA pollutant code to the HERMES_GR pollutant name.

    Parameters
    ----------
    code : str
        Pollutant suffix from a GRETA variable name.

    Returns
    -------
    str
        HERMES_GR name, with NOX, PM2.5, and SO2 mapped to their conventions.
    """
    name = code.upper()
    if name in {"NOX", "NOX_NO2"}:
        return "nox_no2"
    if name in {"PM2.5", "PM2_5", "PM25"}:
        return "pm25"
    if name in {"SOX", "SO2"}:
        return "sox"
    return name.lower()


def _voc_output_name(code):
    """Convert a VOC split-table code to a zero-padded output name.

    Parameters
    ----------
    code : str
        VOC code, for example ``v01``.

    Returns
    -------
    str
        HERMES_GR VOC name, for example ``voc01``.

    Raises
    ------
    ValueError
        If ``code`` contains no digits.
    """
    code = str(code).strip().lower()
    digits = "".join(char for char in code if char.isdigit())
    if not digits:
        raise ValueError(f"Unrecognized GEIA VOC code: {code!r}")
    return f"voc{int(digits):02d}"


def _write_variable(input_path, output_path, year, source_name, output_name,
                    sector, fraction=1.0, crs="EPSG:3035", inventory=None,
                    inventory_source="GRETA v1.6.0.1",
                    references=AIR_POLLUTANT_REFERENCE):
    """Write one pollutant-sector output on the native GRETA grid.

    GRETA values are treated as annual kt per cell. The selected field is
    multiplied by ``fraction`` and converted to an annual mass flux in
    kg m-2 s-1. NES receives the source CRS and coordinate information because
    the input files do not carry those metadata themselves.

    Parameters
    ----------
    input_path : str or os.PathLike
        Source GRETA NetCDF file.
    output_path : str or os.PathLike
        Destination HERMES_GR NetCDF file. Existing files are left unchanged.
    year : int
        Inventory year assigned to the annual time interval.
    source_name : str
        GRETA variable to load, such as ``E_GNFR_A_CO``.
    output_name : str
        HERMES_GR variable name to write.
    sector : str
        GNFR sector represented in the output.
    fraction : float, optional
        Scalar factor applied to the source field, by default 1.0.
    crs : str, optional
        Source CRS identifier, by default ``EPSG:3035``.
    inventory : nes.Nes, optional
        Open source inventory whose grid and cell areas can be reused.
        Its variables are restored after writing; the caller closes it.
    inventory_source : str, optional
        Source inventory name and version stored in the output metadata.
    references : str or None, optional
        Citation stored in the output metadata. Use ``None`` when no verified
        citation is available for the input product.

    Returns
    -------
    None
        The output is written to disk. If it already exists, the function
        returns without writing.
    """
    # A previous run may already have created this output; leave it untouched.
    if os.path.exists(output_path):
        return
    # A shared NES object retains grid geometry and areas between outputs.
    # Only close objects opened here; the yearly loop owns shared objects.
    owns_inventory = inventory is None
    if owns_inventory:
        inventory = nes.open_netcdf(
            input_path,
            # GRETA coordinate variables lack CRS and unit metadata in the
            # source.
            crs=crs,
            coordinate_names={"x": "LON", "y": "LAT"},
            coordinate_units={"x": "m", "y": "m"},
            time_exception=True,
            parallel_method="Y",
        )
    # Writing temporarily replaces variables; retain the source lookup.
    source_variables = inventory.variables
    try:
        if source_name not in inventory.variables:
            raise KeyError(f"GRETA variable {source_name!r} not found in {input_path}")
        inventory.load(source_name)
        data = np.asarray(inventory.variables[source_name]["data"], dtype=np.float64)
        # Cell area is needed because GRETA stores mass per cell, while HERMES
        # expects mass per area and per second.
        area = inventory.cell_measures.get("cell_area", {}).get("data")
        if area is None:
            inventory.calculate_grid_area()
            area = inventory.cell_measures["cell_area"]["data"]
        area = np.asarray(area, dtype=np.float64)
        # Convert annual kt/cell to annual mean kg m-2 s-1 after applying the
        # subsector or chemical-speciation fraction.
        flux = data * float(fraction) * 1.0e6 / area / SECONDS_PER_YEAR

        # Keep the source variable's metadata, then replace its name,
        # description, units, and values for this output.
        metadata = dict(inventory.variables[source_name])
        metadata.update({
            "data": flux,
            "units": "kg.m-2.s-1",
            "description": (
                f"GRETA annual {output_name} emissions, GNFR {sector}, "
                f"Germany, {year}; split factor {fraction:.12g}"
            ),
            "short_name": output_name,
        })
        # Write one pollutant variable per file, matching the other
        # inventories.
        inventory.variables = {output_name: metadata}
        start, end = datetime(year, 1, 1), datetime(year + 1, 1, 1)
        # The source files have no time coordinate; describe them as annual
        # totals.
        inventory.set_time([start])
        inventory.set_time_bnds([[start, end]])
        inventory.global_attrs["Conventions"] = "CF-1.7"
        inventory.global_attrs["title"] = f"GRETA Germany emissions, GNFR {sector}"
        inventory.global_attrs["source"] = inventory_source
        if references is not None:
            inventory.global_attrs["references"] = references
        inventory.global_attrs["source_file"] = os.path.basename(input_path)
        inventory.global_attrs["comment"] = (
            "Native ETRS89 / LAEA Europe grid (EPSG:3035); annual kt/cell "
            "converted to kg m-2 s-1."
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Keep outputs uncompressed as requested for the archive workflow.
        inventory.to_netcdf(output_path, compression_level=0)
    finally:
        # Restore source metadata for the next pollutant and release its data
        # so memory use does not grow with the number of generated files.
        inventory.variables = source_variables
        if source_name in source_variables:
            source_variables[source_name]["data"] = None
        if owns_inventory:
            inventory.close()


def greta_preproc(original_dir=ORIGINAL_DIR, output_path=OUTPUT_PATH,
                  years=(2022, ), cams_reg_path=CAMS_REG_PATH,
                  nmvoc_split_path=NMVOC_SPLIT_PATH, pm_split_path=PM_SPLIT_PATH,
                  overwrite=False):
    """Create HERMES_GR annual files from the selected GRETA inventory years.

    Output files use the layout
    ``<output_path>/yearly_mean/<pollutant>_gnfr_<sector>/``. Aggregate
    GNFR F is retained and also distributed to F1--F4. NMVOC and PM2.5
    speciation is applied to every sector with available split factors,
    excluding aggregate F. The factor reference year is 2022 for every
    inventory year.

    Parameters
    ----------
    original_dir : str or os.PathLike, optional
        Directory containing the GRETA NetCDF input files, or the full path to
        one GRETA NetCDF file. When a file is supplied, ``years`` must contain
        only the year represented by that filename.
    output_path : str or os.PathLike, optional
        Inventory root directory. Years share pollutant-sector directories
        and are identified by the output filenames.
    years : iterable of int, optional
        Inventory years to process. Supported values are 2022 and 2030.
    cams_reg_path : str or os.PathLike, optional
        CAMS-REG inventory CSV used to calculate F1--F4 fractions.
    nmvoc_split_path : str or os.PathLike, optional
        NMVOC speciation-factor CSV.
    pm_split_path : str or os.PathLike, optional
        PM2.5 component-factor CSV.
    overwrite : bool, optional
        If True, remove and recreate existing output files. If False, existing
        outputs are skipped.

    Returns
    -------
    bool
        True when processing completes.

    Raises
    ------
    FileNotFoundError
        If a required input file is missing.
    ValueError
        If a year, pollutant column, or split factor is invalid.
    """
    # Read the reference tables once before processing emissions. Their factors
    # control how each source field is divided among output sectors or species.
    cams_df = _read_csv(cams_reg_path, sep=";")
    _column(cams_df, "ISO3")
    _column(cams_df, "GNFR_Category")
    pollutant_columns = {str(col).strip().upper(): col for col in cams_df.columns}
    nmvoc_df, pm_df = _read_csv(nmvoc_split_path, sep=","), _read_csv(pm_split_path, sep=",")
    # Filter once to DEU 2022 so repeated lookups scan only relevant rows.
    # These reference factors are also required for the 2030 projection.
    for table in (cams_df, nmvoc_df, pm_df):
        iso_col = _column(table, "ISO3")
        year_col = _column(table, "year")
        selected = ((table[iso_col].astype(str).str.upper() == "DEU") &
                    (pd.to_numeric(table[year_col], errors="coerce") == YEAR_FOR_SPLITS))
        table.drop(index=table.index[~selected], inplace=True)
    voc_col = _column(nmvoc_df, "vcode")
    pm_col = _column(pm_df, "pmcode")

    years = tuple(years)

    # A direct input filename identifies its own year; it cannot provide the
    # source file for a second requested year.
    input_is_file = os.path.isfile(original_dir)
    if input_is_file:
        input_basename = os.path.basename(original_dir)
        input_years = [year for year, filename in ORIGINAL_FILES.items()
                       if filename == input_basename]
        if not input_years:
            raise ValueError(
                f"Input file {input_basename!r} is not one of the supported GRETA files."
            )
        if years != (input_years[0],):
            raise ValueError(
                f"The input file is for {input_years[0]}; request only that year."
            )

    # Both years share sector directories; the year belongs in the filename.
    # Process only the years requested by the caller, using the matching file.
    for year in years:
        filename = ORIGINAL_FILES.get(year)
        if filename is None:
            raise ValueError(f"Unsupported GRETA year {year}; expected one of {sorted(ORIGINAL_FILES)}")
        input_path = original_dir if input_is_file else os.path.join(original_dir, filename)
        if not os.path.isfile(input_path):
            raise FileNotFoundError(input_path)
        # Keep one NES object open per year to reuse geometry and cell areas.
        inventory = nes.open_netcdf(
            input_path,
            crs="EPSG:3035",
            coordinate_names={"x": "LON", "y": "LAT"},
            coordinate_units={"x": "m", "y": "m"},
            time_exception=True,
            parallel_method="Y",
        )
        try:
            source_variables = list(inventory.variables)
            for source_name in source_variables:
                # GRETA also contains E_SUM_* totals. Only E_GNFR_* fields are
                # needed.
                if not source_name.startswith("E_GNFR_"):
                    continue
                bits = source_name.split("_")
                if len(bits) < 4:
                    continue
                # For example, E_GNFR_F_NOX becomes sector F and pollutant NOX.
                sector, pollutant_code = bits[2].upper(), "_".join(bits[3:]).upper()
                # M and SUM are aggregate/unsupported sectors requested for
                # exclusion.
                if sector in IGNORED_SECTORS:
                    continue
                if (sector not in {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"}
                        and sector not in GNFR_F_SUBSECTORS):
                    continue
                pollutant = _output_pollutant(pollutant_code)
                print(f"Processing GRETA {year} {source_name} -> {pollutant} GNFR {sector}")
                # Most sectors keep their source sector. Aggregate F
                # additionally
                # gets four outputs, each containing its estimated share of F.
                sector_names = [(sector, 1.0)]
                # CAMS reports sulfur oxides as SOX; GRETA uses SO2.
                cams_code = {
                    "nox_no2": "NOX", "pm25": "PM2_5", "sox": "SOX",
                }.get(pollutant, pollutant_code)
                if sector == "F":
                    if cams_code not in pollutant_columns:
                        raise ValueError(f"No CAMS-REG pollutant column for GRETA {pollutant_code}")
                    # Pass the preloaded reference table to avoid repeated I/O.
                    fractions = get_gnfr_f_fractions(cams_reg_path, pollutant_columns[cams_code], table=cams_df)
                    sector_names.extend((subsector, fraction) for subsector, fraction in fractions.items())

                for out_sector, fraction in sector_names:
                    # Start with the unspeciated pollutant. Add component files
                    # below only when the input and split table support them.
                    out_pollutants = [(pollutant, fraction)]
                    # The issue requires speciation for all sectors with
                    # factors,
                    # not just F1--F4. Aggregate F has no factors and is
                    # excluded.
                    if pollutant == "nmvoc" and out_sector != "F":
                        for code in sorted(nmvoc_df[voc_col].dropna().astype(str).unique()):
                            factor = _split_factor(nmvoc_split_path, "vcode", code, out_sector, table=nmvoc_df)
                            if factor is not None:
                                out_pollutants.append((_voc_output_name(code), fraction * factor))
                    if pollutant == "pm25" and out_sector != "F":
                        for code in sorted(pm_df[pm_col].dropna().astype(str).unique()):
                            factor = _split_factor(pm_split_path, "pmcode", code, out_sector, table=pm_df)
                            if factor is not None:
                                out_pollutants.append((code.lower(), fraction * factor))

                    for out_name, out_fraction in out_pollutants:
                        # Keep related outputs together in one directory per
                        # pollutant and sector, with the year in the filename.
                        # No year directory: follow the archive inventory
                        # layout.
                        path = os.path.join(
                            output_path, "yearly_mean", f"{out_name}_gnfr_{out_sector.upper()}",
                            f"{out_name}_{year}.nc",
                        )
                        if overwrite and os.path.exists(path):
                            # Explicit overwrite mode replaces an existing
                            # result.
                            os.remove(path)
                        _write_variable(input_path, path, year, source_name, out_name,
                                        out_sector, out_fraction, inventory=inventory)
        finally:
            # Close the shared input even if one of its output writes fails.
            inventory.close()
    return True


def greta_ghg_preproc(
    original_file=GHG_ORIGINAL_FILE,
    output_path=OUTPUT_PATH,
    year=2022,
    cams_reg_path=CAMS_REG_PATH,
    overwrite=False,
):
    """Pre-process the GRETA 1.7.1.1 greenhouse-gas inventory.

    Only CO2 and CH4 are written; N2O and aggregate ``E_SUM_*`` fields are
    ignored. Aggregate GNFR F is retained and additionally split across F1--F4
    using DEU CAMS-REG 2022 totals. CO2 shares use the combined ``CO2_ff`` and
    ``CO2_bf`` columns; CH4 shares use the ``CH4`` column.

    Parameters
    ----------
    original_file : str or os.PathLike, optional
        Path to ``2022_Sub2026_GHG_GNFR_Greta1.7.1.1.nc``.
    output_path : str or os.PathLike, optional
        GRETA inventory output root. Files are placed under
        ``yearly_mean/<pollutant>_gnfr_<sector>/``.
    year : int, optional
        Inventory year. The available GHG file represents 2022.
    cams_reg_path : str or os.PathLike, optional
        CAMS-REG inventory CSV used to calculate the F1--F4 shares.
    overwrite : bool, optional
        If True, remove and recreate existing outputs. Otherwise, existing
        files are skipped.

    Returns
    -------
    bool
        True when processing completes.

    Raises
    ------
    FileNotFoundError
        If the GHG inventory or CAMS-REG table is missing.
    ValueError
        If the requested year is not 2022 or an F-split total is invalid.
    """
    if year != 2022:
        raise ValueError("The configured GRETA GHG inventory is available for 2022 only.")
    if not os.path.isfile(original_file):
        raise FileNotFoundError(original_file)

    # Load the CAMS reference table once; its rows are filtered by the share
    # helper to DEU, 2022, and the requested pollutant's F1--F4 categories.
    cams_df = _read_csv(cams_reg_path, sep=";")

    # As for the air-pollutant file, provide the missing CRS and coordinate
    # metadata explicitly and keep the source open while outputs are written.
    inventory = nes.open_netcdf(
        original_file,
        crs="EPSG:3035",
        coordinate_names={"x": "LON", "y": "LAT"},
        coordinate_units={"x": "m", "y": "m"},
        time_exception=True,
        parallel_method="Y",
    )
    try:
        source_variables = list(inventory.variables)
        for source_name in source_variables:
            if not source_name.startswith("E_GNFR_"):
                continue
            bits = source_name.split("_")
            if len(bits) < 4:
                continue
            sector = bits[2].upper()
            pollutant_code = "_".join(bits[3:]).upper()

            # This inventory contains N2O as well, but only CO2 and CH4 are
            # requested. Also omit unsupported and aggregate sectors.
            if pollutant_code not in {"CO2", "CH4"}:
                continue
            if sector in IGNORED_SECTORS:
                continue
            if (sector not in {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"}
                    and sector not in GNFR_F_SUBSECTORS):
                continue

            pollutant = _output_pollutant(pollutant_code)
            sector_names = [(sector, 1.0)]
            if sector == "F":
                # The helper combines CO2_ff + CO2_bf for CO2, and reads CH4
                # directly for methane.
                fractions = get_gnfr_f_fractions(
                    cams_reg_path, pollutant_code, table=cams_df
                )
                sector_names.extend(fractions.items())

            for output_sector, fraction in sector_names:
                output_file = os.path.join(
                    output_path,
                    "yearly_mean",
                    f"{pollutant}_gnfr_{output_sector.upper()}",
                    f"{pollutant}_{year}.nc",
                )
                if overwrite and os.path.exists(output_file):
                    os.remove(output_file)
                _write_variable(
                    original_file,
                    output_file,
                    year,
                    source_name,
                    pollutant,
                    output_sector,
                    fraction,
                    inventory=inventory,
                    inventory_source="GRETA v1.7.1.1",
                    references=None,
                )
    finally:
        inventory.close()
    return True


if __name__ == "__main__":
    greta_preproc()
