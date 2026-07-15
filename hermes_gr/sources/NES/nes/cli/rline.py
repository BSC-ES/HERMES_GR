# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

from typing import List, Optional, Tuple

from geopandas import GeoDataFrame, GeoSeries, read_file
from mpi4py.MPI import COMM_WORLD

# set_option('display.max_columns', None)
from numpy import float32 as precision
from pandas import DataFrame, Index, MultiIndex, Series, set_option
from shapely.geometry import box

from nes import Nes
from nes.load_nes import open_netcdf

from .cli_logger import get_cli_logger

logger = get_cli_logger("nc2rline")
rank = COMM_WORLD.Get_rank()


def load_netcdf_emissions(input_file: str, var_list: Optional[List[str]]) -> Nes:
    """
    Prepare and validate input data by opening the NetCDF file in lazy mode.

    Parameters
    ----------
    input_file : str
        Path to the NetCDF file to transform.
    var_list : list of str or None
        List of variables to include from NetCDF.

    Returns
    -------
    Nes
        Loaded NetCDF data object opened in lazy mode.
    """
    # Open the NetCDF (metadata) in lazy mode
    if rank == 0:
        logger.info(f"Loading NetCDF from {input_file}...")
    nessy = open_netcdf(input_file)
    # Traffic emissions always happen on the first vertical layer
    nessy.sel(lev_min=0, lev_max=0)

    if var_list is not None:
        nessy.keep_vars(var_list)
    error_list = []
    for var_name in nessy.variables.keys():
        if nessy.variables[var_name]['units'] not in ['kg/s', 'Kg/s', 'kg.s-1', 'Kg.s-1']:
            error_list.append(var_name)
    if len(error_list) > 0:
        raise ValueError(f"Units os {error_list} variables are not kg/s")

    # Loading data
    nessy.load()

    return nessy


def prepare_geometry(
    geometry_path: str,
    index_column: Optional[str],
    weight_column: Optional[str],
    bbox: Optional[Tuple[float, float, float, float]] = None,
    geometry_file_path: Optional[str] = None,
    crs: Optional[str] = None,
) -> GeoDataFrame:
    """
    Load geometry data and optionally filter by bounding box and attach weight information.

    Parameters
    ----------
    geometry_path : str
        Path to the geometry file with line features.
    index_column : str or None
        Name of the index column if provided. If not given, feature order will be used as the index.
    weight_column : str or None
        Name of the weight column if provided. If not provided, a default weight of 1.0 is assigned to all features.
    bbox : tuple of float or None, optional
        Bounding box in the form (min_lon, min_lat, max_lon, max_lat), expressed in geographic coordinates
        (longitude, latitude).
    geometry_file_path : str or None, optional
        If provided, the processed geometry will be saved to this file as a CSV formatted for RLINE.
    crs : str or None, optional
        If provided, the geometry will be reprojected to this coordinate reference system.

    Returns
    -------
    GeoDataFrame
        Geometry data with index, 'geometry', and 'weight' columns. Filtered and weighted as requested.
    """
    if rank == 0:
        logger.info(f"Loading geometry from {geometry_path}...")

    if index_column is not None:
        # Use fast partial read with mask and CRS-aware bounding box
        # Read header to get CRS
        orig_crs = read_file(geometry_path, rows=0).crs
        # Build bbox as geometry and reproject to match source CRS
        bbox_geom = box(*bbox)
        bbox_geom = GeoDataFrame(geometry=[bbox_geom], crs="EPSG:4326").to_crs(orig_crs)
        mask = [bbox_geom]
        gdf = read_file(geometry_path, mask=mask)

        if index_column not in gdf.columns:
            raise ValueError(f"[ERROR] index_column '{index_column}' not found in geometry file.")
        gdf = gdf.set_index(index_column)
    else:
        # Fallback to full read and CRS-aware bbox filtering
        gdf = read_file(geometry_path)
        if bbox is not None:
            orig_crs = gdf.crs
            bbox_geom = box(*bbox)
            bbox_geom = GeoDataFrame(geometry=[bbox_geom], crs="EPSG:4326").to_crs(orig_crs)
            gdf = gdf[gdf.geometry.intersects(bbox_geom.geometry.values[0])]

    # Set CRS if provided
    if crs is not None:
        gdf = gdf.to_crs(crs)

    # Ensure index is named 'index'
    gdf.index.name = "index"

    # Write geometry file path to CSV for RLINE to read
    if geometry_file_path is not None:
        write_geometry_file(gdf, geometry_file_path)

    # Ensure only index, geometry, and weight columns remain
    # Handle weight column
    if weight_column is not None:
        if weight_column not in gdf.columns:
            raise ValueError(f"[ERROR] weight_column '{weight_column}' not found in geometry file.")
        gdf = gdf.rename(columns={weight_column: "weight"})
    else:
        gdf["weight"] = 1.0

    # Compose the output GeoDataFrame with index, geometry, and weight only
    # If index is set, keep it as index, else use generated_index
    # Always return geometry and weight columns (index as index)
    return gdf[["geometry", "weight"]].set_geometry("geometry")


def read_weights(path: str) -> DataFrame:
    """
    Read a precomputed weight matrix from a CSV file and validate its format.

    The weight file is expected to have:
    - A MultiIndex of (FID_src, FID_dst)
    - A single column named 'fraction'

    Parameters
    ----------
    path : str
        Path to the weight CSV file.

    Returns
    -------
    DataFrame
        DataFrame with MultiIndex (FID_src, FID_dst) and a single column 'fraction'.

    Raises
    ------
    TypeError
        If the index is not a MultiIndex.
    ValueError
        If the index names or columns do not match the expected format.
    """
    from pandas import MultiIndex, read_csv
    logger.info(f"Reading pre-computed weights from {path}...")

    weights = read_csv(path, index_col=[0, 1])

    # Checking format
    if not isinstance(weights.index, MultiIndex):
        raise TypeError("[ERROR] Weights DataFrame must have a MultiIndex (FID_src, FID_dst).")

    if weights.index.names != ["FID_src", "FID_dst"]:
        raise ValueError("[ERROR] MultiIndex must be named ['FID_src', 'FID_dst'].")

    if list(weights.columns) != ["fraction"]:
        raise ValueError("[ERROR] Weights DataFrame must contain a single column named 'fraction'.")

    return weights


def get_emission_weights(gridded_emissions: Nes, geometry: GeoDataFrame, weight_file: Optional[str]) -> DataFrame:
    """
    Retrieve or compute spatial emission weights for mapping gridded emissions to road segments.

    If `weight_file` exists, weights are read from the file. Each MPI rank reads the file and
    filters rows to keep only those `FID_src` entries relevant to its subset of the domain,
    as determined by `gridded_emissions.get_fids()`.

    If `weight_file` does not exist, weights are computed by intersecting `geometry` (road segments)
    with the grid cells from `gridded_emissions`. The computed weights are normalized so that
    all fractions for a given `FID_src` sum to 1. The result can be written to `weight_file`.

    Parameters
    ----------
    gridded_emissions : Nes
        Loaded gridded emissions dataset (NES object) that provides geometry and FID list.
    geometry : GeoDataFrame
        Road segment geometries. Only required if `weight_file` does not exist.
    weight_file : str or None
        Path to the weight CSV file. If present, it is read; if absent, weights are computed
        and optionally written to this path.

    Returns
    -------
    DataFrame
        A DataFrame with a MultiIndex (FID_src, FID_dst) and one column:
        - fraction : float
            Normalized weight for mapping grid cell FID_src to road segment FID_dst.
    """
    import os
    if weight_file is not None and os.path.exists(weight_file):
        weights = read_weights(weight_file)
        logger.info(f"\n{gridded_emissions.get_fids().flatten()}")
        weights = weights.loc[weights.index.get_level_values("FID_src").isin(gridded_emissions.get_fids().flatten())]
        logger.info(f"{weights}")
    else:
        if geometry is None:
            raise ValueError("Geometry must be provided when weight_file does not exist.")
        weights = calculate_weights(gridded_emissions, geometry)
        if weight_file is not None:
            write_weights(weights, weight_file)
    return weights


def calculate_weights(gridded_emissions: Nes, geometry: GeoDataFrame) -> DataFrame:
    """
    Compute spatial weights for redistributing emissions from a grid to road segments.

    For each grid cell (FID_src), the function computes the intersection length with all
    overlapping road segments (FID_dst) and multiplies this length by the road's weight.
    The resulting values are normalized so that all fractions for a given FID_src sum to 1.

    Parameters
    ----------
    gridded_emissions : Nes
        NES object representing the gridded emissions dataset.
    geometry : GeoDataFrame
        Road segment geometries with a 'weight' column in the same CRS as the grid.

    Returns
    -------
    DataFrame
        A DataFrame with MultiIndex (FID_src, FID_dst) and column:
        - fraction : float
            Normalized weight for distributing emissions.
    """
    def compute_intersection_length(row: Series) -> float:
        """
        Compute length of the intersection between a polygon and a line segment.

        Parameters
        ----------
        row : Series
            Must contain 'geometry_src' (polygon) and 'geometry_dst' (line).

        Returns
        -------
        float
            Length of the intersected line segment
        """
        try:
            inter = row["geometry_dst"].intersection(row["geometry_src"])
        except Exception:
            inter = row["geometry_dst"].buffer(0).intersection(row["geometry_src"].buffer(0))
        return inter.length

    import numpy as np
    # Convert emissions to GeoDataFrame and match CRS
    gdf = gridded_emissions.create_shapefile().to_crs(geometry.crs)
    # Assign temporary integer-based IDs to facilitate join
    gdf["FID_src"] = gdf.index
    geometry["FID_dst"] = geometry.index

    # Identify grid cells and road segments that potentially intersect using spatial index
    gdf = gdf.reset_index()
    geometry = geometry.reset_index()
    fid_src, fid_dst = geometry.sindex.query(gdf.geometry, predicate="intersects")

    # Calculate intersected areas and fractions
    intersection_df = DataFrame(columns=["FID_src", "FID_dst"])

    intersection_df["FID_src"] = np.array(gdf.loc[fid_src, "FID_src"], dtype=np.uint32)
    intersection_df["FID_dst"] = np.array(geometry.loc[fid_dst, "FID_dst"], dtype=np.uint32)
    intersection_df["geometry_src"] = gdf.loc[fid_src, "geometry"].values
    intersection_df["geometry_dst"] = geometry.loc[fid_dst, "geometry"].values
    intersection_df["weight"] = geometry.loc[fid_dst, "weight"].values

    # Use the total length to normalize each segment fraction so that the result represents mass per meter,
    # instead of total Kg when applying the distribution.
    intersection_df["tot_length"] = GeoSeries(intersection_df["geometry_dst"].values).length.values

    # Compute the intersection length
    intersection_df["fraction"] = intersection_df.apply(compute_intersection_length, axis=1)
    # Multiply intersection length by the road segment's weight
    intersection_df["fraction"] *= intersection_df["weight"]
    # Normalize fractions so that the sum of fractions for each FID_src equals 1
    intersection_df["fraction"] = (
            intersection_df["fraction"] / intersection_df.groupby("FID_src")["fraction"].transform("sum"))

    # Divide per length to pass from Kg to Kg/m
    intersection_df["fraction"] = intersection_df["fraction"] / intersection_df["tot_length"]

    # Kg to g
    intersection_df["fraction"] *= 1000

    # Drop geometry columns to save memory
    del intersection_df["geometry_dst"], intersection_df["geometry_src"]
    del intersection_df["weight"], intersection_df["tot_length"]

    intersection_df = intersection_df.set_index(["FID_src", "FID_dst"])

    return intersection_df


def distribute_emissions_to_roads_slow(gridded_emissions: Nes, weights: DataFrame) -> DataFrame:
    """
    Apply the emission weights to distribute emissions from the NetCDF data to road segments.

    This function maps gridded emissions to line-based segments using precomputed spatial weights.

    Parameters
    ----------
    gridded_emissions : Nes
        The loaded NetCDF emissions dataset, containing variables and their associated 4D data arrays.
    weights : DataFrame
        DataFrame indexed by MultiIndex (FID_src, FID_dst) with a single column 'fraction'.

    Returns
    -------
    DataFrame
        DataFrame indexed by FID_dst, with columns as MultiIndex (variable, timestamp),
        containing total emissions per road segment.
    """
    import numpy as np

    logger.info(f"Starting to distribute emissions to road links")

    # Extract useful objects
    fids_array = gridded_emissions.get_fids()
    time_index = gridded_emissions.time
    variables = gridded_emissions.variables

    # Prepare result container: indexed by (FID_src, FID_dst)
    result = DataFrame(
        index=weights.index,
        columns=MultiIndex.from_product([variables.keys(), time_index], names=["variable", "timestamp"]),
        dtype=np.float32
    )

    # Main loop: apply weights
    for var_name, var_obj in variables.items():
        data = var_obj["data"]  # Shape: (time, level, y, x)

        for i_time, timestamp in enumerate(time_index):
            for (fid_src, fid_dst), fraction in weights.itertuples():
                # Find position of fid_src in the 2D grid
                match = np.argwhere(fids_array == fid_src)
                if match.size == 0:
                    continue  # FID not in this partition
                y, x = match[0]

                value = data[i_time, 0, y, x] * fraction
                result.loc[(fid_src, fid_dst), (var_name, timestamp)] = np.float32(value)

    # Group by FID_dst and sum all FID_src contributions (transpose-safe)
    result = result.groupby(level="FID_dst").sum().T
    COMM_WORLD.Barrier()
    logger.info(f"{result}")
    return result


def distribute_emissions_to_roads(gridded_emissions: Nes, weights: DataFrame) -> dict[str, DataFrame]:
    """
    Distribute gridded emissions onto road segments using spatial weights.

    This function projects gridded emissions from a NetCDF dataset onto linear road segments
    using precomputed spatial weights. For each emission variable, it aggregates the
    contributions from all grid cells (FID_src) to each road segment (FID_dst) based on
    the fraction defined in the `weights` DataFrame.

    Parameters
    ----------
    gridded_emissions : Nes
        A `Nes` object representing the loaded NetCDF emissions file.
        It must contain:
            - `.variables`: dictionary of variable names to 4D arrays (time, level, y, x)
            - `.time`: list or index of time steps
            - `.get_fids()`: method returning the 2D array of FID values (shape: [ny, nx])
    weights : DataFrame
        A pandas DataFrame with MultiIndex (FID_src, FID_dst) and one column:
            - 'fraction': the fraction of the grid cell emission that should be
              assigned to each road segment.

    Returns
    -------
    dict[str, DataFrame]
        A dictionary where each key is a variable name and each value is a DataFrame:
            - Index: FID_dst (int)
            - Columns: timestamp (datetime)
            - Values: total emissions assigned to each road segment for each time step.

        Each DataFrame can be directly exported to a separate CSV file per variable.
    """
    if rank == 0:
        logger.info("Starting to distribute emissions to road links")

    # Extract basic data
    gridded_fids = gridded_emissions.get_fids().flatten()
    valid_fids = weights.index.get_level_values("FID_src").unique()
    time_index = gridded_emissions.time
    variables = gridded_emissions.variables

    # Initialize result container
    final_result = {}

    # Loop over each variable (pollutant)
    for var_name, var_info in variables.items():

        # Extract 4D array and reshape to (FID_src, time)
        data_aux = var_info["data"][:, 0, :, :]                  # Remove level → (time, y, x)
        data_aux = data_aux.transpose(1, 2, 0)                   # → (y, x, time)
        data_aux = data_aux.reshape(-1, data_aux.shape[-1])     # → (y * x, time)

        # Build DataFrame of shape (FID_src, timestamp)
        data_df = DataFrame(
            data=data_aux,
            index=Index(gridded_fids, name="FID_src"),
            columns=Index(time_index, name="timestamp"),
            dtype=precision
        )

        # Filter to only the relevant FID_src (those in weights)
        data_df = data_df.loc[data_df.index.isin(valid_fids)]

        # Merge weights with emissions by FID_src
        weights_reset = weights.reset_index()  # columns: FID_src, FID_dst, fraction
        merged = weights_reset.merge(data_df, how="left", left_on="FID_src", right_index=True)

        # Apply fraction to each timestamp
        weighted = merged.loc[:, data_df.columns] * merged["fraction"].values[:, None]

        # Aggregate by FID_dst
        result = weighted.groupby(merged["FID_dst"]).sum()
        result.index.name = "FID_dst"
        result.columns.name = "timestamp"

        # Store result per variable
        final_result[var_name] = result.copy().T

    logger.info("Finished distributing emissions to road links")
    return final_result


def mass_balance_check(gridded_emissions: Nes, weights: DataFrame, unmapped_output_path: Optional[str] = None) -> bool:
    """
    Perform a mass-balance check to ensure total emissions are conserved after
    the grid-to-line redistribution.

    Parameters
    ----------
    gridded_emissions : Nes
        Original gridded emissions loaded from the NetCDF file.
        Used as the reference total against which the distributed totals are compared.
    weights : DataFrame
        Weights from Cell ID to road line segments.
    unmapped_output_path : str or None, optional
        If provided, a NetCDF file will be written containing the emissions from grid cells
        that could not be assigned to any road segment.

    Returns
    -------
    bool
        True if the mass-balance difference is within the accepted tolerance,
        False otherwise.
    """
    if rank == 0:
        logger.info("Performing mass balance check (not yet implemented).")
    # TODO: Implement logic for mass balance validation and unmapped cell output
    return True


def write_weights(weights: DataFrame, weight_file: str) -> None:
    """
    Write spatial emission weights to a CSV file.

    The function gathers partial weights from all MPI ranks (if running in parallel),
    merges them, validates the DataFrame format, and writes the final weights.

    Parameters
    ----------
    weights : DataFrame
        ADataFrame with MultiIndex (FID_src, FID_dst) and one column 'fraction'.
        Must not contain duplicate index pairs.
    weight_file : str
        Path to the CSV file where the weights will be saved.

    Returns
    -------
    None
    """
    from pandas import concat

    # Checking format
    if not isinstance(weights.index, MultiIndex):
        raise TypeError("[ERROR] Weights DataFrame must have a MultiIndex (FID_src, FID_dst).")

    if weights.index.names != ["FID_src", "FID_dst"]:
        raise ValueError("[ERROR] MultiIndex must be named ['FID_src', 'FID_dst'].")

    if list(weights.columns) != ["fraction"]:
        raise ValueError("[ERROR] Weights DataFrame must contain a single column named 'fraction'.")

    # Gather all weights from MPI ranks to the root process
    weights_out = COMM_WORLD.gather(weights.reset_index(), root=0)
    if rank == 0:
        weights_out = concat(weights_out)
        weights_out = weights_out.set_index(["FID_src", "FID_dst"]).sort_index()
        if weights.index.duplicated().any():
            raise ValueError("[ERROR] Duplicate indices found in weight DataFrame before saving.")
        weights_out.to_csv(weight_file, index=True)
        logger.info(f"Weights successfully written to '{weight_file}'")
    return None


def write_geometry_file(gdf: GeoDataFrame, geometry_file_path: str) -> None:
    """
    Write the geometry data to a CSV file formatted for RLINE input.

    The output CSV contains:
    - Group: default "G1" for all rows
    - X_b, Y_b, Z_b: coordinates of the beginning of the line (Z_b = 1)
    - X_e, Y_e, Z_e: coordinates of the end of the line (Z_e = 1)
    - Link_ID: unique identifier for the road segment (taken from the index)

    This function drops duplicated Link_ID rows before saving.

    Parameters
    ----------
    gdf : GeoDataFrame
        Geometry data to be saved. Must have a unique index per road segment.
    geometry_file_path : str
        Path where the geometry CSV will be saved.

    Returns
    -------
    None
        This function does not return a value.
    """
    from pandas import DataFrame, concat

    if rank == 0:
        logger.info("Writing RLINE geometry output file...")

    columns = [
        "Group", "X_b", "Y_b", "Z_b", "X_e", "Y_e", "Z_e", "dCL", "sigmaz0", "#lanes", "lanewidth", "Emis",
        "Hw1", "dw1", "Hw2", "dw2", "Depth", "Wtop", "Wbottom", "l_bh2sw", "l_avgbh", "l_avgbdensity",
        "l_bhdev", "X0_af", "X45_af", "X90_af", "X135_af", "X180_af", "X225_af", "X270_af", "X315_af",
        "l_maxbh", "Link_ID"
    ]
    gdf_out = DataFrame(columns=columns)

    for i, (idx, row) in enumerate(gdf.iterrows()):
        if i % max(1, len(gdf) // 10) == 0:
            pct = int((i + 1) / len(gdf) * 100)
            logger.info(f"Progress: {pct}% ({i + 1}/{len(gdf)})")

        try:
            geom = row.geometry
            values = {
                "Group": "G1",
                "X_b": round(geom.coords[0][0], 3),
                "Y_b": round(geom.coords[0][1], 3),
                "Z_b": 1,
                "X_e": round(geom.coords[-1][0], 3),
                "Y_e": round(geom.coords[-1][1], 3),
                "Z_e": 1,
                "dCL": round(row.get("dCL", 0), 3),
                "sigmaz0": round(row.get("sigmaz0", 0), 3),
                "#lanes": row.get("lanes", 0),
                "lanewidth": round(row.get("lanewidth", 0), 3),
                "Emis": round(row.get("Emis", 0), 3),
                "Hw1": round(row.get("Hw1", 0), 3),
                "dw1": round(row.get("dw1", 0), 3),
                "Hw2": round(row.get("Hw2", 0), 3),
                "dw2": round(row.get("dw2", 0), 3),
                "Depth": round(row.get("Depth", 0), 3),
                "Wtop": round(row.get("Wtop", 0), 3),
                "Wbottom": round(row.get("Wbottom", 0), 3),
                "l_bh2sw": round(row.get("bh_2_sw", 0), 3),
                "l_avgbh": round(row.get("avgbh", 0), 3),
                "l_avgbdensity": round(row.get("avgbdensity", 0), 3),
                "l_bhdev": round(row.get("bhdev", 0), 3),
                "X0_af": round(row.get("X0_af", 0), 3),
                "X45_af": round(row.get("X45_af", 0), 3),
                "X90_af": round(row.get("X90_af", 0), 3),
                "X135_af": round(row.get("X135_af", 0), 3),
                "X180_af": round(row.get("X180_af", 0), 3),
                "X225_af": round(row.get("X225_af", 0), 3),
                "X270_af": round(row.get("X270_af", 0), 3),
                "X315_af": round(row.get("X315_af", 0), 3),
                "l_maxbh": round(row.get("maxbh", 0), 3),
                "Link_ID": idx
            }
            gdf_out.loc[len(gdf_out)] = values
        except Exception as e:
            if rank == 0:
                logger.warning(f"Could not process geometry for index {idx}: {e}")
            continue

    gdf_to_write = COMM_WORLD.gather(gdf_out, root=0)
    if rank == 0:
        gdf_to_write = concat(gdf_to_write)
        gdf_to_write = gdf_to_write.drop_duplicates(subset=["Link_ID"], keep='first')
        gdf_to_write.sort_values("Link_ID", inplace=True)
        gdf_to_write.drop(columns=["Link_ID"], inplace=True)
        gdf_to_write.to_csv(geometry_file_path, index=False, sep=" ")
        logger.info(f"Geometry CSV saved to: {geometry_file_path}")

    return None


def write_rline_output(rline_emissions: dict[str, DataFrame], out_file_path: str) -> None:
    """
    Write distributed R-LINE emissions to individual CSV files per variable.

    This version gathers data across MPI ranks and sums values by road segment.

    Parameters
    ----------
    rline_emissions : dict of {str: DataFrame}
        Dictionary with variable names as keys and DataFrames as values.
        Each DataFrame must have FID_dst as columns and timestamps as index.
    out_file_path : str
        Path where the R-LINE formatted CSV files will be saved.
    """
    import os
    from pathlib import Path

    from pandas import concat, to_datetime

    Path(out_file_path).parent.mkdir(parents=True, exist_ok=True)

    # Generate timestamp metadata once
    reference_df = next(iter(rline_emissions.values()))
    time_df = reference_df.index.to_frame(name="timestamp")
    time_df["timestamp"] = to_datetime(time_df["timestamp"])
    time_df["Year"] = time_df["timestamp"].dt.year
    time_df["Mon"] = time_df["timestamp"].dt.month
    time_df["Day"] = time_df["timestamp"].dt.day
    time_df["JDay"] = time_df["timestamp"].dt.dayofyear
    time_df["Hr"] = time_df["timestamp"].dt.hour
    time_info = time_df[["Year", "Mon", "Day", "JDay", "Hr"]]

    # Process each pollutant
    for var_name, df in rline_emissions.items():
        sorted_fids = sorted(df.columns)
        emission_df = df[sorted_fids].copy()
        emission_df.columns = [f"L_{fid:05d}" for fid in sorted_fids]

        final_df = time_info.join(emission_df)

        # Gather all pieces in rank 0
        df_out = COMM_WORLD.gather(final_df, root=0)

        if COMM_WORLD.Get_rank() == 0:
            # Concatenate all DataFrames
            df_out = concat(df_out, axis=0)

            # Group by timestamp and sum L_xxxxx values (in case of duplicates from different ranks)
            time_keys = ["Year", "Mon", "Day", "JDay", "Hr"]
            df_out = df_out.groupby(time_keys, sort=True).sum(numeric_only=True).reset_index()

            # Force column order: time keys first, then L_XXXXX sorted
            l_columns = sorted([col for col in df_out.columns if col.startswith("L_")])
            df_out = df_out[time_keys + l_columns]

            # Save to CSV
            if '<pollutant>' in out_file_path:
                csv_path = out_file_path.replace('<pollutant>', var_name)
            else:
                csv_path = out_file_path.replace(".csv", f"_{var_name}.csv")
            df_out.to_csv(csv_path, index=False)
            logger.info(f"Saved {var_name} RLINE CSV: {csv_path}")
    return None


def nc2rline(
    input_file: str,
    geometry: str,
    output_file: str,
    output_geometry_file: Optional[str] = None,
    index_column: Optional[str] = None,
    weight_column: Optional[str] = None,
    weight_file: Optional[str] = None,
    var_list: Optional[List[str]] = None,
    crs: Optional[str] = None,
) -> None:
    """
    Convert gridded emissions from a NetCDF file into line-based emissions suitable for RLINE modeling.

    This function performs the following steps:
    1. Opens and filters the input NetCDF emissions dataset.
    2. Loads the geometry of road segments and any optional weighting information.
    3. Retrieves or computes spatial weights to map grid cells to road segments.
    4. Applies the weights to distribute emissions across the line geometries.
    5. Performs a mass balance check to quantify any emission loss due to unmapped areas.
    6. Writes the result to a CSV file in a format compatible with RLINE.

    Parameters
    ----------
    input_file : str
        Path to the NetCDF file containing gridded emissions data.
    geometry : str
        Path to the geometry file (e.g., Shapefile or GeoJSON) with road line features.
    output_file : str
        Path to save the final CSV file in RLINE format.
    output_geometry_file : Optional[str], optional
        If provided, the geometry data will be saved to this path as a CSV file
        formatted for input into the RLINE model. If None, no geometry file will be written.
    index_column : str or None, optional
        Name of the column in the geometry file to use as a unique identifier for line segments.
        If not provided, the row index will be used.
    weight_column : str or None, optional
        Optional column in the geometry file indicating pre-assigned weights per line segment
        (e.g., traffic volume or AADT).
    weight_file : str or None, optional
        Path to a CSV file that stores or receives the spatial interpolation weights.
        If it exists, weights will be read; otherwise, they will be computed and saved here.
    var_list : Optional[List[str]], optional
        List of variable names from the NetCDF to process. If not provided, all available variables are used.
    crs : str or None, optional
        Target CRS to reproject the geometry before processing. If not provided, original CRS is used.

    Returns
    -------
    None
    """
    import os

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"[ERROR] Input NetCDF file not found: {input_file}")
    if (weight_file is None or not os.path.exists(weight_file)) and not os.path.exists(geometry):
        raise FileNotFoundError(f"[ERROR] Geometry file required when weight_file is not provided or missing.")
    if rank == 0:
        logger.info("===== nc2rline =====")
        # Mandatory parameters
        logger.info(f"  input_file: {input_file}")
        logger.info(f"  geometry: {geometry}")
        logger.info(f"  output_file: {output_file}")
        # Optional parameters
        if var_list is not None:
            logger.info(f"  var_list: {var_list}")
        if weight_file is not None:
            logger.info(f"  weight_file: {weight_file}")
        if output_geometry_file is not None:
            logger.info(f"  output_geometry_file: {output_geometry_file}")
        if index_column is not None:
            logger.info(f"  index_column: {index_column}")
        if weight_column is not None:
            logger.info(f"  weight_column: {weight_column}")
        if crs is not None:
            logger.info(f"  crs: {crs}")
    COMM_WORLD.Barrier()

    # Step 1: Load gridded emissions from NetCDF
    nessy = load_netcdf_emissions(input_file=input_file, var_list=var_list)

    # Step 2: Load geometry data only if weights need to be calculated
    if weight_file is not None and os.path.exists(weight_file):
        if rank == 0:
            logger.info(f"Weight file '{weight_file}' exists. Skipping geometry loading.")
        geometry = None
    else:
        # If geometry_file_path is provided, the geometry will be saved for RLINE usage
        # Load geometry for road segments and optional weights
        geometry = prepare_geometry(
            geometry_path=geometry,
            index_column=index_column,
            weight_column=weight_column,
            bbox=nessy.get_bbox(),
            geometry_file_path=output_geometry_file,
            crs=crs,
        )

    # Step 3: Get or compute weights
    weights = get_emission_weights(
        gridded_emissions=nessy,
        geometry=geometry,
        weight_file=weight_file
    )

    # Clean up geometry if it was loaded
    # (del geometry is not needed in Python; let GC handle it)

    # Step 4: Apply weights to distribute emissions
    rline_emissions = distribute_emissions_to_roads(
        gridded_emissions=nessy,
        weights=weights
    )

    # # Step 5: Perform mass balance check
    # mass_balance_check(
    #     gridded_emissions=nessy,
    #     weights=weights
    # )

    # Step 6: Save the output in RLINE format
    write_rline_output(
        rline_emissions=rline_emissions,
        out_file_path=output_file
    )

    if rank == 0:
        logger.info(f"nc2rline completed successfully. Output saved at: {output_file}")
    return None
