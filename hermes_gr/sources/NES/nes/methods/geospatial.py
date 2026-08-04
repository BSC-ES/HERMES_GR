#!/usr/bin/env python
# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

from ..nc_projections import Nes
import sys
from warnings import warn, filterwarnings
from geopandas import sjoin_nearest, sjoin, read_file, GeoDataFrame
from pandas import DataFrame
from numpy import array, uint32, nan
from shapely.errors import TopologicalError
from typing import Union, Optional, List


def nes_2_geometry(self, ext_geo: Union[str, GeoDataFrame], var_list: Optional[List[str]] = None) -> GeoDataFrame:
    """
    Converts NES data to a GeoDataFrame by mapping NetCDF variable values to geometries.

    This function takes an external geometry source—either a file path to a vector dataset or a GeoDataFrame—
    and intersects or joins it with the NES NetCDF data. The result is a new GeoDataFrame that includes the
    original geometries enriched with values of the selected NES variables.

    Parameters
    ----------
    self : Nes
        A NES object that holds gridded data from a NetCDF file. It must provide spatial metadata such as grid
        coordinates and variable arrays.
    ext_geo : Union[str, GeoDataFrame]
        Input geometries to which the NES data will be mapped. This can be:
            - A string path to a supported file (e.g., .shp, .geojson, .gpkg).
            - A GeoDataFrame containing the geometry column and optional attributes.
    var_list : Optional[List[str]], default=None
        A list of variable names (as strings) from the NES dataset to extract and map.
        If None, all variables available in the NES object will be included.

    Returns
    -------
    GeoDataFrame
        A new GeoDataFrame with the same geometry as `ext_geo`, and columns containing the values of the
        selected NES variables.
    """
    self.create_shapefile()
    self.add_variables_to_shapefile(var_list=var_list)
    return GeoDataFrame()  # Placeholder for implementation
