# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

__date__ = "2026-11-20"
__version__ = "1.3.9"

__all__ = [
    'LAEANes',
    'LCCNes',
    'LatLonNes',
    'MercatorNes',
    'Nes',
    'PointsNes',
    'PointsNesGHOST',
    'PointsNesProvidentia',
    'RotatedNes',
    'RotatedNestedNes',
    'calculate_geometry_area',
    'concatenate_netcdfs',
    'create_nes',
    'from_shapefile',
    'open_netcdf'
]

# from .load_nes import open_raster
from .create_nes import create_nes, from_shapefile
from .default_nes import Nes
from .load_nes import concatenate_netcdfs, open_netcdf
from .methods.cell_measures import calculate_geometry_area
from .projections import (
    LAEANes,
    LatLonNes,
    LCCNes,
    MercatorNes,
    PointsNes,
    PointsNesGHOST,
    PointsNesProvidentia,
    RotatedNes,
    RotatedNestedNes,
)
