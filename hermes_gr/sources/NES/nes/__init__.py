# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

__date__ = "2026-07-10"
__version__ = "1.3.0"
__all__ = [
    'open_netcdf', 'concatenate_netcdfs', 'create_nes', 'from_shapefile', 'calculate_geometry_area', 'Nes', 'LatLonNes',
    'LCCNes', 'RotatedNes', 'RotatedNestedNes', 'MercatorNes', 'PointsNesProvidentia', 'PointsNesGHOST', 'PointsNes'
]

from .load_nes import open_netcdf, concatenate_netcdfs
# from .load_nes import open_raster
from .create_nes import create_nes, from_shapefile
from .methods.cell_measures import calculate_geometry_area
from .nc_projections import (Nes, LatLonNes, LCCNes, RotatedNes, RotatedNestedNes, MercatorNes, PointsNesProvidentia,
                             PointsNes, PointsNesGHOST)
