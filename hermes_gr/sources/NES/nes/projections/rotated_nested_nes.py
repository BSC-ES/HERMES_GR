#!/usr/bin/env python
# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

"""Nested rotated latitude-longitude grids derived from an existing parent domain."""

from numpy import linspace, float64
from netCDF4 import Dataset
from .rotated_nes import RotatedNes
from ._coordinates import EARTH_FIELDS

from ._coordinates import validate_creation


class RotatedNestedNes(RotatedNes):

    """Represent a rotated latitude-longitude child grid resolved from its parent.

    Notes
    -----

    Creation requires parent_grid_path, positive integer parent_ratio, one-based i_parent_start/j_parent_start
    and n_rlat/n_rlon counts. Pole and Earth attributes are resolved from the parent; supplied overrides must
    agree with it.
    """
    def __init__(self, comm=None, path=None, info=False, dataset=None, parallel_method="Y",
                 avoid_first_hours=0, avoid_last_hours=0, first_level=0, last_level=None, create_nes=False,
                 balanced=False, times=None, **kwargs):
        """Initialize the RotatedNestedNes class.

        Parameters
        ----------
        comm : mpi4py.MPI.Comm or None, optional
            Communicator for collective operations. None selects MPI.COMM_WORLD.
        path : str or None, optional
            Input/output NetCDF path, as required by the operation.
        info : bool, optional
            Print per-rank processing diagnostics. Default is False.
        dataset : netCDF4.Dataset or None, optional
            Already open input dataset; takes precedence over path when reading.
        parallel_method : {'X', 'Y', 'T'}, optional
            Domain decomposition axis. Structured grids default to Y; station datasets default to X and retain
            their format-specific restrictions.
        avoid_first_hours : int, optional
            Number of hours excluded from the beginning of the input. Default is zero.
        avoid_last_hours : int, optional
            Number of hours excluded from the end of the input. Default is zero.
        first_level : int, optional
            First retained vertical index. Default is zero.
        last_level : int or None, optional
            Last-level selection limit, following base NES slicing. None retains remaining levels.
        create_nes : bool, optional
            Create from geographical parameters if True; otherwise read an existing dataset. Default is False.
        balanced : bool, optional
            Balance the distributed domain. Default is False; chunked output does not support balanced decomposition.
        times : sequence of datetime or None, optional
            Creation time coordinates. Defaults to 1996-12-31 when omitted.
        **kwargs : dict
            Projection-specific creation arguments or documented options forwarded to the underlying operation.

        Notes
        -----
        All ranks of comm must participate in initialization.

        Creation requires parent_grid_path, positive integer parent_ratio, one-based
        i_parent_start/j_parent_start and n_rlat/n_rlon counts. Pole and Earth attributes are resolved from the
        parent; supplied overrides must agree with it.
        """

        if create_nes:
            validate_creation('rotated-nested', kwargs)
        super(RotatedNestedNes, self).__init__(comm=comm, path=path,
                                               info=info, dataset=dataset, balanced=balanced,
                                               parallel_method=parallel_method,
                                               avoid_first_hours=avoid_first_hours, avoid_last_hours=avoid_last_hours,
                                               first_level=first_level, last_level=last_level, create_nes=create_nes,
                                               times=times, **kwargs)

    @staticmethod
    def _get_parent_attributes(projection_data):
        """Get projection attributes from parent grid.

        Parameters
        ----------
        projection_data : dict
            Dictionary with the projection information.

        Returns
        -------
        projection_data : dict
            Dictionary with the projection information, including parameters from the parent grid.
        """

        # Read variables from parent grid
        with Dataset(projection_data["parent_grid_path"], mode="r") as netcdf:
            rlat = next(
                (v for v in netcdf.variables.values() if getattr(v, "standard_name", "") == "grid_latitude"),
                netcdf.variables.get("rlat"),
            )
            rlon = next(
                (v for v in netcdf.variables.values() if getattr(v, "standard_name", "") == "grid_longitude"),
                netcdf.variables.get("rlon"),
            )
            if rlat is None or rlon is None:
                raise ValueError("Nested grids require rotated parent coordinates")
            rlat, rlon = rlat[:], rlon[:]
            mappings = [
                v
                for v in netcdf.variables.values()
                if getattr(v, "grid_mapping_name", "") == "rotated_latitude_longitude"
            ]
            mapping = mappings[0] if len(mappings) == 1 else netcdf.variables["rotated_pole"]
            ratio = projection_data["parent_ratio"]
            if ratio <= 0 or int(ratio) != ratio or len(rlat) < 2 or len(rlon) < 2:
                raise ValueError(
                    "Nested grids require a positive integer parent ratio and nonsingleton parent axes"
                )
            i, j = int(projection_data["i_parent_start"]) - 1, int(projection_data["j_parent_start"]) - 1
            if i < 0 or j < 0 or i >= len(rlon) or j >= len(rlat):
                raise ValueError("Nested parent start indices are one-based and must lie inside the parent grid")
            projection_data.update(
                inc_rlat=float(rlat[1] - rlat[0]) / ratio,
                inc_rlon=float(rlon[1] - rlon[0]) / ratio,
                **{"1st_rlat": float(rlat[j]), "1st_rlon": float(rlon[i])}
            )
            for key in (
                "grid_north_pole_longitude",
                "grid_north_pole_latitude",
                "north_pole_grid_longitude",
            ) + EARTH_FIELDS:
                if key in mapping.ncattrs():
                    projection_data[key] = mapping.getncattr(key)
        return projection_data

    def _get_projection_data(self, create_nes, **kwargs):
        """Retrieves projection data based on grid details.

        Parameters
        ----------
        create_nes : bool
            Flag indicating whether to create new object (True) or use existing (False).
        **kwargs : dict
            Additional keyword arguments for specifying projection details.

        Returns
        -------
        projection_data : dict or None
            CF mapping attributes for structured grids; None for the base or station-only layouts.
        """
        if create_nes:
            projection_data = {"grid_mapping_name": "rotated_latitude_longitude",
                               "parent_grid_path": kwargs["parent_grid_path"],
                               "parent_ratio": kwargs["parent_ratio"],
                               "i_parent_start": kwargs["i_parent_start"],
                               "j_parent_start": kwargs["j_parent_start"],
                               "n_rlat": kwargs["n_rlat"],
                               "n_rlon": kwargs["n_rlon"]
                               }
            projection_data = self._get_parent_attributes(projection_data)
            for key in EARTH_FIELDS + ('north_pole_grid_longitude',):
                if key in kwargs:
                    if key in projection_data and kwargs[key] != projection_data[key]:
                        raise ValueError(f'Nested grid {key} must agree with its parent')
                    projection_data[key] = kwargs[key]
        else:
            projection_data = super()._get_projection_data(create_nes, **kwargs)

        return projection_data

    def _create_rotated_coordinates(self):
        """Calculate rotated latitudes and longitudes from grid details.

        Returns
        -------
        _rlat : dict
            Rotated latitudes dictionary with the "data" key for all the values and the rest of the attributes.
        _rlon : dict
            Rotated longitudes dictionary with the "data" key for all the values and the rest of the attributes.
        """

        # Get grid resolution
        inc_rlon = self.projection_data["inc_rlon"]
        inc_rlat = self.projection_data["inc_rlat"]

        # Get number of rotated coordinates
        n_rlat = self.projection_data["n_rlat"]
        n_rlon = self.projection_data["n_rlon"]

        # Get first coordinates
        first_rlat = self.projection_data["1st_rlat"]
        first_rlon = self.projection_data["1st_rlon"]

        # Calculate rotated latitudes
        rlat = linspace(first_rlat, first_rlat + (inc_rlat * (n_rlat - 1)), n_rlat, dtype=float64)

        # Calculate rotated longitudes
        rlon = linspace(first_rlon, first_rlon + (inc_rlon * (n_rlon - 1)), n_rlon, dtype=float64)

        return {"data": rlat}, {"data": rlon}
