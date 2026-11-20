# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

"""Print global NetCDF variable totals with their original units."""

from typing import List, Optional

from nes import open_netcdf


def total(input_file: str, var_list: Optional[List[str]] = None) -> None:
    """Sum all time steps, levels and cells for selected (or all) data variables.

    NaNs are ignored by NES.get_totals(). Units are reported as stored in the
    file; no area/time integration or unit conversion is performed. Only MPI
    rank zero prints results. Missing units are reported as 'unknown units'.
    """
    nessy = open_netcdf(input_file)
    try:
        if var_list is not None:
            missing = [name for name in var_list if name not in nessy.variables]
            if missing:
                raise ValueError(f"Variables not found in NetCDF: {', '.join(missing)}")
            nessy.keep_vars(var_list)
        nessy.load()
        totals = nessy.get_totals()
        if nessy.rank == 0:
            for name, value in totals.items():
                units = nessy.variables[name].get("units", "unknown units")
                print(f"{name}: {value} {units}".rstrip())
    finally:
        nessy.close()
