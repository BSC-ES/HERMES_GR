#!/usr/bin/env python

# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import os
import sys
from netCDF4 import Dataset


def get_arguments():
    """
    Parse command-line arguments.

    Returns
    -------
    tuple
        A tuple containing:
        nc_out : str
            Path to the output NetCDF file.
        nc_list : list of str
            List of input NetCDF file paths to be summed.
    """
    from argparse import ArgumentParser

    argp = ArgumentParser()
    argp.add_argument('nc_out', help='Output NetCDF file')
    argp.add_argument('nc_list', help='List of netCDF files to sum (space-separated)')

    args = argp.parse_args()

    nc_out = args.nc_out
    nc_list = args.nc_list.split()

    return nc_out, nc_list


def copy_first_emis_file(nc_out, base_emis_file):
    """
    Copy the first NetCDF file as the base for output.

    Parameters
    ----------
    nc_out : str
        Path to the output NetCDF file.
    base_emis_file : str
        Path to the base NetCDF file to be copied.

    Returns
    -------
    bool
        True if the copy was successful.
    """
    from shutil import copyfile

    # If the output file already exists, delete it
    if os.path.isfile(nc_out):
        os.remove(nc_out)

    # Copy the base emissions file to the output path
    copyfile(base_emis_file, nc_out)
    print(f'Copying {base_emis_file} file')

    return True


def add_emis(nc_out, nc_list):
    """
    Add emissions data from each NetCDF file in the list to the output NetCDF file.

    Parameters
    ----------
    nc_out : str
        Path to the output NetCDF file.
    nc_list : list of str
        List of NetCDF files to add to the output file.

    Notes
    -----
    Variables such as time and coordinates are skipped. If a variable is missing in
    the output file, it is created and copied; otherwise, it is summed.
    """
    nc_out = Dataset(nc_out, mode='r+')

    for nc_file in nc_list:
        print(f'Adding emissions from {nc_file}')
        nc = Dataset(nc_file, mode='r')

        out_variables = nc_out.variables
        in_variables = list(nc.variables.keys())

        for var in in_variables:
            print(f'\t{var}')
            sys.stdout.flush()

            # Skip non-emission variables
            if var not in ['time', 'lat', 'latitudes', 'lon', 'longitudes', 'lat_bnds', 'lon_bnds', 'rlat', 'rlon',
                           'lev', 'rotated_pole', 'crs', 'cell_area', 'TFLAG', 'Times']:
                if var not in out_variables:
                    # Create the variable if it does not exist in the output file
                    var_new = nc_out.createVariable(var, nc.variables[var].datatype, nc.variables[var].dimensions)
                    for ncattr in nc.variables[var].ncattrs():
                        var_new.setncattr(ncattr, str(nc.variables[var].getncattr(ncattr)))
                    var_new[:] = nc[var][:]
                else:
                    # Sum the variable values, treating masked data as zero
                    try:
                        var_aux = nc[var][:].data
                        var_aux[nc[var][:].mask] = 0
                    except AttributeError:
                        var_aux = nc[var][:]
                    nc_out[var][:] += var_aux

        nc.close()

    nc_out.close()


def run():
    """
    Execute the main process: parse arguments, copy the base file, and add emissions.
    """
    output_netcdf, netcdf_file_list = get_arguments()
    copy_first_emis_file(output_netcdf, netcdf_file_list[0])
    add_emis(output_netcdf, netcdf_file_list[1:])


if __name__ == '__main__':
    run()
