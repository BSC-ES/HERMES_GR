#!/usr/bin/env python

# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.


from configargparse import ArgParser
from pandas import read_csv
from warnings import warn
import os
from shutil import rmtree
from datetime import datetime
from mpi4py import MPI
from hermes_gr import __version__


class ConfigGr(ArgParser):
    """
    Initialization of the arguments that the parser can handle.

    Attributes
    ----------
    comm : MPI.COMMUNICATOR
    options : Namespace
        Configuration options
    """
    def __init__(self, comm=None):
        """
        Initialize the Config class

        Parameters
        ----------
        comm : MPI.COMMUNICATOR
        """
        if comm is None:
            comm = MPI.COMM_WORLD
        self.comm = comm

        super(ConfigGr, self).__init__()
        description_csv_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arguments_description.csv")

        args_csv = read_csv(description_csv_file, index_col="Parameter", keep_default_na=False, comment="#")
        self.options = self.read_options(args_csv)
        import hermes_gr
        hermes_gr.DEBUG = self.options.log_level == 5

    def read_options(self, args_csv):
        """
        Reads all the arguments from command line or from the configuration file.
        The value of an argument given by command line has high priority that the one that appear in the
        configuration file.

        Returns
        -------
        Namespace
            Configuration options
        """
        p = ArgParser()
        p.add_argument('-c', '--my-config', required=False, is_config_file=True, help='Path to the configuration file.')
        p.add_argument('--version', '-V', action='version', version="%(prog)s " + __version__)
        # Process each argument in args_csv
        for arg_name, csv_row in args_csv.iterrows():
            arg_name = arg_name.strip()
            # Parse choices
            if csv_row["Choices"].strip() == "None":
                csv_row["Choices"] = None
            else:
                csv_row["Choices"] = self._parse_list(csv_row["Choices"])
            # Add to the parser
            p.add_argument(f"--{arg_name}",
                           dest=arg_name,
                           help=csv_row["Description"],
                           type=eval(csv_row["Data Type"]),
                           choices=csv_row["Choices"],
                           required=csv_row["Required"],
                           default=eval(csv_row["Default"]))

        arguments, unknown = p.parse_known_args()
        if len(unknown) > 0:
            warn(f"Unrecognized arguments: {unknown}")

        for item in vars(arguments):
            is_str = isinstance(arguments.__dict__[item], str)
            if is_str:
                # Parse paths
                arguments.__dict__[item] = self._parse_path(path=arguments.__dict__[item],
                                                            data_path=arguments.data_path,
                                                            input_dir=arguments.input_dir,
                                                            version="v" + __version__,
                                                            domain_type=arguments.domain_type)
                # Check domain options and parse <resolution>
                self.process_domain_resolution_arguments(arguments, item)

        if arguments.output_timestep_type != 'hourly':
            # This change is to try not take into account daylight saving time errors.
            # at 00:00 depending on the timezone you can be in a place or in another.
            arguments.start_date.replace(hour=12)
        arguments.end_date = self._parse_end_date(arguments.end_date, arguments.start_date)

        self.create_dir(arguments.output_dir)
        if arguments.erase_auxiliary_files:
            if os.path.exists(arguments.auxiliary_files_path):
                if self.comm.Get_rank() == 0:
                    rmtree(arguments.auxiliary_files_path)
                self.comm.Barrier()
        self.create_dir(arguments.auxiliary_files_path)

        return arguments

    def get_output_name(self, date):
        """
        Generates the full path of the output replacing <date> by YYYYMMDDHH, YYYYMMDD, YYYYMM or YYYY depending on the
        output_timestep_type.

        Parameters
        ----------
        date : datetime
            Starting date to simulate

        Returns
        -------
        str
            Complete output path
        """
        if self.options.output_timestep_type == 'hourly':
            file_name = self.options.output_name.replace('<date>', date.strftime('%Y%m%d%H'))
        elif self.options.output_timestep_type == 'daily':
            file_name = self.options.output_name.replace('<date>', date.strftime('%Y%m%d'))
        elif self.options.output_timestep_type == 'monthly':
            file_name = self.options.output_name.replace('<date>', date.strftime('%Y%m'))
        elif self.options.output_timestep_type == 'yearly':
            file_name = self.options.output_name.replace('<date>', date.strftime('%Y'))
        else:
            file_name = self.options.output_name
        full_path = os.path.join(self.options.output_dir, file_name)
        return full_path

    @staticmethod
    def create_dir(path):
        """
        Create the given folder if it is not created yet.

        Parameters
        ----------
        path : str
            Path to be created

        """
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    @staticmethod
    def process_domain_resolution_arguments(arguments, item):
        """
        Processes and updates the resolution string in `arguments` based on the `domain_type` and the
        provided resolution increments.

        Parameters
        ----------
        arguments : Namespace
            An object containing various attributes, including `domain_type` and resolution increment
            attributes (e.g.,`domain_type`, `inc_lat`, `inc_lon`, `inc_rlat`, `inc_rlon`, etc.).
        item : str
            The key in the `arguments` dictionary whose value contains the placeholder (`<resolution>`)
            that will be replaced with the actual resolution string.

        Returns
        ------
        None

        Raises
        ------
        RuntimeError
            If the required resolution increment attributes (e.g., `inc_lat`, `inc_lon`, etc.)
            for the specified `domain_type` are not provided.
            If the `domain_type` is unsupported.
        """
        domain_type = arguments.domain_type

        if domain_type in ['global', 'regular', 'global_monarch']:
            if arguments.inc_lat is not None and arguments.inc_lon is not None:
                arguments.__dict__[item] = arguments.__dict__[item].replace(
                    '<resolution>', f'{arguments.inc_lat}_{arguments.inc_lon}')
            else:
                raise RuntimeError(
                    f"Arguments 'inc_lat' and 'inc_lon' are needed for the specified domain_type '{domain_type}'")

        elif domain_type == 'rotated':
            if arguments.inc_rlat is not None and arguments.inc_rlon is not None:
                arguments.__dict__[item] = arguments.__dict__[item].replace(
                    '<resolution>', f'{arguments.inc_rlat}_{arguments.inc_rlon}')
            else:
                raise RuntimeError(
                    f"Arguments 'inc_rlat' and 'inc_rlon' are needed for the specified domain_type '{domain_type}'")

        elif domain_type == 'rotated_nested':
            if arguments.n_rlat is not None and arguments.n_rlon is not None:
                arguments.__dict__[item] = arguments.__dict__[item].replace(
                    '<resolution>', f'{arguments.n_rlat}_{arguments.n_rlon}')
            else:
                raise RuntimeError(
                    f"Arguments 'n_rlat' and 'n_rlon' are needed for the specified domain_type '{domain_type}'")

        elif domain_type in ['lcc', 'mercator']:
            if arguments.inc_x is not None and arguments.inc_y is not None:
                arguments.__dict__[item] = arguments.__dict__[item].replace(
                    '<resolution>', f'{item}_{arguments.inc_x}_{arguments.inc_y}')
            else:
                raise RuntimeError(
                    f"Arguments 'inc_x' and 'inc_y' are needed for the specified domain_type '{domain_type}'")

        else:
            raise RuntimeError(f"Unsupported domain_type: '{domain_type}'")

        return None

    @staticmethod
    def _parse_end_date(end_date, start_date):
        """
        Parse the end date.
        If it's not defined it will be the same date that start_date (to do only one day).

        Parameters
        ----------
        end_date : str or datetime
            Date of the last day to simulate.
        start_date : datetime
            Date of the first day to simulate.

        Returns
        -------
        datetime
            Date to the last day to simulate in datetime format.
        """
        if end_date is None:
            return start_date
        else:
            return parse_start_date(end_date)

    @staticmethod
    def _parse_list(string: str) -> list:
        """
        Parses a string into a list considering ',', ':' and ';' as delimiters.
        Evaluates each element of the list (strings).

        Parameters
        ----------
        string : str, list
            String to convert into a list

        Returns
        -------
        values : list
            Parsed list
        """
        delimiters = [',', ':', ';']

        for delimiter in delimiters:
            if delimiter in string:
                items = string.split(delimiter)
                break
        else:
            # If no delimiter is found, return the string itself
            return [string]
        # Attempt to evaluate each item; if evaluation fails, keep it as a string
        values = []
        for item in items:
            try:
                values.append(eval(item.strip()))
            except Exception:
                values.append(item.strip())
        return values

    @staticmethod
    def _parse_path(path, data_path=None, input_dir=None, version=None, domain_type=None) -> str:
        """
        Parses path using specific patterns.

        This function replaces the following patterns within the input path string with the provided values:
        - '<data_path>'
        - '<input_dir>'
        - '<version>'
        - '<domain_type>'

        Parameters
        ----------
        path : str
            Path with patterns to parse.
        data_path : str, optional
            String to replace the pattern '<data_path>' (default: None).
        input_dir : str, optional
            String to replace the pattern '<input_dir>' (default: None).
        version : str, optional
            String to replace the pattern '<version>' (default: None).
        domain_type : str, optional
            String to replace the pattern '<domain_type>' (default: None).

        Returns
        -------
        path : str
            Parsed path with replaced patterns.
        """

        if data_path is not None:
            path = path.replace("<data_path>", data_path)
        if input_dir is not None:
            path = path.replace("<input_dir>", input_dir)
        if version is not None:
            path = path.replace("<version>", version)
        if domain_type is not None:
            path = path.replace("<domain_type>", domain_type)

        return path


def parse_bool(str_bool):
    """
    Parse the giving string into a boolean.
    The accepted options for a True value are: 'True', 'true', 'T', 't', 'Yes', 'yes', 'Y', 'y', '1'
    The accepted options for a False value are: 'False', 'false', 'F', 'f', 'No', 'no', 'N', 'n', '0'

    If the sting is not in the options it will release a WARNING and the return value will be False.

    Parameters
    ----------
    str_bool : bool or str or int
        Bool to parse
    Returns
    -------
    bool
        Parsed boolean

    """
    true_options = ['True', 'true', 'T', 't', 'Yes', 'yes', 'Y', 'y', '1', 1, True]
    false_options = ['False', 'false', 'F', 'f', 'No', 'no', 'N', 'n', '0', 0, False, None]

    if str_bool in true_options:
        return True
    elif str_bool in false_options:
        return False
    else:
        print('WARNING: Boolean value not contemplated use {0} for True values and {1} for the False ones'.format(
            true_options, false_options
        ))
        print('/t Using False as default')
        return False


def parse_start_date(str_date):
    """
    Parse the date form string to datetime.
    It accepts several ways to introduce the date:
        YYYYMMDD, YYYY/MM/DD, YYYYMMDDhh, YYYYYMMDD.hh, YYYY/MM/DD_hh:mm:ss, YYYY-MM-DD_hh:mm:ss,
        YYYY/MM/DD hh:mm:ss, YYYY-MM-DD hh:mm:ss, YYYY/MM/DD_hh, YYYY-MM-DD_hh.

    Parameters
    ----------
    str_date : str or datetime
        Date to parse

    Returns
    -------
    datetime
        Parsed time-step date-time
    """
    format_types = ['%Y%m%d', '%Y%m%d%H', '%Y%m%d.%H', '%Y/%m/%d_%H:%M:%S', '%Y-%m-%d_%H:%M:%S',
                    '%Y/%m/%d %H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y/%m/%d_%H', '%Y-%m-%d_%H', '%Y/%m/%d']

    date = None
    for date_format in format_types:
        try:
            date = datetime.strptime(str_date, date_format)
            break
        except ValueError as e:
            if str(e) == 'day is out of range for month':
                raise ValueError(e)

    if date is None:
        raise ValueError(f"Date format '{str_date}' not contemplated. Use one of this: {format_types}")

    return date
