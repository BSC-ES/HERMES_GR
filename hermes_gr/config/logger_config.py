# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import logging
import sys
import os
import psutil
from warnings import warn
from mpi4py import MPI
from timeit import default_timer
import pandas as pd
import hermes_gr

logger = None  # Global logger instance

# Initialize Series to store time and memory records, each rank will have its own
my_comm = MPI.COMM_WORLD
my_rank = my_comm.Get_rank()

time_records = pd.Series(
    dtype=float, index=pd.MultiIndex.from_tuples([], names=["section", "subsection"]), name=my_rank)
memory_records = pd.Series(
    dtype=float, index=pd.MultiIndex.from_tuples([], names=["section", "subsection"]), name=my_rank)

# Custom log levels mapping
CUSTOM_LOG_LEVELS = {
    1: logging.INFO,   # Basic workflow info
    2: logging.INFO,   # Extended workflow info (1 tab)
    3: logging.INFO,   # Deep workflow info (2 tabs)
    5: logging.DEBUG,  # Debug info
    7: logging.WARNING,
    9: logging.CRITICAL,
}


def configure_logger(log_path: str, log_detail: int = 1) -> None:
    """
    Configures a unique logger for each MPI process.

    This function ensures that each MPI process has its own dedicated log file.
    The directory for logs is created if it does not exist. The logging level is
    set based on the `log_detail` parameter, with `rank 0` also logging to stdout.

    Parameters
    ----------
    log_path : str
        Path to the log files. The string '<rank>' in the path will be replaced
        by the corresponding MPI rank (formatted with three digits).
    log_detail : int, optional
        Level of log detail:
        - 1: Basic workflow information.
        - 2: Extended workflow information (one tab indentation).
        - 3: Deep workflow information (two tab indentations).
        - 5: Debugging details.
        - 7: Warnings.
        - 9: Critical errors.
        Default is 1.

    Returns
    -------
    None
    """
    global logger
    if logger is not None:
        return  # Skip reconfiguration if logger is already set

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Ensure the log directory exists
    log_dir = os.path.dirname(log_path)
    if not os.path.exists(log_dir):
        if rank == 0:  # Only rank 0 creates the directory
            os.makedirs(log_dir, exist_ok=True)
            print(f"Log directory created: {log_dir}")
    comm.Barrier()  # Synchronize all ranks after creating the directory

    # Map custom log detail to logging levels
    log_level = CUSTOM_LOG_LEVELS.get(log_detail, logging.INFO)

    # Create a logger instance specific to the process rank
    logger = logging.getLogger(f"process_{rank}")
    logger.setLevel(log_level)

    # File handler for logging to a file
    log_path = log_path.replace('<rank>', str(rank).zfill(3))
    file_handler = logging.FileHandler(log_path, mode='w')
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Add a stream handler for rank 0 to log to stdout
    if rank == 0:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_formatter = logging.Formatter("%(message)s")  # Simple format for stdout
        stream_handler.setFormatter(stream_formatter)
        stream_handler.flush = sys.stdout.flush  # Ensure flushing happens after every log
        logger.addHandler(stream_handler)

    logger.info(f"Logger configured for process {rank}. Log level: {logging.getLevelName(log_level)}")
    if rank == 0:
        logger.info("Process 0 will also log messages to stdout.")
    return


def log_message(message: str, level: int = 1) -> None:
    """
    Logs a message at the specified custom log level.

    Messages can be indented based on the log level to reflect workflow depth.
    Warnings and critical errors are logged independently of the logger configuration.

    Parameters
    ----------
    message : str
        The log message to be recorded.
    level : int, optional
        Custom log level:
        - 1: Basic workflow info.
        - 2: Extended workflow info (one tab).
        - 3: Deep workflow info (two tabs).
        - 5: Debug information.
        - 7: Warning messages (also raised as a `warn`).
        - 9: Critical errors (also written to `sys.stderr`).
        Default is 1.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If the logger has not been configured before calling this function.
    """
    if not isinstance(logger, logging.Logger):
        raise RuntimeError("Logger is not configured. Call `configure_logger` first.")

    # Map custom level to Python's logging levels
    python_log_level = CUSTOM_LOG_LEVELS.get(level, logging.INFO)

    # Determine tabulation prefix based on the level of the message
    tab_prefix = ""
    if level == 2:
        tab_prefix = "\t"
    elif level == 3:
        tab_prefix = "\t\t"

    # Log messages at levels 7 and 9 independently of logger configuration
    formatted_message = f"{tab_prefix}{message}"
    if level == 7:  # WARNING level
        logger.warning(formatted_message)
        warn(formatted_message)
    elif level == 9:  # CRITICAL level
        logger.critical(formatted_message)
        sys.stderr.write(formatted_message)
        sys.stderr.flush()
    else:
        # For other levels, log with standard behavior
        if python_log_level == logging.INFO:
            logger.info(formatted_message)
        elif python_log_level == logging.DEBUG:
            logger.debug(formatted_message)
        else:
            logger.log(python_log_level, formatted_message)
    return None


def get_time_stamp() -> float:
    """
    Returns the current timestamp in seconds with high precision.

    This function uses `timeit.default_timer()`, which provides a high-resolution
    timer optimized for performance measurements.

    Returns
    -------
    float
        The current time in seconds with high precision.
    """
    return default_timer()


def get_memory_stamp() -> float:
    """
    Returns the current memory usage of the process in MB.

    This function retrieves the Resident Set Size (RSS), which represents
    the amount of memory occupied by the process in RAM.

    Returns
    -------
    float
        The memory usage of the current process in megabytes (MB).
    """
    if hermes_gr.DEBUG:
        return psutil.Process().memory_info().rss / (1024 ** 2)  # Convert bytes to MB
    else:
        return 0


def finalize_log(output_path: str, records: pd.Series, log_type: str) -> None:
    """
    Gathers and finalizes log data from all MPI processes.

    This function concatenates all records across ranks, computes
    min, mean, and max for each (section, subsection), and writes
    the result to a CSV file.

    Parameters
    ----------
    output_path : str
        Path where the CSV file will be saved.
    records : pd.Series
        A Series containing the recorded values (time or memory).
    log_type : str
        The type of log being processed, e.g., "Time" or "Memory".

    Returns
    -------
    None
    """
    if hermes_gr.DEBUG:
        all_records = my_comm.gather(records, root=0)

        if my_rank == 0:
            df = pd.concat(all_records, axis=1)
            df['min'] = df.min(axis=1)
            df['mean'] = df.mean(axis=1)
            df['max'] = df.max(axis=1)
            df.to_csv(output_path)
            log_message(f"{log_type} records saved to {output_path}")
    return


def finalize_time_log(output_path: str) -> None:
    """
    Gathers and finalizes execution time logs from all MPI processes.

    This function concatenates all the time records across ranks, computes
    min, mean, and max for each (section, subsection), and writes the
    result to a CSV file.

    Parameters
    ----------
    output_path : str
        Path where the CSV file will be saved.

    Returns
    -------
    None
    """
    finalize_log(output_path, time_records, "Time")
    return


def finalize_mem_log(output_path: str) -> None:
    """
    Gathers and finalizes memory usage logs from all MPI processes.

    This function concatenates all memory records across ranks, computes
    min, mean, and max for each (section, subsection), and writes the
    result to a CSV file.

    Parameters
    ----------
    output_path : str
        Path where the CSV file will be saved.

    Returns
    -------
    None
    """
    finalize_log(output_path, memory_records, "Memory")
    return


def record_time(section: str, subsection: str, time_value: float) -> None:
    """
    Records the execution time for a given section and subsection.
    If the entry exists, the time is accumulated.

    Parameters
    ----------
    section : str
        The main section name.
    subsection : str
        The subsection name.
    time_value : float
        The measured time value in seconds.

    Returns
    -------
    None
    """
    if hermes_gr.DEBUG:
        global time_records
        index = (section, subsection)

        # Accumulate time if entry exists
        if index in time_records.index:
            time_records.at[index] += time_value
        else:
            time_records.at[index] = time_value
    return


def record_memory(section: str, subsection: str, mem_value: float) -> None:
    """
    Records the memory usage for a given section and subsection.
    If the entry exists, the memory is accumulated.

    Parameters
    ----------
    section : str
        The main section name.
    subsection : str
        The subsection name.
    mem_value : float
        The measured memory usage in MB.

    Returns
    -------
    None
    """
    if hermes_gr.DEBUG:
        global memory_records
        index = (section, subsection)

        # Accumulate memory if entry exists
        if index in memory_records.index:
            memory_records.at[index] += mem_value
        else:
            memory_records.at[index] = mem_value
    return
