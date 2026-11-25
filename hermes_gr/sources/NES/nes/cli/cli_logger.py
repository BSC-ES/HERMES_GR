# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import logging
import sys
from mpi4py.MPI import COMM_WORLD


def get_cli_logger(name: str = "cli", level: int = logging.INFO) -> logging.Logger:
    """
    Returns a logger instance for CLI output, configured with rank and flushable stdout.

    Parameters
    ----------
    name : str
        Name of the logger instance.
    level : int
        Logging level to use (e.g., logging.INFO, logging.ERROR).

    Returns
    -------
    logging.Logger
        Configured logger for CLI usage.
    """
    rank = COMM_WORLD.Get_rank()
    handler = logging.StreamHandler(sys.stdout)
    handler.flush = sys.stdout.flush
    if COMM_WORLD.Get_size() > 1:
        handler.setFormatter(logging.Formatter(f"[Rank {rank}] [%(levelname)s] %(message)s"))
    else:
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)
        logger.propagate = False
    return logger
