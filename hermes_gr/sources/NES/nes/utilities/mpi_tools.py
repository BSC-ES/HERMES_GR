# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

"""
MPI synchronization and communication tools for ensuring parallel consistency.

This module provides utilities to enforce synchronization barriers and test
object broadcasting between MPI ranks, primarily used for debugging and validating
parallel behavior in distributed environments.
"""
import numpy as np
from mpi4py.MPI import Comm, COMM_WORLD
from typing import Optional


class MyObject(object):
    """
    Dummy class used for broadcast testing across MPI ranks.

    Attributes
    ----------
    name : str
        A test name identifier for the object.
    """
    def __init__(self):
        self.name = "TestObject"


def sync_check(comm: Comm = COMM_WORLD, msg: str = "", abort: bool = False) -> None:
    """
    Perform an MPI synchronization check with an optional abort.

    This function enforces MPI barriers before and after printing a message
    from each rank. It also tests object broadcasting to ensure all ranks are
    able to receive the same Python object type.

    Parameters
    ----------
    comm : MPI.Comm, optional
        The MPI communicator to synchronize across. Default is COMM_WORLD.
    msg : str, optional
        Message to print before and after the synchronization. Default is empty.
    abort : bool, optional
        If True, aborts execution after synchronization. Default is False.

    Returns
    -------
    None
    """
    comm.Barrier()
    print(f"[Rank {comm.Get_rank()}] {msg}", flush=True)
    comm.Barrier()

    check_bcast(comm, msg=msg)

    if abort:
        print(f"[Rank {comm.Get_rank()}] ABORTING {msg}", flush=True)
        comm.Abort(1)

    return None


def check_bcast(comm: Comm = COMM_WORLD, msg=""):
    """
    Test broadcasting of a Python object across MPI ranks.

    This function broadcasts a dummy object from rank 0 to all other ranks,
    and checks that the received object is an instance of the expected type.

    Parameters
    ----------
    comm : MPI.Comm, optional
        The MPI communicator used for broadcasting. Default is COMM_WORLD.
    msg : str, optional
        Message used to label failure output if the check fails.

    Returns
    -------
    bool
        True if broadcast succeeds and object is correctly received.
    """
    if comm.Get_rank() == 0:
        a = MyObject()
    else:
        a = None
    a = comm.bcast(a, root=0)
    if not isinstance(a, MyObject):
        print(f"FAIL {msg}: Received object {a}", flush=True)
        comm.Abort(1)
    return True
