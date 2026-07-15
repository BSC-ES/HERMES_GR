# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import numpy as np
from mpi4py.MPI import Comm, COMM_WORLD
from typing import Optional


class MyObject():
    def __init__(self):
        self.name = "TestObject"


def sync_check(comm: Comm = COMM_WORLD, msg: str = "", abort: bool = False) -> None:
    """
    Prints a synchronization message with the MPI rank, flushes stdout,
    and applies a strong MPI barrier.

    Parameters
    ----------
    comm : MPI.Comm
        The MPI communicator to synchronize across.
    msg : str
        The message to print before the barrier.
    abort : bool
        If True it stops the execution.
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

    if comm.Get_rank() == 0:
        a = MyObject()
    else:
        a = None
    a = comm.bcast(a, root=0)
    if not isinstance(a, MyObject):
        print(f"FAIL {msg}: Received object {a}", flush=True)
        comm.Abort(1)
    return True


def scatter_array(comm, data, master=0):
    if comm.Get_size() == 1:
        data = data
    else:
        if comm.Get_rank() == master:
            data = np.array_split(data, comm.Get_size())
        else:
            data = None
        data = comm.scatter(data, root=master)

    return data


def bcast_array(comm, data, master=0):
    if comm.Get_size() == 1:
        data = data
    else:
        data = comm.bcast(data, root=master)

    return data


def balance_dataframe(comm, data, master=0):
    import pandas as pd
    data = comm.gather(data, root=master)
    if comm.Get_rank() == master:
        data = pd.concat(data)
        data = np.array_split(data, comm.Get_size())
    else:
        data = None

    data = comm.scatter(data, root=master)

    return data


def get_balanced_distribution(processors, shape):
    while len(shape) < 4:
        shape = (0,) + shape

    fid_dist = {}
    total_rows = shape[2]

    procs_rows = total_rows // processors
    procs_rows_extended = total_rows-(procs_rows*processors)

    rows_sum = 0
    for proc in range(processors):
        if proc < procs_rows_extended:
            aux_rows = procs_rows + 1
        else:
            aux_rows = procs_rows

        total_rows -= aux_rows
        if total_rows < 0:
            rows = total_rows + aux_rows
        else:
            rows = aux_rows

        min_fid = proc * aux_rows * shape[3]
        max_fid = (proc + 1) * aux_rows * shape[3]

        fid_dist[proc] = {
            'y_min': rows_sum,
            'y_max': rows_sum + rows,
            'x_min': 0,
            'x_max': shape[3],
            'fid_min': min_fid,
            'fid_max': max_fid,
            'shape': (shape[0], shape[1], rows, shape[3]),
        }

        rows_sum += rows

    return fid_dist
