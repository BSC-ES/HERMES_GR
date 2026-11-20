The NES object
==============

Creating a NES object
----------------------

.. automodule:: nes.create_nes
   :members:
   :undoc-members:
   :show-inheritance:

Loading a NES object
--------------------

.. automodule:: nes.load_nes
   :members:
   :undoc-members:
   :show-inheritance:

Canonical grid configuration
----------------------------

Structured grids expose a global, versioned geographic definition, independent
of time, vertical levels, file paths, writer options and MPI decomposition::

    from nes import create_nes, open_netcdf

    options = dict(lat_orig=40., lon_orig=0., inc_lat=.1, inc_lon=.1,
                   n_lat=4, n_lon=4)
    created = create_nes(projection="regular", **options)
    expected = created.get_grid_configuration()
    created.to_netcdf("grid.nc", serial=True)
    created.comm.Barrier()  # serial multi-rank output must finish before reopening
    loaded = open_netcdf("grid.nc")
    loaded.validate_grid_configuration(expected)
    loaded.validate_grid_configuration(projection="regular", **options)

Both public methods are collective: every rank in the object's communicator
must participate in the same order. Only the master examines global native 1D
axes and CF parameters, then broadcasts a small dictionary or error. Validation
never constructs a reference grid, computes areas/geometry, transforms a full
coordinate mesh, or broadcasts coordinate arrays. The axis check is O(nx + ny).
Mismatches raise the same ValueError on every rank with expected/found values.
Default absolute tolerances are 1e-6 degrees and 1e-3 m (zero relative tolerance).

Native NES/DEFAULT output writes the JSON attribute ``nes_grid_configuration``
(schema version 1), refreshed from actual axes/metadata before opening output.
The canonical dictionary contains projection parameters, global shape, and each
native axis's first/last value, increment, length and layout. Earth shape,
prime meridian and false offsets are included. Projected axes are normalized
to metres and rotated axes to degrees. Constant native cell-bound offsets
are compared as well as centres, so equal centres with different cell widths
are incompatible. Global Monarch's
special polar/end cells are represented by their layout. Rotated-nested grids
store the resolved rotated axes and pole rather than parent paths. Validation
against nested creation arguments reads the parent via NES's existing reader.

Legacy files without the JSON attribute are inspected through their existing
axes and CF metadata. Unsupported grids or missing required information fail
explicitly. Singleton-axis increments require creation metadata, saved metadata
or recoverable coordinate bounds. Saved origins and dimensions never override
actual coordinates; unknown schema versions are rejected. Arbitrary curvilinear grids,
nonuniform native axes and unstructured station grids are not covered by this structured-grid API and
continue to use their existing output behavior.

Configuration compatibility does not establish integrity of every 2D coordinate
or cell polygon. Those require separate, potentially expensive diagnostics.
The native regression module can be run under ``mpiexec -n 2 python -m unittest
nes.tests.projections.test_grid_configuration_roundtrip``; the Makefile target
``run_grid_mpi_test`` and CI include this check.

Native coordinate bounds and storage precision
--------------------------------------------

When explicit native axis bounds are absent, NES infers them only for regular,
strictly monotonic axes. Uniformity checks retain the input floating-point dtype
when determining the rounding tolerance. This matters for MONARCH output, which
encodes native coordinates as float32 before writing dimension variables.
The step is inferred from the complete axis span rather than the first decoded
interval. Both adjacent differences and the full affine sequence must agree
within that tolerance. Coordinate centres and scientific data are not rounded,
regularized or otherwise replaced. Genuinely nonuniform axes still require
explicit two-vertex bounds; a singleton requires bounds or a known increment.
