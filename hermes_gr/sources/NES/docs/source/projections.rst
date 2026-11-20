Projections
===========================

The base ``Nes`` class is defined in ``nes.default_nes``. Concrete grids and
observation layouts are available from ``nes.projections``. Public imports such
as ``from nes import Nes, LatLonNes`` are unchanged. Internal imports must use
the canonical paths documented here.

Structured projection classes share CF grid-mapping discovery, Earth-shape
validation, native bounds handling and double-precision coordinate output.
Readers follow coordinate and mapping references instead of requiring fixed
variable names. Projected axes accept metres, kilometres, centimetres and
millimetres; rotated angles are normalized to degrees. Geographical centres
and cell corners can be reconstructed from native axes and their CRS.

Explicit native bounds are retained during selection and writing. Without
bounds, uniform spacing is required to infer cell edges. Expansion uses native
axes and requires uniform spacing; contraction preserves the selected cells.
Default Earth geometry remains the historical 6371000-metre sphere where no
explicit shape is supplied, except Lambert Azimuthal Equal Area, which requires
an explicit Earth definition. Nested rotated grids inherit the parent shape.

``new`` preserves the receiving subclass. Structured classes support X, Y and T
MPI decomposition. Station datasets support X and T; GHOST and Providentia
retain their observation layouts and gather data for serial output. Station
locations do not define polygon bounds and cannot be expanded or contracted
as raster grids.

Base grid class
---------------------------------------

.. automodule:: nes.default_nes
   :members:
   :undoc-members:
   :show-inheritance:

Latitude/longitude grids
--------------------------------------

.. automodule:: nes.projections.latlon_nes
   :members:
   :undoc-members:
   :show-inheritance:

Lambert Conformal Conic (LCC)
-----------------------------------

.. automodule:: nes.projections.lcc_nes
   :members:
   :undoc-members:
   :show-inheritance:

Mercator projection
----------------------------------------

.. automodule:: nes.projections.mercator_nes
   :members:
   :undoc-members:
   :show-inheritance:

Unstructured station coordinates
--------------------------------------

.. automodule:: nes.projections.points_nes
   :members:
   :undoc-members:
   :show-inheritance:

GHOST observation layout
---------------------------------------------

.. automodule:: nes.projections.points_nes_ghost
   :members:
   :undoc-members:
   :show-inheritance:

Providentia observation layout
---------------------------------------------------

.. automodule:: nes.projections.points_nes_providentia
   :members:
   :undoc-members:
   :show-inheritance:

Rotated projection
---------------------------------------

.. automodule:: nes.projections.rotated_nes
   :members:
   :undoc-members:
   :show-inheritance:

Rotated nested projection
-----------------------------------------------

.. automodule:: nes.projections.rotated_nested_nes
   :members:
   :undoc-members:
   :show-inheritance:

Lambert Azimuthal Equal Area (LAEA)
----------------------------------------

This projection is supported through ``nes.open_netcdf`` and
``nes.create_nes(projection="laea", ...)``. It preserves the projected grid;
no intermediate regridding is performed.

Reading CF files
~~~~~~~~~~~~~~~~

The reader follows the data variable's ``grid_mapping`` reference and identifies
``grid_mapping_name = "lambert_azimuthal_equal_area"``. Mapping and coordinate
variable names are arbitrary. The projected axes must be one-dimensional,
strictly monotonic, and identified by ``standard_name`` values
``projection_x_coordinate`` and ``projection_y_coordinate``. Their dimensions
must be the final two dimensions of the data, in Y, X order. Ascending and
descending axes are accepted. Supported length units are metres (``m``),
kilometres (``km``), centimetres (``cm``), and millimetres (``mm``).

The following CF projection attributes are required:

* ``latitude_of_projection_origin`` and ``longitude_of_projection_origin``.
* An explicit Earth shape: ``earth_radius``, or ``semi_major_axis`` together
  with ``inverse_flattening`` or ``semi_minor_axis``. An inverse flattening
  of zero describes a sphere. Inconsistent ellipsoid definitions are rejected.

``false_easting`` and ``false_northing`` are optional and default to zero.
They use the units of the corresponding projected axis. Earth axes and radii
are in metres. ``longitude_of_prime_meridian`` is also supported. No WGS84
or EPSG:3035 defaults are applied. WKT alone does not replace the required
CF projection and Earth-shape attributes. When present, ``crs_wkt`` is retained
and checked for consistency, including projected units; named datum ensembles
can therefore survive read/write cycles.

Projected axis bounds are read through each coordinate's ``bounds`` reference
and must have shape ``(coordinate, 2)``. Explicit bounds support singleton and
nonuniform axes. Without bounds, only uniformly spaced axes with at least two
centres have an unambiguous supported reconstruction. Geographical cell corners
are transformed from projected bounds, never inferred from geographical centres.
Geographical centres are calculated from the projected axes and CRS.

Reading files with incomplete metadata
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a source whose CRS is known externally, ``open_netcdf`` accepts ``crs``,
``coordinate_names`` and ``coordinate_units``. These options currently support
Lambert Azimuthal Equal Area. CRS values accepted by ``pyproj.CRS`` include
``"EPSG:3035"``, WKT and CRS objects. Coordinate dictionaries use ``x`` and ``y``
keys and refer to source variable names, independently of their dimension names.
Unspecified names and units are read from CF metadata. Contradictory existing
units or projection attributes cause an error rather than being overwritten.

.. code-block:: python

   from datetime import datetime
   import nes

   grid = nes.open_netcdf(
       "original.nc",
       crs="EPSG:3035",
       coordinate_names={"x": "LON", "y": "LAT"},
       coordinate_units={"x": "m", "y": "m"},
       time_exception=True,
   )
   grid.set_time([datetime(2022, 1, 1)])
   grid.set_time_bnds([[datetime(2022, 1, 1), datetime(2023, 1, 1)]])
   grid.load()
   # Supply scientific units and valid standard_name/long_name metadata here.
   grid.to_netcdf("formatted.nc")
   grid.close()

``time_exception=True`` provides a placeholder, which ``set_time`` replaces;
``set_time`` takes a list of datetimes, including for a single annual field.
Time bounds describe the annual interval without distributing emissions in time.
Use X or Y decomposition when setting a previously missing time coordinate.

The source file stays read-only. Coordinates are interpreted as cell centres;
explicit bounds are retained and missing bounds require uniform nonsingleton
axes. Per-rank source slices are transposed lazily into Y/X order before
writing. Data may have two, three or four dimensions: non-spatial dimensions
must follow the existing time or time/level convention. This does not regrid,
rescale emissions or guess units, sector names, years or pollutants. Coordinate
units and emission units are independent. Invalid scientific ``standard_name``
attributes must be corrected or removed by the inventory preprocessor; describing
the CRS alone does not guarantee compliance of all scientific metadata.

Creating and writing a grid
~~~~~~~~~~~~~~~~~~~~~~~~~~

Creation accepts the CF attributes above. ``lat_0`` and ``lon_0`` are aliases
for the projection origin. Geometry parameters follow the LCC/Mercator API:
``x_0`` and ``y_0`` are the first cell's edge coordinates, ``inc_x`` and
``inc_y`` are signed increments, and ``nx`` and ``ny`` are cell counts.
All creation geometry and false offsets use metres. Grid edge origins are
separate from the projection's false easting and false northing.

.. code-block:: python

   import numpy as np
   import nes

   grid = nes.create_nes(
       projection="laea",
       latitude_of_projection_origin=52.0,
       longitude_of_projection_origin=10.0,
       false_easting=4321000.0, false_northing=3210000.0,
       semi_major_axis=6378137.0, inverse_flattening=298.257222101,
       x_0=4000000.0, y_0=3000000.0,
       inc_x=1000.0, inc_y=1000.0, nx=10, ny=8,
   )
   grid.variables["emission"] = {
       "data": np.ones((1, 1) + grid.lat["data"].shape),
       "units": "kg", "long_name": "emission per grid cell",
   }
   grid.global_attrs.update(title="Emissions on a Lambert Azimuthal Equal Area grid", history="Created with NES")
   grid.to_netcdf("emissions.nc")
   reopened = nes.open_netcdf("emissions.nc")
   reopened.load()
   reopened.close()

The standard NES writer produces CF-1.7 files with a scalar projection mapping,
projection parameters, projected coordinate bounds, double-precision
``lat(y, x)``/``lon(y, x)`` and their four-corner bounds. Each data variable
references the mapping and geographical auxiliary coordinates. Projected units,
coordinate values, Earth shape and false offsets survive a read/write cycle.
NES's default level is written as a dimensionless model index; applications
using physical vertical coordinates should supply their units and metadata.
Data units and scientific metadata remain the application's responsibility.

Spatial operations and limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These grids reuse the structured-grid shapefile, intersection, cell-area and
conservative interpolation infrastructure and the X, Y and T MPI decomposition.
Use ``kind="Conservative"`` for extensive cell totals. For intensive quantities,
``flux=True`` additionally applies NES's source/destination cell-area ratio.
Only common domain coverage is conserved; a destination enclosing the source
is useful when checking complete emission totals.

NES represents spatial cells by four geographical corners. Polygon edges and
intersection weights use the existing NES geographical polygon approximations,
not exact integrations of curved projected edges. The existing limitations for
antimeridian/polar polygons and area approximations also apply to these grids.
Only a single structured mapping per dataset is supported; two-dimensional
projected axes and mixed grids are rejected. Expansion requires uniform axes.
Coordinates or corners outside the finite projection domain are rejected.
Missing projection and axis metadata may be supplied through the explicit input
options above. Inventory-specific scientific metadata and transformations remain
in the inventory preprocessor, without changing the original projected grid.

The projection attributes follow `CF Appendix F
<https://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/build/apf.html>`_.
The optional independent validation test uses IOOS ``compliance-checker`` with
``cf:1.7``; it is a test tool, not a NES runtime dependency.

.. automodule:: nes.projections.laea_nes
   :members:
   :undoc-members:
   :show-inheritance:
