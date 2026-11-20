# CHANGELOG

## 1.3.9

- Release date: 2026/11/20
- Changes and new features:
  - Proxy types and options:
    - Added vector proxies for points, lines, polygons and mixed-geometry files. Mixed-geometry
      features from one source cell are applied together, without priority between geometry types.
    - Added numerical and categorical raster proxies. Categorical proxies use `proxy_categories`
      to select the classes to keep; the categorical-mask/numerical-weight proxy uses a second
      raster to distribute weights within those classes. Both rasters must have the same CRS,
      geotransform and shape; NES does not reproject or resample them and raises `ValueError` if
      their grids differ.
    - Proxy types can be combined in a priority list. Each source cell uses the first proxy that
      covers it and falls back to plain-area conservative weights if none does, preserving mass
      for uncovered cells.
    - Added flux interpolation (`flux=True`) for all proxy types and mixed priority lists. Mass
      weights are converted to flux weights as `w_flux = w_mass * A_src / A_dst`.
    - Added the `column_name` option to vector proxies (`points`, `lines`, `polygons` and
      `multigeometry`) to select feature weights. It defaults to `"weight"`; when that column is
      absent, features receive a weight of 1.0.
  - Proxy behaviour:
    - Point proxies, including point features in multigeometry files, assign each point to one
      source and one destination cell using half-open cells on each grid's native axes. Points on
      shared edges or corners belong to the east/north cell, independently of MPI decomposition
      and grid cropping.
    - Proxy read bounds now include complete intersecting source cells.
    - Line proxies use geodesic lengths in metres on the CRS ellipsoid (WGS84 by default), both
      within source cells and when splitting line portions among destination cells. The line
      component of a multigeometry proxy continues to use planar lengths.
    - Numerical raster proxies honour nodata values: nodata, NaN and non-positive pixels receive
      no weight, while positive fractional values are retained. This prevents invalid pixels from
      receiving artificial weights, for example when redistributing ship emissions with density
      maps.
    - Point proxies with `MultiPoint` or other non-discrete point definitions are rejected.
    - Point-proxy weight matrices record the `nes_point_proxy_ownership` attribute. Older cached
      matrices are recomputed in memory with a warning and are not overwritten. Delete them to
      cache the new weights, and regenerate cached downscaled emissions.
  - Interpolation and numerical results:
    - Plain-area conservative interpolation calculates overlap areas and flux-conversion cell
      areas in m² with the same equal-area projection. Nested destination cells retain the source
      flux while conserving total mass.
    - Equal-area coordinates are calculated directly from longitude and latitude without longitude
      wrapping, fixing interpolation on 0–360 grids with cells at or across 180°.
    - Horizontal interpolation precision follows the input grids. Interpolated data use the least
      precise floating-point dtype among each source variable and the source/destination
      coordinates (float64 when no dtype is inferred); weighted sums remain float64. Nearest-
      neighbour coordinates use the inferred coordinate dtype instead of always using float32.
    - Previously saved plain-area conservative weight matrices must be regenerated to use the
      updated area calculations.
    - Fixed LAEA-to-LCC downscaling with point proxies.

## 1.3.8

- Release date: 2026/11/02
- Changes and new features:
  - Added explicit source CRS, coordinate-name and coordinate-unit options to `open_netcdf` for
    Lambert Azimuthal Equal Area files with incomplete CF metadata. Read spatial dimensions in native
    order and transpose rank-local slices to Y/X without changing the source file or interpolating.
    Conflicting source metadata are rejected; untimed annual inventories retain the existing
    `time_exception=True` and `set_time` workflow.
  - Moved the base `Nes` implementation to `nes.default_nes` and renamed the projection and output-format
    packages to `nes.projections` and `nes.formats`. Updated internal imports and API documentation while
    preserving public exports. Internal imports must use the new module paths.
  - Added per-variable and overall messages to `nes diff` and `nes diffper` when no differences are detected,
    with MPI-wide comparisons, matching NaN handling and a warning when no variables are selected.
  - Added collective `get_grid_configuration()` and `validate_grid_configuration()` APIs for structured grids.
    They compare global native axes and CF projection metadata without creating another grid or broadcasting arrays.
  - Native NES/DEFAULT NetCDF output stores a versioned `nes_grid_configuration` JSON attribute refreshed from
    current axes before writing. Legacy files reconstruct the definition from coordinates/CF metadata, with explicit
    errors when required information is unavailable. Dates, levels, paths, writer options and MPI layout are excluded.
  - Added configuration, legacy metadata and synchronized-rank tests, plus native round trips for regular, global,
    global Monarch, rotated, rotated-nested, LCC and Mercator grids with one and two MPI ranks.
  - Added Lambert Azimuthal Equal Area (LAEA) support through `create_nes(projection="laea", ...)` and automatic
    CF grid-mapping detection in `open_netcdf()`, with spherical and ellipsoidal Earth definitions.
    Native NetCDF output preserves projection metadata, projected coordinates and cell bounds without regridding.
  - Added coordinate, round-trip, independent CF-1.7 compliance and conservative regridding tests for this projection,
    including two-process X/Y/T MPI checks, and documented the projection parameters and supported grid geometry.
  - Fixed a deadlock in time-decomposed horizontal weight construction by using local communicators for
    rank-zero geometry and weight calculations before distributing the weights.
  - Aligned structured projection readers and writers around referenced CF mappings, explicit Earth definitions,
    native coordinate units and bounds, including files without geographical auxiliary coordinates.
    Preserved native corners during selection, expansion and contraction; corrected global Monarch edge cells.
  - Account for original axis encoding precision when inferring native coordinate bounds. Uniform float32 axes
    written by MONARCH no longer fail the nonuniform-axis check. Use full-span spacing, retain stored centres,
    and continue requiring explicit bounds for genuinely nonuniform axes.
  - Extended canonical grid comparisons to Earth shape, false offsets and native cell bounds, including
    Lambert Azimuthal Equal Area grids. Standardized subclass factories, coordinate precision and NumPy-style API docs.
  - Fixed observation metadata and flag handling across MPI ranks and point-layout conversions; enabled
    time-decomposed point creation and independent NetCDF writes. Added cross-projection serial and MPI regressions.

## 1.3.7

- Release date: 2026/10/19
- Changes and new features:
  - Added `nes total -i input.nc [-l VAR ...]` to print global sums across all
    time steps, levels and cells with each variable's original units.
  - Uses MPI reductions and prints results only on rank zero.
  - Updated CLI documentation and wiki with variable selection and usage examples.

## 1.3.6

- Release date: 2026/10/07
- Changes and new features:
  - Fixed CMAQ and WRF-Chem global attributes and dimensions on non-master MPI ranks by using the full-coordinate broadcast methods.

## 1.3.5

- Release date: 2026/08/19
- Changes and new features:
  - Added optional timing diagnostics to `append_time_step_data`, reporting
    unit conversion, communication, NetCDF write and total elapsed times.
  - Added `parallel_io_mode` to select collective or independent NetCDF
    variable writes. Independent mode uses a fixed time dimension to avoid
    extending unlimited dimensions during independent I/O.
  - Added `netcdf_write_unaccounted` timing diagnostics to expose time spent
    inside `append_time_step_data` but outside the measured unit conversion,
    communication, and NetCDF assignment blocks.
  - Improved serial time-step writing for distributed 4D variables by caching
    gather metadata and avoiding repeated shape/count communication.
  - Kept generic gather support for string and non-4D variables through the
    previous gather implementations.

## 1.3.1

- Release date: 2026/07/24
- Changes and new features:
  - nc2rline CLI now uses 7 digits to create the link IDs.
  - Added testing support for global monarch projection.

## 1.3.0

- Release date: 2026/07/10
- Changes and new features:
  - Updated project licensing to Apache-2.0 with a standard root `LICENSE` file.
  - Converted root README and CHANGELOG files from reStructuredText to Markdown.
  - Standardized source, configuration, and script file headers.

## 1.2.7

- Release date: 2026/07/07
- Changes and new features:
  - Create global domain with no-regular grid using MONARCH needs.

## 1.2.6

- Release date: 2026/06/30
- Changes and new features:
  - Fix on inverse flattening projection attribute. If it is not present
    it will be omitted.

## 1.2.5

- Release date: 2026/06/09
- Changes and new features:
  - Fix use of character array in WRF_CHEM
  - Adjust cell filtering method in MBTiles creation to include cells
    with any sector values greater than zero, instead of using a
    threshold for the summation of all sector values.

## 1.2.4b

- Release date: 2026/05/11
- Changes and new features:
  - Fix use of character array in WRF_CHEM

## 1.2.3

- Release date: 2026/04/29
- Changes and new features:
  - Added time_exception to load NetCDF without time dimension

## 1.2.2

- Release date: 2026/03/27
- Changes and new features:
  - create_shapefile(points=False) by default same behaviour but you can
    now create a points geostructure.
  - CLI nc2geostructure sum axis is index == -1
  - CLI nes gridarea
  - MBTiles - Add total sector values from grid into MBTiles as metadata

## 1.2.1

- Release date: 2026/03/23
- Changes and new features:
  - Reintroduced MBTiles creation functionality
  - Vectorize create_shapefile function to improve performance
  - Adding the ability to handle malformed time steps in NetCDFs,
    returning a warning and skipping the malformed time step if there's
    any problems with parsing instead of erroring out
  - Mask cli

## 1.2.0

- Release date: 2026/03/05
- Changes and new features:
  - Added unitest
  - Test and tutorials restructured

## 1.1.19

- Release date: 2026/02/17
- Changes and new features:
  - Updated nes nc2rline CLI to use GeoSeries length

## 1.1.18

- Release date: 2026/02/10

- Changes and new features:

- Added option to avoid N hours on the CLI checkers:

  ``` bash
  nes check -f <input_file.nc> [--nan] [--no-nan] [--inf] [--no-inf] [--avoid_first_hours N]
  nes check_min_max -i <input_file.nc> -c <config_file.yaml> [--avoid_first_hours N]
  ```

## 1.1.17

- Release date: 2026/01/20
- Changes and new features:
  - Add visualization functionality

## 1.1.16

- Release date: 2026/01/09
- Changes and new features:
  - Fixed GitLab pipeline

## 1.1.15

- Release date: 2025/12/16
- Changes and new features:
  - Bugfixes:
    - CLI: nes check_min_max to check variable values ranges.

## 1.1.14

- Release date: 2025/11/29
- Changes and new features:
  - Bugfixes:
    - It accepts now time units as 'years since' but NES transforms it
      to 'days since'
    - climatology_bounds read as metadata
    - It now accepts 'lon' and 'longitude' in the convert_longitude
      method
    - CLI: nes nc2rline force input and output units
  - New Command Line Interface nes with:
    - nes diff
    - nes diffper
  - New Functionalities:
    - Nes().get_totals() -> Dictionary with variable names as keys and
      total sum as value.
    - Nes().get_min(var_name)
    - Nes().get_max(var_name)
    - Nes().daily_statistic(op=[nanmax, nanmean, nanmin]) -> new
      functionalities

## 1.1.13

- Release date: 2025/09/30
- Changes and new features:
  - New Command Line Interface nes with:
    - nes nc2rline (Not validated yet)
    - nes nc2mbtiles
    - Some recatoring
  - New Functionalities:
    - Nes().get_full_bbox(only_master=False)
    - CHIMERE format

## 1.1.11

- Release date: 2025/07/07
- Changes and new features:
  - New Command Line Interface nes with:
    - nes nc2geostructure
    - nes check
    - nes reorder
    - nes interpolate

## 1.1.10

- Release date: 2025/07/03
- Changes and new features:
  - Expand & Contract
  - Bugfix on Coordinates metadata conventions.
  - New entry point to check NaN and Inf values:
    `nes_check`

## 1.1.9

- Release date: 2025/04/22
- Changes and new features:
  - Add additional names for the time variable
  - Added MOCAGE format
  - Bugfix on vertical interpolation.
  - Selecting function allows now to select negative latitudes on 0-360
    ones.
  - Reorder functionality (0 360 to -180 180) as entry point
  - Coordinates metadata conventions.

## 1.1.8

- Release date: 2024/10/07
- Changes and new features:
  - Update installation instructions
  - Rename project from NES to nes

## 1.1.7.post2

- Release date: 2024/10/02
- Changes and new features:
  - Remove mpich requirement

## 1.1.7.post1

- Release date: 2024/09/27
- Changes and new features:
  - Remove import errors on installation using pip

## 1.1.7

- Release date: 2024/09/25
- Changes and new features:
  - Final setup to upload package to PyPI

## 1.1.6

- Release date: 2024/09/25
- Changes and new features:
  - Tests to upload package to PyPI

## 1.1.5

- Release date: 2024/09/20
- Changes and new features:
  - to_netcdf function changes the type argument to nc_type
  - Memory usage optimization
  - Bugfixes

## 1.1.4

- Release date: 2024/05/31
- Changes and new features:
  - Statistics:
    - Rolling mean
  - Documentation
  - Removed negative values on the horizontal interpolation due to
    unmapped NaNs values.
  - Improved load_nes.py removing redundant code
  - Direct access to variable data.
    ([\#77](https://earth.bsc.es/gitlab/es/nes/-/issues/77))
  - New functionalities for vertical extrapolation.
    ([\#74](https://earth.bsc.es/gitlab/es/nes/-/issues/74))
  - Removed cfunits and psutil dependencies.
  - Updated the requirements and environment.yml for the Conda
    environment created in MN5
    ([\#78](https://earth.bsc.es/gitlab/es/nes/-/issues/78))
  - Bugfix:
    - Vertical interpolation for descendant level values (Pressure)
      ([\#71](https://earth.bsc.es/gitlab/es/nes/-/issues/71))
    - Removed lat-lon dimension on the NetCDF projections that not need
      them ([\#72](https://earth.bsc.es/gitlab/es/nes/-/issues/72))
    - Fixed the bug when creating the spatial bounds after selecting a
      region ([\#68](https://earth.bsc.es/gitlab/es/nes/-/issues/68))
    - Fixed the bug related to Shapely deprecated function
      TopologicalError([\#76](https://earth.bsc.es/gitlab/es/nes/-/issues/76))
    - Fixed the bug related to NumPy deprecated
      np.object([\#76](https://earth.bsc.es/gitlab/es/nes/-/issues/76))
    - Removed DeprecationWarnings from shapely and pyproj libraries,
      needed for the porting to MN5
      ([\#78](https://earth.bsc.es/gitlab/es/nes/-/issues/78))

## 1.1.3

- Release date: 2023/06/16
- Changes and new features:
  - Rotated nested projection
  - Improved documentation
  - New function get_fids()
  - Climatology options added
  - Milliseconds, seconds, minutes and days time units accepted
  - Option to change the time units' resolution.
  - Bugs fixing:
    - The input arguments in function new() have been corrected
    - Months to day time units fixed

## 1.1.2

- Release date: 2023/05/15
- Changes and new features:
  - Minor bug fixes
  - Tutorial updates
  - Writing formats (CMAQ, MONARCH, and WRF_CHEM added)
    ([\#63](https://earth.bsc.es/gitlab/es/nes/-/issues/63))

## 1.1.1

- Release date: 2023/04/12
- Changes and new features:
  - Sum of Nes objects
    ([\#48](https://earth.bsc.es/gitlab/es/nes/-/issues/48))
  - Write 2D string data to save variables from shapefiles after doing a
    spatial join
    ([\#49](https://earth.bsc.es/gitlab/es/nes/-/issues/49))
  - Horizontal Interpolation Conservative: Improvement on memory usage
    when calculating the weight matrix
    ([\#54](https://earth.bsc.es/gitlab/es/nes/-/issues/54))
  - Improved time on **concatenate_netcdfs** function
    ([\#55](https://earth.bsc.es/gitlab/es/nes/-/issues/55))
  - Write by time step to avoid memory issues
    ([\#57](https://earth.bsc.es/gitlab/es/nes/-/issues/57))
  - Flux conservative horizontal interpolation
    ([\#60](https://earth.bsc.es/gitlab/es/nes/-/issues/60))
  - Bugs fixing:
    - Bug on cell_methods</span> serial write
      ([\#53](https://earth.bsc.es/gitlab/es/nes/-/issues/53))
    - Bug on avoid_first_hours that where not filtered after read the
      dimensions
      ([\#59](https://earth.bsc.es/gitlab/es/nes/-/issues/59))
    - Bug while reading masked data.
    - grid_mapping NetCDF variable as integer instead of character.

## 1.1.0

- Release date: 2023/03/02
- Changes and new features:
  - Improve Lat-Lon to Cartesian coordinates method (used in
    Providentia).
  - Horizontal interpolation: Conservative
  - Function to_shapefile() to create shapefiles from a NES object
    without losing data from the original grid and being able to select
    the time and level.
  - Function from_shapefile() to create a new grid with data from a
    shapefile after doing a spatial join.
  - Function create_shapefile() can now be used in parallel.
  - Function calculate_grid_area() to calculate the area of each cell in
    a grid.
  - Function calculate_geometry_area() to calculate the area of each
    cell given a set of geometries.
  - Function get_spatial_bounds_mesh_format() to get the lon-lat
    boundaries in a mesh format (used in pcolormesh).
  - Bugs fixing:
    - Correct the dimensions of the resulting points datasets from any
      interpolation.
    - Amend the interpolation method to take into account the cases in
      which the distance among points equals zero.
    - Correct the way we retrieve the level positive value.
    - Correct how to calculate the spatial bounds of LCC and Mercator
      grids: the dimensions were flipped.
    - Correct how to calculate the spatial bounds for all grids: use
      read axis limits instead of write axis limits.
    - Calculate centroids from coordinates in the creation of
      shapefiles, instead of using the geopandas function 'centroid',
      that raises a warning for possible errors.
    - Enable selection of variables on the creation of shapefiles.
    - Correct read and write parallel limits.
    - Correct data type in the parallelization of points datasets.
    - Correct error that appear when trying to select coordinates and
      write the file.

## 1.0.0

- Release date: 2022/11/24
- Changes and new features:
  - First beta release
  - Open:
    - NetCDF:
      - Regular Latitude-Longitude
      - Rotated Lat-Lon
      - Lambert Conformal Conic
      - Mercator
      - Points
      - Points in GHOST format
      - Points in PROVIDENTIA format
  - Parallelization:
    - Balanced / Unbalanced
    - By time axis
    - By Y axis
    - By X axis
  - Create:
    - NetCDF:
      - Regular Latitude-Longitude
      - Rotated Lat-Lon
      - Lambert Conformal Conic
      - Mercator
      - Points
    - Shapefile
  - Write:
    - NetCDF
      - CAMS REANALYSIS format
    - Grib2
    - Shapefile
  - Interpolation:
    - Vertical interpolation
    - Horizontal interpolation
      - Nearest Neighbours
    - Providentia interpolation
  - Statistics:
    - Daily_mean
    - Daily_max
    - Daily_min
    - Last time step
  - Methods:
    - Concatenate (variables of the same period in different files)
