# Changelog

## 3.0.8

- Release date: 2026/11/20
- Changes and new features:
  - Added regional rescaling of area emissions to yearly or monthly local totals, with region selection
    (`rescaling_regions`), date-aware CSV paths and a `default` fallback. Incompatible temporal units
    are rejected. Regions must be fully covered by the domain unless `rescaling_allow_partial_regions`
    is enabled; partial coverage then scales in-domain emissions to the full regional totals.
    Regional masks can be reused across selections, with missing-region validation. DEBUG CSVs report
    inventory totals and applied factors.
  - Added proxy-weighted conservative downscaling for area inventories through a YAML
    `proxy_catalogue` and the cross-table `proxy` column. Raster and vector proxies can be prioritized
    per inventory and sector, with area-conservative fallback and separate proxy auxiliary caches.
    Point inventories and interpolation with `double_proxy` are unsupported. Changes to proxy file
    contents require removing the corresponding caches. Added the `pyyaml` dependency.
  - Added optional national holiday handling (`holidays=False` by default): national holidays use
    Sunday temporal profiles; subdivision holidays are excluded. Grid auxiliaries store `country_iso2`,
    with ISO3-to-ISO2 conversion through the new `pycountry` dependency.
  - Strengthened auxiliary-file validation, including grid origin, resolution, projection, dimensions
    and MPI decomposition. Cached grids require paired NetCDF/GeoJSON files; DEBUG also checks their
    geometries. Incompatible auxiliaries are rejected without automatic regeneration.
  - Validate `factor_mask` and `regrid_mask` country codes: invalid ISO3 codes raise errors, while valid
    codes outside the domain produce warnings, including when regridded auxiliaries are reused.
  - Added comma-separated inline layer heights to `vertical_description`, retaining CSV support.
  - Added early speciation CSV validation for missing, NaN or blank expressions, with file, row,
    profile and species diagnostics. Species with no contribution require explicit zero expressions.
  - Fixed GFAS point geographic scaling and masking for daily/hourly inputs, preserving vertical
    allocation fractions and unscaled emissions when masks are omitted. Unsupported GFAS area
    inventories now fail during initialization.
  - Fixed vertical profiles extending above the model top: discard above-top emissions with one
    warning per calculation, preserving retained fractions without redistribution or renormalization.
  - Fixed date-loop termination for non-aligned end dates while preserving inclusive boundaries and
    reinitialization for each valid daily, monthly or yearly period.
  - Fixed CMAQ and WRF-Chem output metadata and dimensions on non-master MPI ranks.
  - Fixed MONARCH output with regular native axes rounded to float32 using precision-aware NES bounds
    inference, preserving coordinate centres and emission data.
  - Fixed cross-table `ref_year` parsing for integer-like floats; empty or non-integer years now raise
    explicit errors.
  - Added the `hermes_gr_scalability` CLI for timing, speedup, efficiency and outlier summaries and
    plots, including replicated runs and legacy v2 timing logs. Added regional rescaling and NES
    writing diagnostics to timing reports.
  - Added `parallel_io_mode` for `collective` or `independent` NetCDF writes (default `independent`).
  - Updated bundled NES to 1.3.9, including the point-proxy boundary ownership fix and
    conservative interpolation fixes.
  - Updated deprecated NumPy/GeoPandas operations without changing scientific calculations.
  - Unified Makefile and GitLab CI execution of the complete pytest suite, expanded regression
    coverage and adjusted test dependencies for Python 3.10 environment compatibility.
  - Added `hermes_gr_preproc GRETA` for Germany's GRETA v1.6.0.1 inventory (2022 and 2030 WM) on the
    native ETRS89 / LAEA Europe grid. Converts annual emissions to HERMES_GR flux files, splits GNFR F
    into F1–F4 using CAMS-REG v8.1 German 2022 totals and speciates NMVOC/PM2.5 where factors exist,
    excluding aggregate F. Supports configurable paths, year selection and optional overwrite.
  - Added `hermes_gr_preproc GRETA_GHG` for GRETA v1.7.1.1 2022 CO2 and CH4, using pollutant-specific
    CAMS-REG F1–F4 shares (`CO2_ff + CO2_bf` for CO2); N2O is skipped.
  - Corrected the default EDGAR v4.3.2 output directory to
    `/esarchive/recon/jrc/edgarv432_ap/monthly_mean`.

## 3.0.1

- Release date: 2026/07/29
- Changes and new features:
  - Updated version of CAMS_REG_AP preprocess to use v9.1 version of the original source

## 3.0.0

- Release date: 2026/07/13
- Changes and new features:
  - Uses the [NES](https://pypi.org/project/NES/) library as the main Python object for I/O and emission handling during calculations.
    - The library is included in HERMES_GR as a source dependency.
  - Custom conservative remapping.
  - Removed CDO `gridarea` dependency.
  - Fixed random MPI freeze errors.
  - Improved timezone handling.
  - Added factor masking for an entire domain.
  - Accepted regular global grids and non-regular grids, as in 2.1.6, for the MONARCH global domain (`global_monarch`).

## 2.1.6

- Release date: 2025/12/05
- Changes and new features:
  - Bugfix on GFAS data precision.

## 2.1.5

- Release date: 2024/05/31
- Changes and new features:
  - Porting to MareNostrum5.
  - MPI abort when some process fails. It stops the execution now.
  - Rank 0 log now in `sys.stdout` with prints.
  - Bugfix with random deadlocks when creating a directory.
  - Added MOCAGE writer.
  - Added MONARCH output grid-mapping metadata.

## 2.1.4

- Release date: Unknown
- Changes and new features:
  - Bug on get gridded temporal profile in rotated-nested projections.

## 2.1.3

- Release date: 2022/03/17
- Changes and new features:
  - NEI preproc.
  - HEMCO preproc.

## 2.1.2

- Release date: 2021/12/17
- Changes and new features:
  - Corrected error while processing GFAS emissions.
  - Added new CAMS-GLOB-SHIP_v3.1 preproc emission inventory.
  - Added new CAMS-REG-AP_v5.1_Refv22 preproc emission inventory.
  - Added new CAMS-GLOB-OCEAN_v3.1 preproc emission inventory.

## 2.1.1

- Release date: 2021/07/29
- Changes and new features:
  - Added new Timezones.
  - Corrected bug with integer on some grid projection creations.

## 2.1.0

- Release date: 2020/03/12
- Changes and new features:
  - Rotated nested grid.
  - GFAS hourly working.
  - Compression level added to the configuration file.
  - Added regional emission inventories. It is mandatory to add coverage column with global or regional in the cross-table file.

## 2.0.3

- Release date: 2020/02/07
- Changes and new features:
  - Corrected bug on grid calculation for some versions of NumPy.
  - Corrected bug on reading vertical description for point source emission inventories.

## 2.0.2

- Release date: 2020/01/14
- Changes and new features:
  - Corrected bug on GFAS as point emissions.

## 2.0.1

- Release date: 2019/12/18
- Changes and new features:
  - Added unrecognized arguments.
  - Dropped some MPI barriers.
  - Changed vertical description from `;` to `,`.
  - Solved bug on 2.0.0 on the creation of rotated grids.
  - Solved bug on 2.0.0 on the creation of the weight matrix for non-global domains.

## 2.0.0

- Release date: 2019/11/20
- Changes and new features:
  - Python 3.
  - `first_time` option.
  - `erase_auxiliary_files` option.
  - Option to return emissions in memory.

## 1.0.4

- Release date: 2019/07/19
- Changes and new features:
  - Specified `timezonefinder` version to be less than 4.0.0 because newer versions do not support Python 2.7.X.
  - Solved bug on some grid creation that sometimes added one extra cell.

## 1.0.3

- Release date: 2019/06/07
- Changes and new features:
  - Solved bug on WRF-Chem unit change.
  - Solved issue on gridded temporal profile dimension names.

## 1.0.2

- Release date: 2019/04/30
- Changes and new features:
  - Corrected error on MONARCH rotated NetCDF metadata.
  - Regional regular latitude-longitude domain.

## 1.0.1

- Release date: 2019/03/12
- Changes and new features:
  - Temporal gridded profiles with destination resolution.
  - Fixed bug for GFAS point in parallel mode.
  - Daily profiles to weekly profiles with 7 elements.
  - New daily profiles with 365/366 elements, gridded and CSV.
  - Added checks for the used temporal profiles.
  - Fixed bug on hourly profile.
  - Added new preprocess.

## 1.0.0

- Release date: 2019/03/01
- Changes and new features:
  - HEMRESv3_GR stable version.
  - GFAS emission inventory as point source emission.

## 0.0.0

- Release date: 2018/09/18
- HERMES_GR beta version first release.
