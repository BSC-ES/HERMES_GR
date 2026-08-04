# Changelog

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
