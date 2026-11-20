# HERMES_GR

[![DOI](https://img.shields.io/badge/DOI-10.82201%2FTQF44A-blue)](https://doi.org/10.82201/TQF44A)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

**HERMES_GR (High-Elective Resolution Modelling Emission System - Global-Regional)** is an open-source, Python-based emission processing system designed to transform officially reported emission inventories into model-ready datasets for air quality and greenhouse gas (GHG) simulations.

It provides high-resolution **spatial**, **temporal**, **vertical**, and **chemical speciation** processing, enabling integration with state-of-the-art atmospheric models for research and regulatory applications.

HERMES_GR can write NetCDF emission output files following the conventions of the **CMAQ**, **MONARCH**, and **WRF-Chem** atmospheric chemistry models.

## Main Features

- **End-to-end emission processing framework** from inventory ingestion to CTM-ready outputs.
- **Flexible domain definition** for global and regional domains, including global, regular latitude-longitude, rotated latitude-longitude, rotated nested, Mercator, and Lambert conformal conic grids.
- **Multi-inventory processing** for global and regional emission inventories covering multiple source types, pollutants, and base years, with user-defined scaling and masking factors for inventory adjustment and combination.
- **Advanced temporal allocation** using default or user-defined profiles, including spatially gridded temporal factors to represent local variability such as temperature-dependent emissions.
- **Vertical distribution** through literature-based or user-defined vertical profiles for emission injection heights by sector.
- **Chemical speciation** from base pollutants into user-defined gas-phase and aerosol chemical mappings.
- **Parallel execution** with MPI-based domain decomposition for efficient HPC workflows and large-scale simulations.
- **CTM-compatible outputs** through generic NetCDF files and model-specific NetCDF conventions for **CMAQ**, **MONARCH**, and **WRF-Chem**.

## Dependencies

HERMES_GR requires the software and Python packages listed in [`environment.yml`](environment.yml). The recommended installation path uses Conda and includes MPI-enabled builds for packages such as NetCDF4 and HDF5.

Key dependencies include:

- Python 3.10
- MPI-enabled `libnetcdf`, `netCDF4`, and `h5py`
- `mpi4py`
- `numpy`, `geopandas`, `shapely`, `pyproj`, `rasterio`, and `gdal`
- `eccodes` and `python-eccodes`
- `configargparse`, `pyyaml`, `openpyxl`, `holidays`, and `numba`

## Quick Start

Clone the repository and install the full Conda environment with the provided `Makefile`:

```bash
git clone https://github.com/BSC-ES/HERMES_GR.git
cd HERMES_GR
make full_installation
```

For advanced installation on HPC systems, see the [installation guide](https://github.com/BSC-ES/HERMES_GR/wiki/HERMES_GR_Installation_Makefile).

## Running HERMES_GR

HERMES_GR is run from the command line with a configuration file:

```bash
hermes_gr --my-config /path/to/your_config.ini
```

Parameters can be defined in the INI file or overridden through the command line. When a parameter is defined in both places, the command-line value takes precedence.

Configuration details are available in the [HERMES_GR configuration guide](https://github.com/BSC-ES/HERMES_GR/wiki/HERMES_GR_HowToConfigure).

## Documentation

Full documentation is available in the [project Wiki](https://github.com/BSC-ES/HERMES_GR/wiki), including:

- [User guide and configuration](https://github.com/BSC-ES/HERMES_GR/wiki/HERMES_GR_HowToConfigure)
- [Input file formats](https://github.com/BSC-ES/HERMES_GR/wiki/HERMES_GR_InputFiles)
- [Installation guide](https://github.com/BSC-ES/HERMES_GR/wiki/HERMES_GR_Installation_Makefile)
- [Short examples](https://github.com/BSC-ES/HERMES_GR/wiki/HERMES_GR_ShortExamples)
- [Change log](CHANGELOG.md)

## Availability and License

HERMES_GR is distributed under the [Apache License 2.0](LICENSE).

## How to Cite

If you use **HERMES_GR** in your research, please cite the software release as the primary reference:

```text
Tena, C., Gehlen, J., Rizza, L., & Guevara, M. (2026).
HERMES_GR: High-Elective Resolution Modelling Emission System - Global-Regional
(version v3.0.0). BSC Dataverse.
https://doi.org/10.82201/TQF44A
```

For the scientific methodology and model description, please also cite Guevara et al. (2019):

```text
Guevara, M., Tena, C., Porquet, M., Jorba, O., and Pérez García-Pando, C.:
HERMES, a stand-alone multi-scale atmospheric emission modelling framework - Part 1:
global and regional module, Geosci. Model Dev., 12, 1885-1907,
https://doi.org/10.5194/gmd-12-1885-2019, 2019.
```

Paper link: <https://gmd.copernicus.org/articles/12/1885/2019/>

## Related Resources

- Software DOI: <https://doi.org/10.82201/TQF44A>
- Methodology paper: <https://gmd.copernicus.org/articles/12/1885/2019/>
- Source code: <https://github.com/BSC-ES/HERMES_GR>
- Documentation: <https://github.com/BSC-ES/HERMES_GR/wiki>
- Benchmark data: <https://dataverse.bsc.es/dataset.xhtml?persistentId=perma:BSC/WQ1714>

## Contact and Support

For questions, support, or contributions:

- [Carles Tena](https://www.bsc.es/es/tena-carles) - <carles.tena@bsc.es>
- [Marc Guevara](https://www.bsc.es/guevara-marc) - <marc.guevara@bsc.es>
- [Johanna Gehlen](https://www.bsc.es/gehlen-johanna) - <johanna.gehlen@bsc.es>
- [Luca Rizza](https://www.bsc.es/rizza-luca) - <luca.rizza@bsc.es>

Contributions are welcome via GitHub pull requests.
