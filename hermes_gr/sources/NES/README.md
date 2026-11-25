[![Latest release](https://img.shields.io/github/v/release/BSC-ES/NES)](https://github.com/BSC-ES/NES/releases)
[![PyPI version](https://img.shields.io/pypi/v/nes)](https://pypi.org/project/NES/)

# NES

## About

NES (NetCDF for Earth Science) is the Python I/O library used by SNES,
the framework that implements the data post-processing pipelines at the
Earth Sciences department, to read and write netCDF files.

## How to install

Use the following command to install the library:

    pip install nes

Read the [FAQ](https://github.com/BSC-ES/NES/wiki/FAQ) if you
encounter any problems. If they do not resolve your issues, please
contact the developers.

## How to run

For installation guidance, usage instructions, tutorials, and CLI documentation,
see the [GitHub wiki](https://github.com/BSC-ES/NES/wiki).

Calculate totals over all time steps, levels and grid cells:

```bash
nes total -i input.nc
nes total -i input.nc -l NOx CO
```

Omit `-l` to include all data variables. Each line prints the variable name,
total and original units. NaNs are ignored; no unit conversion or area/time
integration is applied. Missing units are shown as `unknown units`.

## Package layout

The base class lives in `nes.default_nes`, concrete grids and observation layouts
in `nes.projections`, output adapters in `nes.formats`, and numerical operations
in `nes.methods`. Public imports such as `from nes import Nes, LatLonNes` remain
unchanged. Internal imports must use the new module paths.

## Releases

Stable source releases are available on [GitHub](https://github.com/BSC-ES/NES/releases).
Published Python packages are available on [PyPI](https://pypi.org/project/NES/).
