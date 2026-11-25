#!/usr/bin/env python
# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

from setuptools import find_packages
from setuptools import setup


def read_version():
    with open("nes/__init__.py") as f:
        for line in f:
            if line.startswith("__version__"):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


with open("README.md", "r") as f:
    long_description = f.read()


REQUIREMENTS = {
    "install": [
        "geopandas",
        "rtree>=0.9.0",
        "pandas",
        "netcdf4",
        "numpy",
        "pyproj",
        "setuptools",
        "scipy",
        "filelock",
        "eccodes",
        "mpi4py",
        "shapely",
        "python-dateutil"
    ],
    "setup": [
        "setuptools_scm",
    ],
}


setup(
    name="NES",
    license="Apache-2.0",
    version=read_version(),
    description="NES (NetCDF for Earth Science) is a library designed to operate, read, and write NetCDF datasets in a "
                "parallel and efficient yet transparent way for the user, using domain decomposition as the "
                "parallelization strategy with customizable axis splitting.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Carles Tena Medina, Alba Vilanova Cortezón, Carmen Piñero Megías",
    author_email="carles.tena@bsc.es, alba.vilanova@bsc.es, carmen.pinero@bsc.es",
    url="https://earth.bsc.es/gitlab/es/nes",
    keywords=["Python", "NetCDF4", "Grib2", "Earth"],
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "License :: OSI Approved :: Apache Software License"
    ],
    package_data={"": [
        "README.md",
        "CHANGELOG.md",
        "LICENSE",
    ]
    },
    setup_requires=REQUIREMENTS["setup"],
    install_requires=REQUIREMENTS["install"],
    python_requires=">=3.7",

    entry_points={
        "console_scripts": [
            "nes=nes.cli.cli:main",
        ]
    }
)
