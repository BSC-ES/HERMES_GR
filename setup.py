#!/usr/bin/env python

# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

from setuptools import find_packages
from setuptools import setup


# Get the version number from the relevant file
def read_version():
    with open("hermes_gr/__init__.py") as f:
        for line in f:
            if line.startswith("__version__"):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name='hermes_gr',
    license='Apache-2.0',
    # platforms=['GNU/Linux Debian'],
    version=read_version(),
    description='HERMES Global/Regional',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Carles Tena Medina',
    author_email='carles.tena@bsc.es',
    url='https://github.com/BSC-ES/HERMES_GR',

    keywords=['emissions', 'emission model', 'top-down', 'cmaq', 'monarch', 'wrf-chem', 'atmospheric composition',
              'air quality', 'earth science'],
    install_requires=[
        'numpy',
        'netCDF4>=1.3.1',
        'pandas',
        'fiona',
        'Rtree',
        'geopandas',
        'pyproj',
        'configargparse',
        'eccodes-python',
        'holidays',
        'pytz',
        'mpi4py',
    ],
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Atmospheric Science"
    ],
    package_data={'': [
        'README.md',
        'CHANGELOG.md',
        'LICENSE',
        'CITATION.cff',
    ]
    },
    data_files=[('.', ['LICENSE', 'CHANGELOG.md', ]),
                ('hermes_gr/config', ['hermes_gr/config/arguments_description.csv']),
                ('conf', ['conf/hermes.ini',
                          'conf/EI_configuration.csv', ]),
                ],

    include_package_data=True,

    entry_points={
        'console_scripts': [
            'hermes_gr = hermes_gr.hermes:run',
            'hermes_gr_preproc = preproc.cli:main',
            'hermes_sum = hermes_gr.utilities.hermes_sum:run',
        ],
    },
)
