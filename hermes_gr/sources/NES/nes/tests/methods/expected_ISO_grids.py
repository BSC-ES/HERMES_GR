# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of NES.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import numpy as np

# Expected ISO assignments for centroid method.
# Grid cell centres are at lat 10.05..10.95, lon 10.05..10.95.
# Quadrants split at lat=10.5, lon=10.5:
#   REG_A: lon < 10.5 and lat < 10.5
#   REG_B: lon >= 10.5 and lat < 10.5
#   REG_C: lon < 10.5 and lat >= 10.5
#   REG_D: lon >= 10.5 and lat >= 10.5
# Cell centres at 10.05,10.15,10.25,10.35,10.45 fall in the <10.5 half
# Cell centres at 10.55,10.65,10.75,10.85,10.95 fall in the >=10.5 half
EXPECTED_ISO_GRID = np.array(
    [
        [
            "REG_A",
            "REG_A",
            "REG_A",
            "REG_A",
            "REG_A",
            "REG_B",
            "REG_B",
            "REG_B",
            "REG_B",
            "REG_B",
        ],
        [
            "REG_A",
            "REG_A",
            "REG_A",
            "REG_A",
            "REG_A",
            "REG_B",
            "REG_B",
            "REG_B",
            "REG_B",
            "REG_B",
        ],
        [
            "REG_A",
            "REG_A",
            "REG_A",
            "REG_A",
            "REG_A",
            "REG_B",
            "REG_B",
            "REG_B",
            "REG_B",
            "REG_B",
        ],
        [
            "REG_A",
            "REG_A",
            "REG_A",
            "REG_A",
            "REG_A",
            "REG_B",
            "REG_B",
            "REG_B",
            "REG_B",
            "REG_B",
        ],
        [
            "REG_A",
            "REG_A",
            "REG_A",
            "REG_A",
            "REG_A",
            "REG_B",
            "REG_B",
            "REG_B",
            "REG_B",
            "REG_B",
        ],
        [
            "REG_C",
            "REG_C",
            "REG_C",
            "REG_C",
            "REG_C",
            "REG_D",
            "REG_D",
            "REG_D",
            "REG_D",
            "REG_D",
        ],
        [
            "REG_C",
            "REG_C",
            "REG_C",
            "REG_C",
            "REG_C",
            "REG_D",
            "REG_D",
            "REG_D",
            "REG_D",
            "REG_D",
        ],
        [
            "REG_C",
            "REG_C",
            "REG_C",
            "REG_C",
            "REG_C",
            "REG_D",
            "REG_D",
            "REG_D",
            "REG_D",
            "REG_D",
        ],
        [
            "REG_C",
            "REG_C",
            "REG_C",
            "REG_C",
            "REG_C",
            "REG_D",
            "REG_D",
            "REG_D",
            "REG_D",
            "REG_D",
        ],
        [
            "REG_C",
            "REG_C",
            "REG_C",
            "REG_C",
            "REG_C",
            "REG_D",
            "REG_D",
            "REG_D",
            "REG_D",
            "REG_D",
        ],
    ]
)
