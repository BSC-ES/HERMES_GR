# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

__all__ = [
    "TemporalDistribution",
    "get_country_holiday_calendar",
    "iso3_to_iso2",
    "normalize_iso2",
    "missing_iso3_codes",
    "normalize_iso3",
]


def __getattr__(name):
    if name == "TemporalDistribution":
        from .temporal import TemporalDistribution

        return TemporalDistribution
    if name in [
        "get_country_holiday_calendar",
        "iso3_to_iso2",
        "missing_iso3_codes",
        "normalize_iso2",
        "normalize_iso3",
    ]:
        from hermes_gr.tools.country_codes import (
            get_country_holiday_calendar,
            iso3_to_iso2,
            missing_iso3_codes,
            normalize_iso2,
            normalize_iso3,
        )

        return {
            "get_country_holiday_calendar": get_country_holiday_calendar,
            "iso3_to_iso2": iso3_to_iso2,
            "missing_iso3_codes": missing_iso3_codes,
            "normalize_iso2": normalize_iso2,
            "normalize_iso3": normalize_iso3,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
