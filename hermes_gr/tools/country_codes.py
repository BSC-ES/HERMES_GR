#!/usr/bin/env python

# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

"""Country code helpers shared by masking, grid and temporal modules."""

from functools import lru_cache

import pycountry


def normalize_iso3(country_code):
    """
    Normalize a country code expected to be ISO 3166-1 alpha-3.

    Parameters
    ----------
    country_code : str
        Country code to normalize.

    Returns
    -------
    str or None
        Uppercase ISO3 country code, or None for missing values.
    """
    if country_code is None:
        return None
    if isinstance(country_code, bytes):
        country_code = country_code.decode()

    country_code = str(country_code).replace("\x00", "").strip().upper()
    if country_code in ("", "NAN", "NONE", "UNK"):
        return None

    return country_code


def normalize_iso2(country_code):
    """
    Normalize a country code expected to be ISO 3166-1 alpha-2.

    Parameters
    ----------
    country_code : str
        Country code to normalize.

    Returns
    -------
    str or None
        Uppercase ISO2 country code, or None for missing values.
    """
    if country_code is None:
        return None
    if isinstance(country_code, bytes):
        country_code = country_code.decode()

    country_code = str(country_code).replace("\x00", "").strip().upper()
    if country_code in ("", "NAN", "NONE", "UNK"):
        return None

    return country_code


@lru_cache(maxsize=None)
def get_iso3166_country(iso3_code):
    """
    Get an ISO 3166-1 country from its alpha-3 code.

    Parameters
    ----------
    iso3_code : str
        ISO3 country code.

    Returns
    -------
    pycountry.db.Country or None
        ISO 3166-1 country object, or None when the code is unknown.
    """
    iso3_code = normalize_iso3(iso3_code)
    if iso3_code is None:
        return None
    return pycountry.countries.get(alpha_3=iso3_code)


def iso3_to_iso2(country_code, strict=False):
    """
    Convert an ISO 3166-1 alpha-3 country code to ISO 3166-1 alpha-2.

    Parameters
    ----------
    country_code : str
        ISO3 country code.
    strict : bool
        If True, raise a KeyError when the ISO3 code is unknown. If False,
        return None for unknown or missing values.

    Returns
    -------
    str or None
        ISO2 country code.
    """
    iso3_code = normalize_iso3(country_code)
    if iso3_code is None:
        return None

    country = get_iso3166_country(iso3_code)
    if country is None and strict:
        raise KeyError(f"Unknown ISO3 country code: {country_code}")
    if country is None:
        return None

    return country.alpha_2


def missing_iso3_codes(country_codes):
    """
    Find country codes that cannot be converted from ISO3 to ISO2.

    Parameters
    ----------
    country_codes : iterable
        Country codes to check.

    Returns
    -------
    set
        Normalized ISO3 country codes missing from the ISO 3166-1 country codes database.
    """
    missing_codes = set()
    for country_code in country_codes:
        iso3_code = normalize_iso3(country_code)
        if iso3_code is not None and get_iso3166_country(iso3_code) is None:
            missing_codes.add(iso3_code)

    return missing_codes


def validate_iso3_codes(country_codes, context="ISO3 code list"):
    """
    Validate that country codes exist in the ISO 3166-1 country codes database.

    Parameters
    ----------
    country_codes : iterable
        Country codes to validate.
    context : str
        Context to include in the error message.

    Returns
    -------
    None
    """
    invalid_codes = sorted(missing_iso3_codes(country_codes))
    if len(invalid_codes) > 0:
        raise ValueError(
            "ERROR: {context} contains ISO3 codes that do not exist in the ISO 3166-1 country codes database: "
            "{invalid_codes}. Check for misspelled ISO3 codes.".format(
                context=context,
                invalid_codes=", ".join(invalid_codes),
            )
        )


def normalize_iso3_mapping(country_values, default_key="default"):
    """
    Normalize ISO3 keys from a country-value mapping.

    Parameters
    ----------
    country_values : dict or None
        Mapping with ISO3 country codes as keys.
    default_key : str
        Key that stores the default value and must not be interpreted as ISO3.

    Returns
    -------
    dict or None
        Mapping with normalized ISO3 keys.
    """
    if country_values is None:
        return None

    normalized_country_values = {}
    if default_key in country_values:
        normalized_country_values[default_key] = country_values[default_key]

    for country_code, value in country_values.items():
        if country_code == default_key:
            continue

        country_code = normalize_iso3(country_code)
        if country_code is None:
            continue
        normalized_country_values[country_code] = value

    return normalized_country_values


def get_country_holiday_calendar(country_iso2, years, holiday_cache, unsupported_countries):
    """
    Get national holidays for a country and year set.

    Parameters
    ----------
    country_iso2 : str
        ISO 3166-1 alpha-2 country code.
    years : iterable
        Years to pre-calculate.
    holiday_cache : dict
        Cache keyed by ``(country_iso2, tuple(years))``.
    unsupported_countries : set
        Countries already known to be unsupported by the holidays package.

    Returns
    -------
    holidays.HolidayBase or None
        Holiday calendar, or None when the country is unknown or unsupported.
    """
    country_iso2 = normalize_iso2(country_iso2)
    if country_iso2 is None:
        return None
    if country_iso2 in unsupported_countries:
        return None

    years = tuple(sorted(set(years)))
    cache_key = (country_iso2, years)
    if cache_key in holiday_cache:
        return holiday_cache[cache_key]

    import holidays

    try:
        holiday_cache[cache_key] = holidays.country_holidays(country_iso2, years=years)
    except (KeyError, NotImplementedError):
        unsupported_countries.add(country_iso2)
        return None

    return holiday_cache[cache_key]
