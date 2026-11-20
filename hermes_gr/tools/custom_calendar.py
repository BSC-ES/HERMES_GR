#!/usr/bin/env python

# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.


import os
import sys
import datetime
import holidays


def custom_holidays(zone, year):
    """
    Calculate the festivity days that appear in the library holidays adding the Maundy Thursday and the God Friday

    :param zone: Name of the country. It has to appear and has to have the same format (capital letters) of the library
                holidays: https://pypi.python.org/pypi/holidays
    :type zone: str

    :param year: Year to get the festivities.
    :type year: int

    :return: Dictionary with the festivity days.
    :rtype: dict
    """
    c_holidays = get_holidays(zone, year)

    # Adding more festivities than appear in the library
    pascua_sunday = pascua(year)
    c_holidays.update({pascua_sunday - datetime.timedelta(days=3): 'Maundy Thursday'})  # Jueves Santo
    c_holidays.update({pascua_sunday - datetime.timedelta(days=2): 'God Friday'})  # Viernes Santo

    return c_holidays


def get_holidays(zone, year):
    """
    Find the holidays for the selected zone and year.

    :param zone: Name of the country. It has to appear and has to have the same format (capital letters) of the library
                holidays: https://pypi.python.org/pypi/holidays
    :type zone: str

    :param year: Year to found the Pascua.
    :type year: int

    :return: Dictionary with the festivity days.
    :rtype: dict
    """
    method_to_call = getattr(holidays, zone)
    result = method_to_call(years=year)
    return result


def pascua(year):
    """
    Calculate the "Pascua" date

    :param year: Year to found the Pascua.
    :type year: int

    :return: Sunday of Pascua.
    :rtype: datetime.date
    """
    # Magic constants
    m = 24
    n = 5

    # Remainders
    a = year % 19
    b = year % 4
    c = year % 7
    d = (19 * a + m) % 30
    e = (2 * b + 4 * c + 6 * d + n) % 7

    if d + e < 10:
        day = d + e + 22
        month = 3
    else:
        day = d + e - 9
        month = 4

    # Special exceptions
    if day == 26 and month == 4:
        day = 19
    if day == 25 and month == 4 and d == 28 and e == 6 and a > 10:
        day = 18

    return datetime.date(year, month, day)
