#!/usr/bin/env python

# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import os
from datetime import timedelta, date, datetime
from calendar import (
    monthrange,
    isleap,
    weekday,
    MONDAY,
    TUESDAY,
    WEDNESDAY,
    THURSDAY,
    FRIDAY,
    SATURDAY,
    SUNDAY,
)
from hermes_gr.config import precision, log_message
from hermes_gr.config import get_time_stamp, record_time
from netCDF4 import Dataset
import numpy as np
import pandas as pd
import re
from datetime import timezone as dtz
from nes import Nes


class TemporalDistribution(object):
    """
    TemporalDistribution class that contains all the information for the temporal disaggregation.

    Attributes
    ----------
    grid : Nes
        Destination grid
    date_array : List[datetime]
        List of date-times to simulate

    monthly_profile : dict
        Temporal factors to go from yearly to monthly emissions
        It has 12 keys (from 1 to 12) corresponding to the monthly numerical ID and their corresponding factor.
        e.g.: {1: 0.0, 2: 2.0, 3: 4.75, 4: 2.9, 5: 0.5, 6: 0.4, 7: 0.2, 8: 0.5, 9: 0.75, 10: 0.0, 11: 0.0, 12: 0.0}

    weekly_profile : dict
        e.g.: {0: {0: 1.06, 1: 1.06, 2: 1.06, 3: 1.06, 4: 1.06, 5: 0.85, 6: 0.85},
               1: {0: 1.060967741935484, 1: 1.060967741935484, ... 6: 0.8509677419354839}
               }

    daily_profile : dict
        Temporal factors to pass from yearly to daily emissions
        It has 365 (or 366) keys corresponding to each Julian day of the year.
        e.g.: {1: 1.0, 2: 1.0, ... , 365: 1.0}

    hourly_profile : dict
        e.g.: {'weekday': {0: 0.79, 1: 0.72, 2: 0.72, ... , 22: 0.96, 23: 0.88},
               'saturday': {0: 0.79, 1: 0.72, 2: 0.72, ... , 22: 0.96, 23: 0.88},
               'sunday': {0: 0.79, 1: 0.72, 2: 0.72, ... , 22: 0.96, 23: 0.88}
               }
    """

    def __init__(self, grid, starting_date, timestep_type, timestep_num, timestep_freq, profile_info,):
        """
        Parameters
        ----------
        grid : Nes
            Destination grid
        starting_date : datetime
            Date of the first timestep.
        timestep_type : str
            Relation between time-steps. It can be hourly, monthly or yearly.
        timestep_num : int
            Quantity of time-steps.
        timestep_freq : int
            Quantity of timestep_type between time-steps.
            eg: If timestep_type = hourly; timestep_freq = 2; The difference between time of each timestep is 2 hours.
        profile_info : dict
            Profile information.
            profiles = {
                            'Month': {'profile_id': p_month, 'path': options.p_month},
                            'Week': {'profile_id': p_week, 'path': options.p_week},
                            'Day': {'profile_id': p_day, 'path': options.p_day},
                            'Hour': {'profile_id': p_hour, 'path': options.p_hour}
                       }
        """
        grid.comm.Barrier()
        st_time = get_time_stamp()
        # start_increment("Temporal", "Init")
        # change_labels("Temporal", "Init")
        self.grid = grid

        self.date_array = self.calculate_date_array(starting_date, timestep_type, timestep_num, timestep_freq)
        log_message("Date array", level=3)

        # Daily profile initialization
        self.daily_profile = self.get_daily_profile(
            profile_info["Day"]["profile_id"],
            profile_info["Day"]["path"],
            timestep_type,
        )
        log_message("Daily profile", level=3)

        if self.daily_profile is None:
            # Monthly profile initialization
            self.monthly_profile = self.get_monthly_profile(
                profile_info["Month"]["profile_id"],
                profile_info["Month"]["path"],
                timestep_type,
            )
            grid.comm.Barrier()
            log_message("Monthly profile", level=3)

            # Weekly profile initialization
            self.weekly_profile = self.get_weekly_profile(
                profile_info["Week"]["profile_id"],
                profile_info["Week"]["path"],
                timestep_type,
            )
            log_message("Weekly profile", level=3)
        else:
            log_message("Test Log Temporal 2.B")

            # Disabling monthly & weekly profiles if daily profile is set.
            self.monthly_profile = None
            self.weekly_profile = None
            if profile_info["Month"]["profile_id"] is not None or profile_info["Week"]["profile_id"] is not None:
                log_message("WARNING: Daily profile is set. Monthly or Weekly profiles will be ignored.", level=7,)

        # Hourly profile initialization
        self.hourly_profile = self.get_hourly_profile(
            profile_info["Hour"]["profile_id"],
            profile_info["Hour"]["path"],
            timestep_type,
        )
        log_message("Hourly Profile", level=3)

        record_time("Temporal", "__init__", get_time_stamp() - st_time)
        # stop_increment("Temporal", "Init")

    @staticmethod
    def calculate_date_array(starting_date, timestep_type, timestep_num, timestep_freq):
        """
        Generates the date-times to simulate

        Parameters
        ----------
        starting_date : datetime
            Date of the first timestep.
        timestep_type : str
            Relation between time-steps. It can be hourly, monthly or yearly.
        timestep_num : int
            Quantity of time-steps.
        timestep_freq : int
            Quantity of timestep_type between time-steps.
            eg: If timestep_type = hourly; timestep_freq = 2; The difference between time of each timestep is 2 hours.

        Returns
        -------
        List[datetime]
            List of date-times to simulate
        """
        if timestep_type == "hourly":
            end_date = starting_date + (timestep_num - 1) * timedelta(hours=timestep_freq)
        elif timestep_type == "daily":
            end_date = starting_date + (timestep_num - 1) * timedelta(hours=timestep_freq * 24)
        elif timestep_type == "monthly":
            delta_year = (timestep_num - 1) * timestep_freq // 12
            delta_month = (timestep_num - 1) * timestep_freq % 12
            end_date = starting_date.replace(year=starting_date.year + delta_year,
                                             month=starting_date.month + delta_month)
        elif timestep_type == "yearly":
            delta_year = (timestep_num - 1) * timestep_freq
            end_date = starting_date.replace(year=starting_date.year + delta_year)
        else:
            end_date = starting_date

        date_aux = starting_date
        date_array = []
        while date_aux <= end_date:
            date_array.append(date_aux)  # 3600 seconds per hour

            if timestep_type == "hourly":
                delta = timedelta(hours=timestep_freq)
            elif timestep_type == "daily":
                delta = timedelta(hours=timestep_freq * 24)
            elif timestep_type == "monthly":
                days = monthrange(date_aux.year, date_aux.month)[1]
                delta = timedelta(hours=days * 24)
            elif timestep_type == "yearly":
                if isleap(date_aux.year):
                    delta = timedelta(hours=366 * 24)
                else:
                    delta = timedelta(hours=365 * 24)
            else:
                delta = None

            date_aux = date_aux + delta
        return date_array

    def get_month_array(self):
        """
        Generate the list of involved months to simulate

        Returns
        -------
        List[date]
            List of date of the involved months (first day of each month)
        """
        month_array = [
            date(year=self.date_array[0].year, month=month, day=1)
            for month in np.unique(self.grid.variables["month"]["data"])
        ]

        return month_array

    @staticmethod
    def get_csv_temporal_profile(profile_id, profile_path):
        """
        Obtain the profile data

        Parameters
        ----------
        profile_id : str
            Temporal profile ID
        profile_path : str
            Path to the file that contains all the temporal profiles

        Returns
        -------
        DataSeries
            Temporal profile
        """
        profile = pd.read_csv(profile_path, sep=",", index_col="ID")

        # Check for duplicated index
        if profile.index.has_duplicates:
            raise IndexError(f"ERROR: {profile_path} has duplicated index")

        # Check for ID not in profile table
        try:
            profile = profile.loc[profile_id, :]
        except KeyError:
            raise KeyError(f"ERROR: {profile_id} is not appearing in the {profile_path} profile")

        return profile

    def get_gridded_temporal_profile(self, profile_value, profile_path):
        """
        Read the 3D gridded profile

        3D gridded profile expressed as 2D with the time as first dimension and the second dimension is
        the number of grid cells. e.g.: (24, 1800 * 3600)

        Parameters
        ----------
        profile_value : str
            Profile variable name
        profile_path : str
            Path to the NetCDF that contains the profile data

        Returns
        -------
        np.ndarray
            2D array profile with time as first dimension and the second dimension is the number of grid cells
            e.g.: (365, 1800 * 3600)
        """
        axis_limits = self.grid.read_axis_limits

        profile_nc = Dataset(profile_path)

        profile = profile_nc.variables[profile_value][
            :,
            axis_limits["y_min"]:axis_limits["y_max"],
            axis_limits["x_min"]:axis_limits["x_max"],
        ]
        profile_nc.close()
        profile = profile.reshape((profile.shape[0], profile.shape[-2], profile.shape[-1]))
        return profile

    def get_daily_profile(self, profile_id, profile_path, timestep_type):
        """
        Function to get the daily profiles

        Return options:
            - None -> If no daily profile set.
            - dict -> 365 (or 366) dict keys with the Julian day as key.
            - numpy.ndarray -> 3D array with the gridded profiles.

        Parameters
        ----------
        profile_id : str
            That parameter can be None if no daily profile to set; A path to the gridded profile;
            Or a string (length=4) with the ID of the profile (it have to appear in the profile path).
        profile_path : str
            Path to the CSV file that contains all the daily profiles. The sum of the profile have to be the amount
            of days of the year to simulate (365 or 366 for leap years).
        timestep_type : str
            Type of time step: 'daily' or 'hourly'

        Returns
        -------
        dict
            Temporal factors to pass from yearly to daily emissions

            It has 365 (or 366) keys corresponding to each Julian day of the year.
            e.g.: {1: 1.0, 2: 1.0, ... , 365: 1.0}
        """
        if timestep_type not in ["daily", "hourly"]:
            return None
        if isleap(self.date_array[0].year):
            days_of_year = 366
        else:
            days_of_year = 365

        try:
            if np.isnan(profile_id):
                profile_id = None
        except TypeError:
            profile_id = profile_id

        if profile_id is None:
            # No Daily profile
            return None
        elif os.path.exists(profile_id):
            # Gridded monthly profile
            profile_aux = self.get_gridded_temporal_profile("Fday", profile_id)
            profile_sum = profile_aux.sum() / (profile_aux.shape[-1] * profile_aux.shape[-2])
            profile = {}
            for jday in np.unique(self.grid.variables["julian"]["data"]):
                profile[jday] = profile_aux[jday - 1]
                # if jday == 366 and isleap(self.date_array[0].year) and 366 != profile_aux.shape[0]:
                #     profile[366] = profile_aux[365-1]

        elif len(profile_id) == 4:
            # Plane daily profile
            profile_aux = self.get_csv_temporal_profile(profile_id, profile_path)
            profile_sum = profile_aux.sum()
            profile_aux.index = profile_aux.index.astype(int)
            profile = {}
            for jday in np.unique(self.grid.variables["julian"]["data"]):
                profile[jday] = profile_aux[jday - 1]
                # if jday == 366 and isleap(self.date_array[0].year) and 366 not in profile.keys():
                #     profile[366] = profile[365]

        else:
            log_message("ERROR: Check the .err file to get more info.")
            raise AttributeError(f"ERROR: An error has occurred in the profile {profile_id}.")

        # CHECK if it is normalized
        if profile is not None and round(days_of_year, 2) != round(profile_sum, 2):
            log_message(f"ERROR: The sum of the temporal weight factors is not {days_of_year} "
                        f"(is {profile_sum}).", level=9,)

        return profile

    def get_monthly_profile(self, profile_id, profile_path, timestep_type):
        """
        Function to get the monthly profile

        Return options:
            - None -> If no monthly profile set.
            - dict -> 12 dict keys with the month number (from 1 to 12) as key.
            - numpy.ndarray -> 3D array with the gridded profiles.

        Parameters
        ----------
        profile_id : str
            That parameter can be None if no monthly profile to set; A path to the gridded profile;
            Or a string (length=4) with the ID of the profile (it have to appear in the profile path).
        profile_path : str
            Path to the CSV file that contains all the monthly profiles. The sum of the profile have to be 12.
        timestep_type : str
            Type of time step: 'monthly', 'daily' or 'hourly'

        Returns
        -------
        dict
            Temporal factors to go from yearly to monthly emissions

            It has 12 keys (from 1 to 12) corresponding to the monthly numerical ID and their corresponding factor.
            e.g.: {1: 0.0, 2: 2.0, 3: 4.75, 4: 2.9, 5: 0.5, 6: 0.4, 7: 0.2, 8: 0.5, 9: 0.75, 10: 0.0, 11: 0.0, 12: 0.0}
        """
        if timestep_type not in ["monthly", "daily", "hourly"]:
            return None

        try:
            if np.isnan(profile_id):
                profile_id = None
        except TypeError:
            profile_id = profile_id

        if profile_id is None:
            # No Daily profile
            return None
        elif os.path.exists(profile_id):
            # Gridded monthly profile netCDF
            profile_aux = self.get_gridded_temporal_profile("Fmonth", profile_id)
            profile_sum = profile_aux.sum() / (profile_aux.shape[-2] * profile_aux.shape[-1])
            profile = {}
            for month in np.unique(self.grid.variables["month"]["data"]):
                profile[month] = profile_aux[month - 1]
        elif len(profile_id) == 4:
            # Plane daily profile CSV
            profile_aux = self.get_csv_temporal_profile(profile_id, profile_path)
            profile_sum = profile_aux.sum()
            profile_aux.rename(
                index={
                    "January": 1,
                    "February": 2,
                    "March": 3,
                    "April": 4,
                    "May": 5,
                    "June": 6,
                    "July": 7,
                    "August": 8,
                    "September": 9,
                    "October": 10,
                    "November": 11,
                    "December": 12,
                }, inplace=True)
            profile = {}
            for month in np.unique(self.grid.variables["month"]["data"]):
                profile[month] = profile_aux[month]
        else:
            log_message(f"ERROR: An error has occurred in the profile {profile_id}.", level=9)
            raise RuntimeError(f"ERROR: An error has occurred in the profile {profile_id}.")
        # CHECK if it is normalized
        if profile is not None and round(12, 2) != round(profile_sum, 2):
            log_message(f"ERROR: The sum of the temporal weight factors is not 12 (is {profile_sum}).", level=9)

        return profile

    def rebalance_weekly_profile(self, month_date, profile):
        """
        Re-balance the monthly to day profile takin into account the number of each day type.

        Parameters
        ----------
        month_date : date or datetime
            Date of the first day of the month
        profile : DataFrame
            Temporal profile

        Returns
        -------
        DataFrame
            Balances temporal profile
        """
        weekdays_count = self.calculate_weekdays(month_date)
        if isinstance(profile, dict):
            factor = self.calculate_weekday_factor(profile, weekdays_count)
            for dict_key in profile.keys():
                profile[dict_key] = profile[dict_key] + factor
        elif isinstance(profile, np.ndarray):
            # Gridded
            factor = self.calculate_weekday_gridded_factor(profile, weekdays_count)
            for week_day in range(7):
                profile[week_day, :] = profile[week_day, :] + factor
        else:
            log_message("ERROR: Check the .err file to get more info.")
            raise TypeError(f"ERROR: Profile type '{type(profile)}' not implemented")
        return profile

    def get_weekly_profile(self, profile_id, profile_path, timestep_type):
        """
        Function to get the weekly profiles

        Parameters
        ----------
        profile_id : str
            That parameter can be None if no daily profile to set; A path to the gridded profile;
            Or a string (length=4) with the ID of the profile (it have to appear in the profile path).
        profile_path : str
            Path to the CSV file that contains all the daily profiles. The sum of the profile have to be the amount
            of days of the year to simulate (365 or 366 for leap years).
        timestep_type : str
            Type of time step: 'daily' or 'hourly'

        Returns
        -------
        dict
            Temporal factors to pass from monthly to weekday emissions

            It has as key 0 the basis week profile and with the month number key the balanced one.
                Taking into account how many weekdays types (Mondays, Tuesdays, ...) are in the selected month
            e.g.: {0: {0: 1.06, 1: 1.06, 2: 1.06, 3: 1.06, 4: 1.06, 5: 0.85, 6: 0.85},
                   11: {0: 1.056, 1: 1.056, 2: 1.056, 3: 1.056, 4: 1.056, 5: 0.8460000000000001, 6: 0.8460000000000001},
                   12: {0: 1.0677419354838709, 1: 1.0677419354838709, ... , 6: 0.8577419354838709}}

        """
        if timestep_type not in ["daily", "hourly"]:
            return None
        if isinstance(profile_id, str):
            profile = {}
            log_message(f"Month array: {self.get_month_array()}", level=5)
            for month_date in self.get_month_array():
                log_message(f"Profile ID: {profile_id}", level=5)
                if os.path.exists(profile_id):
                    # Gridded  profile
                    log_message(f"Gridded temporal profile {profile_id}", level=5)
                    profile_aux = self.get_gridded_temporal_profile("Fweek", profile_id)
                    profile_sum = profile_aux.sum() / (profile_aux.shape[-2] * profile_aux.shape[-1])
                    profile_aux = self.rebalance_weekly_profile(month_date, profile_aux)
                    profile_month = {}
                    for weekday in np.unique(self.grid.variables["weekday"]["data"]):
                        profile_month[weekday] = profile_aux[weekday]
                elif len(profile_id) == 4:
                    # Plane  profile
                    log_message(f"Vector temporal profile {profile_id}", level=5)
                    profile_month = self.get_csv_temporal_profile(profile_id, profile_path)
                    profile_sum = profile_month.sum()
                    profile_month.rename(
                        index={
                            "Monday": 0,
                            "Tuesday": 1,
                            "Wednesday": 2,
                            "Thursday": 3,
                            "Friday": 4,
                            "Saturday": 5,
                            "Sunday": 6,
                        }, inplace=True)
                    profile_month = profile_month.to_dict()
                    profile_month = self.rebalance_weekly_profile(month_date, profile_month)
                else:
                    raise AttributeError(f"ERROR: An error has occurred in the profile {profile_id}.")

                # CHECK if it is normalized
                if profile_month is not None and round(7, 2) != round(profile_sum, 2):
                    raise AttributeError(f"ERROR: The sum of the temporal weight factors is not 7 is {profile_sum}.")
                else:
                    profile[month_date.month] = profile_month
        else:
            profile = None
        return profile

    @staticmethod
    def parse_hourly_profile_id(profile_id):
        """
        Parses the profile string in order to provide different IDs for weekday, saturday or sunday

        Parameters
        ----------
        profile_id : str
            That parameter can be None if no daily profile to set; A path to the gridded profile;
            Or a string (length=4) with the ID of the profile (it have to appear in the profile path).

        Returns
        -------
        dict
            Dictionary with the day-type as key and the profile as value
        """
        dict_aux = {}
        try:
            list_aux = list(map(str, re.split(" , | ,|, |,| ", profile_id)))
        except TypeError:
            return None
        if len(list_aux) == 1:
            dict_aux["default"] = profile_id
        else:
            for element in list_aux:
                key_value_list = list(map(str, re.split(":| :|: | : |=| =|= | = ", element)))
                if key_value_list[0] in ["default", "Default", "DEFAULT", "weekday", "Weekday", "WEEKDAY"]:
                    dict_aux["default"] = key_value_list[1]
                elif key_value_list[0] in ["Monday", "Monday", "MONDAY"]:
                    dict_aux[0] = key_value_list[1]
                elif key_value_list[0] in ["Tuesday", "tuesday", "TUESDAY"]:
                    dict_aux[1] = key_value_list[1]
                elif key_value_list[0] in ["Wednesday", "wednesday", "WEDNESDAY"]:
                    dict_aux[2] = key_value_list[1]
                elif key_value_list[0] in ["Thursday", "thursday", "THURSDAY"]:
                    dict_aux[3] = key_value_list[1]
                elif key_value_list[0] in ["Friday", "friday", "FRIDAY"]:
                    dict_aux[4] = key_value_list[1]
                elif key_value_list[0] in ["Saturday", "saturday", "SATURDAY"]:
                    dict_aux[5] = key_value_list[1]
                elif key_value_list[0] in ["Sunday", "sunday", "SUNDAY"]:
                    dict_aux[6] = key_value_list[1]

        return dict_aux

    def get_hourly_profile(self, profile_id, profile_path, timestep_type):
        """
        Function to read the hourly profile.
        Return options:
            - None -> If no hourly profile set.
            - dict -> 24 dict keys with the month number (from 0 to 23) as key.
            - numpy.ndarray -> 3D array with the gridded profiles.

        Parameters
        ----------
        profile_id : str
            That parameter can be None if no daily profile to set; A path to the gridded profile;
            Or a string (length=4) with the ID of the profile (it have to appear in the profile path).
        profile_path : str
            Path to the CSV file that contains all the daily profiles. The sum of the profile have to be the amount
            of days of the year to simulate (365 or 366 for leap years).
        timestep_type : str
            Type of time step: 'daily' or 'hourly'

        Returns
        -------
        dict
            Hourly profile (numpy.ndarray if gridded or dict for the rest)
        """
        if timestep_type not in ["hourly"]:
            return None
        if isinstance(profile_id, str):
            profile = self.parse_hourly_profile_id(profile_id)

            for profile_type, profile_id_aux in profile.items():
                # Gridded monthly profile
                if os.path.exists(profile_id_aux):
                    profile_aux = self.get_gridded_temporal_profile("Fhour", profile_id_aux)
                    profile_sum = profile_aux.sum() / (profile_aux.shape[-2] * profile_aux.shape[-1])
                    profile_hour = {}
                    for hour in np.unique(self.grid.variables["hour"]["data"]):
                        profile_hour[hour] = profile_aux[hour]
                # Plane daily profile
                elif len(profile_id_aux) == 4:
                    profile_hour = self.get_csv_temporal_profile(profile_id_aux, profile_path)
                    profile_sum = profile_hour.sum()
                    profile_hour.index = profile_hour.index.astype(int)
                    profile_hour = profile_hour.to_dict()
                else:
                    log_message("ERROR: Check the .err file to get more info.")
                    raise AttributeError(f"ERROR: An error has occurred in the profile {profile_id_aux}.")

                # CHECK if it is normalized
                if profile_hour is not None and round(24, 2) != round(profile_sum, 2):
                    log_message("ERROR: Check the .err file to get more info.")
                    raise AttributeError(f"ERROR: The sum of the temporal weight factors profile_id_aux is not 24 "
                                         f"(is {profile_sum}).")
                else:
                    profile[profile_type] = profile_hour
        else:
            profile = None

        return profile

    @staticmethod
    def parse_tz(timezone):
        """
        Parse the timezone (string format).

        It is needed because some libraries have more timezones than others, and it
        tries to simplify setting the strange ones into the nearest common one.
        Examples:
            'America/Punta_Arenas': 'America/Santiago',
            'Europe/Astrakhan': 'Europe/Moscow',
            'Asia/Atyrau': 'Asia/Aqtau',
            'Asia/Barnaul': 'Asia/Almaty',
            'Europe/Saratov': 'Europe/Moscow',
            'Europe/Ulyanovsk': 'Europe/Moscow',
            'Europe/Kirov': 'Europe/Moscow',
            'Asia/Tomsk': 'Asia/Novokuznetsk',
            'America/Fort_Nelson': 'America/Vancouver'

        Parameters
        ----------
        timezone : str
            Not parsed timezone.

        Returns
        -------
        str
            Parsed timezone
        """

        tz_dict = {
            "America/Punta_Arenas": "America/Santiago",
            "Europe/Astrakhan": "Europe/Moscow",
            "Asia/Atyrau": "Asia/Aqtau",
            "Asia/Barnaul": "Asia/Almaty",
            "Europe/Saratov": "Europe/Moscow",
            "Europe/Ulyanovsk": "Europe/Moscow",
            "Europe/Kirov": "Europe/Moscow",
            "Asia/Tomsk": "Asia/Novokuznetsk",
            "America/Fort_Nelson": "America/Vancouver",
            "Asia/Famagusta": "Asia/Nicosia",
            "America/Nuuk": dtz(timedelta(hours=-3)),
        }

        if timezone in iter(tz_dict.keys()):
            timezone = tz_dict[timezone]

        return timezone

    def calculate_2d_temporal_factors_old(self, date_aux):
        """
        Calculate the temporal factor to correct the input data of the given date for each cell.

        Parameters
        ----------
        date_aux : datetime.datetime
            Date of the current timestep.

        Returns
        -------
        numpy.array
            2D array with the factors to correct the input data to the date of this timestep.
        """

        st_time = get_time_stamp()
        # start_increment("Temporal", "Calculate 2D temporal factors")
        # change_labels("Temporal", "Calculate 2D temporal factors")

        delta_hours = (date_aux - self.date_array[0]).seconds // 3600

        df = pd.DataFrame(self.grid.shapefile["local"]).reset_index(drop=True)
        df["local"] += timedelta(hours=delta_hours)

        # ===== HOURLY PROFILES =====
        df["weekday"] = df["local"].dt.weekday
        df["hour"] = df["local"].dt.hour
        if self.hourly_profile is not None:
            if isinstance(self.hourly_profile, dict):
                # WEEKDAY
                weekday_profile = self.hourly_profile["weekday"]
                if isinstance(weekday_profile, dict):
                    df["weekday_factor"] = df["hour"].map(weekday_profile)
                else:
                    for hour in range(24):
                        df.loc[df["hour"] == hour, "weekday_factor"] = \
                            weekday_profile[hour, df[df["hour"] == hour].index]
                # SATURDAY
                saturday_profile = self.hourly_profile["saturday"]
                if isinstance(saturday_profile, dict):
                    df["saturday_factor"] = df["hour"].map(saturday_profile)
                else:
                    for hour in range(24):
                        df.loc[df["hour"] == hour, "saturday_factor"] = \
                            saturday_profile[hour, df[df["hour"] == hour].index]

                # SUNDAY
                sunday_profile = self.hourly_profile["sunday"]
                if isinstance(sunday_profile, dict):
                    df["sunday_factor"] = df["hour"].map(sunday_profile)
                else:
                    for hour in range(24):
                        df.loc[df["hour"] == hour, "sunday_factor"] = sunday_profile[hour, df[df["hour"] == hour].index]

                # Selecting profile type depending on the weekday
                df.loc[df["weekday"] <= 4, "hourly_factor"] = df["weekday_factor"][df["weekday"] <= 4].values
                df.loc[df["weekday"] == 5, "hourly_factor"] = df["saturday_factor"][df["weekday"] == 5].values
                df.loc[df["weekday"] == 6, "hourly_factor"] = df["sunday_factor"][df["weekday"] == 6].values

                df.drop(columns=["weekday_factor", "saturday_factor", "sunday_factor"], inplace=True,)
        else:
            df["hourly_factor"] = 1

        df.drop(columns=["hour"], inplace=True)

        # ===== WEEKLY PROFILES =====
        df["month"] = df["local"].dt.month
        if self.weekly_profile is not None:
            for month in np.unique(df["month"]):
                if month not in self.weekly_profile:
                    self.weekly_profile[month] = self.rebalance_weekly_profile(
                        date(year=df.loc[df["month"] == month, "local"].dt.year.values[0],
                             month=month,
                             day=1,), self.weekly_profile[0])

                if isinstance(self.weekly_profile[month], dict):
                    df.loc[df["month"] == month, "weekly_factor"] = df["weekday"].map(self.weekly_profile[month])
                else:
                    for week_day in range(7):
                        df.loc[df["weekday"] == week_day, "weekly_factor"] = \
                            self.weekly_profile[month][week_day, df[df["weekday"] == week_day].index]

        else:
            df["weekly_factor"] = 1

        df.drop(columns=["weekday"], inplace=True)

        # ===== MONTHLY PROFILES =====
        if self.monthly_profile is not None:
            if isinstance(self.monthly_profile, dict):
                df["monthly_factor"] = df["month"].map(self.monthly_profile)
            else:
                for month in np.unique(df["month"]):
                    df.loc[df["month"] == month, "monthly_factor"] = (
                        self.monthly_profile)[month - 1, df[df["month"] == month].index]
        else:
            df["monthly_factor"] = 1

        df.drop(columns=["month"], inplace=True)

        # ===== DAILY PROFILES =====
        df["day"] = df["local"].dt.dayofyear
        if self.daily_profile is not None:
            if isinstance(self.daily_profile, dict):
                df["daily_factor"] = df["day"].map(self.daily_profile)
            else:
                for day in np.unique(df["day"]):
                    df.loc[df["day"] == day, "daily_factor"] = self.daily_profile[day - 1, df[df["day"] == day].index]
        else:
            df["daily_factor"] = 1

        df.drop(columns=["day"], inplace=True)

        df["factor"] = df["monthly_factor"] * df["weekly_factor"] * df["daily_factor"] * df["hourly_factor"]

        df.drop(columns=["monthly_factor", "weekly_factor", "daily_factor", "hourly_factor"], inplace=True)

        factors = np.array(df["factor"].values, dtype=precision).reshape(
            (self.grid.lat["data"].shape[0], self.grid.lon["data"].shape[-1]))
        del df

        record_time("TemporalDistribution", "calculate_2d_temporal_factors", get_time_stamp() - st_time)
        # stop_increment("Temporal", "Calculate 2D temporal factors")

        return factors

    def calculate_2d_temporal_factors(self, i_time):
        """
        Calculate the temporal factor to correct the input data of the given date for each cell.

        Parameters
        ----------
        i_time : int
            Index of the time-step

        Returns
        -------
        numpy.array
            2D array with the factors to correct the input data to the date of this timestep.
        """

        st_time = get_time_stamp()
        # start_increment("Temporal", "Calculate 2D temporal factors")
        # change_labels("Temporal", "Calculate 2D temporal factors")

        shape_2d = (self.grid.lat["data"].shape[0], self.grid.lon["data"].shape[-1])

        factors = np.empty(shape_2d, precision)

        # MONTHLY profiles
        if self.monthly_profile is None:
            factors[:] = 1
        else:
            for month in np.unique(self.grid.variables["month"]["data"][i_time]):
                location = np.where(
                    self.grid.variables["month"]["data"][i_time] == month
                )
                if isinstance(self.monthly_profile[month], np.ndarray):
                    # Gridded profile
                    factors[location] = self.monthly_profile[month][location]
                else:
                    factors[location] = self.monthly_profile[month]

        # WEEKLY profiles
        if self.weekly_profile is not None:
            for month in np.unique(self.grid.variables["month"]["data"][i_time]):
                for weekday in np.unique(
                    self.grid.variables["weekday"]["data"][i_time]
                ):
                    location = np.where(
                        (self.grid.variables["month"]["data"][i_time] == month)
                        & (self.grid.variables["weekday"]["data"][i_time] == weekday)
                    )
                    if isinstance(self.weekly_profile[month][weekday], np.ndarray):
                        # Gridded profile
                        factors[location] *= self.weekly_profile[month][weekday][
                            location
                        ]
                    else:
                        factors[location] *= self.weekly_profile[month][weekday]
        # DAILY profiles
        if self.daily_profile is not None:
            for jday in np.unique(self.grid.variables["julian"]["data"][i_time]):
                location = np.where(
                    self.grid.variables["julian"]["data"][i_time] == jday
                )
                if isinstance(self.daily_profile[jday], np.ndarray):
                    # Gridded profile
                    factors[location] *= self.daily_profile[jday][location]
                else:
                    factors[location] *= self.daily_profile[jday]

        # HOURLY profiles
        if self.hourly_profile is not None:
            for weekday in np.unique(self.grid.variables["weekday"]["data"][i_time]):
                for hour in np.unique(self.grid.variables["hour"]["data"][i_time]):
                    location = np.where(
                        (self.grid.variables["weekday"]["data"][i_time] == weekday)
                        & (self.grid.variables["hour"]["data"][i_time] == hour)
                    )
                    if weekday in self.hourly_profile.keys():
                        # The selected weekday has unique profile
                        if isinstance(self.hourly_profile[weekday][hour], np.ndarray):
                            # Gridded profile
                            factors[location] *= self.hourly_profile[weekday][hour][
                                location
                            ]
                        else:
                            factors[location] *= self.hourly_profile[weekday][hour]
                    else:
                        if isinstance(self.hourly_profile["default"][hour], np.ndarray):
                            # Gridded profile
                            factors[location] *= self.hourly_profile["default"][hour][
                                location
                            ]
                        else:
                            factors[location] *= self.hourly_profile["default"][hour]

        record_time("TemporalDistribution", "calculate_2d_temporal_factors", get_time_stamp() - st_time)
        # stop_increment("Temporal", "Calculate 2D temporal factors")

        return factors

    @staticmethod
    def calculate_weekday_factor(profile, weekdays):
        """
        Obtain the factors to re-balance the weekday profile

        Parameters
        ----------
        profile : DataFrame
            Temporal profile
        weekdays : dict
            Dictionary with teh weekday type as key and the quantity of them as value

        Returns
        -------
        numpy.array
            Factors to re-balance the weekday profile taking into account all the days of the month
        """

        weekdays_factors = 0
        num_days = 0
        for week_day in range(7):
            weekdays_factors += profile[week_day] * weekdays[week_day]
            num_days += weekdays[week_day]

        return (num_days - weekdays_factors) / num_days

    @staticmethod
    def calculate_weekday_gridded_factor(profile, weekdays):
        """
        Obtain the factors to re-balance the weekday profile of a gridded profile

        Parameters
        ----------
        profile : DataFrame
            Temporal profile
        weekdays : dict
            Dictionary with teh weekday type as key and the quantity of them as value

        Returns
        -------
        numpy.array
            Factors to re-balance the weekday profile taking into account all the days of the month
        """
        weekdays_factors = np.zeros((profile.shape[-2], profile.shape[-1]))
        num_days = 0
        for week_day in range(7):
            weekdays_factors += profile[week_day, :] * weekdays[week_day]
            num_days += weekdays[week_day]

        factor = (num_days - weekdays_factors) / num_days

        return factor

    @staticmethod
    def calculate_weekdays(current_date):
        """
        Obtain the quantity of each weekday type for the month of the selected date

        Parameters
        ----------
        current_date : datetime
            Date to simulate

        Returns
        -------
        dict
            Dictionary with teh weekday type as key and the quantity of them as value
        """
        weekdays = [MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY]
        days = [
            weekday(current_date.year, current_date.month, d + 1)
            for d in range(monthrange(current_date.year, current_date.month)[1])
        ]
        weekdays_dict = {}
        count = 0
        for day in weekdays:
            weekdays_dict[count] = days.count(day)

            count += 1

        return weekdays_dict
