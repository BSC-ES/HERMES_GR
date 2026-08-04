#!/usr/bin/env python

# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import unittest


class TestTemporalDistribution(unittest.TestCase):
    def setUp(self):
        pass

    # def testing_calculate_ending_date_1hour(self):
    #     temporal = TemporalDistribution(
    #         datetime(year=2016, month=01, day=01, hour=0, minute=0, second=0), 'hourly', 1, 1,
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Monthly.csv', 'M001',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Daily.csv', 'D000',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Hourly.csv', 'H000',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/tz_world_country_iso3166.csv',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/auxiliar_files/global_1.0_1.40625')
    #     self.assertEqual(
    #         temporal.calculate_ending_date(),
    #         datetime(year=2016, month=01, day=01, hour=0, minute=0, second=0))
    #
    # def testing_calculate_ending_date_24hours(self):
    #     temporal = TemporalDistribution(
    #         datetime(year=2016, month=01, day=01, hour=0, minute=0, second=0), 'hourly', 24, 1,
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Monthly.csv', 'M001',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Daily.csv', 'D000',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Hourly.csv', 'H000',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/tz_world_country_iso3166.csv',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/auxiliar_files/global_1.0_1.40625')
    #     self.assertEqual(
    #         temporal.calculate_ending_date(),
    #         datetime(year=2016, month=01, day=01, hour=23, minute=0, second=0))
    #
    # def testing_calculate_ending_date_3hour_each2(self):
    #     temporal = TemporalDistribution(
    #         datetime(year=2016, month=01, day=01, hour=0, minute=0, second=0), 'hourly', 3, 2,
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Monthly.csv', 'M001',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Daily.csv', 'D000',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Hourly.csv', 'H000',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/tz_world_country_iso3166.csv',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/auxiliar_files/global_1.0_1.40625')
    #     self.assertEqual(
    #         temporal.calculate_ending_date(),
    #         datetime(year=2016, month=01, day=01, hour=4, minute=0, second=0))
    #
    # def testing_def_calculate_timedelta_3hour_each2(self):
    #     temporal = TemporalDistribution(
    #         datetime(year=2016, month=01, day=01, hour=0, minute=0, second=0), 'hourly', 3, 2,
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Monthly.csv', 'M001',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Daily.csv', 'D000',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Hourly.csv', 'H000',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/tz_world_country_iso3166.csv',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/auxiliar_files/global_1.0_1.40625')
    #     self.assertEqual(
    #         temporal.calculate_timedelta(datetime(year=2016, month=01, day=01, hour=0, minute=0, second=0)),
    #         timedelta(hours=2))
    #
    # def testing_def_calculate_timedelta_month(self):
    #     temporal = TemporalDistribution(
    #         datetime(year=2017, month=02, day=01, hour=0, minute=0, second=0), 'monthly', 1, 1,
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Monthly.csv', 'M001',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Daily.csv', 'D000',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Hourly.csv', 'H000',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/tz_world_country_iso3166.csv',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/auxiliar_files/global_1.0_1.40625')
    #     self.assertEqual(
    #         temporal.calculate_timedelta(datetime(year=2017, month=02, day=01, hour=0, minute=0, second=0)),
    #         timedelta(hours=24*28))
    #
    # def testing_def_calculate_timedelta_month_leapyear(self):
    #     temporal = TemporalDistribution(
    #         datetime(year=2016, month=02, day=01, hour=0, minute=0, second=0), 'monthly', 1, 1,
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Monthly.csv', 'M001',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Daily.csv', 'D000',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Hourly.csv', 'H000',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/tz_world_country_iso3166.csv',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/auxiliar_files/global_1.0_1.40625')
    #     self.assertEqual(
    #         temporal.calculate_timedelta(datetime(year=2016, month=02, day=01, hour=0, minute=0, second=0)),
    #         timedelta(hours=24*29))
    #
    # def testing_get_tz_from_id(self):
    #     temporal = TemporalDistribution(
    #         datetime(year=2016, month=01, day=01), 'monthly', 48, 1,
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Monthly.csv', 'M001',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Daily.csv', 'D000',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Hourly.csv', 'H000',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/tz_world_country_iso3166.csv',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/auxiliar_files/global_1.0_1.40625')
    #
    #     self.assertEqual(
    #         temporal.get_tz_from_id(309),
    #         "Europe/Andorra")
    #
    # def testing_get_id_from_tz(self):
    #     temporal = TemporalDistribution(
    #         datetime(year=2016, month=01, day=01), 'monthly', 48, 1,
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Monthly.csv', 'M001',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Daily.csv', 'D000',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Hourly.csv', 'H000',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/tz_world_country_iso3166.csv',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/auxiliar_files/global_1.0_1.40625')
    #
    #     self.assertEqual(
    #         temporal.get_id_from_tz("Europe/Andorra"),
    #         309)
    #
    # def testing_parse_tz(self):
    #     temporal = TemporalDistribution(
    #         datetime(year=2016, month=01, day=01), 'monthly', 48, 1,
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Monthly.csv', 'M001',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Daily.csv', 'D000',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Hourly.csv', 'H000',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/tz_world_country_iso3166.csv',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/auxiliar_files/global_1.0_1.40625')
    #
    #     self.assertEqual(
    #         temporal.parse_tz("America/Fort_Nelson"),
    #         'America/Vancouver')
    #
    # def testing_find_closest_timezone_BCN(self):
    #     temporal = TemporalDistribution(
    #         datetime(year=2016, month=01, day=01), 'monthly', 48, 1,
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Monthly.csv', 'M001',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Daily.csv', 'D000',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Hourly.csv', 'H000',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/tz_world_country_iso3166.csv',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/auxiliar_files/global_1.0_1.40625')
    #
    #     self.assertEqual(
    #         temporal.find_closest_timezone(41.390205, 2.154007),
    #         'Europe/Madrid')
    #
    # def testing_find_closest_timezone_MEX(self):
    #     temporal = TemporalDistribution(
    #         datetime(year=2016, month=01, day=01), 'monthly', 48, 1,
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Monthly.csv', 'M001',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Daily.csv', 'D000',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Hourly.csv', 'H000',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/tz_world_country_iso3166.csv',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/auxiliar_files/global_1.0_1.40625')
    #
    #     self.assertEqual(
    #         temporal.find_closest_timezone(19.451054, -99.125519),
    #         "America/Mexico_City")
    #
    # def testing_find_closest_timezone_Kuwait(self):
    #     temporal = TemporalDistribution(
    #         datetime(year=2016, month=01, day=01), 'monthly', 48, 1,
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Monthly.csv', 'M001',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Daily.csv', 'D000',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Hourly.csv', 'H000',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/tz_world_country_iso3166.csv',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/auxiliar_files/global_1.0_1.40625')
    #
    #     self.assertEqual(
    #         temporal.find_closest_timezone(29.378586, 47.990341),
    #         "Asia/Kuwait")
    #
    # def testing_find_closest_timezone_Shanghai(self):
    #     temporal = TemporalDistribution(
    #         datetime(year=2016, month=01, day=01), 'monthly', 48, 1,
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Monthly.csv', 'M001',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Daily.csv', 'D000',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Hourly.csv', 'H000',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/tz_world_country_iso3166.csv',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/auxiliar_files/global_1.0_1.40625')
    #
    #     self.assertEqual(
    #         temporal.find_closest_timezone(31.267401, 121.522179),
    #         "Asia/Shanghai")
    #
    # def testing_create_netcdf_timezones(self):
    #     import numpy as np
    #     from hermes_gr.modules.grids.grid import Grid
    #     from hermes_gr.tools.netcdf_tools import extract_vars
    #
    #     aux_path = '/home/Earth/ctena/Models/HERMES/IN/data/auxiliar_files/testing'
    #     if not os.path.exists(aux_path):
    #         os.makedirs(aux_path)
    #
    #     grid = Grid('global', aux_path)
    #     grid.center_latitudes = np.array([[41.390205, 19.451054], [29.378586, 31.267401]])
    #     grid.center_longitudes = np.array([[2.154007, -99.125519], [47.990341, 121.522179]])
    #
    #     temporal = TemporalDistribution(
    #         datetime(year=2016, month=01, day=01), 'monthly', 48, 1,
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Monthly.csv', 'M001',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Daily.csv', 'D000',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Hourly.csv', 'H000',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/tz_world_country_iso3166.csv',
    #         aux_path)
    #
    #     self.assertTrue(temporal.create_netcdf_timezones(grid))
    #
    #     [timezones] = extract_vars(temporal.netcdf_timezones, ['timezone_id'])
    #     timezones = list(timezones['data'][0, :].astype(int).flatten())
    #
    #     self.assertEqual(timezones,
    #                      [335, 147, 247, 268])
    #
    # def testing_calculate_timezones(self):
    #     self.testing_create_netcdf_timezones()
    #
    #     temporal = TemporalDistribution(
    #         datetime(year=2016, month=01, day=01), 'monthly', 48, 1,
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Monthly.csv', 'M001',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Daily.csv', 'D000',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Hourly.csv', 'H000',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/tz_world_country_iso3166.csv',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/auxiliar_files/testing')
    #     self.assertEqual(temporal.calculate_timezones().tolist(),
    #                      [['Europe/Madrid', "America/Mexico_City"], ["Asia/Kuwait", "Asia/Shanghai"]])
    #
    # def testing_calculate_2d_temporal_factors(self):
    #     self.testing_create_netcdf_timezones()
    #
    #     temporal = TemporalDistribution(
    #         datetime(year=2016, month=01, day=01), 'monthly', 48, 1,
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Monthly.csv', 'M001',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Daily.csv', 'D000',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Hourly.csv', 'H000',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/tz_world_country_iso3166.csv',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/auxiliar_files/testing')
    #     timezones = temporal.calculate_timezones()
    #
    #     temporal.monthly_profile = {1: 1.,
    #                                 2: 1.,
    #                                 3: 1.,
    #                                 4: 1.,
    #                                 5: 1.,
    #                                 6: 1.,
    #                                 7: 1.,
    #                                 8: 1.,
    #                                 9: 1.,
    #                                 10: 1.,
    #                                 11: 1.,
    #                                 12: 1.}
    #     temporal.daily_profile_id = {0: 1.,
    #                               1: 1.,
    #                               2: 1.,
    #                               3: 1.,
    #                               4: 1.,
    #                               5: 1.,
    #                               6: 1.}
    #     temporal.hourly_profile = {0: 1.,
    #                                1: 1.,
    #                                2: 1.,
    #                                3: 1.,
    #                                4: 1.,
    #                                5: 1.,
    #                                6: 1.,
    #                                7: 1.,
    #                                8: 1.,
    #                                9: 1.,
    #                                10: 1.,
    #                                11: 1.,
    #                                12: 1.,
    #                                13: 20.,
    #                                14: 1.,
    #                                15: 1.,
    #                                16: 1.,
    #                                17: 1.,
    #                                18: 1.,
    #                                19: 1.,
    #                                20: 1.,
    #                                21: 1.,
    #                                22: 1.,
    #                                23: 1.}
    #
    #    self.assertEqual(
    #        temporal.calculate_2d_temporal_factors(
    #            datetime(year=2017, month=6, day=23, hour=11, minute=0, second=0), timezones).tolist(),
    #        [[20., 1.], [1., 1.]])
    #
    # def testing_do_temporal(self):
    #     import numpy as np
    #     from hermes_gr.modules.grids.grid import Grid
    #     self.testing_create_netcdf_timezones()
    #
    #     aux_path = '/home/Earth/ctena/Models/HERMES/IN/data/auxiliar_files/testing'
    #
    #     temporal = TemporalDistribution(
    #         datetime(year=2017, month=6, day=23, hour=11, minute=0, second=0), 'hourly', 1, 1,
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Monthly.csv', 'M001',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Daily.csv', 'D000',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Hourly.csv', 'H000',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/tz_world_country_iso3166.csv',
    #         aux_path)
    #     temporal.monthly_profile = {1: 1.,
    #                                 2: 1.,
    #                                 3: 1.,
    #                                 4: 1.,
    #                                 5: 1.,
    #                                 6: 1.,
    #                                 7: 1.,
    #                                 8: 1.,
    #                                 9: 1.,
    #                                 10: 1.,
    #                                 11: 1.,
    #                                 12: 1.}
    #     temporal.daily_profile_id = {0: 1.,
    #                               1: 1.,
    #                               2: 1.,
    #                               3: 1.,
    #                               4: 1.,
    #                               5: 1.,
    #                               6: 1.}
    #     temporal.hourly_profile = {0: 1.,
    #                                1: 1.,
    #                                2: 1.,
    #                                3: 1.,
    #                                4: 1.,
    #                                5: 1.,
    #                                6: 1.,
    #                                7: 1.,
    #                                8: 1.,
    #                                9: 1.,
    #                                10: 1.,
    #                                11: 1.,
    #                                12: 1.,
    #                                13: 20.,
    #                                14: 1.,
    #                                15: 1.,
    #                                16: 1.,
    #                                17: 1.,
    #                                18: 1.,
    #                                19: 1.,
    #                                20: 1.,
    #                                21: 1.,
    #                                22: 1.,
    #                                23: 1.}
    #
    #     grid = Grid('global', aux_path)
    #     grid.center_latitudes = np.array([[41.390205, 19.451054], [29.378586, 31.267401]])
    #     grid.center_longitudes = np.array([[2.154007, -99.125519], [47.990341, 121.522179]])
    #     data_in = [{'data': np.array([[10., 10.], [10., 10.]])}]
    #     # data_out = [{'data': np.array([[200., 10.], [10., 10.]])}]
    #     data_out = temporal.do_temporal(data_in, grid)
    #
    #     self.assertEqual(data_out[0]['data'].tolist(), [[[200., 10.], [10., 10.]]])
    #
    # def testing_calculate_weekdays_no_leap_year(self):
    #     from datetime import datetime
    #     temporal = TemporalDistribution(
    #         datetime(year=2016, month=01, day=01), 'monthly', 48, 1,
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Monthly.csv', 'M001',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Daily.csv', 'D000',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Hourly.csv', 'H000',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/tz_world_country_iso3166.csv',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/auxiliar_files/testing')
    #     self.assertEqual(temporal.calculate_weekdays(datetime(year=2017, month=02, day=1)),
    #                      {0: 4, 1: 4, 2: 4, 3: 4, 4: 4, 5: 4, 6: 4})
    #
    # def testing_calculate_weekdays_leap_year(self):
    #     from datetime import datetime
    #     temporal = TemporalDistribution(
    #         datetime(year=2016, month=01, day=01), 'monthly', 48, 1,
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Monthly.csv', 'M001',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Daily.csv', 'D000',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Hourly.csv', 'H000',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/tz_world_country_iso3166.csv',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/auxiliar_files/testing')
    #     self.assertEqual(temporal.calculate_weekdays(datetime(year=2016, month=02, day=1)),
    #                      {0: 5, 1: 4, 2: 4, 3: 4, 4: 4, 5: 4, 6: 4})
    #
    # def testing_calculate_weekdays_factors_full_month(self):
    #     from datetime import datetime
    #     temporal = TemporalDistribution(
    #         datetime(year=2016, month=01, day=01), 'monthly', 48, 1,
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Monthly.csv', 'M001',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Daily.csv', 'D000',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Hourly.csv', 'H000',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/tz_world_country_iso3166.csv',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/auxiliar_files/testing')
    #
    #     self.assertEqual(round(temporal.calculate_weekday_factor(
    #         {0: 0.8, 1: 1.2, 2: 0.5, 3: 1.5, 4: 0.9, 5: 0.9, 6: 1.2}, {0: 5, 1: 4, 2: 4, 3: 4, 4: 4, 5: 4, 6: 4}), 5),
    #         round(0.2/29, 5))
    #
    # def testing_calculate_rebalance_factor(self):
    #     from datetime import datetime
    #     temporal = TemporalDistribution(
    #         datetime(year=2016, month=01, day=01), 'monthly', 48, 1,
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Monthly.csv', 'M001',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Daily.csv', 'D000',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Hourly.csv', 'H000',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/tz_world_country_iso3166.csv',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/auxiliar_files/testing')
    #
    #     self.assertEqual(round(temporal.calculate_rebalance_factor(
    #         {0: 0.8, 1: 1.2, 2: 0.5, 3: 1.5, 4: 0.9, 5: 0.9, 6: 1.2}, datetime(year=2016, month=02, day=1)), 5),
    #         round(0.2/29, 5))

    # def testing_get_temporal_daily_profile(self):
    #     from datetime import datetime
    #     from calendar import monthrange
    #     date = datetime(year=2016, month=02, day=1)
    #     temporal = TemporalDistribution(
    #         date, 'monthly', 48, 1,
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Monthly.csv', 'M001',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Daily.csv', 'D000',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/TemporalProfile_Hourly.csv', 'H000',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/profiles/tz_world_country_iso3166.csv',
    #         '/home/Earth/ctena/Models/HERMES/IN/data/auxiliar_files/testing')
    #
    #     print temporal.get_temporal_weekly_profile(date)
