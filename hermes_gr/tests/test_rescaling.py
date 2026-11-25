#!/usr/bin/env python

# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import os
import tempfile
import unittest
from calendar import monthrange
from datetime import datetime
from unittest.mock import Mock, patch

try:
    import hermes_gr
    import numpy as np
    import pandas as pd

    from hermes_gr.modules.rescaling.rescaling import (
        EI_COLUMN,
        FACTORS_FILE_TEMPLATE,
        INVENTORY_TOTALS_FILE_TEMPLATE,
        OTHERS_REGION_ID,
        REGION_ID_COLUMN,
        REGION_MASK_VAR,
        SECONDS_PER_YEAR,
        SECTOR_COLUMN,
        UNITS_COLUMN,
        ReScaling,
    )

    RESCALING_IMPORT_ERROR = None
except ImportError as import_error:
    RESCALING_IMPORT_ERROR = import_error


class DummyComm(object):
    def gather(self, value, root=0):
        return [value]

    def Barrier(self):
        return None

    def bcast(self, value, root=0):
        return value

    def Allreduce(self, sendbuf, recvbuf, op=None):
        recvbuf[...] = sendbuf
        return None


class DummyGrid(object):
    def __init__(self):
        self.cell_measures = {
            "cell_area": {
                "data": np.array(
                    [
                        [2.0, 2.0],
                        [2.0, 2.0],
                    ]
                )
            }
        }


class DummyEmissions(object):
    def __init__(self):
        self.variables = {
            "co": {
                "data": np.array(
                    [
                        [
                            [
                                [1.0, 1.0],
                                [1.0, 1.0],
                            ]
                        ]
                    ]
                )
            },
            "nox_no2": {
                "data": np.array(
                    [
                        [1.0, 2.0],
                        [3.0, 4.0],
                    ]
                )
            },
        }


@unittest.skipIf(
    RESCALING_IMPORT_ERROR is not None,
    f"ReScaling dependencies missing: {RESCALING_IMPORT_ERROR}",
)
class TestReScaling(unittest.TestCase):
    def setUp(self):
        self.rescaling = ReScaling.__new__(ReScaling)
        self.rescaling.master = True
        self.rescaling.comm = DummyComm()
        self.rescaling.apply_to_all_emissions = False
        self.rescaling.selected_regions = None
        self.rescaling.allow_partial_regions = False
        self.rescaling.validated_domain_regions = set()
        self.rescaling.domain_geometry = None
        self.rescaling.domain_geometry_initialized = False
        self.rescaling.domain_region_geometries = None

    def test_validate_domain_rejects_partial_outside_and_detached_region_parts(self):
        import geopandas as gpd
        from shapely.geometry import Polygon, box

        # A diamond domain rejects regions within its bbox but outside its actual footprint.
        domain = Polygon([(0, -2), (2, 0), (0, 2), (-2, 0)])
        grid = Mock()
        grid.shapefile = gpd.GeoDataFrame(geometry=[domain], crs="EPSG:4326")
        self.rescaling.grid = grid
        self.rescaling.rescaling_geometries = "/regions.shp"
        self.rescaling.rescaling_geometries_attribute = "NAME"
        geometries = gpd.GeoDataFrame(
            {"NAME": ["Inside", "Partial", "Outside", "Detached", "Detached", "Boundary"]},
            geometry=[box(-0.2, -0.2, 0.2, 0.2), box(0.5, 0.5, 1.5, 1.5), box(3, 3, 4, 4),
                      box(-0.1, -0.1, 0.1, 0.1), box(4, 4, 5, 5), domain], crs="EPSG:4326",
        )
        # Validation is required even when an auxiliary mask was already loaded.
        self.rescaling.region_mask = Mock()
        with patch("hermes_gr.modules.rescaling.rescaling.gpd.read_file", return_value=geometries) as read:
            with self.assertRaisesRegex(ValueError, r"Of the 4 selected.*3.*\[Detached, Outside, Partial\]"):
                self.rescaling._validate_regions_in_domain(["Inside", "Partial", "Outside", "Detached"])
            self.assertEqual(self.rescaling.validated_domain_regions, set())
            self.rescaling._validate_regions_in_domain(["Inside", "Boundary"])
            self.rescaling._validate_regions_in_domain(["Inside"])
            read.assert_called_once_with("/regions.shp")
            with self.assertRaisesRegex(ValueError, "missing from the geometry file: Missing"):
                self.rescaling._validate_regions_in_domain(["Missing"])
        self.assertEqual(self.rescaling.validated_domain_regions, {"Inside", "Boundary"})

    def test_domain_validation_error_is_received_by_non_master(self):
        self.rescaling.master = False
        self.rescaling.domain_geometry_initialized = True
        self.rescaling.comm = Mock()
        message = "ERROR: Of the 12 selected rescaling regions, 10 are not completely inside the simulation domain."
        self.rescaling.comm.bcast.return_value = message
        with self.assertRaisesRegex(ValueError, "12.*10"):
            self.rescaling._validate_regions_in_domain(["Milano"])
        self.assertEqual(self.rescaling.validated_domain_regions, set())

    def test_allow_partial_regions_warns_once_and_preserves_geometry_errors(self):
        import geopandas as gpd
        from shapely.geometry import box

        self.rescaling.allow_partial_regions = True
        self.rescaling.grid = Mock()
        self.rescaling.grid.shapefile = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
        self.rescaling.rescaling_geometries = "/regions.shp"
        self.rescaling.rescaling_geometries_attribute = "NAME"
        geometries = gpd.GeoDataFrame(
            {"NAME": ["Partial", "Outside", "Invalid"]},
            geometry=[box(0, 0, 2, 2), box(3, 3, 4, 4), None], crs="EPSG:4326",
        )
        with patch("hermes_gr.modules.rescaling.rescaling.gpd.read_file", return_value=geometries), \
                patch("hermes_gr.modules.rescaling.rescaling.log_message") as log:
            self.rescaling._validate_regions_in_domain(["Partial", "Outside"])
            self.rescaling._validate_regions_in_domain(["Partial", "Outside"])
            log.assert_called_once()
            self.assertEqual(log.call_args.kwargs["level"], 7)
            self.assertIn("Of the 2 selected rescaling regions, 2", log.call_args.args[0])
            self.assertIn("[Outside, Partial]", log.call_args.args[0])
            self.assertIn("full regional local totals", log.call_args.args[0])
            with self.assertRaisesRegex(ValueError, "missing from the geometry file: Missing"):
                self.rescaling._validate_regions_in_domain(["Missing"])
            with self.assertRaisesRegex(ValueError, "missing, empty or invalid geometries"):
                self.rescaling._validate_regions_in_domain(["Invalid"])
        self.assertEqual(self.rescaling.validated_domain_regions, {"Partial", "Outside"})

    def test_parse_regions_supports_all_and_comma_separated_names(self):
        for value in (None, "", "  ", "ALL", "all"):
            with self.subTest(value=value):
                self.assertIsNone(ReScaling._parse_regions(value))
        self.assertEqual(
            ReScaling._parse_regions(" Milano, Monza Brianza, Milano "), ["Milano", "Monza Brianza"],
        )
        for value in ("ALL, Milano", "Milano,", "OTHERS"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    ReScaling._parse_regions(value)

    def test_region_selection_filters_sectors_without_changing_mask_regions(self):
        totals = pd.DataFrame(
            {"co": [10.0, 20.0, 30.0]},
            index=pd.MultiIndex.from_tuples(
                [("Milano", "CAMS", "gnfr_A"), ("Lecco", "CAMS", "gnfr_A"),
                 ("Lecco", "CAMS", "gnfr_B")],
                names=[REGION_ID_COLUMN, EI_COLUMN, SECTOR_COLUMN],
            ),
        )
        self.rescaling.region_totals = totals
        self.rescaling.selected_regions = ["Milano"]
        self.assertTrue(self.rescaling.do_i_have_to_rescale("CAMS", "gnfr_A"))
        self.assertFalse(self.rescaling.do_i_have_to_rescale("CAMS", "gnfr_B"))
        selected = self.rescaling._get_inventory_sector_totals("CAMS", "gnfr_A")
        self.assertEqual(list(selected.index), ["Milano"])
        self.assertEqual(self.rescaling._get_region_ids(totals), ["Lecco", "Milano"])
        self.rescaling.selected_regions = ["Milano", "Lecco"]
        self.assertTrue(self.rescaling.do_i_have_to_rescale("CAMS", "gnfr_B"))
        self.rescaling.selected_regions = ["Unknown"]
        with self.assertRaisesRegex(ValueError, "absent.*Unknown"):
            self.rescaling.do_i_have_to_rescale("CAMS", "gnfr_A")

    def test_ensure_region_mask_rejects_missing_regions_on_first_load_and_reuse(self):
        mask = type("DummyMask", (), {
            "variables": {REGION_MASK_VAR: {"region_ids": "Milano;OTHERS"}},
        })()
        self.rescaling.auxiliary_files_path = "/auxiliary"
        self.rescaling.rescaling_geometries = "/regions.shp"
        self.rescaling.rescaling_geometries_attribute = "NAME_2"

        for first_load in (True, False):
            with self.subTest(first_load=first_load):
                self.rescaling.region_mask = None if first_load else mask
                with patch.object(self.rescaling, "_obtain_region_mask", return_value=mask) as obtain_mask:
                    with self.assertRaises(ValueError) as error:
                        self.rescaling._ensure_region_mask(["Milano", "Bergamo", "Brescia"])

                message = str(error.exception)
                self.assertIn("Missing regions: Bergamo, Brescia", message)
                self.assertIn("/auxiliary/ReScaling_RegionsMask.nc", message)
                self.assertIn("all required regions", message)
                self.assertIn("other auxiliary files can be reused", message)
                self.assertEqual(obtain_mask.call_count, int(first_load))

    def test_ensure_region_mask_accepts_subset_on_first_load_and_reuse(self):
        mask = type("DummyMask", (), {
            "variables": {REGION_MASK_VAR: {"region_ids": "Bergamo;Milano;OTHERS"}},
        })()
        self.rescaling.auxiliary_files_path = "/auxiliary"
        self.rescaling.rescaling_geometries = "/regions.shp"
        self.rescaling.rescaling_geometries_attribute = "NAME_2"

        for first_load in (True, False):
            with self.subTest(first_load=first_load):
                self.rescaling.region_mask = None if first_load else mask
                with patch.object(self.rescaling, "_obtain_region_mask", return_value=mask):
                    self.assertIsNone(self.rescaling._ensure_region_mask(["Milano"]))
                self.assertIs(self.rescaling.region_mask, mask)

    def test_calculate_rescaling_factors_builds_shared_mask_for_all_sector_regions(self):
        region_totals = pd.DataFrame(
            {"co": [20.0, 30.0, 100.0]},
            index=pd.MultiIndex.from_tuples(
                [("Milano", "CAMS", "gnfr_A"), ("Lecco", "CAMS", "gnfr_B"),
                 (OTHERS_REGION_ID, "CAMS", "gnfr_B")],
                names=[REGION_ID_COLUMN, EI_COLUMN, SECTOR_COLUMN],
            ),
        )
        inventory_totals = pd.DataFrame({"co": [10.0, 10.0]}, index=["Milano", "Lecco"])
        self.rescaling.apply_to_all_emissions = False
        self.rescaling.region_mask = None
        self.rescaling.auxiliary_files_path = "/auxiliary"
        self.rescaling.rescaling_geometries = "/regions.shp"
        self.rescaling.rescaling_geometries_attribute = "NAME_2"

        def obtain_mask(auxiliary_path, geometries_path, attribute, region_ids):
            return type("DummyMask", (), {
                "variables": {REGION_MASK_VAR: {"region_ids": ";".join(region_ids + [OTHERS_REGION_ID])}},
            })()

        with patch.object(hermes_gr, "DEBUG", False), \
                patch.object(self.rescaling, "_validate_regions_in_domain") as validate_domain, \
                patch.object(self.rescaling, "_get_region_totals_for_period", return_value=region_totals), \
                patch.object(self.rescaling, "_calculate_inventory_totals", return_value=inventory_totals), \
                patch.object(self.rescaling, "_obtain_region_mask", side_effect=obtain_mask) as obtain, \
                patch("hermes_gr.modules.rescaling.rescaling.log_message"):
            factors_a = self.rescaling.calculate_rescaling_factors(
                DummyEmissions(), "CAMS", "gnfr_A", 2015, "yearly", datetime(2026, 1, 1),
                "/auxiliary", "/output",
            )
            factors_b = self.rescaling.calculate_rescaling_factors(
                DummyEmissions(), "CAMS", "gnfr_B", 2015, "yearly", datetime(2026, 1, 1),
                "/auxiliary", "/output",
            )

        obtain.assert_called_once_with("/auxiliary", "/regions.shp", "NAME_2", ["Lecco", "Milano"])
        self.assertEqual(factors_a, {"Milano": {"co": 2.0}})
        self.assertEqual(factors_b, {"Lecco": {"co": 3.0}})
        self.assertEqual(validate_domain.call_args_list[0].args, (["Lecco", "Milano"],))

    def test_read_region_totals_converts_units_to_kg_year(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as csv_file:
            csv_file.write("Region,ei,sector,units,co,nox_no2\n")
            csv_file.write("Barcelona,CAMS,gnfr_A,t/year,2,3\n")
            csv_file.write("Girona,CAMS,gnfr_A,kg/year,4,5\n")
            csv_path = csv_file.name

        try:
            region_totals = self.rescaling._read_region_totals(csv_path)
        finally:
            os.remove(csv_path)

        self.assertEqual(region_totals.index.names, [REGION_ID_COLUMN, EI_COLUMN, SECTOR_COLUMN])
        self.assertEqual(region_totals.loc[("Barcelona", "CAMS", "gnfr_A"), "co"], 2000.0)
        self.assertEqual(region_totals.loc[("Barcelona", "CAMS", "gnfr_A"), "nox_no2"], 3000.0)
        self.assertEqual(region_totals.loc[("Girona", "CAMS", "gnfr_A"), "co"], 4.0)

    def test_read_region_totals_converts_monthly_units_to_kg_month(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as csv_file:
            csv_file.write("Region,ei,sector,units,co,nox_no2\n")
            csv_file.write("Barcelona,HTAP,energy,t/month,2,3\n")
            csv_file.write("Girona,HTAP,energy,kg/month,4,5\n")
            csv_path = csv_file.name

        try:
            region_totals = self.rescaling._read_region_totals(csv_path, input_frequency="monthly")
        finally:
            os.remove(csv_path)

        self.assertEqual(region_totals.loc[("Barcelona", "HTAP", "energy"), "co"], 2000.0)
        self.assertEqual(region_totals.loc[("Barcelona", "HTAP", "energy"), "nox_no2"], 3000.0)
        self.assertEqual(region_totals.loc[("Girona", "HTAP", "energy"), "co"], 4.0)

    def test_read_region_totals_rejects_yearly_units_for_monthly_inventory(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as csv_file:
            csv_file.write("Region,ei,sector,units,co\n")
            csv_file.write("Barcelona,HTAP,energy,t/year,2\n")
            csv_path = csv_file.name

        try:
            with self.assertRaisesRegex(ValueError, "Incompatible regional emission rescaling units"):
                self.rescaling._read_region_totals(csv_path, input_frequency="monthly")
        finally:
            os.remove(csv_path)

    def test_read_region_totals_rejects_monthly_units_for_yearly_inventory(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as csv_file:
            csv_file.write("Region,ei,sector,units,co\n")
            csv_file.write("Barcelona,CAMS,gnfr_A,t/month,2\n")
            csv_path = csv_file.name

        try:
            with self.assertRaisesRegex(ValueError, "Incompatible regional emission rescaling units"):
                self.rescaling._read_region_totals(csv_path, input_frequency="yearly")
        finally:
            os.remove(csv_path)

    def test_inventory_sector_helpers_exclude_others(self):
        self.rescaling.region_totals = pd.DataFrame(
            {
                "co": [10.0, 20.0, 30.0],
            },
            index=pd.MultiIndex.from_tuples(
                [
                    ("Barcelona", "CAMS", "gnfr_A"),
                    ("Girona", "CAMS", "gnfr_A"),
                    (OTHERS_REGION_ID, "CAMS", "gnfr_A"),
                ],
                names=[REGION_ID_COLUMN, EI_COLUMN, SECTOR_COLUMN],
            ),
        )

        self.assertEqual(self.rescaling._get_region_ids(), ["Barcelona", "Girona"])
        self.assertTrue(self.rescaling.do_i_have_to_rescale("CAMS", "gnfr_A"))
        self.assertFalse(self.rescaling.do_i_have_to_rescale("CAMS", "gnfr_B"))
        local_totals = self.rescaling._get_inventory_sector_totals("CAMS", "gnfr_A")
        self.assertEqual(list(local_totals.index), ["Barcelona", "Girona"])

    def test_calculate_inventory_totals_uses_mask_cell_area_and_seconds_per_year(self):
        self.rescaling.comm = DummyComm()
        self.rescaling.grid = DummyGrid()
        self.rescaling.region_mask = type(
            "DummyMask",
            (),
            {
                "variables": {
                    REGION_MASK_VAR: {
                        "region_ids": f"Barcelona;{OTHERS_REGION_ID}",
                        "data": np.array(
                            [
                                [
                                    [
                                        [1, 0],
                                        [1, 0],
                                    ],
                                    [
                                        [0, 1],
                                        [0, 1],
                                    ],
                                ]
                            ],
                            dtype=np.int8,
                        ),
                    }
                }
            },
        )()

        inventory_totals = self.rescaling._calculate_inventory_totals(DummyEmissions())

        self.assertEqual(list(inventory_totals.index), ["Barcelona", OTHERS_REGION_ID])
        self.assertEqual(inventory_totals.loc["Barcelona", "co"], 4.0 * SECONDS_PER_YEAR)
        self.assertEqual(inventory_totals.loc[OTHERS_REGION_ID, "co"], 4.0 * SECONDS_PER_YEAR)
        self.assertEqual(inventory_totals.loc["Barcelona", "nox_no2"], 8.0 * SECONDS_PER_YEAR)
        self.assertEqual(inventory_totals.loc[OTHERS_REGION_ID, "nox_no2"], 12.0 * SECONDS_PER_YEAR)

    def test_calculate_inventory_totals_can_use_month_seconds(self):
        self.rescaling.comm = DummyComm()
        self.rescaling.grid = DummyGrid()
        self.rescaling.region_mask = type(
            "DummyMask",
            (),
            {
                "variables": {
                    REGION_MASK_VAR: {
                        "region_ids": "Barcelona",
                        "data": np.array(
                            [
                                [
                                    [
                                        [1, 1],
                                        [1, 1],
                                    ],
                                ]
                            ],
                            dtype=np.int8,
                        ),
                    }
                }
            },
        )()
        date = datetime(2025, 6, 1)
        seconds_per_month = monthrange(2025, 6)[1] * 24 * 60 * 60

        inventory_totals = self.rescaling._calculate_inventory_totals(
            DummyEmissions(),
            seconds_per_period=ReScaling._get_inventory_total_seconds(2025, "monthly", date),
        )

        self.assertEqual(inventory_totals.loc["Barcelona", "co"], 8.0 * seconds_per_month)
        self.assertEqual(inventory_totals.loc["Barcelona", "nox_no2"], 20.0 * seconds_per_month)

    def test_calculate_factors_from_totals_excludes_others_and_handles_missing_values(self):
        inventory_totals = pd.DataFrame(
            {
                "co": [10.0, 30.0, 50.0],
                "nox_no2": [20.0, 30.0, 60.0],
                "so2": [40.0, 0.0, 70.0],
            },
            index=["Barcelona", "Girona", OTHERS_REGION_ID],
        )
        local_totals = pd.DataFrame(
            {
                "co": [20.0, 0.0, 40.0],
                "nox_no2": [np.nan, 90.0, 100.0],
            },
            index=["Barcelona", "Girona", OTHERS_REGION_ID],
        )

        factors = ReScaling._calculate_factors_from_totals(inventory_totals, local_totals)

        self.assertNotIn(OTHERS_REGION_ID, factors)
        self.assertEqual(factors["Barcelona"]["co"], 2.0)
        self.assertIsNone(factors["Barcelona"]["nox_no2"])
        self.assertIsNone(factors["Barcelona"]["so2"])
        self.assertEqual(factors["Girona"]["co"], 0)
        self.assertEqual(factors["Girona"]["nox_no2"], 3.0)
        self.assertIsNone(factors["Girona"]["so2"])

    def test_calculate_factors_warns_when_local_total_exists_but_inventory_is_zero(self):
        inventory_totals = pd.DataFrame(
            {
                "co": [0.0, 0.0],
            },
            index=["Barcelona", "Girona"],
        )
        local_totals = pd.DataFrame(
            {
                "co": [10.0, 0.0],
            },
            index=["Barcelona", "Girona"],
        )

        with patch("hermes_gr.modules.rescaling.rescaling.log_message") as mocked_log_message:
            factors = ReScaling._calculate_factors_from_totals(
                inventory_totals,
                local_totals,
                inventory_name="CAMS-REG-AP_v221",
                sector="gnfr_H",
            )

        self.assertEqual(factors["Barcelona"]["co"], 0)
        self.assertEqual(factors["Girona"]["co"], 0)
        mocked_log_message.assert_called_once()
        self.assertIn("inventory=CAMS-REG-AP_v221", mocked_log_message.call_args[0][0])
        self.assertIn("sector=gnfr_H", mocked_log_message.call_args[0][0])
        self.assertIn("region_id=Barcelona", mocked_log_message.call_args[0][0])
        self.assertEqual(mocked_log_message.call_args[1]["level"], 7)

    def test_calculate_rescaling_factors_writes_debug_factors_csv(self):
        region_totals = pd.DataFrame(
            {
                "co": [20.0, 0.0, 10.0, 60.0],
                "nox_no2": [np.nan, 90.0, 40.0, 30.0],
            },
            index=pd.MultiIndex.from_tuples(
                [
                    ("Barcelona", "CAMS", "gnfr_A"),
                    ("Girona", "CAMS", "gnfr_A"),
                    ("Barcelona", "CAMS", "gnfr_B"),
                    ("Girona", "CAMS", "gnfr_B"),
                ],
                names=[REGION_ID_COLUMN, EI_COLUMN, SECTOR_COLUMN],
            ),
        )
        inventory_totals = pd.DataFrame(
            {
                "co": [10.0, 30.0],
                "nox_no2": [20.0, 30.0],
            },
            index=["Barcelona", "Girona"],
        )
        date = datetime(2026, 5, 1)
        previous_debug = hermes_gr.DEBUG
        hermes_gr.DEBUG = True

        try:
            with tempfile.TemporaryDirectory() as auxiliary_files_path:
                with tempfile.TemporaryDirectory() as output_path:
                    with patch.object(self.rescaling, "_get_region_totals_for_period", return_value=region_totals):
                        with patch.object(self.rescaling, "_ensure_region_mask"), \
                                patch.object(self.rescaling, "_validate_regions_in_domain"):
                            with patch.object(
                                self.rescaling,
                                "_calculate_inventory_totals",
                                return_value=inventory_totals,
                            ):
                                with patch("hermes_gr.modules.rescaling.rescaling.log_message"):
                                    factors = self.rescaling.calculate_rescaling_factors(
                                        DummyEmissions(),
                                        "CAMS",
                                        "gnfr_A",
                                        2015,
                                        "monthly",
                                        date,
                                        auxiliary_files_path,
                                        output_path,
                                    )
                                    self.rescaling.calculate_rescaling_factors(
                                        DummyEmissions(),
                                        "CAMS",
                                        "gnfr_B",
                                        2015,
                                        "monthly",
                                        date,
                                        auxiliary_files_path,
                                        output_path,
                                    )

                    factors_file_name = FACTORS_FILE_TEMPLATE.format(date_suffix="201505_to_202605")
                    inventory_totals_file_name = INVENTORY_TOTALS_FILE_TEMPLATE.format(date_suffix="201505")
                    auxiliary_factors_path = os.path.join(auxiliary_files_path, factors_file_name)
                    auxiliary_inventory_totals_path = os.path.join(auxiliary_files_path, inventory_totals_file_name)
                    factors_path = os.path.join(output_path, factors_file_name)
                    inventory_totals_path = os.path.join(output_path, inventory_totals_file_name)
                    self.assertFalse(os.path.exists(auxiliary_factors_path))
                    self.assertFalse(os.path.exists(auxiliary_inventory_totals_path))
                    factors_to_write = pd.read_csv(factors_path)
                    inventory_totals_to_write = pd.read_csv(inventory_totals_path)
        finally:
            hermes_gr.DEBUG = previous_debug

        self.assertEqual(factors["Barcelona"]["co"], 2.0)
        self.assertIsNone(factors["Barcelona"]["nox_no2"])
        self.assertEqual(factors["Girona"]["co"], 0.0)
        self.assertEqual(factors["Girona"]["nox_no2"], 3.0)
        self.assertEqual(list(factors_to_write.columns), [REGION_ID_COLUMN, EI_COLUMN, SECTOR_COLUMN, "co", "nox_no2"])
        self.assertEqual(len(factors_to_write), 4)
        factors_to_write = factors_to_write.set_index([REGION_ID_COLUMN, EI_COLUMN, SECTOR_COLUMN])
        self.assertEqual(factors_to_write.loc[("Barcelona", "CAMS", "gnfr_A"), "co"], 2.0)
        self.assertTrue(pd.isna(factors_to_write.loc[("Barcelona", "CAMS", "gnfr_A"), "nox_no2"]))
        self.assertEqual(factors_to_write.loc[("Girona", "CAMS", "gnfr_A"), "co"], 0.0)
        self.assertEqual(factors_to_write.loc[("Girona", "CAMS", "gnfr_A"), "nox_no2"], 3.0)
        self.assertEqual(factors_to_write.loc[("Barcelona", "CAMS", "gnfr_B"), "co"], 1.0)
        self.assertEqual(factors_to_write.loc[("Barcelona", "CAMS", "gnfr_B"), "nox_no2"], 2.0)
        self.assertEqual(factors_to_write.loc[("Girona", "CAMS", "gnfr_B"), "co"], 2.0)
        self.assertEqual(factors_to_write.loc[("Girona", "CAMS", "gnfr_B"), "nox_no2"], 1.0)
        self.assertEqual(
            list(inventory_totals_to_write.columns),
            [REGION_ID_COLUMN, EI_COLUMN, SECTOR_COLUMN, UNITS_COLUMN, "co", "nox_no2"],
        )
        self.assertEqual(len(inventory_totals_to_write), 4)
        inventory_totals_to_write = inventory_totals_to_write.set_index([REGION_ID_COLUMN, EI_COLUMN, SECTOR_COLUMN])
        self.assertEqual(inventory_totals_to_write.loc[("Barcelona", "CAMS", "gnfr_A"), UNITS_COLUMN], "kg/month")
        self.assertEqual(inventory_totals_to_write.loc[("Barcelona", "CAMS", "gnfr_A"), "co"], 10.0)
        self.assertEqual(inventory_totals_to_write.loc[("Girona", "CAMS", "gnfr_B"), "nox_no2"], 30.0)

    def test_apply_factors_to_all_emissions_uses_single_region_factor(self):
        emissions = DummyEmissions()
        self.rescaling.apply_to_all_emissions = True
        factors = {
            "Barcelona": {
                "co": 2.0,
                "nox_no2": None,
            }
        }

        with patch("hermes_gr.modules.rescaling.rescaling.log_message"):
            rescaled_emissions = self.rescaling.apply_factors(emissions, factors)

        self.assertTrue(np.array_equal(rescaled_emissions.variables["co"]["data"], np.ones((1, 1, 2, 2)) * 2.0))
        self.assertTrue(
            np.array_equal(
                rescaled_emissions.variables["nox_no2"]["data"],
                np.array(
                    [
                        [1.0, 2.0],
                        [3.0, 4.0],
                    ]
                ),
            )
        )

    def test_apply_factors_by_region_uses_mask_and_preserves_none_factors(self):
        emissions = DummyEmissions()
        self.rescaling.apply_to_all_emissions = False
        self.rescaling.region_mask = type(
            "DummyMask",
            (),
            {
                "variables": {
                    REGION_MASK_VAR: {
                        "region_ids": f"Barcelona;Girona;{OTHERS_REGION_ID}",
                        "data": np.array(
                            [
                                [
                                    [
                                        [1, 0],
                                        [0, 0],
                                    ],
                                    [
                                        [0, 1],
                                        [0, 0],
                                    ],
                                    [
                                        [0, 0],
                                        [1, 1],
                                    ],
                                ]
                            ],
                            dtype=np.int8,
                        ),
                    }
                }
            },
        )()
        factors = {
            "Barcelona": {
                "co": 2.0,
                "nox_no2": None,
            },
            "Girona": {
                "co": 3.0,
                "nox_no2": 10.0,
            },
        }

        with patch("hermes_gr.modules.rescaling.rescaling.log_message"):
            rescaled_emissions = self.rescaling.apply_factors(emissions, factors)

        expected_co = np.array(
            [
                [
                    [
                        [2.0, 3.0],
                        [1.0, 1.0],
                    ]
                ]
            ]
        )
        expected_nox_no2 = np.array(
            [
                [1.0, 20.0],
                [3.0, 4.0],
            ]
        )
        self.assertTrue(np.array_equal(rescaled_emissions.variables["co"]["data"], expected_co))
        self.assertTrue(np.array_equal(rescaled_emissions.variables["nox_no2"]["data"], expected_nox_no2))

    def test_date_suffix_and_file_name_helpers(self):
        date = datetime(2025, 6, 27, 3)

        self.assertEqual(ReScaling._get_inventory_totals_date_suffix(2015, "yearly", date), "2015")
        self.assertEqual(ReScaling._get_inventory_totals_date_suffix(2015, "monthly", date), "201506")
        self.assertEqual(ReScaling._get_inventory_totals_date_suffix(2015, "daily", date), "20150627")
        self.assertEqual(ReScaling._get_inventory_totals_date_suffix(2015, "hourly", date), "2015062703")
        self.assertEqual(
            ReScaling._get_inventory_totals_path("/output", 2015, "yearly", date),
            "/output/ReScaling_InventoryTotals_2015.csv",
        )

    def test_local_totals_date_suffix_uses_simulation_date_not_reference_year(self):
        date = datetime(2026, 6, 27, 3)

        self.assertEqual(ReScaling._get_local_totals_date_suffix(2015, "yearly", date), "2026")
        self.assertEqual(ReScaling._get_local_totals_date_suffix(2015, "monthly", date), "202606")

    def test_resolve_local_totals_path_replaces_date_placeholder(self):
        date = datetime(2025, 6, 27, 3)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path_template = os.path.join(tmp_dir, "local_totals_<date>.csv")
            expected_path = os.path.join(tmp_dir, "local_totals_202506.csv")
            with open(expected_path, "w"):
                pass

            with patch("hermes_gr.modules.rescaling.rescaling.log_message") as mocked_log_message:
                resolved_path = self.rescaling._resolve_local_totals_path(path_template, 2025, "monthly", date)

        self.assertEqual(resolved_path, expected_path)
        self.assertIn(expected_path, mocked_log_message.call_args[0][0])

    def test_resolve_local_totals_path_without_date_placeholder_warns_and_uses_configured_file(self):
        date = datetime(2025, 6, 27, 3)
        with tempfile.TemporaryDirectory() as tmp_dir:
            fixed_path = os.path.join(tmp_dir, "local_totals.csv")

            with patch("hermes_gr.modules.rescaling.rescaling.log_message") as mocked_log_message:
                resolved_path = self.rescaling._resolve_local_totals_path(fixed_path, 2025, "monthly", date)

        self.assertEqual(resolved_path, fixed_path)
        self.assertIn("does not contain <date>", mocked_log_message.call_args[0][0])
        self.assertIn(fixed_path, mocked_log_message.call_args[0][0])
        self.assertEqual(mocked_log_message.call_args[1]["level"], 7)

    def test_resolve_local_totals_path_falls_back_to_default_with_warning(self):
        date = datetime(2025, 6, 27, 3)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path_template = os.path.join(tmp_dir, "local_totals_<date>.csv")
            default_path = os.path.join(tmp_dir, "local_totals_default.csv")
            with open(default_path, "w"):
                pass

            with patch("hermes_gr.modules.rescaling.rescaling.log_message") as mocked_log_message:
                resolved_path = self.rescaling._resolve_local_totals_path(path_template, 2025, "monthly", date)

        self.assertEqual(resolved_path, default_path)
        self.assertIn("Falling back to default file", mocked_log_message.call_args[0][0])
        self.assertEqual(mocked_log_message.call_args[1]["level"], 7)


if __name__ == "__main__":
    unittest.main()
