# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import numpy as np

    from hermes_gr.modules.masking.masking import Masking

    MASKING_IMPORT_ERROR = None
except ImportError as import_error:
    MASKING_IMPORT_ERROR = import_error


class DummyCountries(object):
    def __init__(self):
        self.lat = {"data": np.zeros((2, 2))}
        self.lon = {"data": np.zeros((2, 2))}
        self.variables = {
            "ISO": {
                "data": np.array(
                    [
                        [
                            [
                                ["ESP", "FRA"],
                                ["UNK", "ESP"],
                            ]
                        ]
                    ],
                    dtype=object,
                )
            }
        }

    def load(self, var_name):
        return None


class DummyCharEncodedCountries(DummyCountries):
    def __init__(self):
        super().__init__()
        self.variables["ISO"]["data"] = np.array(
            [
                [
                    [
                        [["E", "S", "P"], ["A", "N", "D"]],
                        [["U", "N", "K"], ["E", "S", "P"]],
                    ]
                ]
            ],
            dtype="S1",
        )


class DummyObjectArrayCountries(DummyCountries):
    def __init__(self):
        super().__init__()
        country_data = np.empty((1, 1, 2, 2), dtype=object)
        country_data[0, 0, 0, 0] = np.array([b"E", b"S", b"P"])
        country_data[0, 0, 0, 1] = np.array([b"A", b"N", b"D"])
        country_data[0, 0, 1, 0] = np.array([b"U", b"N", b"K"])
        country_data[0, 0, 1, 1] = np.array([b"E", b"S", b"P"])
        self.variables["ISO"]["data"] = country_data


class DummyParallelCountries(DummyCountries):
    def __init__(self):
        super().__init__()
        self.variables["ISO"]["data"] = np.array(
            [
                [
                    [
                        ["FRA", "FRA"],
                        ["FRA", "FRA"],
                    ]
                ]
            ],
            dtype=object,
        )
        self.comm = DummyAllGatherComm()
        self.master = True


class DummyAllGatherComm(object):
    def allgather(self, value):
        return [
            value,
            {"ESP", "AND"},
        ]


class DummyCountriesWithPortugal(DummyCountries):
    def __init__(self):
        super().__init__()
        self.variables["ISO"]["data"] = np.array(
            [
                [
                    [
                        ["ESP", "PRT"],
                        ["AND", "ESP"],
                    ]
                ]
            ],
            dtype=object,
        )


class DummyDestinationGrid(object):
    def __init__(self, country_iso2):
        self.variables = {
            "country_iso2": {
                "data": np.array(country_iso2, dtype=object),
            }
        }


class DummyMaskingGrid(object):
    comm = None
    parallel_method = None
    balanced = False


@unittest.skipIf(
    MASKING_IMPORT_ERROR is not None,
    f"Masking dependencies missing: {MASKING_IMPORT_ERROR}",
)
class TestMasking(unittest.TestCase):
    def test_calculate_mask_array_warns_for_country_codes_missing_from_domain(self):
        masking = Masking.__new__(Masking)
        countries = DummyCountries()

        with patch("hermes_gr.modules.masking.masking.log_message") as log_message:
            mask_array = masking.calculate_mask_array(
                {
                    "default": 1,
                    "ESP": 0.5,
                    "AND": 0.2,
                },
                countries,
                "factor_mask",
            )

        self.assertEqual(mask_array[0, 0, 0, 0], 0.5)
        log_message.assert_called_once()
        message = log_message.call_args.args[0]
        self.assertIn("factor_mask", message)
        self.assertIn("AND", message)
        self.assertNotIn("ESP,", message)
        self.assertEqual(log_message.call_args.kwargs["level"], 7)

    def test_calculate_mask_array_errors_for_invalid_iso3_country_codes(self):
        masking = Masking.__new__(Masking)
        countries = DummyCountries()

        with self.assertRaises(ValueError) as error:
            masking.calculate_mask_array(
                {
                    "default": 1,
                    "ESP": 0.5,
                    "SPA": 0.2,
                },
                countries,
                "factor_mask",
            )

        message = str(error.exception)
        self.assertIn("factor_mask", message)
        self.assertIn("SPA", message)
        self.assertIn("do not exist", message)

    def test_calculate_mask_array_does_not_warn_for_country_codes_present_in_domain(self):
        masking = Masking.__new__(Masking)
        countries = DummyCountries()

        with patch("hermes_gr.modules.masking.masking.log_message") as log_message:
            masking.calculate_mask_array(
                {
                    "default": 0,
                    "ESP": 1,
                    "FRA": 1,
                },
                countries,
                "regrid_mask",
            )

        log_message.assert_not_called()

    def test_calculate_mask_array_normalizes_country_code_case(self):
        masking = Masking.__new__(Masking)
        countries = DummyCountries()

        with patch("hermes_gr.modules.masking.masking.log_message") as log_message:
            mask_array = masking.calculate_mask_array(
                {
                    "default": 0,
                    "esp": 1,
                    "fra": 1,
                },
                countries,
                "regrid_mask",
            )

        self.assertEqual(mask_array[0, 0, 0, 0], 1)
        self.assertEqual(mask_array[0, 0, 0, 1], 1)
        self.assertEqual(mask_array[0, 0, 1, 1], 1)
        log_message.assert_not_called()

    def test_calculate_mask_array_stores_normalized_mask_info(self):
        masking = Masking.__new__(Masking)
        countries = DummyCountries()

        masking.calculate_mask_array(
            {
                "default": 0,
                "esp": 1,
                "unk": 1,
            },
            countries,
            "regrid_mask",
        )

        self.assertEqual(countries.variables["mask_0"]["mask_info"], "{'default': 0, 'ESP': 1}")

    def test_normalize_country_code_ignores_no_data_values_case_insensitively(self):
        self.assertIsNone(Masking._normalize_country_code("unk"))
        self.assertIsNone(Masking._normalize_country_code("NaN"))
        self.assertIsNone(Masking._normalize_country_code("none"))

    def test_calculate_mask_array_supports_char_encoded_country_mask(self):
        masking = Masking.__new__(Masking)
        countries = DummyCharEncodedCountries()

        with patch("hermes_gr.modules.masking.masking.log_message") as log_message:
            mask_array = masking.calculate_mask_array(
                {
                    "default": 0,
                    "ESP": 1,
                    "AND": 1,
                },
                countries,
                "regrid_mask",
            )

        self.assertEqual(mask_array[0, 0, 0, 0], 1)
        self.assertEqual(mask_array[0, 0, 0, 1], 1)
        self.assertEqual(mask_array[0, 0, 1, 0], 0)
        log_message.assert_not_called()

    def test_calculate_mask_array_supports_object_array_country_codes(self):
        masking = Masking.__new__(Masking)
        countries = DummyObjectArrayCountries()

        with patch("hermes_gr.modules.masking.masking.log_message") as log_message:
            mask_array = masking.calculate_mask_array(
                {
                    "default": 0,
                    "ESP": 1,
                    "AND": 1,
                },
                countries,
                "regrid_mask",
            )

        self.assertEqual(mask_array[0, 0, 0, 0], 1)
        self.assertEqual(mask_array[0, 0, 0, 1], 1)
        self.assertEqual(mask_array[0, 0, 1, 0], 0)
        log_message.assert_not_called()

    def test_calculate_mask_array_uses_global_country_codes_for_warning(self):
        masking = Masking.__new__(Masking)
        countries = DummyParallelCountries()

        with patch("hermes_gr.modules.masking.masking.log_message") as log_message:
            masking.calculate_mask_array(
                {
                    "default": 0,
                    "ESP": 1,
                    "AND": 1,
                },
                countries,
                "regrid_mask",
            )

        log_message.assert_not_called()

    def test_calculate_mask_array_warns_for_country_codes_missing_from_destination_domain(self):
        masking = Masking.__new__(Masking)
        countries = DummyCountriesWithPortugal()
        dst_grid = DummyDestinationGrid([["ES", "AD"], ["ES", "ES"]])

        with patch("hermes_gr.modules.masking.masking.log_message") as log_message:
            masking.calculate_mask_array(
                {
                    "default": 1,
                    "AND": 0,
                    "PRT": 0,
                },
                countries,
                "regrid_mask",
                dst_grid=dst_grid,
            )

        log_message.assert_called_once()
        message = log_message.call_args.args[0]
        self.assertIn("output domain", message)
        self.assertIn("PRT", message)
        self.assertIn("valid ISO3", message)
        self.assertNotIn("misspelled", message)
        self.assertNotIn("AND", message)

    def test_existing_country_mask_checks_dimensions_against_source_grid_path(self):
        masking = Masking.__new__(Masking)
        masking.factors_mask_values = {"default": 1}
        masking.regrid_mask_values = None
        existing_country_mask = SimpleNamespace(load=lambda: None)
        source_grid = SimpleNamespace()

        with patch("hermes_gr.modules.masking.masking.os.path.exists", return_value=True), \
                patch(
                    "hermes_gr.modules.masking.masking.open_netcdf",
                    side_effect=[existing_country_mask, source_grid],
                ), \
                patch("hermes_gr.modules.masking.masking.check_aux_file", return_value=True) as check_aux_file:
            masking.get_countries_data(
                DummyMaskingGrid(),
                "/path/to/source_grid.nc",
                "/path/to/countries.shp",
                "/path/to/existing_world_mask.nc",
            )

        self.assertIs(check_aux_file.call_args.kwargs["nes_object_to_compare"], source_grid)
        self.assertEqual(check_aux_file.call_args.kwargs["var_list"], ["ISO"])


if __name__ == "__main__":
    unittest.main()
