#!/usr/bin/env python

# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    from hermes_gr.modules.emision_inventories.emission_inventory import (
        AUXILIARY_WARNINGS_ATTR,
        EmissionInventory,
    )

    EMISSION_INVENTORY_IMPORT_ERROR = None
except ImportError as import_error:
    EMISSION_INVENTORY_IMPORT_ERROR = import_error


@unittest.skipIf(
    EMISSION_INVENTORY_IMPORT_ERROR is not None,
    f"EmissionInventory dependencies missing: {EMISSION_INVENTORY_IMPORT_ERROR}",
)
class TestEmissionInventoryAuxiliaryWarnings(unittest.TestCase):
    def test_store_auxiliary_warning_messages_deduplicates_existing_warnings(self):
        nessy = SimpleNamespace(global_attrs={AUXILIARY_WARNINGS_ATTR: "warning A"})

        EmissionInventory._store_auxiliary_warning_messages(nessy, ["warning A", "warning B"])

        self.assertEqual(nessy.global_attrs[AUXILIARY_WARNINGS_ATTR], "warning A\nwarning B")
        self.assertEqual(
            EmissionInventory._get_auxiliary_warning_messages(nessy),
            ["warning A", "warning B"],
        )

    def test_emit_auxiliary_warning_messages_logs_stored_warnings_on_master(self):
        inventory = EmissionInventory.__new__(EmissionInventory)
        inventory.grid = SimpleNamespace(master=True)
        nessy = SimpleNamespace(global_attrs={AUXILIARY_WARNINGS_ATTR: "warning A\nwarning B"})

        with patch("hermes_gr.modules.emision_inventories.emission_inventory.log_message") as log_message:
            inventory._emit_auxiliary_warning_messages(nessy)

        self.assertEqual(log_message.call_count, 2)
        self.assertEqual(log_message.call_args_list[0].args[0], "warning A")
        self.assertEqual(log_message.call_args_list[0].kwargs["level"], 7)
        self.assertEqual(log_message.call_args_list[1].args[0], "warning B")

    def test_emit_auxiliary_warning_messages_skips_non_master_ranks(self):
        inventory = EmissionInventory.__new__(EmissionInventory)
        inventory.grid = SimpleNamespace(master=False)
        nessy = SimpleNamespace(global_attrs={AUXILIARY_WARNINGS_ATTR: "warning A"})

        with patch("hermes_gr.modules.emision_inventories.emission_inventory.log_message") as log_message:
            inventory._emit_auxiliary_warning_messages(nessy)

        log_message.assert_not_called()


if __name__ == "__main__":
    unittest.main()
