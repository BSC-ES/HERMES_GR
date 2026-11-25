# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import os
import tempfile
import unittest

import pandas as pd

from hermes_gr.utilities.scalability_analysis import ScalabilityAnalysis
from hermes_gr.utilities.scalability_analysis import run
from hermes_gr.utilities.legacy_scalability_timings import convert_legacy_log


class TestScalabilityAnalysis(unittest.TestCase):

    @staticmethod
    def _write_log(path, total_time):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        log = pd.DataFrame({
            "section": ["Init", "HERMES"],
            "subsection": ["TOTAL", "TOTAL"],
            "0": [0.5, total_time - 1.0],
            "1": [1.0, total_time],
            "min": [0.5, total_time - 1.0],
            "mean": [0.75, total_time - 0.5],
            "max": [1.0, total_time],
        })
        log.to_csv(path, index=False)

    def test_discovers_and_aggregates_replications(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._write_log(os.path.join(tmp_dir, "1", "HERMES_Times_1.csv"), 10.0)
            self._write_log(os.path.join(tmp_dir, "2", "HERMES_Times_1.csv"), 14.0)
            self._write_log(os.path.join(tmp_dir, "1", "HERMES_Times_2.csv"), 5.0)
            self._write_log(os.path.join(tmp_dir, "2", "HERMES_Times_2.csv"), 7.0)

            analysis = ScalabilityAnalysis(
                os.path.join(tmp_dir, "<replication>", "HERMES_Times_<processes>.csv"),
                os.path.join(tmp_dir, "out"),
            )
            times = analysis.analyze()
            summary = analysis.summarize(style="simple")
            speedup = analysis.get_speedup(summary)
            efficiency = analysis.get_efficiency(summary)

            self.assertEqual(len(analysis.logs), 4)
            self.assertAlmostEqual(times.loc[("HERMES", "TOTAL"), 1], 12.0)
            self.assertAlmostEqual(times.loc[("HERMES", "TOTAL"), 2], 6.0)
            self.assertAlmostEqual(speedup.loc["TOTAL", 2], 2.0)
            self.assertAlmostEqual(efficiency.loc["TOTAL", 2], 1.0)

    def test_filters_zero_based_replications(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            for replication_id in range(8):
                self._write_log(os.path.join(tmp_dir, str(replication_id), "HERMES_Times_1.csv"), 10.0)

            logs = ScalabilityAnalysis.discover_logs(
                os.path.join(tmp_dir, "<replication>", "HERMES_Times_<processes>.csv"),
                replications=7,
            )

            self.assertEqual(list(range(7)), [log_record["replication"] for log_record in logs])

    def test_cli_writes_csv_outputs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._write_log(os.path.join(tmp_dir, "1", "HERMES_Times_1.csv"), 10.0)
            self._write_log(os.path.join(tmp_dir, "1", "HERMES_Times_2.csv"), 5.0)
            out_dir = os.path.join(tmp_dir, "out")

            run([
                "--log-pattern", os.path.join(tmp_dir, "<replication>", "HERMES_Times_<processes>.csv"),
                "--out-dir", out_dir,
                "--skip-plots",
            ])

            self.assertTrue(os.path.exists(os.path.join(out_dir, "Times_Summary.csv")))
            self.assertTrue(os.path.exists(os.path.join(out_dir, "Times_Detail.csv")))
            self.assertTrue(os.path.exists(os.path.join(out_dir, "Outlier_Summary.csv")))
            self.assertTrue(os.path.exists(os.path.join(out_dir, "Speedup_Summary.csv")))
            self.assertTrue(os.path.exists(os.path.join(out_dir, "Efficiency_Summary.csv")))

    def test_converts_legacy_v216_timing_log(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            legacy_path = os.path.join(
                tmp_dir,
                "rep_000_np_0001",
                "logs",
                "Times_HERMESv3_GR_times_Procs0001.csv",
            )
            os.makedirs(os.path.dirname(legacy_path), exist_ok=True)
            pd.DataFrame({
                "Class": [
                    "Grid",
                    "EmissionInventory",
                    "TemporalDistribution",
                    "Writer",
                    "HERMES",
                    "HERMES",
                ],
                "Function": [
                    "Init",
                    "do_regrid",
                    "calculate_2d_temporal_factors",
                    "write",
                    "Init",
                    "TOTAL",
                ],
                "0": [1.0, 4.0, 3.0, 2.0, 0.5, 10.0],
                "min": [1.0, 4.0, 3.0, 2.0, 0.5, 10.0],
                "mean": [1.0, 4.0, 3.0, 2.0, 0.5, 10.0],
                "max": [1.0, 4.0, 3.0, 2.0, 0.5, 10.0],
            }).to_csv(legacy_path)

            converted = convert_legacy_log(legacy_path)

            self.assertIn("section", converted.columns)
            self.assertIn("subsection", converted.columns)
            detail = converted.set_index(["section", "subsection"])
            self.assertAlmostEqual(detail.loc[("Initialization", "Grid"), "max"], 1.0)
            self.assertAlmostEqual(detail.loc[("Initialization", "init_horizontal_regrid"), "max"], 4.0)
            self.assertAlmostEqual(detail.loc[("Writing", "TOTAL"), "max"], 2.0)
            self.assertAlmostEqual(detail.loc[("HERMES", "TOTAL"), "max"], 10.0)

    def test_scalability_analysis_reads_legacy_v216_timing_logs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            legacy_path = os.path.join(
                tmp_dir,
                "0",
                "logs",
                "Times_HERMESv3_GR_times_Procs0002.csv",
            )
            os.makedirs(os.path.dirname(legacy_path), exist_ok=True)
            pd.DataFrame({
                "Class": ["HERMES", "HERMES", "Writer"],
                "Function": ["Init", "TOTAL", "write"],
                "0": [1.0, 8.0, 2.0],
                "1": [1.0, 9.0, 3.0],
                "min": [1.0, 8.0, 2.0],
                "mean": [1.0, 8.5, 2.5],
                "max": [1.0, 9.0, 3.0],
            }).to_csv(legacy_path)

            analysis = ScalabilityAnalysis(
                os.path.join(tmp_dir, "<replication>", "logs", "Times_HERMESv3_GR_times_Procs<processes>.csv"),
                os.path.join(tmp_dir, "out"),
                timing_schema="auto",
            )

            times = analysis.analyze()

            self.assertAlmostEqual(times.loc[("HERMES", "TOTAL"), 2], 9.0)
            self.assertAlmostEqual(times.loc[("Writing", "TOTAL"), 2], 3.0)

    def test_auto_converts_early_v3_phase_timers(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "0", "HERMES_Times_2.csv")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            pd.DataFrame({
                "section": ["Init", "HERMES", "HERMES", "HERMES", "Write"],
                "subsection": ["TOTAL", "Main", "Write", "TOTAL", "Creation"],
                "0": [10.0, 40.0, 15.0, 50.0, 2.0],
                "1": [12.0, 44.0, 17.0, 56.0, 3.0],
                "min": [10.0, 40.0, 15.0, 50.0, 2.0],
                "mean": [11.0, 42.0, 16.0, 53.0, 2.5],
                "max": [12.0, 44.0, 17.0, 56.0, 3.0],
            }).to_csv(path, index=False)

            analysis = ScalabilityAnalysis(
                os.path.join(tmp_dir, "<replication>", "HERMES_Times_<processes>.csv"),
                os.path.join(tmp_dir, "out"),
                timing_schema="auto",
            )
            times = analysis.analyze()
            summary = analysis.summarize(style="grouped")

            self.assertAlmostEqual(times.loc[("Calculation", "TOTAL"), 2], 27.0)
            self.assertAlmostEqual(times.loc[("Writing", "write_time_step_emissions"), 2], 14.0)
            self.assertAlmostEqual(times.loc[("HERMES", "TOTAL"), 2], 56.0)
            self.assertEqual(["Initialization", "Calculation", "Writing"], list(summary.index))

    def test_discovers_logs_with_repeated_process_placeholder(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = os.path.join(
                tmp_dir,
                "rep_000_np_0032",
                "logs",
                "Times_HERMESv3_GR_2025112900_times_Procs0032.csv",
            )
            self._write_log(log_path, 10.0)

            logs = ScalabilityAnalysis.discover_logs(
                os.path.join(
                    tmp_dir,
                    "rep_<replication>_np_<processes>",
                    "logs",
                    "Times_HERMESv3_GR_2025112900_times_Procs<processes>.csv",
                ),
            )

            self.assertEqual(1, len(logs))
            self.assertEqual(0, logs[0]["replication"])
            self.assertEqual(32, logs[0]["process"])

    def test_scalability_cli_reads_explicit_legacy_v216_schema(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            legacy_path = os.path.join(
                tmp_dir,
                "0",
                "logs",
                "Times_HERMESv3_GR_times_Procs0001.csv",
            )
            os.makedirs(os.path.dirname(legacy_path), exist_ok=True)
            pd.DataFrame({
                "Class": ["HERMES", "HERMES", "Writer"],
                "Function": ["Init", "TOTAL", "write"],
                "0": [1.0, 8.0, 2.0],
                "min": [1.0, 8.0, 2.0],
                "mean": [1.0, 8.0, 2.0],
                "max": [1.0, 8.0, 2.0],
            }).to_csv(legacy_path)

            out_dir = os.path.join(tmp_dir, "out")
            run([
                "--log-pattern",
                os.path.join(tmp_dir, "<replication>", "logs", "Times_HERMESv3_GR_times_Procs<processes>.csv"),
                "--out-dir", out_dir,
                "--timing-schema", "v2",
                "--skip-plots",
            ])

            self.assertTrue(os.path.exists(os.path.join(out_dir, "Times_Summary.csv")))

    def test_summarizes_new_workflow_timing_sections(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = os.path.join(tmp_dir, "1", "HERMES_Times_1.csv")
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            pd.DataFrame({
                "section": ["Initialization", "Calculation", "Writing", "HERMES"],
                "subsection": ["TOTAL", "TOTAL", "TOTAL", "TOTAL"],
                "0": [1.0, 4.0, 2.0, 9.0],
                "1": [2.0, 5.0, 3.0, 10.0],
                "min": [1.0, 4.0, 2.0, 9.0],
                "mean": [1.5, 4.5, 2.5, 9.5],
                "max": [2.0, 5.0, 3.0, 10.0],
            }).to_csv(log_path, index=False)

            analysis = ScalabilityAnalysis(
                os.path.join(tmp_dir, "<replication>", "HERMES_Times_<processes>.csv"),
                os.path.join(tmp_dir, "out"),
            )
            analysis.analyze()
            summary = analysis.summarize(style="simple")
            detail = analysis.get_ordered_times()

            self.assertEqual(["Initialization", "Calculation", "Writing"], list(summary.index))
            self.assertEqual(
                [
                    ("Initialization", "TOTAL"),
                    ("Calculation", "TOTAL"),
                    ("Writing", "TOTAL"),
                    ("HERMES", "TOTAL"),
                ],
                list(detail.index),
            )

    def test_groups_new_workflow_timing_sections(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = os.path.join(tmp_dir, "1", "HERMES_Times_1.csv")
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            pd.DataFrame({
                "section": [
                    "Initialization",
                    "Initialization",
                    "Initialization",
                    "Initialization",
                    "Initialization",
                    "Initialization",
                    "Initialization",
                    "Initialization",
                    "Initialization",
                    "Calculation",
                    "Calculation",
                    "Calculation",
                    "Calculation",
                    "Calculation",
                    "Calculation",
                    "Writing",
                    "Writing",
                    "Writing",
                    "Writing",
                    "HERMES",
                ],
                "subsection": [
                    "Grid",
                    "init_vertical",
                    "init_speciation",
                    "init_temporal",
                    "load_emissions",
                    "init_horizontal_get_weight_matrix",
                    "init_horizontal_regrid",
                    "rescaling",
                    "TOTAL",
                    "do_vertical",
                    "do_speciation",
                    "do_temporal",
                    "get_4d_emissions",
                    "combine_inventories",
                    "TOTAL",
                    "prepare_output",
                    "write_time_step_emissions",
                    "close_output",
                    "TOTAL",
                    "TOTAL",
                ],
                "0": [
                    1.0,
                    0.5,
                    1.0,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    1.0,
                    17.0,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    1.0,
                    14.0,
                    1.0,
                    4.0,
                    0.5,
                    7.0,
                    37.0,
                ],
                "1": [
                    2.0,
                    1.0,
                    1.5,
                    1.5,
                    3.0,
                    4.0,
                    5.0,
                    2.0,
                    22.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    2.0,
                    18.0,
                    2.0,
                    5.0,
                    1.0,
                    9.0,
                    47.0,
                ],
                "min": [
                    1.0,
                    0.5,
                    1.0,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    1.0,
                    17.0,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    1.0,
                    14.0,
                    1.0,
                    4.0,
                    0.5,
                    7.0,
                    37.0,
                ],
                "mean": [
                    1.5,
                    0.75,
                    1.25,
                    1.25,
                    2.5,
                    3.5,
                    4.5,
                    1.5,
                    19.0,
                    1.5,
                    2.5,
                    3.5,
                    4.5,
                    1.5,
                    16.0,
                    1.5,
                    4.5,
                    0.75,
                    8.0,
                    42.0,
                ],
                "max": [
                    2.0,
                    1.0,
                    1.5,
                    1.5,
                    3.0,
                    4.0,
                    5.0,
                    2.0,
                    22.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    2.0,
                    18.0,
                    2.0,
                    5.0,
                    1.0,
                    9.0,
                    47.0,
                ],
            }).to_csv(log_path, index=False)

            analysis = ScalabilityAnalysis(
                os.path.join(tmp_dir, "<replication>", "HERMES_Times_<processes>.csv"),
                os.path.join(tmp_dir, "out"),
            )
            analysis.analyze()
            grouped_summary = analysis.summarize(style="grouped")
            detailed_summary = analysis.summarize(style="detailed")
            raw_summary = analysis.summarize(style="raw")

            self.assertEqual(["Initialization", "Calculation", "Writing"], list(grouped_summary.index))
            self.assertAlmostEqual(grouped_summary.loc["Initialization", 1], 22.0)
            self.assertAlmostEqual(grouped_summary.loc["Calculation", 1], 18.0)
            self.assertAlmostEqual(grouped_summary.loc["Writing", 1], 9.0)
            self.assertAlmostEqual(detailed_summary.loc["Initialization/Grid", 1], 2.0)
            self.assertAlmostEqual(detailed_summary.loc["Initialization/Horizontal", 1], 14.0)
            self.assertAlmostEqual(detailed_summary.loc["Calculation/CalculateTimeStepEmissions", 1], 7.0)
            self.assertNotIn("Initialization/General", detailed_summary.index)
            self.assertNotIn("Calculation/General", detailed_summary.index)
            self.assertNotIn("Calculation/4D", detailed_summary.index)
            self.assertNotIn("Calculation/CombineInventories", detailed_summary.index)
            self.assertAlmostEqual(detailed_summary.loc["Writing/Output", 1], 8.0)
            self.assertNotIn("Writing/General", detailed_summary.index)
            self.assertNotIn("Writing/PrepareOutput", detailed_summary.index)
            self.assertNotIn("Writing/TimeStepEmissions", detailed_summary.index)
            self.assertNotIn("Writing/CloseOutput", detailed_summary.index)
            self.assertAlmostEqual(raw_summary.loc["Calculation/combine_inventories", 1], 2.0)
            self.assertAlmostEqual(raw_summary.loc["Initialization/rescaling", 1], 2.0)
            self.assertNotIn("Calculation/TOTAL", raw_summary.index)
            self.assertNotIn("Writing/close_output", raw_summary.index)
            self.assertLess(
                list(raw_summary.index).index("Initialization/init_horizontal_regrid"),
                list(raw_summary.index).index("Calculation/do_vertical"),
            )
            self.assertLess(
                list(raw_summary.index).index("Calculation/combine_inventories"),
                list(raw_summary.index).index("Writing/prepare_output"),
            )

            plot_times = analysis.add_total_reference(raw_summary, analysis.get_ordered_times())
            boundaries = ScalabilityAnalysis._get_phase_boundary_series(plot_times, raw_summary.index)
            self.assertAlmostEqual(boundaries["Initialization"].loc[1], 22.0)
            self.assertAlmostEqual(boundaries["Calculation"].loc[1], 40.0)

    def test_raw_summary_avoids_write_step_double_counting(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = os.path.join(tmp_dir, "1", "HERMES_Times_1.csv")
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            pd.DataFrame({
                "section": [
                    "Writing",
                    "Writing",
                    "Writing",
                    "Writing",
                    "Writing",
                    "HERMES",
                ],
                "subsection": [
                    "write_time_step_emissions",
                    "unit_conversion",
                    "communication",
                    "netcdf_write",
                    "TOTAL",
                    "TOTAL",
                ],
                "0": [10.0, 1.0, 6.0, 3.0, 10.0, 10.0],
                "1": [10.0, 1.0, 6.0, 3.0, 10.0, 10.0],
                "min": [10.0, 1.0, 6.0, 3.0, 10.0, 10.0],
                "mean": [10.0, 1.0, 6.0, 3.0, 10.0, 10.0],
                "max": [10.0, 1.0, 6.0, 3.0, 10.0, 10.0],
            }).to_csv(log_path, index=False)

            analysis = ScalabilityAnalysis(
                os.path.join(tmp_dir, "<replication>", "HERMES_Times_<processes>.csv"),
                os.path.join(tmp_dir, "out"),
            )
            analysis.analyze()
            raw_summary = analysis.summarize(style="raw")
            detailed_summary = analysis.summarize(style="detailed")

            self.assertNotIn("Writing/write_time_step_emissions", raw_summary.index)
            self.assertIn("Writing/unit_conversion", raw_summary.index)
            self.assertIn("Writing/communication", raw_summary.index)
            self.assertIn("Writing/netcdf_write", raw_summary.index)
            self.assertAlmostEqual(detailed_summary.loc["Writing/Output", 1], 10.0)

    def test_adds_total_reference_to_plot_times(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = os.path.join(tmp_dir, "1", "HERMES_Times_1.csv")
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            pd.DataFrame({
                "section": ["Initialization", "Calculation", "Writing", "HERMES"],
                "subsection": ["TOTAL", "TOTAL", "TOTAL", "TOTAL"],
                "0": [1.0, 2.0, 3.0, 6.0],
                "1": [2.0, 3.0, 4.0, 9.0],
                "min": [1.0, 2.0, 3.0, 6.0],
                "mean": [1.5, 2.5, 3.5, 7.5],
                "max": [2.0, 3.0, 4.0, 9.0],
            }).to_csv(log_path, index=False)

            analysis = ScalabilityAnalysis(
                os.path.join(tmp_dir, "<replication>", "HERMES_Times_<processes>.csv"),
                os.path.join(tmp_dir, "out"),
            )
            analysis.analyze()
            summary = analysis.summarize(style="grouped")
            plot_times = analysis.add_total_reference(summary, analysis.get_ordered_times())

            self.assertNotIn("HERMES/TOTAL", summary.index)
            self.assertIn("HERMES/TOTAL", plot_times.index)
            self.assertAlmostEqual(plot_times.loc["HERMES/TOTAL", 1], 9.0)

    def test_adds_hermes_total_reference_to_metric_times(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            for process_id, factor in [(1, 1.0), (2, 0.5)]:
                log_path = os.path.join(tmp_dir, "1", f"HERMES_Times_{process_id}.csv")
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                pd.DataFrame({
                    "section": ["Initialization", "Calculation", "Writing", "HERMES"],
                    "subsection": ["TOTAL", "TOTAL", "TOTAL", "TOTAL"],
                    "0": [1.0 * factor, 2.0 * factor, 3.0 * factor, 6.0 * factor],
                    "1": [2.0 * factor, 3.0 * factor, 4.0 * factor, 9.0 * factor],
                    "min": [1.0 * factor, 2.0 * factor, 3.0 * factor, 6.0 * factor],
                    "mean": [1.5 * factor, 2.5 * factor, 3.5 * factor, 7.5 * factor],
                    "max": [2.0 * factor, 3.0 * factor, 4.0 * factor, 9.0 * factor],
                }).to_csv(log_path, index=False)

            analysis = ScalabilityAnalysis(
                os.path.join(tmp_dir, "<replication>", "HERMES_Times_<processes>.csv"),
                os.path.join(tmp_dir, "out"),
            )
            analysis.analyze()
            summary = analysis.summarize(style="grouped")
            metric_times = analysis.add_hermes_total_reference(summary, analysis.get_ordered_times())
            speedup = analysis.get_speedup(metric_times)
            efficiency = analysis.get_efficiency(metric_times)

            self.assertNotIn("HERMES/TOTAL", summary.index)
            self.assertIn("HERMES/TOTAL", metric_times.index)
            self.assertAlmostEqual(speedup.loc["HERMES/TOTAL", 2], 2.0)
            self.assertAlmostEqual(efficiency.loc["HERMES/TOTAL", 2], 1.0)

    def test_cli_writes_workflow_detail_plots(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            for process_id, factor in [(1, 1.0), (2, 0.5)]:
                log_path = os.path.join(tmp_dir, "1", f"HERMES_Times_{process_id}.csv")
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                pd.DataFrame({
                    "section": [
                        "Initialization",
                        "Initialization",
                        "Initialization",
                        "Calculation",
                        "Calculation",
                        "Calculation",
                        "Writing",
                        "Writing",
                        "Writing",
                        "HERMES",
                    ],
                    "subsection": [
                        "Grid",
                        "load_emissions",
                        "TOTAL",
                        "do_temporal",
                        "get_4d_emissions",
                        "TOTAL",
                        "prepare_output",
                        "write_time_step_emissions",
                        "TOTAL",
                        "TOTAL",
                    ],
                    "0": [
                        1.0 * factor,
                        2.0 * factor,
                        4.0 * factor,
                        0.5 * factor,
                        3.0 * factor,
                        4.0 * factor,
                        0.5 * factor,
                        1.0 * factor,
                        2.0 * factor,
                        11.0 * factor,
                    ],
                    "1": [
                        2.0 * factor,
                        3.0 * factor,
                        5.0 * factor,
                        1.0 * factor,
                        4.0 * factor,
                        5.0 * factor,
                        1.0 * factor,
                        2.0 * factor,
                        3.0 * factor,
                        13.0 * factor,
                    ],
                    "min": [
                        1.0 * factor,
                        2.0 * factor,
                        4.0 * factor,
                        0.5 * factor,
                        3.0 * factor,
                        4.0 * factor,
                        0.5 * factor,
                        1.0 * factor,
                        2.0 * factor,
                        11.0 * factor,
                    ],
                    "mean": [
                        1.5 * factor,
                        2.5 * factor,
                        4.5 * factor,
                        0.75 * factor,
                        3.5 * factor,
                        4.5 * factor,
                        0.75 * factor,
                        1.5 * factor,
                        2.5 * factor,
                        12.0 * factor,
                    ],
                    "max": [
                        2.0 * factor,
                        3.0 * factor,
                        5.0 * factor,
                        1.0 * factor,
                        4.0 * factor,
                        5.0 * factor,
                        1.0 * factor,
                        2.0 * factor,
                        3.0 * factor,
                        13.0 * factor,
                    ],
                }).to_csv(log_path, index=False)

            out_dir = os.path.join(tmp_dir, "out")
            run([
                "--log-pattern", os.path.join(tmp_dir, "<replication>", "HERMES_Times_<processes>.csv"),
                "--out-dir", out_dir,
                "--summary-style", "detailed",
            ])

            self.assertTrue(os.path.exists(os.path.join(out_dir, "HERMES_Times.png")))
            self.assertTrue(os.path.exists(os.path.join(out_dir, "HERMES_Speedup.png")))
            self.assertTrue(os.path.exists(os.path.join(out_dir, "HERMES_Efficiency.png")))
            self.assertFalse(os.path.exists(os.path.join(out_dir, "HERMES_Initialization_Times.png")))
            self.assertFalse(os.path.exists(os.path.join(out_dir, "HERMES_Calculation_Times.png")))
            self.assertFalse(os.path.exists(os.path.join(out_dir, "HERMES_Writing_Times.png")))

    def test_cli_uses_selected_timing_statistic(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = os.path.join(tmp_dir, "1", "HERMES_Times_1.csv")
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            pd.DataFrame({
                "section": ["Initialization", "Calculation", "Writing"],
                "subsection": ["TOTAL", "TOTAL", "TOTAL"],
                "0": [1.0, 2.0, 3.0],
                "1": [19.0, 38.0, 57.0],
                "min": [1.0, 2.0, 3.0],
                "mean": [10.0, 20.0, 30.0],
                "max": [100.0, 200.0, 300.0],
            }).to_csv(log_path, index=False)
            out_dir = os.path.join(tmp_dir, "out")

            run([
                "--log-pattern", os.path.join(tmp_dir, "<replication>", "HERMES_Times_<processes>.csv"),
                "--out-dir", out_dir,
                "--summary-style", "grouped",
                "--timing-stat", "mean",
                "--skip-plots",
            ])

            summary = pd.read_csv(os.path.join(out_dir, "Times_Summary.csv"), index_col=0)

            self.assertAlmostEqual(summary.loc["Initialization", "1"], 10.0)
            self.assertAlmostEqual(summary.loc["Calculation", "1"], 20.0)
            self.assertAlmostEqual(summary.loc["Writing", "1"], 30.0)

    def test_cli_computes_median_timing_statistic_from_rank_columns(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = os.path.join(tmp_dir, "1", "HERMES_Times_1.csv")
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            pd.DataFrame({
                "section": ["Initialization", "Calculation", "Writing"],
                "subsection": ["TOTAL", "TOTAL", "TOTAL"],
                "0": [1.0, 2.0, 3.0],
                "1": [3.0, 6.0, 9.0],
                "2": [100.0, 200.0, 300.0],
                "min": [1.0, 2.0, 3.0],
                "mean": [34.6666667, 69.3333333, 104.0],
                "max": [100.0, 200.0, 300.0],
            }).to_csv(log_path, index=False)
            out_dir = os.path.join(tmp_dir, "out")

            run([
                "--log-pattern", os.path.join(tmp_dir, "<replication>", "HERMES_Times_<processes>.csv"),
                "--out-dir", out_dir,
                "--summary-style", "grouped",
                "--timing-stat", "median",
                "--skip-plots",
            ])

            summary = pd.read_csv(os.path.join(out_dir, "Times_Summary.csv"), index_col=0)

            self.assertAlmostEqual(summary.loc["Initialization", "1"], 3.0)
            self.assertAlmostEqual(summary.loc["Calculation", "1"], 6.0)
            self.assertAlmostEqual(summary.loc["Writing", "1"], 9.0)


if __name__ == "__main__":
    unittest.main()
