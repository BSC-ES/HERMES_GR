#!/usr/bin/env python

# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

"""CLI utilities to analyze HERMES scalability timing logs."""

import argparse
import glob
import os
import re
import sys
import tempfile

import numpy as np
import pandas as pd

from hermes_gr.utilities.legacy_scalability_timings import convert_legacy_log
from hermes_gr.utilities.legacy_scalability_timings import is_legacy_log
from hermes_gr.utilities.legacy_scalability_timings import is_early_v3_log
from hermes_gr.utilities.legacy_scalability_timings import convert_early_v3_log


TIMING_METADATA_COLUMNS = ["section", "subsection"]
TIMING_STAT_COLUMNS = ["min", "mean", "max", "median"]
TIMING_SCHEMAS = ["auto", "v3", "v2"]
GROUPED_COLORS = {
    "Initialization": "#8EDDE8",
    "Calculation": "#FFE27A",
    "Writing": "#F3A0CF",
}
TOTAL_LINE_STYLES = {
    "Initialization": {"marker": "s", "linestyle": "-"},
    "Calculation": {"marker": "^", "linestyle": "-"},
    "HERMES": {"marker": "o", "linestyle": "-"},
}
DETAIL_COLORS = [
    "#4E79A7",
    "#F28E2B",
    "#59A14F",
    "#E15759",
    "#B07AA1",
    "#76B7B2",
    "#EDC948",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
    "#1F77B4",
    "#FF7F0E",
    "#2CA02C",
    "#D62728",
    "#9467BD",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
    "#BCBD22",
    "#17BECF",
]
WORKFLOW_TIMING_ORDER = [
    ("Initialization", "Grid"),
    ("Initialization", "init_vertical"),
    ("Initialization", "init_speciation"),
    ("Initialization", "init_temporal"),
    ("Initialization", "load_emissions"),
    ("Initialization", "init_horizontal_get_weight_matrix"),
    ("Initialization", "init_horizontal_regrid"),
    ("Initialization", "rescaling"),
    ("Initialization", "TOTAL"),
    ("Calculation", "do_vertical"),
    ("Calculation", "do_speciation"),
    ("Calculation", "do_temporal"),
    ("Calculation", "get_4d_emissions"),
    ("Calculation", "combine_inventories"),
    ("Calculation", "TOTAL"),
    ("Writing", "prepare_output"),
    ("Writing", "write_time_step_emissions"),
    ("Writing", "unit_conversion"),
    ("Writing", "communication"),
    ("Writing", "netcdf_write"),
    ("Writing", "netcdf_write_unaccounted"),
    ("Writing", "close_output"),
    ("Writing", "TOTAL"),
    ("HERMES", "TOTAL"),
]
WORKFLOW_PHASES = [
    ("Initialization", "TOTAL", "Initialization"),
    ("Calculation", "TOTAL", "Calculation"),
    ("Writing", "TOTAL", "Writing"),
]
RAW_EXCLUDED_TIMERS = {
    ("Writing", "close_output"),
}
WRITE_TIME_STEP_TIMER = ("Writing", "write_time_step_emissions")
WRITE_TIME_STEP_INTERNAL_TIMERS = [
    ("Writing", "unit_conversion"),
    ("Writing", "communication"),
    ("Writing", "netcdf_write"),
    ("Writing", "netcdf_write_unaccounted"),
]
WORKFLOW_GROUPS = [
    ("Initialization", "Grid", [("Initialization", "Grid")], []),
    ("Initialization", "Vertical", [("Initialization", "init_vertical")], []),
    ("Initialization", "Temporal", [("Initialization", "init_temporal")], []),
    (
        "Initialization",
        "Horizontal",
        [
            ("Initialization", "load_emissions"),
            ("Initialization", "init_horizontal_get_weight_matrix"),
            ("Initialization", "init_horizontal_regrid"),
            ("Initialization", "rescaling"),
        ],
        [],
    ),
    ("Initialization", "Speciation", [("Initialization", "init_speciation")], []),
    ("Calculation", "Vertical", [("Calculation", "do_vertical")], []),
    ("Calculation", "Speciation", [("Calculation", "do_speciation")], []),
    ("Calculation", "Temporal", [("Calculation", "do_temporal")], []),
    (
        "Calculation",
        "CalculateTimeStepEmissions",
        [
            ("Calculation", "get_4d_emissions"),
            ("Calculation", "combine_inventories"),
        ],
        [],
    ),
    (
        "Writing",
        "Output",
        [
            ("Writing", "prepare_output"),
            ("Writing", "write_time_step_emissions"),
            ("Writing", "close_output"),
        ],
        [],
    ),
]


class ScalabilityAnalysis(object):
    """Analyze execution-time logs produced by HERMES scalability experiments."""

    def __init__(
            self,
            log_pattern,
            out_dir,
            replications=None,
            outlier_method=None,
            outlier_threshold=1.5,
            aggregation_method="mean",
            min_replications_for_filtering=4,
            timing_stat="max",
            timing_schema="auto"):
        if timing_stat not in TIMING_STAT_COLUMNS:
            raise ValueError("Unsupported timing statistic: {}".format(timing_stat))
        if timing_schema not in TIMING_SCHEMAS:
            raise ValueError("Unsupported timing schema: {}".format(timing_schema))
        self.log_pattern = log_pattern
        self.out_dir = out_dir
        self.replications = replications
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.aggregation_method = aggregation_method
        self.min_replications_for_filtering = min_replications_for_filtering
        self.timing_stat = timing_stat
        self.timing_schema = timing_schema
        self.logs = self.discover_logs(log_pattern, replications=replications)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        self.total_times = None
        self.outlier_summary = None

    @staticmethod
    def _compile_log_pattern(log_pattern):
        """Build a regular expression from a log pattern containing placeholders."""
        if "<processes>" not in log_pattern:
            raise ValueError("The log pattern must contain the <processes> placeholder.")

        escaped_pattern = re.escape(os.path.abspath(log_pattern))

        for placeholder, group_name in [
                ("<replication>", "replication"),
                ("<processes>", "processes")]:
            placeholder_pattern = re.escape(placeholder)
            replacement_count = 0

            def replace_placeholder(_match):
                nonlocal replacement_count
                replacement_count += 1
                if replacement_count == 1:
                    return r"(?P<{}>[^/]+)".format(group_name)
                return r"(?P={})".format(group_name)

            escaped_pattern = re.sub(
                placeholder_pattern,
                replace_placeholder,
                escaped_pattern,
            )
        return re.compile(r"^{}$".format(escaped_pattern))

    @classmethod
    def discover_logs(cls, log_pattern, replications=None):
        """
        Discover timing logs and extract their replication and process metadata.

        Parameters
        ----------
        log_pattern : str
            File path pattern. It must contain ``<processes>`` and can contain
            ``<replication>``.
        replications : int or None
            Optional number of expected integer replication folders.

        Returns
        -------
        list of dict
            One record per discovered file.
        """
        glob_pattern = log_pattern.replace("<replication>", "*").replace("<processes>", "*")
        log_paths = sorted(glob.glob(glob_pattern))
        pattern_regex = cls._compile_log_pattern(log_pattern)
        log_records = []

        for log_path in log_paths:
            abs_log_path = os.path.abspath(log_path)
            match = pattern_regex.match(abs_log_path)
            if match is None:
                continue

            process_value = match.group("processes")
            try:
                process_id = int(process_value)
            except ValueError:
                continue

            if "<replication>" in log_pattern:
                replication_value = match.group("replication")
                try:
                    replication_id = int(replication_value)
                except ValueError:
                    replication_id = replication_value
            else:
                replication_id = 1

            log_records.append({
                "path": abs_log_path,
                "replication": replication_id,
                "process": process_id,
            })

        if replications is not None:
            integer_replications = [
                log_record["replication"] for log_record in log_records
                if isinstance(log_record["replication"], int)
            ]
            first_replication = 0 if 0 in integer_replications else 1
            expected_replications = set(range(first_replication, first_replication + replications))
            log_records = [
                log_record for log_record in log_records
                if log_record["replication"] in expected_replications
            ]

        if not log_records:
            raise FileNotFoundError("No timing logs were found with the provided pattern.")

        return sorted(log_records, key=lambda item: (item["replication"], item["process"], item["path"]))

    def read_timing_log(self, path):
        """Read one timing log, converting legacy schemas when requested."""
        if self.timing_schema == "v2" or (
                self.timing_schema == "auto" and is_legacy_log(path)):
            log = convert_legacy_log(path)
        else:
            log = pd.read_csv(path)
            if self.timing_schema == "auto" and is_early_v3_log(log):
                log = convert_early_v3_log(log)

        if self.timing_stat == "median" and "median" not in log.columns:
            rank_columns = [
                column for column in log.columns
                if column not in TIMING_METADATA_COLUMNS + ["min", "mean", "max"]
            ]
            if not rank_columns:
                raise ValueError(
                    "Cannot compute median timing statistic because {} has no rank columns.".format(path))
            log["median"] = log[rank_columns].median(axis=1)

        timing_columns = TIMING_METADATA_COLUMNS + [self.timing_stat]
        missing_columns = [column for column in timing_columns if column not in log.columns]
        if missing_columns:
            raise ValueError(
                "Timing log {} is missing required columns: {}".format(
                    path,
                    ", ".join(missing_columns),
                ))
        return log[timing_columns]

    def read_logs(self):
        """
        Read discovered logs into a single DataFrame with replication/process columns.

        Returns
        -------
        pandas.DataFrame
            Timing values indexed by ``section`` and ``subsection``.
        """
        total_times = []

        for log_record in self.logs:
            log = self.read_timing_log(log_record["path"])
            log = log.set_index(["section", "subsection"])
            log = log.rename(columns={self.timing_stat: (log_record["replication"], log_record["process"])})
            log.columns = pd.MultiIndex.from_tuples(log.columns, names=["Replication", "Process"])
            total_times.append(log)

        return pd.concat(total_times, axis=1).sort_index(axis=1)

    @staticmethod
    def _get_inlier_mask(values, outlier_method=None, outlier_threshold=1.5, min_replications_for_filtering=4):
        """Return a boolean mask indicating which replication values are kept."""
        valid_values = values.dropna()
        keep_mask = pd.Series(True, index=values.index, dtype=bool)

        if outlier_method in (None, "none") or valid_values.size < min_replications_for_filtering:
            return keep_mask

        if outlier_method not in {"iqr", "mad", "zscore"}:
            raise ValueError("Unsupported outlier method: {}".format(outlier_method))

        if outlier_method == "iqr":
            quantile_25 = valid_values.quantile(0.25)
            quantile_75 = valid_values.quantile(0.75)
            iqr_value = quantile_75 - quantile_25
            if iqr_value == 0:
                return keep_mask
            lower_bound = quantile_25 - outlier_threshold * iqr_value
            upper_bound = quantile_75 + outlier_threshold * iqr_value
            inlier_mask = (valid_values >= lower_bound) & (valid_values <= upper_bound)
        elif outlier_method == "mad":
            median_value = valid_values.median()
            mad_value = (valid_values - median_value).abs().median()
            if mad_value == 0:
                return keep_mask
            modified_zscore = 0.6745 * (valid_values - median_value).abs() / mad_value
            inlier_mask = modified_zscore <= outlier_threshold
        else:
            std_value = valid_values.std(ddof=0)
            if std_value == 0:
                return keep_mask
            zscore = ((valid_values - valid_values.mean()) / std_value).abs()
            inlier_mask = zscore <= outlier_threshold

        keep_mask.loc[valid_values.index] = inlier_mask
        return keep_mask

    @classmethod
    def aggregate_replications(
            cls,
            total_times_df,
            outlier_method=None,
            outlier_threshold=1.5,
            aggregation_method="mean",
            min_replications_for_filtering=4):
        """
        Aggregate timing values across replications for every timing block/process.

        Returns
        -------
        tuple
            Aggregated timing DataFrame and an outlier bookkeeping DataFrame.
        """
        if aggregation_method not in {"mean", "median"}:
            raise ValueError("Unsupported aggregation method: {}".format(aggregation_method))

        aggregated_times = {}
        summary_records = []

        process_ids = sorted(total_times_df.columns.get_level_values("Process").unique())
        for process_id in process_ids:
            process_frame = total_times_df.xs(process_id, axis=1, level="Process")
            aggregated_column = []

            for timing_block, row_values in process_frame.iterrows():
                keep_mask = cls._get_inlier_mask(
                    row_values,
                    outlier_method=outlier_method,
                    outlier_threshold=outlier_threshold,
                    min_replications_for_filtering=min_replications_for_filtering,
                )
                original_count = int(row_values.dropna().size)
                filtered_values = row_values[keep_mask].dropna()
                kept_count = int(filtered_values.size)

                if filtered_values.empty:
                    filtered_values = row_values.dropna()
                    kept_count = int(filtered_values.size)

                if aggregation_method == "mean":
                    aggregated_value = filtered_values.mean()
                else:
                    aggregated_value = filtered_values.median()

                aggregated_column.append(aggregated_value)
                summary_records.append({
                    "section": timing_block[0],
                    "subsection": timing_block[1],
                    "Process": process_id,
                    "replications_total": original_count,
                    "replications_kept": kept_count,
                    "replications_removed": original_count - kept_count,
                    "outlier_method": outlier_method if outlier_method is not None else "none",
                    "aggregation_method": aggregation_method,
                })

            aggregated_times[process_id] = pd.Series(aggregated_column, index=process_frame.index)

        aggregated_times_df = pd.DataFrame(aggregated_times)
        aggregated_times_df.columns.name = "Process"
        outlier_summary_df = pd.DataFrame(summary_records)
        outlier_summary_df = outlier_summary_df.set_index(["section", "subsection", "Process"])

        return aggregated_times_df, outlier_summary_df

    def analyze(self):
        """Read the logs and aggregate them according to the configured options."""
        total_times_df = self.read_logs()
        self.total_times, self.outlier_summary = self.aggregate_replications(
            total_times_df,
            outlier_method=self.outlier_method,
            outlier_threshold=self.outlier_threshold,
            aggregation_method=self.aggregation_method,
            min_replications_for_filtering=self.min_replications_for_filtering,
        )
        return self.total_times

    @staticmethod
    def _row_exists(times, section, subsection):
        return (section, subsection) in times.index

    def has_workflow_timing(self):
        """Return whether the aggregated logs use the new workflow timing schema."""
        if self.total_times is None:
            self.analyze()
        return any(self._row_exists(self.total_times, section, subsection)
                   for section, subsection in WORKFLOW_TIMING_ORDER)

    def get_ordered_times(self):
        """Return the aggregated timing table ordered by the known workflow schema."""
        if self.total_times is None:
            self.analyze()

        ordered_index = [
            timing_block for timing_block in WORKFLOW_TIMING_ORDER
            if timing_block in self.total_times.index
        ]
        remaining_index = [
            timing_block for timing_block in self.total_times.index
            if timing_block not in ordered_index
        ]
        return self.total_times.reindex(ordered_index + remaining_index)

    def get_section_detail(self, section):
        """Return a flat-index timing table for one workflow section."""
        ordered_times = self.get_ordered_times()
        try:
            section_times = ordered_times.xs(section, level="section")
        except KeyError:
            return pd.DataFrame()

        return section_times

    def _sum_timing_blocks(self, timing_blocks):
        """Sum available timing blocks from the aggregated table."""
        selected_rows = [
            self.total_times.loc[timing_block]
            for timing_block in timing_blocks
            if timing_block in self.total_times.index
        ]
        if not selected_rows:
            return None
        return pd.concat(selected_rows, axis=1).sum(axis=1)

    @staticmethod
    def _has_positive_timers(times, timing_blocks):
        """Return whether any timing block exists with positive values."""
        return any(
            (
                timing_block in times.index
                and pd.to_numeric(times.loc[timing_block], errors="coerce").fillna(0).sum() > 0
            )
            for timing_block in timing_blocks
        )

    def get_phase_summary(self):
        """Return one row per high-level workflow phase."""
        workflow_summary = {}
        for section, subsection, label in WORKFLOW_PHASES:
            if self._row_exists(self.total_times, section, subsection):
                workflow_summary[label] = self.total_times.loc[(section, subsection)]

        if not workflow_summary:
            return pd.DataFrame()

        workflow_order = ["Initialization", "Calculation", "Writing"]
        workflow_df = pd.DataFrame.from_dict(workflow_summary, orient="index")
        return workflow_df.reindex([label for label in workflow_order if label in workflow_df.index])

    def get_grouped_summary(self):
        """Return workflow timings grouped by phase and domain component."""
        grouped_summary = {}

        for section, group, timing_blocks, residual_blocks in WORKFLOW_GROUPS:
            if section == "Writing" and group == "Output":
                if self._has_positive_timers(self.total_times, WRITE_TIME_STEP_INTERNAL_TIMERS):
                    timing_blocks = [("Writing", "prepare_output")] + WRITE_TIME_STEP_INTERNAL_TIMERS
            group_values = self._sum_timing_blocks(timing_blocks)
            if group_values is None:
                continue

            if residual_blocks:
                component_values = self._sum_timing_blocks(residual_blocks)
                if component_values is not None:
                    group_values = group_values - component_values
                    group_values[group_values < 0] = 0

            if not group_values.isna().all() and group_values.sum() > 0:
                grouped_summary["{}/{}".format(section, group)] = group_values

        if not grouped_summary:
            return pd.DataFrame()

        return pd.DataFrame.from_dict(grouped_summary, orient="index")

    def get_raw_summary(self):
        """Return measured function timers in workflow order."""
        detail = self.get_ordered_times()
        raw_rows = []
        excluded_timers = set(RAW_EXCLUDED_TIMERS)
        has_write_step_internal_timers = self._has_positive_timers(
            detail, WRITE_TIME_STEP_INTERNAL_TIMERS)
        if has_write_step_internal_timers:
            excluded_timers.add(WRITE_TIME_STEP_TIMER)

        for section in ["Initialization", "Calculation", "Writing"]:
            section_rows = [
                timing_block for timing_block in detail.index
                if (
                    timing_block[0] == section
                    and timing_block[1] != "TOTAL"
                    and timing_block not in excluded_timers
                )
            ]
            if section_rows:
                section_detail = detail.reindex(section_rows)
                section_detail.index = [
                    "{}/{}".format(row_section, subsection)
                    for row_section, subsection in section_detail.index
                ]
                raw_rows.append(section_detail)

        remaining_rows = [
            timing_block for timing_block in detail.index
            if (
                timing_block[1] != "TOTAL"
                and timing_block[0] not in {"Initialization", "Calculation", "Writing"}
                and timing_block not in excluded_timers
            )
        ]
        if remaining_rows:
            remaining_detail = detail.reindex(remaining_rows)
            remaining_detail.index = [
                "{}/{}".format(section, subsection)
                for section, subsection in remaining_detail.index
            ]
            raw_rows.append(remaining_detail)

        if raw_rows:
            return pd.concat(raw_rows)
        return pd.DataFrame()

    @staticmethod
    def add_total_reference(plot_times, detailed_times):
        """Add workflow totals as plot-only reference lines when available."""
        if "HERMES/TOTAL" in plot_times.index:
            return plot_times

        total_rows = []
        for section in ["Initialization", "Calculation", "HERMES"]:
            total_index = (section, "TOTAL")
            total_label = "{}/TOTAL".format(section)
            if total_index in detailed_times.index and total_label not in plot_times.index:
                total_rows.append(pd.DataFrame(
                    [detailed_times.loc[total_index]],
                    index=[total_label],
                ))

        if not total_rows:
            return plot_times
        return pd.concat([plot_times] + total_rows)

    @staticmethod
    def add_hermes_total_reference(plot_times, detailed_times):
        """Add the HERMES total as a plot-only reference line when available."""
        total_index = ("HERMES", "TOTAL")
        total_label = "HERMES/TOTAL"
        if total_label in plot_times.index or total_index not in detailed_times.index:
            return plot_times

        total_row = pd.DataFrame(
            [detailed_times.loc[total_index]],
            index=[total_label],
        )
        return pd.concat([plot_times, total_row])

    def summarize(self, style="auto"):
        """
        Build a plotting/reporting summary from the aggregated timings.

        ``grouped`` returns Initialization/Calculation/Writing. ``detailed``
        returns one extra level of grouped detail. ``raw`` keeps measured
        function timers in workflow order.
        """
        if self.total_times is None:
            self.analyze()

        if style not in {"auto", "simple", "phase", "grouped", "detailed", "detail", "raw"}:
            raise ValueError("Unsupported summary style: {}".format(style))

        if style == "raw":
            return self.get_raw_summary()

        if self.has_workflow_timing():
            if style in {"detailed", "detail"}:
                grouped_summary = self.get_grouped_summary()
                if not grouped_summary.empty:
                    return grouped_summary

            phase_summary = self.get_phase_summary()
            if not phase_summary.empty:
                return phase_summary

        summary = {}
        if self._row_exists(self.total_times, "Init", "TOTAL"):
            summary["Initialization"] = self.total_times.loc[("Init", "TOTAL")]

        for subsection in ["PointSourcesAllocation", "Speciation", "Vertical", "Temporal", "Horizontal"]:
            try:
                summary[subsection] = self.total_times.xs(subsection, level="subsection").sum()
            except KeyError:
                pass

        if self._row_exists(self.total_times, "Write", "TOTAL"):
            summary["Write"] = self.total_times.loc[("Write", "TOTAL")]
        if self._row_exists(self.total_times, "HERMES", "TOTAL"):
            summary["TOTAL"] = self.total_times.loc[("HERMES", "TOTAL")]

        if summary:
            return pd.DataFrame.from_dict(summary, orient="index")
        if style == "simple":
            raise KeyError("No known HERMES timing blocks were found for the simple summary.")

        return self.get_raw_summary()

    @staticmethod
    def get_speedup(times):
        """Compute strong-scaling speedup as T(1) / T(p)."""
        if 1 not in times.columns:
            raise KeyError("Process count 1 is required to compute speedup and efficiency.")
        return times.div(times[1], axis=0).rdiv(1)

    @staticmethod
    def get_efficiency(times):
        """Compute parallel efficiency as speedup / p."""
        speedup = ScalabilityAnalysis.get_speedup(times)
        return speedup.div(speedup.columns, axis=1)

    @staticmethod
    def _load_pyplot():
        if "MPLCONFIGDIR" not in os.environ:
            os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="hermes_matplotlib_")
        if "XDG_CACHE_HOME" not in os.environ:
            os.environ["XDG_CACHE_HOME"] = tempfile.mkdtemp(prefix="hermes_cache_")
        import matplotlib as mpl
        mpl.use("Agg")
        import matplotlib.pyplot as plt
        return plt

    @staticmethod
    def _is_grouped_categories(categories):
        return set(map(str, categories)).issubset(set(GROUPED_COLORS.keys()))

    @staticmethod
    def _get_phase_boundary_series(times, categories):
        boundaries = {}
        if "Initialization/TOTAL" in times.index:
            boundaries["Initialization"] = times.loc["Initialization/TOTAL"]
        if "Calculation/TOTAL" in times.index:
            calculation_total = times.loc["Calculation/TOTAL"]
            if "Initialization" in boundaries:
                calculation_total = boundaries["Initialization"] + calculation_total
            boundaries["Calculation"] = calculation_total
        return boundaries

    @classmethod
    def _get_category_colors(cls, categories):
        categories = list(categories)
        if cls._is_grouped_categories(categories):
            return {
                category: GROUPED_COLORS[str(category)]
                for category in categories
            }
        return {
            category: DETAIL_COLORS[idx % len(DETAIL_COLORS)]
            for idx, category in enumerate(categories)
        }

    @staticmethod
    def _blend_hex_color(hex_color, blend_with="#FFFFFF", factor=0.35):
        """Blend a hex color with another color."""
        color = hex_color.lstrip("#")
        other = blend_with.lstrip("#")
        rgb = [int(color[idx:idx + 2], 16) for idx in range(0, 6, 2)]
        other_rgb = [int(other[idx:idx + 2], 16) for idx in range(0, 6, 2)]
        blended = [
            int(round(component * (1.0 - factor) + other_component * factor))
            for component, other_component in zip(rgb, other_rgb)
        ]
        return "#{:02X}{:02X}{:02X}".format(*blended)

    @classmethod
    def _get_fill_and_edge_colors(cls, color):
        return (
            cls._blend_hex_color(color, blend_with="#FFFFFF", factor=0.25),
            cls._blend_hex_color(color, blend_with="#000000", factor=0.45),
        )

    @classmethod
    def plot_times(cls, times, output_path, title=None):
        """Generate a stacked execution-time figure."""
        plt = cls._load_pyplot()
        fig, ax = plt.subplots(figsize=(13.5, 8.5))
        categories = [
            category for category in times.index
            if category != "TOTAL" and not str(category).endswith("/TOTAL")
        ]
        x_values = list(times.columns)
        accumulated = pd.Series(0.0, index=times.columns)
        colors = cls._get_category_colors(categories)

        for category in categories:
            upper = accumulated + times.loc[category]
            fill_color, edge_color = cls._get_fill_and_edge_colors(colors[category])
            ax.fill_between(
                x_values, accumulated.values, upper.values,
                color=fill_color, alpha=0.82, edgecolor=edge_color, linewidth=0.7, label=category)
            accumulated = upper

        if not cls._is_grouped_categories(categories):
            for phase, phase_boundary in cls._get_phase_boundary_series(times, categories).items():
                line_style = TOTAL_LINE_STYLES[phase]
                ax.plot(
                    x_values,
                    phase_boundary,
                    marker=line_style["marker"],
                    color="black",
                    linestyle=line_style["linestyle"],
                    linewidth=1.4,
                    label="{} TOTAL".format(phase),
                )

        total_label = None
        for candidate in ["TOTAL", "HERMES/TOTAL"]:
            if candidate in times.index:
                total_label = candidate
                break

        if total_label is not None:
            line_style = TOTAL_LINE_STYLES["HERMES"]
            ax.plot(
                x_values,
                times.loc[total_label],
                marker=line_style["marker"],
                color="black",
                linestyle=line_style["linestyle"],
                linewidth=1.7,
                label="HERMES TOTAL",
            )

        ax.set_xscale("log", base=2)
        ax.set_xticks(x_values)
        ax.set_xticklabels([str(value) for value in x_values])
        ax.set_xlabel("MPI processes")
        ax.set_ylabel("Execution time (s)")
        ax.set_title(title or "HERMES scalability timing analysis")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)

    @classmethod
    def plot_section_times(cls, times, output_path, section, title_suffix=""):
        """Generate an execution-time plot for the detailed timers of one section."""
        if times.empty:
            return False

        plot_times = times.drop(index="TOTAL", errors="ignore")
        if plot_times.empty:
            return False

        cls.plot_times(
            plot_times,
            output_path,
            title="HERMES {} timing detail{}".format(section.lower(), title_suffix),
        )
        return True

    @classmethod
    def plot_metric(cls, metric, output_path, ylabel, title=None, ideal=None):
        """Generate a line plot for a scalability metric."""
        plt = cls._load_pyplot()
        fig, ax = plt.subplots(figsize=(13.5, 8.5))
        x_values = list(metric.columns)
        categories = [
            category for category in metric.index
            if category != "TOTAL" and not str(category).endswith("/TOTAL")
        ]
        colors = cls._get_category_colors(categories)

        for category in categories:
            ax.plot(x_values, metric.loc[category], marker="o", color=colors[category], label=category)

        total_label = None
        for candidate in ["TOTAL", "HERMES/TOTAL"]:
            if candidate in metric.index:
                total_label = candidate
                break

        if total_label is not None:
            line_style = TOTAL_LINE_STYLES["HERMES"]
            ax.plot(
                x_values,
                metric.loc[total_label],
                marker=line_style["marker"],
                color="black",
                linestyle=line_style["linestyle"],
                linewidth=1.7,
                label="HERMES TOTAL",
            )

        if ideal == "speedup":
            ax.plot(x_values, x_values, linestyle="--", color="red", label="Ideal")
        elif ideal == "efficiency":
            ax.axhline(1.0, linestyle="--", color="red", label="Ideal")

        ax.set_xscale("log", base=2)
        ax.set_xticks(x_values)
        ax.set_xticklabels([str(value) for value in x_values])
        ax.set_xlabel("MPI processes")
        ax.set_ylabel(ylabel)
        ax.set_title(title or ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)

    def write_outputs(self, summary_style="auto", env=None, write_plots=True):
        """Run the analysis and write CSV/figure outputs to the output directory."""
        times = self.summarize(style=summary_style)
        detailed_times = self.get_ordered_times()
        times_path = os.path.join(self.out_dir, "Times_Summary.csv")
        detailed_times_path = os.path.join(self.out_dir, "Times_Detail.csv")
        outlier_path = os.path.join(self.out_dir, "Outlier_Summary.csv")
        speedup_path = os.path.join(self.out_dir, "Speedup_Summary.csv")
        efficiency_path = os.path.join(self.out_dir, "Efficiency_Summary.csv")

        times.to_csv(times_path)
        detailed_times.to_csv(detailed_times_path)
        self.outlier_summary.to_csv(outlier_path)

        output_files = [times_path, detailed_times_path, outlier_path]
        title_suffix = " ({})".format(env) if env else ""

        try:
            speedup = self.get_speedup(times)
            efficiency = self.get_efficiency(times)
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            speedup = None
            efficiency = None
        else:
            speedup.to_csv(speedup_path)
            efficiency.to_csv(efficiency_path)
            output_files.extend([speedup_path, efficiency_path])

        if write_plots:
            times_plot = os.path.join(self.out_dir, "HERMES_Times.png")
            plot_times = self.add_total_reference(times, detailed_times)
            self.plot_times(plot_times, times_plot, title="HERMES execution times{}".format(title_suffix))
            output_files.append(times_plot)

            if speedup is not None and efficiency is not None:
                speedup_plot = os.path.join(self.out_dir, "HERMES_Speedup.png")
                efficiency_plot = os.path.join(self.out_dir, "HERMES_Efficiency.png")
                metric_times = self.add_hermes_total_reference(times, detailed_times)
                speedup_plot_data = self.get_speedup(metric_times)
                efficiency_plot_data = self.get_efficiency(metric_times)
                self.plot_metric(
                    speedup_plot_data, speedup_plot, "Speedup",
                    title="HERMES speedup{}".format(title_suffix), ideal="speedup")
                self.plot_metric(
                    efficiency_plot_data, efficiency_plot, "Efficiency",
                    title="HERMES parallel efficiency{}".format(title_suffix), ideal="efficiency")
                output_files.extend([speedup_plot, efficiency_plot])

        return output_files


def get_arguments(args=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze HERMES scalability timing logs.")
    parser.add_argument(
        "--log-pattern", required=True,
        help="Timing log pattern with <processes> and optionally <replication> placeholders.")
    parser.add_argument(
        "--out-dir", required=True,
        help="Directory where CSV summaries and plots will be written.")
    parser.add_argument(
        "--replications", type=int, default=None,
        help="Optional number of expected integer replications. Uses 0-based filtering when replication 0 exists.")
    parser.add_argument(
        "--outlier-method", choices=["none", "iqr", "mad", "zscore"], default="none",
        help="Optional outlier filtering method.")
    parser.add_argument(
        "--outlier-threshold", type=float, default=1.5,
        help="Threshold used by the selected outlier method.")
    parser.add_argument(
        "--aggregation-method", choices=["mean", "median"], default="mean",
        help="Method used to aggregate replications.")
    parser.add_argument(
        "--timing-stat", choices=TIMING_STAT_COLUMNS, default="max",
        help="Timing statistic read from each HERMES time log.")
    parser.add_argument(
        "--timing-schema", choices=TIMING_SCHEMAS, default="auto",
        help="Timing log schema. 'auto' maps v2 and early v3 logs when detected.")
    parser.add_argument(
        "--min-replications-for-filtering", type=int, default=4,
        help="Minimum valid replications required before filtering outliers.")
    parser.add_argument(
        "--summary-style", choices=["auto", "simple", "phase", "grouped", "detailed", "detail", "raw"],
        default="auto",
        help="Timing summary style used for output tables and plots.")
    parser.add_argument(
        "--env", default=None,
        help="Optional environment name to include in plot titles.")
    parser.add_argument(
        "--skip-plots", action="store_true",
        help="Only write CSV outputs.")
    return parser.parse_args(args)


def run(args=None):
    """Execute the scalability-analysis CLI."""
    parsed_args = get_arguments(args=args)
    outlier_method = None if parsed_args.outlier_method == "none" else parsed_args.outlier_method
    analysis = ScalabilityAnalysis(
        parsed_args.log_pattern,
        parsed_args.out_dir,
        replications=parsed_args.replications,
        outlier_method=outlier_method,
        outlier_threshold=parsed_args.outlier_threshold,
        aggregation_method=parsed_args.aggregation_method,
        min_replications_for_filtering=parsed_args.min_replications_for_filtering,
        timing_stat=parsed_args.timing_stat,
        timing_schema=parsed_args.timing_schema,
    )
    output_files = analysis.write_outputs(
        summary_style=parsed_args.summary_style,
        env=parsed_args.env,
        write_plots=not parsed_args.skip_plots,
    )

    for output_file in output_files:
        print(output_file)


if __name__ == "__main__":
    run()
