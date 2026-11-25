#!/usr/bin/env python

# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import pandas as pd
import numpy as np
import re
from hermes_gr.config import precision, log_message


class VerticalDistribution(object):
    """
    VerticalDistribution class that contains all the information to do the vertical distribution.

    Attributes
    ----------
    id : str
        ID of the vertical profile that appears in the vertical profile file.
    output_heights : List[float]
        Vertical levels
    vertical_profile : List[tuple]
        List of tuples of two values. The first value of the tuple is the height of the layer and the second
        value is the quantity (%) of pollutant that goes into this layer.
    """

    def __init__(self, vertical_id, vertical_profile_path, vertical_output_profile):
        """
        Initialize the Vertical distribution class.

        Parameters
        ----------
        vertical_id : str
            ID of the vertical profile that appears in the vertical profile file.
        vertical_profile_path : str
            Path to the file that contains all the vertical profiles.
        vertical_output_profile : List[float]
            Vertical levels
        """
        self.id = vertical_id

        self.output_heights = vertical_output_profile
        self.vertical_profile = self.get_vertical_profile(vertical_profile_path)

    def get_vertical_profile(self, path):
        """
        Extract the vertical v_profile from the vertical v_profile file.

        Parameters
        ----------
        path : str
            Path to the file that contains all the vertical profiles.

        Returns
        -------
        List[tuple]
            List of tuples of two values. The first value of the tuple is the height of the layer and the second
            value is the quantity (%) of pollutant that goes into this layer.
        """
        log_message(f"Getting vertical profile id '{self.id}' from {path} .", level=3)

        df = pd.read_csv(path, sep=";")
        try:
            v_profile = df.loc[df[df.ID == self.id].index[0]].to_dict()
        except IndexError:
            log_message("ERROR: Check the .err file to get more info.")
            raise AttributeError(f"ERROR: Vertical profile ID {self.id} is not in the {path} file.")
        v_profile.pop("ID", None)
        v_profile["layers"] = list(map(int, re.split(", |,|; |;| ", v_profile["layers"])))
        v_profile["weights"] = list(map(float, re.split(", |,|; |;| ", v_profile["weights"])))

        if len(v_profile["layers"]) != len(v_profile["weights"]):
            log_message("ERROR: Check the .err file to get more info.")
            raise AttributeError(f"ERROR: The number of layers and numbers os weight have to have the same length."
                                 f" The v_profile '{self.id}' of the '{path}' file doesn't match.")
        else:
            return_value = list(zip(v_profile["layers"], v_profile["weights"]))

        return return_value

    @staticmethod
    def get_weights(prev_layer, layer, in_weight, output_vertical_profile):
        """
        Intersect an input layer with valid model layers without redistributing overflow.

        Each overlap retains the original input-layer weight per unit height.
        Portions above the final output height are omitted; no out-of-range
        index or synthetic top-layer contribution is returned.

        Parameters
        ----------
        prev_layer : float
            Altitude of the lower layer. 0 if it's the first.
        layer : float
            Altitude of the current layer.
        in_weight : float
            Weight og the current layer.
        output_vertical_profile : List[float]
            Output vertical profile (altitude)

        Returns
        -------
        List[dict]
            List of dictionaries with the information of each layer weight.
            Each dictionary contains the 'index' key with the vertical index of the layer and 'weight' with their weight
        """
        origin_diff_factor = in_weight / (layer - prev_layer)
        weight_list = []
        output_bottom = 0
        for index, output_top in enumerate(output_vertical_profile):
            overlap = min(layer, output_top) - max(prev_layer, output_bottom)
            if overlap > 0:
                weight_list.append({"index": index, "weight": overlap * origin_diff_factor})
            output_bottom = output_top

        return weight_list

    def calculate_weights(self):
        """
        Allocate the original profile to model levels, discarding above-top emissions.

        Convert the original profile's weight scale to fractions using its full
        total, including out-of-domain weights. Never divide by the retained sum:
        partial overflow leaves a sum below one and complete overflow leaves zero.
        This preserves existing in-domain normalization for fraction/percent inputs.
        Emit one aggregated warning per calculation if positive weight is discarded,
        rather than one warning per source layer, grid cell or output timestep.

        Returns
        -------
        np.array
            Weights that goes to each layer.
        """
        log_message("Calculating vertical weights.", level=3)

        weights = np.zeros(len(self.output_heights))
        profile_total = sum(weight for layer, weight in self.vertical_profile)
        discarded_weight = 0.0
        prev_layer = 0
        for layer, weight in self.vertical_profile:
            if weight != float(0):
                for element in self.get_weights(prev_layer, layer, weight, self.output_heights):
                    weights[element["index"]] += element["weight"]
                if layer > self.output_heights[-1]:
                    discarded_weight += (
                        weight * (layer - max(prev_layer, self.output_heights[-1])) / (layer - prev_layer)
                    )

            prev_layer = layer

        weights = np.array(weights, dtype=precision)
        # Preserve the original scale, never renormalize the retained fraction.
        if profile_total != 0:
            weights = weights / profile_total
        if discarded_weight > 0:
            log_message(
                f"WARNING: Vertical profile '{self.id}' extends above model top {self.output_heights[-1]} m: "
                f"{discarded_weight / profile_total:.2%} of its emission weight is discarded. "
                "In-domain fractions are preserved; discarded emissions are not redistributed "
                "and retained weights are not renormalized.", level=7,
            )

        return weights
