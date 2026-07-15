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
from hermes_gr.config import get_time_stamp, record_time


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
    def get_vertical_output_profile(path):
        """
        Extract the vertical description of the desired output.

        Parameters
        ----------
        path : str
             Path to the file that contains the output vertical description.

        Returns
        -------
        List
            Heights of the output vertical layers.
        """
        log_message(f"Calculating vertical levels from {path} .")

        df = pd.read_csv(path)

        heights = df.height_magl.values

        return heights

    @staticmethod
    def get_weights(prev_layer, layer, in_weight, output_vertical_profile):
        """
        Calculate the weights for the given layer.

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
        output_vertical_profile_aux = [s for s in output_vertical_profile if s >= prev_layer]
        output_vertical_profile_aux = [s for s in output_vertical_profile_aux if s < layer]

        output_vertical_profile_aux = [prev_layer] + output_vertical_profile_aux + [layer]

        index = len([s for s in output_vertical_profile if s < prev_layer])
        origin_diff_factor = in_weight / (layer - prev_layer)
        weight_list = []
        for i in range(len(output_vertical_profile_aux) - 1):
            weight = abs(output_vertical_profile_aux[i] - output_vertical_profile_aux[i + 1]) * origin_diff_factor
            weight_list.append({"index": index, "weight": weight})
            index += 1

        return weight_list

    def calculate_weights(self):
        """
        Calculate the weights for all the vertical layers.

        Returns
        -------
        np.array
            Weights that goes to each layer.
        """
        st_time = get_time_stamp()
        # start_increment("VerticalDistribution", "calculate_weights")
        # change_labels("VerticalDistribution", "calculate_weights")

        log_message("Calculating vertical weights.", level=3)

        weights = np.zeros(len(self.output_heights))
        prev_layer = 0
        for layer, weight in self.vertical_profile:
            if weight != float(0):
                for element in self.get_weights(prev_layer, layer, weight, self.output_heights):
                    weights[element["index"]] += element["weight"]

            prev_layer = layer

        record_time("VerticalDistribution", "calculate_weights", get_time_stamp() - st_time)
        # stop_increment("VerticalDistribution", "calculate_weights")

        weights = np.array(weights, dtype=precision)
        # Normalizing
        weights = weights / weights.sum()

        return weights
