#!/usr/bin/env python

# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import pandas as pd
import numpy as np
from hermes_gr.config import precision, log_message
from nes import Nes


class Speciation(object):
    """
    Attributes
    ----------
    speciation_profile : List[dict]
        List of dictionaries. Each dictionary has the keys 'name', 'formula', 'units' and 'long_name'.
    molecular_weights : dict
        Dictionary with the name of the pollutant as key and the molecular weight as value.
    """

    def __init__(self, speciation_id, speciation_profile_path, molecular_weights_path):
        """
        Speciation class initialization

        Parameters
        ----------
        speciation_id : str
            ID of the speciation profile that have to be in the speciation profile file.
        speciation_profile_path : str
            Path to the file that contains all the speciation profiles.
        molecular_weights_path : str
            Path to the file that contains all the needed molecular weights.
        """

        self.id = speciation_id
        self.speciation_profile = self.get_speciation_profile(speciation_profile_path)
        self.molecular_weights = self.extract_molecular_weights(molecular_weights_path)

    def get_speciation_profile(self, speciation_profile_path):
        """
        Validate CSV expressions and extract the selected profile with metadata.

        Every species cell in every profile row must contain an expression or an
        explicit zero. Missing values, NaN markers and whitespace-only cells are
        invalid, including in profiles other than the selected one. The ``units``
        and ``short_description`` rows contain metadata and are excluded from this
        expression check. Completely empty rows are ignored.

        Validation runs before profile selection or expression processing. Errors
        list the file, CSV row (header is row one), profile ID and affected species.
        No missing expression is silently interpreted as a zero contribution.

        Parameters
        ----------
        speciation_profile_path : str
            Path to the speciation profile file.

        Returns
        -------
        List[dict]
            List of dictionaries. Each dictionary has the keys 'name', 'formula', 'units' and 'long_name'.

        Raises
        ------
        ValueError
            A profile contains a missing or NaN speciation expression.
        AttributeError
            The requested profile ID is absent from the file.
        """
        log_message(f"Getting speciation profile id '{self.id}' from {speciation_profile_path} .", level=3)
        # Retain blank records in the index so diagnostics retain their CSV rows.
        df = pd.read_csv(speciation_profile_path, sep=";", skip_blank_lines=False)
        profile_rows = df.dropna(how="all")
        profile_rows = profile_rows.loc[~profile_rows["ID"].isin(["units", "short_description"])]
        invalid = []
        for row_index, row in profile_rows.iterrows():
            for species in df.columns.drop("ID"):
                expression = row[species]
                if pd.isna(expression) or str(expression).strip().lower() in ("", "nan"):
                    invalid.append(f"  - Row {row_index + 2}, profile {row['ID']}, species {species}")
        if invalid:
            raise ValueError(
                f"ERROR: Invalid speciation file: {speciation_profile_path}\n\n"
                "NaN or empty values detected in speciation expressions:\n"
                + "\n".join(invalid)
                + "\n\nSpeciation expressions must contain valid values. "
                "Use an explicit 0 for a species with no contribution. "
                "Please correct the input file before running HERMES_GR."
            )

        try:
            formulas_dict = df.loc[df[df.ID == self.id].index[0]].to_dict()
        except IndexError:
            raise AttributeError(f"ERROR: Speciation profile ID {self.id} is not here {speciation_profile_path}.")
        formulas_dict.pop("ID", None)
        units_dict = df.loc[df[df.ID == "units"].index[0]].to_dict()

        units_dict.pop("ID", None)
        long_name_dict = df.loc[df[df.ID == "short_description"].index[0]].to_dict()
        long_name_dict.pop("ID", None)
        profile_list = []
        for key in formulas_dict.keys():
            profile_list.append(
                {
                    "name": key,
                    "formula": formulas_dict[key],
                    "units": units_dict[key],
                    "long_name": long_name_dict[key],
                }
            )

        return profile_list

    @staticmethod
    def extract_molecular_weights(molecular_weights_path):
        """
        Extract the molecular weights for each pollutant as a dictionary with the name of the pollutant as key and the
        molecular weight as value.

        Parameters
        ----------
        molecular_weights_path : str
            Path to the CSV that contains all the molecular weights.
        Returns
        -------
        dict
            Dictionary with the name of the pollutant as key and the molecular weight as value.
        """

        df = pd.read_csv(
            molecular_weights_path,
            sep=",",
            index_col="Specie",
            dtype={"Specie": str, "MW": precision},
        )
        molecular_weights = df.loc[:, "MW"]

        return molecular_weights

    def do_speciation(self, emissions):
        """
        Speciate the emissions

        Parameters
        ----------
        emissions : Nes
            Emissions NES object

        Returns
        -------
        Nes
            Speciated emissions NES object
        """
        log_message("Speciating", level=2)

        for var_name in emissions.variables.keys():
            try:
                emissions.variables[var_name]["data"] = np.array(
                    emissions.variables[var_name]["data"] / self.molecular_weights[var_name],
                    dtype=precision,
                )
            except KeyError:
                raise KeyError(f"ERROR: {var_name} pollutant is not in the molecular weights file.")
            exec(f"{var_name} = np.array(emissions.variables[var_name]['data'], dtype=precision)")

        speciated_emissions = emissions.copy(copy_vars=False)
        num = 0
        input_pollutants = list(emissions.variables.keys())
        for pollutant in self.speciation_profile:
            formula = str(pollutant["formula"])
            used_poll = []
            for in_p in input_pollutants:
                if in_p in formula:
                    used_poll.append(in_p)
            for poll_rem in used_poll:
                input_pollutants.remove(poll_rem)
            num += 1
            if formula != "nan":
                log_message(f"Pollutant {pollutant['name']} using the formula "
                            f"{pollutant['name']}={formula} ({num}/{len(self.speciation_profile)})", level=3)

                dict_aux = {"units": pollutant["units"], "long_name": pollutant["long_name"]}
                if formula == "0" or formula == 0:
                    dict_aux.update({"data": 0})
                else:
                    try:
                        dict_aux.update({"data": np.array(eval(formula), dtype=precision)})
                    except NameError as e:
                        raise AttributeError(
                            f"Error in speciation profile {self.id}: The output specie {pollutant['name']} cannot be "
                            f"calculated with the expression {formula} because {str(e)}")

                speciated_emissions.variables[pollutant["name"]] = dict_aux
            else:
                log_message(f"Pollutant {pollutant['name']} does not have formula. Ignoring. "
                            f"({num}/{len(self.speciation_profile)})", level=3)
        if len(input_pollutants) > 0:
            log_message(f"WARNING: The input pollutants {input_pollutants} "
                        f"do not appear in the speciation profile {self.id}.", level=7)

        return speciated_emissions
