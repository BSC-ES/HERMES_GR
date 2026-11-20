#!/usr/bin/env python

# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.


import sys
import os


def make_conf_file_list():
    main_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir, os.pardir)

    file_list = [
        {'conf': [
            os.path.join(main_dir, 'conf', 'hermes.conf'),
            os.path.join(main_dir, 'conf', 'EI_configuration.csv'),
        ]},
    ]

    return file_list


def make_profiles_file_list():
    main_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir, os.pardir)

    file_list = [
        {'data': [
            os.path.join(main_dir, 'data', 'global_attributes.csv'),
            {'profiles': [{
                'speciation': [
                    os.path.join(main_dir, 'data', 'profiles', 'speciation', 'MolecularWeights.csv'),
                    os.path.join(main_dir, 'data', 'profiles', 'speciation',
                                 'Speciation_profile_cb05_aero5_Benchmark.csv'),
                    os.path.join(main_dir, 'data', 'profiles', 'speciation', 'Speciation_profile_cb05_aero5_CMAQ.csv'),
                    os.path.join(main_dir, 'data', 'profiles', 'speciation',
                                 'Speciation_profile_cb05_aero5_MONARCH.csv'),
                    os.path.join(main_dir, 'data', 'profiles', 'speciation',
                                 'Speciation_profile_cb05_aero6_CMAQ.csv'),
                    os.path.join(main_dir, 'data', 'profiles', 'speciation',
                                 'Speciation_profile_radm2_madesorgam_WRF_CHEM.csv'),
                    os.path.join(main_dir, 'data', 'profiles', 'speciation',
                                 'Speciation_profile_cb05e51_aero6_CMAQ.csv'),
                ]},
                {'temporal': [
                    os.path.join(main_dir, 'data', 'profiles', 'temporal', 'TemporalProfile_Daily.csv'),
                    os.path.join(main_dir, 'data', 'profiles', 'temporal', 'TemporalProfile_Weekly.csv'),
                    os.path.join(main_dir, 'data', 'profiles', 'temporal', 'TemporalProfile_Hourly.csv'),
                    os.path.join(main_dir, 'data', 'profiles', 'temporal', 'TemporalProfile_Monthly.csv'),
                    os.path.join(main_dir, 'data', 'profiles', 'temporal', 'tz_world_country_iso3166.csv'),
                ]},
                {'vertical': [
                    os.path.join(main_dir, 'data', 'profiles', 'vertical',
                                 'Benchmark_15layers_vertical_description.csv'),
                    os.path.join(main_dir, 'data', 'profiles', 'vertical', 'Vertical_profile.csv'),
                ]},
            ]},
        ]},
    ]

    return file_list


def make_preproc_file_list():
    main_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir, os.pardir)

    file_list = [
        os.path.join(main_dir, 'preproc', 'ceds_preproc.py'),
        os.path.join(main_dir, 'preproc', 'eclipsev5a_preproc.py'),
        os.path.join(main_dir, 'preproc', 'edgarv432_ap_preproc.py'),
        os.path.join(main_dir, 'preproc', 'edgarv432_voc_preproc.py'),
        os.path.join(main_dir, 'preproc', 'emep_preproc.py'),
        os.path.join(main_dir, 'preproc', 'GFAS_Parameters.csv'),
        os.path.join(main_dir, 'preproc', 'wiedinmyer_preproc.py'),
        os.path.join(main_dir, 'preproc', 'htapv2_preproc.py'),
        os.path.join(main_dir, 'preproc', 'tno_mac_iii_preproc.py'),
        os.path.join(main_dir, 'preproc', 'tno_mac_iii_preproc_voc_ratios.py'),
        os.path.join(main_dir, 'preproc', 'VOC_split_AIR.csv'),
        os.path.join(main_dir, 'preproc', 'VOC_split_SHIP.csv'),
        os.path.join(main_dir, 'preproc', 'wiedinmyer_preproc.py'),
        os.path.join(main_dir, 'preproc', 'nmvoc', 'ratio_voc01.nc'),
        os.path.join(main_dir, 'preproc', 'nmvoc', 'ratio_voc02.nc'),
        os.path.join(main_dir, 'preproc', 'nmvoc', 'ratio_voc03.nc'),
        os.path.join(main_dir, 'preproc', 'nmvoc', 'ratio_voc04.nc'),
        os.path.join(main_dir, 'preproc', 'nmvoc', 'ratio_voc05.nc'),
        os.path.join(main_dir, 'preproc', 'nmvoc', 'ratio_voc06.nc'),
        os.path.join(main_dir, 'preproc', 'nmvoc', 'ratio_voc07.nc'),
        os.path.join(main_dir, 'preproc', 'nmvoc', 'ratio_voc08.nc'),
        os.path.join(main_dir, 'preproc', 'nmvoc', 'ratio_voc09.nc'),
        os.path.join(main_dir, 'preproc', 'nmvoc', 'ratio_voc12.nc'),
        os.path.join(main_dir, 'preproc', 'nmvoc', 'ratio_voc13.nc'),
        os.path.join(main_dir, 'preproc', 'nmvoc', 'ratio_voc14.nc'),
        os.path.join(main_dir, 'preproc', 'nmvoc', 'ratio_voc15.nc'),
        os.path.join(main_dir, 'preproc', 'nmvoc', 'ratio_voc16.nc'),
        os.path.join(main_dir, 'preproc', 'nmvoc', 'ratio_voc17.nc'),
        os.path.join(main_dir, 'preproc', 'nmvoc', 'ratio_voc18.nc'),
        os.path.join(main_dir, 'preproc', 'nmvoc', 'ratio_voc19.nc'),
        os.path.join(main_dir, 'preproc', 'nmvoc', 'ratio_voc20.nc'),
        os.path.join(main_dir, 'preproc', 'nmvoc', 'ratio_voc21.nc'),
        os.path.join(main_dir, 'preproc', 'nmvoc', 'ratio_voc22.nc'),
        os.path.join(main_dir, 'preproc', 'nmvoc', 'ratio_voc23.nc'),
        os.path.join(main_dir, 'preproc', 'nmvoc', 'ratio_voc24.nc'),
        os.path.join(main_dir, 'preproc', 'nmvoc', 'ratio_voc25.nc'),
    ]

    return file_list


def query_yes_no(question, default="yes"):
    valid = {"yes": True, "y": True, "1": True, 1: True,
             "no": False, "n": False, "0": False, 0: False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")


def check_args(args, exe_str):
    if len(args) == 0:
        print(("Missing destination path after '{0}'. e.g.:".format(exe_str) +
              "\n\t{0} /home/user/HERMES/HERMES_IN".format(exe_str)))
        sys.exit(1)
    elif len(args) > 1:
        print(("Too much arguments through '{0}'. Only destination path is needed e.g.:".format(exe_str) +
              "\n\t{0} /home/user/HERMES/HERMES_IN".format(exe_str)))
        sys.exit(1)
    else:
        dir_path = args[0]

    if not os.path.exists(dir_path):
        if query_yes_no("'{0}' does not exist. Do you want to create it? ".format(dir_path)):
            os.makedirs(dir_path)
        else:
            sys.exit(0)

    return dir_path


def copy_files(file_list, directory):
    from shutil import copy2

    if not os.path.exists(directory):
        os.makedirs(directory)

    for element in file_list:
        if dict == type(element):
            copy_files(list(element.values())[0], os.path.join(directory, list(element.keys())[0]))
        else:
            copy2(element, directory)
    return True


def copy_config_files():
    argv = sys.argv[1:]

    parent_dir = check_args(argv, 'hermes_gr_copy_config_files')

    copy_files(make_conf_file_list(), parent_dir)
    copy_files(make_profiles_file_list(), parent_dir)


def copy_preproc_files():
    argv = sys.argv[1:]

    parent_dir = check_args(argv, 'hermes_gr_copy_preproc_files')

    copy_files(make_preproc_file_list(), parent_dir)


if __name__ == '__main__':
    copy_config_files()
