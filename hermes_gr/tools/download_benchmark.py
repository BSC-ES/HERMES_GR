#!/usr/bin/env python

# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.


import sys
import os


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
              "\n\t{0} /home/user/HERMES".format(exe_str)))
        sys.exit(1)
    elif len(args) > 1:
        print(("Too much arguments through '{0}'. Only destination path is needed e.g.:".format(exe_str) +
              "\n\t{0} /home/user/HERMES".format(exe_str)))
        sys.exit(1)
    else:
        dir_path = args[0]

    if not os.path.exists(dir_path):
        if query_yes_no("'{0}' does not exist. Do you want to create it? ".format(dir_path)):
            os.makedirs(dir_path)
        else:
            sys.exit(0)

    return dir_path


def download_files(parent_path):
    from ftplib import FTP

    ftp = FTP('bscesftp.bsc.es')
    ftp.login()
    dst_file = os.path.join(parent_path, 'HERMES_GR_Benchmark.zip')

    ftp.retrbinary('RETR HERMES_GR_Benchmark.zip', open(dst_file, 'wb').write)

    ftp.quit()

    return dst_file


def unzip_files(zippath, parent_path):
    import zipfile

    zip_file = zipfile.ZipFile(zippath, 'r')
    zip_file.extractall(parent_path)
    zip_file.close()
    os.remove(zippath)


def download_benchmark():
    argv = sys.argv[1:]

    parent_dir = check_args(argv, 'hermes_gr_download_benchmark')

    zippath = download_files(parent_dir)
    unzip_files(zippath, parent_dir)


if __name__ == '__main__':
    download_benchmark()
