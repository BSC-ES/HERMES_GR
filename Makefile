# Copyright 2018-2026 Earth Sciences Department, Barcelona Supercomputing Center (BSC-CNS)
#
# This file is part of HERMES_GR.
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

version=3.0.0

### PATHS: Change them if needed
# Paths to the HERMES_GR software
#--------------------------------------------------------------------------------------
# Extracting directory where the makefile is
MAKEFILE_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
# HERMES_GR  path:
hermes_path=$(MAKEFILE_DIR)

#--------------------------------------------------------------------------------------
# LOCAL or MN5: Change this depending if you want to do the make locally or in MN5. Default is MN5
HERMES_PROJECT_PATH=$(hermes_path)

CONDA_ENV_DIR ?= "."
CONDA_ENV_PATH=$(CONDA_ENV_DIR)/HERMES_GR_v$(version)
#--------------------------------------------------------------------------------------
# If the installation is for development, the name of the user will be added to the end of
# the name of the environment
create_env_for_dev: CONDA_ENV_PATH=$(CONDA_ENV_DIR)/HERMES_GR_$(USER)
set_env_vars_for_dev: CONDA_ENV_PATH=$(CONDA_ENV_DIR)/HERMES_GR_$(USER)
install_software_for_dev: CONDA_ENV_PATH=$(CONDA_ENV_DIR)/HERMES_GR_$(USER)
full_installation_for_dev: CONDA_ENV_PATH=$(CONDA_ENV_DIR)/HERMES_GR_$(USER)

# Default path to the setup script
SETUP_PY=$(HERMES_PROJECT_PATH)/setup.py

# Default conda environment yaml file
ENV_YML="$(HERMES_PROJECT_PATH)/environment.yml"

.PHONY: create_env create_conda_env create_env_for_dev write_activate_vars_script write_deactivate_vars_script install_software install_software_for_dev full_installation full_installation_for_dev full_local_installation
# Task to create the conda environment
create_env:
	@echo "Creating environment in: $(CONDA_ENV_PATH)"
	mamba env create -p $(CONDA_ENV_PATH) -f $(ENV_YML)

create_conda_env:
	@echo "Creating environment in: $(CONDA_ENV_PATH)"
	conda env create -p $(CONDA_ENV_PATH) -f $(ENV_YML)

# Task to create the directories activate.d and deactivate.d inside the environment
create_activate_deactivate_dirs:
	mkdir -p $(CONDA_ENV_PATH)/etc/conda/activate.d
	mkdir -p $(CONDA_ENV_PATH)/etc/conda/deactivate.d

#Task to write the env_vars.sh inside activate.d to set those variables when the environment is activated
write_activate_vars_script:
	echo '#!/bin/bash' > $(CONDA_ENV_PATH)/etc/conda/activate.d/env_vars.sh
	echo 'export SLURM_CPU_BIND=none' >> $(CONDA_ENV_PATH)/etc/conda/activate.d/env_vars.sh
#	echo 'export OMPI_MCA_pml=ob1' >> $(CONDA_ENV_PATH)/etc/conda/activate.d/env_vars.sh
#	echo 'export OMPI_MCA_btl=self,vader,tcp' >> $(CONDA_ENV_PATH)/etc/conda/activate.d/env_vars.sh
	echo 'eval "$$(register-python-argcomplete hermes_gr_preproc)"' >> $(CONDA_ENV_PATH)/etc/conda/activate.d/env_vars.sh


#Task to write the env_vars.sh inside deactivate.d to unset those variables when the environment is deactivated
write_deactivate_vars_script:
	echo '#!/bin/bash' > $(CONDA_ENV_PATH)/etc/conda/deactivate.d/env_vars.sh
	echo 'unset PYTHONPATH' >> $(CONDA_ENV_PATH)/etc/conda/deactivate.d/env_vars.sh
#	echo 'unset OMPI_MCA_pml' >> $(CONDA_ENV_PATH)/etc/conda/deactivate.d/env_vars.sh
#	echo 'unset OMPI_MCA_btl' >> $(CONDA_ENV_PATH)/etc/conda/deactivate.d/env_vars.sh

# Task to set environment variables inside the Conda environment
set_env_vars: create_activate_deactivate_dirs write_activate_vars_script
	@echo "Setting environment variables..."

set_env_vars_for_dev: create_activate_deactivate_dirs write_activate_vars_script
	@echo "Setting environment variables..."

# Task to install the software using pip install inside the Conda environment
install_software:
	# Install NES
	conda run -p $(CONDA_ENV_PATH) python -m pip install $(HERMES_PROJECT_PATH)/hermes_gr/sources/NES
	# Install HERMES_GR
	conda run -p $(CONDA_ENV_PATH) python -m pip install $(HERMES_PROJECT_PATH)
	@echo "Installation completed. Run 'conda activate $(CONDA_ENV_PATH)' to use the environment."

# Task to install the software using pip install inside the Conda environment for development
install_software_for_dev:
	# Install NES
	conda run -p $(CONDA_ENV_PATH) python -m pip install -e $(HERMES_PROJECT_PATH)/hermes_gr/sources/NES
	# Install HERMES_GR
	conda run -p $(CONDA_ENV_PATH) python -m pip install -e $(HERMES_PROJECT_PATH)
	@echo "Installation completed. Run 'conda activate $(CONDA_ENV_PATH)' to use the environment."

# Task to test the imports of the required libraries and the PEP8 conventions compliance
run_main_test:
	conda run -p $(CONDA_ENV_PATH) python $(HERMES_PROJECT_PATH)/hermes_gr/tests/test_imports.py
	conda run -p $(CONDA_ENV_PATH) python $(HERMES_PROJECT_PATH)/hermes_gr/tests/test_lint.py

run_tests: run_main_test
	@echo "Testing completed."

# Combined task to make full installation for user
full_installation: create_env set_env_vars install_software run_tests
	@echo "Installation and testing completed. \
	Run 'conda activate $(CONDA_ENV_PATH)' to use the environment."

full_conda_installation: create_conda_env set_env_vars install_software run_tests
	@echo "Installation and testing completed. \
	Run 'conda activate $(CONDA_ENV_PATH)' to use the environment."

# Combined task to make full installation for developer
full_installation_for_dev: create_env set_env_vars_for_dev install_software_for_dev run_tests
	@echo "Installation for development and testing completed. \
	Run 'conda activate $(CONDA_ENV_PATH)' to use the environment."

# Combined task to make full installation for developer using CONDA instead of MAMBA
full_installation_for_dev_conda: CONDA_ENV_PATH=$(CONDA_ENV_DIR)/HERMES_GR_$(USER)
full_installation_for_dev_conda: create_conda_env set_env_vars_for_dev install_software_for_dev run_tests

	@echo "Installation for development and testing completed. \
	Run 'conda activate $(CONDA_ENV_PATH)' to use the environment."

# Combined task to make full local installation for user
full_local_installation: create_env install_software run_tests
	@echo "Local installation and testing completed. \
	Run 'conda activate $(CONDA_ENV_PATH)' to use the environment."

# Task to clean up the environment
clean:
	conda env remove -p $(CONDA_ENV_PATH) -y
	conda clean --all --yes

all: full_installation
	@echo "Installation completed. Run 'conda activate $(CONDA_ENV_PATH)' to use the environment.\
    If you want to make the installation for development, use 'make full_installation_for_dev'"
