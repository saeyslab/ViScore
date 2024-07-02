#!/bin/bash

usage="Create CPU/GPU virtual environments for benchmark

Only call this script directly if you are running a local benchmark.
For HPC, use ./01_schedule_venv_creation.sh.

-h show this help message and exit
-c either CPU or GPU (message: CPU)
"

## Set hard-coded

CONFIG="config.json"

## Set defaults

CLUSTER="CPU"

## Get user args

while getopts ":c:h" opt; do
  case $opt in
    c) CLUSTER="${OPTARG}" ;;
    h) echo "${usage}" ; exit 0 ;;
  esac
done

## Make sure venvs directory exists

mkdir -p venvs

## Identify (C/G)PU venvs from the config file

VENVS=( $(jq -r '.methods[] | select(.cluster == "'${CLUSTER}'") .venv' "${CONFIG}") )
VENVS=($(printf "%s\n" "${VENVS[@]}" | sort -u))

## Iterate over venvs

for VENV in "${VENVS[@]}"; do

    echo "Creating virtual environment: ${VENV}"

    ## Identify env modules and load them

    IFS=$'\n' ENV_MODULES=($(tr -d '\r' < "./install/${VENV}_environment.txt"))
    IFS= 
    module purge
    for MODULE in "${ENV_MODULES[@]}"; do
        module load ${MODULE}
    done

    ## Create venv and install required Python modules

    python -m venv ./venvs/venv_${VENV}
    source ./venvs/venv_${VENV}/bin/activate
    ./install/${VENV}_install.sh
    deactivate

    ## Unload modules

    module purge
done

## Also create ViScore venv

echo "Creating virtual environment: ViScore"

IFS=$'\n' ENV_MODULES=($(tr -d '\r' < "./install/ViScore_environment.txt"))
IFS= 
module purge
for MODULE in "${ENV_MODULES[@]}"; do
    module load ${MODULE}
done
python -m venv ./venvs/venv_ViScore
source ./venvs/venv_ViScore/bin/activate
./install/ViScore_install.sh
deactivate
module purge

echo "Done"