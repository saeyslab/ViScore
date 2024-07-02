#!/bin/bash

usage="Schedule preparation of scRNA-seq datasets on HPC

-h show this help message and exit
-d path to data directory (default: ./data)
-D path to datasets CSV file (default: ./datasets.csv)
-l path to directory for writing error and output logs (default: ./logs)
"

## Set defaults

DATADIR="./data"
DATASETS_CSV="./datasets.csv"
LOGS="./logs"

## Get user args

while getopts ":d:D:l:h" opt; do
  case $opt in
    d) DATADIR="${OPTARG}" ;;
    D) DATASETS_CSV="${OPTARG}";;
    l) LOGS="${OPTARG}";;
    h) echo "${usage}" ; exit 0 ;;
  esac
done

echo "Target data directory: ${DATADIR}"

## Create a command to activate ViScore venv

VISCORE_VENV_CMD="source ./venvs/venv_ViScore/bin/activate"

## Identify env modules for ViScore and create a command to load them

IFS=$'\n' read -r -d '' -a VISCORE_ENV_MODULES < "./install/ViScore_environment.txt"
if [[ ${#VISCORE_ENV_MODULES[@]} -eq 0 ]]; then
    VISCORE_ENV_MOD_CMD=":"
else
    VISCORE_ENV_MOD_CMD="module load "
    IFS="; "
    for MODULE in "${VISCORE_ENV_MODULES[@]}"; do VISCORE_ENV_MOD_CMD+="$MODULE "; done
    VISCORE_ENV_MOD_CMD=${VISCORE_ENV_MOD_CMD%?}
fi
IFS= 

## Identify dataset names

IFS=$'\n' read -r -d '' -a DATASETS < "./datasets.txt"
IFS= 

## Iterate over datasets

for DATASET in "${DATASETS[@]}"; do

  ## Set up job name and logs

  NAME="02_prepare_dataset_${DATASET}"
  OUTPUTS="${LOGS}/${NAME}_outputs"
  ERRORS="${LOGS}/${NAME}_errors"

  ## Submit job to scheduler
  
  cat <<EOS | qsub -l nodes=1,mem=128gb,walltime=2:00:00 -e ${ERRORS} -o ${OUTPUTS} -N ${NAME}
cd ${PWD}
module purge
eval ${VISCORE_ENV_MOD_CMD}
eval ${VISCORE_VENV_CMD}
python ./00_prepare_datasets.py -n ${DATASET} -d ${DATADIR} -D ${DATASETS_CSV}
EOS
done