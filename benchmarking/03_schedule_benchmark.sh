#!/bin/bash

usage="Schedule a full CPU/GPU benchmark of dimension reduction on HPC

-h show this help message and exit
-c either CPU or CPU (default: CPU)
-i path to input datasets (default: ./data)
-a whether to save trained DR model (0/1, default: 0)
-s number of runs with different random seeds for each setting (default: 5)
-l path to directory for writing error and output logs
-n number of compute nodes to request (default: 1)
-m RAM to request, in GB (default: 16)
-g GPU count per node to request if applicable (default: 1)
-p processors (cores) count per node (default: 2)
-w walltime to specify for the job (default: 1:00:00)
"

## Set hard-coded

OUTPUT="./results"
CONFIG="config.json"
ZDIMS=(2)
DENOISEDS=(0 1)

## Set defaults

CLUSTER="CPU"
INPUT="./data"
SAVE=0
NRUNS=5
LOGS="./logs"
NODES=1
MEMORY=32
GPUS=1
PPN=2
WALLTIME="1:00:00"

## Get user args

while getopts ":c:i:a:s:l:n:m:g:p:w:h" opt; do
  case $opt in
    c) CLUSTER="${OPTARG}";;
    i) INPUT="${OPTARG}";;
    a) SAVE="${OPTARG}";;
    s) NRUNS="${OPTARG}";;
    l) LOGS="${OPTARG}";;
    n) NODES="${OPTARG}";;
    m) MEMORY="${OPTARG}";;
    g) GPUS="${OPTARG}";;
    p) PPN="${OPTARG}";;
    w) WALLTIME="${OPTARG}";;
    h) echo "${usage}" ; exit 0 ;;
  esac
done

## Identify datasets

echo "Identifying datasets"
IFS=$'\n' read -r -d '' -a DATASETS < "./datasets.txt"

## Identify methods to run on this cluster (CPU/GPU)

echo "Identifying ${CLUSTER} methods"
METHODS=( $(jq -r '.methods | to_entries[] | select(.value.cluster == "'${CLUSTER}'") .key' "${CONFIG}") )

## Make sure results and logs directories exist

mkdir -p ${LOGS}
mkdir -p ${OUTPUT}

## Iterate over datasets, methods, dimensionalities and denoising(yes/no)

for DATASET in "${DATASETS[@]}"; do
  for METHOD in "${METHODS[@]}"; do
    for ZDIM in "${ZDIMS[@]}"; do
      for DENOISED in "${DENOISEDS[@]}"; do

        ## Schedule an experiment for each setting

        ./03_schedule_experiment.sh \
          -M ${METHOD} \
          -D ${DATASET} \
          -c ${CLUSTER} \
          -l ${LOGS} \
          -z ${ZDIM} \
          -u ${DENOISED} \
          -s ${NRUNS} \
          -i ${INPUT} \
          -a ${SAVE} \
          -n ${NODES} \
          -m ${MEMORY} \
          -g ${GPUS} \
          -p ${PPN} \
          -w ${WALLTIME}
      done
    done
  done
done