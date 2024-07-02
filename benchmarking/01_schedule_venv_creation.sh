#!/bin/bash

usage="Schedule CPU/GPU virtual environment creation on HPC

-h show this help message and exit
-c either CPU or GPU (default: CPU)
-n number of compute nodes to request (default: 1)
-m RAM to request, in GB (default: 16)
-g GPU count per node to request if applicable (default: 1)
-w walltime to specify for the job (default: 1:00:00)
-l path to directory for writing error and output logs (default: ./logs)
"

## Set hard-coded

CONFIG="./config.json"

## Set defaults

CLUSTER="CPU"
NODES=1
MEMORY=16
GPUS=1
WALLTIME="1:00:00"
LOGS="./logs"

## Get user args

while getopts ":c:n:m:g:w:l:h" opt; do
  case $opt in
    c) CLUSTER="${OPTARG}" ;;
    n) NODES="${OPTARG}";;
    m) MEMORY="${OPTARG}";;
    g) GPUS="${OPTARG}";;
    w) WALLTIME="${OPTARG}";;
    l) LOGS="${OPTARG}";;
    h) echo "${usage}" ; exit 0 ;;
  esac
done

## Set up job name and logs

NAME="01_create_venvs_${CLUSTER}"
OUTPUTS="${LOGS}/${NAME}_outputs"
ERRORS="${LOGS}/${NAME}_errors"

echo "Scheduling ${CLUSTER} virtual env creation"

## Submit CPU/GPU venv installation job to scheduler

if [[ "${CLUSTER}" == "CPU" ]]; then
    cat <<EOS | qsub -l nodes=${NODES},mem=${MEMORY}gb,walltime=${WALLTIME} -e ${ERRORS} -o ${OUTPUTS} -N ${NAME}
cd ${PWD}
./01_create_venvs.sh -c ${CLUSTER}
EOS
elif [[ "${CLUSTER}" == "GPU" ]]; then
    cat <<EOS | qsub -l nodes=${NODES}:ppn=quarter:gpus=${GPUS},mem=${MEMORY}gb,walltime=${WALLTIME} -e ${ERRORS} -o ${OUTPUTS} -N ${NAME}
cd ${PWD}
./01_create_venvs.sh -c ${CLUSTER}
EOS
fi
