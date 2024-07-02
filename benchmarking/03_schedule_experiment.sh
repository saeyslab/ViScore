#!/bin/bash

usage="Schedule a CPU/GPU experiment

This runs a repeated evaluation of a DR method with fixed settings.

-h show this help message and exit
-M method name
-D name (prefix) of scRNA-seq data files, as prepared earlier
-c either CPU or GPU (default: CPU)
-l path to directory for writing error and output logs
-z target embedding dimensionality (default: 2)
-u whether to use denoised inputs (0/1/2 where 2~only for ViVAE, default: 2)
-s number of runs with different random seeds (default: 5)
-i path to input dataset (default: ./data)
-a whether to save the trained DR model (0/1, default: 0)
-n number of compute nodes to request (default: 1)
-m RAM to request, in GB (default: 16)
-g GPU count per node to request if applicable (default: 1)
-p processors (cores) count per node (default: 2)
-w walltime to specify for the job (default: 1:00:00)
"

## Set hard-coded

OUTPUT="./results"
CONFIG="config.json"

## Set defaults

CLUSTER="CPU"
LOGS="./logs"
ZDIM=2
DENOISED=2
NRUNS=5
INPUT="./data"
SAVE=0
NODES=1
MEMORY=16
GPUS=1
PPN=2
WALLTIME="1:00:00"

## Get user args

while getopts ":M:D:c:l:z:u:s:i:a:n:m:g:w:h" opt; do
  case $opt in
    M) METHOD="${OPTARG}";;
    D) DATASET="${OPTARG}";;
    c) CLUSTER="${OPTARG}";;
    l) LOGS="${OPTARG}";;
    z) ZDIM="${OPTARG}";;
    u) DENOISED="${OPTARG}";;
    s) NRUNS="${OPTARG}";;
    i) INPUT="${OPTARG}";;
    a) SAVE="${OPTARG}";;
    n) NODES="${OPTARG}";;
    m) MEMORY="${OPTARG}";;
    g) GPUS="${OPTARG}";;
    p) PPN="${OPTARG}";;
    w) WALLTIME="${OPTARG}";;
    h) echo "${usage}" ; exit 0 ;;
  esac
done

## Make sure logs and results directories exist

mkdir -p ${LOGS}
mkdir -p ${OUTPUT}

## Identify venv for DR method and create a command to activate it

VENV=$(jq -r ".methods[\"$METHOD\"].venv" "${CONFIG}")
DR_VENV_CMD="source ./venvs/venv_${VENV}/bin/activate"

## Identify env modules for DR method and create a command to load them

IFS=$'\n' read -r -d '' -a DR_ENV_MODULES < "./install/${VENV}_environment.txt"
if [[ ${#DR_ENV_MODULES[@]} -eq 0 ]]; then
    DR_ENV_MOD_CMD=":"
else
    DR_ENV_MOD_CMD="module load "
    IFS="; "
    for MODULE in "${DR_ENV_MODULES[@]}"; do DR_ENV_MOD_CMD+="$MODULE "; done
    DR_ENV_MOD_CMD=${DR_ENV_MOD_CMD%?}
fi
IFS= 

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

## Iterate over runs (with different random seeds)

for (( SEED=1; SEED<=$NRUNS; SEED++ )); do
    ## Set up job name and logs

    NAME="${DATASET}_${METHOD}_z${ZDIM}_u${DENOISED}_seed${SEED}"
    OUTPUTS="${LOGS}/${NAME}_outputs"
    ERRORS="${LOGS}/${NAME}_errors"

    ## Submit CPU/GPU running+scoring job to scheduler

    echo "Submitting ${CLUSTER} job: ${NAME}"

    if [[ "${CLUSTER}" == "CPU" ]]; then

        cat <<EOS | qsub -l nodes=${NODES},mem=${MEMORY}gb,walltime=${WALLTIME} -e ${ERRORS} -o ${OUTPUTS} -N ${NAME}
cd ${PWD}

eval ${DR_ENV_MOD_CMD}
eval ${DR_VENV_CMD}
python ./03a_run_method.py \
    --method ${METHOD} \
    --dataset ${DATASET} \
    --zdim ${ZDIM} \
    --denoised ${DENOISED} \
    --seed ${SEED} \
    --input ${INPUT} \
    --save ${SAVE} \
    --verbose 1
deactivate
module purge

eval ${VISCORE_ENV_MOD_CMD}
eval ${VISCORE_VENV_CMD}
python ./03b_score_method.py \
    --method ${METHOD} \
    --dataset ${DATASET} \
    --zdim ${ZDIM} \
    --denoised ${DENOISED} \
    --seed ${SEED} \
    --input ${INPUT} \
    --verbose 1
deactivate
module purge
EOS
    elif [[ "${CLUSTER}" == "GPU" ]]; then

        cat <<EOS | qsub -l nodes=${NODES}:ppn=${PPN}:gpus=${GPUS},mem=${MEMORY}gb,walltime=${WALLTIME} -e ${ERRORS} -o ${OUTPUTS} -N ${NAME}
cd ${PWD}

eval ${DR_ENV_MOD_CMD}
eval ${DR_VENV_CMD}
python ./03a_run_method.py \
    --method ${METHOD} \
    --dataset ${DATASET} \
    --zdim ${ZDIM} \
    --denoised ${DENOISED} \
    --seed ${SEED} \
    --input ${INPUT} \
    --save ${SAVE} \
    --verbose 1
deactivate
module purge

eval ${VISCORE_ENV_MOD_CMD}
eval ${VISCORE_VENV_CMD}
python ./03b_score_method.py \
    --method ${METHOD} \
    --dataset ${DATASET} \
    --zdim ${ZDIM} \
    --denoised ${DENOISED} \
    --seed ${SEED} \
    --input ${INPUT} \
    --verbose 1
deactivate
module purge
EOS
    fi
done