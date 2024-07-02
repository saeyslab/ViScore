import os
import json
import time
import argparse
import pickle
import importlib
import numpy as np

## Set hard-coded

output_path = './results'
config = './config.json'

## Parse user args

parser = argparse.ArgumentParser(
    description='Apply DR method to a prepared dataset'
)
parser.add_argument('-m', '--method',   type=str, help='method name')
parser.add_argument('-d', '--dataset',  type=str, help='name (prefix) of scRNA-seq data files')
parser.add_argument('-z', '--zdim',     type=int, help='target embedding dimensionality')
parser.add_argument('-u', '--denoised', type=int, help='whether to use denoised inputs (0/1/2) with 2~use with ViVAE')
parser.add_argument('-s', '--seed',     type=int, help='random seed for reproducibility')
parser.add_argument('-i', '--input',    type=str, help='path to input dataset')
parser.add_argument('-a', '--save',     type=int, help='whether to save the trained DR model (0/1)')
parser.add_argument('-v', '--verbose',  type=int, help='whether to print progress messages (0/1)')
args = parser.parse_args()

method       = args.method
dataset_name = args.dataset
zdim         = args.zdim
use_denoised = args.denoised
if use_denoised==2:
    use_denoised = method.startswith('ViVAE')
else:
    use_denoised = (use_denoised==1)
seed         = args.seed
input_path   = args.input
save_model   = args.save==1
verbose      = args.verbose==1

if verbose:
    print(f'Running DR\n\tmethod: {method}\n\tdataset: {dataset_name}\n\tseed: {seed}\n\t')

## Resolve output path

if not os.path.exists(output_path):
    try:
        os.mkdir(output_path)
    except:
        print('Output path already exists')
res_path = os.path.join(output_path, f'{dataset_name}_{method}_z{zdim}_u{use_denoised}')
if not os.path.exists(res_path):
    try:
        os.mkdir(res_path)
    except:
        print('Result path already exists')

if verbose:
    print(f'Results file path: {res_path}')

## Load inputs

if verbose:
    print('Loading inputs')
input   = np.load(os.path.join(input_path, f'{dataset_name}_input.npy'),           allow_pickle=True)
input_d = np.load(os.path.join(input_path, f'{dataset_name}_input_denoised.npy'),  allow_pickle=True)
knn     = np.load(os.path.join(input_path, f'{dataset_name}_knn.npy'),          allow_pickle=True)
knn_d   = np.load(os.path.join(input_path, f'{dataset_name}_knn_denoised.npy'), allow_pickle=True)
labels  = np.load(os.path.join(input_path, f'{dataset_name}_labels.npy'),       allow_pickle=True)

## Read method config from JSON

if verbose:
    print('Parsing configuration file')
with open(config, encoding='utf-8') as f:
    conf = json.load(f)
methods = conf['methods']
mconf   = methods[method] # method configuration

## Get model class

mmod = importlib.import_module(mconf['package']) # method module
mcl  = getattr(mmod, mconf['model_class']) # model class

## Resolve arguments to constructor and fit_transform method

init_args    = mconf['init_args']
ft_args      = mconf['fit_transform_args']
ft_args['X'] = input_d if use_denoised else input

## Resolve input dimensionality argument

xdimm = mconf['xdim_arg']['method']
if xdimm=='init':
    init_args[mconf['xdim_arg']['name']] = input.shape[1]
elif xdimm=='fit_transform':
    ft_args[mconf['xdim_arg']['name']] = input.shape[1]

## Resolve target embedding dimensionality argument

zdimm = mconf['zdim_arg']['method']
if zdimm=='init':
    init_args[mconf['zdim_arg']['name']] = zdim
elif zdimm=='fit_transform':
    ft_args[mconf['zdim_arg']['name']] = zdim

## Resolve pre-computed k-NNG argument

knnm = mconf['knn_arg']['method']
if knnm in ['init', 'fit_transform']:
    knng = knn_d if use_denoised else knn
    k = mconf['knn_arg']['k']
    if not isinstance(k, str):
        knng = knn[:,:,range(k)]
    knnformat = mconf['knn_arg']['format']
    if knnformat=='tuple':
        knng = tuple(knng)
    elif knnformat=='array':
        knng = knng[0]
    args = init_args if knnm=='init' else ft_args
    args[mconf['knn_arg']['name']] = knng

## Resolve random seed argument

seedm = mconf['seed_arg']['method']
if seedm in ['init', 'fit_transform']:
    args = init_args if seedm=='init' else ft_args
    args[mconf['seed_arg']['name']] = seed

## Init and train model and transform inputs

if verbose:
    print('Applying model')
t0    = time.time()

model = mcl(**init_args)
emb = model.fit_transform(**ft_args)

t1    = time.time()
t     = t1-t0

## Save results of DR

if verbose:
    print('Saving dimensionality reduction results')
np.save(os.path.join(res_path, f'emb_seed{seed}.npy'), emb, allow_pickle=True)
np.save(os.path.join(res_path, f'time_seed{seed}.npy'), t, allow_pickle=True)

## Save trained DR model if requested

if save_model:
    with open(os.path.join(res_path, f'model_seed{seed}.pkl'), 'wb') as output:
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)
