import os
import numpy as np
import argparse
import viscore as vs

## Set hard-coded

output_path = './results'
config = './config.json'

## Parse user args

parser = argparse.ArgumentParser(
    description='Score results of a DR method'
)
parser.add_argument('-m', '--method',   type=str, help='method name')
parser.add_argument('-d', '--dataset',  type=str, help='name (prefix) of scRNA-seq data files')
parser.add_argument('-z', '--zdim',     type=int, help='target embedding dimensionality')
parser.add_argument('-u', '--denoised', type=int, help='whether to use denoised inputs (0/1/2) with 2~use with ViVAE')
parser.add_argument('-s', '--seed',     type=int, help='random seed for reproducibility')
parser.add_argument('-i', '--input',    type=str, help='path to input dataset')
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
verbose      = args.verbose==1

if verbose:
    print(f'Scoring DR\n\tmethod: {method}\n\tdataset: {dataset_name}\n\tseed: {seed}\n\t')

## Resolve output path

res_path = os.path.join(output_path, f'{dataset_name}_{method}_z{zdim}_u{use_denoised}')
if not os.path.exists(res_path):
    raise ValueError('DR results do not exist')

if verbose:
    print(f'Results file path: {res_path}')

## Load inputs

if verbose:
    print('Loading inputs')
input   = np.load(os.path.join(input_path, f'{dataset_name}_input.npy'),  allow_pickle=True)
knn     = np.load(os.path.join(input_path, f'{dataset_name}_knn.npy'),    allow_pickle=True)
labels  = np.load(os.path.join(input_path, f'{dataset_name}_labels.npy'), allow_pickle=True)

## Load embedding

if verbose:
    print('Loading embedding')
emb = np.load(os.path.join(res_path, f'emb_seed{seed}.npy'), allow_pickle=True)

## Unsupervised evaluation

if verbose:
    print('Evaluating structure preservation')
s = vs.score(hd=input, ld=emb)
sl = s['Sl']
sg = s['Sg']
rnx = s['RNX']
if verbose:
    print(f'\tlocal SP={np.round(sl, 4)}')
    print(f'\tglobal SP={np.round(sg, 4)}')
np.save(os.path.join(res_path, f'sp_local_seed{seed}.npy'),  sl, allow_pickle=True)
np.save(os.path.join(res_path, f'sp_global_seed{seed}.npy'), sg, allow_pickle=True)
np.save(os.path.join(res_path, f'rnx_curve_seed{seed}.npy'), rnx, allow_pickle=True)

## Supervised evaluation

if verbose:
    print('Evaluating population embedding errors')
s = vs.xnpe(hd=input, ld=emb, annot=labels, knn=knn)
np.save(os.path.join(res_path, f'xnpe_seed{seed}.npy'), s, allow_pickle=True)
