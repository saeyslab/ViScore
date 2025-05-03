import time
import scanpy as sc
import numpy as np
import pandas as pd
import os
import viscore as vs
import urllib.request
import pathlib
import argparse

parser = argparse.ArgumentParser(
    description='Prepare scRNA-seq datasets for benchmark'
)
parser.add_argument('-n', '--name',     type=str, default='***ALL***', help='name of single dataset to prepare (otherwise all will be prepared)')
parser.add_argument('-d', '--datadir',  type=str, help='path to data')
parser.add_argument('-D', '--datasets', type=str, help='path to datasets CSV')
args = parser.parse_args()

name    = args.name
datadir = args.datadir
datasets = args.datasets

all_datasets = (name=='***ALL***')

k = 150

print('Reading datasets table')

dtable = pd.read_csv(datasets)
n = dtable.shape[0]

output_path = datadir
if not os.path.exists(output_path):
    os.mkdir(output_path)

def process_dataset(idx):
    dataset = dtable['Name'][idx]
    print(f'Preparing "{dataset}" dataset')

    if not (
        os.path.exists(os.path.join(output_path, f'{dataset}.h5ad')) and
        os.path.exists(os.path.join(output_path, f'{dataset}_input.npy')) and
        os.path.exists(os.path.join(output_path, f'{dataset}_knn.npy')) and
        os.path.exists(os.path.join(output_path, f'{dataset}_input_denoised.npy')) and
        os.path.exists(os.path.join(output_path, f'{dataset}_knn_denoised.npy')) and
        os.path.exists(os.path.join(output_path, f'{dataset}_labels.npy')) and
        os.path.exists(os.path.join(output_path, f'{dataset}_unassigned.npy'))
    ):

        fname_pc    = os.path.join(output_path, f'{dataset}_input.npy')
        fname_labs  = os.path.join(output_path, f'{dataset}_labels.npy')
        fname_unass = os.path.join(output_path, f'{dataset}_unassigned.npy')

        pc_labs_ready = os.path.exists(fname_pc) and os.path.exists(fname_labs) and os.path.exists(fname_unass)
        if pc_labs_ready:

            print('\tUsing existing PCA-transformed data and annotation')
            pc = np.load(fname_pc, allow_pickle=True)
            labels = np.load(fname_labs, allow_pickle=True)
            unassigned = np.load(fname_unass, allow_pickle=True)
        else:

            print('\tDownloading data')
            fname_h5 = os.path.join(output_path, f'{dataset}.h5ad')
            urllib.request.urlretrieve(dtable['DatasetLink'][idx], fname_h5)

            print('\tImporting data')
            hd = sc.read_h5ad(fname_h5)
            filter_cond = dtable['FilteringCondition'][idx]
            filter_val  = dtable['FilteringValue'][idx]
            if isinstance(filter_cond, str) and isinstance(filter_val, str):
                print('\tFiltering by condition')
                hd = hd[hd.obs[filter_cond]==filter_val]

            print('\tScaling and PCA')
            if not os.path.exists(os.path.join(output_path, f'{dataset}_input.npy')):
                sc.pp.scale(hd, max_value=10.)
                sc.tl.pca(hd, svd_solver='arpack', n_comps=100)
                pc = hd.obsm['X_pca']
                np.save(os.path.join(output_path, f'{dataset}_input.npy'), pc, allow_pickle=True)
            else:
                pc = np.load(os.path.join(output_path, f'{dataset}_input.npy'), allow_pickle=True)

            print('\tExtracting annotation')
            unassigned = dtable['UnassignedLabel'][idx]
            colname = dtable['LabelsColumn'][idx]
            labels = hd.obs[colname]
            np.save(os.path.join(output_path, f'{dataset}_labels.npy'), labels, allow_pickle=True)
            np.save(os.path.join(output_path, f'{dataset}_unassigned.npy'), unassigned, allow_pickle=True)

            print('\tDeleting H5AD file')
            del hd
            pathlib.Path.unlink(fname_h5)

        print('\tConstructing k-NNG')
        t0  = time.time()
        knn = vs.make_knn(x=pc, k=k, fname=os.path.join(output_path, f'{dataset}_knn.npy'), verbose=False)
        t1  = time.time()
        t   = t1-t0
        np.save(os.path.join(output_path, f'{dataset}_knn_time.npy'), t, allow_pickle=True)

        print('\tDenoising PCA input')
        pc_d = vs.smooth(pc, knn, k=100, coef=1., n_iter=1)
        np.save(os.path.join(output_path, f'{dataset}_input_denoised.npy'), pc_d, allow_pickle=True)
        del pc

        print('\tRe-constructing k-NNG on denoised data')
        t0  = time.time()
        knn = vs.make_knn(x=pc_d, k=k, fname=os.path.join(output_path, f'{dataset}_knn_denoised.npy'), verbose=False)
        t1  = time.time()
        t   = t1-t0
        np.save(os.path.join(output_path, f'{dataset}_knn_denoised_time.npy'), t, allow_pickle=True)
        del knn

if all_datasets:
    for idx in range(n):
        process_dataset(idx)
else:
    idx = np.where(dtable['Name']==name)[0][0]
    process_dataset(idx)
