
import sys
import os

# training functions
import data_prep
from evaluation_models import DeepSurvEvaluationModel

sys.path.append(os.path.join(sys.path[0], '../'))
import os
import random
import json
import numpy as np
import torch
import pandas as pd
import argparse
from tqdm import tqdm

from get_baseline_ctds import *

DEFAULT_DATA_PREPROC_KWARGS = {
    'train_val_test_splits': [80, 10, 10],
    'split_rule': 'random',
    'random_seed': 0,
    'filter_out_surv_nans': True,
    'train_ones_as_zeros': True,
    
    # default to no preprocessing (All Genes)
    'keep_only_highly_var_genes': False,
    'highly_var_kwargs': {},
    'perform_pca': False,
    'pca_kwargs': {},
    'perform_patient_normalization': False,
    'perform_gene_normalization': False
}

PREPROC_ID_TO_KWARGS = {
    '10_PCs': {'perform_pca': True, 'pca_kwargs': {'n_components': 10}},
    '50_PCs': {'perform_pca': True, 'pca_kwargs': {'n_components': 50}},
    '100_PCs': {'perform_pca': True, 'pca_kwargs': {'n_components': 100}},
    '100_HVGs': {'keep_only_highly_var_genes': True, 'highly_var_kwargs': {'n_top_genes': 100}},
    '1000_HVGs': {'keep_only_highly_var_genes': True, 'highly_var_kwargs': {'n_top_genes': 1000}},
    '10000_HVGs': {'keep_only_highly_var_genes': True, 'highly_var_kwargs': {'n_top_genes': 10000}},
    'All_Genes': {}
}

MODEL_NAMES_TO_TRAINING_FUNCTIONS = {
    'RSF': rsf_res,
    'DeepSurv': deepsurv_res,
    'DeepHit': deephit_res,
    'MultiSurv': multisurv_res
}

NN_METHODS = ['DeepSurv', 'DeepHit', 'MultiSurv']

def initialize_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_hyperparam_combos(hyperparams_grid):
    params = []
    values_list = []
    for param in hyperparams_grid:
        print(param)
        params.append(param)
        values_list.append(hyperparams_grid[param])  
    list_of_permutations = [list(x) for x in np.array(np.meshgrid(*values_list)).T.reshape(-1,len(values_list))]
    
    hyperparams_list = []
    for perms in list_of_permutations:
        adict = {}
        for i, value in enumerate(perms):
            adict[params[i]] = value
                
        hyperparams_list.append(adict)
    return hyperparams_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../datasets/')
    parser.add_argument('--cancer_type', type=str, default='GBM')
    parser.add_argument('--model_name', type=str, default='MultiSurv')
    parser.add_argument('--preproc_id', type=str, default='100_HVGs')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    assert args.model_name in MODEL_NAMES_TO_TRAINING_FUNCTIONS
    assert args.preproc_id in PREPROC_ID_TO_KWARGS

    # get list of hyperparams
    with open('hyperparameters_to_try.json', 'r') as f:
        if args.model_name in NN_METHODS:
            hyperparams_grid = json.load(f)['NN Methods'][args.preproc_id]
        else:
            hyperparams_grid = json.load(f)[args.model_name][args.preproc_id]
        
        hyperparams_list = get_hyperparam_combos(hyperparams_grid)

    # get data preprocessing kwargs
    preproc_kwargs = DEFAULT_DATA_PREPROC_KWARGS
    for kwarg in PREPROC_ID_TO_KWARGS[args.preproc_id]:
        preproc_kwargs[kwarg] = PREPROC_ID_TO_KWARGS[args.preproc_id][kwarg]
    preproc_kwargs['data_dir'] = args.data_dir
    preproc_kwargs['cancer_types'] = [args.cancer_type]

    # train model with all combinations of hyperparams, with 5 different random seeds
    # select hyperparameters that, on average, have the lowest rank
    num_seeds = 5
    seeds_list = list(np.arange(num_seeds))
    training_function = MODEL_NAMES_TO_TRAINING_FUNCTIONS[args.model_name]
    cum_valid_c_index_td = np.zeros((len(hyperparams_list),))
    for i, seed in enumerate(seeds_list):
        print('%d/%d' % (i+1, len(seeds_list)))
        valid_c_index_td = []
        for hyperparams in tqdm(hyperparams_list):
            # print(hyperparams)
            c_index_td = training_function(preproc_kwargs, hyperparams, args.device)
            valid_c_index_td.append(c_index_td)

        cum_valid_c_index_td += np.array(valid_c_index_td)

    mean_valid_c_index_td = cum_valid_c_index_td / num_seeds

    print(mean_valid_c_index_td, file=sys.stderr)
    print('Best c_index_td: %.3f' % (np.max(mean_valid_c_index_td)), file=sys.stderr)

    best_hyperparams = hyperparams_list[np.argmax(mean_valid_c_index_td)]

    # save hyperparameters to json file
    output_dir = 'hyp_optim_results/%s/%s' % (args.cancer_type, args.model_name)
    output_file = os.path.join(output_dir, 'best_hyperparams-preproc_id=%s.json' % (args.preproc_id))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file , 'w+') as f:
        json.dump(best_hyperparams, f, indent=1)