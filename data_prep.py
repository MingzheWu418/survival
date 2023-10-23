
import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA
import torch
from scipy.stats import zscore
from torch.utils.data import Dataset

import matplotlib.pyplot as plt

from typing import List, Dict


def get_and_preprocess_data(data_dir: str = '',
             cancer_types: List[str] = [],
             train_val_test_splits: List[int] = [], # in percentages, so must sum to 100
             split_rule: str = 'by_ct', # random or by_ct
             random_seed: int = 420,
             filter_out_surv_nans: bool = True,
             train_ones_as_zeros: bool = True,
             keep_only_highly_var_genes: bool = True,
             highly_var_kwargs: Dict = {'n_top_genes': 1000},
             perform_pca: bool = False,
             pca_kwargs: Dict = {'n_components': 50},
             perform_patient_normalization: bool = False,
             perform_gene_normalization: bool = False):
    
    
    '''
    Returns 3 TCGA_dataset objects (train, val and test)
    '''
    
    assert np.sum(train_val_test_splits) == 100

    # setup random number generator with seed for reproducibility
    rng = np.random.default_rng(random_seed)

    # collect expression data in patient by gene (P by G) matrix
    expr_data_list = []
    surv_mask_data_list = []
    surv_obs_data_list = []
    cts_list = []
    for ct in cancer_types:
        expr_data = np.loadtxt(data_dir + '{}_expr.txt'.format(ct))
        surv_mask_data = np.loadtxt(data_dir + '{}_maskedSurvival.txt'.format(ct))
        # print(sum(surv_mask_data))
        # print(len(surv_mask_data))
        surv_mask_data = surv_mask_data == 1.0 # convert masks to bools
        surv_obs_data = np.loadtxt(data_dir + '{}_observedSurvival.txt'.format(ct))
        # print(surv_obs_data)
        # filter out patients with nan survival
        if filter_out_surv_nans:
            not_nan_surv_mask = np.logical_and(~np.isnan(surv_obs_data), surv_obs_data >= 0.0)
            expr_data = expr_data[not_nan_surv_mask]
            surv_mask_data = surv_mask_data[not_nan_surv_mask]
            surv_obs_data = surv_obs_data[not_nan_surv_mask]

        expr_data_list.append(expr_data)
        surv_mask_data_list.append(surv_mask_data)
        surv_obs_data_list.append(surv_obs_data)
        # print(surv_obs_data)
        cts_list.append([ct for _ in range(surv_obs_data.shape[0])])

    expr_data_PG = np.vstack(expr_data_list)
    surv_mask_data_P = np.hstack(surv_mask_data_list)
    surv_obs_data_P = np.hstack(surv_obs_data_list)
    cts_data_P = np.hstack(cts_list)

    if train_ones_as_zeros:
        expr_data_PG[expr_data_PG == 1.0] = 0.0

    # print(expr_data_PG.shape)
    # print(surv_mask_data_P.shape)
    # print(surv_obs_data_P.shape)
    # print(expr_data_PG)
    # print(surv_mask_data_P)
    # print(surv_obs_data_P)

    # split data in train, val and test
    if split_rule == 'random':
        train_expr_data, val_expr_data, test_expr_data, train_surv_mask_data, val_surv_mask_data, test_surv_mask_data, train_surv_obs_data, val_surv_obs_data, test_surv_obs_data, train_cts_data, val_cts_data, test_cts_data = random_split(expr_data_PG, surv_mask_data_P, surv_obs_data_P, cts_data_P, train_val_test_splits, rng)
    elif split_rule == 'by_ct':
        train_expr_data, val_expr_data, test_expr_data, train_surv_mask_data, val_surv_mask_data, test_surv_mask_data, train_surv_obs_data, val_surv_obs_data, test_surv_obs_data, train_cts_data, val_cts_data, test_cts_data = by_ct_split(expr_data_PG, surv_mask_data_P, surv_obs_data_P, cts_data_P, train_val_test_splits, rng)
    else:
        raise Exception('Invalid split_type. Must be "random" or "by_ct", not %s.' % (split_rule))
    # preprocess data
    if keep_only_highly_var_genes:
        train_expr_data, val_expr_data, test_expr_data = preprocess__keep_highly_variable_genes(train_expr_data, val_expr_data, test_expr_data, **highly_var_kwargs)
    if perform_pca:
        train_expr_data, val_expr_data, test_expr_data = preprocess__PCA(train_expr_data, val_expr_data, test_expr_data, **pca_kwargs)
    if perform_patient_normalization:
        train_expr_data, val_expr_data, test_expr_data = preprocess__zscore_by_patient(train_expr_data, val_expr_data, test_expr_data)
    if perform_gene_normalization:
        train_expr_data, val_expr_data, test_expr_data = preprocess__zscore_by_gene(train_expr_data, val_expr_data, test_expr_data)

    # create dataset objects
    train_dataset = TCGA_dataset(train_expr_data, train_surv_mask_data, train_surv_obs_data, train_cts_data)
    val_dataset = TCGA_dataset(val_expr_data, val_surv_mask_data, val_surv_obs_data, val_cts_data)
    test_dataset = TCGA_dataset(test_expr_data, test_surv_mask_data, test_surv_obs_data, test_cts_data)

    return train_dataset, val_dataset, test_dataset


def random_split(expr_data_PG, surv_mask_data_P, surv_obs_data_P, cts_data_P, train_val_test_splits, rng):
    P, G = expr_data_PG.shape
    train, val, test = train_val_test_splits
    indices = np.arange(P)

    """
    May 26th:
    Split validation set with only observed instances
    """
    # # print(surv_mask_data_P)
    # observed_indices = indices[surv_mask_data_P == 1]
    # if (len(observed_indices)*100) <= (val*P):
    #     val_indices = observed_indices
    #     train_indices = np.setdiff1d(indices, val_indices)
    # else:
    #     # print(observed_indices)
    #     rng.shuffle(observed_indices)
    #     val_indices = observed_indices[: (val*P)//100]
    #     # print(val_indices)
    #     # print("---")
    #     train_indices = np.setdiff1d(indices, val_indices)
    # test_indices = []


    """
    Jun 3rd:
    Use observed instances for training, and masked for testing
    """
    # observed_indices = indices[surv_mask_data_P == 0]
    # val_indices = observed_indices
    # train_indices = np.setdiff1d(indices, val_indices)
    # test_indices = []

    """
    original
    """
    rng.shuffle(indices)
    train_indices = indices[: (train*P)//100]
    val_indices = indices[(train*P)//100 : ((train+val)*P)//100]
    test_indices = indices[((train+val)*P)//100 :]

    train_expr_data = expr_data_PG[train_indices]
    val_expr_data = expr_data_PG[val_indices]
    test_expr_data = expr_data_PG[test_indices]

    train_surv_mask_data = surv_mask_data_P[train_indices]
    val_surv_mask_data = surv_mask_data_P[val_indices]
    # print(val_surv_mask_data)
    test_surv_mask_data = surv_mask_data_P[test_indices]

    train_surv_obs_data = surv_obs_data_P[train_indices]
    val_surv_obs_data = surv_obs_data_P[val_indices]
    test_surv_obs_data = surv_obs_data_P[test_indices]

    train_cts_data = cts_data_P[train_indices]
    val_cts_data = cts_data_P[val_indices]
    test_cts_data = cts_data_P[test_indices]

    return train_expr_data, val_expr_data, test_expr_data, train_surv_mask_data, val_surv_mask_data, test_surv_mask_data, train_surv_obs_data, val_surv_obs_data, test_surv_obs_data, train_cts_data, val_cts_data, test_cts_data


def by_ct_split(expr_data_PG, surv_mask_data_P, surv_obs_data_P, cts_data_P, train_val_test_splits, rng):
    '''
    Splits the data randomly but keeping the relative proportions of cancer types the same across train, val and test sets.
    '''
    cts = list(set(list(cts_data_P)))
    train, val, test = train_val_test_splits
    train_expr_data, val_expr_data, test_expr_data, train_surv_mask_data, val_surv_mask_data, test_surv_mask_data, train_surv_obs_data, val_surv_obs_data, test_surv_obs_data, train_cts_data, val_cts_data, test_cts_data = [], [], [], [], [], [], [], [], [], [], [], []
    for ct in cts:
        ct_indices = np.squeeze(np.argwhere(cts_data_P == ct))
        ct_P = ct_indices.shape[0]
        rng.shuffle(ct_indices)
        train_indices = ct_indices[: (train*ct_P)//100]
        val_indices = ct_indices[(train*ct_P)//100 : ((train+val)*ct_P)//100]
        test_indices = ct_indices[((train+val)*ct_P)//100 :]

        train_expr_data.append(expr_data_PG[train_indices])
        val_expr_data.append(expr_data_PG[val_indices])
        test_expr_data.append(expr_data_PG[test_indices])

        train_surv_mask_data.append(surv_mask_data_P[train_indices])
        val_surv_mask_data.append(surv_mask_data_P[val_indices])
        test_surv_mask_data.append(surv_mask_data_P[test_indices])

        train_surv_obs_data.append(surv_obs_data_P[train_indices])
        val_surv_obs_data.append(surv_obs_data_P[val_indices])
        test_surv_obs_data.append(surv_obs_data_P[test_indices])

        train_cts_data.append(cts_data_P[train_indices])
        val_cts_data.append(cts_data_P[val_indices])
        test_cts_data.append(cts_data_P[test_indices])
    
    train_expr_data = np.vstack(train_expr_data)
    val_expr_data = np.vstack(val_expr_data)
    test_expr_data = np.vstack(test_expr_data)

    train_surv_mask_data = np.hstack(train_surv_mask_data)
    val_surv_mask_data = np.hstack(val_surv_mask_data)
    test_surv_mask_data = np.hstack(test_surv_mask_data)

    train_surv_obs_data = np.hstack(train_surv_obs_data)
    val_surv_obs_data = np.hstack(val_surv_obs_data)
    test_surv_obs_data = np.hstack(test_surv_obs_data)

    train_cts_data = np.hstack(train_cts_data)
    val_cts_data = np.hstack(val_cts_data)
    test_cts_data = np.hstack(test_cts_data)

    return train_expr_data, val_expr_data, test_expr_data, train_surv_mask_data, val_surv_mask_data, test_surv_mask_data, train_surv_obs_data, val_surv_obs_data, test_surv_obs_data, train_cts_data, val_cts_data, test_cts_data


class TCGA_dataset(Dataset):
    def __init__(self, expr_data, surv_mask_data, surv_obs_data, cts_data):
        super().__init__()
        assert expr_data.shape[0] == surv_obs_data.shape[0]
        assert surv_mask_data.shape[0] == surv_obs_data.shape[0]
        assert cts_data.shape[0] == surv_obs_data.shape[0]
        self.expr_data = expr_data
        self.surv_mask_data = surv_mask_data
        self.surv_obs_data = surv_obs_data
        self.cts_data = cts_data
        self.len = self.expr_data.shape[0]
        self.num_features = self.expr_data.shape[1]
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, i):
        return self.expr_data[i], self.cts_data[i], self.surv_mask_data[i], self.surv_obs_data[i]


def preprocess__keep_highly_variable_genes(train_expr_data, val_expr_data, test_expr_data, **highly_var_kwargs):
    adata = sc.AnnData(X=train_expr_data)
    sc.pp.highly_variable_genes(adata, **highly_var_kwargs) # highly variable genes are determined on training data only
    num_highly_variable = np.sum(adata.var['highly_variable'])
    print(('Keeping %d highly variable genes.' % (num_highly_variable)))

    train_expr_data = train_expr_data[:, adata.var['highly_variable']]
    val_expr_data = val_expr_data[:, adata.var['highly_variable']]
    test_expr_data = test_expr_data[:, adata.var['highly_variable']]

    return train_expr_data, val_expr_data, test_expr_data


def preprocess__PCA(train_expr_data, val_expr_data, test_expr_data, n_components = 50):
    pca = PCA(n_components)
    train_expr_data = pca.fit_transform(train_expr_data) # pca decomposition is fitted on training data only
    try:
        val_expr_data = pca.transform(val_expr_data) # pca decomposition is fitted on training data only
    except:
        val_expr_data = np.empty((0,0))
    try:
        test_expr_data = pca.transform(test_expr_data) # pca decomposition is fitted on training data only
    except:
        test_expr_data = np.empty((0,0))

    return train_expr_data, val_expr_data, test_expr_data


def check_patient_zscore(expr):
    for i in range(len(expr)):
        assert abs(np.mean(np.nan_to_num(expr[i]))) < 1e-3, "Wanted ~0 but got {}".format(np.mean(np.nan_to_num(expr[i])))
        if set(np.unique(np.nan_to_num(expr[i]))) != {0}:
            assert abs(np.std(np.nan_to_num(expr[i])) - 1) < 1e-3, \
                "Wanted ~1 but got {}".format(np.std(np.nan_to_num(expr[i])))


def preprocess__zscore_by_patient(train_expr_data, val_expr_data, test_expr_data):
    zscoretr = zscore(train_expr_data, axis=1)  # preprocessing is all within-patient -> applied independently
    zscoreval = zscore(val_expr_data, axis=1)
    zscoretest = zscore(test_expr_data, axis=1)
    
    check_patient_zscore(zscoretr)
    check_patient_zscore(zscoreval)
    check_patient_zscore(zscoretest)
    
    return np.nan_to_num(zscoretr), np.nan_to_num(zscoreval), np.nan_to_num(zscoretest)  # 0s if no variation


def normalize_z(expr, mu, std):
    N, M = expr.shape
    assert len(mu) == M
    assert len(std) == M
    res = (expr - np.stack([mu for _ in range(N)], axis=0)) / np.stack([std for _ in range(N)], axis=0)
    # remove nans and infs
    res[np.isinf(res)] = 1.  # had no variation in train set
    res = np.nan_to_num(res)
    return res


def preprocess__zscore_by_gene(train_expr_data, val_expr_data, test_expr_data):
    N, M = train_expr_data.shape
    N_val = len(val_expr_data)
    N_test = len(test_expr_data)
    tr_gene_mean = np.mean(train_expr_data, axis=0)  # compute statistics from training data only
    tr_gene_std = np.std(train_expr_data, axis=0)

    transformed_tr = normalize_z(train_expr_data, tr_gene_mean, tr_gene_std)
    transformed_val = normalize_z(val_expr_data, tr_gene_mean, tr_gene_std)
    transformed_test = normalize_z(test_expr_data, tr_gene_mean, tr_gene_std)

    print(np.unique(transformed_tr), np.unique(transformed_val), np.unique(transformed_test))
    # return a 0 if no variation
    return transformed_tr, transformed_val, transformed_test
