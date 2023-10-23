import sys
import os
import numpy as np
import torch
from torch import nn, Tensor
import math
import random
from typing import *
sys.path.append(os.path.join(sys.path[0], '../'))

import data_prep
# from experiments import config


def initialize_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# Different dataset input versions
def load_full_dataset_singleCT(ct: str, seed: int) -> Tuple[data_prep.TCGA_dataset]:
    train_ds, val_ds, test_ds = data_prep.get_and_preprocess_data(
        config.DATA_DIR, cancer_types=[ct], train_val_test_splits=[80, 10, 10],
        split_rule='random', random_seed=seed, keep_only_highly_var_genes=False,
        perform_pca=False, perform_patient_normalization=False, perform_gene_normalization=False)
    return train_ds, val_ds, test_ds


def load_highlyVariable_dataset_singleCT(ct: str, seed: int, num_genes: int) -> Tuple[data_prep.TCGA_dataset]:
    train_ds, val_ds, test_ds = data_prep.get_and_preprocess_data(
        config.DATA_DIR, cancer_types=[ct], train_val_test_splits=[80, 10, 10],
        split_rule='random', random_seed=seed, keep_only_highly_var_genes=True,
        highly_var_kwargs=dict(n_top_genes=num_genes), perform_pca=False, perform_patient_normalization=False,
        perform_gene_normalization=False,)
    return train_ds, val_ds, test_ds


def load_PCA_dataset_singleCT(ct: str, seed: int, num_pcs: int) -> Tuple[data_prep.TCGA_dataset]:
    train_ds, val_ds, test_ds = data_prep.get_and_preprocess_data(
        config.DATA_DIR, cancer_types=[ct], train_val_test_splits=[80, 10, 10],
        split_rule='random', random_seed=seed, keep_only_highly_var_genes=False,
        perform_pca=True, perform_patient_normalization=False, perform_gene_normalization=False,
        pca_kwargs=dict(n_components=num_pcs))
    return train_ds, val_ds, test_ds


def load_patient_zscore_dataset_singleCT(ct: str, seed: int) -> Tuple[data_prep.TCGA_dataset]:
    train_ds, val_ds, test_ds = data_prep.get_and_preprocess_data(
        config.DATA_DIR, cancer_types=[ct], train_val_test_splits=[80, 10, 10],
        split_rule='random', random_seed=seed, keep_only_highly_var_genes=False,
        perform_pca=False, perform_patient_normalization=True, perform_gene_normalization=False)
    return train_ds, val_ds, test_ds


def load_gene_zscore_dataset_singleCT(ct: str, seed: int) -> Tuple[data_prep.TCGA_dataset]:
    train_ds, val_ds, test_ds = data_prep.get_and_preprocess_data(
        config.DATA_DIR, cancer_types=[ct], train_val_test_splits=[80, 10, 10],
        split_rule='random', random_seed=seed, keep_only_highly_var_genes=False,
        perform_pca=False, perform_patient_normalization=False, perform_gene_normalization=True)
    return train_ds, val_ds, test_ds


def remove_all_censored_values_from_ds(ds: data_prep.TCGA_dataset) -> data_prep.TCGA_dataset:
    expr = ds.expr_data
    not_censored_mask = ds.surv_mask_data
    observed_ts = ds.surv_obs_data
    ct = ds.cts_data
    N, M = expr.shape

    new_expr = expr[np.where(not_censored_mask == 1)]
    new_nc = not_censored_mask[np.where(not_censored_mask == 1)]
    new_ob = observed_ts[np.where(not_censored_mask == 1)]
    new_ct = ct[np.where(not_censored_mask == 1)]

    assert (new_nc == 1).all(), 'All events should now be observed!'
    return data_prep.TCGA_dataset(new_expr, new_nc, new_ob, new_ct)


# Different annealing strategies
def no_annealing(n_epochs, base_alpha=0.1, base_beta=0.001, base_gamma=1000):
    return ([base_alpha for _ in range(n_epochs)], [base_beta for _ in range(n_epochs)],
            [base_gamma for _ in range(n_epochs)])


def fine_tuning(n_epochs, stageARatio=0.5, base_alpha=0.1, base_beta=0.01, base_gamma=1000):
    stageA = int(n_epochs * stageARatio)
    stageB = n_epochs - stageA
    return([base_alpha for _ in range(stageA)] + [0 for _ in range(stageB)],
           [base_beta for _ in range(stageA)] + [0 for _ in range(stageB)],
           [0 for _ in range(stageA)] + [base_gamma for _ in range(stageB)])


# Early termination conditions
def check_for_early_termination(val_losses):
    ep_range = 250
    if len(val_losses) < ep_range:
        return False
    return val_losses.index(min(val_losses)) < len(val_losses) - ep_range  # terminate if best val_loss was >250 epochs ago


# Different treedepth weighting strategies
def even_level_weighting(tree_depth):
    return {level: 1.0 for level in range(tree_depth)}


def exponentially_decreasing_level_weighting(tree_depth):
    return {level: 0.5 ** level for level in range(tree_depth)}


def exponentially_increasing_level_weighting(tree_depth):
    return {level: (2. ** level) / tree_depth ** 2 for level in range(tree_depth)}


def bellcurve_level_weighting(tree_depth):
    midpoint = (tree_depth - 1) / 2.
    return {level: (1 - abs(midpoint - level) / (tree_depth / 2.)) ** 2 for level in range(tree_depth)}

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)