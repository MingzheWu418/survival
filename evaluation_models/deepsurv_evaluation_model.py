
import sys
import os
import torch
from torch import Tensor
import numpy as np
import pandas as pd
import json

from typing import Dict, Optional, Tuple

from baselines import utils
from evaluation_models import evaluation_model
from data_prep import TCGA_dataset

HYPERPARAMS_PATH = './baselines/hyp_optim_results'

class DeepSurvEvaluationModel:

    def __init__(self) -> None:
        pass

    def set_device(self, device: str) -> None:
        self.device = device

    def get_best_hyperparams(self,
                             cancer_type: str,
                             n_hvgs: Optional[int] = None,
                             n_PCs: Optional[int] = None,
                             norm_by_gene: Optional[bool] = None,
                             norm_by_patient: Optional[bool] = None) -> Dict:
        if n_hvgs is not None:
            assert (n_PCs is None) and (norm_by_gene is None) and (norm_by_patient is None)
            preproc_id = '%d_HVGs' % n_hvgs

        elif n_PCs is not None:
            assert (n_hvgs is None) and (norm_by_gene is None) and (norm_by_patient is None)
            preproc_id = '%d_PCs' % n_PCs

        elif norm_by_gene is not None:
            assert (n_hvgs is None) and (n_PCs is None) and (norm_by_patient is None)
            raise NotImplementedError('Norm by gene optimal hyperparams not implemented')

        elif norm_by_patient is not None:
            assert (n_hvgs is None) and (n_PCs is None) and (norm_by_gene is None)
            raise NotImplementedError('Norm by patient optimal hyperparams not implemented')

        else:
            preproc_id = 'All_Genes'
        
        print('preproc_id: %s' % preproc_id)
        model_name = 'DeepSurv'

        # for consistency, use hyperparams optimized on BLCA
        with open(os.path.join(HYPERPARAMS_PATH, '%s/%s/best_hyperparams-preproc_id=%s.json' % ('BLCA', model_name, preproc_id)), 'r') as f:
            hyperparams = json.load(f)
        
        return hyperparams

    def init_model_hyperparams(self, hyperparams: Dict, input_dim: int, max_surv_time: float, window_size: float, all_data: Dict) -> None:
        self.model = utils.Baselines('DeepSurv', all_data, n_neurons=hyperparams['hidden_dims'], dropout=hyperparams['dropout'])

    def train(self, train_ds: TCGA_dataset, val_ds: TCGA_dataset) -> None:
        self.model.fit(batch_size=len(train_ds.surv_obs_data), verbose=False) # no batching

    def predict_survival(self, gene_expressions: np.array) -> Tuple[np.array]:
        predictions = self.model.model.predict_surv_df(gene_expressions.astype(np.float32)) # using the model within the model...
        time_index = np.array(predictions.index)
        survival_probas = predictions.values.T
        return time_index, survival_probas
