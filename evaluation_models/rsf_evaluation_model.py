import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from pysurvival.models.survival_forest import RandomSurvivalForestModel

from evaluation_models import evaluation_model
from data_prep import TCGA_dataset

HYPERPARAMS_PATH = './baselines/hyp_optim_results'

class RSFEvaluationModel(evaluation_model.EvaluationModel):
    
    def __init__(self) -> None:
        pass

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
        model_name = 'RSF'
        with open(os.path.join(HYPERPARAMS_PATH, '%s/%s/best_hyperparams-preproc_id=%s.json' % ('BLCA', model_name, preproc_id)), 'r') as f:
            hyperparams = json.load(f)
        return hyperparams
    
    def init_model_hyperparams(self, hyperparams: Dict, input_dim: int, max_surv_time: float, window_size: float, all_data: Dict) -> None:
        self.model = RandomSurvivalForestModel(num_trees=50)
        # print(hyperparams)
        self.min_node_size = hyperparams['min_node_size']

    def train(self, train_ds: TCGA_dataset, val_ds: TCGA_dataset) -> None:
        df = pd.DataFrame(train_ds.expr_data)
        df['time'] = train_ds.surv_obs_data
        df['event'] = train_ds.surv_mask_data

        self.model.fit(
            df.iloc[:, :-2].values,
            df['time'].values,
            df['event'].values, min_node_size=self.min_node_size)
    
    def predict_survival(self, gene_expressions: np.array) -> Tuple[np.array]:
        # predictions = self.model.predict_survival_function(gene_expressions)
        # time_index = np.array(predictions.index)
        # survival_probas = predictions.values.T

        predictions = self.model.predict_survival(gene_expressions)
        time_index = np.array(self.model.times)
        survival_probas = predictions
        # print(predictions.shape)
        # print(time_index.shape)
        # print(survival_probas.shape)
        return time_index, survival_probas
