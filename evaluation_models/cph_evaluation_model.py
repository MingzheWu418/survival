
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from lifelines import CoxPHFitter

from evaluation_models import evaluation_model
from data_prep import TCGA_dataset

class CPHEvaluationModel(evaluation_model.EvaluationModel):
    
    def __init__(self) -> None:
        pass

    def get_best_hyperparams(self,
                             cancer_type: str,
                             n_hvgs: Optional[int] = None,
                             n_PCs: Optional[int] = None,
                             norm_by_gene: Optional[bool] = None,
                             norm_by_patient: Optional[bool] = None) -> Dict:
        return {}
    
    def init_model_hyperparams(self, hyperparams: Dict, input_dim: int, max_surv_time: float, window_size: float, all_data: Dict) -> None:
        self.model = CoxPHFitter()
    
    def train(self, train_ds: TCGA_dataset, val_ds: TCGA_dataset) -> None:
        df = pd.DataFrame(train_ds.expr_data)
        df['time'] = train_ds.surv_obs_data
        df['event'] = train_ds.surv_mask_data

        self.model.fit(df, duration_col = 'time', event_col = 'event')
    
    def predict_survival(self, gene_expressions: np.array) -> Tuple[np.array]:
        predictions = self.model.predict_survival_function(gene_expressions)
        time_index = np.array(predictions.index)
        survival_probas = predictions.values.T
        return time_index, survival_probas
