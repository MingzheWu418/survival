
import numpy as np
import torch
from torch import Tensor
from data_prep import TCGA_dataset
from abc import abstractmethod
from typing import Tuple, Dict, Optional



class EvaluationModel:

    @abstractmethod
    def __init__(self) -> None:
        '''
        Do whatever you need to do here, just without parameters.
        Let me know if you need parameters, but ideally you shouldn't need them
        '''
        pass

    def set_device(self, device: str) -> None:
        '''
        Just sets the device so you can put the model and the data on it
        '''
        self.device = device

    @abstractmethod
    def get_best_hyperparams(self,
                             cancer_type: str,
                             n_hvgs: Optional[int] = None,
                             n_PCs: Optional[int] = None,
                             norm_by_gene: Optional[bool] = None,
                             norm_by_patient: Optional[bool] = None) -> Dict:
        pass

    @abstractmethod
    def init_model_hyperparams(self, hyperparams: Dict, input_dim: int, max_surv_time: float, window_size: float, all_data: Dict) -> None:
        '''
        :param hyperparams: output of `get_best_hyperparams()`
        '''

        pass

    @abstractmethod
    def train(self, train_ds: TCGA_dataset, val_ds: TCGA_dataset) -> None:
        pass

    @abstractmethod
    def predict_survival(self, gene_expressions: np.array) -> Tuple[np.array]:
        '''
        :param expression: gene_expressions, or equivalent lower-dim repreresentation, data matrix; (N, F)
        :return: (time_index, in months; T, 
                  survival_probabilities - survival function - for each time point; (N, T))
        
        Note: please make sure that survival_probabilites in in the same order as gene_expressions!
        '''

        pass
    

