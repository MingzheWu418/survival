
import sys
import os
sys.path.append(os.path.join(sys.path[0], '../../'))

from copy import deepcopy
import json
import torch
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

from models import NonlinearNonpropotionalHazardPredictor
from evaluation_models import evaluation_model
from data_prep import TCGA_dataset

HYPERPARAMS_PATH = '/gscratch/spe/gvisan01/tcga_CLembedding/baselines/hyp_optim_results'

class MultiSurvEvaluationModel(evaluation_model.EvaluationModel):
    
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
        model_name = 'MultiSurv'

        # for consistency, use hyperparams optimized on BLCA
        with open(os.path.join(HYPERPARAMS_PATH, '%s/%s/best_hyperparams-preproc_id=%s.json' % ('BLCA', model_name, preproc_id)), 'r') as f:
            hyperparams = json.load(f)
        
        return hyperparams
    
    
    def init_model_hyperparams(self, hyperparams: Dict, input_dim: int, max_surv_time: float, window_size: float, all_data: Dict) -> None:
        self.model = NonlinearNonpropotionalHazardPredictor(embedding_dim=input_dim,
                                                            hidden_dims=hyperparams['hidden_dims'],
                                                            dropout_rate=hyperparams['dropout'],
                                                            window_size=window_size,
                                                            max_surv_time=max_surv_time,
                                                            device=self.device).to(self.device)
    
    def train(self, train_ds: TCGA_dataset, valid_ds: TCGA_dataset) -> None:
        train_dataloader = DataLoader(train_ds, batch_size=len(train_ds), shuffle=False, drop_last=False)
        valid_dataloader = DataLoader(valid_ds, batch_size=len(valid_ds), shuffle=False, drop_last=False)

        model = deepcopy(self.model)
        optimizer = optim.Adam(model.parameters(), 0.0005)
        
        best_model = deepcopy(model)
        best_valid_loss = np.inf
        best_model_epoch = 0
        break_big_loop = False
        train_loss_trace = []
        valid_loss_trace = []
        ABSOLUTE_MAX_EPOCHS = 5000
        for epoch in range(ABSOLUTE_MAX_EPOCHS):
            for i, (expr, ct, surv_mask, surv_obs) in enumerate(train_dataloader): # this should just run once
                expr = expr.float().to(self.device)
                surv_mask = surv_mask.to(self.device)
                surv_obs = surv_obs.to(self.device)

                optimizer.zero_grad()

                # forward + backward + optimize
                model.train()
                pred = model(expr)
                loss = model.loss_function(surv_mask, surv_obs, pred)
                train_loss_trace.append(loss.cpu().item())
                loss.backward()
                optimizer.step()

                # compute loss on validation data
                for i, (expr, ct, surv_mask, surv_obs) in enumerate(valid_dataloader): # this should just run once
                    expr = expr.float().to(self.device)
                    surv_mask = surv_mask.to(self.device)
                    surv_obs = surv_obs.to(self.device)

                    model.eval()
                    pred = model(expr)
                    valid_loss = model.loss_function(surv_mask, surv_obs, pred).cpu().item()
                    valid_loss_trace.append(valid_loss)
                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        best_model = deepcopy(model)
                        best_model_epoch = epoch
                
                if len(valid_loss_trace) > 250 and best_model_epoch < (epoch - 250): # break if the validation loss hasn't improved for 250 epochs
                    break_big_loop = True
                    break

            if break_big_loop:
                break

        print('Stopped at epoch %d with %.10f best val loss and %.10f train loss' % (epoch, best_valid_loss, train_loss_trace[-250]), file=sys.stderr)
        print('Best epoch: %d' % (best_model_epoch), file=sys.stderr)
        
        self.model = deepcopy(best_model)

    
    def predict_survival(self, gene_expressions: np.array) -> Tuple[np.array]:
        self.model.eval()
        return self.model.predict_survival(torch.tensor(gene_expressions).float().to(self.device))