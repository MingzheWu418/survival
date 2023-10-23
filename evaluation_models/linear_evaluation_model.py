
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

from evaluation_models import evaluation_model
from data_prep import TCGA_dataset
from sklearn.linear_model import LinearRegression
import torch

class LinearRegressor(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.l1 = torch.nn.Linear(input_dim, 1)

    def forward(self, x):
        x = torch.FloatTensor(x)
        x = self.l1(x)
        return x


class LinearEvaluationModel(evaluation_model.EvaluationModel):
    
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
        self.model = LinearRegressor(input_dim)
    
    def train(self, train_ds: TCGA_dataset, val_ds: TCGA_dataset, max_iter=100) -> None:
        # x = train_ds.expr_data
        # y = train_ds.surv_obs_data
        # # print(y)
        # self.model.fit(x, y)

        x = train_ds.expr_data
        y = train_ds.surv_obs_data
        observed = train_ds.surv_mask_data
        # print(observed)
        y = torch.FloatTensor(y)
        observed = torch.IntTensor(observed)
        # print(y)

        """
        Reweight samples during training
        """
        alpha = 1
        weight = np.where(observed, 1, alpha)
        optim = torch.optim.Adam(self.model.parameters())
        # mae = torch.nn.L1Loss()
        # mse = torch.nn.MSELoss()
        for i in range(max_iter):
            optim.zero_grad()
            pred_y = self.model(x).reshape(-1,)
            # print("-----")
            # print(pred_y)
            # print(y)
            # loss = weighted_mae_loss(y, pred_y, torch.Tensor(weight))
            loss = self.loss_fn(y, pred_y, observed)
            # print(loss)
            # loss = loss * torch.Tensor(weight)
            # print(loss)
            loss.backward()
            optim.step()
    
    def predict_survival(self, gene_expressions: np.array) -> Tuple[np.array]:
        # predictions = self.model.predict(gene_expressions)

        # time_index = np.array(predictions.index)
        # survival_probas = predictions.values.T

        predictions = self.model(gene_expressions)
        return [], predictions

    def loss_fn(self, gt, pred, observed, alpha_weight=1.0, reg_lambda=0.0, get_individual_losses=False):
        error = pred - gt
        ae = torch.abs(error)
        abs_error = torch.where(observed == 1.0, ae.float(), alpha_weight*torch.nn.functional.relu(-error))

        if get_individual_losses:
            return abs_error

        L2_reg = torch.tensor(0., requires_grad=True)
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                L2_reg = L2_reg + torch.norm(param)

        return torch.mean(abs_error) + reg_lambda * torch.pow(L2_reg, 2)

