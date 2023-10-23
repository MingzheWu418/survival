
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

from evaluation_models import evaluation_model
from data_prep import TCGA_dataset
from sklearn.neural_network import MLPRegressor
import torch


def kl_div(p, q):
    return (p * ((p + 1e-5).log() - (q + 1e-5).log())).sum(-1)

class MLPRegressor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim: list):
        super().__init__()
        self.l1 = torch.nn.Linear(input_dim, hidden_dim[0])
        self.non_lin1 = torch.nn.ReLU()
        self.layers = torch.nn.ModuleList()
        for i in range(len(hidden_dim)-2):
            self.layers.append(torch.nn.Linear(hidden_dim[i], hidden_dim[i+1]))
            self.layers.append(torch.nn.ReLU())
        self.l2 = torch.nn.Linear(hidden_dim[-1], 1)

    def forward(self, x):
        x = torch.FloatTensor(x)
        x = self.l1(x)
        x = self.non_lin1(x)
        for layer in self.layers:
            x = layer(x)
        x = self.l2(x)
        return x

class MLPClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim: list):
        super().__init__()
        self.l1 = torch.nn.Linear(input_dim, hidden_dim[0])
        self.non_lin1 = torch.nn.ReLU()
        self.layers = torch.nn.ModuleList()
        for i in range(len(hidden_dim)-2):
            self.layers.append(torch.nn.Linear(hidden_dim[i], hidden_dim[i+1]))
            self.layers.append(torch.nn.ReLU())
        self.l2 = torch.nn.Linear(hidden_dim[-1], 2)

    def forward(self, x):
        x = torch.FloatTensor(x)
        x = self.l1(x)
        x = self.non_lin1(x)
        for layer in self.layers:
            x = layer(x)
        x = self.l2(x)
        return x


class CombModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim: list, num_models=2):
        super().__init__()
        self.models = torch.nn.ModuleList()
        self.num_models = num_models
        for i in range(num_models):
            self.model = MLPClassifier(input_dim, hidden_dim)
            self.models.append(self.model)

    def forward(self, x):
        # for i in range(len(self.models)):
        logits = [model(x) for model in self.models]
        probs = [torch.nn.functional.softmax(logit, dim=-1) for logit in logits]
        self.probs = probs
        # print(torch.nn.functional.softmax(self.models[0](x), dim=-1) - probs[0])
        # print(self.models[0](x))
        # return probs[0]
        return logits[0]

    def calc_agreement_loss(self, label):
        avg_prob = torch.stack(self.probs, dim=0).mean(0)
        mask = (label.view(-1) != -1).to(self.probs[0])
        reg_loss = sum([kl_div(avg_prob, prob) * mask for prob in self.probs]) / self.num_models
        reg_loss = reg_loss.sum() / (mask.sum() + 1e-3)
        return reg_loss

class MLPEvaluationModel(evaluation_model.EvaluationModel):

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
        # self.model = MLPRegressor(input_dim, [64])
        self.model = CombModel(input_dim=input_dim, hidden_dim=[64], num_models=4)

    def train(self, train_ds: TCGA_dataset, val_ds: TCGA_dataset, max_iter=100) -> None:
        x = train_ds.expr_data
        print(x.shape)
        y = train_ds.surv_obs_data
        observed = train_ds.surv_mask_data
        # print(observed)
        y = torch.FloatTensor(y)
        observed = torch.IntTensor(observed)
        # print(y)

        """
        Reweight samples during training
        """
        alpha = 0.05
        # weight = np.where(observed, 1, alpha)
        optim = torch.optim.Adam(self.model.parameters())
        # mae = torch.nn.L1Loss()
        # mse = torch.nn.MSELoss()
        for i in range(max_iter):
            optim.zero_grad()
            pred_y = self.model(x)
            # print("-----")
            # print(pred_y)
            # print(torch.argmax(pred_y, 1))
            # print(observed)
            # loss = weighted_mae_loss(y, pred_y, torch.Tensor(weight))
            # loss = self.loss_fn(y, pred_y, observed)
            # print("------")
            # print(pred_y.shape)
            # print(observed.shape)
            L = torch.nn.CrossEntropyLoss()
            # print(i)
            # print(L(pred_y, observed.to(torch.long)))
            # print(self.model.calc_agreement_loss(observed))
            loss = L(pred_y, observed.to(torch.long)) + alpha * self.model.calc_agreement_loss(observed)
            # loss = loss * torch.Tensor(weight)
            # print(loss)
            loss.backward()
            optim.step()
    
    def predict_survival(self, gene_expressions: np.array) -> Tuple[np.array]:
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


def weighted_mae_loss(input, target, weight):
    # print(weight * torch.abs((input - target)))
    return torch.sum(weight * torch.abs((input - target)))
