
import numpy as np
import pandas as pd
import math
import torch
from torch import nn, Tensor
from typing import List, Tuple, Optional
from pycox.evaluation import EvalSurv


class NonlinearNonpropotionalHazardPredictor(nn.Module):

    def __init__(self,
                 embedding_dim: int, # input dim
                 hidden_dims: List[int],
                 window_size: float,
                 max_surv_time: float,
                 windows_partition_method: str = 'uniform',
                 dropout_rate: float = 0.0,
                 use_batch_norm: bool = False,
                 device: str = 'cpu'):
        
        '''
        We model the survival time of a patient by predicting their hazard function: i.e. the probabability of death during a given time 
        window, conditioned on the fact that they survived through all the previous time windows. There are `num_windows` windows spaced
        between `0` and `max_surv_time`, spread according to the rule indicated by `windows_partition_method`.

        Censored patients with a censoring time are given the credit to have survived through the time windows before the censoring time.
        They are given credit to have survived the interval where the censoring time occurs if the censoring time is after the midpoint
        of the window.

        If a patient's survival time is greater than `max_surv_time`, then the patient is just given credit for surviving all windows until
        `max_surv_time`.

        We use a fully connected MLP predictor with input dimension equal to `embedding_dim` and output dimension equal to `num_windows`.
        '''

        super().__init__()

        self.window_size = window_size
        self.num_windows = math.ceil(max_surv_time / window_size)
        self.device = device

        ## build survivial prediction module
        layers = []
        prev_dim = embedding_dim
        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(prev_dim, hidden_dims[i]))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dims[i], momentum=0.01, eps=0.001))
            layers.append(nn.ReLU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dims[i]
        
        # output layer
        layers.append(nn.Linear(prev_dim, self.num_windows))
        layers.append(nn.Sigmoid())
        self.hazard_predictor = nn.Sequential(*layers)


        ## build module to encode float survival times into vector of windows
        if windows_partition_method == 'uniform': # windows of fixed size

            # round up maximum survival time if it's not a multiple of window_size
            self.max_surv_time = self.num_windows * self.window_size
            
            def place_into_windows(survival_times: Tensor, is_observed: Tensor) -> Tensor:
                survival_times = survival_times.view(-1, 1)

                edges = torch.linspace(0.0, self.max_surv_time, self.num_windows+1)[1:].repeat(survival_times.shape[0], 1)
                edges = edges.to(self.device)
                stepsize = edges[0][1] - edges[0][0]
                survival_times_plus_stepsize = survival_times + stepsize
                bool_windows = torch.logical_and(edges > survival_times, edges < survival_times_plus_stepsize)
                onehot_windows = torch.where(bool_windows, 1.0, 0.0)
                
                # set all values to 0 for censored patients since we don't know when they actually died
                onehot_windows[~is_observed, :] = 0.0

                bool_windows_after_death = torch.logical_not(edges < survival_times_plus_stepsize)

                # this is for censored patients. give credit to patient for surviving the window where
                # the censoring time occurs only if it's past the window's midpoint
                edges_censored = edges + (0.5 * stepsize)
                bool_windows_after_censoring = torch.logical_not(edges_censored < survival_times_plus_stepsize)

                # put everything onto device
                onehot_windows = onehot_windows.to(self.device)
                bool_windows_after_death = bool_windows_after_death.to(self.device)
                bool_windows_after_censoring = bool_windows_after_censoring.to(device)
                return onehot_windows, bool_windows_after_death, bool_windows_after_censoring

        else:
            raise NotImplementedError('{} method to create survival windows not implemented.'.format(windows_partition_method))

        self.place_into_windows = place_into_windows

    def forward(self, embeddings: Tensor) -> Tensor:
        return self.hazard_predictor(embeddings)
    
    def predict_survival(self, embeddings: Tensor) -> Tensor:
        '''
        Gets survival function from hazard function using law of conditional probability.
        '''
        edges = np.linspace(0.0, self.max_surv_time, self.num_windows+1)
        time_points = edges[:-1] + ((edges[1:] - edges[:-1]) / 2)

        one = torch.Tensor([1.0]).type(torch.float).to(self.device)
        return time_points, torch.cumprod(one - self.hazard_predictor(embeddings), dim=-1)
    
    def compute_metrics(self, survival_probabilities: Tensor, is_observed: Tensor, gt_survival: Tensor, time_points: Tensor, drop_last_times: int = 25) -> Tuple:
        assert survival_probabilities.shape[0] == is_observed.shape[0]
        assert survival_probabilities.shape[0] == gt_survival.shape[0]
        n_patients = survival_probabilities.shape[0]
        
        survival_probabilities = survival_probabilities.detach().cpu().numpy()
        times = gt_survival.detach().cpu().numpy()
        events = is_observed.detach().cpu().numpy()
    
        ## prepare data for pycox's EvalSurv
        # patients are identified by their order, so it's important we preserve
        # the order of patients across evaluations!!
        predictions = {p: survival_probabilities[p] for p in range(n_patients)}
        predictions = pd.DataFrame.from_dict(predictions)

        # Replace automatic index by time points
        predictions.insert(0, 'time', time_points)
        predictions = predictions.set_index('time')

        ev = EvalSurv(predictions, times, events, censor_surv='km')

        c_index_td = ev.concordance_td('adj_antolini')

        ## The following is the same as MultiSurv's evaluation. They say it's "based on data"
        ## but it's not clear how. Our data constrained to one cancer type at a time, so it
        ## may cause a difference?
        # time_grid = np.array(predictions.index)
        # Use 100-point time grid based on data
        time_grid = np.linspace(times.min(), times.max(), 100)
        # Since the score becomes unstable for the highest times, drop the last
        # time points?
        if drop_last_times > 0:
            time_grid = time_grid[:-drop_last_times]
        ibs = ev.integrated_brier_score(time_grid)
        inbll = ev.integrated_nbll(time_grid)

        return c_index_td, ibs, inbll
    
    def loss_function(self, is_observed: Tensor, gt_survival: Tensor, pred_survival: Tensor) -> Tensor:
        '''
        gt_survival: original survival data (Tensor of floats)
        pred_survival: conditional hazard probabilities

        Negative log likelihood of the patient's true survival time under the model
        '''
        gt_survival_windows, bool_windows_after_death, bool_windows_after_censoring = self.place_into_windows(gt_survival, is_observed)
        gt_survival_windows.to(self.device)

        # little trick: set predicted hazard values to 0.0 for all windows after death so we can use
        # batched binary cross entropy. 
        zero = torch.Tensor([0.0]).type(torch.float).to(self.device)
        observed = torch.where(bool_windows_after_death, zero, pred_survival)
        censored = torch.where(bool_windows_after_censoring, zero, pred_survival)
        pred_survival = torch.where(is_observed.view(-1, 1).repeat(1, self.num_windows), observed, censored)
        
        pred_survival.to(self.device)

        return nn.functional.binary_cross_entropy(pred_survival, gt_survival_windows)