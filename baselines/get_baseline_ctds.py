
import sys
import numpy as np
import pandas as pd
import math
import torch

# import dataset
import baselines.utils as utils
from evaluation import Evaluation
from data_prep import get_and_preprocess_data, TCGA_dataset

import lifelines.utils


from torch.utils.data import Dataset, DataLoader
from torch import optim
from copy import deepcopy

from models import NonlinearNonpropotionalHazardPredictor


def deepsurv_res(preproc_kwargs, hyperparams, device, window_size=3.0): 
    
    train_dataset, val_dataset, test_dataset = get_and_preprocess_data(**preproc_kwargs)

    for dataset in [train_dataset, val_dataset, test_dataset]:     
        dataset.combined_df = pd.DataFrame(dataset.expr_data)
        dataset.combined_df['time'] = dataset.surv_obs_data
        dataset.combined_df['event'] = dataset.surv_mask_data

    all_data = {'train': train_dataset.combined_df, 'val':val_dataset.combined_df, 'test':test_dataset.combined_df}

    max_surv_time = np.max(train_dataset.surv_obs_data)
    num_windows = math.ceil(max_surv_time/window_size)
    max_rounded = num_windows * window_size 
    time_points = torch.linspace(window_size/2, max_rounded - (window_size/2), num_windows)
    print(time_points, file=sys.stderr)
    
    hidden_dims = hyperparams['hidden_dims']
    dropout = hyperparams['dropout']

    baseline = utils.Baselines('DeepSurv', all_data, n_neurons=hidden_dims, dropout=dropout)
    baseline.fit(batch_size=len(train_dataset.surv_obs_data), verbose=False) # no batching 

    # eval                  
    performance = Evaluation(model=baseline.model, dataset=baseline.data['val'])
    performance.compute_metrics(time_points) # get ctd 
    
    return performance.c_index_td

def deephit_res(preproc_kwargs, hyperparams, device, window_size=3.0): 
    
    train_dataset, val_dataset, test_dataset = get_and_preprocess_data(**preproc_kwargs)

    for dataset in [train_dataset, val_dataset, test_dataset]:     
        dataset.combined_df = pd.DataFrame(dataset.expr_data)
        dataset.combined_df['time'] = dataset.surv_obs_data
        dataset.combined_df['event'] = dataset.surv_mask_data

    all_data = {'train': train_dataset.combined_df, 'val':val_dataset.combined_df, 'test':test_dataset.combined_df}

    max_surv_time = np.max(train_dataset.surv_obs_data)
    num_windows = math.ceil(max_surv_time/window_size)
    max_rounded = num_windows * window_size 
    time_points = torch.linspace(window_size/2, max_rounded - (window_size/2), num_windows)
    print(time_points, file=sys.stderr)
    
    hidden_dims = hyperparams['hidden_dims']
    dropout = hyperparams['dropout']

    baseline = utils.Baselines('DeepHit', all_data, n_neurons=hidden_dims, dropout=dropout)
    baseline.fit(batch_size=len(train_dataset.surv_obs_data), verbose=False) # no batching 

    # eval                  
    performance = Evaluation(model=baseline.model, dataset=baseline.data['val'])
    performance.compute_metrics(time_points) # get ctd 
    
    return performance.c_index_td


def rsf_res(preproc_kwargs, hyperparams, device, window_size=3.0): 
   
    train_dataset, val_dataset, test_dataset = get_and_preprocess_data(**preproc_kwargs)

    for dataset in [train_dataset, val_dataset, test_dataset]:     
        dataset.combined_df = pd.DataFrame(dataset.expr_data)
        dataset.combined_df['time'] = dataset.surv_obs_data.clip(min=0) # no neg values 
        dataset.combined_df['event'] = dataset.surv_mask_data

    all_data = {'train': train_dataset.combined_df, 'val':val_dataset.combined_df, 'test':test_dataset.combined_df}

    max_surv_time = np.max(train_dataset.surv_obs_data)
    num_windows = math.ceil(max_surv_time/window_size)
    max_rounded = num_windows * window_size
    time_points = torch.linspace(window_size/2, max_rounded - (window_size/2), num_windows)
    print(time_points, file=sys.stderr)
    
    min_node_size = hyperparams['min_node_size']
    
    baseline = utils.Baselines('RSF', all_data, n_trees=50)
    baseline.fit(min_node_size=min_node_size, max_depth=100) # setting max depth to be high so that only hyperparam is min_node_size
    performance = Evaluation(model=baseline.model, dataset=baseline.data['val'])
    performance.compute_metrics(time_points) # get ctd 

    return performance.c_index_td

def multisurv_res(preproc_kwargs, hyperparams, device, window_size=3.0):

    train_ds, valid_ds, test_ds = get_and_preprocess_data(**preproc_kwargs)

    train_dataloader = DataLoader(train_ds, batch_size=len(train_ds), shuffle=False, drop_last=False)
    valid_dataloader = DataLoader(valid_ds, batch_size=len(valid_ds), shuffle=False, drop_last=False)

    embedding_dim = train_ds.expr_data.shape[1]
    hidden_dims = hyperparams['hidden_dims']
    dropout_rate = hyperparams['dropout']
    max_surv_time = np.max(train_ds.surv_obs_data)

    model = NonlinearNonpropotionalHazardPredictor(embedding_dim=embedding_dim,
                                                   hidden_dims=hidden_dims,
                                                   dropout_rate=dropout_rate,
                                                   window_size=window_size,
                                                   max_surv_time=max_surv_time,
                                                   device=device).to(device)

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
            expr = expr.float().to(device)
            surv_mask = surv_mask.to(device)
            surv_obs = surv_obs.to(device)

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
                expr = expr.float().to(device)
                surv_mask = surv_mask.to(device)
                surv_obs = surv_obs.to(device)

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
    # return c_index_td on validation data
    for i, (expr, ct, surv_mask, surv_obs) in enumerate(valid_dataloader): # this should just run once
        expr = expr.float().to(device)
        surv_mask = surv_mask.to(device)
        surv_obs = surv_obs.to(device)

        best_model.eval()
        time_index, surv_pr = best_model.predict_survival(expr)
    
    c_index_td, ibs, inbll = best_model.compute_metrics(surv_pr, surv_mask, surv_obs, time_index)

    print('c_index_td: %.3f' % (c_index_td), file=sys.stderr)
    
    return c_index_td




    
    
    
    
    
    






    
   

                     

    
    

