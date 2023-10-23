from vae_models import VanillaVAE, Autoencoder
from tqdm import tqdm

import copy
import torch.nn as nn
import os
import sys
import pickle
import random
import json
import numpy as np
import math
import torch
from torch import Tensor
import pandas as pd
import argparse
from tqdm import tqdm
from pycox.evaluation import EvalSurv
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.metrics import roc_auc_score

import models
from evaluation_models import CPHEvaluationModel, MultiSurvEvaluationModel, RSFEvaluationModel, \
    DeepSurvEvaluationModel, DeepHitEvaluationModel, LinearEvaluationModel, MLPEvaluationModel
import data_prep



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_loss(model, dataloader, loss_fn=nn.MSELoss(), conditional=False):
    losses = []
    for batch in dataloader:
        batch = batch[0].to(device)
        # labels = labels.to(device)
        #
        # if conditional:
        #     loss = loss_fn(batch, model(batch, labels))
        # else:
        loss = loss_fn(batch, model(batch))

        losses.append(loss)

    return (sum(losses)/len(losses)).item() # calculate mean

def evaluate(losses, autoencoder, dataloader, vae=False, conditional=False, title=""):
    #     display.clear_output(wait=True)
    if vae and conditional:
        model = lambda x, y: autoencoder(x, y)[0]
    elif vae:
        model = lambda x: autoencoder(x)[0]
    else:
        model = autoencoder

    loss = calculate_loss(model, dataloader, conditional=conditional)
    # show_visual_progress(model, test_dataloader, flatten=flatten, vae=vae, conditional=conditional, title=title)

    losses.append(loss)

def train(net, dataloader, test_dataloader, epochs=5, loss_fn=nn.MSELoss(), title=None):
    optim = torch.optim.Adam(net.parameters())

    train_losses = []
    validation_losses = []

    for i in tqdm(range(epochs)):
        for batch in dataloader:
            batch = batch[0].to(device)

            optim.zero_grad()
            loss = loss_fn(batch, net(batch))
            loss.backward()
            optim.step()
            # print(loss.item())
            train_losses.append(loss.item())
        if title:
            image_title = f'{title} - Epoch {i}'
        evaluate(validation_losses, net, test_dataloader)
        print(validation_losses)


def predict_survival_for_dataset(model, ds):
    # get predictions
    time_index, surv_pr = model.predict_survival(ds.expr_data)
    if type(surv_pr) == Tensor:
        surv_pr = surv_pr.detach().cpu().numpy()

    # interpolation to standard time index
    interp_surv_pr = []
    for i in range(surv_pr.shape[0]):
        interp_surv_pr.append(np.interp(STANDARD_TIME_INDEX, time_index, surv_pr[i, :]))
    interp_surv_pr = np.vstack(interp_surv_pr)

    times = ds.surv_obs_data
    events = ds.surv_mask_data

    return interp_surv_pr, times, events


def initialize_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# def position_encoding_init(n_position, d_pos_vec):
#     ''' Init the sinusoid position encoding table '''
#     # keep dim 0 for padding token position encoding zero vector
#     position_enc = np.array([
#         [pos / np.power(10000, 2*i/d_pos_vec) for i in range(d_pos_vec)]
#         if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])
#
#     position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
#     position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
#     return position_enc

if __name__ == '__main__':

    # print(position_encoding_init(500, 64).shape)

    DEFAULT_DATA_PREPROC_KWARGS = {
        'train_val_test_splits': [80, 10, 10],
        'split_rule': 'random',
        'random_seed': 0,
        'filter_out_surv_nans': True,
        'train_ones_as_zeros': True,
        # default to no preprocessing (All Genes)
        'keep_only_highly_var_genes': False,
        'highly_var_kwargs': {},
        'perform_pca': False,
        'pca_kwargs': {},
        'perform_patient_normalization': False,
        'perform_gene_normalization': False
    }


    PREPROC_ID_TO_KWARGS = {
        'All_Genes': {'perform_pca': False, 'keep_only_highly_var_genes': False}
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='MLP',
                        help='Standard name of the model.')
    parser.add_argument('--pretrain_cancer_type', type=list, default=['BLCA', 'GBM', 'COAD', 'COADREAD', 'HNSC', 'LUSC', 'LUAD', 'SARC', 'KIPAN', 'KIRC'])
    parser.add_argument('--cancer_type', type=str, default='COAD')
    parser.add_argument('--data_dir', type=str, default='datasets/')
    parser.add_argument('--max_iter', type=list, default=[10, 100, 500, 1000, 2000])

    args = parser.parse_args()

    for preproc_id in PREPROC_ID_TO_KWARGS:
        preproc_kwargs = DEFAULT_DATA_PREPROC_KWARGS
        for kwarg in PREPROC_ID_TO_KWARGS[preproc_id]:
            preproc_kwargs[kwarg] = PREPROC_ID_TO_KWARGS[preproc_id][kwarg]
        preproc_kwargs['data_dir'] = args.data_dir
        preproc_kwargs['cancer_types'] = args.pretrain_cancer_type
        print(preproc_kwargs)
        train_ds, valid_ds, test_ds = data_prep.get_and_preprocess_data(**preproc_kwargs)
        in_channels = train_ds.expr_data.shape
        print(in_channels)

    in_channels = train_ds.expr_data.shape[1]
    # model = VanillaVAE(in_channels=in_channels, latent_dim=2048)
    vae_model = Autoencoder(input_size=in_channels, latent_dim=4096, hidden_dims=[8192])
    print(vae_model)

    train_dataset = torch.utils.data.TensorDataset(torch.Tensor(train_ds.expr_data))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 16, shuffle = True, drop_last=False)

    val_dataset = torch.utils.data.TensorDataset(torch.Tensor(valid_ds.expr_data))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 8, shuffle = True, drop_last=False)

    # print(type(next(iter(train_loader))))
    # print(next(iter(train_loader))[0].shape)

    print(torch.Tensor(train_ds.expr_data).shape)
    print(vae_model(torch.Tensor(train_ds.expr_data)))

    train(vae_model, train_loader, val_loader, epochs=3)

    print(vae_model.encode(torch.Tensor(train_ds.expr_data)))
    print(vae_model.encode(torch.Tensor(train_ds.expr_data)).shape)

    for max_iter in args.max_iter:
        print("Now training: " + str(max_iter))
        try:
            NAME_TO_MODEL_DICT = {
                'Linear': 'LinearEvaluationModel',
                'MLP': 'MLPEvaluationModel',
                'CPH': 'CPHEvaluationModel',
                'RSF': 'RSFEvaluationModel',
                'DeepSurv': 'DeepSurvEvaluationModel',
                'DeepHit': 'DeepHitEvaluationModel',
                'MultiSurv': 'MultiSurvEvaluationModel',
                'VAE': 'VAEEvaluationModel',
                'treedepthCL': 'TreedepthCLEvaluationModel',
                'bucketedCL': 'BucketedCLEvaluationModel',
            }

            DEFAULT_DATA_PREPROC_KWARGS = {
                'train_val_test_splits': [80, 10, 10],
                'split_rule': 'random',
                'random_seed': 0,
                'filter_out_surv_nans': True,
                'train_ones_as_zeros': True,
                # default to no preprocessing (All Genes)
                'keep_only_highly_var_genes': False,
                'highly_var_kwargs': {},
                'perform_pca': False,
                'pca_kwargs': {},
                'perform_patient_normalization': False,
                'perform_gene_normalization': False
            }

            PREPROC_ID_TO_KWARGS = {
                '10_PCs': {'perform_pca': True, 'pca_kwargs': {'n_components': 10}},
                '50_PCs': {'perform_pca': True, 'pca_kwargs': {'n_components': 50}},
                '100_PCs': {'perform_pca': True, 'pca_kwargs': {'n_components': 100}},
                '100_HVGs': {'keep_only_highly_var_genes': True, 'highly_var_kwargs': {'n_top_genes': 100}},
                '1000_HVGs': {'keep_only_highly_var_genes': True, 'highly_var_kwargs': {'n_top_genes': 1000}},
                # '10000_HVGs': {'keep_only_highly_var_genes': True, 'highly_var_kwargs': {'n_top_genes': 10000}},
                'All_Genes': {'keep_only_highly_var_genes': False, 'perform_pca': False}
            }

            PREPROC_ID_TO_KWARGS_FOR_EVAL_MODEL = {
                '10_PCs': {'n_PCs': 10},
                '50_PCs': {'n_PCs': 50},
                '100_PCs': {'n_PCs': 100},
                '100_HVGs': {'n_hvgs': 100},
                '1000_HVGs': {'n_hvgs': 1000},
                '10000_HVGs': {'n_hvgs': 10000},
                'All_Genes': {}
            }

            DATA_SHUFFLING_SEED = 42
            BOOTSTRAP_RANDOM_SEED = 42

            # SEED_LIST = np.arange(15)
            SEED_LIST = np.arange(15)
            # SEED_LIST = [42]

            WINDOW_SIZE = 3.0

            # for computing IBS, following the MultiSurv paper
            DROP_LAST_TIMES = 25

            NUM_BOOTSTRAP = 200
            # get standard-time index for cancer-type
            # upper bound max event time in the training set
            # assumes unirom discretization of time
            preproc_kwargs = DEFAULT_DATA_PREPROC_KWARGS
            preproc_kwargs['data_dir'] = args.data_dir
            preproc_kwargs['cancer_types'] = [args.cancer_type]
            train_ds, _, _ = data_prep.get_and_preprocess_data(**preproc_kwargs)
            max_train_survival = np.nanmax(train_ds.surv_obs_data)
            num_windows = math.ceil(max_train_survival / WINDOW_SIZE)
            max_train_survival_rounded_up = num_windows * WINDOW_SIZE
            STANDARD_TIME_INDEX = np.linspace(WINDOW_SIZE / 2, max_train_survival_rounded_up - (WINDOW_SIZE / 2), num_windows)
            print(STANDARD_TIME_INDEX)

            # initialize model
            cox_model = CPHEvaluationModel()
            model = eval(NAME_TO_MODEL_DICT[args.model_name] + '()')
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model.set_device(device)

            # initialize results dict (statistics of metrics)
            results = {'train': {}, 'valid': {}, 'test': {}}
            for split in results:
                results[split]['preprocessing'] = []
                results[split]['C_index_td - mean'] = []
                results[split]['C_index_td - low'] = []
                results[split]['C_index_td - high'] = []
                results[split]['IBS - mean'] = []
                results[split]['IBS - low'] = []
                results[split]['IBS - high'] = []
                results[split]['INBLL - mean'] = []
                results[split]['INBLL - low'] = []
                results[split]['INBLL - high'] = []

            preproc_kwargs['keep_only_highly_var_genes'] = False
            preproc_kwargs['perform_pca'] = False

            vae_train_ds, vae_valid_ds, vae_test_ds = data_prep.get_and_preprocess_data(**preproc_kwargs)
            print(vae_train_ds.expr_data.shape)
            for dataset in [vae_train_ds, vae_valid_ds, vae_test_ds]:
                dataset.expr_data = vae_model.encode(torch.Tensor(dataset.expr_data)).detach().numpy()
                # print(vae_model(torch.Tensor(dataset.expr_data)).detach().numpy().shape)
                # print(dataset.expr_data.shape)
            print(vae_train_ds.expr_data.shape)

            # iterate over the preprocessing steps
            # all_valid_losses = [int(args.max_iter)]
            for preproc_id in PREPROC_ID_TO_KWARGS:
                train_ds = copy.deepcopy(vae_train_ds)
                valid_ds = copy.deepcopy(vae_valid_ds)
                test_ds = copy.deepcopy(vae_test_ds)

                print('Working on: %s' % preproc_id, file=sys.stderr)

                # update preprocessing kwargs and preprocess data
                # preproc_kwargs['keep_only_highly_var_genes'] = False
                # preproc_kwargs['perform_pca'] = False
                preproc_kwargs = DEFAULT_DATA_PREPROC_KWARGS
                for kwarg in PREPROC_ID_TO_KWARGS[preproc_id]:
                    preproc_kwargs[kwarg] = PREPROC_ID_TO_KWARGS[preproc_id][kwarg]
                preproc_kwargs['data_dir'] = args.data_dir
                preproc_kwargs['cancer_types'] = [args.cancer_type]
                print(vae_train_ds.expr_data.shape)
                print(preproc_kwargs)
                if preproc_kwargs['keep_only_highly_var_genes']:
                    train_expr_data, val_expr_data, test_expr_data = data_prep.preprocess__keep_highly_variable_genes(vae_train_ds.expr_data, vae_valid_ds.expr_data, vae_test_ds.expr_data, **preproc_kwargs['highly_var_kwargs'])
                if preproc_kwargs['perform_pca']:
                    train_expr_data, val_expr_data, test_expr_data = data_prep.preprocess__PCA(vae_train_ds.expr_data, vae_valid_ds.expr_data, vae_test_ds.expr_data, **preproc_kwargs['pca_kwargs'])
                # train_ds, valid_ds, test_ds = data_prep.get_and_preprocess_data(**preproc_kwargs)
                try:
                    train_ds.expr_data = train_expr_data
                    valid_ds.expr_data = val_expr_data
                    test_ds.expr_data = test_expr_data
                except:
                    print("No pca nor hvg performed")

                print(train_ds.expr_data.shape)
                # this is for the pycox methods (DeepSurv and DeepHit)
                for dataset in [train_ds, valid_ds, test_ds]:
                    # print(dataset)
                    dataset.combined_df = pd.DataFrame(dataset.expr_data)
                    dataset.combined_df['time'] = dataset.surv_obs_data
                    dataset.combined_df['event'] = dataset.surv_mask_data
                all_data = {'train': train_ds.combined_df, 'val': valid_ds.combined_df, 'test': test_ds.combined_df}

                # get best hyperparams for given cancer_type and preproc_id
                cox_hyperparams = cox_model.get_best_hyperparams(args.cancer_type,
                                                                 **(PREPROC_ID_TO_KWARGS_FOR_EVAL_MODEL[preproc_id]))
                hyperparams = model.get_best_hyperparams(args.cancer_type, **(PREPROC_ID_TO_KWARGS_FOR_EVAL_MODEL[preproc_id]))
                input_dim = train_ds.expr_data.shape[1]
                print('Input dim: %d' % input_dim)

                # train model with different random seeds, save predictions
                survival_probas = {'train': [], 'valid': [], 'test': []}
                times = {'train': [], 'valid': [], 'test': []}
                events = {'train': [], 'valid': [], 'test': []}
                print('Training...', file=sys.stderr)

                observed_count = 0
                masked_count = 0
                observed_to_masked = 0
                masked_to_observed = 0

                for seed in tqdm(SEED_LIST):
                    initialize_random_seed(seed)

                    # initialize model
                    model.init_model_hyperparams(hyperparams, input_dim, max_train_survival, WINDOW_SIZE, all_data)

                    # train model
                    if args.model_name in ["MLP", "Linear"]:
                        # print(train)
                        model.train(train_ds, valid_ds, max_iter=max_iter)
                    else:
                        model.train(train_ds, valid_ds)

                    loss_observed = []
                    loss_masked = []
                    loss = []
                    true_survival = []
                    # get survival functions for all splits
                    for split, ds in zip(['train', 'valid', 'test'], [train_ds, valid_ds, test_ds]):
                        if len(ds.expr_data) == 0:
                            pass
                        elif args.model_name in ['Linear', 'MLP']:
                            # print(ds.expr_data.shape)
                            _, pred_y = model.predict_survival(ds.expr_data)
                            if args.model_name in ['Linear', 'MLP']:
                                pred_y = pred_y.detach().numpy()

                            y = ds.surv_obs_data
                            observed = ds.surv_mask_data
                            # print(y.shape)
                            # loss = np.abs(y.reshape(-1, ) - pred_y.reshape(-1, ))
                            if split == 'train':
                                pred = np.argmax(pred_y, axis=1)
                                # print(pred)
                                # print(observed.astype(int) - pred)
                            #     train_loss = np.abs(y.reshape(-1, ) - pred_y.reshape(-1, ))
                            #     # gmm_in = np.stack([train_loss.reshape(-1,), np.reciprocal(y.reshape(-1,) + 1)])
                            #     # gmm_in = np.stack([train_loss.reshape(-1,), np.log(y.reshape(-1,) + 1)])
                            #     gmm_in = np.stack([train_loss.reshape(-1,), pred_y.reshape(-1,)])
                            #     # gmm_in = np.stack([train_loss.reshape(-1,)])
                            #     # print(train_loss.shape)
                            #     train_observed = ds.surv_mask_data
                            #     train_pred_y = pred_y
                            #     train_y = y
                            # true_survival.append(y.reshape(-1, 1))
                            """ Original """
                            # loss_observed.append(loss[observed.astype(bool)].reshape(-1, 1))
                            # loss_masked.append(loss[np.logical_not(observed.astype(bool))].reshape(-1, 1))
                        else:
                            interp_surv_pr_seed, times_seed, events_seed = predict_survival_for_dataset(model, ds)
                            survival_probas[split].append(interp_surv_pr_seed)
                            times[split].append(times_seed)
                            events[split].append(events_seed)

                    # train_mask = train_ds.surv_mask_data
                    #
                    # loss_observed_final = np.concatenate([item for item in loss_observed])
                    # loss_masked_final = np.concatenate([item for item in loss_masked])
                    # # print(loss_observed_final.shape)
                    # # print(loss_masked_final.shape)
                    # true_survival_final = np.concatenate([item for item in true_survival])
                    # # st, pval = scipy.stats.ttest_ind(loss_observed_final, loss_masked_final, alternative="less")
                    # # if not os.path.exists("./t_test/" + str(args.cancer_type)):
                    # #     os.makedirs("./t_test/" + str(args.cancer_type))
                    # #
                    # # with open('./t_test/' + str(args.cancer_type) + "/" + 't-test.txt', 'a') as f:
                    # #     f.write('----------\n')
                    # #     f.write('Using model:    ' + str(args.model_name) + '\n')
                    # #     f.write('Preprocess ID:  ' + str(preproc_id) + '\n')
                    # #     f.write('Number of Epochs:  ' + str(args.max_iter) + '\n')
                    # #     f.write('t statistic is: ' + str(st) + '\n')
                    # #     f.write('p-value is:     ' + str(pval) + '\n')
                    #
                    # # print(loss_masked_final)
                    # print(loss.shape)
                    #
                    # order = train_pred_y.reshape(-1).argsort()
                    # ranks = order.argsort()
                    # # print(len(ranks))
                    # encoded_pred_y = []
                    # enc = position_encoding_init(len(train_pred_y), 128)
                    #
                    # for ele in ranks:
                    #     encoded_pred_y.append(enc[ele])
                    #
                    # # print(enc)
                    # # print(ranks)
                    # # print(encoded_pred_y)
                    # gmm_in = np.stack([np.concatenate([[train_loss[i]], encoded_pred_y[i]]) for i in range(len(train_loss))])
                    # # print(gmm_in.shape)
                    # """ Compute AUC Score"""
                    # pred = GaussianMixture(n_components=2).fit_predict(gmm_in)
                    # # print(pred)
                    # # print(pred.shape)
                    #
                    # # prob = GaussianMixture(n_components=2).fit(train_loss.reshape(-1, 1)).predict_proba(train_loss.reshape(-1, 1))[:, 0]
                    # # print(prob)
                    # print(roc_auc_score(train_mask.astype(int), pred))
                    # if roc_auc_score(train_mask.astype(int), pred) < 0.5:
                    #     pred = 1 - pred
                    # # print(pred.shape)
                    # # print(gmm_in.T.shape)
                    # # print(np.stack([gmm_in.T, pred.reshape(-1,1)], axis=1))
                    # # pred = pred.astype(bool)
                    # altered_mask = np.where(train_observed.astype(bool), train_mask, pred)
                    # # np.logical_or(train_mask[train_observed], pred[np.logical_not(train_observed)])
                    # observed_count += sum(train_observed)
                    # masked_count += (len(train_observed) - sum(train_observed))
                    # observed_to_masked += sum((train_observed.astype(int) - pred.astype(int)) == 1)
                    # masked_to_observed += sum((train_observed.astype(int) - pred.astype(int)) == -1)
                    #
                    # # print("------")
                    # # print(train_loss)
                    # # print(train_loss[pred])
                    # loss_observed_pred = train_loss[pred.astype(bool)]  # lower loss
                    # # print(loss_observed_pred.shape)
                    # loss_masked_pred = train_loss[np.logical_not(pred.astype(bool))]  # higher loss
                    # # print(loss_masked_pred.shape)
                    # bins = np.logspace(-2, 3, 500)
                    # # plt.hist()
                    # # plt.xlim(min_bound, max_bound)
                    # plt.xscale('log')
                    # plt.yscale('log')

                    # plt.hist(loss_observed_final, bins, alpha=0.5, color="blue", label="observed")
                    # plt.hist(loss_observed_pred, bins, alpha=0.5, color="blue", label="clean")
                    # plt.hist(loss_masked_pred, bins, alpha=0.5, color="green", label="noisy")

                    # print(gmm_in.T)
                    # df = pd.DataFrame(gmm_in.T)
                    # df.columns = ['loss', 'surv']
                    # df['pred'] = pred
                    # df['label'] = train_mask
                    #
                    # sns.histplot(df, x='loss', y='surv', hue='pred', bins=25, label='pred', legend=True)
                    #
                    # plt.title(str(args.cancer_type) + "_" + str(args.model_name) + "_" + str(preproc_id) + "_" + str(
                    #     args.max_iter) + "_iters_loss")
                    # # plt.show()
                    # plt.savefig("./visualization_gmm/" + str(args.cancer_type) + "/" + str(args.cancer_type) + "_" + str(
                    #     args.model_name) + "_" + str(preproc_id) + "_loss_" + str(args.max_iter) + "_iters_observed_masked")
                    # plt.clf()
                    #
                    #
                    # plt.xscale('log')
                    # plt.yscale('log')
                    # sns.histplot(df, x='loss', y='surv', hue='label', bins=25, label='label', legend=True)
                    # plt.title(str(args.cancer_type) + "_" + str(args.model_name) + "_" + str(preproc_id) + "_" + str(
                    #     args.max_iter) + "_iters")
                    # if not os.path.exists("./visualization_survival/" + str(args.cancer_type)):
                    #     os.makedirs("./visualization_survival/" + str(args.cancer_type))
                    # plt.legend()
                    # plt.savefig("./visualization_survival/" + str(args.cancer_type) + "/" + str(args.cancer_type) + "_" + str(
                    #     args.model_name) + "_" + str(preproc_id) + "_pred_survival" + "_" + str(args.max_iter) + "_iters")
                    # plt.clf()



                    #
                    # bins = np.logspace(-2, 3, 50)
                    # # plt.hist()
                    # # plt.xlim(min_bound, max_bound)
                    # plt.xscale('log')
                    # # plt.hist(loss_observed_final, bins, alpha=0.5, color="blue", label="observed")
                    # plt.hist(loss_observed_final, bins, alpha=0.5, color="blue", label="observed")
                    # plt.hist(loss_masked_final, bins, alpha=0.5, color="green", label="masked")
                    # # plt.title(str(args.cancer_type) + "_" + str(args.model_name) + "_" + str(preproc_id))
                    # plt.title(str(args.cancer_type) + "_" + str(args.model_name) + "_" + str(preproc_id) + "_" + str(
                    #     args.max_iter) + "_iters")
                    #
                    # if not os.path.exists("./visualization_survival/" + str(args.cancer_type)):
                    #     os.makedirs("./visualization_survival/" + str(args.cancer_type))
                    # plt.legend()
                    # plt.savefig("./visualization_survival/" + str(args.cancer_type) + "/" + str(args.cancer_type) + "_" + str(
                    #     args.model_name) + "_" + str(preproc_id) + "_pred_survival" + "_" + str(args.max_iter) + "_iters")
                    # plt.clf()

                    # min_bound = np.amin(loss_observed_final)
                    # bins = np.linspace(min_bound, 200, 50)
                    # plt.hist(loss_observed_final, bins, alpha=0.5, color="blue", label="observed")
                    # plt.hist(true_survival_final, bins, alpha=0.5, color="purple", label="true survival")
                    # plt.title(str(args.cancer_type) + "_" + str(args.model_name) + "_" + str(preproc_id) + "_" + str(args.max_iter) + "_iters")
                    #
                    # if not os.path.exists("./visualization_survival/" + str(args.cancer_type)):
                    #     os.makedirs("./visualization_survival/" + str(args.cancer_type))
                    # plt.legend()
                    # plt.savefig("./visualization_survival/" + str(args.cancer_type) + "/" + str(args.cancer_type) + "_" + str(args.model_name) + "_" + str(preproc_id) + "_valid_loss" + "_" + str(args.max_iter) + "_iters")
                    # plt.clf()
                    #
                    # min_bound = np.amin(true_survival_final)
                    # bins = np.linspace(min_bound, 200, 50)
                    # plt.hist(true_survival_final[observed], bins, alpha=0.5, color="blue", label="observed")
                    # plt.hist(true_survival_final[np.logical_not(observed)], bins, alpha=0.5, color="green", label="masked")
                    # plt.title(str(args.cancer_type))
                    # plt.legend()
                    # plt.savefig("./visualization_gmm/" + str(args.cancer_type) + "/" + str(args.cancer_type) + "true_survival")
                    # plt.clf()

                    # Cox regression
                    # First, modify the dataset based on predicted noisy/clean set
                    # print(pred)
                    for dataset in [train_ds, valid_ds, test_ds]:
                        # print(dataset)
                        if dataset == train_ds:
                            dataset.surv_mask_data = pred
                        dataset.combined_df = pd.DataFrame(dataset.expr_data)
                        dataset.combined_df['time'] = dataset.surv_obs_data
                        dataset.combined_df['event'] = dataset.surv_mask_data

                        # print("-------")
                        # print(dataset.surv_mask_data)
                        # print(pred)
                    all_data = {'train': train_ds.combined_df, 'val': valid_ds.combined_df, 'test': test_ds.combined_df}

                    # Then, fit the dataset using cox model
                    initialize_random_seed(seed)

                    # initialize model
                    cox_model.init_model_hyperparams(cox_hyperparams, input_dim, max_train_survival, WINDOW_SIZE, all_data)
                    try:
                        cox_model.train(train_ds, valid_ds)
                        break_loop = 0
                    except:
                        break_loop = 1
                        break

                    empty_set = {}
                    loss_observed = []
                    loss_masked = []
                    loss = []
                    true_survival = []
                    # get survival functions for all splits
                    for split, ds in zip(['train', 'valid', 'test'], [train_ds, valid_ds, test_ds]):
                        if len(ds.expr_data) == 0:
                            empty_set[split] = 1
                        else:
                            empty_set[split] = 0
                            interp_surv_pr_seed, times_seed, events_seed = predict_survival_for_dataset(cox_model, ds)
                            survival_probas[split].append(interp_surv_pr_seed)
                            times[split].append(times_seed)
                            events[split].append(events_seed)
                if not break_loop:
                    for split in ['train', 'valid', 'test']:
                        if empty_set[split] == 1:
                            pass
                        else:
                            survival_probas[split] = np.vstack(survival_probas[split])
                            times[split] = np.hstack(times[split])
                            events[split] = np.hstack(events[split])

                            # print(survival_probas[split].shape)
                            # print(times[split].shape)
                            # print(events[split].shape)

                    ## run bootstrapped evaluation for dataset split, always with the same random seed

                    # initialize results dict (all bootstrapped metrics)
                    results_boot = {'train': {}, 'valid': {}, 'test': {}}
                    for split in results_boot:
                        results_boot[split]['C_index_td'] = []
                        results_boot[split]['IBS'] = []
                        results_boot[split]['INBLL'] = []

                    print('Running bootstrapped evaluation...', file=sys.stderr)
                    for split, ds in zip(['train', 'valid', 'test'], [train_ds, valid_ds, test_ds]):
                        # print(ds)
                        if empty_set[split] == 1:
                            pass
                        else:
                            print(split, file=sys.stderr)
                            initialize_random_seed(BOOTSTRAP_RANDOM_SEED)
                            for i in tqdm(range(NUM_BOOTSTRAP)):
                                # print(survival_probas[split])
                                idxs = np.random.choice(survival_probas[split].shape[0], size=len(ds))
                                surv_pr = survival_probas[split][idxs, :]

                                predictions = {p: surv_pr[p] for p in range(surv_pr.shape[0])}
                                predictions = pd.DataFrame.from_dict(predictions)

                                # Replace automatic index by time points
                                predictions.insert(0, 'time', STANDARD_TIME_INDEX)
                                predictions = predictions.set_index('time')

                                ev = EvalSurv(predictions, times[split][idxs], events[split][idxs], censor_surv='km')
                                c_index_td = ev.concordance_td('adj_antolini')

                                ## The following is the same as MultiSurv's evaluation. They say it's "based on data"
                                ## but it's not clear how. Our data constrained to one cancer type at a time, so it
                                ## may cause a difference?

                                # time_grid = np.array(predictions.index)
                                # Use 100-point time grid based on data
                                time_grid = np.linspace(times[split][idxs].min(), times[split][idxs].max(), 100)
                                # Since the score becomes unstable for the highest times, drop the last
                                # time points?
                                if DROP_LAST_TIMES > 0:
                                    time_grid = time_grid[:-DROP_LAST_TIMES]
                                ibs = ev.integrated_brier_score(time_grid)
                                inbll = ev.integrated_nbll(time_grid)

                                results_boot[split]['C_index_td'].append(c_index_td)
                                results_boot[split]['IBS'].append(ibs)
                                results_boot[split]['INBLL'].append(inbll)

                            # save all bootstrapping results
                            results_dir = 'results/%s/%s' % (split, args.cancer_type)
                            if not os.path.exists(results_dir):
                                os.makedirs(results_dir)
                            if args.model_name in ["Linear", "MLP"]:
                                with open(os.path.join(results_dir, 'metrics_boot-model=%s-preproc_id=%s-iter=%s.pkl' % (
                                        args.model_name, preproc_id, str(max_iter))), 'wb') as f:
                                    pickle.dump(results_boot[split], f)
                            else:
                                with open(os.path.join(results_dir,
                                                       'metrics_boot-model=%s-preproc_id=%s.pkl' % (args.model_name, preproc_id)),
                                          'wb') as f:
                                    pickle.dump(results_boot[split], f)

                            # compute and record bootstrapping statistics
                            results[split]['preprocessing'].append(preproc_id)
                            for metric, metric_vals_list in zip(['C_index_td', 'IBS', 'INBLL'],
                                                                [results_boot[split]['C_index_td'], results_boot[split]['IBS'],
                                                                 results_boot[split]['INBLL']]):
                                num_nan = np.sum(np.isnan(metric_vals_list))
                                if num_nan > 0:
                                    print('Warning, %d/%d nan values in %s for %s split.' % (
                                        num_nan, len(metric_vals_list), metric, split))

                                results[split][metric + ' - mean'].append(np.nanmean(metric_vals_list))
                                results[split][metric + ' - low'].append(np.nanpercentile(metric_vals_list, 5))
                                results[split][metric + ' - high'].append(np.nanpercentile(metric_vals_list, 95))

                            # save results (statistics of metrics)
                            df = pd.DataFrame(results[split])
                            df = df.set_index('preprocessing')
                            results_dir = 'results/%s/%s' % (split, args.cancer_type)
                            if not os.path.exists(results_dir):
                                os.makedirs(results_dir)
                            if args.model_name in ["Linear", "MLP"]:
                                df.to_csv(os.path.join(results_dir, 'metrics_stats-model=%s-iter=%s.csv' % (
                                    args.model_name, str(max_iter))))
                            else:
                                df.to_csv(os.path.join(results_dir, 'metrics_stats-model=%s.csv' % (args.model_name)))


                    if not os.path.exists("./visualization_gmm/" + str(args.cancer_type)):
                        os.makedirs("./visualization_gmm/" + str(args.cancer_type))

                    with open('./visualization_gmm/' + str(args.cancer_type) + "/" + 'altered_count.txt', 'a') as f:
                        f.write('----------\n')
                        f.write('Using model:           ' + str(args.model_name) + '\n')
                        f.write('Preprocess ID:         ' + str(preproc_id) + '\n')
                        f.write('Number of Epochs:      ' + str(max_iter) + '\n')
                        f.write('Total Observed:        ' + str(observed_count) + '\n')
                        f.write('Total Masked:          ' + str(masked_count) + '\n')
                        f.write('Observed -> Masked:    ' + str(observed_to_masked) + '\n')
                        f.write('Masked -> Observed:    ' + str(masked_to_observed) + '\n')
                        # f.write('Observed -> Masked:' + str(len((train_observed - pred) == 1)) + '\n')
                        # f.write('auc:               ' + str("{:.3f}".format(roc_auc_score(observed.astype(int), pred)) + '\n'))


            # filename = "./t_test/" + str(args.cancer_type) + "/" + "mean_loss_linear"
            # col = [key for key in PREPROC_ID_TO_KWARGS]
            # col.insert(0, "Iterations")
            # print(col)
            # print(all_valid_losses)
            # df = pd.DataFrame([all_valid_losses], columns=col)
            # print(df)
            # if os.path.isfile(filename):
            #     df.to_csv(filename, mode='a', header=not os.path.exists(filename))
            # else:
            #     df.to_csv(filename, header=not os.path.exists(filename))
        except:
            pass
