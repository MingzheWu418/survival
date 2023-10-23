from data_prep import get_and_preprocess_data
import pandas as pd
import numpy as np
from baseline_models import Baselines
# from utils import load_full_dataset_singleCT
from evaluation import Evaluation
import torch
import random

# torch.set_default_dtype(torch.float64)
data = {}
ds_name = ['train', 'val', 'test']
ctd = []
ibs = []

SEED_LIST = np.arange(15)
for epoch in range(15):
    seed = SEED_LIST[epoch]
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    ds = get_and_preprocess_data(data_dir="./datasets/", cancer_types=["BLCA"],
                                 train_val_test_splits=[80, 10, 10],
                                 split_rule='random',
                                 random_seed=seed,
                                 keep_only_highly_var_genes=True,
                                 highly_var_kwargs={'n_top_genes': 100},
                                 # perform_pca=True,
                                 # pca_kwargs={'n_components': 50},
                                 )

    for i in range(len(ds)):
        # print(ds.expr_data.shape)
        df_time = pd.DataFrame(ds[i].surv_obs_data, columns=['time'])
        df_event = pd.DataFrame(ds[i].surv_mask_data, columns=['event'])
        col = []
        for j in range(ds[i].num_features):
            col.append("x"+str(j))
        df_x = pd.DataFrame(ds[i].expr_data, columns=col)
        data[ds_name[i]] = pd.concat([df_time, df_event, df_x], axis=1)

    # print(data)

    # Able to use: CPH and RSF

    ''' CPH '''
    base_model = Baselines("CPH", data)
    base_model.fit()

    ''' RSF '''
    # base_model = Baselines("RSF", data, n_trees=100)
    # base_model.fit()

    ''' DeepSurv '''
    # base_model = Baselines("DeepSurv", data, n_trees=100, n_neurons=100)
    # base_model.fit(batch_size=100)

    ''' DeepHit '''
    # base_model = Baselines("DeepHit", data, n_trees=100, n_neurons=100)
    # base_model.fit(batch_size=100)

    ''' CoxTime '''
    # base_model = Baselines("CoxTime", data, n_trees=100, n_neurons=100)
    # base_model.fit(batch_size=100)

    evaluator = Evaluation(base_model.model, data['test'])
    evaluator.compute_metrics()
    evaluator.show_results()
    # print(evaluator.results)
    for algo, res in evaluator.results.items():
        # print(algo + ' ' * (10 - len(algo)) + res)
        if algo == 'Ctd':
            ctd.append(float(res))
        if algo == "IBS":
            ibs.append(float(res))


ctd = np.asarray(ctd)
print("Ctd: mean " + str(np.mean(ctd)) + ", std " + str(np.std(ctd)))

ibs = np.asarray(ibs)
print("IBS: mean " + str(np.mean(ibs)) + ", std " + str(np.std(ibs)))