# encoding utf-8
import numpy as np
import pandas as pd
from Modeling.utils.utils import Z_Score
from Modeling.utils.utils import generate_dataset


def Data_load(config, timesteps_input, timesteps_output):

    # X = pd.read_csv(config['V_nodes'], header=None).to_numpy(np.float32)
    # X = pd.read_csv(config['V_avg'], header=None).to_numpy(np.float32)
    X = pd.read_csv(config['V_50'], header=None).to_numpy(np.float32)
    # NATree = np.load(config['NATree']).astype(np.float32)
    # NATree = np.load(config['Adj_Matrix']).astype(np.float32)
    NATree = np.load(config['NATree_50']).astype(np.float32)
    # NATree = np.load(config['A  _50']).astype(np.float32)

    X = np.reshape(X, (X.shape[0], X.shape[1], 1)).transpose((1, 2, 0))
    X, X_mean, X_std = Z_Score(X)

    index_1 = int(X.shape[2] * 0.8)
    index_2 = int(X.shape[2] * 0.2)

    train_original_data = X[:, :, :index_1]
    val_original_data = X[:, :, index_1:]

    train_input, train_target = generate_dataset(train_original_data,
                                                 num_timesteps_input=timesteps_input,
                                                 num_timesteps_output=timesteps_output)
    evaluate_input, evaluate_target = generate_dataset(val_original_data,
                                                       num_timesteps_input=timesteps_input,
                                                       num_timesteps_output=timesteps_output)
    data_set = {}
    data_set['train_input'], data_set['train_target'], data_set['eval_input'], data_set[
        'eval_target'], data_set['X_mean'], data_set['X_std'], \
        = train_input, train_target, evaluate_input, evaluate_target, X_mean, X_std

    return NATree, data_set

