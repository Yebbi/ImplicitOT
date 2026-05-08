import pandas as pd
import pickle
import numpy as np
from collections import Counter
from os.path import join

class HEPMASS:
    """
    The HEPMASS data set.
    http://archive.ics.uci.edu/ml/datasets/HEPMASS
    """

    class Data:

        def __init__(self, data):

            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self, normalization_type):
        path = '/home/yesom/Codes/Implicit_OT/data/hepmass'
        # path = datasets.root + 'hepmass/'
        trn, val, tst = load_data_no_discrete_normalised_as_array(path, normalization_type)

        self.trn = self.Data(trn)
        self.val = self.Data(val)
        self.tst = self.Data(tst)

        self.n_dims = self.trn.x.shape[1]


def load_data(path):

    data_train = pd.read_csv(filepath_or_buffer=join(path, "1000_train.csv"), index_col=False)
    data_test = pd.read_csv(filepath_or_buffer=join(path, "1000_test.csv"), index_col=False)

    return data_train, data_test


def load_data_no_discrete(path):
    """
    Loads the positive class examples from the first 10 percent of the dataset.
    """
    data_train, data_test = load_data(path)

    # Gets rid of any background noise examples i.e. class label 0.
    data_train = data_train[data_train[data_train.columns[0]] == 1]
    data_train = data_train.drop(data_train.columns[0], axis=1)
    data_test = data_test[data_test[data_test.columns[0]] == 1]
    data_test = data_test.drop(data_test.columns[0], axis=1)
    # Because the data set is messed up!
    data_test = data_test.drop(data_test.columns[-1], axis=1)

    return data_train, data_test

def normalize_data_svd(path, data_train, data_test):
    mean_data_train = np.mean(data_train, axis=0)
    cov_data_train = np.matmul((data_train - mean_data_train).T, (data_train - mean_data_train)) / data_train.shape[0]
    U, s, V = np.linalg.svd(cov_data_train, full_matrices=True)

    idx = s > 1e-4
    U_reduced = U[:, idx]
    V_reduced = V[idx, :]
    s_reduced = s[idx]

    sqr = data_train ** 2
    std = np.sqrt(np.mean(sqr, axis=0) - mean_data_train ** 2)  # Trace of Covariance matrix

    with open(join(path,'SVD_U.pkl'), 'wb') as file:
        pickle.dump(U_reduced, file)
    with open(join(path,'SVD_V.pkl'), 'wb') as file:
        pickle.dump(V_reduced, file)
    with open(join(path,'SVD_s.pkl'), 'wb') as file:
        pickle.dump(s_reduced, file)
    with open(join(path,'mean_encoded_x_all.pkl'), 'wb') as file:
        pickle.dump(mean_data_train, file)

    with open(join(path,'std.pkl'), 'wb') as file:
        pickle.dump(std, file)

    # effective_dim = sum(idx)
    # effective_dim_using_std = data_train.shape[1]

    data_train_normalized = np.matmul((data_train - mean_data_train), U_reduced)/np.sqrt(s_reduced)
    data_test_normalized = np.matmul((data_test - mean_data_train), U_reduced)/np.sqrt(s_reduced)
    
    return data_train_normalized, data_test_normalized
    
    
def load_data_no_discrete_normalised(path, normalization_type):

    data_train, data_test = load_data_no_discrete(path)
    
    if normalization_type == 'static':
        mu = data_train.mean()
        s = data_train.std()
    
    if normalization_type == 'scale':
        s = (np.max(data_train,axis=0) - np.min(data_train,axis=0))/2
        mu = (np.max(data_train,axis=0) + np.min(data_train,axis=0))/2
    
    if normalization_type == 'svd':
        data_train, data_test = normalize_data_svd(path, data_train, data_test)
        s = (np.max(data_train,axis=0) - np.min(data_train,axis=0))/2
        mu = (np.max(data_train,axis=0) + np.min(data_train,axis=0))/2
        
    data_train = (data_train - mu) / s
    data_test = (data_test - mu) / s

    return data_train, data_test


def load_data_no_discrete_normalised_as_array(path, normalization_type):
    normalization_type = normalization_type.lower()
    data_train, data_test = load_data_no_discrete_normalised(path, normalization_type)
    data_train, data_test = data_train.values, data_test.values

    i = 0
    # Remove any features that have too many re-occurring real values.
    features_to_remove = []
    for feature in data_train.T:
        c = Counter(feature)
        max_count = np.array([v for k, v in sorted(c.items())])[0]
        if max_count > 5:
            features_to_remove.append(i)
        i += 1
    data_train = data_train[:, np.array([i for i in range(data_train.shape[1]) if i not in features_to_remove])]
    data_test = data_test[:, np.array([i for i in range(data_test.shape[1]) if i not in features_to_remove])]

    N = data_train.shape[0]
    N_validate = int(N * 0.1)
    data_validate = data_train[-N_validate:]
    data_train = data_train[0:-N_validate]

    return data_train, data_validate, data_test