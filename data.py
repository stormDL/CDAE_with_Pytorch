import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def read_ml100k(train_path, test_path, sep, header):
    train_set_dict, test_set_dict = {}, {}
    df_train = pd.read_csv(train_path, sep=sep, header=header)-1
    df_test = pd.read_csv(test_path, sep=sep, header=header)-1
    # 处理训练集
    for item in df_train.itertuples():
        uid, i_id = item[1], item[2]
        train_set_dict.setdefault(uid, {}).setdefault(i_id, 1)
    for item in df_test.itertuples():
        uid, i_id = item[1], item[2]
        test_set_dict.setdefault(uid, {}).setdefault(i_id, 1)
    return train_set_dict, test_set_dict


def read_ml1m(filepath, sep='::', header='infer'):
    train_set_dict, test_set_dict = {}, {}
    df = pd.read_csv(filepath, sep=sep, header=header).iloc[:, :3]-1
    df = df.values.tolist()
    train_set, test_set = train_test_split(df, test_size=0.2, random_state=1231)
    for uid, iid, score in train_set:
        train_set_dict.setdefault(uid, {}).setdefault(iid, 1)
    for uid, iid, score in test_set:
        test_set_dict.setdefault(uid, {}).setdefault(iid, 1)
    return train_set_dict, test_set_dict


def get_matrix(train_set_dict, test_set_dict, nb_user, nb_item):
    train_set, test_set = np.zeros(shape=(nb_user, nb_item)), np.zeros(shape=(nb_user, nb_item))
    for u in train_set_dict.keys():
        for i in train_set_dict[u].keys():
            train_set[u][i] = 1
    for u in test_set_dict.keys():
        for i in test_set_dict[u]:
            test_set[u][i] = 1
    return train_set, test_set

