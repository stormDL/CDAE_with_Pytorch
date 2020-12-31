import random

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from data import read_ml1m, get_matrix, read_ml100k
from layers import CDAE
from matplotlib import pyplot as plt


def plot_precision(epoche_list, precision_list):
    plt.title('precision')
    plt.plot(epoche_list, precision_list, marker='o')
    plt.savefig('precision_ml100k.png')
    plt.show()


class M_Dataset(Dataset):
    def __init__(self, train_set):
        self.train_set = train_set

    def __getitem__(self, idx):
        purchase_vec = torch.tensor(self.train_set[idx], dtype=torch.float)
        uid = torch.tensor([idx,], dtype=torch.long)
        return purchase_vec, uid

    def __len__(self):
        return len(self.train_set)


def select_negative_items(batch_history_data, nb):
    data = np.array(batch_history_data)
    idx = np.zeros_like(data)
    for i in range(data.shape[0]):
        # 得到所有为0的项目下标
        items = np.where(data[i] == 0)[0].tolist()
        # 随机抽取一定数量的下标
        tmp_zr = random.sample(items, nb)
        # 这些位置的值为1
        idx[i][tmp_zr] = 1
    return idx


def test(model, test_set_dict, train_set, top_k=5):
    model.eval()
    users = list(test_set_dict.keys())
    input_data = torch.tensor(train_set[users], dtype=torch.float)
    uids = torch.tensor(users, dtype=torch.long).view(-1, 1)
    out = model(uids, input_data)
    out = (out - 999*input_data).detach().numpy()
    precisions = 0
    recalls = 0
    hits = 0
    total_purchase_nb = 0
    for i, u in enumerate(users):
        hit = 0
        tmp_list = [(idx, value) for idx, value in enumerate(out[i])]
        tmp_list = sorted(tmp_list, key=lambda x:x[1], reverse=True)[:top_k]
        for k, v in tmp_list:
            if k in test_set_dict[u]:
                hit += 1
        recalls += hit/len(test_set_dict[u])
        precisions += hit/top_k
        hits += hit
        total_purchase_nb += len(test_set_dict[u])
    recall = recalls/len(users)
    precision = precisions/len(users)
    print('recall:{}, precision:{}'.format(recall, precision))
    return precision, recall


def train(nb_user, nb_item, nb_hidden,epoches, dataloader, lr, nb_mask, train_set, test_set_dict, top_k):
    # 收集数据
    epoche_list, precision_list = [], []
    # 建模
    model = CDAE(nb_item, nb_user, nb_hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for e in range(epoches):
        model.train()
        for purchase_vec, uid in dataloader:
            mask_vec = torch.tensor(select_negative_items(purchase_vec, nb_mask)) + purchase_vec
            out = model(uid, purchase_vec) * mask_vec
            loss = torch.sum((out - purchase_vec).square())/mask_vec.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            # print(loss.item())
        if (e+1)%5 == 0:
            print(e + 1, '\t', '==' * 24)
            precision, _  = test(model, test_set_dict, train_set, top_k=top_k)
            epoche_list.append(e + 1)
            precision_list.append(precision)
    plot_precision(epoche_list, precision_list)


if __name__ == '__main__':

    nb_user = 6040
    nb_item = 3952
    nb_hidden = 24
    train_set_dict, test_set_dict = read_ml1m('dataset/ml-1m/ratings.dat')
    train_set, test_set = get_matrix(train_set_dict, test_set_dict, nb_user=nb_user, nb_item=nb_item)

    # nb_user=943
    # nb_item=1682
    # nb_hidden = 12
    # train_set_dict, test_set_dict = read_ml100k('dataset/ml-100k/u1.base', 'dataset/ml-100k/u1.test', sep='\t', header=None)
    # train_set, test_set = get_matrix(train_set_dict, test_set_dict, nb_user=nb_user, nb_item=nb_item)
    dataset = M_Dataset(train_set)
    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
    train(nb_user, nb_item, nb_hidden, epoches=2000, dataloader=dataloader, lr=0.0001, nb_mask=128, train_set=train_set, test_set_dict=test_set_dict, top_k=5)