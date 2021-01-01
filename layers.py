import torch
import torch.nn as nn


class CDAE(nn.Module):
    def __init__(self, nb_item, nb_user, nb_hidden, drop_rate=0.3):
        '''
        :param nb_item: 项目的数量
        '''
        super(CDAE, self).__init__()
        self.item2hidden = nn.Sequential(
            nn.Linear(nb_item, nb_hidden),
            nn.Dropout(drop_rate)
        )
        self.id2hidden = nn.Embedding(nb_user, nb_hidden)
        self.hidden2out = nn.Linear(nb_hidden, nb_item)
        self.sigmoid = nn.Sigmoid()

    def forward(self, uid, purchase_vec):
        hidden = self.sigmoid(self.id2hidden(uid).squeeze(dim=1)+self.item2hidden(purchase_vec))
        out = self.hidden2out(hidden)
        out = self.sigmoid(out)
        return out