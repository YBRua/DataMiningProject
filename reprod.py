import copy
import pickle
import random
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
from typing import Dict, List
from types import SimpleNamespace
from torch.utils.data import DataLoader, IterableDataset

from baseline.metric import batch_ndcg_torch
import data_io
from common import *


device = torch.device("cuda:0")


class SelfAttention(nn.Module):
    def __init__(self, nhead, i, kq, v):
        super().__init__()
        self.attn_k = nn.Linear(i, kq, bias=False)
        self.attn_q = nn.Linear(i, kq, bias=False)
        self.attn_v = nn.Linear(i, v, bias=False)
        self.nhead = nhead

    def forward(self, x):
        # x: ...ni
        k = self.attn_k(x).reshape(*x.shape[:-1], self.nhead, -1)
        q = self.attn_q(x).reshape(*x.shape[:-1], self.nhead, -1)
        v = self.attn_v(x).reshape(*x.shape[:-1], self.nhead, -1)
        attn = torch.softmax(
            torch.einsum("...nhd,...mhd->...nmh", k, q) / k.shape[-1] ** 0.5,
            -2
        )
        return torch.einsum("...nmh,...mhd->...nhd", attn, v).flatten(-2)



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.user_embed = nn.Embedding(90000, 300)
        self.item_embed = nn.Embedding(90000, 300)
        with torch.no_grad():
            self.user_embed.weight.uniform_(-0.05, 0.05)
            self.item_embed.weight.uniform_(-0.05, 0.05)
        self.attn = SelfAttention(5, 300, 600, 600)
        self.drop = nn.Dropout(0.4)
        self.proj = nn.Linear(600, 300)
        self.linear = nn.Linear(300, 1)
        self.norm = nn.LayerNorm(300)

    def forward(self, user_ids, item_ids, user_features, item_features, labels=None):
        # bNMd
        item = self.item_embed(item_features)
        # bL4d
        user = self.user_embed(user_features).mean(2)  # bLd
        user = user.unsqueeze(1).repeat(1, item.shape[1], 1, 1)  # bNLd
        sa_pre = torch.cat([user, item], 2)  # bN (L+M) d
        sa_post = self.norm(self.proj(self.attn(sa_pre)) + sa_pre).mean(-2)  # bNd
        return self.linear(self.drop(sa_post)).squeeze()


def get_train_instances(train, ufd, ifd):
    user_input, item_input, labels = [], [], []
    num_users = train.shape[0]
    user_ids = []
    item_ids = []
    for (u, i) in train.keys():
        # positive instance
        user_ids.append(u)
        item_ids.append(i)
        user_input.append(numpy.array(ufd[u]).reshape(-1, 4))
        item_input.append(numpy.array(ifd[i][:4])[None])
        # assert ifd[i][:4] == ifd[i][4:8] == ifd[i][8:12] == ifd[i][12:], [i, ifd[i]]
        if train[(u, i)] == 1:
            labels.append(1)
        if train[(u, i)] == -1:
            labels.append(0)

    return [*zip(user_ids, item_ids, user_input, item_input, labels)]


class Rekommand(IterableDataset):
    def __init__(
        self, entries: List[data_io.TestEntry],
        item_feature_dict: Dict[int, List[int]],
        user_feature_dict: Dict[int, List[int]],
        pos_per_entry: int = 5, neg_per_entry: int = 50,
        background_neg_samples: int = 0,
        full: bool = False
    ) -> None:
        super().__init__()
        self.full = full
        self.ppe = pos_per_entry
        self.npe = neg_per_entry
        self.bns = background_neg_samples
        self.entries = entries
        self.ifd = item_feature_dict
        self.ufd = user_feature_dict
        self.valid_items = list(self.ifd.keys())
        self.cum_weights = [*itertools.accumulate(map(lambda x: len(x.positives), self.entries))]

    def __iter__(self):
        rng = len(self.entries) if self.full else 2 ** 30
        for i in range(rng):
            entry = self.entries[i] if self.full else\
                random.choices(self.entries, cum_weights=self.cum_weights, k=1)[0]
            entry = copy.deepcopy(entry)
            if not self.full:
                if not len(entry.positives):
                    continue
                if random.random() < 0.2:
                    entry.positives = random.choices(entry.positives, k=1)
                    entry.negatives = []
                else:
                    entry.positives = []
                    entry.negatives = random.choices(entry.negatives, k=1)
                # entry.negatives = random.choices(self.valid_items, k=self.npe)
            items = numpy.array(list(entry.positives) + list(entry.negatives))
            item_features = numpy.array([self.ifd[x][:4] for x in items])
            user_features = numpy.array(self.ufd[entry.id]).reshape(-1, 4)
            labels = numpy.concatenate([
                numpy.ones_like(entry.positives),
                numpy.zeros_like(entry.negatives)
            ])
            randperm = numpy.random.permutation(len(items))
            yield entry.id, items[randperm], user_features, item_features[randperm], labels[randperm]

    def __length_hint__(self):
        return len(self.entries) if self.full else 2 * len(self.entries)


class Loss(Backprop):
    def __call__(self, inputs, model_return):
        *_, y = inputs
        return F.binary_cross_entropy_with_logits(model_return.reshape(*y.shape), y.float())


class NDCG3(Metric):
    def __call__(self, inputs, model_return):
        *_, y = inputs
        topk = torch.topk(model_return.reshape(*y.shape), 3, -1).indices
        return batch_ndcg_torch(topk, y).mean()


def main():
    # train = data_io.load_train_entries('data/bookcross.train.rating')
    train_raw = pickle.load(open("train.pkl", "rb"))
    test = data_io.load_test_entries('data/bookcross', False)
    features_dict = pickle.load(open('data/book_info_bookcross', 'rb'))
    user_features_dict = pickle.load(open('data/user_hist_withinfo_bookcross', 'rb'))
    # train = Rekommand(train, features_dict, user_features_dict, 11, 44, 44)
    test = Rekommand(test, features_dict, user_features_dict, full=True)
    B = 64
    test_loader = DataLoader(test, B, num_workers=1)

    model = Net()
    # weights = pickle.load(open("param_lists.pkl", "rb"))
    '''with torch.no_grad():
        model.user_embed.weight.set_(torch.tensor(weights[2][0]))
        model.item_embed.weight.set_(torch.tensor(weights[3][0]))
        model.attn.attn_k.weight.set_(torch.tensor(weights[-4][0].T))
        model.attn.attn_q.weight.set_(torch.tensor(weights[-4][1].T))
        model.attn.attn_v.weight.set_(torch.tensor(weights[-4][2].T))
        model.linear.weight.set_(torch.tensor(weights[-1][0].T))
        model.linear.bias.set_(torch.tensor(weights[-1][1]))'''
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters())
    best_acc = -1
    for epoch in range(100):
        train = get_train_instances(train_raw, user_features_dict, features_dict)
        train_loader = DataLoader(train, B, num_workers=2, shuffle=True)
        train_sub = train_loader  # TruncatedIter(train_loader, train.__length_hint__() // 4)
        train_stats = run_epoch(model, train_sub, [Loss()], epoch, opt)
        val_stats = run_epoch(model, test_loader, [Loss(), NDCG3()], epoch)
        if epoch == 0:
            write_log("epoch", "train_loss", "val_loss", "train_ndcg@3", "val_ndcg@3")
        write_log(
            epoch,
            train_stats['Loss'], val_stats['Loss'],
            float('nan'), val_stats['NDCG3'],
        )
        if val_stats['NDCG3'] > best_acc:
            best_acc = val_stats['NDCG3']
            torch.save(model.state_dict(), "e2e.dat")
            print("New best!")


if __name__ == "__main__":
    main()
