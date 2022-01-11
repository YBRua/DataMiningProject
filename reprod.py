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
from dgcnn import DGCNN
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


class SAF(nn.Module):
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

    def encode(self, user_features, item_features):
        # bNMd
        item = self.item_embed(item_features)
        # bL4d
        user = self.user_embed(user_features).mean(2)  # bLd
        user = user.unsqueeze(1).repeat(1, item.shape[1], 1, 1)  # bNLd
        sa_pre = torch.cat([user, item], 2)  # bN (L+M) d
        sa_post = self.norm(self.proj(self.attn(sa_pre)) + sa_pre).mean(-2)  # bNd
        return sa_post

    def forward(self, user_ids, item_ids, user_features, item_features, labels=None):
        sa_post = self.encode(user_features, item_features)
        return self.linear(self.drop(sa_post)).squeeze()


class RankingAwareNet(nn.Module):
    def __init__(self, encoder: SAF):
        super().__init__()
        self.encoder = encoder
        self.impf = nn.Sequential(
            nn.Linear(300, 1024),
            nn.LayerNorm(1024), nn.ReLU(),
        )
        self.cls = nn.Sequential(
            nn.Dropout(), nn.Linear(1324, 1)
        )

    def forward(self, user_ids, item_ids, user_features, item_features, labels=None):
        with torch.no_grad():
            sa_post = self.encoder.encode(user_features, item_features)
        impf = self.impf(sa_post).max(-2)[0].unsqueeze(-2).repeat(1, sa_post.shape[1], 1)
        pf = torch.cat([impf, sa_post], dim=-1)
        return self.cls(pf).squeeze()


def get_train_instances(train, ufd, ifd):
    user_input, item_input, labels = [], [], []
    user_ids = []
    item_ids = []
    for (u, i) in train.keys():
        user_ids.append(u)
        item_ids.append(i)
        user_input.append(numpy.array(ufd[u]).reshape(-1, 4))
        item_input.append(numpy.array(ifd[i][:4])[None])
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
        self.cum_weights = [*itertools.accumulate(map(
            lambda x: len(x.positives) + len(x.negatives), self.entries
        ))]

    def __iter__(self):
        rng = len(self.entries) if self.full else 2 ** 30
        for i in range(rng):
            entry = self.entries[i] if self.full else\
                random.choices(self.entries, cum_weights=self.cum_weights, k=1)[0]
            entry = copy.deepcopy(entry)
            if not self.full:
                if not len(entry.positives):
                    continue
                entry.positives = random.choices(entry.positives, k=self.ppe)
                # entry.negatives = random.choices(entry.negatives, k=self.npe)
                entry.negatives = random.choices(self.valid_items, k=self.npe)
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
    train_raw = pickle.load(open("data/train.pkl", "rb"))
    train_smp = data_io.load_train_entries('data/bookcross.train.rating')
    test = data_io.load_test_entries('data/bookcross', False)
    features_dict = pickle.load(open('data/book_info_bookcross', 'rb'))
    user_features_dict = pickle.load(open('data/user_hist_withinfo_bookcross', 'rb'))
    train_smp = Rekommand(train_smp, features_dict, user_features_dict)
    test = Rekommand(test, features_dict, user_features_dict, full=True)
    B = 64
    train_loader_smp = DataLoader(train_smp, B, num_workers=4)
    test_loader = DataLoader(test, B, num_workers=1)

    model = RankingAwareNet(SAF()).to(device)
    target = RankingAwareNet(SAF()).to(device)
    enc_opt = torch.optim.Adam(model.encoder.parameters())
    opt = torch.optim.Adam(model.parameters())
    best_acc = -1
    train = get_train_instances(train_raw, user_features_dict, features_dict)
    for epoch in range(20):
        train_loader = DataLoader(train, B, num_workers=2, shuffle=True)
        ent_stats = run_epoch(model.encoder, train_loader, [Loss()], epoch, enc_opt)
        train_sub = TruncatedIter(train_loader_smp, train_smp.__length_hint__() // B + 1)
        frt_stats = run_epoch(model, train_sub, [Loss()], epoch, opt)
        lerp = epoch / (epoch + 1)
        with torch.no_grad():
            target.load_state_dict(merge_state_dicts([
                scale_state_dict(target.state_dict(), lerp),
                scale_state_dict(model.state_dict(), 1 - lerp)
            ]))
        val_stats = run_epoch(target, test_loader, [Loss(), NDCG3()], epoch)
        if epoch == 0:
            write_log("epoch", "encoder_loss", "finetune_loss", "val_loss", "val_ndcg@3")
        write_log(
            epoch,
            ent_stats['Loss'], frt_stats['Loss'],
            val_stats['Loss'], val_stats['NDCG3']
        )
        if val_stats['NDCG3'] > best_acc:
            best_acc = val_stats['NDCG3']
            torch.save(model.state_dict(), "e2e.dat")
            print("New best!")


if __name__ == "__main__":
    main()
