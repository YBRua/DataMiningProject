import copy
import pickle
import random
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
from typing import Dict, List
from torch.utils.data import DataLoader, IterableDataset
from freq_counter import FreqCounter

from argparse import ArgumentParser
from baseline.metric import batch_ndcg_torch, batched_average_precision, batched_hit_ratio, batched_reciprocal_rank
import data_io
from common import *
from metrics import MAP, NDCG, HR, RR, Loss


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--epoch', '-e', type=int, default=20,
        help='Number of training epoches')
    parser.add_argument(
        '--batchsize', '-b', type=int, default=64,
        help='Batch size')
    parser.add_argument(
        '--dataset', '-d', choices=['bookcross', 'music'],
        default='bookcross',
        help='Dataset, can be one of bobookcrossok or music')
    parser.add_argument(
        '--device', type=str, default='cuda:0',
        help='PyTorch style device to run the model')

    return parser.parse_args()


def prepare_dataset(dataset: str):
    assert dataset in ['bookcross', 'music'], f'Invalid dataset: {dataset}'

    train_raw = data_io.load_rating_file_as_matrix(f'data/{dataset}.train.rating')
    train_smp = data_io.load_train_entries(f'data/{dataset}.train.rating')
    test = data_io.load_test_entries(f'data/{dataset}', False)
    if dataset == 'bookcross':
        feature_dict_path = 'data/book_info_bookcross'
        user_feat_path = 'data/user_hist_withinfo_bookcross'
    else:
        feature_dict_path = 'data/song_info_music_HK'
        user_feat_path = 'data/user_hist_withinfo_music_HK'
    features_dict = pickle.load(open(feature_dict_path, 'rb'))
    user_features_dict = pickle.load(open(user_feat_path, 'rb'))

    return train_raw, train_smp, test, features_dict, user_features_dict


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


def main():
    args = parse_args()
    DEVICE = torch.device(args.device)
    BATCH_SIZE = args.batchsize
    EPOCHES = args.epoch

    train_raw, train_smp, test, features_dict, user_features_dict = prepare_dataset(args.dataset)

    target = FreqCounter(train_smp)

    train_smp = Rekommand(train_smp, features_dict, user_features_dict)
    test = Rekommand(test, features_dict, user_features_dict, full=True)
    train_loader_smp = DataLoader(train_smp, BATCH_SIZE, num_workers=4)
    test_loader = DataLoader(test, BATCH_SIZE, num_workers=1)
    val_stats = run_epoch(
        target,
        test_loader,
        [
            Loss(),
            RR(5),
            MAP(3), MAP(5),
            NDCG(3), NDCG(5),
            HR(3), HR(5)
        ],
        1
    )
    print(val_stats)

    model = RankingAwareNet(SAF()).to(DEVICE)
    target = RankingAwareNet(SAF()).to(DEVICE)
    enc_opt = torch.optim.Adam(model.encoder.parameters())
    opt = torch.optim.Adam(model.parameters())
    best_acc = -1
    train = get_train_instances(train_raw, user_features_dict, features_dict)

    for epoch in range(EPOCHES):
        train_loader = DataLoader(train, BATCH_SIZE, num_workers=2, shuffle=True)
        ent_stats = run_epoch(model.encoder, train_loader, [Loss()], epoch, enc_opt)
        train_sub = TruncatedIter(train_loader_smp, train_smp.__length_hint__() // BATCH_SIZE + 1)
        frt_stats = run_epoch(model, train_sub, [Loss()], epoch, opt)
        lerp = epoch / (epoch + 1)
        with torch.no_grad():
            target.load_state_dict(merge_state_dicts([
                scale_state_dict(target.state_dict(), lerp),
                scale_state_dict(model.state_dict(), 1 - lerp)
            ]))
        val_stats = run_epoch(
            target,
            test_loader,
            [
                Loss(),
                RR(5),
                MAP(3), MAP(5),
                NDCG(3), NDCG(5),
                HR(3), HR(5)
            ],
            epoch
        )
        if epoch == 0:
            write_log(
                "epoch",
                "encoder_loss", "finetune_loss", "val_loss",
                "val_rr@5",
                "val_map@3", "val_map@5",
                "val_ndcg@3", "val_ndcg@5",
                "val_hr@3", "val_hr@5"
            )
        write_log(
            epoch,
            ent_stats['Loss'], frt_stats['Loss'], val_stats['Loss'],
            val_stats['RR@5'],
            val_stats['MAP@3'], val_stats['MAP@5'],
            val_stats['NDCG@3'], val_stats['NDCG@5'],
            val_stats['HR@3'], val_stats['HR@5']
        )
        if val_stats['NDCG@3'] > best_acc:
            best_acc = val_stats['NDCG@3']
            torch.save(model.state_dict(), "e2e.dat")
            print("New best!")


if __name__ == "__main__":
    main()
