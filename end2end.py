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


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()

    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


class DGCNN(nn.Module):
    def __init__(self, args):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k
        self.input_dims = args.input_dims
        output_channels = args.output_channels

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(self.input_dims*2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Conv1d(args.emb_dims, 512, 1, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Conv1d(512, 256, 1)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Conv1d(256, output_channels, 1)

    def forward(self, x):
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)

        # x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        # x = self.dp1(x)
        # x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        # x = self.dp2(x)
        # x = self.linear3(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.user_embed = nn.Embedding(10000, 64)
        self.item_embed = nn.Embedding(90000, 32)
        self.item_bias = nn.Embedding(90000, 1)
        self.dgcnn = DGCNN(SimpleNamespace(
            k=9,
            emb_dims=128,
            input_dims=192,
            output_channels=1,
            dropout=0.3
        ))

    def forward(self, user_ids, item_ids, user_features, item_features, labels=None):
        # b34d
        user = self.item_embed(user_features).mean(1).flatten(1)
        item = self.item_embed(item_features).flatten(2).transpose(1, 2)
        # hodgepodge = torch.cat([
        #     user.unsqueeze(1).repeat(1, item.shape[1], 1), item
        # ], -1).transpose(1, 2)
        iemb = item
        uemb = user
        return torch.einsum("bdn,bd->bn", iemb, uemb) * 0.1 + self.item_bias(item_ids).squeeze(-1).detach()


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
            entry = self.entries[i] if self.full else random.choice(self.entries)
            entry = copy.deepcopy(entry)
            if not self.full:
                if not len(entry.positives):
                    continue
                entry.positives = random.choices(entry.positives, k=self.ppe)
                entry.negatives = random.choices(entry.negatives, k=self.npe)
                # entry.negatives = random.choices(self.valid_items, k=self.npe)
            items = numpy.array(list(entry.positives) + list(entry.negatives))
            item_features = numpy.array([self.ifd[x][:4] for x in items])
            user_features = numpy.array(self.ufd[entry.id]).reshape(3, 4)
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
        return F.binary_cross_entropy_with_logits(model_return, y.float())


class NDCG3(Metric):
    def __call__(self, inputs, model_return):
        *_, y = inputs
        topk = torch.topk(model_return, 3, -1).indices
        return batch_ndcg_torch(topk, y).mean()


class TestModel(nn.Module):
    def __init__(self, train):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.cnt = collections.Counter([y for x in train for y in x.positives])
        self.srt = sorted(self.cnt, key=self.cnt.get)
        self.table = {x: i for i, x in enumerate(self.srt)}

    def forward(self, user_ids, item_ids, *_dc):
        scores = torch.zeros_like(item_ids).float()
        for b in range(item_ids.shape[0]):
            for i in range(item_ids.shape[1]):
                scores[b, i] = self.table.get(item_ids[b, i].item(), -1)
        return scores


def main():
    train = data_io.load_train_entries('data/bookcross.train.rating')
    test = data_io.load_test_entries('data/bookcross', False)
    features_dict = pickle.load(open('data/book_info_bookcross', 'rb'))
    user_features_dict = pickle.load(open('data/user_hist_withinfo_bookcross', 'rb'))
    test_model = TestModel(train)
    train = Rekommand(train, features_dict, user_features_dict, 11, 44, 44)
    test = Rekommand(test, features_dict, user_features_dict, full=True)
    B = 32
    train_loader = DataLoader(train, B, num_workers=4)
    test_loader = DataLoader(test, B, num_workers=1)
    run_epoch(test_model, test_loader, [Loss(), NDCG3()], 0)

    model = Net().to(device)
    model.item_bias.weight.set_(torch.tensor(
        [numpy.log(test_model.cnt.get(x, 0.1)) for x in range(90000)],
        device=device, dtype=torch.float32
    ).unsqueeze(-1))
    opt = torch.optim.Adam(model.parameters())
    best_acc = -1
    for epoch in range(100):
        train_sub = TruncatedIter(train_loader, train.__length_hint__() // B + 1)
        train_stats = run_epoch(model, train_sub, [Loss(), NDCG3()], epoch, opt)
        val_stats = run_epoch(model, test_loader, [Loss(), NDCG3()], epoch)
        if epoch == 0:
            write_log("epoch", "train_loss", "val_loss", "train_ndcg@3", "val_ndcg@3")
        write_log(
            epoch,
            train_stats['Loss'], val_stats['Loss'],
            train_stats['NDCG3'], val_stats['NDCG3'],
        )
        if val_stats['NDCG3'] > best_acc:
            best_acc = val_stats['NDCG3']
            torch.save(model.state_dict(), "e2e.dat")
            print("New best!")


if __name__ == "__main__":
    main()
