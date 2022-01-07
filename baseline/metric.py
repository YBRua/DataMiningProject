import torch
import random
import torch.nn as nn
import math
import numpy
from torch.utils.data import DataLoader
from .sparse_graph import SparseGraph


def get_negative_tests(G: SparseGraph, size: int) -> torch.Tensor:
    n_nodes = G.coo.shape[0]
    negatives = []
    for i in range(size):
        src = choose_a_node(n_nodes)
        neighbours = G.get_neighbours(src)
        dst = choose_a_node(n_nodes)
        while dst == src or dst in neighbours:
            dst = choose_a_node(n_nodes)
        negatives.append([src, dst])

    return torch.tensor(negatives)


def choose_a_node(high, low=0) -> int:
    return random.randint(low, high-1)


def normalized_cosine_similiarty(x: torch.Tensor, y: torch.Tensor):
    # cosine_sim = torch.cosine_similarity(x, y)
    # cosine_sim = cosine_sim - cosine_sim.min()
    # cosine_sim = cosine_sim / cosine_sim.max()

    # return cosine_sim
    return (torch.cosine_similarity(x, y) + 1) / 2


def calc_auc(D0: torch.LongTensor, D1: torch.LongTensor, model: nn.Module):
    model.eval()
    EVAL_BATCH_SIZE = 2048

    d0_ldr = DataLoader(D0, EVAL_BATCH_SIZE)
    d1_ldr = DataLoader(D1, EVAL_BATCH_SIZE)

    b = 0
    tot_auc = 0
    for d0, d1 in zip(d0_ldr, d1_ldr):
        b += 1
        denominator = d0.shape[0] * d1.shape[0]
        pred0 = model.forward(d0)  # B,2,Emb
        pred1 = model.forward(d1)  # B,2,Emb

        prob0 = normalized_cosine_similiarty(pred0[:, 0, :], pred0[:, 1, :])
        prob1 = normalized_cosine_similiarty(pred1[:, 0, :], pred1[:, 1, :])
        prob1_ext = prob1.repeat_interleave(prob0.shape[0])
        prob0_ext = prob0.repeat(prob1.shape[0])
        auc = torch.sum(prob0_ext < prob1_ext).float() / denominator
        tot_auc += auc

    tot_auc /= b
    return tot_auc


def ndcg(ranked, gt, at=5):
    score = 0.0
    total = 0.0
    for i in range(at):
        t = math.log(2) / math.log(i + 2)
        total += t
        if ranked[i] in gt:
            score += t
    return score / total


def batch_ndcg_torch(topk: torch.Tensor, labels01: torch.Tensor):
    t = math.log(2) / (torch.arange(
        topk.shape[-1], dtype=torch.float32, device=topk.device
    ) + 2)
    batch_idx = topk.new_tensor(numpy.arange(topk.shape[0])).unsqueeze(-1)
    return (t * labels01[batch_idx, topk].to(t)).sum(-1) / t.sum(-1)
