import sys
import torch
import random
import numpy as np
# import pandas as pd
import torch.nn as nn
import seaborn as sns
import torch.optim as optim
import scipy.sparse as sparse
import matplotlib.pyplot as plt

import data_io as io

from tqdm import tqdm
from typing import List
from torch.utils.data import DataLoader
from baseline.metric import calc_auc, normalized_cosine_similiarty
from baseline.metric import get_negative_tests, ndcg

from baseline.model import SkipGram
from baseline.utils import load_edges, parse_args
from baseline.sparse_graph import SparseGraph
from baseline.walker import RandomWalker, UniformRandomWalker
from baseline.loss import NegativeSamplingLoss


def train_procedure(
        args,
        eval: np.ndarray,
        train: np.ndarray,
        graph: SparseGraph,
        model: SkipGram,
        optimizer: optim.Optimizer,
        walker: RandomWalker,
        loss_metric: NegativeSamplingLoss):
    # training configs
    N_NODES = graph.coo.shape[0]
    N_EPOCHS = args.epoch
    BATCH_SIZE = args.batchsize
    DEVICE = args.device

    # random walk hyper params
    WALK_LENGTH = args.walk_length
    WINDOW_SIZE = args.window_size

    losses = []
    aucs = []

    for e in range(N_EPOCHS):
        trajectory: List[List[int]] = []
        isolated_nodes = set([])
        nodes = list(range(N_NODES))
        random.shuffle(nodes)
        for node in nodes:
            if graph.get_degree(node) == 0:
                isolated_nodes.add(node)
                continue
            walk = walker(node, WALK_LENGTH)
            for cent in range(WINDOW_SIZE, WALK_LENGTH - WINDOW_SIZE):
                trajectory.append(
                    walk[cent - WINDOW_SIZE: cent + WINDOW_SIZE + 1])
        trajectory = torch.LongTensor(trajectory)

        data_loader = DataLoader(
            trajectory, batch_size=BATCH_SIZE, shuffle=True)

        tot_loss = 0
        t = tqdm(data_loader)
        model.train()
        bidx = 1
        for x in t:
            current_bsize = x.shape[0]
            optimizer.zero_grad()
            x = x.to(DEVICE)
            preds = model.forward(x)
            loss = loss_metric(model, preds, current_bsize)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
            t.set_description(f'Epoch: {e:2d} Loss: {tot_loss / (bidx+1):.4f}')
            bidx += 1
        model.eval()

        with torch.no_grad():
            if eval is not None:
                k = eval.shape[0]
                pos_tests = torch.tensor(eval)
                neg_tests = get_negative_tests(graph, k)
            else:
                k = 5000
                neg_tests = get_negative_tests(graph, k)
                pos_tests = torch.tensor(
                    train[random.choices(list(range(train.shape[0])), k=k)])

            neg_tests = neg_tests.to(DEVICE)
            pos_tests = pos_tests.to(DEVICE)
            auc = calc_auc(neg_tests, pos_tests, model)

        losses.append(tot_loss / bidx)
        aucs.append(auc.item())

        print(f'Epoch: {e:2d} AUC: {auc:.4f}', file=sys.stderr)

    return model, losses, aucs


def inference_procedure(model: nn.Module, tests: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        preds = model.forward(tests)
        probs = normalized_cosine_similiarty(preds[:, 0, :], preds[:, 1, :])
        return probs


def output_to_csv(res: np.ndarray, path: str):
    with open(path, 'w') as fo:
        for idx, probs in enumerate(res):
            fo.write(f'{idx}, {probs:.4f}\n')
    print(f'Result written to {path}')


def main():
    args = parse_args()
    # training configs

    N_NEG_SAMPLES = args.neg_samples
    DEVICE = args.device

    # I/O configs
    INPUT_PATH = args.dataset_path
    # TEST_PATH = args.testset_path
    # OUTPUT_PATH = args.file_output
    SAVE_PATH = args.model_save
    PRETRAINED_MODEL = args.pretrained_path

    # hyper params
    EMBEDDING_DIM = 64
    LR = args.lr  # learning rate
    MMT = args.mmt  # momentum

    # hyper params for biased random walker
    RETURN_PARAM = args.walker_return_param
    IO_PARAM = args.walker_io_param

    # build sparse graph
    data = io.filter_dataset(io.load_dataset(INPUT_PATH))
    edges = data[:, :2]
    n_vertices = np.max(edges) + 1
    print("edges", edges.shape)
    coo = sparse.coo_matrix(
        (np.ones(edges.shape[0]), (edges.T[0], edges.T[1])),
        shape=(n_vertices, n_vertices)
    )
    print("coo", coo.shape)
    G = SparseGraph(coo)
    # G = SparseGraph.build_graph_from_csv(EDGE_PATH)
    # edges = load_edges(EDGE_PATH)

    N_NODES = G.coo.shape[0]
    IN_DIM = N_NODES

    if args.split_dataset:
        print('Dataset will be splitted for training and evaluation.')
        np.random.shuffle(edges)
        split = int(edges.shape[0] * 0.9)
        train_x = edges[:split, :]
        eval_x = edges[split:, :]
        print("Train_x", train_x.shape)
        print("Eval_x", eval_x.shape)
    else:
        print('Will be using the entire dataset for training.')
        train_x = edges
        eval_x = None

    # build the training graph with training data
    graph = SparseGraph(sparse.coo_matrix(
        (np.ones(train_x.shape[0]), (train_x.T[0], train_x.T[1])),
        shape=(n_vertices, n_vertices)))

    # test_set = pd.read_csv(TEST_PATH).to_numpy()
    # test_x = torch.tensor(test_set[:, 1:]).to(DEVICE)

    # walker = BiasedRandomWalker(graph, RETURN_PARAM, IO_PARAM)
    walker = UniformRandomWalker(graph)
    model = SkipGram(IN_DIM, EMBEDDING_DIM).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MMT)
    loss_m = NegativeSamplingLoss(N_NODES, N_NEG_SAMPLES)

    if PRETRAINED_MODEL == '':
        # training mode
        model.train()
        model, losses, aucs = train_procedure(
            args, eval_x, train_x, graph, model, optimizer, walker, loss_m)
        torch.save(model.state_dict(), SAVE_PATH)
        print(f'Model saved to {SAVE_PATH}')
        model.eval()
    else:
        # inference mode
        model.load_state_dict(torch.load(PRETRAINED_MODEL, map_location=torch.device(DEVICE)))
        model.eval()

    if args.fancy:
        print('Going fancy.')
        sns.set_theme('notebook', 'white', 'pastel')
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].plot(losses, label='loss')
        ax[0].set_xlabel('epoch')
        ax[0].set_ylabel('loss')
        ax[1].plot(aucs, label='auc')
        ax[1].set_title('AUC-Epoch')
        ax[1].set_xlabel('epoch')
        ax[1].set_ylabel('auc')
        fig.tight_layout()
        sns.despine()
        plt.show()

    test_x = io.load_test_entries(args.testset_path)
    ndcg_log = []
    prog_test = tqdm(test_x)
    for entry in prog_test:
        user = entry.id
        tn = torch.empty(
            [len(entry.positives) + len(entry.negatives), 2],
            dtype=torch.int64,
            device=DEVICE
        )
        tn[:, 0] = user
        tn[:len(entry.positives), 1] = tn.new_tensor(entry.positives)
        tn[len(entry.positives):, 1] = tn.new_tensor(entry.negatives)
        probs = inference_procedure(model, tn)

        ndcg_k = 3
        probsort = probs.detach().cpu().numpy().argsort()[::-1]
        ranklist = np.array(entry.positives + entry.negatives)[probsort]
        ndcg_log.append(ndcg(ranklist, set(entry.positives), ndcg_k))
        if random.random() < 0.01:
            prog_test.set_description(f"NDCG@{ndcg_k}: {np.mean(ndcg_log):.4f}")
    print("Final", f"NDCG@{ndcg_k}: {np.mean(ndcg_log):.4f}")

    # output_to_csv(probs, OUTPUT_PATH)


if __name__ == '__main__':
    main()
