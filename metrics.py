import torch
import torch.nn.functional as F
from common import Backprop, kMetric
from baseline.metric import batch_ndcg_torch, batched_reciprocal_rank
from baseline.metric import batched_hit_ratio, batched_average_precision


class Loss(Backprop):
    def __call__(self, inputs, model_return):
        *_, y = inputs
        return F.binary_cross_entropy_with_logits(model_return.reshape(*y.shape), y.float())


class NDCG(kMetric):
    def __call__(self, inputs, model_return):
        *_, y = inputs
        topk = torch.topk(model_return.reshape(*y.shape), self.k, -1).indices
        return batch_ndcg_torch(topk, y).mean()


class RR(kMetric):
    def __call__(self, inputs, model_return) -> torch.Tensor:
        *_, y = inputs
        topk = torch.topk(model_return.reshape(*y.shape), self.k, -1).indices
        return batched_reciprocal_rank(topk, y).mean()


class HR(kMetric):
    def __call__(self, inputs, model_return) -> torch.Tensor:
        *_, y = inputs
        topk = torch.topk(model_return.reshape(*y.shape), self.k, -1).indices
        return batched_hit_ratio(topk, y).mean()


class MAP(kMetric):
    def __call__(self, inputs, model_return) -> torch.Tensor:
        *_, y = inputs
        topk = torch.topk(model_return.reshape(*y.shape), self.k, -1).indices
        return batched_average_precision(topk, y).mean()
