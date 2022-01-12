import torch
import torch.nn as nn
import collections
from deepwalk_baseline import inference_procedure


class DeepWalkWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, user_ids, item_ids, *_dc):
        x = torch.stack(user_ids.unsqueeze(-1).expand_as(item_ids), item_ids, dim=-1)
        return inference_procedure(x, x.flatten(0, 1)).reshape(*x.shape[:-1]).squeeze()
