import torch
import torch.nn as nn
import collections


class FreqCounter(nn.Module):
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
