import time
import collections
import torch
import tqdm
from abc import abstractmethod
from typing import Optional, Sequence


class Meter:
    def __init__(self) -> None:
        self.k = collections.defaultdict(int)
        self.v = collections.defaultdict(int)

    def __getitem__(self, k):
        if self.k[k] == 0:
            return 0
        return self.v[k] / self.k[k]

    def u(self, k, v):
        self.k[k] += 1
        self.v[k] += v


class kMetric:
    def __init__(self, k) -> None:
        self.k = k

    @abstractmethod
    def __call__(self, inputs, model_return) -> torch.Tensor:
        raise NotImplementedError

    def __str__(self) -> str:
        if self.k is not None:
            return f'{self.__class__.__name__}@{self.k}'
        else:
            return self.__class__.__name__


class Backprop(kMetric):
    def __init__(self, k=None) -> None:
        super().__init__(k)


class TruncatedIter:
    def __init__(self, iterable, max_items) -> None:
        self.iterable = iterable
        self.max_items = max_items

    def __len__(self):
        try:
            len(self.iterable)
        except TypeError:
            return self.max_items
        else:
            return min(self.max_items, len(self.iterable))

    def __iter__(self):
        for _, x in zip(range(self.max_items), self.iterable):
            yield x


train_begin = int(time.time())


def write_log(*data):
    with open("training_%d.csv" % train_begin, "a") as fo:
        for d in data:
            fo.write(str(d))
            fo.write(",")
        fo.write("\n")


def run_epoch(
    model: torch.nn.Module,
    loader, metrics: Sequence[kMetric],
    epoch: int,
    optimizer: Optional[torch.optim.Optimizer]=None
):
    training = optimizer is not None
    model.train(training)
    meter = Meter()
    ref_pt = next(model.parameters())
    torch.set_grad_enabled(training)
    prog = tqdm.tqdm(loader)
    for d in prog:
        d = [x.to(ref_pt.device) for x in d]
        output = model(*d)
        for met in metrics:
            mval = met(d, output)
            if isinstance(met, Backprop) and training:
                optimizer.zero_grad()
                mval.backward()
                optimizer.step()
            meter.u(str(met), mval.item())
        desc = "VT"[training] + " %02d" % epoch
        for k in sorted(meter.k):
            desc += " %s: %.4f" % (k, meter[k])
        prog.set_description(desc)
    return {k: meter[k] for k in sorted(meter.k)}


def merge_state_dicts(dicts):
    refd = dicts[0]
    result = dict()
    if isinstance(refd, dict):
        for k in refd.keys():
            result[k] = merge_state_dicts([d[k] for d in dicts])
        return result
    else:
        return sum(dicts)


def scale_state_dict(d, n):
    result = dict()
    if isinstance(d, dict):
        for k in d.keys():
            result[k] = scale_state_dict(d[k], n)
        return result
    else:
        return torch.multiply(d, n)
