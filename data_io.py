import numpy as np
import pandas as pd

import typing as t


class TestEntry():
    def __init__(
            self,
            id: int,
            positives: t.List[int] = None,
            negatives: t.List[int] = None):
        self.id = id
        self.positives = positives
        self.negatives = negatives


def load_dataset(path: str) -> np.ndarray:
    """Loads a dataset from given `path` and returns an `np.ndarray`
    The dataset should be a csv-like text file with `\t` as seperator.

    Args:
        path (str): Path to the dataset

    Returns:
        np.ndarray: Loaded dataset, converted to a numpy array
    """
    csv_f: pd.DataFrame = pd.read_csv(path, sep='\t', header=None)
    return csv_f.to_numpy()


def filter_dataset(data: np.ndarray) -> np.ndarray:
    """Filters the given dataset. Retains only samples with rating > 0.
    This function also re-indexes all nodes
    so that user nodes and item nodes have different ids.

    Args:
        data (np.ndarray): Original dataset, expected to be a (N, 3) array
          Columns corresponds to (user, item, rating)

    Returns:
        np.ndarray: Filtered dataset.
          (N', 3) array containing only positive rating samples.
    """
    n_users = np.max(data, axis=0)[0] + 1  # maximum of user column
    # user nodes: [0 : n_users]
    # item nodes: [n_users:]
    data[:, 1] += n_users
    positives = data[data[:, -1] > 0]
    # print(positives.shape)
    return positives


def load_test_entries(path: str, offset=True) -> t.List[TestEntry]:
    """Loads test.ratings and test.negative from given `path`.

    Args:
        path (str): Path to *.test.negative and *.test.rating

    Returns:
        List[TestEntry]: A list of TestEntry objects.
        - Each `TestEntry` consists of
            - an `id`: Id of user
            - a list of `positives`: Item ids of pos ratings from this user
            - a list of `negatives`: Item ids of neg ratings from this user

    NOTE: This function appends `.test.rating` and `.test.negative` to path,
    so `path` should be like `./data/bookcross`
    """
    pos_path = path + '.test.rating'
    neg_path = path + '.test.negative'
    pos_ratings = pd.read_csv(pos_path, sep='\t', header=None).to_numpy()
    neg_ratings = pd.read_csv(neg_path, sep='\t', header=None).to_numpy()

    assert pos_ratings.shape[0] == neg_ratings.shape[0], "?"

    n_entries = pos_ratings.shape[0]
    entries: t.List[TestEntry] = []
    for e in range(n_entries):
        entries.append(TestEntry(e))

    _load_test_positives(pos_ratings, entries, offset)
    _load_test_negatives(neg_ratings, entries, offset)

    return entries


def _load_test_positives(ratings: str, entries: t.List[TestEntry], offset=True):
    n_users = len(entries)
    for id, entry in enumerate(entries):
        entry.positives = (ratings[id, 1:] + n_users * offset).tolist()


def _load_test_negatives(ratings: str, entries: t.List[TestEntry], offset=True):
    n_users = len(entries)
    for id, entry in enumerate(entries):
        entry.negatives = (ratings[id, 1:] + n_users * offset).tolist()


def load_train_entries(path: str) -> t.List[TestEntry]:
    data = load_dataset(path)
    entries = [TestEntry(x, [], []) for x in range(np.max(data, axis=0)[0] + 1)]
    for user, item, review in data:
        if review > 0:
            entries[user].positives.append(item)
        else:
            entries[user].negatives.append(item)
    return entries


if __name__ == '__main__':
    path = './data/bookcross'
    entries = load_test_entries(path)

    for i in range(10):
        entry = entries[i]
        print(entry.id, entry.positives, entry.negatives)
