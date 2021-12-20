import numpy as np
import pandas as pd


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


def load_test_set(path: str) -> np.ndarray:
    pass
