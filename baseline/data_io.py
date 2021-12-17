import numpy as np
import pandas as pd


def load_training_set(path: str) -> np.ndarray:
    csv_f: pd.DataFrame = pd.read_csv(path, sep='\t', header=None)
    return csv_f.to_numpy()


def filter_dataset(data: np.ndarray) -> np.ndarray:
    n_users = np.max(data, axis=0)[0] + 1  # maximum of user column
    # user nodes: [0 : n_users]
    # item nodes: [n_users:]
    data[:, 1] += n_users
    positives = data[data[:, -1] > 0]
    # print(positives.shape)
    return positives


def load_test_set(path: str) -> np.ndarray:
    pass
