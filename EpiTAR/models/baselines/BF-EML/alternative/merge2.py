import pickle
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_pickle(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def merge_datasets(base_dir=BASE_DIR):
    x_merged = {0: [], 1: [], 2: []}
    y_merged = []

    excluded_files = {6}

    for i in range(1, 25):
        if i not in excluded_files:
            file_path = base_dir / f'dataset/eeg/data/domain_feature/data{i}_fold0_train.pkl'
            x, y = load_pickle(file_path)
            for k in x_merged.keys():
                x_merged[k].append(x[k])
            y_merged.append(y)

    for k in x_merged.keys():
        x_merged[k] = np.concatenate(x_merged[k], axis=0)
    y_merged = np.concatenate(y_merged, axis=0)

    save_pickle((x_merged, y_merged), base_dir / 'dataset/eeg/data/domain_feature/merged_data_exclude4_fold0_train.pkl')

if __name__ == '__main__':
    merge_datasets()