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


def adjust_data_dimensions(x, target_shape):
    adjusted_x = {}
    for k, v in x.items():
        if v.shape[1] != target_shape[k][1]:
            adjusted_x[k] = np.resize(v, (v.shape[0], target_shape[k][1]))
        else:
            adjusted_x[k] = v
    return adjusted_x


def merge_datasets(start=7, end=24, base_dir=BASE_DIR):
    x_merged = {0: [], 1: [], 2: []}
    y_merged = []
    target_shape = None

    for i in range(start, end + 1):
        file_path = base_dir / f'dataset/eeg/data/domain_feature/data{i}_fold0_train.pkl'
        x, y = load_pickle(file_path)

        if target_shape is None:
            target_shape = {k: v.shape for k, v in x.items()}


        for k in x.keys():
            print(f"data{i} - x[{k}] shape: {x[k].shape}")


        x = adjust_data_dimensions(x, target_shape)

        for k in x_merged.keys():
            x_merged[k].append(x[k])
        y_merged.append(y)


    for k in x_merged.keys():
        print(f"Before concatenation - x_merged[{k}] shapes: {[arr.shape for arr in x_merged[k]]}")

        try:
            x_merged[k] = np.concatenate(x_merged[k], axis=0)
        except ValueError as e:
            print(f"Error concatenating key {k}: {e}")
            for arr in x_merged[k]:
                print(f"Array shape: {arr.shape}")

    y_merged = np.concatenate(y_merged, axis=0)
    save_pickle((x_merged, y_merged), base_dir / 'dataset/eeg/data/domain_feature/merged_data7_24_fold0_train.pkl')


if __name__ == '__main__':
    merge_datasets()
