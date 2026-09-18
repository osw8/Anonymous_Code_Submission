from pathlib import Path
import pickle
from sklearn.preprocessing import MinMaxScaler
import scipy.io
import numpy as np
import h5py

BASE_DIR = Path(__file__).resolve().parent

def split_dataset(x: dict, y, fold=0, num_fold=5):
    assert 0 <= fold < num_fold
    
    x_train = dict()
    for k in x.keys():
        x_train[k] = list()
    y_train = list()
    for f in range(num_fold):
        if f != fold:
            for k in x.keys():
                x_train[k].append(x[k][f::num_fold, :])
            y_train.append(y[f::num_fold])
    for k in x.keys():
        x_train[k] = np.concatenate(x_train[k], axis=0)
    y_train = np.concatenate(y_train, axis=0)
    
    x_valid = dict()
    for k in x.keys():
        x_valid[k] = x[k][fold::num_fold, :]
    y_valid = y[fold::num_fold]
    return x_train, y_train, x_valid, y_valid

def read_h5_data(file, key):
    if isinstance(file[key], h5py.Dataset):
        return file[key][:]
    elif isinstance(file[key], h5py.Group):
        data = {k: read_h5_data(file[key], k) for k in file[key].keys()}
        return data
    elif isinstance(file[key], h5py.Reference):
        return file[file[key]][:]
    else:
        raise ValueError(f"Unexpected data type in HDF5 file for key {key}")

def read_referenced_data(file, refs):
    data = []
    for ref in refs:
        if isinstance(ref, h5py.Reference):
            data.append(file[ref][:])
        else:
            data.append(ref)
    return data

def process_eeg_domain(path=BASE_DIR / 'dataset/eeg/data/domain_feature/train_data1.mat', fold=0, saving_name='dataname'):
    print(f'Processing file: {path}')
    with h5py.File(str(path), 'r') as f:
        data = {key: read_h5_data(f, key) for key in f.keys()}
        data['X'] = read_referenced_data(f, data['X'][0])

    x = dict()
    x[0] = MinMaxScaler((0, 1)).fit_transform(data['X'][0]).astype(np.float32)
    x[1] = MinMaxScaler((0, 1)).fit_transform(data['X'][1]).astype(np.float32)
    x[2] = MinMaxScaler((0, 1)).fit_transform(data['X'][2]).astype(np.float32)
    y = np.argmax(data['Y'], axis=1)
    x_train, y_train, x_valid, y_valid = split_dataset(x, y, fold=fold, num_fold=5)
    pickle.dump([x_train, y_train], open(path.parent / f'{saving_name}_fold{fold}_train.pkl', 'wb'))
    pickle.dump([x_valid, y_valid], open(path.parent / f'{saving_name}_fold{fold}_valid.pkl', 'wb'))
    print('---- ', path.name)
    print('domain data shape:', x[0].shape, x[1].shape, x[2].shape, y.shape)
    print('domain training data shape:', x_train[0].shape, x_train[1].shape, x_train[2].shape, y_train.shape)
    print('domain validating data shape:', x_valid[0].shape, x_valid[1].shape, x_valid[2].shape, y_valid.shape)

if __name__ == '__main__':
    process_eeg_domain(BASE_DIR / 'dataset/eeg/data/domain_feature/train_data8.mat', saving_name='data8')
    process_eeg_domain(BASE_DIR / 'dataset/eeg/data/domain_feature/train_data11.mat', saving_name='data11')
    process_eeg_domain(BASE_DIR / 'dataset/eeg/data/domain_feature/train_data15.mat', saving_name='data15')

