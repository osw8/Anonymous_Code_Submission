import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.util import to_tensor
import os
import random
import lmdb
import pickle
from scipy import signal


class CustomDataset(Dataset):
    def __init__(
            self,
            data_dir,
            files,
    ):
        super(CustomDataset, self).__init__()
        self.data_dir = data_dir
        self.files = files

    def __len__(self):
        return len((self.files))

    def __getitem__(self, idx):
        file = self.files[idx]
        data_dict = pickle.load(open(os.path.join(self.data_dir, file), "rb"))
        data = data_dict['signal']
        label = int(data_dict['label'][0]-1)
        
        data = data.reshape(16, 5, 200)
        return data/100, label

    def collate(self, batch):
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        return to_tensor(x_data), to_tensor(y_label).long()


class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.datasets_dir = params.datasets_dir

    def get_data_loader(self):
        train_files = os.listdir(os.path.join(self.datasets_dir, "processed_train"))
        val_files = os.listdir(os.path.join(self.datasets_dir, "processed_eval"))
        test_files = os.listdir(os.path.join(self.datasets_dir, "processed_test"))

        train_set = CustomDataset(os.path.join(self.datasets_dir, "processed_train"), train_files)
        val_set = CustomDataset(os.path.join(self.datasets_dir, "processed_eval"), val_files)
        test_set = CustomDataset(os.path.join(self.datasets_dir, "processed_test"), test_files)

        print(len(train_set), len(val_set), len(test_set))
        print(len(train_set)+len(val_set)+len(test_set))

        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.params.batch_size,
                collate_fn=train_set.collate,
                shuffle=True,
            ),
            'val': DataLoader(
                val_set,
                batch_size=self.params.batch_size,
                collate_fn=val_set.collate,
                shuffle=False,
            ),
            'test': DataLoader(
                test_set,
                batch_size=self.params.batch_size,
                collate_fn=test_set.collate,
                shuffle=False,
            ),
        }
        return data_loader
